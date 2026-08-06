"""
Simulation script to test scoring_v2.py against real trading history.

1. Loads data/trading_history.json
2. Extracts miner UIDs / hotkeys
3. Runs the current epoch through score_miners()
4. Replays every historical epoch to produce a payout timeline
5. Prints per-pool tables, mechanism diagnostics, and the on-chain weight vector

Differences vs the v1 simulator, all downstream of the mechanism change:
  - no kappa columns (no price to report)
  - no Phase 1 / Phase 2 status, T*, or dual diagnostics (no solver)
  - no dust-gate x1->x2 tracking (dust is now an explicit reserve, not a
    constraint that Phase 2 can quietly drop)
  + tier accounting (active / dust / gated), edge distribution, cap binding,
    and distributed-vs-burned budget, which are the things that can actually
    go wrong now
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import requests
from tabulate import tabulate

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.validator.market.scoring import (  # noqa: E402
    score_miners,
    calculate_weights,
    compute_edge,
    pool_epoch_fees,
    ROLLING_HISTORY_IN_DAYS,
    PARETO_ALPHA,
    EDGE_DECAY,
    CONCENTRATION_CAP,
    CAP_RELAX_FACTOR,
    FEE_FLOOR_MULTIPLIER,
    FEE_FLOOR_MIN_ROI,
    ENABLE_GENERAL_POOL_SCORING,
    DUST_RESERVE_SHARE,
    DUST_MIN_RATIO,
    INACTIVITY_EPOCHS,
    MIN_EPOCHS_FOR_ELIGIBILITY,
    MIN_TRADES_FOR_ELIGIBILITY,
    MINER_POOL_WEIGHT_BOOST_PERCENTAGE,
    U16_QUANT_FLOOR,
    BURN_UID,
)

TOTAL_MINER_ALPHA_PER_DAY = 2952
EXCESS_MINER_WEIGHT_UID = None
FALLBACK_ALPHA_PRICE_USD = 5.0  # used with --offline

_TRADING_HISTORY_PATH = _REPO_ROOT / "data" / "trading_history.json"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def extract_miner_info(trading_history):
    """Return (all_uids, all_hotkeys) where all_hotkeys[uid] is that UID's hotkey."""
    miner_map = {}
    for trade in trading_history:
        if trade.get("is_general_pool", False):
            continue
        miner_id = trade.get("miner_id")
        hotkey = trade.get("miner_hotkey")
        if miner_id is None or hotkey is None:
            continue
        if miner_id not in miner_map:
            miner_map[miner_id] = hotkey
        elif miner_map[miner_id] != hotkey:
            print(f"WARNING: inconsistent hotkey for miner_id {miner_id}")

    all_uids = sorted(miner_map)
    if not all_uids:
        return [], []
    all_hotkeys = [""] * (max(all_uids) + 1)
    for uid, hk in miner_map.items():
        all_hotkeys[uid] = hk
    return all_uids, all_hotkeys


def resolve_epoch_budget(offline: bool):
    """Subnet emission budget in USD for the epoch."""
    if offline:
        print(f"[offline] assuming alpha price ${FALLBACK_ALPHA_PRICE_USD:.2f}")
        return FALLBACK_ALPHA_PRICE_USD * TOTAL_MINER_ALPHA_PER_DAY

    import bittensor as bt

    subtensor = bt.Subtensor(network="finney")
    metagraph = subtensor.subnets.metagraph(41)
    tao_price = requests.get(
        "https://api.coingecko.com/api/v3/simple/price"
        "?ids=bittensor&vs_currencies=usd",
        timeout=15,
    ).json()["bittensor"]["usd"]
    alpha_price = metagraph.moving_price * tao_price
    print(f"TAO price:   ${tao_price:,.2f}")
    print(f"Alpha price: ${alpha_price:,.4f}")
    return alpha_price * TOTAL_MINER_ALPHA_PER_DAY


# ---------------------------------------------------------------------------
# Historical replay
# ---------------------------------------------------------------------------

def _trade_date(trade):
    raw = trade.get("completed_at")
    if not raw:
        return None
    raw = str(raw)
    return raw.split("T")[0] if "T" in raw else raw


def calculate_historical_payouts(miner_history, all_uids, all_hotkeys, trading_history, debug=False):
    """
    Replay each epoch as it would have scored on the day, returning per-epoch
    payout and diagnostic arrays.

    Much cheaper than the v1 replay: there is no solver in the loop, so this is
    O(epochs x traders) rather than 30 sequential convex programs.
    """
    n_epochs = miner_history["n_epochs"]
    epoch_dates = miner_history["epoch_dates"]

    out = {
        k: np.zeros(n_epochs)
        for k in ("mp_payout", "gp_payout", "mp_budget", "gp_budget",
                  "mp_active", "mp_dust", "mp_undist")
    }

    print(f"Replaying {n_epochs} epochs...")
    for epoch_idx in range(n_epochs):
        epoch_date = epoch_dates[epoch_idx]
        epoch_trades = [
            t for t in trading_history
            if t.get("is_completed") and (_trade_date(t) or "9999") <= epoch_date
        ]
        if not epoch_trades:
            continue

        try:
            _, _, m_scores, g_scores, m_budget, g_budget = score_miners(
                all_uids=all_uids,
                all_hotkeys=all_hotkeys,
                trading_history=epoch_trades,
                verbose=False,
                target_epoch_idx=epoch_idx,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: epoch {epoch_idx} ({epoch_date}) failed: {exc}")
            continue

        out["mp_payout"][epoch_idx] = float(np.sum(m_scores["tokens"]))
        out["gp_payout"][epoch_idx] = float(np.sum(g_scores["tokens"]))
        out["mp_budget"][epoch_idx] = m_budget
        out["gp_budget"][epoch_idx] = g_budget
        out["mp_active"][epoch_idx] = int(np.sum(m_scores["active"]))
        out["mp_dust"][epoch_idx] = int(np.sum(m_scores["dormant"]))
        out["mp_undist"][epoch_idx] = m_scores["undistributed"]

        if debug:
            print(
                f"  epoch {epoch_idx:>2} {epoch_date}: {len(epoch_trades):>6} trades  "
                f"budget ${m_budget:>10,.2f}  paid ${out['mp_payout'][epoch_idx]:>10,.2f}  "
                f"active {int(out['mp_active'][epoch_idx]):>3}  dust {int(out['mp_dust'][epoch_idx]):>3}"
            )

    return out


def print_daily_stats(miner_history, general_pool_history, hist, miners_scores, general_pool_scores):
    """Per-epoch volume, budget, payout and tier counts."""
    n_epochs = miner_history["n_epochs"]
    dates = miner_history["epoch_dates"]
    rows = []

    for i in range(n_epochs):
        def _vol(h):
            if h["n_entities"] == 0:
                return 0.0
            return float(np.sum(h["qualified_prev"][i]) + np.sum(h["unqualified_prev"][i]))

        mp_vol, gp_vol = _vol(miner_history), _vol(general_pool_history)
        mp_budget, gp_budget = hist["mp_budget"][i], hist["gp_budget"][i]

        if i == n_epochs - 1:
            mp_pay = float(np.sum(miners_scores["tokens"]))
            gp_pay = float(np.sum(general_pool_scores["tokens"]))
            active = int(np.sum(miners_scores["active"]))
            dust = int(np.sum(miners_scores["dormant"]))
            undist = miners_scores["undistributed"]
        else:
            mp_pay, gp_pay = hist["mp_payout"][i], hist["gp_payout"][i]
            active, dust = int(hist["mp_active"][i]), int(hist["mp_dust"][i])
            undist = hist["mp_undist"][i]

        total_budget = mp_budget + gp_budget
        total_pay = mp_pay + gp_pay
        rows.append([
            i, dates[i],
            f"${mp_vol:,.0f}", f"${gp_vol:,.0f}",
            f"${mp_budget:,.0f}", f"${gp_budget:,.0f}",
            active, dust,
            f"${mp_pay:,.0f}", f"${gp_pay:,.0f}",
            f"${undist:,.0f}",
            f"{(total_pay / total_budget * 100) if total_budget > 0 else 0:.1f}%",
        ])

    headers = ["Ep", "Date", "MP Vol", "GP Vol", "MP Budget", "GP Budget",
               "Active", "Dust", "MP Payout", "GP Payout", "MP Burned", "Used %"]
    print(tabulate(rows, headers=headers, tablefmt="grid", stralign="right"))


# ---------------------------------------------------------------------------
# Pool tables
# ---------------------------------------------------------------------------

def print_pool_table(history, scores, budget, label, top_n=None):
    """Per-trader breakdown: history, edge, epoch activity, payout."""
    n = history["n_entities"]
    if n == 0:
        print(f"\n--- {label} --- (no entities)")
        return

    cur = history["n_epochs"] - 1
    vol_m, pnl_m, fee_m, trd_m = (
        history["volume_prev"], history["profit_prev"],
        history["fees_prev"], history["trade_counts"],
    )
    edge = scores["edge"]
    tokens = scores["tokens"]

    rows = []
    for j, eid in enumerate(history["entity_ids"]):
        tv = float(vol_m[:, j].sum())
        tp = float(pnl_m[:, j].sum())
        ev, ep, ef = float(vol_m[cur, j]), float(pnl_m[cur, j]), float(fee_m[cur, j])
        et = int(trd_m[cur, j])
        tier = "active" if scores["active"][j] else ("dust" if scores["dormant"][j] else "gated")
        rows.append([
            str(eid), tier,
            int(np.sum(trd_m[:, j] > 0)), int(trd_m[:, j].sum()),
            f"${tv:,.0f}", f"${tp:,.2f}",
            f"{(tp / tv * 100) if tv else 0:.2f}%",
            f"{edge[j] * 100:.2f}%",
            et, f"${ev:,.0f}", f"${ep:,.2f}",
            f"{(ep / ev * 100) if ev else 0:.2f}%",
            f"${ef:,.2f}",
            f"{tokens[j]:,.2f}",
            f"{(tokens[j] / budget * 100) if budget > 0 else 0:.2f}%",
            f"{tokens[j] / ef:.2f}x" if ef > 0 else "-",
        ])

    rows.sort(key=lambda r: -float(r[13].replace(",", "")))
    if top_n:
        rows = rows[:top_n]

    print(f"\n--- {label} (budget ${budget:,.2f}) ---")
    print(tabulate(rows, headers=[
        "ID", "Tier", "Eps", "Preds", "30d Vol", "30d PnL", "30d ROI",
        "Edge", "Ep Preds", "Ep Vol", "Ep PnL", "Ep ROI", "Ep Fees",
        "Earnings", "Share", "vs Fees",
    ], tablefmt="grid", stralign="right"))


# ---------------------------------------------------------------------------
# Mechanism diagnostics
# ---------------------------------------------------------------------------

def print_mechanism_diagnostics(miner_history, miners_scores, miner_budget):
    print("\n--- MECHANISM DIAGNOSTICS ---")
    print(
        f"alpha={PARETO_ALPHA}  pnl_decay={EDGE_DECAY}  "
        f"cap={CONCENTRATION_CAP:.0%} (relax {CAP_RELAX_FACTOR}x)\n"
        f"fee_floor={FEE_FLOOR_MULTIPLIER:.0%} (gate roi>={FEE_FLOOR_MIN_ROI:.2%})  "
        f"dust_reserve={DUST_RESERVE_SHARE:.0%}  "
        f"inactivity={INACTIVITY_EPOCHS} epochs"
    )

    tokens, active, dormant = miners_scores["tokens"], miners_scores["active"], miners_scores["dormant"]
    n = miner_history["n_entities"]
    gated = n - int(active.sum()) - int(dormant.sum())
    print(
        f"\nTiers: active={int(active.sum())}  dust={int(dormant.sum())}  "
        f"gated/inactive={gated}  total={n}"
    )
    print(f"Paid (tokens > 0): {int(np.sum(tokens > 0))}")

    # --- budget accounting ---
    dist = miners_scores["distributed"]
    print(
        f"\nBudget: ${miner_budget:,.2f}  distributed ${dist:,.2f} "
        f"({(dist / miner_budget * 100) if miner_budget else 0:.1f}%)  "
        f"burned ${miners_scores['undistributed']:,.2f}"
    )
    assert dist <= miner_budget + 1e-6, "BUDGET VIOLATION"
    print("Budget constraint: OK")

    # --- concentration ---
    n_scoring = max(int(np.sum(miners_scores["scores"] > 0)), 1)
    cap_eff = max(CONCENTRATION_CAP, CAP_RELAX_FACTOR / n_scoring)
    if miner_budget > 0:
        top = tokens.max() / miner_budget
        n_at_cap = int(np.sum(tokens / miner_budget >= cap_eff - 1e-6))
        print(
            f"Concentration: top share {top:.2%}, effective cap {cap_eff:.2%} "
            f"({n_scoring} scoring), {n_at_cap} at cap"
        )
        if n_at_cap >= max(3, n_scoring // 2):
            print(
                "  NOTE: cap is binding for most payees — payouts are flattening. "
                "Raise CAP_RELAX_FACTOR or lower CONCENTRATION_CAP deliberately."
            )

    # --- edge distribution ---
    edge = miners_scores["edge"]
    live = edge[active | dormant]
    if live.size:
        print(
            f"Edge (decayed ROI): zero={int(np.sum(live <= 0))}  "
            f"below floor gate={int(np.sum((live > 0) & (live < FEE_FLOOR_MIN_ROI)))}  "
            f"median={np.median(live[live > 0]) * 100 if np.any(live > 0) else 0:.2f}%  "
            f"max={live.max() * 100:.2f}%"
        )

    # --- dust ranking ---
    if dormant.any():
        d = np.sort(tokens[dormant])[::-1]
        print(
            f"Dust: {d.size} miners, total ${d.sum():,.2f}, "
            f"range ${d.min():,.4f}-${d.max():,.4f} "
            f"(ratio {d.min() / d.max():.2f}, target {DUST_MIN_RATIO})"
        )
        assert np.all(d > 0), "DUST FAILURE: dormant miner scored zero"
        print("Dust floor: OK (no dormant miner at zero)")

    # --- fee floor ---
    cur = miner_history["n_epochs"] - 1
    fees = miner_history["fees_prev"][cur]
    floored = active & (miners_scores["edge"] >= FEE_FLOOR_MIN_ROI) & (fees > 0)
    if floored.any():
        ratio = tokens[floored] / fees[floored]
        print(
            f"Fee return (active, positive edge): min {ratio.min():.2f}x  "
            f"median {np.median(ratio):.2f}x  max {ratio.max():.2f}x"
        )

    # --- build-up gate cost ---
    trd = miner_history["trade_counts"]
    blocked = (
        (np.sum(trd > 0, axis=0) < MIN_EPOCHS_FOR_ELIGIBILITY)
        | (np.sum(trd, axis=0) < MIN_TRADES_FOR_ELIGIBILITY)
    ) & (miner_history["volume_prev"][cur] > 0)
    if blocked.any():
        print(
            f"Build-up gate: {int(blocked.sum())} miners traded this epoch but are "
            f"still in build-up (paid ${fees[blocked].sum():,.2f} in fees, earned nothing)"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Simulate scoring_v2 against trading history")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--offline", action="store_true",
                        help="Skip subtensor/coingecko; use a fixed alpha price")
    parser.add_argument("--no-replay", action="store_true",
                        help="Score the current epoch only, skip the historical replay")
    parser.add_argument("--top", type=int, default=None,
                        help="Limit pool tables to the top N by payout")
    parser.add_argument("--history", type=Path, default=_TRADING_HISTORY_PATH)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    print(f"Loading trading history from {args.history}...")
    if not args.history.exists():
        raise FileNotFoundError(f"Trading history not found: {args.history}")
    with open(args.history) as f:
        trading_history = json.load(f)
    if isinstance(trading_history, dict):
        trading_history = trading_history.get("data", trading_history)
    print(f"Loaded {len(trading_history)} trades")

    all_uids, all_hotkeys = extract_miner_info(trading_history)
    print(f"Found {len(all_uids)} unique miners")

    # Pad hotkeys / UID list for the special UIDs.
    max_uid = max([BURN_UID] + all_uids + ([EXCESS_MINER_WEIGHT_UID] if EXCESS_MINER_WEIGHT_UID else []))
    if len(all_hotkeys) <= max_uid:
        all_hotkeys.extend([""] * (max_uid + 1 - len(all_hotkeys)))
    weight_uids = list(all_uids)
    if EXCESS_MINER_WEIGHT_UID is not None:
        weight_uids.insert(0, EXCESS_MINER_WEIGHT_UID)
    weight_uids.append(BURN_UID)

    current_epoch_budget = resolve_epoch_budget(args.offline)
    print(f"Subnet epoch (24h) emission budget: ${current_epoch_budget:,.2f}\n")

    print("Scoring current epoch...")
    (miner_history, general_pool_history, miners_scores,
     general_pool_scores, miner_budget, gp_budget) = score_miners(
        all_uids=all_uids,
        all_hotkeys=all_hotkeys,
        trading_history=trading_history,
        current_epoch_budget=current_epoch_budget,
        verbose=True,
    )

    if args.no_replay:
        n = miner_history["n_epochs"]
        hist = {k: np.zeros(n) for k in
                ("mp_payout", "gp_payout", "mp_budget", "gp_budget",
                 "mp_active", "mp_dust", "mp_undist")}
        hist["mp_budget"][-1] = miner_budget
        hist["gp_budget"][-1] = gp_budget
    else:
        hist = calculate_historical_payouts(
            miner_history, all_uids, all_hotkeys, trading_history, debug=args.debug
        )

    print("\n" + "=" * 80)
    print("SCORING v2 SIMULATION RESULTS")
    print("=" * 80)

    print(f"\n--- DAILY STATS (last {ROLLING_HISTORY_IN_DAYS} epochs) ---")
    print_daily_stats(miner_history, general_pool_history, hist, miners_scores, general_pool_scores)

    print("\n--- BUDGET ---")
    print(f"Miner pool (fees):   ${miner_budget:,.2f}")
    print(f"General pool (fees): ${gp_budget:,.2f}")
    print(f"Total distributable: ${miner_budget + gp_budget:,.2f}")
    print(f"Subnet emission:     ${current_epoch_budget:,.2f}")

    print_pool_table(miner_history, miners_scores, miner_budget, "MINER POOL", args.top)
    gp_label = "GENERAL POOL" if ENABLE_GENERAL_POOL_SCORING else "GENERAL POOL (scoring disabled)"
    print_pool_table(general_pool_history, general_pool_scores, gp_budget, gp_label, args.top)

    print_mechanism_diagnostics(miner_history, miners_scores, miner_budget)

    print("\n--- WEIGHTS ---")
    weights = calculate_weights(
        miners_scores,
        general_pool_scores,
        current_epoch_budget,
        miner_budget,
        gp_budget,
        [],
        weight_uids,
        verbose=True,
    )
    print(f"Miner pool weight boost: {MINER_POOL_WEIGHT_BOOST_PERCENTAGE:.0%}")
    print(f"Total weight sum: {sum(weights):.6f}")
    print("-" * 40)
    for uid, w in zip(weight_uids, weights):
        if w > 1e-9 or uid in (BURN_UID, EXCESS_MINER_WEIGHT_UID):
            tag = " (burn)" if uid == BURN_UID else ""
            print(f"{str(uid):<6} {w:.8f}{tag}")
    print("-" * 40)

    nz = [w for w in weights if w > 0]
    if nz and min(nz) / max(nz) < U16_QUANT_FLOOR:
        print(
            f"\nWARNING: smallest weight is {min(nz) / max(nz):.2e} of the largest, "
            f"below the u16 quantisation floor ({U16_QUANT_FLOOR:.2e}). "
            "Dustings will round to zero on chain."
        )


if __name__ == "__main__":
    main()
    print("\nSimulation complete.")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)