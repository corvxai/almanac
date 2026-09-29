"""
Simulation script to test scoring.py against real trading history.

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
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from tabulate import tabulate

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.constants import VALIDATOR_LOOP  # noqa: E402
from src.validator.market.constants import (  # noqa: E402
    BURN_UID,
    ENABLE_GENERAL_POOL_SCORING,
    EXCESS_MINER_WEIGHT_UID,
    ROLLING_HISTORY_IN_DAYS,
    TOTAL_MINER_ALPHA_PER_DAY,
    U16_QUANT_FLOOR,
)
from src.validator.market.scoring import (  # noqa: E402
    calculate_weights,
    pool_epoch_fees,
    print_mechanism_diagnostics,
    print_pool_table,
    score_miners,
    utc_epoch_boundary,
)

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

    from src.validator.market.loop import fetch_tao_price

    subtensor = bt.Subtensor(network="finney")
    metagraph = subtensor.subnets.metagraph(41)
    tao_price = fetch_tao_price()
    alpha_price = metagraph.moving_price * tao_price
    print(f"TAO price:   ${tao_price:,.2f}")
    print(f"Alpha price: ${alpha_price:,.4f}")
    return alpha_price * TOTAL_MINER_ALPHA_PER_DAY


# ---------------------------------------------------------------------------
# Historical replay
# ---------------------------------------------------------------------------

def calculate_historical_payouts(miner_history, all_uids, all_hotkeys, trading_history, debug=False):
    """
    Replay each epoch as it would have scored on the day, returning per-epoch
    payout and diagnostic arrays.

    Each day D is scored with as_of = D+1 (exclusive UTC midnight) so the
    trailing window is always ROLLING_HISTORY_IN_DAYS long — not a truncated
    prefix of today's matrix.
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
        as_of = datetime.strptime(epoch_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        ) + timedelta(days=1)

        try:
            _, _, m_scores, g_scores, m_budget, g_budget = score_miners(
                all_uids=all_uids,
                all_hotkeys=all_hotkeys,
                trading_history=trading_history,
                verbose=False,
                as_of=as_of,
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
                f"  epoch {epoch_idx:>2} {epoch_date}: as_of {as_of.date()}  "
                f"budget ${m_budget:>10,.2f}  paid ${out['mp_payout'][epoch_idx]:>10,.2f}  "
                f"active {int(out['mp_active'][epoch_idx]):>3}  dust {int(out['mp_dust'][epoch_idx]):>3}"
            )

    return out


def _u16_weights(weights):
    """Quantise a weight vector the way chain set_weights effectively does."""
    arr = np.asarray(weights, dtype=float)
    if arr.size == 0 or arr.sum() <= 0:
        return np.zeros_like(arr, dtype=int)
    scaled = arr / arr.sum()
    return np.floor(scaled * 65535 + 1e-12).astype(int)


def export_settled_epoch(
    path: Path,
    *,
    as_of,
    snapshot_at: str,
    epoch_budget_usd: float,
    miner_budget: float,
    gp_budget: float,
    miner_history,
    miners_scores,
    weight_uids,
    weights,
):
    """Write a checkable end-to-end epoch artefact (per account + burn + u16)."""
    u16 = _u16_weights(weights)
    weight_by_uid = {int(uid): float(w) for uid, w in zip(weight_uids, weights)}
    u16_by_uid = {int(uid): int(q) for uid, q in zip(weight_uids, u16)}

    accounts = []
    for eid, tok, active, dormant, edge, score in zip(
        miners_scores["entity_ids"],
        miners_scores["tokens"],
        miners_scores["active"],
        miners_scores["dormant"],
        miners_scores["edge"],
        miners_scores["scores"],
    ):
        uid = int(eid)
        accounts.append({
            "uid": uid,
            "tokens": float(tok),
            "tier": "active" if active else ("dust" if dormant else "gated"),
            "edge": float(edge),
            "score": float(score),
            "weight": weight_by_uid.get(uid, 0.0),
            "weight_u16": u16_by_uid.get(uid, 0),
        })

    burn_w = weight_by_uid.get(BURN_UID, 0.0)
    payload = {
        "snapshot_at": snapshot_at,
        "as_of": as_of.isoformat() if hasattr(as_of, "isoformat") else str(as_of),
        "epoch_budget_usd": float(epoch_budget_usd),
        "miner_pool_fees": float(miner_budget),
        "general_pool_fees": float(gp_budget),
        "distributed": float(miners_scores["distributed"]),
        "undistributed": float(miners_scores["undistributed"]),
        "burn_uid": BURN_UID,
        "burn_weight": burn_w,
        "burn_weight_u16": u16_by_uid.get(BURN_UID, 0),
        "weight_sum": float(sum(weights)),
        "weight_sum_u16": int(u16.sum()),
        "n_epochs": int(miner_history["n_epochs"]),
        "epoch_dates": list(miner_history["epoch_dates"]),
        "accounts": accounts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote settled-epoch export to {path}")


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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Simulate scoring against trading history")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--offline", action="store_true",
                        help="Skip subtensor and the live TAO price; use a fixed alpha price")
    parser.add_argument("--no-replay", action="store_true",
                        help="Score the current epoch only, skip the historical replay")
    parser.add_argument("--top", type=int, default=None,
                        help="Limit pool tables to the top N by payout")
    parser.add_argument("--history", type=Path, default=_TRADING_HISTORY_PATH)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--export-epoch",
        type=Path,
        default=None,
        help="Write a settled-epoch JSON (per-account tokens, burn, u16 weights)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    print(f"Loading trading history from {args.history}...")
    if not args.history.exists():
        raise FileNotFoundError(f"Trading history not found: {args.history}")
    with open(args.history) as f:
        raw = json.load(f)

    snapshot_at = None
    exported_effective = None
    exported_subnet = None
    exported_share = None
    if isinstance(raw, dict):
        snapshot_at = raw.get("snapshot_at")
        exported_effective = raw.get("epoch_budget_usd")
        exported_subnet = raw.get("subnet_epoch_budget_usd")
        exported_share = raw.get("budget_share")
        trading_history = raw.get("data", raw)
        if snapshot_at or exported_effective is not None:
            print(
                f"Export metadata: snapshot_at={snapshot_at!r}  "
                f"epoch_budget_usd={exported_effective}  "
                f"subnet_epoch_budget_usd={exported_subnet}  "
                f"budget_share={exported_share}"
            )
    else:
        trading_history = raw
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

    share = float(VALIDATOR_LOOP.market_weight_share)
    forecast_share = float(VALIDATOR_LOOP.forecasting_weight_share)
    if exported_subnet is not None:
        subnet_budget = float(exported_subnet)
    elif exported_effective is not None and exported_share:
        subnet_budget = float(exported_effective) / float(exported_share)
    elif exported_effective is not None:
        subnet_budget = float(exported_effective)
        print(
            "WARNING: export has no subnet_epoch_budget_usd/budget_share; "
            "treating epoch_budget_usd as the full subnet emission"
        )
    else:
        subnet_budget = resolve_epoch_budget(args.offline)
    current_epoch_budget = subnet_budget * float(np.clip(share, 0.0, 1.0))
    print(f"Subnet epoch (24h) emission: ${subnet_budget:,.2f}")
    print(
        f"market_weight_share={share:.4f} → market slice ${current_epoch_budget:,.2f}  "
        f"forecasting_weight_share={forecast_share:.4f} reserved "
        f"${subnet_budget - current_epoch_budget:,.2f}"
    )
    if exported_share is not None and abs(float(exported_share) - share) > 1e-9:
        print(
            f"NOTE: export was written with budget_share={float(exported_share):.4f}; "
            f"scoring with market_weight_share={share:.4f} from src.core.constants"
        )
    print()

    as_of = utc_epoch_boundary()
    print("Scoring current epoch...")
    (miner_history, general_pool_history, miners_scores,
     general_pool_scores, miner_budget, gp_budget) = score_miners(
        all_uids=all_uids,
        all_hotkeys=all_hotkeys,
        trading_history=trading_history,
        current_epoch_budget=current_epoch_budget,
        verbose=True,
        as_of=as_of,
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
    print(f"Subnet emission:     ${subnet_budget:,.2f}")
    print(f"Market slice:        ${current_epoch_budget:,.2f}  (share={share:.4f})")
    print(
        f"Forecasting reserve: ${subnet_budget - current_epoch_budget:,.2f}  "
        f"(share={forecast_share:.4f})"
    )

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
        budget_share=share,
    )
    print(
        f"Market vector sum={sum(weights):.6f} (100% of the market slice). "
        f"On-chain blend keeps {share:.4f} of this vector and "
        f"{forecast_share:.4f} for forecasting."
    )
    print("-" * 40)
    for uid, w in zip(weight_uids, weights):
        if w > 1e-9 or uid in (BURN_UID, EXCESS_MINER_WEIGHT_UID):
            tag = " (burn)" if uid == BURN_UID else ""
            print(f"{str(uid):<6} {w:.8f} * share = {w * share:.8f}{tag}")
    print("-" * 40)

    nz = [w for w in weights if w > 0]
    if nz and min(nz) / max(nz) < U16_QUANT_FLOOR:
        print(
            f"\nWARNING: smallest weight is {min(nz) / max(nz):.2e} of the largest, "
            f"below the u16 quantisation floor ({U16_QUANT_FLOOR:.2e}). "
            "Dustings will round to zero on chain."
        )

    if args.export_epoch is not None:
        export_settled_epoch(
            args.export_epoch,
            as_of=as_of,
            snapshot_at=snapshot_at or datetime.now(timezone.utc).isoformat(),
            epoch_budget_usd=current_epoch_budget,
            miner_budget=miner_budget,
            gp_budget=gp_budget,
            miner_history=miner_history,
            miners_scores=miners_scores,
            weight_uids=weight_uids,
            weights=weights,
        )


if __name__ == "__main__":
    main()
    print("\nSimulation complete.")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)