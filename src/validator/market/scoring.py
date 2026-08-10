"""
scoring_v2.py

DESIGN
------
The old mechanism solved a two-phase convex program (Phase 1 maximise routed
volume s.t. budget; Phase 2 redistribute toward ROI) with an endogenous price
kappa acting as the exchange rate between qualified flow and token budget.

That is a correct way to trace the volume/ROI Pareto frontier, but it needs a
solver, a price, a ramp, an entropy smoother, retention floors and a dust
constraint just to stay feasible and non-cliffy.

This version keeps the frontier and drops the machinery.

    score_i  =  volume_share_i ** ALPHA  *  pnl_share_i ** (1 - ALPHA)

A weighted geometric mean of two normalised objectives is a Cobb-Douglas
utility: maximising it over the allocation simplex lands on the Pareto frontier
of (routed volume, PnL), and ALPHA slides along that frontier. ALPHA = 1 is a
pure pro-rata fee rebate, ALPHA = 0 is pure skill. No solver, no price, no
cliff.

Both axes are ADDITIVE (decayed volume, decayed positive PnL). That makes the
score invariant to splitting one book across k identities: the exponents sum
to 1, so k entities holding (v/k, p/k) score exactly what one entity holding
(v, p) does. Volume uses the same exponential memory as the edge gate and
dust ranking, so a steady daily cadence outranks an equal monthly total
dumped in one burst when both show up today. An earlier draft used a shrunk
ROI estimate ("edge") as the skill axis; the per-entity shrinkage constant
made splitting strictly profitable (sum of P/(V/k + S) grows with k), so it
was replaced with raw PnL share. The residual benefit of splitting is
escaping the per-entity concentration cap, which is bounded at plain
pro-rata and priced by UID registration.

BUDGET
------
Per pool, per epoch:  B = fees collected by that pool this epoch (1% of volume).
    - dust reserve is taken off the top (small, bounded)
    - the rest is distributed by score, subject to a per-trader concentration
      cap and a fee-return floor
    - when caps bind before the residual is exhausted, leftover budget is not
      distributed and falls through to burn. Note the effective per-entity cap
      is max(CONCENTRATION_CAP, CAP_RELAX_FACTOR / n_scoring), so with few
      scorers total cap capacity can exceed the pool — the cap redistributes
      rather than forcing burn in that regime.
Only the final miner-pool weight boost is allowed to exceed fee budget
(clamped so miner + general-pool weight never exceeds 1.0).

TIERS
-----
    ACTIVE    traded this epoch, past build-up  -> full score
    DORMANT   no trades this epoch, inside the inactivity window, positive
              trailing edge                      -> ranked dust
    INACTIVE  nothing in INACTIVITY_EPOCHS       -> zero

The general pool is a separate track that is being retired: its history is
still built for fee accounting and reporting, but scoring is disabled
(ENABLE_GENERAL_POOL_SCORING) and it earns zero tokens.

build_epoch_history() buckets the rolling UTC-day window; pass an explicit
``as_of`` boundary so live scoring and historical replay share one definition.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

AsOf = Optional[Union[datetime, date]]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROLLING_HISTORY_IN_DAYS = 30
VOLUME_FEE = 0.01

# --- the Pareto knob -------------------------------------------------------
# 1.0 = pay volume only, 0.0 = pay PnL only. Everything between is on the
# frontier. This is the single most important number in the file.
PARETO_ALPHA = 0.65

# --- skill / presence estimation ------------------------------------------
# One decay, applied to both PnL and volume. Half-life ~9.5 days at 0.93.
# Volume memory is the presence axis in the active score (and dust ranking);
# budget and fee floors still key off this epoch's fees only.
EDGE_DECAY = 0.93

# --- concentration ---------------------------------------------------------
# Max share of the epoch pool any single trader can take.
CONCENTRATION_CAP = 0.06
# A hard cap flattens the top of the distribution into equal payouts whenever
# it binds for everyone, which is what happens in thin epochs. So the effective
# cap never tightens below CAP_RELAX_FACTOR x the equal share of the traders who
# actually scored. With many scorers CONCENTRATION_CAP binds; with few, the cap
# only clips genuine outliers and the ranking survives.
CAP_RELAX_FACTOR = 2.5

# --- fee-return floor ------------------------------------------------------
# Active traders with real trailing edge get back at least this much of the
# fees they paid this epoch, even on a losing day. Must stay <= 1 / (1 + boost)
# so the boosted floor never exceeds 1.0x fees — otherwise PnL-neutral wash
# volume becomes a guaranteed money pump (validated at import time below).
FEE_FLOOR_MULTIPLIER = 0.57
# Floor eligibility: decayed PnL / decayed volume must clear this. `> 0` is not
# enough — a single old win plus PnL-neutral churn keeps decayed PnL positive
# forever. The trader's own volume grows the denominator, so holding the gate
# while churning requires maintaining real, proportional wins.
FEE_FLOOR_MIN_ROI = 0.005
# Floors may never consume more than this share of the active pool.
FEE_FLOOR_MAX_POOL_SHARE = 0.40

# --- dust ------------------------------------------------------------------
# Total reserve for dormant-but-historically-positive miners.
DUST_RESERVE_SHARE = 0.02
# Worst-ranked dormant miner gets this fraction of the best-ranked one's dust.
DUST_MIN_RATIO = 0.25
# Sanity check only: emitted dust weight relative to the largest weight in the
# vector. Below ~1/65535 the u16 quantisation in set_weights rounds it to zero
# and the dusting does nothing.
U16_QUANT_FLOOR = 1.0 / 65535.0

# --- gates -----------------------------------------------------------------
MIN_EPOCH_VOLUME = 1.0
MIN_EPOCHS_FOR_ELIGIBILITY = 3
MIN_TRADES_FOR_ELIGIBILITY = 5
INACTIVITY_EPOCHS = 10

# --- pool split / boost ----------------------------------------------------
# Dynamic split: each pool's budget is the fees that pool generated.
MINER_POOL_WEIGHT_BOOST_PERCENTAGE = 0.75
BURN_UID = 210

# General pool is a separate track and is being retired: its history is still
# built (fees, reporting) but it earns zero tokens.
ENABLE_GENERAL_POOL_SCORING = False

# Fail fast: if the boosted floor exceeds 1.0x fees, wash trading turns +EV.
if FEE_FLOOR_MULTIPLIER * (1 + MINER_POOL_WEIGHT_BOOST_PERCENTAGE) > 1.0:
    raise ValueError(
        f"FEE_FLOOR_MULTIPLIER ({FEE_FLOOR_MULTIPLIER}) x boost "
        f"(1 + {MINER_POOL_WEIGHT_BOOST_PERCENTAGE}) exceeds 1.0x fees; "
        "this makes fee-churning profitable. Lower one of them."
    )


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

def _decayed(matrix: np.ndarray, decay: float) -> np.ndarray:
    """Exponentially decayed column sums, most recent epoch at weight 1.0."""
    n_epochs = matrix.shape[0]
    if n_epochs == 0:
        return np.zeros(matrix.shape[1])
    weights = decay ** np.arange(n_epochs - 1, -1, -1.0)
    return weights @ matrix


def compute_pnl(history: Dict[str, Any]) -> np.ndarray:
    """
    Decayed, non-negative PnL per entity — the skill mass in the score.

    PnL is additive, so pnl_share (unlike a per-entity ROI estimate) cannot be
    inflated by splitting one book across identities. Negative-PnL traders get
    exactly zero, which zeroes their whole score — but it arrives as a limit
    rather than a cliff, so no entropy smoothing is needed.
    """
    return np.maximum(_decayed(history["profit_prev"], EDGE_DECAY), 0.0)


def compute_volume_memory(history: Dict[str, Any]) -> np.ndarray:
    """
    Decayed volume per entity — the presence mass in the score.

    Same decay as PnL / edge. Active traders must still trade today to be
    eligible; among those who do, trailing cadence outranks a one-day burst.
    """
    return _decayed(history["volume_prev"], EDGE_DECAY)


def compute_edge(history: Dict[str, Any]) -> np.ndarray:
    """
    Decayed PnL / decayed volume per entity. Used for the fee-floor gate and
    diagnostics only — never inside the score, where a ratio would reintroduce
    small-sample ROI outliers.
    """
    pnl = _decayed(history["profit_prev"], EDGE_DECAY)
    vol = compute_volume_memory(history)
    return np.maximum(pnl, 0.0) / np.maximum(vol, 1.0)


def pareto_score(volume: np.ndarray, pnl: np.ndarray, alpha: float | None = None) -> np.ndarray:
    """Cobb-Douglas blend of normalised volume share and normalised PnL share."""
    alpha = PARETO_ALPHA if alpha is None else alpha
    v_tot, p_tot = volume.sum(), pnl.sum()
    if v_tot <= 0 or p_tot <= 0:
        return np.zeros_like(volume)
    v_share = volume / v_tot
    p_share = pnl / p_tot
    return (v_share ** alpha) * (p_share ** (1.0 - alpha))


def _project_to_budget(
    shares: np.ndarray,
    total: float,
    floors: np.ndarray,
    cap_fraction: float,
    max_iter: int = 64,
) -> np.ndarray:
    """
    Distribute `total` proportional to `shares`, respecting per-entity floors and
    a per-entity cap, by water-filling.

    Starts at floors (scaled to fit if needed), then allocates the residual
    proportional to shares with per-entity headroom up to the cap. By
    construction ``0 <= sum(alloc) <= total``. Undistributable mass (when caps
    bind before the residual is exhausted) is left on the table for burn.
    """
    n = shares.size
    if n == 0 or total <= 0:
        return np.zeros(n)

    shares = np.maximum(np.nan_to_num(shares), 0.0)
    n_scoring = max(int(np.sum(shares > 0)), 1)
    cap = max(float(cap_fraction), CAP_RELAX_FACTOR / n_scoring) * total
    cap = min(cap, total)
    floors = np.clip(np.nan_to_num(floors), 0.0, cap)
    if floors.sum() > total:
        floors = floors * (total / floors.sum())

    # Floors first so a simultaneous cap+floor pin cannot overshoot the budget.
    alloc = floors.copy()
    remaining = float(total - alloc.sum())
    free = np.ones(n, dtype=bool)

    for _ in range(max_iter):
        if remaining <= 1e-12:
            break
        headroom = np.where(free, cap - alloc, 0.0)
        free = free & (headroom > 1e-12)
        w = np.where(free, shares, 0.0)
        if w.sum() <= 0:
            break

        add = remaining * w / w.sum()
        over = free & (add > headroom + 1e-9)
        if not over.any():
            alloc = alloc + add
            break

        alloc = np.where(over, cap, alloc)
        free = free & ~over
        remaining = float(total - alloc.sum())

    return alloc


def utc_epoch_boundary(as_of: AsOf = None) -> datetime:
    """Exclusive end of the scoring window, floored to 00:00 UTC.

    Epochs are calendar UTC days. Pass an explicit ``as_of`` so validators and
    historical replays share the same window; default is wall-clock now.
    """
    if as_of is None:
        as_of = datetime.now(timezone.utc)
    if isinstance(as_of, datetime):
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)
        else:
            as_of = as_of.astimezone(timezone.utc)
        return as_of.replace(hour=0, minute=0, second=0, microsecond=0)
    return datetime(as_of.year, as_of.month, as_of.day, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Tiers
# ---------------------------------------------------------------------------

def classify_entities(history: Dict[str, Any], edge: np.ndarray, allow_dust: bool):
    """Return boolean masks (active, dormant) for the current epoch."""
    n = history["n_entities"]
    if n == 0:
        empty = np.zeros(0, dtype=bool)
        return empty, empty

    trades = history["trade_counts"]
    volume = history["volume_prev"]
    n_epochs = history["n_epochs"]
    cur = n_epochs - 1

    epochs_traded = np.sum(trades > 0, axis=0)
    total_trades = np.sum(trades, axis=0)
    built_up = (epochs_traded >= MIN_EPOCHS_FOR_ELIGIBILITY) & (
        total_trades >= MIN_TRADES_FOR_ELIGIBILITY
    )

    lookback = max(1, INACTIVITY_EPOCHS)
    recent = np.any(trades[max(0, n_epochs - lookback):, :] > 0, axis=0)

    traded_now = volume[cur] >= MIN_EPOCH_VOLUME

    active = built_up & traded_now
    dormant = built_up & recent & ~traded_now & (edge > 0) if allow_dust else np.zeros(n, dtype=bool)
    return active, dormant


def dust_allocations(history: Dict[str, Any], dormant: np.ndarray, pnl: np.ndarray, reserve: float) -> np.ndarray:
    """
    Rank dormant miners by their trailing quality and pay a linear ramp of dust
    from DUST_MIN_RATIO (worst) to 1.0 (best). Ranked rather than value-scaled
    so a single outlier cannot flatten everyone else's dusting to nothing.
    """
    n = dormant.size
    out = np.zeros(n)
    idx = np.flatnonzero(dormant)
    if idx.size == 0 or reserve <= 0:
        return out

    v_mem = compute_volume_memory(history)
    quality = pareto_score(v_mem[idx], pnl[idx])
    if quality.sum() <= 0:
        quality = np.ones(idx.size)

    order = np.argsort(np.argsort(quality))  # 0 = worst
    pct = order / max(idx.size - 1, 1)
    ramp = DUST_MIN_RATIO + (1.0 - DUST_MIN_RATIO) * pct

    out[idx] = reserve * ramp / ramp.sum()
    return out


# ---------------------------------------------------------------------------
# Pool scoring
# ---------------------------------------------------------------------------

def score_pool(history: Dict[str, Any], budget: float, allow_dust: bool, verbose: bool = False) -> Dict[str, Any]:
    """Score one pool for the current epoch. Returns tokens (alpha) per entity."""
    n = history["n_entities"]
    if n == 0 or budget <= 0:
        return {
            "entity_ids": history.get("entity_ids", []),
            "tokens": np.zeros(n),
            "scores": np.zeros(n),
            "edge": np.zeros(n),
            "active": np.zeros(n, dtype=bool),
            "dormant": np.zeros(n, dtype=bool),
            "distributed": 0.0,
            "undistributed": float(max(budget, 0.0)),
        }

    cur = history["n_epochs"] - 1
    epoch_fees = history["fees_prev"][cur]

    edge = compute_edge(history)
    pnl = compute_pnl(history)
    v_mem = compute_volume_memory(history)
    active, dormant = classify_entities(history, edge, allow_dust)

    # --- dust reserve off the top ------------------------------------------
    reserve = DUST_RESERVE_SHARE * budget if dormant.any() else 0.0
    dust = dust_allocations(history, dormant, pnl, reserve)
    active_pool = budget - dust.sum()

    # --- Pareto score over active traders ----------------------------------
    scores = np.zeros(n)
    if active.any():
        scores[active] = pareto_score(v_mem[active], pnl[active])

    # --- fee-return floors --------------------------------------------------
    floors = np.zeros(n)
    eligible_floor = active & (edge >= FEE_FLOOR_MIN_ROI)
    floors[eligible_floor] = FEE_FLOOR_MULTIPLIER * epoch_fees[eligible_floor]
    max_floor = FEE_FLOOR_MAX_POOL_SHARE * active_pool
    if floors.sum() > max_floor > 0:
        floors *= max_floor / floors.sum()

    tokens = _project_to_budget(scores, active_pool, floors, CONCENTRATION_CAP)
    tokens = tokens + dust

    distributed = float(tokens.sum())
    if verbose:
        print(
            f"  budget={budget:,.2f}  active={int(active.sum())}  dormant={int(dormant.sum())}  "
            f"dust={dust.sum():,.2f}  distributed={distributed:,.2f}  "
            f"undistributed={budget - distributed:,.2f}"
        )

    return {
        "entity_ids": history["entity_ids"],
        "tokens": tokens,
        "scores": scores,
        "edge": edge,
        "active": active,
        "dormant": dormant,
        "distributed": distributed,
        "undistributed": float(budget - distributed),
    }


def build_epoch_history(
    trading_history: List[Dict[str, Any]],
    all_uids: List[int],
    all_hotkeys: List[str],
    is_miner_pool: bool,
    target_epoch_idx: int = None,
    as_of: AsOf = None,
) -> Dict[str, Any]:
    """
    Bucket settled trades into (epoch, entity) matrices.

    Carried over from the v1 mechanism with two changes: defensive .get()
    lookups, and trades flagged by the local hygiene rules (fee underpayment,
    off-Almanac position top-ups) now count their losses against profit_prev
    even though their volume is excluded — see the comment at the flagged
    branch below.

    ``as_of`` is the exclusive end of the window (floored to 00:00 UTC). The
    trailing ``ROLLING_HISTORY_IN_DAYS`` calendar days before that boundary are
    scored. Pass it explicitly for historical replay and so a single scoring
    run does not drift across midnight mid-computation.

    ``target_epoch_idx`` is retained for older call sites: when ``as_of`` is
    omitted it selects the exclusive boundary so epoch ``k`` of a live-built
    history is replayed against a full trailing window ending on that day,
    rather than a truncated ``k+1``-day matrix.
    """
    if as_of is not None:
        today = utc_epoch_boundary(as_of)
    elif target_epoch_idx is not None:
        today = utc_epoch_boundary(None) - timedelta(
            days=ROLLING_HISTORY_IN_DAYS - 1 - int(target_epoch_idx)
        )
    else:
        today = utc_epoch_boundary(None)

    start_date = today - timedelta(days=ROLLING_HISTORY_IN_DAYS)
    n_epochs = ROLLING_HISTORY_IN_DAYS
    epoch_dates = [(start_date + timedelta(days=i)).date() for i in range(n_epochs)]

    entity_set = set()
    epoch_trades = defaultdict(list)
    miner_profiles: Dict[int, str] = {}
    account_map: Dict[Any, Any] = {}

    for trade in trading_history:
        if trade.get("account_id") is None or not trade.get("is_completed"):
            continue

        completed = trade.get("completed_at")
        if not completed:
            continue
        if isinstance(completed, str):
            completed = datetime.fromisoformat(completed.replace("Z", "+00:00"))
        trade_date = completed.date()
        if trade_date < epoch_dates[0] or trade_date >= today.date():
            continue
        epoch_idx = (trade_date - epoch_dates[0]).days
        if epoch_idx >= n_epochs:
            continue

        account_id = trade["account_id"]

        if is_miner_pool:
            if trade.get("is_general_pool"):
                continue
            miner_id = trade.get("miner_id")
            miner_hotkey = trade.get("miner_hotkey")
            if miner_id is None or miner_hotkey is None:
                continue
            if (
                miner_id not in all_uids
                or miner_id >= len(all_hotkeys)
                or all_hotkeys[miner_id] != miner_hotkey
            ):
                continue
            entity_id = miner_id
            if trade.get("is_reward_eligible"):
                pid = str(trade.get("profile_id", "")).lower()
                if miner_id not in miner_profiles:
                    miner_profiles[miner_id] = pid
                elif pid not in miner_profiles[miner_id].split(","):
                    miner_profiles[miner_id] += f",{pid}"
                account_map.setdefault(entity_id, account_id)
        else:
            if not trade.get("is_general_pool"):
                continue
            entity_id = trade["profile_id"]
            account_map.setdefault(entity_id, account_id)

        entity_set.add(entity_id)
        epoch_trades[epoch_idx].append((entity_id, trade))

    entity_ids = sorted(entity_set, key=str)
    entity_map = {eid: i for i, eid in enumerate(entity_ids)}
    n = len(entity_ids)

    shape = (n_epochs, n)
    volume_prev = np.zeros(shape)
    qualified_prev = np.zeros(shape)
    unqualified_prev = np.zeros(shape)
    profit_prev = np.zeros(shape)
    fees_prev = np.zeros(shape)
    trade_counts = np.zeros(shape)
    correct_trade_counts = np.zeros(shape)

    for epoch_idx, rows in epoch_trades.items():
        for entity_id, trade in rows:
            j = entity_map[entity_id]

            volume = float(trade.get("volume") or 0.0)
            expected_volume = float(trade.get("expected_volume") or 0.0)
            actual_fees = float(trade.get("actual_fees") or 0.0)
            expected_fees = float(trade.get("expected_fees") or 0.0)
            pnl = float(trade.get("pnl") or 0.0)
            is_correct = bool(trade.get("is_correct"))
            eligible = bool(trade.get("is_reward_eligible"))

            hygiene_flagged = False
            # Position was topped up outside Almanac ($5 rounding buffer).
            if eligible and volume > expected_volume and abs(volume - expected_volume) > 5:
                eligible = False
                hygiene_flagged = True
            # Underpaid fees (10% buffer).
            if eligible and expected_fees > 0 and actual_fees < expected_fees:
                if (actual_fees / expected_fees) < 0.9:
                    eligible = False
                    hygiene_flagged = True

            # Fees are always collected — they fund the pool regardless.
            fees_prev[epoch_idx, j] += actual_fees

            if eligible and actual_fees > 0:
                volume_prev[epoch_idx, j] += volume
                if is_correct:
                    qualified_prev[epoch_idx, j] += volume - actual_fees
                    correct_trade_counts[epoch_idx, j] += 1
                else:
                    unqualified_prev[epoch_idx, j] += volume
                profit_prev[epoch_idx, j] += pnl
                trade_counts[epoch_idx, j] += 1
            elif hygiene_flagged:
                # Trades the trader made ineligible post-hoc (external top-up,
                # underpaid fees) still count their LOSSES against PnL —
                # otherwise flipping a losing trade ineligible censors the loss
                # out of the skill signal. Wins stay uncounted; volume and
                # trade counts stay excluded. Trades that arrive ineligible
                # from the API are untouched.
                profit_prev[epoch_idx, j] += min(pnl, 0.0)

    return {
        "volume_prev": volume_prev,
        "qualified_prev": qualified_prev,
        "unqualified_prev": unqualified_prev,
        "profit_prev": profit_prev,
        "fees_prev": fees_prev,
        "trade_counts": trade_counts,
        "correct_trade_counts": correct_trade_counts,
        "entity_ids": entity_ids,
        "entity_map": entity_map,
        "epoch_dates": [str(d) for d in epoch_dates],
        "n_epochs": n_epochs,
        "n_entities": n,
        "miner_profiles": miner_profiles,
        "account_map": account_map,
    }


def pool_epoch_fees(history: Dict[str, Any]) -> float:
    """Fees generated by a pool in its current epoch — this pool's budget."""
    if history["n_entities"] == 0 or history["n_epochs"] == 0:
        return 0.0
    return float(np.sum(history["fees_prev"][history["n_epochs"] - 1]))


def score_miners(
    all_uids: List[int],
    all_hotkeys: List[str],
    trading_history: List[Dict[str, Any]],
    current_epoch_budget: float = None,
    verbose: bool = False,
    target_epoch_idx: int = None,
    as_of: AsOf = None,
):
    """
    Drop-in replacement for the v1 entry point. Same signature, same 6-tuple.

    `current_epoch_budget` is the *subnet* emission budget and is not used for
    distribution — each pool's distributable budget is the fees that pool
    generated. It is retained because calculate_weights needs it as the
    denominator when converting tokens to on-chain weight.

    Pass ``as_of`` (UTC day boundary) so both pools share one pinned window.
    """
    if trading_history is None:
        raise ValueError("trading_history is required")
    if isinstance(trading_history, dict):
        if "data" not in trading_history:
            raise ValueError(f"trading_history dict missing 'data': {list(trading_history.keys())}")
        trading_history = trading_history["data"]
    if not isinstance(trading_history, list):
        raise ValueError(f"trading_history must be a list, got {type(trading_history)}")
    if all_uids is None or all_hotkeys is None:
        raise ValueError("all_uids and all_hotkeys are required")

    # One pinned boundary for both pools — never call datetime.now() twice.
    if as_of is not None:
        boundary = utc_epoch_boundary(as_of)
    elif target_epoch_idx is not None:
        boundary = utc_epoch_boundary(None) - timedelta(
            days=ROLLING_HISTORY_IN_DAYS - 1 - int(target_epoch_idx)
        )
    else:
        boundary = utc_epoch_boundary(None)

    miner_history = build_epoch_history(
        trading_history, all_uids, all_hotkeys, True, as_of=boundary
    )
    general_pool_history = build_epoch_history(
        trading_history, all_uids, all_hotkeys, False, as_of=boundary
    )

    miner_budget = pool_epoch_fees(miner_history)
    gp_budget = pool_epoch_fees(general_pool_history)

    if verbose:
        print(f"Epoch fees — miner: {miner_budget:,.2f}  general: {gp_budget:,.2f}")
        print("Miner pool:")
    miners_scores = score_pool(miner_history, miner_budget, allow_dust=True, verbose=verbose)
    if verbose:
        print("General pool:" + ("" if ENABLE_GENERAL_POOL_SCORING else " (scoring disabled)"))
    # Passing a zero budget yields a correctly-shaped all-zero result, so the
    # return contract and downstream reporting are unchanged when disabled.
    general_pool_scores = score_pool(
        general_pool_history,
        gp_budget if ENABLE_GENERAL_POOL_SCORING else 0.0,
        allow_dust=False,
        verbose=verbose,
    )

    return (
        miner_history,
        general_pool_history,
        miners_scores,
        general_pool_scores,
        miner_budget,
        gp_budget,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_pool(label: str, history: Dict[str, Any], scores: Dict[str, Any] | None, include_current_epoch: bool) -> None:
    n = history["n_entities"]
    print(f"--- {label} ---")
    if n == 0:
        print("(no entities)")
        return

    cur = history["n_epochs"] - 1
    vol, pnl, fees = history["volume_prev"], history["profit_prev"], history["fees_prev"]
    tokens = scores["tokens"] if scores is not None else np.zeros(n)

    order = np.argsort(-tokens) if scores is not None else np.argsort(-pnl.sum(axis=0))
    header = f"{'ID':>12} {'30d Vol':>12} {'30d PnL':>12}"
    if include_current_epoch:
        header += f" {'Ep Vol':>10} {'Ep PnL':>10} {'Ep Fees':>9} {'Tokens':>10}"
    print(header)
    for j in order:
        line = (
            f"{str(history['entity_ids'][j]):>12} "
            f"{vol[:, j].sum():>12,.0f} {pnl[:, j].sum():>12,.2f}"
        )
        if include_current_epoch:
            line += (
                f" {vol[cur, j]:>10,.0f} {pnl[cur, j]:>10,.2f}"
                f" {fees[cur, j]:>9,.2f} {tokens[j]:>10,.2f}"
            )
        print(line)


def print_pool_stats(
    miner_history: Dict[str, Any],
    general_pool_history: Dict[str, Any],
    include_current_epoch: bool = False,
    miner_scores: Dict[str, Any] = None,
    general_pool_scores: Dict[str, Any] = None,
) -> None:
    """Plain-text pool summaries. Same call surface the v1 module exposed."""
    _print_pool("MINER POOL", miner_history, miner_scores, include_current_epoch)
    gp_label = "GENERAL POOL" if ENABLE_GENERAL_POOL_SCORING else "GENERAL POOL (scoring disabled)"
    _print_pool(gp_label, general_pool_history, general_pool_scores, include_current_epoch)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

def calculate_weights(
    miner_scores: Dict[str, Any],
    gp_scores: Dict[str, Any],
    total_epoch_budget: float,
    miner_budget: float = 0.0,
    general_pool_budget: float = 0.0,
    miners_to_penalize: List[int] = None,
    all_uids: List[int] = None,
    verbose: bool = False,
) -> List[float]:
    """
    Convert token allocations to an on-chain weight vector.

    Weight is denominated as a fraction of the full subnet epoch budget, so an
    epoch that generates few fees emits proportionally less and burns the rest.
    The miner-pool boost may lift payouts above fee budget, but is clamped so
    miner + general-pool weight never exceeds 1.0 (no all-or-nothing cliff).
    """
    miners_to_penalize = miners_to_penalize or []
    all_uids = all_uids or []

    weights: Dict[int, float] = {}
    if total_epoch_budget <= 0:
        return [0.0] * len(all_uids)

    entity_ids = list(miner_scores["entity_ids"])
    if BURN_UID in entity_ids:
        raise AssertionError(
            f"BURN_UID ({BURN_UID}) collides with a scored entity id; "
            "refusing to mix burn residual into a miner weight"
        )

    for uid, tok in zip(entity_ids, miner_scores["tokens"]):
        if uid in miners_to_penalize:
            continue
        weights[uid] = float(tok) / total_epoch_budget

    miner_weight = sum(weights.values())
    gp_weight = float(np.sum(gp_scores["tokens"])) / total_epoch_budget  # burned

    # Partial boost up to the remaining headroom — avoids a hard cliff where
    # one dollar of distributed volume suddenly drops the whole pool's boost.
    if MINER_POOL_WEIGHT_BOOST_PERCENTAGE > 0 and miner_weight > 0:
        room = max(1.0 - gp_weight, 0.0)
        factor = min(1.0 + MINER_POOL_WEIGHT_BOOST_PERCENTAGE, room / miner_weight)
        if factor > 1.0:
            weights = {u: w * factor for u, w in weights.items()}
            miner_weight = sum(weights.values())

    weights[BURN_UID] = weights.get(BURN_UID, 0.0) + max(1.0 - miner_weight, 0.0)

    vec = [weights.get(uid, 0.0) for uid in all_uids]
    total = sum(vec)
    if total > 0:
        vec = [w / total for w in vec]

    if verbose:
        nz = [w for w in vec if w > 0]
        peak = max(nz) if nz else 0.0
        smallest = min(nz) if nz else 0.0
        print(
            f"miner weight={miner_weight:.4f}  burn={weights[BURN_UID]:.4f}  "
            f"nonzero uids={len(nz)}  smallest/largest={smallest / peak if peak else 0:.2e} "
            f"(u16 floor {U16_QUANT_FLOOR:.2e})"
        )
        if peak and smallest / peak < U16_QUANT_FLOOR:
            print("  WARNING: smallest emitted weight rounds to zero under u16 quantisation")

    return vec