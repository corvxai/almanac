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

    score_i  =  volume_share_i ** ALPHA  *  edge_share_i ** (1 - ALPHA)

A weighted geometric mean of two normalised objectives is a Cobb-Douglas
utility: maximising it over the allocation simplex lands on the Pareto frontier
of (routed volume, edge), and ALPHA slides along that frontier. ALPHA = 1 is a
pure pro-rata fee rebate, ALPHA = 0 is pure edge, ALPHA = 0.65 leans to volume
while still paying small sharp traders. No solver, no price, no cliff.

BUDGET
------
Per pool, per epoch:  B = fees collected by that pool this epoch (1% of volume).
    - dust reserve is taken off the top (small, bounded)
    - the rest is distributed by score, subject to a per-trader concentration
      cap and a fee-return floor
    - anything the cap prevents distributing is simply not distributed; it
      falls through to burn. Concentrated epochs pay out less. That is correct:
      the cap is what makes volume-dumping unprofitable.
Only the final miner-pool weight boost is allowed to exceed budget.

TIERS
-----
    ACTIVE    traded this epoch, past build-up  -> full score
    DORMANT   no trades this epoch, inside the inactivity window, positive
              trailing edge                      -> ranked dust
    INACTIVE  nothing in INACTIVITY_EPOCHS       -> zero

The general pool has no dust tier (no UIDs to deregister) and requires epoch
activity. That is the only structural difference between the two pipelines.

build_epoch_history() from the current codebase is unchanged and still upstream
of everything here; this module consumes the same dict.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROLLING_HISTORY_IN_DAYS = 30
VOLUME_FEE = 0.01

# --- the Pareto knob -------------------------------------------------------
# 1.0 = pay volume only, 0.0 = pay edge only. Everything between is on the
# frontier. This is the single most important number in the file.
PARETO_ALPHA = 0.65

# --- edge estimation -------------------------------------------------------
# One decay, applied once, to both PnL and volume. Half-life ~9.5 days at 0.93.
EDGE_DECAY = 0.93
# Shrinkage pseudo-volume: roughly "the decayed volume below which you are not
# yet credible". A trader with $200 of settled volume and a 300% ROI gets shrunk
# toward zero; a trader with $200k is barely touched. This replaces the
# credibility weights, the min-trades gate and the ROI cliff. Tune to the pool's
# volume distribution — a sensible start is the median trader's decayed volume.
EDGE_SHRINKAGE_VOLUME = 10_000.0
# Hard ceiling on estimated edge so one longshot cannot own the edge component.
EDGE_CAP = 0.40

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
# Active traders with positive trailing edge get back at least this much of the
# fees they paid this epoch, even on a losing day.
FEE_FLOOR_MULTIPLIER = 0.70
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


def compute_edge(history: Dict[str, Any]) -> np.ndarray:
    """
    Shrunk, decayed, non-negative ROI estimate per entity.

        edge = max(0, decayed_pnl) / (decayed_volume + SHRINKAGE)

    Negative-PnL traders get exactly zero, which is the ROI_MIN gate — but it
    arrives as a limit rather than a cliff, so no entropy smoothing is needed.
    """
    pnl = _decayed(history["profit_prev"], EDGE_DECAY)
    vol = _decayed(history["volume_prev"], EDGE_DECAY)
    edge = np.maximum(pnl, 0.0) / (vol + EDGE_SHRINKAGE_VOLUME)
    return np.minimum(edge, EDGE_CAP)


def pareto_score(volume: np.ndarray, edge: np.ndarray, alpha: float | None = None) -> np.ndarray:
    """Cobb-Douglas blend of normalised volume share and normalised edge share."""
    alpha = PARETO_ALPHA if alpha is None else alpha
    v_tot, e_tot = volume.sum(), edge.sum()
    if v_tot <= 0 or e_tot <= 0:
        return np.zeros_like(volume)
    v_share = volume / v_tot
    e_share = edge / e_tot
    return (v_share ** alpha) * (e_share ** (1.0 - alpha))


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

    Replaces the LP diversity constraints, the dust constraint, the retention
    floors and the post-hoc rescaling passes. Undistributable mass (when caps
    bind) is intentionally left on the table and falls through to burn.
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

    alloc = np.zeros(n)
    pinned = np.zeros(n, dtype=bool)
    remaining = total

    for _ in range(max_iter):
        if remaining <= 1e-12:
            break
        free = ~pinned
        w = np.where(free, shares, 0.0)
        if w.sum() <= 0:
            need = np.where(free, floors, 0.0)
            if need.sum() > 0:
                alloc = alloc + need * (min(remaining, need.sum()) / need.sum())
            break

        candidate = np.where(free, remaining * w / w.sum(), alloc)
        over = free & (candidate > cap + 1e-9)
        under = free & (candidate < floors - 1e-9)

        if not over.any() and not under.any():
            alloc = candidate
            break

        alloc = np.where(over, cap, np.where(under, floors, alloc))
        pinned = pinned | over | under
        remaining = max(total - alloc[pinned].sum(), 0.0)

    return alloc


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


def dust_allocations(history: Dict[str, Any], dormant: np.ndarray, edge: np.ndarray, reserve: float) -> np.ndarray:
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

    v_mem = _decayed(history["volume_prev"], EDGE_DECAY)
    quality = pareto_score(v_mem[idx], edge[idx])
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
    epoch_volume = history["volume_prev"][cur]
    epoch_fees = history["fees_prev"][cur]

    edge = compute_edge(history)
    active, dormant = classify_entities(history, edge, allow_dust)

    # --- dust reserve off the top ------------------------------------------
    reserve = DUST_RESERVE_SHARE * budget if dormant.any() else 0.0
    dust = dust_allocations(history, dormant, edge, reserve)
    active_pool = budget - dust.sum()

    # --- Pareto score over active traders ----------------------------------
    scores = np.zeros(n)
    if active.any():
        scores[active] = pareto_score(epoch_volume[active], edge[active])

    # --- fee-return floors --------------------------------------------------
    floors = np.zeros(n)
    eligible_floor = active & (edge > 0)
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
) -> Dict[str, Any]:
    """
    Bucket settled trades into (epoch, entity) matrices.

    Carried over unchanged from the v1 mechanism apart from defensive .get()
    lookups — the data-hygiene rules here (fee underpayment, off-Almanac
    position top-ups, hotkey/UID match) are orthogonal to how scoring works and
    are still exactly what you want.
    """
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = today - timedelta(days=ROLLING_HISTORY_IN_DAYS)

    n_epochs = target_epoch_idx + 1 if target_epoch_idx is not None else ROLLING_HISTORY_IN_DAYS
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

            # Position was topped up outside Almanac ($5 rounding buffer).
            if eligible and volume > expected_volume and abs(volume - expected_volume) > 5:
                eligible = False
            # Underpaid fees (10% buffer).
            if eligible and expected_fees > 0 and actual_fees < expected_fees:
                if (actual_fees / expected_fees) < 0.9:
                    eligible = False

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
):
    """
    Drop-in replacement for the v1 entry point. Same signature, same 6-tuple.

    `current_epoch_budget` is the *subnet* emission budget and is not used for
    distribution — each pool's distributable budget is the fees that pool
    generated. It is retained because calculate_weights needs it as the
    denominator when converting tokens to on-chain weight.
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

    miner_history = build_epoch_history(
        trading_history, all_uids, all_hotkeys, True, target_epoch_idx
    )
    general_pool_history = build_epoch_history(
        trading_history, all_uids, all_hotkeys, False, target_epoch_idx
    )

    miner_budget = pool_epoch_fees(miner_history)
    gp_budget = pool_epoch_fees(general_pool_history)

    if verbose:
        print(f"Epoch fees — miner: {miner_budget:,.2f}  general: {gp_budget:,.2f}")
        print("Miner pool:")
    miners_scores = score_pool(miner_history, miner_budget, allow_dust=True, verbose=verbose)
    if verbose:
        print("General pool:")
    general_pool_scores = score_pool(
        general_pool_history, gp_budget, allow_dust=False, verbose=verbose
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
    The miner-pool boost is the only step permitted to exceed budget.
    """
    miners_to_penalize = miners_to_penalize or []
    all_uids = all_uids or []

    weights: Dict[int, float] = {}
    if total_epoch_budget <= 0:
        return [0.0] * len(all_uids)

    for uid, tok in zip(miner_scores["entity_ids"], miner_scores["tokens"]):
        if uid in miners_to_penalize:
            continue
        weights[uid] = float(tok) / total_epoch_budget

    miner_weight = sum(weights.values())
    gp_weight = float(np.sum(gp_scores["tokens"])) / total_epoch_budget  # burned

    if MINER_POOL_WEIGHT_BOOST_PERCENTAGE > 0 and miner_weight + gp_weight < 1.0:
        boosted = {u: w * (1 + MINER_POOL_WEIGHT_BOOST_PERCENTAGE) for u, w in weights.items()}
        if sum(boosted.values()) + gp_weight <= 1.0:
            weights = boosted
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