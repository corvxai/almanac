"""Almanac Forecasting scoring from orchestrator scored predictions.

Composite score per miner is a weighted blend of two pillars, computed over
that miner's most recent predictions (``MAX_EVENTS_PER_MINER``) within the
orchestrator fetch window:

    1. Accuracy     (weight 0.70) - Brier skill against the market price
                     stored at prediction time, on the same events, plus a
                     small allowance. An exact match scores the allowance.
                     The anchor.
    2. Edge         (weight 0.30) - centered market-relative Brier-difference
                     + PnL, scaled so a few points of edge move the composite.
                     0 matches the market at prediction time.

Pillar weights are renormalised over their sum at compute time, so only their relative sizes matter.

Pipeline per scoring tick (stateless; a pure function of the fetched rows):

    gates -> per-event pillar scores -> recency-weighted pillar means ->
    composite -> rho significance -> inactivity -> slice pools

Eligibility gates (any failure -> score 0):
    * Invalid-rate gate : >= INVALID_RATE_THRESHOLD of the miner's capped
      window is invalid -> 0. Invalids below the gate are excluded from every
      pillar; they are never scored at a baseline (which would quietly reward
      laziness).
    * Minimum-sample gate: fewer than MIN_VALID_PREDICTIONS_HARD_FLOOR valid
      predictions in the capped window -> 0.
    * Baseline-accuracy gate: recency-weighted mean Brier at or above
      ACCURACY_BASELINE_BRIER (the always-0.5 coin baseline) -> 0. Skill and
      edge can still be positive there (a miner can beat a weak market
      without beating a coin flip), so the gate is what zeroes the miner.
    * Inactivity gate: minerLastPredictedAt older than INACTIVITY_ZERO_HOURS
      -> 0, linear fade after INACTIVITY_GRACE_HOURS. A null or missing
      timestamp does not penalize.

Post-gate scoring details:
    * Recency-weighted pillar means use exponential calendar-time decay
      (``RECENCY_HALF_LIFE_DAYS``), not an EMA. The function is stateless and
      order-independent across validator ticks.
    * Rho significance multiplies the composite by a logistic S-curve in
      recency-weighted effective sample size. The floor holds through ~15-20
      predictions, then rho ramps toward saturation by ~170
      (``RHO_THRESHOLD_PREDICTIONS`` is the midpoint, not full rho).
    * Inactivity applies after rho, using the orchestrator's
      ``minerLastPredictedAt`` (last submission, including unresolved).
      Linear fade beyond ``INACTIVITY_GRACE_HOURS``, hard zero beyond
      ``INACTIVITY_ZERO_HOURS``. Null leaves the multiplier at 1.
    * The forecasting slice is spent through two pools. Every positive
      composite draws an allowance share. A miner who beats the market and
      reaches ``SKILL_MIN_EFFECTIVE_N`` is paid at least that share; only the
      amount above it comes from the skill pool. A shorter positive skill
      draws the allowance share alone.
      Unspent slice weight is added to ``BURN_UID`` when that UID is in the
      metagraph. :func:`_apply_pareto` is not used for the payout.
    * Edge pillar blends centered Brier-difference vs market (60%) with
      winner-side PnL (40%), then scales by ``EDGE_SIGNAL_GAIN``.

Coverage (research-trace quality) is not a scoring pillar. Invalid
predictions are marked orchestrator-side via ``predictionIsInvalid`` and
enforced by the invalid-rate gate.

A row counts only when ``minerUid`` is in the current metagraph and
``minerHotkey`` is the hotkey registered at that UID. Predictions from a
deregistered miner that still carry a recycled UID are dropped, including
their ``minerLastPredictedAt``.

Returns a ``np.ndarray`` aligned to the metagraph UIDs. Each entry is a
share of the forecasting slice. With ``BURN_UID`` in the metagraph the
shares sum to 1.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
from tabulate import tabulate

from src.validator.market.constants import BURN_UID

logger = logging.getLogger("forecasting.scoring")


# Orchestrator fetch/cutoff window. This is a data-retrieval bound, not a
# scoring dial: it just needs to be wide enough to cover MAX_EVENTS_PER_MINER
# at current assignment cadence and keep the recency decay well-populated.
DEFAULT_ROLLING_WINDOW_DAYS = 30

# --------------------------------------------------------------------------- #
# Tunable scoring constants.                                                   #
# --------------------------------------------------------------------------- #

# Composite pillar weights. Renormalised over their sum at compute time, so
# only relative sizes matter.
WEIGHT_ACCURACY = 0.70
WEIGHT_EDGE = 0.30

# Edge pillar internals: blend of Brier-difference vs market and winner-side
# PnL. Sub-weights sum to 1.0. Brier-difference weighted higher: bounded,
# lower-variance signal vs PnL's product of two differences.
EDGE_WEIGHT_BRIER_DIFF = 0.60
EDGE_WEIGHT_PNL = 0.40

# Eligibility gates.
MIN_VALID_PREDICTIONS_HARD_FLOOR = 10 # tiny-sample floor; below this scoring is too noisy
INVALID_RATE_THRESHOLD = 0.15         # >= this fraction invalid in window -> 0

# Per-miner event window. A safety valve for uniformity and bounded compute;
# the recency half-life below is the intended "how much history matters" dial.
# At ~50 assignments/miner/day this covers ~20 days, past the half-life.
MAX_EVENTS_PER_MINER = 1000

# Recency decay (calendar-time clock, deliberately NOT event-count based so
# throughput changes never rescale it). Drives both the per-event weights in
# the pillar means and rho's effective prediction count.
RECENCY_HALF_LIFE_DAYS = 14.0

# Time-based significance (rho): logistic in recency-weighted sample size.
# Orchestration caps how many predictions a miner can make, so rho is a
# small-sample prior, not a grind incentive. Daily scored volume will vary;
# the curve is in effective count, not calendar time.
#   ~15-20  leave the floor (rho just above RHO_FLOOR)
#   ~100    logistic midpoint (rho≈0.55) = RHO_THRESHOLD_PREDICTIONS
#   ~170    saturation (rho≈0.97)
RHO_THRESHOLD_PREDICTIONS = 100.0
RHO_ALPHA = 0.05
RHO_FLOOR = 0.10                      # minimum rho once miner passes hard gates

# Inactivity policy (separate from rho): fade stale miners then switch off.
# Clock is minerLastPredictedAt, the last submission across open and settled
# predictions. This is the anti-perpetuity guard - nobody coasts on old history.
# A null timestamp means the orchestrator did not know; do not zero the miner.
INACTIVITY_GRACE_HOURS = 24.0         # no inactivity penalty while within grace
INACTIVITY_ZERO_HOURS = 72.0          # score forced to 0 after this staleness age

# Accuracy is a Brier skill score against the market on the same events:
# clip(1 - (agent_mean_brier / market_mean_brier) ** gamma + allowance, -1, 1).
# Gamma 1 leaves the ratio equal to the skill score. The power is on the
# ratio of the two means, not on each event: a per-event ratio pays an
# uninformed extremizer when the market is confident. Events with no market
# price drop out of both means. A miner with none scores 0. There is no
# fallback to the coin flip. The allowance shifts the zero: an exact match
# scores +allowance, and a miner that much worse than the market scores 0.
# A zero market mean with a worse agent is -1 before the allowance.
# ACCURACY_BASELINE_BRIER is the coin-flip gate, not this pillar's denominator.
ACCURACY_BASELINE_BRIER = 0.25        # Brier of always predicting 0.5
ACCURACY_HARDNESS_GAMMA = 1.0
ACCURACY_SKILL_ALLOWANCE = 0.2       # skill added after the ratio; 0.2 worse than the market scores 0

# Forecasting-slice budget. Shares are of the slice, not of the subnet.
# The allowance pool is the only budget a miner can draw when they have not
# beaten the market, or have beaten it on fewer than SKILL_MIN_EFFECTIVE_N
# recency-weighted predictions. Past that floor the skill pool pays only the
# amount above the miner's allowance share, and saturates at SKILL_SATURATION.
# Caps do not redistribute.
# Anything neither pool spends is burn.
ALLOWANCE_POOL_SHARE = 0.25
ALLOWANCE_MINER_CAP = 0.03            # of the slice
SKILL_POOL_SHARE = 0.75
SKILL_SATURATION = 0.10               # 10% lower mean Brier than the market fills the cap
SKILL_MINER_CAP = 0.40                # of the skill pool, before rho
SKILL_MIN_EFFECTIVE_N = 50.0          # below this, positive skill draws allowance

# Edge is centered at 0 (matching the market). The raw Brier-difference / PnL
# blend is only a few hundredths in practice; mapping it through (x+1)/2 used
# to pin every miner near 0.5. Gain 10 saturates the pillar at a 0.10 blended
# edge, so a few points of market-relative Brier move the composite on the
# same scale as accuracy's within-cohort range. The pillar is clipped to [-1, 1].
EDGE_SIGNAL_GAIN = 10.0

# Predictions are rounded once, at the input boundary, for cross-validator
# determinism. Never reapplied downstream (spec section 9.5).
PREDICTION_DECIMALS = 2

# Post-composite Pareto shaping (positive scores only). Power-law mode scales
# each miner relative to the current leader: out = MU + BOOST * (score/leader)^GAMMA.
# GAMMA > 1 mildly concentrates emissions toward absolute top performers while
# keeping gradual separation for everyone else (no rank-only dead zone).
PARETO_MU = 0.15          # Output floor (leader-relative score of 0 maps here).
PARETO_BOOST = 0.70       # Output lift from MU to MU+BOOST at the leader.
PARETO_GAMMA = 1.75        # Power exponent; 1.0 is linear, 2.0 is quadratic.

# Legacy rank-knee parameters (used only by ``_apply_pareto_by_rank``). Useful
# when many mature miners cluster at similar composites and you want the
# concentration profile to depend on rank, not absolute gaps.
PARETO_KNEE = 0.5         # Normalized rank where the curve begins rising faster.
PARETO_SHARPNESS = 10.0   # Higher values make the knee transition steeper.
PARETO_TAIL = 1.35        # >1 steepens the very top tail; 1 is linear in knee-space.

# Pillar internals.
NEUTRAL_PILLAR_SCORE = 0.0        # edge fallback when no market prices exist

_WEIGHTS = {
    "accuracy": WEIGHT_ACCURACY,
    "edge": WEIGHT_EDGE,
}


@dataclass
class _Record:
    """A single valid, resolved, in-window prediction, normalized for scoring."""

    p_win: float              # agent prob on the resolved (winning) outcome, rounded
    p_pred: float             # agent prob on its own predicted outcome, rounded
    hit: int                  # 1 if predicted outcome == resolved outcome else 0
    scored_at: datetime       # when this resolved prediction was scored
    market_p_win: Optional[float]   # market price on the winning outcome at prediction
    market_p_pred: Optional[float]  # market price on predicted side at prediction


def score_agent_predictions(
    *,
    metagraph,
    scored_predictions,
    rolling_window_days: int = DEFAULT_ROLLING_WINDOW_DAYS,
    now: Optional[datetime] = None,
    return_pre_pareto: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Return forecasting slice shares from orchestrator ``scored-predictions`` rows.

    Output is a float ``np.ndarray`` indexed by metagraph UID position.
    Positive composites are paid from the skill pool or the allowance pool.
    Gated miners stay at 0. Unspent slice weight is added to ``BURN_UID``
    when that UID is present. With ``return_pre_pareto`` the pre-pool
    composite (after rho and inactivity, before the pools), each miner's
    pool label (``skill``, ``allow``, or ``-``), and the composite before
    rho are returned alongside, for diagnostics and simulation tooling.
    """
    _validate_weights()

    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=rolling_window_days)

    uids = _metagraph_uids(metagraph)
    n = len(uids)
    if n == 0:
        return np.zeros(0, dtype=float)
    uid_to_idx = {uid: idx for idx, uid in enumerate(uids)}
    hotkeys = _metagraph_hotkeys(metagraph)
    if len(hotkeys) < n:
        logger.warning(
            "forecasting scoring: metagraph hotkeys (%d) shorter than uids (%d); "
            "rows without a registered hotkey at their UID are dropped",
            len(hotkeys),
            n,
        )

    # Attributable rows per miner: (scored_at, record-or-None, invalid_flag).
    # record is None for invalid-flagged and unparseable rows; both stay in
    # the capped population so the invalid-rate gate and the scored pillars
    # always observe the same window.
    rows_by_idx: list[list[tuple[datetime, Optional[_Record], bool]]] = [
        [] for _ in range(n)
    ]
    last_predicted_by_idx: list[Optional[datetime]] = [None for _ in range(n)]
    dropped_hotkey = 0

    for item in scored_predictions:
        if getattr(item, "resolutionStatus", None) != "resolved":
            continue

        scored_at = getattr(item, "scoredAt", None)
        if scored_at is None:
            continue
        if scored_at.tzinfo is None:
            scored_at = scored_at.replace(tzinfo=timezone.utc)
        if scored_at < cutoff:
            continue

        uid = getattr(item, "minerUid", None)
        if uid is None:
            continue
        idx = uid_to_idx.get(int(uid))
        if idx is None:
            continue
        miner_hotkey = getattr(item, "minerHotkey", None)
        if (
            not isinstance(miner_hotkey, str)
            or idx >= len(hotkeys)
            or hotkeys[idx] != miner_hotkey
        ):
            dropped_hotkey += 1
            continue

        last_predicted = _as_utc(getattr(item, "minerLastPredictedAt", None))
        if last_predicted is not None:
            prev = last_predicted_by_idx[idx]
            if prev is None or last_predicted > prev:
                last_predicted_by_idx[idx] = last_predicted

        if getattr(item, "predictionIsInvalid", None) is True:
            rows_by_idx[idx].append((scored_at, None, True))
            continue
        rows_by_idx[idx].append((scored_at, _parse_record(item, scored_at=scored_at), False))

    # Diagnostics + outputs, aligned to metagraph idx.
    scores = np.zeros(n, dtype=float)
    accuracy = np.zeros(n, dtype=float)
    edge = np.zeros(n, dtype=float)
    rho_by_idx = np.zeros(n, dtype=float)
    effective_n_by_idx = np.zeros(n, dtype=float)
    latest_age_hours_by_idx = np.full(n, np.nan, dtype=float)
    mean_brier_by_idx = np.full(n, np.nan, dtype=float)
    skill_by_idx = np.full(n, np.nan, dtype=float)
    significance_by_idx = np.zeros(n, dtype=float)
    composite_by_idx = np.zeros(n, dtype=float)
    total_in_window = np.zeros(n, dtype=int)
    invalid_in_window = np.zeros(n, dtype=int)
    records_by_idx: list[list[_Record]] = [[] for _ in range(n)]

    gated_invalid = 0
    gated_sample = 0
    gated_baseline = 0
    gated_inactive = 0
    missing_last_predicted = 0
    scored_miners = 0

    weight_sum = sum(_WEIGHTS.values())

    for idx in range(n):
        rows = rows_by_idx[idx]
        if not rows:
            continue

        # Most recent MAX_EVENTS_PER_MINER attributable rows form the window.
        rows.sort(key=lambda t: t[0], reverse=True)
        rows = rows[:MAX_EVENTS_PER_MINER]
        total = len(rows)
        invalid = sum(1 for _, _, inv in rows if inv)
        total_in_window[idx] = total
        invalid_in_window[idx] = invalid

        recs = [r for _, r, _ in rows if r is not None]
        records_by_idx[idx] = recs

        if invalid / total >= INVALID_RATE_THRESHOLD:
            gated_invalid += 1
            continue

        if len(recs) < MIN_VALID_PREDICTIONS_HARD_FLOOR:
            gated_sample += 1
            continue

        rec_weights = _recency_weights(recs, now=now)
        skill = _market_skill(recs, rec_weights)
        acc = _accuracy_from_skill(skill)
        edg = _edge_score(recs, rec_weights)
        mean_brier = _mean_brier(recs, rec_weights)
        mean_brier_by_idx[idx] = mean_brier
        accuracy[idx] = acc
        edge[idx] = edg
        if skill is not None:
            skill_by_idx[idx] = skill

        effective_n = float(rec_weights.sum())
        rho = compute_significance_score(
            num_miner_predictions=effective_n,
            num_threshold_predictions=RHO_THRESHOLD_PREDICTIONS,
            alpha=RHO_ALPHA,
        )
        latest_age_hours = _submission_age_hours(last_predicted_by_idx[idx], now=now)
        inactivity_mult = _inactivity_multiplier(latest_age_hours)
        effective_n_by_idx[idx] = effective_n
        rho_by_idx[idx] = rho
        if latest_age_hours is None:
            missing_last_predicted += 1
        else:
            latest_age_hours_by_idx[idx] = latest_age_hours

        # Mean Brier at or above the coin flip. Skill and edge can still be
        # positive here (beating a weak market is not the same as beating a
        # coin flip), so the gate is what zeroes the miner.
        if mean_brier >= ACCURACY_BASELINE_BRIER:
            gated_baseline += 1
            continue

        if inactivity_mult <= 0.0:
            gated_inactive += 1
            continue

        composite = (WEIGHT_ACCURACY * acc + WEIGHT_EDGE * edg) / weight_sum
        composite_by_idx[idx] = float(composite)
        significance_by_idx[idx] = rho * inactivity_mult
        scores[idx] = float(np.clip(composite * significance_by_idx[idx], 0.0, 1.0))
        scored_miners += 1

    pre_pareto = scores.copy()
    scores, pool_by_idx, allowance_spent, skill_spent = _allocate_slice(
        pre_pareto,
        skill_by_idx,
        significance_by_idx,
        effective_n_by_idx,
    )
    spent = float(scores.sum())
    burn = max(0.0, 1.0 - spent)
    burn_idx = uid_to_idx.get(BURN_UID)
    if burn_idx is not None:
        if scores[burn_idx] > 0.0:
            logger.warning(
                "forecasting scoring: BURN_UID %d also has a miner claim of %.4f; "
                "burn %.4f is added on top",
                BURN_UID,
                float(scores[burn_idx]),
                burn,
            )
        scores[burn_idx] += burn
    elif burn > 0.0:
        logger.warning(
            "forecasting scoring: BURN_UID %d is not in the metagraph; "
            "%.4f of the forecasting slice is unassigned",
            BURN_UID,
            burn,
        )

    logger.info(
        "forecasting composite scoring: %d valid rows | %d miners scored, "
        "%d gated (invalid-rate), %d gated (min-sample-hard-floor), %d gated (baseline-accuracy), "
        "%d gated (inactive) "
        "| cutoff=%s window=%dd cap=%d | %d dropped (hotkey mismatch) "
        "| slice allowance %.3f/%.2f, skill %.3f/%.2f, burn %.3f",
        sum(len(r) for r in records_by_idx),
        scored_miners,
        gated_invalid,
        gated_sample,
        gated_baseline,
        gated_inactive,
        cutoff.isoformat(),
        rolling_window_days,
        MAX_EVENTS_PER_MINER,
        dropped_hotkey,
        allowance_spent,
        ALLOWANCE_POOL_SHARE,
        skill_spent,
        SKILL_POOL_SHARE,
        burn,
    )
    if missing_last_predicted:
        logger.warning(
            "forecasting scoring: %d miner(s) missing minerLastPredictedAt; "
            "inactivity gate not applied to them",
            missing_last_predicted,
        )
    _log_score_table(
        uids=uids,
        scores=scores,
        pre_pareto=pre_pareto,
        accuracy=accuracy,
        edge=edge,
        rho_by_idx=rho_by_idx,
        effective_n_by_idx=effective_n_by_idx,
        latest_age_hours_by_idx=latest_age_hours_by_idx,
        mean_brier_by_idx=mean_brier_by_idx,
        records_by_idx=records_by_idx,
        total_in_window=total_in_window,
        invalid_in_window=invalid_in_window,
        pool_by_idx=pool_by_idx,
    )
    if return_pre_pareto:
        return scores, pre_pareto, pool_by_idx, composite_by_idx
    return scores


# --------------------------------------------------------------------------- #
# Parsing                                                                      #
# --------------------------------------------------------------------------- #

def _parse_record(item, scored_at: datetime) -> Optional[_Record]:
    """Normalize a single DTO row into a ``_Record``, or ``None`` if unusable."""
    outcome_probs = getattr(item, "outcomeProbabilities", None) or {}
    resolved_outcome_id = getattr(item, "resolvedOutcomeId", None)
    predicted_outcome_id = getattr(item, "predictedOutcomeId", None)

    if not isinstance(outcome_probs, dict):
        return None
    if not isinstance(resolved_outcome_id, str) or resolved_outcome_id not in outcome_probs:
        return None

    p_win = _coerce_prob(outcome_probs.get(resolved_outcome_id))
    if p_win is None:
        return None

    # Probability the agent placed on its OWN chosen side, plus whether it hit.
    if isinstance(predicted_outcome_id, str) and predicted_outcome_id in outcome_probs:
        p_pred = _coerce_prob(outcome_probs.get(predicted_outcome_id))
        hit = 1 if predicted_outcome_id == resolved_outcome_id else 0
    else:
        # No chosen side: treat the winner as the pick so hit is 1.
        p_pred = p_win
        hit = 1
    if p_pred is None:
        return None

    # Agent probabilities are rounded exactly once, here at the input
    # boundary, for cross-validator determinism. Market prices are reference
    # data, not predictions, and are left untouched.
    p_win = round(p_win, PREDICTION_DECIMALS)
    p_pred = round(p_pred, PREDICTION_DECIMALS)

    # Market baseline at prediction time (edge pillar + ROI diagnostics).
    prices_at_pred = getattr(item, "outcomePricesAtPrediction", None) or {}
    market_p_win = None
    market_p_pred = None
    if isinstance(prices_at_pred, dict) and resolved_outcome_id in prices_at_pred:
        market_p_win = _coerce_prob(prices_at_pred.get(resolved_outcome_id))
    if isinstance(prices_at_pred, dict) and isinstance(predicted_outcome_id, str):
        if predicted_outcome_id in prices_at_pred:
            market_p_pred = _coerce_prob(prices_at_pred.get(predicted_outcome_id))

    return _Record(
        p_win=p_win,
        p_pred=p_pred,
        hit=hit,
        scored_at=scored_at,
        market_p_win=market_p_win,
        market_p_pred=market_p_pred,
    )


def _coerce_prob(value) -> Optional[float]:
    """Coerce to a float probability in [0, 1], or ``None`` if invalid."""
    try:
        p = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(p) or p < 0.0 or p > 1.0:
        return None
    return p


# --------------------------------------------------------------------------- #
# Pillar 1: Accuracy (Brier skill vs the market, recency-weighted means)      #
# --------------------------------------------------------------------------- #

def _market_skill(recs: list[_Record], weights: np.ndarray) -> Optional[float]:
    """Pre-allowance Brier skill against the market, or ``None`` if no prices.

    ``1 - (agent_mean / market_mean) ** gamma`` over events that have a
    market price, using the same recency weights for both means. The ratio
    is of the two means, not the mean of per-event ratios. Both means exactly
    0 is an exact match (skill 0). A zero market mean with a worse agent is -1.
    """
    paired = [
        (r, w)
        for r, w in zip(recs, weights)
        if r.market_p_win is not None
    ]
    if not paired:
        return None
    paired_recs = [r for r, _ in paired]
    paired_weights = np.array([w for _, w in paired], dtype=float)
    if float(paired_weights.sum()) <= 0.0:
        return None

    agent_brier = _mean_brier(paired_recs, paired_weights)
    market_briers = np.array(
        [(1.0 - r.market_p_win) ** 2 for r in paired_recs],
        dtype=float,
    )
    market_brier = float(np.average(market_briers, weights=paired_weights))
    if market_brier <= 0.0:
        return 0.0 if agent_brier <= 0.0 else -1.0
    return float(1.0 - (agent_brier / market_brier) ** ACCURACY_HARDNESS_GAMMA)


def _accuracy_from_skill(skill: Optional[float]) -> float:
    """Accuracy pillar: skill plus the allowance, clipped. No skill is 0."""
    if skill is None:
        return 0.0
    return float(np.clip(skill + ACCURACY_SKILL_ALLOWANCE, -1.0, 1.0))


def _accuracy_score(recs: list[_Record], weights: np.ndarray) -> float:
    """Brier skill against the market on the same events, plus the allowance.

    ``clip(skill + allowance, -1, 1)`` where ``skill`` is
    ``1 - (agent_mean / market_mean) ** gamma`` and both means are the
    recency-weighted mean of ``(1 - p)^2`` over events that have a market
    price. An exact match scores the allowance. A miner
    ``ACCURACY_SKILL_ALLOWANCE`` worse than the market scores 0.

    The ratio is of the two means, not the mean of per-event ratios. A
    per-event ratio has a large positive expectation for an uninformed
    extremizer when the market is confident. Events with no market price are
    excluded from both means. No priced events scores 0, with no allowance.
    Both means exactly 0 is an exact match, so it scores the allowance. A
    zero market mean with a worse agent is -1 before the allowance.

    With the market mean fixed, this is still maximized by reporting the true
    probability: the power is applied to the ratio of means, and gamma > 0
    keeps that ratio strictly increasing in the agent's mean Brier.
    """
    return _accuracy_from_skill(_market_skill(recs, weights))


def _allocate_slice(
    pre_pareto: np.ndarray,
    skill: np.ndarray,
    significance: np.ndarray,
    effective_n: np.ndarray,
) -> tuple[np.ndarray, list[str], float, float]:
    """Map pre-pool composites into forecasting-slice shares.

    Every positive composite takes a share of ``ALLOWANCE_POOL_SHARE``,
    proportional to that composite and capped at ``ALLOWANCE_MINER_CAP``.
    A miner with positive pre-allowance skill and at least
    ``SKILL_MIN_EFFECTIVE_N`` recency-weighted predictions also has a skill
    claim, ``min(skill / SKILL_SATURATION, 1) * SKILL_MINER_CAP * SKILL_POOL_SHARE * significance``,
    scaled down if those claims exceed the skill pool. They are paid the
    larger of the two, not the sum. The allowance share is funded by the
    allowance pool, and only the excess by the skill pool. Caps and unused
    pool budget are not redistributed.

    Returns weights, pool labels, allowance spent, and skill spent.
    """
    n = len(pre_pareto)
    weights = np.zeros(n, dtype=float)
    pools = ["-"] * n
    skill_claims = np.zeros(n, dtype=float)
    eligible: list[int] = []

    for i in range(n):
        if pre_pareto[i] <= 0.0 or np.isnan(skill[i]):
            continue
        eligible.append(i)
        if skill[i] > 0.0 and float(effective_n[i]) >= SKILL_MIN_EFFECTIVE_N:
            saturated = min(float(skill[i]) / SKILL_SATURATION, 1.0)
            skill_claims[i] = (
                saturated * (SKILL_MINER_CAP * SKILL_POOL_SHARE) * float(significance[i])
            )
            pools[i] = "skill"
        else:
            pools[i] = "allow"

    skill_total = float(skill_claims.sum())
    if skill_total > SKILL_POOL_SHARE:
        skill_claims *= SKILL_POOL_SHARE / skill_total

    allowance_of = np.zeros(n, dtype=float)
    if eligible:
        raw = np.array([pre_pareto[i] for i in eligible], dtype=float)
        raw_sum = float(raw.sum())
        if raw_sum > 0.0:
            shares = ALLOWANCE_POOL_SHARE * raw / raw_sum
            for i, share in zip(eligible, shares):
                allowance_of[i] = min(float(share), ALLOWANCE_MINER_CAP)

    allowance_spent = 0.0
    skill_spent = 0.0
    for i in eligible:
        share = float(allowance_of[i])
        claim = float(skill_claims[i])
        if pools[i] == "skill" and claim >= share:
            weights[i] = claim
            allowance_spent += share
            skill_spent += claim - share
        else:
            weights[i] = share
            pools[i] = "allow"
            allowance_spent += share
    return weights, pools, allowance_spent, skill_spent


# --------------------------------------------------------------------------- #
# Pillar 2: Market-relative edge (Brier-difference + PnL, recency-weighted)   #
# --------------------------------------------------------------------------- #

def _edge_score(recs: list[_Record], weights: np.ndarray) -> float:
    """Recency-weighted market-relative edge, centered at 0 and scaled.

    Per event with a known market price on the winner (``m``) and the agent's
    prob on the winner (``a``):

        brier_diff = (1 - m)**2 - (1 - a)**2    # market error minus agent error
        pnl        = (a - m) * (1 - m)          # winner side, outcome = 1
        signal     = 0.60 * brier_diff + 0.40 * pnl

    Both inputs are in [-1, 1] and martingale-fair: if the market is calibrated,
    uninformed divergence has zero expected value in either metric. The
    spec's ratio-BSS is deliberately not used - it has a large positive
    expectation for zero-information "extremize the market" strategies when
    the market is confident.

    ``signal`` is 0 when the agent matches the market. It is multiplied by
    ``EDGE_SIGNAL_GAIN`` and clipped to [-1, 1] so a few points of realized
    edge move the composite instead of sitting on 0.5. Events without a
    market price are skipped; a miner with none falls back to 0.
    """
    num = 0.0
    den = 0.0
    for r, w in zip(recs, weights):
        m = r.market_p_win
        if m is None:
            continue
        a = r.p_win
        brier_diff = (1.0 - m) ** 2 - (1.0 - a) ** 2
        pnl = (a - m) * (1.0 - m)
        signal = EDGE_WEIGHT_BRIER_DIFF * brier_diff + EDGE_WEIGHT_PNL * pnl
        num += w * signal
        den += w
    if den <= 0.0:
        return NEUTRAL_PILLAR_SCORE
    scaled = EDGE_SIGNAL_GAIN * (num / den)
    return float(np.clip(scaled, -1.0, 1.0))


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _validate_weights() -> None:
    """Pillar weights must be non-negative with a positive sum (they are
    renormalised over that sum at compute time)."""
    if any(w < 0.0 for w in _WEIGHTS.values()) or sum(_WEIGHTS.values()) <= 0.0:
        raise ValueError(
            f"Composite pillar weights must be non-negative with a positive sum; "
            f"got {_WEIGHTS}. Adjust the WEIGHT_* constants."
        )
    pool_shares = ALLOWANCE_POOL_SHARE + SKILL_POOL_SHARE
    if not 0.0 < pool_shares <= 1.0 + 1e-9:
        raise ValueError(
            f"Allowance and skill pool shares must be positive and sum to at most 1; "
            f"got {ALLOWANCE_POOL_SHARE} + {SKILL_POOL_SHARE}."
        )
    if not 0.0 < ALLOWANCE_MINER_CAP <= ALLOWANCE_POOL_SHARE:
        raise ValueError(
            f"ALLOWANCE_MINER_CAP must be in (0, ALLOWANCE_POOL_SHARE]; "
            f"got {ALLOWANCE_MINER_CAP}."
        )
    if (
        not 0.0 < SKILL_MINER_CAP <= 1.0
        or SKILL_SATURATION <= 0.0
        or SKILL_MIN_EFFECTIVE_N < 0.0
    ):
        raise ValueError(
            f"SKILL_MINER_CAP must be in (0, 1], SKILL_SATURATION must be positive, "
            f"and SKILL_MIN_EFFECTIVE_N must be non-negative; "
            f"got cap {SKILL_MINER_CAP}, saturation {SKILL_SATURATION}, "
            f"min effective n {SKILL_MIN_EFFECTIVE_N}."
        )


def _mean_brier(recs: list[_Record], weights: np.ndarray) -> float:
    """Recency-weighted mean Brier score on the winning outcome."""
    briers = np.array([(1.0 - r.p_win) ** 2 for r in recs], dtype=float)
    return float(np.average(briers, weights=weights))


def _recency_weights(recs: list[_Record], *, now: datetime) -> np.ndarray:
    """Exponential recency weight per record (calendar-time half-life).

    The sum of these weights is also rho's effective prediction count.
    """
    ages_days = np.array(
        [max(0.0, (now - r.scored_at).total_seconds() / 86400.0) for r in recs],
        dtype=float,
    )
    return np.power(0.5, ages_days / RECENCY_HALF_LIFE_DAYS)


def compute_significance_score(
    num_miner_predictions: float,
    num_threshold_predictions: float,
    alpha: float,
    floor: float = RHO_FLOOR,
) -> float:
    """Logistic significance score blended above a configurable floor.

    ``floor + (1 - floor) * logistic(...)`` per the spec, so rho approaches
    the floor smoothly for tiny samples instead of clamping at it.
    """
    exponent = -alpha * (num_miner_predictions - num_threshold_predictions)
    exponent = float(np.clip(exponent, -60.0, 60.0))
    raw = 1.0 / (1.0 + math.exp(exponent))
    floor = float(np.clip(floor, 0.0, 1.0))
    return floor + (1.0 - floor) * raw


def _as_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _submission_age_hours(submitted_at: Optional[datetime], *, now: datetime) -> Optional[float]:
    """Hours since minerLastPredictedAt, or None when the field is null."""
    submitted_at = _as_utc(submitted_at)
    if submitted_at is None:
        return None
    return max(0.0, (now - submitted_at).total_seconds() / 3600.0)


def _inactivity_multiplier(age_hours: Optional[float]) -> float:
    """Linear fade after grace window; hard off at inactivity zero threshold.

    ``None`` (minerLastPredictedAt missing) is not evidence of silence -> 1.0.
    """
    if age_hours is None:
        return 1.0
    if age_hours <= INACTIVITY_GRACE_HOURS:
        return 1.0
    if age_hours >= INACTIVITY_ZERO_HOURS:
        return 0.0
    span = max(INACTIVITY_ZERO_HOURS - INACTIVITY_GRACE_HOURS, 1e-6)
    return float(np.clip(1.0 - ((age_hours - INACTIVITY_GRACE_HOURS) / span), 0.0, 1.0))


def _metagraph_hotkeys(metagraph) -> list[str]:
    """Hotkeys parallel to ``_metagraph_uids`` (index ``i`` is UID ``uids[i]``)."""
    hotkeys = getattr(metagraph, "hotkeys", None)
    if hotkeys is None:
        return []
    return [hk if isinstance(hk, str) else str(hk) for hk in hotkeys]


def _metagraph_uids(metagraph) -> list[int]:
    uids = getattr(metagraph, "uids", None)
    if uids is None:
        return [int(neuron.uid) for neuron in metagraph]
    try:
        return [int(u) for u in uids.tolist()]
    except AttributeError:
        return [int(u) for u in uids]


def _apply_pareto(
    scores: np.ndarray,
    *,
    mu: float = PARETO_MU,
    boost: float = PARETO_BOOST,
    gamma: float = PARETO_GAMMA,
) -> np.ndarray:
    """Shape positive scores with a leader-relative power law.

    Steps for positive scores only:
    1) Divide by the current leader score (maps leader to 1.0).
    2) Raise by ``gamma`` (``gamma`` > 1 mildly concentrates toward the leader).
    3) Scale/shift with ``mu`` + ``boost``.

    Preserves absolute quality gaps vs the leader instead of min-max rank
    knees. Non-positive scores remain at zero.
    """
    out = np.zeros_like(scores, dtype=float)
    positive_mask = scores > 0
    positive_scores = scores[positive_mask]
    if positive_scores.size == 0:
        return out

    leader = float(np.max(positive_scores))
    if leader <= 0.0:
        return out

    x = np.clip(positive_scores / leader, 0.0, 1.0)
    g = max(0.0, float(gamma))
    z = np.power(x, g) if g > 0.0 else x
    out[positive_mask] = mu + max(0.0, float(boost)) * z
    return out


def _apply_pareto_by_rank(
    scores: np.ndarray,
    *,
    mu: float = PARETO_MU,
    boost: float = PARETO_BOOST,
    knee: float = PARETO_KNEE,
    sharpness: float = PARETO_SHARPNESS,
    tail: float = PARETO_TAIL,
) -> np.ndarray:
    """Legacy rank-knee Pareto shaping (min-max normalised).

    Prefer :func:`_apply_pareto` during ramp-up when rho already separates
    low-sample miners. Switch to this when many mature miners cluster at
    similar composites and you need drift-invariant rank concentration.
    """
    out = np.zeros_like(scores, dtype=float)
    positive_mask = scores > 0
    positive_scores = scores[positive_mask]
    if positive_scores.size == 0:
        return out

    lo = float(np.min(positive_scores))
    hi = float(np.max(positive_scores))
    if hi <= lo:
        out[positive_mask] = mu
        return out

    x = (positive_scores - lo) / (hi - lo)

    k = max(0.0, float(sharpness))
    knee = float(np.clip(knee, 0.0, 1.0))
    if k <= 0.0:
        z = x
    else:
        sig = lambda v: 1.0 / (1.0 + np.exp(-v))  # noqa: E731
        a = sig(k * (x - knee))
        a0 = sig(k * (0.0 - knee))
        a1 = sig(k * (1.0 - knee))
        denom = a1 - a0
        if abs(denom) <= 1e-12:
            z = x
        else:
            z = np.clip((a - a0) / denom, 0.0, 1.0)

    t = max(0.0, float(tail))
    shaped = np.power(z, t) if t > 0.0 else np.ones_like(z)
    out[positive_mask] = mu + max(0.0, float(boost)) * shaped
    return out


def _log_score_table(
    *,
    uids: list[int],
    scores: np.ndarray,
    pre_pareto: np.ndarray,
    accuracy: np.ndarray,
    edge: np.ndarray,
    rho_by_idx: np.ndarray,
    effective_n_by_idx: np.ndarray,
    latest_age_hours_by_idx: np.ndarray,
    mean_brier_by_idx: np.ndarray,
    records_by_idx: list[list[_Record]],
    total_in_window: np.ndarray,
    invalid_in_window: np.ndarray,
    pool_by_idx: list[str],
) -> None:
    """Emit per-miner diagnostics table at INFO level every scoring tick."""
    rows: list[list[object]] = []
    # Only miners with attributable predictions in the window — not the full
    # metagraph UID map. Rank among that subset by slice weight (desc).
    ranked = sorted(
        (
            (idx, uid, float(scores[idx]))
            for idx, uid in enumerate(uids)
            if int(total_in_window[idx]) > 0
        ),
        key=lambda x: x[2],
        reverse=True,
    )
    any_invalid_gate = False
    any_stale_age = False
    any_baseline_gate = False
    any_inactive_gate = False
    for rank, (idx, uid, score) in enumerate(ranked, start=1):
        total = int(total_in_window[idx])
        invalid = int(invalid_in_window[idx])
        invalid_rate = (invalid / total) if total > 0 else 0.0
        invalid_preds = "-"
        if invalid > 0:
            invalid_preds = f"{invalid}/{total} ({invalid_rate:.2f})"
        if total > 0 and invalid_rate >= INVALID_RATE_THRESHOLD and invalid > 0:
            invalid_preds = f"{invalid_preds}*"
            any_invalid_gate = True

        recs = records_by_idx[idx]

        w_brier: object = "-"
        w_brier_val = mean_brier_by_idx[idx]
        if not np.isnan(w_brier_val):
            w_brier = f"{float(w_brier_val):.3f}"
            if float(w_brier_val) >= ACCURACY_BASELINE_BRIER:
                w_brier = f"{w_brier}!"
                any_baseline_gate = True

        m_brier: object = "-"
        priced = [r for r in recs if r.market_p_win is not None]
        if priced:
            m_brier = round(
                float(np.mean([(1.0 - r.market_p_win) ** 2 for r in priced])),
                3,
            )

        pnl = 0.0
        pnl_trades = 0
        for r in recs:
            if r.market_p_pred is None or r.market_p_pred <= 0.0:
                continue
            pnl_trades += 1
            pnl += ((1.0 / r.market_p_pred) - 1.0) if r.hit == 1 else -1.0
        roi: object = "-"
        if pnl_trades > 0:
            roi = round(pnl / float(pnl_trades), 3)

        age_h = latest_age_hours_by_idx[idx]
        age_display: object = "-"
        if not np.isnan(age_h):
            rounded_age = int(round(float(age_h)))
            age_display = str(rounded_age)
            if float(age_h) >= INACTIVITY_ZERO_HOURS:
                age_display = f"{age_display}\u2021"
                any_inactive_gate = True
            elif float(age_h) > INACTIVITY_GRACE_HOURS:
                age_display = f"{age_display}\u2020"
                any_stale_age = True

        rows.append(
            [
                rank,
                uid,
                score,
                pool_by_idx[idx],
                float(pre_pareto[idx]),
                float(accuracy[idx]),
                float(edge[idx]),
                w_brier,
                m_brier,
                roi,
                total,
                float(rho_by_idx[idx]),
                float(effective_n_by_idx[idx]),
                age_display,
                invalid_preds,
            ]
        )

    weight_sum = sum(_WEIGHTS.values())
    w_acc = WEIGHT_ACCURACY / weight_sum * 100.0
    w_edge = WEIGHT_EDGE / weight_sum * 100.0
    headers = [
        "rank",
        "uid",
        "score",
        "pool",
        "raw",
        f"acc. ({w_acc:.0f}%)",
        f"edge ({w_edge:.0f}%)",
        "w_brier",
        "m_brier",
        "roi",
        "# preds",
        "rho",
        "eff_n",
        "age_h",
        "invalid",
    ]
    floatfmt = (
        ".0f", ".0f", ".3f", "", ".3f", ".3f", ".3f",
        "", ".3f", ".3f", ".0f", ".3f", ".0f", "", "",
    )
    table = tabulate(
        rows,
        headers=headers,
        tablefmt="grid",
        stralign="right",
        floatfmt=floatfmt,
    )
    legends: list[str] = []
    if any_invalid_gate:
        legends.append("* too many invalid predictions tripped the INVALID_RATE_THRESHOLD gate. Setting score to 0.")
    if any_stale_age:
        legends.append(
            f"\u2020 {INACTIVITY_GRACE_HOURS:.0f}h < age_h < {INACTIVITY_ZERO_HOURS:.0f}h "
            "(inactivity decay region)."
        )
    if any_inactive_gate:
        legends.append(
            f"\u2021 age_h > {INACTIVITY_ZERO_HOURS:.0f}h (inactivity gate tripped, setting score to 0)."
        )
    if any_baseline_gate:
        legends.append(
            f"! w_brier is the recency-weighted (half-life={RECENCY_HALF_LIFE_DAYS:.0f}d) "
            f"mean Brier used by the baseline gate; >= {ACCURACY_BASELINE_BRIER} sets score to 0."
        )

    if legends:
        logger.info("forecasting scoring miner table:\n%s\n%s\n", table, "\n".join(legends))
    else:
        logger.info("forecasting scoring miner table:\n%s", table)
