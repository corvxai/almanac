"""Almanac Forecasting scoring from orchestrator scored predictions.

Composite score per miner is a weighted blend of three pillars, computed over
that miner's most recent predictions (``MAX_EVENTS_PER_MINER``) within the
orchestrator fetch window:

    1. Accuracy     (weight 0.50) - hardness-curved Brier closeness to truth.
                     The anchor pillar.
    2. Calibration  (weight 0.30) - do stated confidences match realized hit
                     rates? (1 - ECE over reliability bins)
    3. Edge         (weight 0.20) - did the agent beat the market price at
                     prediction time? Brier-difference + PnL blend.

Pillar weights are renormalised over their sum at compute time, so only their relative sizes matter.

Pipeline per scoring tick (stateless; a pure function of the fetched rows):

    gates -> per-event pillar scores -> recency-weighted pillar means ->
    composite -> rho significance -> inactivity -> Pareto shaping

Eligibility gates (any failure -> score 0):
    * Invalid-rate gate : >= INVALID_RATE_THRESHOLD of the miner's capped
      window is invalid -> 0. Invalids below the gate are excluded from every
      pillar; they are never scored at a baseline (which would quietly reward
      laziness).
    * Minimum-sample gate: fewer than MIN_VALID_PREDICTIONS_HARD_FLOOR valid
      predictions in the capped window -> 0.
    * Baseline-accuracy gate: recency-weighted mean Brier at or above
      ACCURACY_BASELINE_BRIER (the always-0.5 coin baseline) -> 0. Gating on
      mean Brier rather than on the accuracy pillar matters: the pillar's
      per-event max(0, ...) floor would let a zero-information "lottery"
      miner (predict 1.0/0.0 everywhere) keep half its events at full score,
      while its mean Brier of ~0.5 is plainly worse than the coin baseline.
    * Inactivity gate: latest scored prediction older than
      INACTIVITY_ZERO_HOURS -> 0, linear fade after INACTIVITY_GRACE_HOURS.

Post-gate scoring details:
    * Recency-weighted pillar means use exponential calendar-time decay
      (``RECENCY_HALF_LIFE_DAYS``), not an EMA. The function is stateless and
      order-independent across validator ticks.
    * Rho significance multiplies the composite by a logistic ramp in
      recency-weighted effective sample size (``RHO_THRESHOLD_PREDICTIONS``),
      so new or low-volume miners earn proportionally less until they build
      history.
    * Inactivity applies after rho: linear fade beyond ``INACTIVITY_GRACE_HOURS``,
      hard zero beyond ``INACTIVITY_ZERO_HOURS``.
    * Pareto shaping maps positive scores with a leader-relative power law
      (``PARETO_GAMMA``) into ``[PARETO_MU, PARETO_MU + PARETO_BOOST]``.
      :func:`_apply_pareto_by_rank` is an alternate rank-knee curve for
      environments where many miners cluster at similar composites.
    * Edge pillar blends bounded Brier-difference vs market (60%) with
      winner-side PnL (40%).

Coverage (research-trace quality) is not a scoring pillar. Invalid
predictions are marked orchestrator-side via ``predictionIsInvalid`` and
enforced by the invalid-rate gate.

Returns a ``np.ndarray`` of per-miner scores aligned to the metagraph UIDs.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
from tabulate import tabulate

logger = logging.getLogger("forecasting.scoring")


# Orchestrator fetch/cutoff window. This is a data-retrieval bound, not a
# scoring dial: it just needs to be wide enough to cover MAX_EVENTS_PER_MINER
# at current assignment cadence and keep the recency decay well-populated.
DEFAULT_ROLLING_WINDOW_DAYS = 30

# --------------------------------------------------------------------------- #
# Tunable scoring constants.                                                   #
# --------------------------------------------------------------------------- #

# Composite pillar weights. Renormalised over their sum at compute time, so
# only relative sizes matter (currently 4/9 accuracy, 2/9 calibration, 3/9 edge).
WEIGHT_ACCURACY = 0.50
WEIGHT_CALIBRATION = 0.30
WEIGHT_EDGE = 0.20

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

# Time-based significance (rho): confidence from recency-weighted sample size.
# Threshold calibrated for orchestrator cadence of ~50 predictions/miner/day
# at ~7 days to rho≈0.5 and ~12+ days toward saturation (eff_n≈350-500).
# Retune proportionally if the orchestrator's daily cap changes.
RHO_THRESHOLD_PREDICTIONS = 350.0
RHO_ALPHA = 0.06
RHO_FLOOR = 0.10                      # minimum rho once miner passes hard gates

# Inactivity policy (separate from rho): fade stale miners then switch off.
# This is the anti-perpetuity guard - nobody coasts on old history.
INACTIVITY_GRACE_HOURS = 24.0         # no inactivity penalty while within grace
INACTIVITY_ZERO_HOURS = 72.0          # score forced to 0 after this staleness age

# Accuracy hardening. Per-event score is max(0, 1 - (brier/baseline)**gamma):
# zero at baseline-or-worse Brier, sharpened reward gradient above it.
ACCURACY_BASELINE_BRIER = 0.25        # Brier of always predicting 0.5
ACCURACY_HARDNESS_GAMMA = 1.7

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
NUM_CALIBRATION_BINS = 5          # reliability bins for the calibration pillar
NEUTRAL_PILLAR_SCORE = 0.5        # edge fallback when no market prices exist

_WEIGHTS = {
    "accuracy": WEIGHT_ACCURACY,
    "calibration": WEIGHT_CALIBRATION,
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
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return forecasting composite scores from orchestrator ``scored-predictions`` rows.

    Output is a float ``np.ndarray`` indexed by metagraph UID position.
    Positive composite scores are shaped via :func:`_apply_pareto` before
    return; gated miners remain at 0. With ``return_pre_pareto`` the raw
    (pre-shaping) composite vector is returned alongside, for diagnostics
    and simulation tooling.
    """
    _validate_weights()

    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=rolling_window_days)

    uids = _metagraph_uids(metagraph)
    n = len(uids)
    if n == 0:
        return np.zeros(0, dtype=float)
    uid_to_idx = {uid: idx for idx, uid in enumerate(uids)}

    # Attributable rows per miner: (scored_at, record-or-None, invalid_flag).
    # record is None for invalid-flagged and unparseable rows; both stay in
    # the capped population so the invalid-rate gate and the scored pillars
    # always observe the same window.
    rows_by_idx: list[list[tuple[datetime, Optional[_Record], bool]]] = [
        [] for _ in range(n)
    ]

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

        if getattr(item, "predictionIsInvalid", None) is True:
            rows_by_idx[idx].append((scored_at, None, True))
            continue
        rows_by_idx[idx].append((scored_at, _parse_record(item, scored_at=scored_at), False))

    # Diagnostics + outputs, aligned to metagraph idx.
    scores = np.zeros(n, dtype=float)
    accuracy = np.zeros(n, dtype=float)
    calibration = np.zeros(n, dtype=float)
    edge = np.zeros(n, dtype=float)
    rho_by_idx = np.zeros(n, dtype=float)
    effective_n_by_idx = np.zeros(n, dtype=float)
    latest_age_hours_by_idx = np.full(n, np.nan, dtype=float)
    total_in_window = np.zeros(n, dtype=int)
    invalid_in_window = np.zeros(n, dtype=int)
    records_by_idx: list[list[_Record]] = [[] for _ in range(n)]

    gated_invalid = 0
    gated_sample = 0
    gated_baseline = 0
    gated_inactive = 0
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
        acc = _accuracy_score(recs, rec_weights)
        cal = _calibration_score(recs)
        edg = _edge_score(recs, rec_weights)
        mean_brier = _mean_brier(recs, rec_weights)
        accuracy[idx] = acc
        calibration[idx] = cal
        edge[idx] = edg

        effective_n = float(rec_weights.sum())
        rho = compute_significance_score(
            num_miner_predictions=effective_n,
            num_threshold_predictions=RHO_THRESHOLD_PREDICTIONS,
            alpha=RHO_ALPHA,
        )
        latest_age_hours = _latest_prediction_age_hours(recs, now=now)
        inactivity_mult = _inactivity_multiplier(latest_age_hours)
        effective_n_by_idx[idx] = effective_n
        rho_by_idx[idx] = rho
        if latest_age_hours is not None:
            latest_age_hours_by_idx[idx] = latest_age_hours

        # Do not reward miners whose Brier is baseline-or-worse. Gate on mean
        # Brier, NOT on the accuracy pillar: the pillar's per-event floor
        # would let lottery-style extremizing (half the events at full score,
        # half at zero) pass despite a mean Brier far worse than baseline.
        if mean_brier >= ACCURACY_BASELINE_BRIER:
            gated_baseline += 1
            continue

        if inactivity_mult <= 0.0:
            gated_inactive += 1
            continue

        composite = (
            WEIGHT_ACCURACY * acc
            + WEIGHT_CALIBRATION * cal
            + WEIGHT_EDGE * edg
        ) / weight_sum
        scores[idx] = float(np.clip(composite * rho * inactivity_mult, 0.0, 1.0))
        scored_miners += 1

    pre_pareto = scores.copy()
    scores = _apply_pareto(scores)

    logger.info(
        "forecasting composite scoring: %d valid rows | %d miners scored, "
        "%d gated (invalid-rate), %d gated (min-sample-hard-floor), %d gated (baseline-accuracy), "
        "%d gated (inactive) "
        "| cutoff=%s window=%dd cap=%d",
        sum(len(r) for r in records_by_idx),
        scored_miners,
        gated_invalid,
        gated_sample,
        gated_baseline,
        gated_inactive,
        cutoff.isoformat(),
        rolling_window_days,
        MAX_EVENTS_PER_MINER,
    )
    _log_score_table(
        uids=uids,
        scores=scores,
        pre_pareto=pre_pareto,
        accuracy=accuracy,
        calibration=calibration,
        edge=edge,
        rho_by_idx=rho_by_idx,
        effective_n_by_idx=effective_n_by_idx,
        latest_age_hours_by_idx=latest_age_hours_by_idx,
        records_by_idx=records_by_idx,
        total_in_window=total_in_window,
        invalid_in_window=invalid_in_window,
    )
    if return_pre_pareto:
        return scores, pre_pareto
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
        # Fall back to the winning side; hit is then trivially the p_win calibration.
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
# Pillar 1: Accuracy (hardness-curved Brier, recency-weighted mean)           #
# --------------------------------------------------------------------------- #

def _accuracy_score(recs: list[_Record], weights: np.ndarray) -> float:
    """Recency-weighted mean of per-event accuracy scores.

    Per event: ``max(0, 1 - (brier / B0) ** gamma)`` with
    ``brier = (1 - p_win)^2`` and ``B0 = ACCURACY_BASELINE_BRIER``. Zero at
    baseline-or-worse Brier, sharpened gradient above it.

    The spec's per-event "Brier < 0.5" accuracy gate is subsumed here: any
    event with brier >= 0.25 already scores exactly 0. Gate-failing events are
    deliberately NEVER dropped from the window - their zero stays in the
    denominator and their (usually negative) edge contribution still counts.
    Dropping them would let a miner's worst predictions vanish from their own
    average.
    """
    per_event = np.empty(len(recs), dtype=float)
    for i, r in enumerate(recs):
        brier = (1.0 - r.p_win) ** 2
        per_event[i] = max(
            0.0,
            1.0 - (brier / ACCURACY_BASELINE_BRIER) ** ACCURACY_HARDNESS_GAMMA,
        )
    return float(np.average(per_event, weights=weights))


# --------------------------------------------------------------------------- #
# Pillar 2: Calibration (reliability of stated confidence)                    #
# --------------------------------------------------------------------------- #

def _calibration_score(recs: list[_Record]) -> float:
    """1 - Expected Calibration Error over the miner's predicted-side confidences.

    For each prediction we take the probability the agent placed on the side
    it actually chose (``p_pred``) and whether that side won (``hit``).
    Predictions are binned by ``p_pred`` (half-open bins, 1.0 in the top bin);
    within each bin we compare mean predicted confidence to the realized hit
    rate. ECE is the sample-weighted mean absolute gap; empty bins contribute
    nothing. A perfectly calibrated miner scores 1.0.
    """
    bin_edges = np.linspace(0.0, 1.0, NUM_CALIBRATION_BINS + 1)
    preds = np.array([r.p_pred for r in recs], dtype=float)
    hits = np.array([r.hit for r in recs], dtype=float)

    bin_ids = np.clip(
        np.digitize(preds, bin_edges[1:-1], right=False),
        0,
        NUM_CALIBRATION_BINS - 1,
    )

    total = len(recs)
    ece = 0.0
    for b in range(NUM_CALIBRATION_BINS):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            continue
        gap = abs(float(preds[mask].mean()) - float(hits[mask].mean()))
        ece += (count / total) * gap

    return float(np.clip(1.0 - ece, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Pillar 3: Market-relative edge (Brier-difference + PnL, recency-weighted)   #
# --------------------------------------------------------------------------- #

def _edge_score(recs: list[_Record], weights: np.ndarray) -> float:
    """Recency-weighted market-relative edge, mapped to [0, 1] with 0.5 neutral.

    Per event with a known market price on the winner (``m``) and the agent's
    prob on the winner (``a``):

        brier_diff = (1 - m)**2 - (1 - a)**2    # market error minus agent error
        pnl        = (a - m) * (1 - m)          # winner side, outcome = 1

    Both are in [-1, 1] and martingale-fair: if the market is calibrated,
    uninformed divergence has zero expected value in either metric. The
    spec's ratio-BSS is deliberately not used - it has a large positive
    expectation for zero-information "extremize the market" strategies when
    the market is confident.

    Events without a market price are skipped; a miner with none falls back
    to the neutral pillar score.
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
        event_edge = (
            EDGE_WEIGHT_BRIER_DIFF * (brier_diff + 1.0) / 2.0
            + EDGE_WEIGHT_PNL * (pnl + 1.0) / 2.0
        )
        num += w * event_edge
        den += w
    if den <= 0.0:
        return NEUTRAL_PILLAR_SCORE
    return float(np.clip(num / den, 0.0, 1.0))


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


def _latest_prediction_age_hours(records: list[_Record], *, now: datetime) -> Optional[float]:
    """Hours since miner's latest valid prediction, or None if no records."""
    if not records:
        return None
    latest = max(r.scored_at for r in records)
    return max(0.0, (now - latest).total_seconds() / 3600.0)


def _inactivity_multiplier(age_hours: Optional[float]) -> float:
    """Linear fade after grace window; hard off at inactivity zero threshold."""
    if age_hours is None:
        return 0.0
    if age_hours <= INACTIVITY_GRACE_HOURS:
        return 1.0
    if age_hours >= INACTIVITY_ZERO_HOURS:
        return 0.0
    span = max(INACTIVITY_ZERO_HOURS - INACTIVITY_GRACE_HOURS, 1e-6)
    return float(np.clip(1.0 - ((age_hours - INACTIVITY_GRACE_HOURS) / span), 0.0, 1.0))


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
    calibration: np.ndarray,
    edge: np.ndarray,
    rho_by_idx: np.ndarray,
    effective_n_by_idx: np.ndarray,
    latest_age_hours_by_idx: np.ndarray,
    records_by_idx: list[list[_Record]],
    total_in_window: np.ndarray,
    invalid_in_window: np.ndarray,
) -> None:
    """Emit per-miner diagnostics table at INFO level every scoring tick."""
    rows: list[list[object]] = []
    ranked = sorted(
        ((idx, uid, float(scores[idx])) for idx, uid in enumerate(uids)),
        key=lambda x: x[2],
        reverse=True,
    )
    any_invalid_gate = False
    any_stale_age = False
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

        raw_brier: object = "-"
        recs = records_by_idx[idx]
        if recs:
            raw_brier = float(np.mean([(1.0 - r.p_win) ** 2 for r in recs]))

        pnl = 0.0
        pnl_trades = 0
        for r in recs:
            if r.market_p_pred is None or r.market_p_pred <= 0.0:
                continue
            pnl_trades += 1
            pnl += ((1.0 / r.market_p_pred) - 1.0) if r.hit == 1 else -1.0
        roi: object = "-"
        if pnl_trades > 0:
            roi = pnl / float(pnl_trades)

        age_h = latest_age_hours_by_idx[idx]
        age_display: object = "-"
        if not np.isnan(age_h):
            rounded_age = int(round(float(age_h)))
            age_display = str(rounded_age)
            if float(age_h) > INACTIVITY_GRACE_HOURS:
                age_display = f"{age_display}\u2020"
                any_stale_age = True

        rows.append(
            [
                rank,
                uid,
                score,
                float(pre_pareto[idx]),
                float(accuracy[idx]),
                float(calibration[idx]),
                float(edge[idx]),
                raw_brier,
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
    w_cal = WEIGHT_CALIBRATION / weight_sum * 100.0
    w_edge = WEIGHT_EDGE / weight_sum * 100.0
    headers = [
        "rank",
        "uid",
        "score",
        "raw",
        f"acc. ({w_acc:.0f}%)",
        f"calib. ({w_cal:.0f}%)",
        f"edge ({w_edge:.0f}%)",
        "brier",
        "roi",
        "# preds",
        "rho",
        "eff_n",
        "age_h",
        "invalid",
    ]
    floatfmt = (
        ".0f", ".0f", ".3f", ".3f", ".3f", ".3f", ".3f",
        ".3f", ".3f", ".0f", ".3f", ".0f", "", "",
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
        legends.append(f"\u2020 age_h > {INACTIVITY_GRACE_HOURS:.0f}h (inactivity decay region).")

    if legends:
        logger.info("forecasting scoring miner table:\n%s\n%s", table, "\n".join(legends))
    else:
        logger.info("forecasting scoring miner table:\n%s", table)
