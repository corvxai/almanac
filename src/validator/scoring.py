"""Arcratio forecasting scoring from orchestrator scored predictions.

Composite score per miner is a weighted blend of four pillars, computed over a
rolling window of resolved predictions:

    1. Accuracy        - Brier-style closeness to truth (the anchor).
    2. Calibration     - do stated confidences match realized hit rates?
    3. Early signal    - did the agent beat the market price at prediction time,
                         weighted by how much uncertainty was actually available?
    4. Ensemble        - does the agent move the swarm consensus toward truth
                         (leave-one-out contribution within each market)?

Eligibility and confidence handling before final shaping:
    * Invalid-rate gate : >= INVALID_RATE_THRESHOLD of a miner's in-window
                          predictions are invalid -> score 0.
    * Minimum-sample gate: fewer than MIN_VALID_PREDICTIONS_HARD_FLOOR valid
                          predictions in window -> score 0.
    * Baseline-accuracy gate: baseline-or-worse Brier -> score 0.
    * Time-based significance: score is scaled by ``rho`` from a recency-weighted
      effective prediction count within the rolling window.

Invalid predictions that don't trip the rate gate are simply excluded from every
pillar; they are never scored at a baseline (which would quietly reward laziness).

The public contract (signature, np.ndarray-aligned-to-metagraph-UIDs return,
``_metagraph_uids`` helper) is unchanged from the previous implementation.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
from tabulate import tabulate

logger = logging.getLogger("arcratio.scoring")


DEFAULT_ROLLING_WINDOW_DAYS = 30

# --------------------------------------------------------------------------- #
# Tunable scoring constants. Edit freely; weights are validated to sum to 1.0. #
# --------------------------------------------------------------------------- #

# Composite pillar weights. MUST sum to 1.0 (validated at call time).
WEIGHT_ACCURACY = 0.40
WEIGHT_CALIBRATION = 0.20
WEIGHT_EARLY_SIGNAL = 0.30
WEIGHT_ENSEMBLE = 0.10

# Eligibility gates.
MIN_VALID_PREDICTIONS_HARD_FLOOR = 10 # tiny-sample floor; below this scoring is too noisy
INVALID_RATE_THRESHOLD = 0.15         # >= this fraction invalid in window -> 0

# Time-based significance (rho) controls confidence from sample size + recency.
RHO_THRESHOLD_PREDICTIONS = 25.0      # effective count where rho ~= 0.5 (relaxed)
RHO_ALPHA = 0.06                      # gentler logistic ramp around the threshold
RHO_HALF_LIFE_DAYS = 14.0             # slower recency decay for effective count
RHO_FLOOR = 0.10                      # minimum rho once miner passes hard gates

# Inactivity policy (separate from rho): fade stale miners then switch off.
INACTIVITY_GRACE_HOURS = 24.0         # no inactivity penalty while within grace
INACTIVITY_ZERO_HOURS = 72.0          # score forced to 0 after this staleness age

# Accuracy hardening.
ACCURACY_BASELINE_BRIER = 0.25
ACCURACY_HARDNESS_GAMMA = 1.7

# Post-composite Pareto-like shaping (applied to positive scores only).
PARETO_MU = 1.0           # Floor value assigned to the lowest positive score.
PARETO_BOOST = 0.25       # Total vertical lift above MU for top-ranked scores.
PARETO_KNEE = 0.7        # Normalized rank where the curve begins rising faster.
PARETO_SHARPNESS = 18.0   # Higher values make the knee transition steeper.
PARETO_TAIL = 1.35        # >1 steepens the very top tail; 1 is linear in knee-space.

# Pillar internals.
NUM_CALIBRATION_BINS = 5          # reliability bins for the calibration pillar
NEUTRAL_PILLAR_SCORE = 0.5        # fallback when a pillar is undefined for a miner

_WEIGHTS = {
    "accuracy": WEIGHT_ACCURACY,
    "calibration": WEIGHT_CALIBRATION,
    "early_signal": WEIGHT_EARLY_SIGNAL,
    "ensemble": WEIGHT_ENSEMBLE,
}


@dataclass
class _Record:
    """A single valid, resolved, in-window prediction, normalized for scoring."""

    idx: int                  # metagraph index for this miner
    market_id: str
    p_win: float              # agent prob assigned to the resolved (winning) outcome
    p_pred: float             # agent prob assigned to its own predicted outcome
    hit: int                  # 1 if predicted outcome == resolved outcome else 0
    scored_at: datetime       # when this resolved prediction was scored
    market_p_win: Optional[float]  # market price on the winning outcome at prediction
    market_p_pred: Optional[float]  # market price on predicted side at prediction


def score_agent_predictions(
    *,
    metagraph,
    scored_predictions,
    rolling_window_days: int = DEFAULT_ROLLING_WINDOW_DAYS,
    now: Optional[datetime] = None,
    return_pre_pareto: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return arcratio composite scores from orchestrator ``scored-predictions`` rows.

    Output is a float ``np.ndarray`` indexed by metagraph UID position. Positive
    composite scores are shaped via :func:`_apply_pareto` (see ``PARETO_*``
    constants) before return; gated miners remain at 0.
    """
    _validate_weights()

    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=rolling_window_days)

    uids = _metagraph_uids(metagraph)
    n = len(uids)
    if n == 0:
        return np.zeros(0, dtype=float)
    uid_to_idx = {uid: idx for idx, uid in enumerate(uids)}

    # In-window tallies for the invalid-rate gate (count BOTH valid and invalid).
    total_in_window = np.zeros(n, dtype=int)
    invalid_in_window = np.zeros(n, dtype=int)

    # Valid records, grouped per miner and per market.
    records_by_idx: list[list[_Record]] = [[] for _ in range(n)]
    records_by_market: dict[str, list[_Record]] = defaultdict(list)

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

        # This row is in-window and attributable -> counts toward the ratio gate.
        total_in_window[idx] += 1
        if getattr(item, "predictionIsInvalid", None) is True:
            invalid_in_window[idx] += 1
            continue

        record = _parse_record(item, idx, scored_at=scored_at)
        if record is None:
            continue
        records_by_idx[idx].append(record)
        records_by_market[record.market_id].append(record)

    # Per-pillar scores (each returns an [0, 1] array aligned to metagraph idx).
    accuracy = _accuracy_scores(records_by_idx, n)
    calibration = _calibration_scores(records_by_idx, n)
    early_signal = _early_signal_scores(records_by_idx, n)
    ensemble = _ensemble_scores(records_by_idx, records_by_market, n)

    scores = np.zeros(n, dtype=float)
    rho_by_idx = np.zeros(n, dtype=float)
    effective_n_by_idx = np.zeros(n, dtype=float)
    inactivity_mult_by_idx = np.zeros(n, dtype=float)
    latest_age_hours_by_idx = np.full(n, np.nan, dtype=float)
    gated_invalid = 0
    gated_sample = 0
    gated_baseline = 0
    gated_inactive = 0
    scored_miners = 0

    for idx in range(n):
        total = total_in_window[idx]
        if total == 0:
            continue

        if invalid_in_window[idx] / total >= INVALID_RATE_THRESHOLD:
            gated_invalid += 1
            continue

        recs = records_by_idx[idx]
        valid_count = len(recs)
        if valid_count < MIN_VALID_PREDICTIONS_HARD_FLOOR:
            gated_sample += 1
            continue

        # Do not reward miners whose Brier is baseline-or-worse.
        if accuracy[idx] <= 0.0:
            gated_baseline += 1
            continue

        effective_n = _effective_prediction_count(recs, now=now)
        rho = compute_significance_score(
            num_miner_predictions=effective_n,
            num_threshold_predictions=RHO_THRESHOLD_PREDICTIONS,
            alpha=RHO_ALPHA,
        )
        latest_age_hours = _latest_prediction_age_hours(recs, now=now)
        inactivity_mult = _inactivity_multiplier(latest_age_hours)
        effective_n_by_idx[idx] = effective_n
        rho_by_idx[idx] = rho
        inactivity_mult_by_idx[idx] = inactivity_mult
        if latest_age_hours is not None:
            latest_age_hours_by_idx[idx] = latest_age_hours

        if inactivity_mult <= 0.0:
            gated_inactive += 1
            continue

        raw_score = (
            WEIGHT_ACCURACY * accuracy[idx]
            + WEIGHT_CALIBRATION * _centered_pillar(float(calibration[idx]))
            + WEIGHT_EARLY_SIGNAL * _centered_pillar(float(early_signal[idx]))
            + WEIGHT_ENSEMBLE * _centered_pillar(float(ensemble[idx]))
        )
        scores[idx] = max(0.0, raw_score * rho * inactivity_mult)
        scored_miners += 1

    scores = np.clip(scores, 0.0, 1.0)
    pre_pareto = scores.copy()
    scores = _apply_pareto(scores)

    logger.info(
        "arcratio composite scoring: %d valid rows | %d miners scored, "
        "%d gated (invalid-rate), %d gated (min-sample-hard-floor), %d gated (baseline-accuracy), "
        "%d gated (inactive) "
        "| cutoff=%s window=%dd",
        sum(len(r) for r in records_by_idx),
        scored_miners,
        gated_invalid,
        gated_sample,
        gated_baseline,
        gated_inactive,
        cutoff.isoformat(),
        rolling_window_days,
    )
    _log_score_table(
        uids=uids,
        scores=scores,
        accuracy=accuracy,
        calibration=calibration,
        early_signal=early_signal,
        ensemble=ensemble,
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

def _parse_record(item, idx: int, scored_at: datetime) -> Optional[_Record]:
    """Normalize a single DTO row into a ``_Record``, or ``None`` if unusable."""
    outcome_probs = getattr(item, "outcomeProbabilities", None) or {}
    resolved_outcome_id = getattr(item, "resolvedOutcomeId", None)
    predicted_outcome_id = getattr(item, "predictedOutcomeId", None)
    market_id = getattr(item, "marketId", None)

    if not isinstance(outcome_probs, dict):
        return None
    if not isinstance(resolved_outcome_id, str) or resolved_outcome_id not in outcome_probs:
        return None
    if not isinstance(market_id, str):
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

    # Market baseline on the winning outcome at prediction time (early signal).
    prices_at_pred = getattr(item, "outcomePricesAtPrediction", None) or {}
    market_p_win = None
    market_p_pred = None
    if isinstance(prices_at_pred, dict) and resolved_outcome_id in prices_at_pred:
        market_p_win = _coerce_prob(prices_at_pred.get(resolved_outcome_id))
    if isinstance(prices_at_pred, dict) and isinstance(predicted_outcome_id, str):
        if predicted_outcome_id in prices_at_pred:
            market_p_pred = _coerce_prob(prices_at_pred.get(predicted_outcome_id))

    return _Record(
        idx=idx,
        market_id=market_id,
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
# Pillar 1: Accuracy (Brier-skill above baseline, hardness-curved)            #
# --------------------------------------------------------------------------- #

def _accuracy_scores(records_by_idx: list[list[_Record]], n: int) -> np.ndarray:
    """Brier skill above baseline, then hardness-curved.

    Per miner we compute mean Brier ``B = mean((1 - p_win)^2)`` and map it to
    skill above baseline:

        s = max(0, (B0 - B) / B0), where B0 = ACCURACY_BASELINE_BRIER

    Then apply a hardness curve:

        accuracy = s ** ACCURACY_HARDNESS_GAMMA

    This enforces zero reward for baseline-or-worse Brier and emphasizes
    genuinely strong probabilistic accuracy.
    """
    out = np.zeros(n, dtype=float)
    for idx, recs in enumerate(records_by_idx):
        if not recs:
            continue
        errs = [(1.0 - r.p_win) ** 2 for r in recs]
        brier = float(np.mean(errs))
        baseline = ACCURACY_BASELINE_BRIER
        if baseline <= 0.0:
            continue
        skill = max(0.0, (baseline - brier) / baseline)
        out[idx] = float(np.clip(skill, 0.0, 1.0) ** ACCURACY_HARDNESS_GAMMA)
    return out


# --------------------------------------------------------------------------- #
# Pillar 2: Calibration (reliability of stated confidence)                    #
# --------------------------------------------------------------------------- #

def _calibration_scores(records_by_idx: list[list[_Record]], n: int) -> np.ndarray:
    """1 - Expected Calibration Error over the miner's predicted-side confidences.

    For each prediction we take the probability the agent placed on the side it
    actually chose (``p_pred``) and whether that side won (``hit``). Predictions
    are binned by ``p_pred``; within each bin we compare mean predicted confidence
    to the realized hit rate. ECE is the sample-weighted mean absolute gap.

    A perfectly calibrated miner (confidence == realized frequency) scores 1.0.
    """
    out = np.full(n, NEUTRAL_PILLAR_SCORE, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, NUM_CALIBRATION_BINS + 1)

    for idx, recs in enumerate(records_by_idx):
        if not recs:
            continue
        preds = np.array([r.p_pred for r in recs], dtype=float)
        hits = np.array([r.hit for r in recs], dtype=float)

        # bin index in [0, NUM_CALIBRATION_BINS - 1]
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

        out[idx] = float(np.clip(1.0 - ece, 0.0, 1.0))
    return out


# --------------------------------------------------------------------------- #
# Pillar 3: Early signal (edge over the market, uncertainty-weighted)         #
# --------------------------------------------------------------------------- #

def _early_signal_scores(records_by_idx: list[list[_Record]], n: int) -> np.ndarray:
    """Uncertainty-weighted Brier edge of the agent over the market at prediction.

    Per prediction with a known market price on the winner (``market_p_win = m``)
    and the agent's prob on the winner (``p_win = a``):

        edge = (1 - m)**2 - (1 - a)**2     # market error minus agent error
        u    = 4 * m * (1 - m)             # available uncertainty (1 at 0.5, 0 at extremes)

    The pillar aggregates ``sum(u * edge) / sum(u)`` so near-resolved markets
    (where the market is already ~right and there's no room to beat it) barely
    move the score in either direction. Result is mapped to [0, 1] with 0.5 =
    matched the market on average, >0.5 = beat it.

    Predictions without a market price are skipped; a miner with none falls back
    to the neutral pillar score.
    """
    out = np.full(n, NEUTRAL_PILLAR_SCORE, dtype=float)
    for idx, recs in enumerate(records_by_idx):
        weighted_edge_sum = 0.0
        weight_sum = 0.0
        for r in recs:
            if r.market_p_win is None:
                continue
            m = r.market_p_win
            a = r.p_win
            edge = (1.0 - m) ** 2 - (1.0 - a) ** 2
            u = 4.0 * m * (1.0 - m)
            weighted_edge_sum += u * edge
            weight_sum += u
        if weight_sum <= 0.0:
            continue  # no usable signal (all markets at extremes or no prices)
        avg_edge = weighted_edge_sum / weight_sum  # in [-1, 1]
        out[idx] = float(np.clip((avg_edge + 1.0) / 2.0, 0.0, 1.0))
    return out


# --------------------------------------------------------------------------- #
# Pillar 4: Ensemble contribution (leave-one-out, within market)              #
# --------------------------------------------------------------------------- #

def _ensemble_scores(
    records_by_idx: list[list[_Record]],
    records_by_market: dict[str, list[_Record]],
    n: int,
) -> np.ndarray:
    """Does the agent pull the swarm consensus toward truth?

    For each market with >= 2 valid agents, and each agent in it:

        others_mean = mean of OTHER agents' p_win on the winning outcome
        all_mean    = mean of ALL agents' p_win (including this one)
        contribution = (1 - others_mean)**2 - (1 - all_mean)**2

    Positive contribution means including this agent lowered the ensemble's error
    (moved consensus toward the correct answer). Per miner we average their
    contributions across multi-agent markets and map to [0, 1] with 0.5 = neutral.

    Miners whose markets never had a second valid agent fall back to the neutral
    pillar score (they can't be measured against a consensus they're alone in).
    """
    contrib_sum = np.zeros(n, dtype=float)
    contrib_count = np.zeros(n, dtype=int)

    for recs in records_by_market.values():
        if len(recs) < 2:
            continue
        p_wins = np.array([r.p_win for r in recs], dtype=float)
        total = p_wins.sum()
        k = len(recs)
        all_mean = total / k
        ensemble_err_with = (1.0 - all_mean) ** 2

        for i, r in enumerate(recs):
            others_mean = (total - p_wins[i]) / (k - 1)
            ensemble_err_without = (1.0 - others_mean) ** 2
            contribution = ensemble_err_without - ensemble_err_with  # in [-1, 1]
            contrib_sum[r.idx] += contribution
            contrib_count[r.idx] += 1

    out = np.full(n, NEUTRAL_PILLAR_SCORE, dtype=float)
    measured = contrib_count > 0
    avg_contrib = np.zeros(n, dtype=float)
    avg_contrib[measured] = contrib_sum[measured] / contrib_count[measured]
    out[measured] = np.clip((avg_contrib[measured] + 1.0) / 2.0, 0.0, 1.0)
    return out


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _validate_weights() -> None:
    """Ensure the four composite weights sum to 1.0 (within tolerance)."""
    total = sum(_WEIGHTS.values())
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(
            f"Composite scoring weights must sum to 1.0; got {total:.6f} "
            f"({_WEIGHTS}). Adjust the WEIGHT_* constants."
        )

def _centered_pillar(value: float) -> float:
    """Map [0, 1] to [-1, 1] so neutral 0.5 contributes 0."""
    return float(np.clip((value - NEUTRAL_PILLAR_SCORE) / NEUTRAL_PILLAR_SCORE, -1.0, 1.0))


def compute_significance_score(
    num_miner_predictions: float,
    num_threshold_predictions: float,
    alpha: float,
    floor: float = RHO_FLOOR,
) -> float:
    """Relaxed logistic significance score with configurable floor."""
    exponent = -alpha * (num_miner_predictions - num_threshold_predictions)
    exponent = float(np.clip(exponent, -60.0, 60.0))
    denominator = 1.0 + math.exp(exponent)
    raw = 1.0 / denominator
    floor = float(np.clip(floor, 0.0, 1.0))
    return max(floor, raw)


def _effective_prediction_count(records: list[_Record], *, now: datetime) -> float:
    """Recency-weighted prediction count using exponential half-life decay."""
    if not records:
        return 0.0
    half_life = max(RHO_HALF_LIFE_DAYS, 1e-6)
    decay_rate = math.log(2.0) / half_life
    total = 0.0
    for r in records:
        age_days = max(0.0, (now - r.scored_at).total_seconds() / 86400.0)
        total += math.exp(-decay_rate * age_days)
    return total


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
        return []
    try:
        return [int(u) for u in uids.tolist()]
    except AttributeError:
        return [int(u) for u in uids]


def _apply_pareto(
    scores: np.ndarray,
    *,
    mu: float = PARETO_MU,
    boost: float = PARETO_BOOST,
    knee: float = PARETO_KNEE,
    sharpness: float = PARETO_SHARPNESS,
    tail: float = PARETO_TAIL,
) -> np.ndarray:
    """Shape positive scores with a soft-knee Pareto-like curve.

    Steps for positive scores only:
    1) Min-max normalize into [0, 1].
    2) Apply a logistic knee (controlled by ``knee`` + ``sharpness``).
    3) Raise by ``tail`` to tune top-end steepness.
    4) Scale/shift with ``mu`` + ``boost``.

    This yields a controllable hockey-stick curve while preserving monotonic rank.
    Non-positive scores remain at zero.
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
        # Normalize logistic output so x=0 maps to 0 and x=1 maps to 1.
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
    accuracy: np.ndarray,
    calibration: np.ndarray,
    early_signal: np.ndarray,
    ensemble: np.ndarray,
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
                float(accuracy[idx]),
                float(calibration[idx]),
                float(early_signal[idx]),
                float(ensemble[idx]),
                raw_brier,
                #pnl,
                roi,
                total,
                float(rho_by_idx[idx]),
                float(effective_n_by_idx[idx]),
                age_display,
                invalid_preds,
            ]
        )

    w_acc = WEIGHT_ACCURACY * 100.0
    w_cal = WEIGHT_CALIBRATION * 100.0
    w_early = WEIGHT_EARLY_SIGNAL * 100.0
    w_ens = WEIGHT_ENSEMBLE * 100.0
    headers = [
        "rank",
        "uid",
        "score",
        f"acc. ({w_acc:.0f}%)",
        f"calib. ({w_cal:.0f}%)",
        f"e sig. ({w_early:.0f}%)",
        f"ens. ({w_ens:.0f}%)",
        "brier",
        #"PnL",
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
        logger.info("arcratio scoring miner table:\n%s\n%s", table, "\n".join(legends))
    else:
        logger.info("arcratio scoring miner table:\n%s", table)
    