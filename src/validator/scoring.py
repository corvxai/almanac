"""Arcratio forecasting scoring from orchestrator scored predictions."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger("arcratio.scoring")


DEFAULT_ROLLING_WINDOW_DAYS = 30


def score_agent_predictions(
    *,
    metagraph,
    scored_predictions,
    rolling_window_days: int = DEFAULT_ROLLING_WINDOW_DAYS,
    now: Optional[datetime] = None,
) -> np.ndarray:
    """Return arcratio scores from orchestrator ``scored-predictions`` DTO rows."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=rolling_window_days)

    uids = _metagraph_uids(metagraph)
    n = len(uids)
    if n == 0:
        return np.zeros(0, dtype=float)
    uid_to_idx = {uid: idx for idx, uid in enumerate(uids)}

    sum_sq_err = np.zeros(n, dtype=float)
    counts = np.zeros(n, dtype=int)

    for item in scored_predictions:
        if getattr(item, "predictionIsInvalid", None) is True:
            continue
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

        outcome_probs = getattr(item, "outcomeProbabilities", None) or {}
        resolved_outcome_id = getattr(item, "resolvedOutcomeId", None)
        if not isinstance(outcome_probs, dict) or not isinstance(resolved_outcome_id, str):
            continue
        if resolved_outcome_id not in outcome_probs:
            continue

        try:
            p_win = float(outcome_probs[resolved_outcome_id])
        except (TypeError, ValueError):
            continue
        if p_win < 0.0 or p_win > 1.0:
            continue

        sq_err = (1.0 - p_win) ** 2
        sum_sq_err[idx] += sq_err
        counts[idx] += 1

    scores = np.zeros(n, dtype=float)
    nonzero = counts > 0
    scores[nonzero] = np.clip(1.0 - (sum_sq_err[nonzero] / counts[nonzero]), 0.0, 1.0)

    logger.info(
        "arcratio scored-predictions scoring: %d rows across %d UIDs (cutoff=%s, window=%dd)",
        int(counts.sum()),
        int(nonzero.sum()),
        cutoff.isoformat(),
        rolling_window_days,
    )
    return scores


def _metagraph_uids(metagraph) -> list[int]:
    uids = getattr(metagraph, "uids", None)
    if uids is None:
        return []
    try:
        return [int(u) for u in uids.tolist()]
    except AttributeError:
        return [int(u) for u in uids]
