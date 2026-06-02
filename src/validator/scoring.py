"""Arcratio forecasting (Brier) scoring.

v1 stub. Walks the trace store, looks at every resolved trace inside the
rolling window, and accumulates a per-UID Brier score. The result is
converted into a non-negative weight vector of length
``len(metagraph.uids)`` that the main validator blends with the Almanac
mechanism's weight vector.

This module is deliberately small. The interesting calibration / penalty
work (timing, coverage, conviction, etc.) lives outside v1 — once miner
agents start submitting traces tagged with their UID/hotkey we can extend
this in a follow-up.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from src.core.schemas import EvidenceDigest
from src.storage.json_store import JsonTraceStore
from src.validator import uid_map

logger = logging.getLogger("arcratio.scoring")


DEFAULT_ROLLING_WINDOW_DAYS = 30


def score_arcratio(
    *,
    metagraph,
    trace_store: JsonTraceStore,
    rolling_window_days: int = DEFAULT_ROLLING_WINDOW_DAYS,
    now: Optional[datetime] = None,
) -> np.ndarray:
    """Return a non-negative arcratio weight vector aligned to ``metagraph.uids``.

    For each resolved trace within ``rolling_window_days``:

    - Resolve the miner UID via ``uid_map.resolve``. Skip + log-once if
      unresolved.
    - Compute Brier ``(prediction - outcome)**2``.
    - Average per UID, then map ``mean_brier`` -> ``score = 1 - mean_brier``
      clipped to ``[0, 1]``. Aligns with the chain's expectation of a
      higher-is-better score.

    UIDs with no qualifying traces get a score of 0.0. The returned vector
    is the *raw* score vector — normalisation happens inside the blend.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=rolling_window_days)

    uids = _metagraph_uids(metagraph)
    n = len(uids)
    if n == 0:
        return np.zeros(0, dtype=float)

    sum_sq_err = np.zeros(n, dtype=float)
    counts = np.zeros(n, dtype=int)

    for trace in _iter_traces(trace_store):
        if not _is_eligible_for_brier(trace, cutoff):
            continue

        uid = uid_map.resolve(trace, metagraph)
        if uid is None:
            continue

        try:
            idx = uids.index(uid)
        except ValueError:
            continue

        outcome = trace.resolution_record.resolution_outcome
        prediction = trace.prediction_output.final_probability
        if outcome is None:
            continue

        sq_err = (float(prediction) - (1.0 if outcome else 0.0)) ** 2
        sum_sq_err[idx] += sq_err
        counts[idx] += 1

    scores = np.zeros(n, dtype=float)
    nonzero = counts > 0
    scores[nonzero] = np.clip(1.0 - (sum_sq_err[nonzero] / counts[nonzero]), 0.0, 1.0)

    total_traces = int(counts.sum())
    scored_uids = int(nonzero.sum())
    logger.info(
        "arcratio scoring: %d traces across %d UIDs (cutoff=%s, window=%dd)",
        total_traces,
        scored_uids,
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


def _iter_traces(trace_store: JsonTraceStore):
    return trace_store._iter_all_traces()  # noqa: SLF001 — sole API for now


def _is_eligible_for_brier(trace: EvidenceDigest, cutoff: datetime) -> bool:
    """Return True iff ``trace`` is a resolved trace inside the rolling window."""
    rr = trace.resolution_record
    if not rr.resolved or rr.resolution_outcome is None:
        return False

    resolved_at = rr.resolution_timestamp or trace.event_snapshot.resolution_deadline
    if resolved_at is None:
        return False
    if resolved_at.tzinfo is None:
        resolved_at = resolved_at.replace(tzinfo=timezone.utc)
    if resolved_at < cutoff:
        return False

    return True
