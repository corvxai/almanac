from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np

from src.validator import scoring


class _StubUids:
    def __init__(self, vals: list[int]) -> None:
        self._vals = vals

    def tolist(self) -> list[int]:
        return list(self._vals)


class _StubMetagraph:
    def __init__(self, uids: list[int]) -> None:
        self.uids = _StubUids(uids)


def _row(
    *,
    uid: int,
    p_win: float,
    now: datetime,
    invalid: bool = False,
    resolved: bool = True,
):
    return SimpleNamespace(
        minerUid=uid,
        predictionIsInvalid=invalid,
        resolutionStatus="resolved" if resolved else "voided",
        scoredAt=now - timedelta(hours=1),
        outcomeProbabilities={"yes": p_win, "no": 1.0 - p_win},
        resolvedOutcomeId="yes",
    )


def test_score_agent_predictions_basic() -> None:
    now = datetime.now(timezone.utc)
    metagraph = _StubMetagraph([0, 1, 2])
    rows = [
        _row(uid=1, p_win=1.0, now=now),
        _row(uid=1, p_win=0.0, now=now),
        _row(uid=2, p_win=0.75, now=now),
    ]
    out = scoring.score_agent_predictions(
        metagraph=metagraph,
        scored_predictions=rows,
        rolling_window_days=30,
        now=now,
    )
    # uid=1 mean sq_err=(0^2 + 1^2)/2=0.5 => score=0.5
    # uid=2 mean sq_err=(1-0.75)^2=0.0625 => score=0.9375
    np.testing.assert_allclose(out, np.array([0.0, 0.5, 0.9375]))


def test_score_agent_predictions_filters_invalid_rows() -> None:
    now = datetime.now(timezone.utc)
    metagraph = _StubMetagraph([0])
    rows = [
        _row(uid=0, p_win=1.0, now=now, invalid=True),
        _row(uid=0, p_win=1.0, now=now, resolved=False),
        SimpleNamespace(
            minerUid=0,
            predictionIsInvalid=False,
            resolutionStatus="resolved",
            scoredAt=now - timedelta(days=1000),
            outcomeProbabilities={"yes": 1.0},
            resolvedOutcomeId="yes",
        ),
    ]
    out = scoring.score_agent_predictions(
        metagraph=metagraph,
        scored_predictions=rows,
        rolling_window_days=30,
        now=now,
    )
    np.testing.assert_allclose(out, np.array([0.0]))
