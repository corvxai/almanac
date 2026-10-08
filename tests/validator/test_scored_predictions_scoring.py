from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np

from src.validator.forecasting import scoring


class _StubUids:
    def __init__(self, vals: list[int]) -> None:
        self._vals = vals

    def tolist(self) -> list[int]:
        return list(self._vals)


class _StubMetagraph:
    def __init__(self, uids: list[int]) -> None:
        self.uids = _StubUids(uids)
        self.hotkeys = [f"hotkey_{uid}" for uid in uids]


def _row(
    *,
    uid: int,
    p_win: float,
    now: datetime,
    invalid: bool = False,
    resolved: bool = True,
    market_p_win: float | None = None,
):
    return SimpleNamespace(
        minerUid=uid,
        minerHotkey=f"hotkey_{uid}",
        predictionIsInvalid=invalid,
        resolutionStatus="resolved" if resolved else "voided",
        scoredAt=now - timedelta(hours=1),
        marketId=f"mkt_{uid}",
        outcomeProbabilities={"yes": p_win, "no": 1.0 - p_win},
        predictedOutcomeId="yes",
        resolvedOutcomeId="yes",
        outcomePricesAtPrediction=(
            None if market_p_win is None else {"yes": market_p_win, "no": 1.0 - market_p_win}
        ),
    )


def test_score_agent_predictions_basic() -> None:
    now = datetime.now(timezone.utc)
    metagraph = _StubMetagraph([0, 1, 2])
    # 0.51 stays under saturation against a 0.50 market. 0.52 reaches it.
    # Both clear the sample floor, so the stronger forecast claims more.
    n = int(scoring.SKILL_MIN_EFFECTIVE_N) + 10
    rows = [_row(uid=1, p_win=0.51, market_p_win=0.50, now=now) for _ in range(n)] + [
        _row(uid=2, p_win=0.52, market_p_win=0.50, now=now) for _ in range(n)
    ]
    out = scoring.score_agent_predictions(
        metagraph=metagraph,
        scored_predictions=rows,
        rolling_window_days=30,
        now=now,
    )
    assert out[0] == 0.0
    assert 0.0 < out[1] < out[2]


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
