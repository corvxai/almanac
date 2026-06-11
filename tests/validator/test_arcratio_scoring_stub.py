"""Tests for arcratio scored-predictions scoring."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.validator.scoring import score_agent_predictions


class _StubMetagraph:
    """Minimal metagraph stub that satisfies ``score_agent_predictions``."""

    def __init__(self, uids: list[int], hotkeys: list[str] | None = None) -> None:
        self._uids = uids
        self.hotkeys = hotkeys or [f"hotkey_{u}" for u in uids]

    @property
    def uids(self):
        # bittensor's metagraph exposes ``uids`` as a tensor with ``.tolist()``.
        # Provide a small shim that matches both shapes the code accepts.
        class _UidsArr:
            def __init__(self, vals: list[int]) -> None:
                self._vals = vals

            def tolist(self) -> list[int]:
                return list(self._vals)

            def __iter__(self):
                return iter(self._vals)

            def __len__(self) -> int:
                return len(self._vals)

        return _UidsArr(self._uids)


def _row(
    *,
    miner_uid: int | None,
    p_win: float,
    now: datetime,
    invalid: bool = False,
    resolved: bool = True,
):
    return type(
        "ScoredRow",
        (),
        {
            "minerUid": miner_uid,
            "predictionIsInvalid": invalid,
            "resolutionStatus": "resolved" if resolved else "voided",
            "scoredAt": now - timedelta(hours=1),
            "outcomeProbabilities": {"yes": p_win, "no": 1.0 - p_win},
            "resolvedOutcomeId": "yes",
        },
    )()


def test_empty_rows_returns_zeros() -> None:
    metagraph = _StubMetagraph(uids=[0, 1, 2])
    out = score_agent_predictions(metagraph=metagraph, scored_predictions=[])

    np.testing.assert_allclose(out, np.zeros(3))


def test_perfect_prediction_yields_score_of_one() -> None:
    metagraph = _StubMetagraph(uids=[0, 1, 2])

    now = datetime.now(timezone.utc)
    rows = [_row(miner_uid=1, p_win=1.0, now=now)]
    out = score_agent_predictions(metagraph=metagraph, scored_predictions=rows, now=now)

    assert out[0] == 0.0
    assert pytest.approx(out[1]) == 1.0
    assert out[2] == 0.0


def test_worst_prediction_yields_score_of_zero() -> None:
    metagraph = _StubMetagraph(uids=[7])

    now = datetime.now(timezone.utc)
    rows = [_row(miner_uid=7, p_win=0.0, now=now)]
    out = score_agent_predictions(metagraph=metagraph, scored_predictions=rows, now=now)

    assert pytest.approx(out[0]) == 0.0


def test_multiple_rows_average_their_briers() -> None:
    metagraph = _StubMetagraph(uids=[5])

    now = datetime.now(timezone.utc)
    rows = [
        _row(miner_uid=5, p_win=1.0, now=now),
        _row(miner_uid=5, p_win=0.0, now=now),
    ]
    out = score_agent_predictions(metagraph=metagraph, scored_predictions=rows, now=now)

    # Brier_avg = (0 + 1) / 2 = 0.5 -> score = 1 - 0.5 = 0.5
    assert pytest.approx(out[0]) == 0.5


def test_row_outside_rolling_window_is_ignored() -> None:
    metagraph = _StubMetagraph(uids=[0])

    now = datetime.now(timezone.utc)
    old = _row(miner_uid=0, p_win=1.0, now=now)
    old.scoredAt = now - timedelta(days=100)
    out = score_agent_predictions(metagraph=metagraph, scored_predictions=[old], now=now)
    np.testing.assert_allclose(out, np.zeros(1))


def test_unresolved_row_is_ignored() -> None:
    metagraph = _StubMetagraph(uids=[0])

    now = datetime.now(timezone.utc)
    rows = [_row(miner_uid=0, p_win=1.0, now=now, resolved=False)]
    out = score_agent_predictions(metagraph=metagraph, scored_predictions=rows, now=now)
    np.testing.assert_allclose(out, np.zeros(1))


def test_unmapped_uid_is_skipped_not_counted() -> None:
    metagraph = _StubMetagraph(uids=[0])

    now = datetime.now(timezone.utc)
    rows = [_row(miner_uid=None, p_win=1.0, now=now)]
    out = score_agent_predictions(metagraph=metagraph, scored_predictions=rows, now=now)
    np.testing.assert_allclose(out, np.zeros(1))
