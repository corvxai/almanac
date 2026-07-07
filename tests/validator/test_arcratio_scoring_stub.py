"""Tests for arcratio scored-predictions scoring."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.validator.scoring import PARETO_BOOST, PARETO_MU, _apply_pareto, _apply_pareto_by_rank, score_agent_predictions


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
            "marketId": f"mkt_{miner_uid}",
            "outcomeProbabilities": {"yes": p_win, "no": 1.0 - p_win},
            "predictedOutcomeId": "yes",
            "resolvedOutcomeId": "yes",
        },
    )()


def test_empty_rows_returns_zeros() -> None:
    metagraph = _StubMetagraph(uids=[0, 1, 2])
    out = score_agent_predictions(metagraph=metagraph, scored_predictions=[])

    np.testing.assert_allclose(out, np.zeros(3))


def test_perfect_prediction_yields_pareto_leader_score() -> None:
    metagraph = _StubMetagraph(uids=[0, 1, 2])

    now = datetime.now(timezone.utc)
    rows = [_row(miner_uid=1, p_win=1.0, now=now) for _ in range(12)]
    out = score_agent_predictions(metagraph=metagraph, scored_predictions=rows, now=now)

    assert out[0] == 0.0
    assert out[1] == pytest.approx(PARETO_MU + PARETO_BOOST)
    assert out[2] == 0.0


def test_worst_prediction_yields_score_of_zero() -> None:
    metagraph = _StubMetagraph(uids=[7])

    now = datetime.now(timezone.utc)
    rows = [_row(miner_uid=7, p_win=0.0, now=now)]
    out = score_agent_predictions(metagraph=metagraph, scored_predictions=rows, now=now)

    assert pytest.approx(out[0]) == 0.0


def test_multiple_rows_average_then_pareto_floor() -> None:
    metagraph = _StubMetagraph(uids=[5])

    now = datetime.now(timezone.utc)
    rows = [
        _row(miner_uid=5, p_win=1.0, now=now),
        _row(miner_uid=5, p_win=0.0, now=now),
    ] * 6
    out = score_agent_predictions(metagraph=metagraph, scored_predictions=rows, now=now)

    # Mean Brier = 0.5 is baseline-or-worse, so miner gets no positive reward.
    assert out[0] == pytest.approx(0.0)


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


def test_apply_pareto_stretches_positive_scores() -> None:
    raw = np.array([0.0, 0.2, 0.4, 0.8], dtype=float)
    out = _apply_pareto(raw, mu=0.0, boost=1.0, gamma=2.0)

    np.testing.assert_allclose(out[0], 0.0)
    np.testing.assert_allclose(out[3], 1.0)
    np.testing.assert_allclose(out[1], 0.0625)
    assert out[1] < out[2] < out[3]


def test_apply_pareto_by_rank_stretches_positive_scores() -> None:
    raw = np.array([0.0, 0.4, 0.6, 0.8], dtype=float)
    out = _apply_pareto_by_rank(raw, mu=1.0, boost=0.30, knee=0.5, sharpness=8.0, tail=1.2)

    np.testing.assert_allclose(out[0], 0.0)
    np.testing.assert_allclose(out[1], 1.0)
    np.testing.assert_allclose(out[3], 1.3)
    assert out[1] < out[2] < out[3]


def test_apply_pareto_defaults_cap_at_mu_plus_boost() -> None:
    raw = np.array([0.1, 0.2, 0.3], dtype=float)
    out = _apply_pareto(raw)
    assert np.max(out) == pytest.approx(PARETO_MU + PARETO_BOOST)


def test_baseline_brier_gets_zero_score() -> None:
    metagraph = _StubMetagraph(uids=[1])
    now = datetime.now(timezone.utc)
    rows = [_row(miner_uid=1, p_win=0.5, now=now) for _ in range(12)]
    out = score_agent_predictions(metagraph=metagraph, scored_predictions=rows, now=now)
    np.testing.assert_allclose(out, np.array([0.0]))


def test_time_based_rho_rewards_recent_volume() -> None:
    metagraph = _StubMetagraph(uids=[1, 2])
    now = datetime.now(timezone.utc)

    low_volume = [_row(miner_uid=1, p_win=0.9, now=now) for _ in range(12)]
    high_volume = [_row(miner_uid=2, p_win=0.9, now=now) for _ in range(40)]

    out = score_agent_predictions(
        metagraph=metagraph,
        scored_predictions=low_volume + high_volume,
        now=now,
    )
    assert out[1] > out[0] > 0.0


def test_inactivity_gate_zeroes_stale_miner() -> None:
    metagraph = _StubMetagraph(uids=[1, 2])
    now = datetime.now(timezone.utc)

    fresh = [_row(miner_uid=1, p_win=0.9, now=now) for _ in range(20)]
    stale = [_row(miner_uid=2, p_win=0.9, now=now) for _ in range(20)]
    for row in stale:
        row.scoredAt = now - timedelta(hours=120)

    out = score_agent_predictions(
        metagraph=metagraph,
        scored_predictions=fresh + stale,
        now=now,
    )
    assert out[0] > 0.0
    assert out[1] == 0.0


def test_unmapped_uid_is_skipped_not_counted() -> None:
    metagraph = _StubMetagraph(uids=[0])

    now = datetime.now(timezone.utc)
    rows = [_row(miner_uid=None, p_win=1.0, now=now)]
    out = score_agent_predictions(metagraph=metagraph, scored_predictions=rows, now=now)
    np.testing.assert_allclose(out, np.zeros(1))
