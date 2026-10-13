from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np

from src.validator.forecasting import scoring

T = datetime(2026, 10, 13, 2, 7, tzinfo=timezone.utc)
H = timedelta(hours=1)


class _StubUids:
    def __init__(self, vals: list[int]) -> None:
        self._vals = vals

    def tolist(self) -> list[int]:
        return list(self._vals)


class _StubMetagraph:
    def __init__(self, uids: list[int]) -> None:
        self.uids = _StubUids(uids)
        self.hotkeys = [f"hotkey_{uid}" for uid in uids]


def _rows(
    uid: int,
    n: int,
    p_win: float,
    price: float = 0.5,
    scored_at: datetime = T,
    invalid: bool = False,
) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            minerUid=uid,
            minerHotkey=f"hotkey_{uid}",
            predictionIsInvalid=invalid,
            resolutionStatus="resolved",
            scoredAt=scored_at,
            marketId=f"mkt_{uid}_{i}",
            outcomeProbabilities={"yes": p_win, "no": 1.0 - p_win},
            predictedOutcomeId="yes",
            resolvedOutcomeId="yes",
            outcomePricesAtPrediction={"yes": price, "no": 1.0 - price},
        )
        for i in range(n)
    ]


def _score(
    rows: list[SimpleNamespace],
    uid: int,
    now: datetime,
    last_predicted: datetime | None = None,
) -> float:
    stamp = now - H if last_predicted is None else last_predicted
    for r in rows:
        r.minerLastPredictedAt = stamp
    out = scoring.score_agent_predictions(
        metagraph=_StubMetagraph([uid]),
        scored_predictions=rows,
        rolling_window_days=30,
        now=now,
    )
    return float(out[0])


def _live_effective_n(rows: list[SimpleNamespace], now: datetime) -> float:
    recs = [
        scoring._parse_record(r, scored_at=r.scoredAt) for r in rows if not r.predictionIsInvalid
    ]
    return float(scoring._recency_weights(recs, now=now).sum())


def test_miner_over_50_at_last_resolution_stays_in_pool_as_clock_runs() -> None:
    rows = _rows(226, 51, 0.8)
    assert _live_effective_n(rows, T + 6 * H) > scoring.SKILL_MIN_EFFECTIVE_N
    assert _live_effective_n(rows, T + 12 * H) < scoring.SKILL_MIN_EFFECTIVE_N
    early = _score(rows, 226, T + 6 * H)
    late = _score(rows, 226, T + 12 * H)
    assert scoring.ALLOWANCE_MINER_CAP < late < early


def test_miner_under_50_at_last_resolution_draws_allowance() -> None:
    rows = _rows(242, 49, 0.8)
    assert _score(rows, 242, T + H) == scoring.ALLOWANCE_MINER_CAP


def test_gate_still_requires_positive_skill() -> None:
    rows = _rows(3, 51, 0.79, price=0.8)
    assert _score(rows, 3, T + H) == scoring.ALLOWANCE_MINER_CAP


def test_gate_freezes_48_hours_then_decays() -> None:
    rows = _rows(160, 51, 0.8)
    assert _live_effective_n(rows, T + 48 * H) < scoring.SKILL_MIN_EFFECTIVE_N
    assert _score(rows, 160, T + 48 * H) > scoring.ALLOWANCE_MINER_CAP
    assert _score(rows, 160, T + 72 * H) == scoring.ALLOWANCE_MINER_CAP


def test_invalid_newest_row_does_not_move_reference() -> None:
    rows = _rows(18, 50, 0.8) + _rows(18, 1, 0.8, scored_at=T + 10 * H, invalid=True)
    assert _live_effective_n(rows, T + 12 * H) < scoring.SKILL_MIN_EFFECTIVE_N
    assert _score(rows, 18, T + 12 * H) > scoring.ALLOWANCE_MINER_CAP


def test_staleness_is_left_to_inactivity_gate() -> None:
    rows = _rows(222, 51, 0.8)
    assert _score(rows, 222, T + 30 * H, last_predicted=T) > scoring.ALLOWANCE_MINER_CAP
    assert _score(rows, 222, T + 73 * H, last_predicted=T) == 0.0


def test_allocate_slice_default_gate_is_live_effective_n() -> None:
    pre = np.array([0.5])
    skill = np.array([0.5])
    sig = np.array([0.5])
    _, pools, _, _ = scoring._allocate_slice(pre, skill, sig, np.array([49.9]))
    assert pools[0] == "allow"
    _, pools, _, _ = scoring._allocate_slice(
        pre, skill, sig, np.array([49.9]), gate_effective_n=np.array([51.9])
    )
    assert pools[0] == "skill"
