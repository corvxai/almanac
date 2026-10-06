"""Tests for Almanac Forecasting scored-predictions scoring."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.validator.forecasting.scoring import (
    ACCURACY_SKILL_ALLOWANCE,
    ALLOWANCE_MINER_CAP,
    ALLOWANCE_POOL_SHARE,
    CALIBRATION_BASELINE_ECE,
    EDGE_SIGNAL_GAIN,
    SKILL_MINER_CAP,
    SKILL_POOL_SHARE,
    SKILL_SATURATION,
    PARETO_BOOST,
    PARETO_MU,
    RHO_ALPHA,
    RHO_FLOOR,
    RHO_THRESHOLD_PREDICTIONS,
    _Record,
    WEIGHT_ACCURACY,
    WEIGHT_CALIBRATION,
    WEIGHT_EDGE,
    _accuracy_score,
    _allocate_slice,
    _apply_pareto,
    _apply_pareto_by_rank,
    _calibration_score,
    _edge_score,
    _mean_brier,
    _parse_record,
    compute_significance_score,
    score_agent_predictions,
)


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
    predicted_outcome_id: str = "yes",
    resolved_outcome_id: str = "yes",
    miner_hotkey: str | None = None,
    market_p_win: float | None = None,
):
    # ``p_win`` is the probability on the resolved outcome.
    if resolved_outcome_id == "yes":
        probs = {"yes": p_win, "no": 1.0 - p_win}
    else:
        probs = {"yes": 1.0 - p_win, "no": p_win}
    return type(
        "ScoredRow",
        (),
        {
            "minerUid": miner_uid,
            "minerHotkey": (
                miner_hotkey
                if miner_hotkey is not None
                else (f"hotkey_{miner_uid}" if miner_uid is not None else None)
            ),
            "predictionIsInvalid": invalid,
            "resolutionStatus": "resolved" if resolved else "voided",
            "scoredAt": now - timedelta(hours=1),
            "marketId": f"mkt_{miner_uid}",
            "outcomeProbabilities": probs,
            "predictedOutcomeId": predicted_outcome_id,
            "resolvedOutcomeId": resolved_outcome_id,
            "outcomePricesAtPrediction": (
                None
                if market_p_win is None
                else (
                    {"yes": market_p_win, "no": 1.0 - market_p_win}
                    if resolved_outcome_id == "yes"
                    else {"yes": 1.0 - market_p_win, "no": market_p_win}
                )
            ),
        },
    )()


def test_empty_rows_returns_zeros() -> None:
    metagraph = _StubMetagraph(uids=[0, 1, 2])
    out = score_agent_predictions(metagraph=metagraph, scored_predictions=[])

    np.testing.assert_allclose(out, np.zeros(3))


def test_perfect_prediction_is_paid_from_the_skill_pool() -> None:
    metagraph = _StubMetagraph(uids=[0, 1, 2])

    now = datetime.now(timezone.utc)
    rows = [_row(miner_uid=1, p_win=1.0, market_p_win=0.5, now=now) for _ in range(12)]
    out = score_agent_predictions(metagraph=metagraph, scored_predictions=rows, now=now)

    assert out[0] == 0.0
    assert 0.0 < out[1] < 1.0
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


def test_rho_leaves_floor_near_twenty_and_saturates_near_one_seventy() -> None:
    rho = lambda n: compute_significance_score(
        n, RHO_THRESHOLD_PREDICTIONS, RHO_ALPHA
    )
    assert RHO_FLOOR < rho(15) < 0.15
    assert rho(20) > rho(15)
    assert rho(RHO_THRESHOLD_PREDICTIONS) == pytest.approx(0.55, abs=0.01)
    assert rho(170) > 0.95


def test_time_based_rho_rewards_recent_volume() -> None:
    metagraph = _StubMetagraph(uids=[1, 2])
    now = datetime.now(timezone.utc)

    low_volume = [_row(miner_uid=1, p_win=0.9, market_p_win=0.5, now=now) for _ in range(18)]
    high_volume = [_row(miner_uid=2, p_win=0.9, market_p_win=0.5, now=now) for _ in range(150)]

    out = score_agent_predictions(
        metagraph=metagraph,
        scored_predictions=low_volume + high_volume,
        now=now,
    )
    assert out[1] > out[0] > 0.0


def test_inactivity_gate_uses_last_submission_not_scored_at() -> None:
    metagraph = _StubMetagraph(uids=[1, 2])
    now = datetime.now(timezone.utc)

    fresh = [_row(miner_uid=1, p_win=0.9, market_p_win=0.5, now=now) for _ in range(20)]
    stale = [_row(miner_uid=2, p_win=0.9, market_p_win=0.5, now=now) for _ in range(20)]
    for row in fresh:
        row.minerLastPredictedAt = now - timedelta(hours=1)
        row.scoredAt = now - timedelta(hours=120)
    for row in stale:
        row.minerLastPredictedAt = now - timedelta(hours=120)
        row.scoredAt = now - timedelta(hours=1)

    out = score_agent_predictions(
        metagraph=metagraph,
        scored_predictions=fresh + stale,
        now=now,
    )
    assert out[0] > 0.0
    assert out[1] == 0.0


def test_mismatched_hotkey_is_dropped_and_does_not_refresh_inactivity() -> None:
    metagraph = _StubMetagraph(uids=[1])
    now = datetime.now(timezone.utc)

    registered = [_row(miner_uid=1, p_win=0.9, now=now) for _ in range(20)]
    for row in registered:
        row.minerLastPredictedAt = now - timedelta(hours=120)
        row.scoredAt = now - timedelta(hours=1)

    # A recycled UID still carrying the previous occupant's history, plus one
    # fresh row from a hotkey that is not the one registered at UID 1.
    deregistered = [_row(miner_uid=1, p_win=1.0, now=now, miner_hotkey="old_hotkey") for _ in range(20)]
    for row in deregistered:
        row.minerLastPredictedAt = now - timedelta(hours=1)

    out = score_agent_predictions(
        metagraph=metagraph,
        scored_predictions=registered + deregistered,
        now=now,
    )
    assert out[0] == 0.0


def test_only_the_registered_hotkey_is_scored() -> None:
    metagraph = _StubMetagraph(uids=[1])
    now = datetime.now(timezone.utc)
    registered = [_row(miner_uid=1, p_win=0.9, market_p_win=0.5, now=now) for _ in range(20)]
    other = [
        _row(miner_uid=1, p_win=1.0, now=now, miner_hotkey="other_hotkey")
        for _ in range(20)
    ]
    missing = [_row(miner_uid=1, p_win=1.0, now=now, miner_hotkey="") for _ in range(20)]

    matched = score_agent_predictions(
        metagraph=metagraph,
        scored_predictions=registered,
        now=now,
    )
    mixed = score_agent_predictions(
        metagraph=metagraph,
        scored_predictions=registered + other + missing,
        now=now,
    )
    np.testing.assert_allclose(mixed, matched)
    assert mixed[0] > 0.0


def test_null_last_submission_does_not_zero_miner(caplog) -> None:
    metagraph = _StubMetagraph(uids=[1])
    now = datetime.now(timezone.utc)
    rows = [_row(miner_uid=1, p_win=0.9, market_p_win=0.5, now=now) for _ in range(20)]
    for row in rows:
        row.minerLastPredictedAt = None

    with caplog.at_level("WARNING", logger="forecasting.scoring"):
        out = score_agent_predictions(metagraph=metagraph, scored_predictions=rows, now=now)
    assert out[0] > 0.0
    assert "1 miner(s) missing minerLastPredictedAt" in caplog.text


def _record(
    *,
    p_win: float,
    now: datetime,
    p_pred: float | None = None,
    hit: int | None = None,
    market_p_win: float | None = None,
) -> _Record:
    if p_pred is None:
        p_pred = p_win
    if hit is None:
        hit = 1 if p_win >= 0.5 else 0
    return _Record(
        p_win=p_win,
        p_pred=p_pred,
        hit=hit,
        scored_at=now,
        market_p_win=market_p_win,
        market_p_pred=market_p_win,
    )


def test_accuracy_curve_rewards_the_true_probability() -> None:
    now = datetime.now(timezone.utc)
    # Six wins and four losses of the chosen side. The market price on the
    # winner is 0.5 on every event, so the skill denominator is fixed and
    # mean Brier is minimized by stating 0.60.
    def stated(q: float) -> list[_Record]:
        hits = [_record(p_win=q, now=now, p_pred=q, hit=1, market_p_win=0.5) for _ in range(6)]
        misses = [
            _record(p_win=1.0 - q, now=now, p_pred=q, hit=0, market_p_win=0.5)
            for _ in range(4)
        ]
        return hits + misses

    weights = np.ones(10)
    honest = _accuracy_score(stated(0.60), weights)
    assert honest > 0.0
    assert honest > _accuracy_score(stated(0.55), weights)
    assert honest > _accuracy_score(stated(0.65), weights)
    assert _accuracy_score(
        [_record(p_win=1.0, now=now, market_p_win=0.5)], np.ones(1)
    ) == pytest.approx(1.0)

    # Exact copy, and a perfect copy of a certainty, score the allowance.
    assert _accuracy_score(
        [_record(p_win=0.6, now=now, market_p_win=0.6)], np.ones(1)
    ) == pytest.approx(ACCURACY_SKILL_ALLOWANCE)
    assert _accuracy_score(
        [_record(p_win=1.0, now=now, market_p_win=1.0)], np.ones(1)
    ) == pytest.approx(ACCURACY_SKILL_ALLOWANCE)
    # One allowance worse than the market is the new zero.
    # Market brier 0.25; agent brier = 0.25 * (1 + allowance).
    worse_p = 1.0 - (0.25 * (1.0 + ACCURACY_SKILL_ALLOWANCE)) ** 0.5
    assert _accuracy_score(
        [_record(p_win=worse_p, now=now, market_p_win=0.5)], np.ones(1)
    ) == pytest.approx(0.0, abs=1e-3)
    # No market price is 0, and the allowance does not apply.
    assert _accuracy_score([_record(p_win=1.0, now=now)], np.ones(1)) == pytest.approx(0.0)
    # Market was certain and right; agent was certain and wrong.
    assert _accuracy_score(
        [_record(p_win=0.0, now=now, market_p_win=1.0)], np.ones(1)
    ) == pytest.approx(-1.0 + ACCURACY_SKILL_ALLOWANCE)

    # Two events, market 0.90, agent extremized to 0.99. The ratio of means
    # is negative. The mean of the per-event ratios is positive (~0.39) and
    # must not be what this pillar returns.
    win = _record(p_win=0.99, now=now, market_p_win=0.90)
    lose = _record(p_win=0.01, now=now, market_p_win=0.10)
    assert _accuracy_score([win, lose], np.ones(2)) < 0.0


def test_market_copy_scores_the_allowance_and_a_better_forecast_leads() -> None:
    metagraph = _StubMetagraph(uids=[1, 2])
    now = datetime.now(timezone.utc)
    copies = [_row(miner_uid=1, p_win=0.6, market_p_win=0.6, now=now) for _ in range(12)]
    better = [_row(miner_uid=2, p_win=0.8, market_p_win=0.6, now=now) for _ in range(12)]
    out = score_agent_predictions(
        metagraph=metagraph,
        scored_predictions=copies + better,
        now=now,
    )
    assert 0.0 < out[0] < out[1]


def test_unpriced_forecast_scores_zero() -> None:
    metagraph = _StubMetagraph(uids=[1])
    now = datetime.now(timezone.utc)
    rows = [_row(miner_uid=1, p_win=1.0, now=now) for _ in range(12)]
    out = score_agent_predictions(metagraph=metagraph, scored_predictions=rows, now=now)
    assert out[0] == pytest.approx(0.0)


def test_baseline_gate_zeros_positive_composite() -> None:
    metagraph = _StubMetagraph(uids=[1])
    now = datetime.now(timezone.utc)
    # Agent Brier is 0.36, worse than a coin flip, but better than a market
    # that put 0.20 on the winner. Skill and edge would pay; the gate returns 0.
    rows = [_row(miner_uid=1, p_win=0.4, market_p_win=0.2, now=now) for _ in range(10)]
    recs = [_parse_record(row, scored_at=row.scoredAt) for row in rows]
    assert all(rec is not None for rec in recs)
    weights = np.ones(len(recs))
    assert _mean_brier(recs, weights) >= 0.25
    blend = (
        WEIGHT_ACCURACY * _accuracy_score(recs, weights)
        + WEIGHT_CALIBRATION * _calibration_score(recs, weights)
        + WEIGHT_EDGE * _edge_score(recs, weights)
    )
    assert blend > 0.0

    out = score_agent_predictions(metagraph=metagraph, scored_predictions=rows, now=now)
    assert out[0] == pytest.approx(0.0)


def test_calibration_is_excess_over_baseline() -> None:
    now = datetime.now(timezone.utc)
    # Stated 0.90 and always won: ECE = 0.10, which is the baseline, so the pillar is 0.
    at_baseline = [_record(p_win=0.9, now=now, p_pred=0.9, hit=1) for _ in range(20)]
    assert _calibration_score(at_baseline, np.ones(20)) == pytest.approx(0.0)

    # Stated 0.60 and hit 60% of the time: ECE = 0, excess = the baseline itself.
    calibrated = []
    for i in range(10):
        calibrated.append(_record(p_win=0.6, now=now, p_pred=0.6, hit=1 if i < 6 else 0))
    assert _calibration_score(calibrated, np.ones(10)) == pytest.approx(CALIBRATION_BASELINE_ECE)

    # Recent misses count more than old hits, so this miner is worse than the unweighted ECE.
    confident = _record(p_win=0.9, now=now, p_pred=0.9, hit=1)
    recent_miss = _record(p_win=0.1, now=now, p_pred=0.9, hit=0)
    unweighted = _calibration_score([confident, recent_miss], np.ones(2))
    recency = _calibration_score([confident, recent_miss], np.array([0.1, 1.0]))
    assert recency < unweighted


def test_edge_is_centered_on_the_market_and_scaled() -> None:
    now = datetime.now(timezone.utc)
    matched = _record(p_win=0.6, now=now, market_p_win=0.6)
    assert _edge_score([matched], np.ones(1)) == pytest.approx(0.0)

    # Agent put 0.70 on a winner the market priced at 0.60.
    # brier_diff = 0.16 - 0.09 = 0.07; pnl = 0.10 * 0.40 = 0.04
    # signal = 0.6 * 0.07 + 0.4 * 0.04 = 0.058; gain 10 keeps it inside [-1, 1].
    better = _record(p_win=0.7, now=now, market_p_win=0.6)
    assert _edge_score([better], np.ones(1)) == pytest.approx(0.058 * EDGE_SIGNAL_GAIN)

    assert _edge_score([_record(p_win=0.7, now=now)], np.ones(1)) == pytest.approx(0.0)


def test_allowance_pool_caps_a_single_miner_and_does_not_redistribute() -> None:
    pre = np.array([0.05, 0.01, 0.0])
    skill = np.array([0.0, -0.02, np.nan])
    sig = np.ones(3)
    weights, pools = _allocate_slice(pre, skill, sig)

    assert pools == ["allow", "allow", "-"]
    assert weights[0] == pytest.approx(ALLOWANCE_MINER_CAP)
    assert weights[1] == pytest.approx(ALLOWANCE_POOL_SHARE * 0.01 / 0.06)
    assert weights.sum() < ALLOWANCE_POOL_SHARE


def test_skill_pool_saturates_and_scales_when_oversubscribed() -> None:
    full = SKILL_MINER_CAP * SKILL_POOL_SHARE
    pre = np.ones(5)
    skill = np.full(5, SKILL_SATURATION)
    sig = np.ones(5)
    weights, pools = _allocate_slice(pre, skill, sig)

    assert pools == ["skill"] * 5
    assert weights.sum() == pytest.approx(SKILL_POOL_SHARE)
    assert weights[0] == pytest.approx(SKILL_POOL_SHARE / 5)
    assert weights[0] < full


def test_skill_claim_shrinks_with_significance() -> None:
    pre = np.array([0.2])
    skill = np.array([SKILL_SATURATION])
    half, _ = _allocate_slice(pre, skill, np.array([0.5]))
    full, _ = _allocate_slice(pre, skill, np.array([1.0]))
    assert half[0] == pytest.approx(full[0] * 0.5)
    assert full[0] == pytest.approx(SKILL_MINER_CAP * SKILL_POOL_SHARE)


def test_unspent_slice_goes_to_burn_uid() -> None:
    from src.validator.forecasting.scoring import BURN_UID

    metagraph = _StubMetagraph(uids=[1, BURN_UID])
    now = datetime.now(timezone.utc)
    rows = [_row(miner_uid=1, p_win=0.6, market_p_win=0.6, now=now) for _ in range(12)]
    out = score_agent_predictions(metagraph=metagraph, scored_predictions=rows, now=now)

    assert out[0] == pytest.approx(ALLOWANCE_MINER_CAP)
    assert out[1] == pytest.approx(1.0 - ALLOWANCE_MINER_CAP)
    assert out.sum() == pytest.approx(1.0)


def test_unmapped_uid_is_skipped_not_counted() -> None:
    metagraph = _StubMetagraph(uids=[0])

    now = datetime.now(timezone.utc)
    rows = [_row(miner_uid=None, p_win=1.0, now=now)]
    out = score_agent_predictions(metagraph=metagraph, scored_predictions=rows, now=now)
    np.testing.assert_allclose(out, np.zeros(1))
