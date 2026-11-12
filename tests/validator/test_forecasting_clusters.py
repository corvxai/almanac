"""Duplicate-agent clusters: detection and the one-claim-per-cluster payout."""

from __future__ import annotations

import random
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from src.validator.forecasting import scoring
from src.validator.forecasting.scoring import (
    ALLOWANCE_MINER_CAP,
    CLUSTER_AGREE_DEN,
    CLUSTER_AGREE_NUM,
    CLUSTER_ENFORCE_FROM,
    CLUSTER_MIN_INFORMATIVE,
    SKILL_MINER_CAP,
    SKILL_POOL_SHARE,
    _allocate_slice,
    _detect_clusters,
    _Record,
    score_agent_predictions,
)

NOW = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc)


def _rec(market: int | str, p_win: float, market_p_win: float | None = 0.50) -> _Record:
    return _Record(
        p_win=p_win,
        p_pred=p_win,
        hit=1,
        scored_at=NOW - timedelta(hours=1),
        market_p_win=market_p_win,
        market_p_pred=market_p_win,
        market_id=market if isinstance(market, str) else f"m{market}",
    )


def _clusters(records_by_uid: dict[int, list[_Record]]) -> dict[int, tuple[int, int]]:
    uids = list(records_by_uid)
    ids, sizes = _detect_clusters(uids, [records_by_uid[u] for u in uids])
    return {uid: (int(ids[i]), int(sizes[i])) for i, uid in enumerate(uids)}


def _row(uid: int, market: int, p_win: float, market_p_win: float, *, now: datetime = NOW):
    return SimpleNamespace(
        minerUid=uid,
        minerHotkey=f"hk-{uid}",
        marketId=f"m{market}",
        predictionIsInvalid=False,
        resolutionStatus="resolved",
        scoredAt=now - timedelta(hours=1, seconds=market),
        outcomeProbabilities={"yes": p_win, "no": 1.0 - p_win},
        outcomePricesAtPrediction={"yes": market_p_win, "no": 1.0 - market_p_win},
        predictedOutcomeId="yes",
        resolvedOutcomeId="yes",
    )


def _metagraph(uids: list[int]):
    return SimpleNamespace(uids=uids, hotkeys=[f"hk-{u}" for u in uids])


# 100 rows keeps rho near 0.55, so four full claims fit inside the skill pool
# and the tests below see the cluster rule without the oversubscription scale.
def _skilled_rows(uid: int, n: int = 100, *, p_win: float = 0.80, now: datetime = NOW):
    """Off-market and better than the market (0.60) on every event."""
    return [_row(uid, m, p_win, 0.60, now=now) for m in range(n)]


def _independent_rows(uid: int, n: int = 100, *, seed: int, now: datetime = NOW):
    """Skilled, off-market, and unlike any other miner."""
    rng = random.Random(seed)
    return [_row(uid, m, rng.choice([0.70, 0.75, 0.85, 0.90, 0.95]), 0.60, now=now) for m in range(n)]


# --------------------------------------------------------------------------- #
# Detection                                                                    #
# --------------------------------------------------------------------------- #

def test_constants() -> None:
    assert scoring.CLUSTER_TICK == 0.01
    assert CLUSTER_MIN_INFORMATIVE == 30
    assert (CLUSTER_AGREE_NUM, CLUSTER_AGREE_DEN) == (3, 5)
    assert CLUSTER_ENFORCE_FROM.tzinfo is not None


def test_echo_pair_is_not_linked() -> None:
    # Both copy the market on 200 shared markets: no off-market record, so no
    # informative event, however alike they are.
    echo = [_rec(m, 0.50 + (m % 40) / 100, 0.50 + (m % 40) / 100) for m in range(200)]
    got = _clusters({1: list(echo), 2: list(echo)})
    assert got == {1: (-1, 1), 2: (-1, 1)}


def test_within_one_tick_of_the_market_is_still_an_echo() -> None:
    # abs(p - m) == 0.01 is not off-market (0.51 - 0.50 in floats is just
    # above 0.01; the 1e-9 slack absorbs it).
    a = [_rec(m, 0.51, 0.50) for m in range(100)]
    b = [_rec(m, 0.49, 0.50) for m in range(100)]
    assert _clusters({1: a, 2: b}) == {1: (-1, 1), 2: (-1, 1)}


def test_independent_pair_is_not_linked() -> None:
    rng = random.Random(7)
    a = [_rec(m, rng.choice([0.70, 0.75, 0.80]), 0.50) for m in range(200)]
    b = [_rec(m, rng.choice([0.20, 0.25, 0.30]), 0.50) for m in range(200)]
    assert _clusters({1: a, 2: b}) == {1: (-1, 1), 2: (-1, 1)}


@pytest.mark.parametrize(("informative", "linked"), [(29, False), (30, True)])
def test_informative_floor(informative: int, linked: bool) -> None:
    # Identical off-market predictions on `informative` markets, plus shared
    # market echoes that must not count toward the floor.
    recs = [_rec(m, 0.80, 0.50) for m in range(informative)]
    recs += [_rec(1000 + m, 0.50, 0.50) for m in range(60)]
    got = _clusters({5: list(recs), 3: list(recs)})
    assert got == ({5: (3, 2), 3: (3, 2)} if linked else {5: (-1, 1), 3: (-1, 1)})


@pytest.mark.parametrize(("agree", "linked"), [(29, False), (30, True), (31, True)])
def test_agreement_boundary(agree: int, linked: bool) -> None:
    # 50 informative events; agree * 5 >= 3 * 50 needs agree >= 30.
    a = [_rec(m, 0.80, 0.50) for m in range(50)]
    b = [_rec(m, 0.80 if m < agree else 0.20, 0.50) for m in range(50)]
    got = _clusters({1: a, 2: b})
    assert got == ({1: (1, 2), 2: (1, 2)} if linked else {1: (-1, 1), 2: (-1, 1)})


@pytest.mark.parametrize(("cents_apart", "linked"), [(1, True), (2, False)])
def test_agreement_is_within_one_cent(cents_apart: int, linked: bool) -> None:
    a = [_rec(m, 0.80, 0.50) for m in range(40)]
    b = [_rec(m, round(0.80 + cents_apart / 100, 2), 0.50) for m in range(40)]
    assert (_clusters({1: a, 2: b})[1] == (1, 2)) is linked


def test_one_sided_off_market_event_is_informative() -> None:
    # A is off-market, B echoes the market: informative, and they disagree.
    a = [_rec(m, 0.80, 0.50) for m in range(40)]
    b = [_rec(m, 0.50, 0.50) for m in range(40)]
    assert _clusters({1: a, 2: b}) == {1: (-1, 1), 2: (-1, 1)}


def test_chain_forms_one_cluster_with_the_smallest_uid() -> None:
    # 30 and 9 share 40 markets, 9 and 12 share another 40, 30 and 12 share
    # none: one component through 9.
    a = [_rec(m, 0.80, 0.50) for m in range(0, 40)]
    b = [_rec(m, 0.80, 0.50) for m in range(0, 80)]
    c = [_rec(m, 0.80, 0.50) for m in range(40, 80)]
    lone = [_rec(m, 0.10, 0.50) for m in range(0, 80)]
    got = _clusters({30: a, 9: b, 12: c, 4: lone})
    assert got == {30: (9, 3), 9: (9, 3), 12: (9, 3), 4: (-1, 1)}


def test_records_without_a_market_id_or_price_are_skipped() -> None:
    no_id = [
        _Record(0.80, 0.80, 1, NOW, 0.50, 0.50) for _ in range(60)
    ]
    no_price = [_rec(m, 0.80, None) for m in range(60)]
    assert _clusters({1: no_id, 2: list(no_id)}) == {1: (-1, 1), 2: (-1, 1)}
    assert _clusters({1: no_price, 2: list(no_price)}) == {1: (-1, 1), 2: (-1, 1)}


def test_detection_is_fast_at_subnet_scale() -> None:
    rng = np.random.default_rng(0)
    records = []
    for _ in range(256):
        p = rng.integers(5, 96, size=1000) / 100
        records.append([_rec(m, float(p[m]), 0.50) for m in range(1000)])
    start = time.perf_counter()
    ids, _ = _detect_clusters(list(range(256)), records)
    assert time.perf_counter() - start < 2.0
    assert (ids == -1).all()


# --------------------------------------------------------------------------- #
# Payout                                                                       #
# --------------------------------------------------------------------------- #

def test_allocate_slice_splits_the_largest_claim() -> None:
    pre = np.array([0.5, 0.5, 0.5, 0.5])
    skill = np.array([0.2, 0.2, 0.2, 0.2])
    sig = np.array([1.0, 0.8, 0.6, 1.0])
    eff_n = np.array([100.0, 100.0, 100.0, 100.0])
    full = SKILL_MINER_CAP * SKILL_POOL_SHARE

    ids = np.array([7, 7, 7, -1])
    weights, pools, _, _ = _allocate_slice(pre, skill, sig, eff_n, cluster_ids=ids)
    assert pools == ["skill"] * 4
    np.testing.assert_allclose(weights[:3], [full / 3] * 3)
    assert weights[:3].sum() == pytest.approx(full)
    assert weights[3] == pytest.approx(full)


def test_member_without_a_skill_claim_gets_nothing_extra() -> None:
    pre = np.array([0.5, 0.5, 0.5])
    skill = np.array([0.2, 0.2, 0.2])
    sig = np.array([1.0, 0.5, 1.0])
    eff_n = np.array([100.0, 100.0, 10.0])  # third is under SKILL_MIN_EFFECTIVE_N
    full = SKILL_MINER_CAP * SKILL_POOL_SHARE
    ids = np.array([2, 2, 2])

    weights, pools, _, _ = _allocate_slice(pre, skill, sig, eff_n, cluster_ids=ids)

    # Two claimants split the larger claim; the third keeps its allowance.
    assert pools == ["skill", "skill", "allow"]
    np.testing.assert_allclose(weights[:2], [full / 2, full / 2])
    assert weights[2] == pytest.approx(ALLOWANCE_MINER_CAP)


def test_single_claimant_in_a_cluster_keeps_its_claim() -> None:
    pre = np.array([0.5, 0.5])
    skill = np.array([0.2, 0.2])
    sig = np.array([1.0, 1.0])
    eff_n = np.array([100.0, 10.0])
    ids = np.array([0, 0])

    weights, pools, _, _ = _allocate_slice(pre, skill, sig, eff_n, cluster_ids=ids)
    baseline, _, _, _ = _allocate_slice(pre, skill, sig, eff_n)

    assert pools == ["skill", "allow"]
    np.testing.assert_allclose(weights, baseline)


def test_split_claim_is_compared_against_the_allowance() -> None:
    n = 20
    pre = np.full(n, 0.5)
    skill = np.full(n, 0.2)
    sig = np.full(n, 1.0)
    eff_n = np.full(n, 100.0)
    ids = np.zeros(n, dtype=int)

    weights, pools, allowance_spent, skill_spent = _allocate_slice(
        pre, skill, sig, eff_n, cluster_ids=ids
    )

    # 0.30 / 20 = 0.015 each, against a 0.25 / 20 = 0.0125 allowance: the
    # 0.0025 above the allowance is all the skill pool pays.
    np.testing.assert_allclose(weights, np.full(n, 0.015))
    assert pools == ["skill"] * n
    assert skill_spent == pytest.approx(n * 0.0025)
    assert allowance_spent == pytest.approx(0.25)

    # A weak claim falls under the allowance once split: significance 0.1
    # gives 0.03 / 20 = 0.0015, so every member is paid as an allowance miner.
    weights, pools, _, skill_spent = _allocate_slice(
        pre, skill, np.full(n, 0.1), eff_n, cluster_ids=ids
    )
    np.testing.assert_allclose(weights, np.full(n, 0.0125))
    assert pools == ["allow"] * n
    assert skill_spent == 0.0


def test_penalty_is_off_by_default() -> None:
    assert scoring.CLUSTER_PENALTY_PCT == 0.0


def test_penalty_cuts_every_cluster_member_and_nobody_else() -> None:
    pre = np.array([0.5, 0.5, 0.5, 0.5])
    skill = np.array([0.2, 0.2, 0.2, 0.2])
    sig = np.array([1.0, 0.5, 1.0, 1.0])
    eff_n = np.array([100.0, 100.0, 10.0, 100.0])  # third has no skill claim
    ids = np.array([2, 2, 2, -1])

    base, base_pools, base_allow, base_skill = _allocate_slice(
        pre, skill, sig, eff_n, cluster_ids=ids
    )
    cut, pools, allow_spent, skill_spent = _allocate_slice(
        pre, skill, sig, eff_n, cluster_ids=ids, cluster_penalty_pct=25.0
    )

    # Claimants and the allowance-only member alike lose a quarter.
    np.testing.assert_allclose(cut[:3], base[:3] * 0.75)
    assert cut[3] == base[3]
    assert pools == base_pools == ["skill", "skill", "allow", "skill"]
    # The cut is unspent, not handed to anyone.
    assert base.sum() - cut.sum() == pytest.approx(0.25 * base[:3].sum())
    assert allow_spent + skill_spent == pytest.approx(cut.sum())
    assert allow_spent < base_allow and skill_spent < base_skill


def test_penalty_needs_enforcement() -> None:
    pre = np.array([0.5, 0.5])
    skill = np.array([0.2, 0.2])
    sig = np.array([1.0, 1.0])
    eff_n = np.array([100.0, 100.0])

    base, _, _, _ = _allocate_slice(pre, skill, sig, eff_n)
    cut, _, _, _ = _allocate_slice(pre, skill, sig, eff_n, cluster_penalty_pct=25.0)
    np.testing.assert_array_equal(cut, base)


def test_penalty_holds_when_the_skill_pool_is_oversubscribed() -> None:
    # Four full claims (1.2) against a 0.75 pool: the penalty is applied after
    # the scale, so the cluster is not scaled back up at the others' expense.
    pre = np.full(4, 0.5)
    skill = np.full(4, 0.2)
    sig = np.full(4, 1.0)
    eff_n = np.full(4, 100.0)
    ids = np.array([0, -1, -1, 0])
    sig_split = sig.copy()

    base, _, _, _ = _allocate_slice(pre, skill, sig_split, eff_n, cluster_ids=ids)
    cut, _, _, _ = _allocate_slice(
        pre, skill, sig_split, eff_n, cluster_ids=ids, cluster_penalty_pct=40.0
    )
    np.testing.assert_allclose(cut[[0, 3]], base[[0, 3]] * 0.6)
    np.testing.assert_array_equal(cut[[1, 2]], base[[1, 2]])


def test_full_penalty_zeroes_the_cluster() -> None:
    pre = np.array([0.5, 0.5, 0.5])
    skill = np.array([0.2, 0.2, 0.2])
    sig = np.ones(3)
    eff_n = np.full(3, 100.0)
    cut, _, _, _ = _allocate_slice(
        pre, skill, sig, eff_n, cluster_ids=np.array([1, 1, -1]), cluster_penalty_pct=100.0
    )
    assert cut[:2].tolist() == [0.0, 0.0]
    assert cut[2] > 0.0


def test_penalty_override_end_to_end() -> None:
    uids = [3, 8, 21, 40]
    rows = [r for uid in (3, 8, 21) for r in _skilled_rows(uid)]
    rows += _independent_rows(40, seed=40)

    def run(enforce: bool, pct: float | None = None):
        return score_agent_predictions(
            metagraph=_metagraph(uids),
            scored_predictions=rows,
            now=NOW,
            cluster_enforce=enforce,
            cluster_penalty_pct=pct,
            return_pre_pareto=True,
        )

    on = run(True)
    cut = run(True, 25.0)
    np.testing.assert_array_equal(on[0], run(True, 0.0)[0])  # default is 0
    np.testing.assert_allclose(cut[0][:3], on[0][:3] * 0.75)
    assert cut[0][3] == on[0][3]
    assert cut[4]["penalty_pct"] == 25.0
    assert on[4]["penalty_pct"] == 0.0
    # Not enforced: no penalty, and the report says so.
    off = run(False, 25.0)
    np.testing.assert_array_equal(off[0], run(False)[0])
    assert off[4]["penalty_pct"] == 0.0


@pytest.mark.parametrize("pct", [-1.0, 100.5, float("nan")])
def test_penalty_out_of_range_is_rejected(pct: float) -> None:
    with pytest.raises(ValueError):
        score_agent_predictions(
            metagraph=_metagraph([3]), scored_predictions=[], now=NOW, cluster_penalty_pct=pct
        )


def test_clone_cluster_collapses_to_one_claim() -> None:
    clones = [3, 8, 21]
    rows = [r for uid in clones for r in _skilled_rows(uid)]
    rows += _independent_rows(40, seed=40)
    uids = [3, 8, 21, 40]

    solo = score_agent_predictions(
        metagraph=_metagraph(uids),
        scored_predictions=_skilled_rows(3) + _independent_rows(40, seed=40),
        now=NOW,
        cluster_enforce=True,
    )
    off = score_agent_predictions(
        metagraph=_metagraph(uids), scored_predictions=rows, now=NOW, cluster_enforce=False
    )
    on, _, pools, _, clusters = score_agent_predictions(
        metagraph=_metagraph(uids),
        scored_predictions=rows,
        now=NOW,
        cluster_enforce=True,
        return_pre_pareto=True,
    )

    assert clusters["cluster_id"].tolist() == [3, 3, 3, -1]
    assert clusters["cluster_size"].tolist() == [3, 3, 3, 1]
    assert clusters["enforced"] is True
    assert pools == ["skill"] * 4
    # Not enforced: three full claims. Enforced: the three share one.
    assert off[:3].sum() == pytest.approx(3 * solo[0])
    assert on[:3].sum() == pytest.approx(solo[0])
    np.testing.assert_allclose(on[:3], np.full(3, solo[0] / 3))
    # The independent miner is untouched (the pool is not oversubscribed).
    assert on[3] == pytest.approx(off[3]) == pytest.approx(solo[3])


def test_before_and_after_the_enforce_date() -> None:
    uids = [3, 8, 21, 40]

    def run(now: datetime, enforce: bool | None = None):
        rows = [r for uid in (3, 8, 21) for r in _skilled_rows(uid, now=now)]
        rows += _independent_rows(40, seed=40, now=now)
        return score_agent_predictions(
            metagraph=_metagraph(uids),
            scored_predictions=rows,
            now=now,
            cluster_enforce=enforce,
            return_pre_pareto=True,
        )

    before_now = CLUSTER_ENFORCE_FROM - timedelta(seconds=1)
    before, _, _, _, before_clusters = run(before_now)
    after, _, _, _, after_clusters = run(CLUSTER_ENFORCE_FROM)

    # Clusters are reported either side of the date; only the payout changes.
    assert before_clusters["cluster_id"].tolist() == [3, 3, 3, -1]
    assert after_clusters["cluster_id"].tolist() == [3, 3, 3, -1]
    assert before_clusters["enforced"] is False
    assert after_clusters["enforced"] is True
    np.testing.assert_allclose(before, run(before_now, False)[0])
    np.testing.assert_allclose(after, run(CLUSTER_ENFORCE_FROM, True)[0])
    assert after[:3].sum() == pytest.approx(before[:3].sum() / 3)


def test_result_does_not_depend_on_row_order() -> None:
    uids = [3, 8, 21, 40, 55]
    rows = [r for uid in (3, 8, 21) for r in _skilled_rows(uid)]
    rows += _independent_rows(40, seed=40) + _independent_rows(55, seed=55)

    def run(seq):
        return score_agent_predictions(
            metagraph=_metagraph(uids),
            scored_predictions=seq,
            now=NOW,
            cluster_enforce=True,
            return_pre_pareto=True,
        )

    base, _, base_pools, _, base_clusters = run(rows)
    for seed in range(3):
        shuffled = list(rows)
        random.Random(seed).shuffle(shuffled)
        got, _, pools, _, clusters = run(shuffled)
        np.testing.assert_allclose(got, base, rtol=0, atol=1e-12)
        assert pools == base_pools
        assert clusters["cluster_id"].tolist() == base_clusters["cluster_id"].tolist()
        assert clusters["cluster_size"].tolist() == base_clusters["cluster_size"].tolist()

    # Metagraph order does not move a miner's cluster or payout either.
    flipped = list(reversed(uids))
    got, _, _, _, clusters = score_agent_predictions(
        metagraph=_metagraph(flipped),
        scored_predictions=rows,
        now=NOW,
        cluster_enforce=True,
        return_pre_pareto=True,
    )
    np.testing.assert_allclose(got[::-1], base, rtol=0, atol=1e-12)
    assert clusters["cluster_id"].tolist()[::-1] == base_clusters["cluster_id"].tolist()
