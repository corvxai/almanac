"""Anti-gaming invariants for the v2 market scoring mechanism.

These lock in the properties established during the scoring-v2 audit:

1. The score is invariant to splitting one book across identities (Sybil).
2. PnL-neutral wash volume returns < 1.0x fees even after the miner weight
   boost — churn is never self-funding.
3. The fee floor is gated on real trailing edge, not merely positive PnL.
4. Trades flipped ineligible by the local hygiene checks still count their
   losses against PnL (no self-serve loss censoring).
5. General pool scoring is disabled but its fee accounting survives.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

pytest.importorskip("bittensor")

from src.validator.market.scoring import (  # noqa: E402
    FEE_FLOOR_MIN_ROI,
    FEE_FLOOR_MULTIPLIER,
    MINER_POOL_WEIGHT_BOOST_PERCENTAGE,
    build_epoch_history,
    compute_edge,
    score_miners,
    score_pool,
)

BOOST = 1 + MINER_POOL_WEIGHT_BOOST_PERCENTAGE
N_EPOCHS = 15


def make_history(volume: np.ndarray, pnl: np.ndarray) -> dict:
    volume = np.asarray(volume, dtype=float)
    pnl = np.asarray(pnl, dtype=float)
    n = volume.shape[1]
    return {
        "volume_prev": volume,
        "profit_prev": pnl,
        "fees_prev": 0.01 * volume,
        "trade_counts": (volume > 0).astype(float) * 5,
        "qualified_prev": np.zeros_like(volume),
        "unqualified_prev": np.zeros_like(volume),
        "correct_trade_counts": np.zeros_like(volume),
        "entity_ids": list(range(n)),
        "entity_map": {i: i for i in range(n)},
        "epoch_dates": [],
        "n_epochs": volume.shape[0],
        "n_entities": n,
        "miner_profiles": {},
        "account_map": {},
    }


def honest_pool(n: int = 30, vol: float = 20_000.0):
    volume = np.full((N_EPOCHS, n), vol)
    roi = np.linspace(0.005, 0.05, n)
    return volume, volume * roi[None, :]


def run_pool(volume: np.ndarray, pnl: np.ndarray) -> tuple[dict, float]:
    hist = make_history(volume, pnl)
    budget = float(hist["fees_prev"][N_EPOCHS - 1].sum())
    return score_pool(hist, budget, allow_dust=False), budget


def test_score_is_split_invariant():
    """Splitting one book into k identities must not change aggregate payout."""
    hv, hp = honest_pool()
    subject_vol, subject_roi = 40_000.0, 0.03

    results = {}
    for k in (1, 4):
        sv = np.full((N_EPOCHS, k), subject_vol / k)
        sp = sv * subject_roi
        r, _ = run_pool(np.hstack([hv, sv]), np.hstack([hp, sp]))
        results[k] = float(r["tokens"][hv.shape[1]:].sum())

    assert results[1] > 0
    np.testing.assert_allclose(results[4], results[1], rtol=1e-6)


def test_volume_memory_rewards_trailing_cadence():
    """Same volume/PnL today: the book with trailing cadence must out-earn a newcomer."""
    hv, hp = honest_pool(n=10)
    today_vol, roi = 20_000.0, 0.03

    steady_v = np.full((N_EPOCHS, 1), today_vol)
    steady_p = steady_v * roi

    # Clear eligibility (3 epochs) without building meaningful volume memory.
    burst_v = np.zeros((N_EPOCHS, 1))
    burst_p = np.zeros((N_EPOCHS, 1))
    for i in range(3):
        burst_v[i, 0] = 100.0
        burst_p[i, 0] = 100.0 * roi
    burst_v[-1, 0] = today_vol
    burst_p[-1, 0] = today_vol * roi

    r, _ = run_pool(
        np.hstack([hv, steady_v, burst_v]),
        np.hstack([hp, steady_p, burst_p]),
    )
    steady_idx = hv.shape[1]
    burst_idx = steady_idx + 1
    assert r["active"][steady_idx] and r["active"][burst_idx]
    assert r["tokens"][steady_idx] > r["tokens"][burst_idx]


def wash_fixture():
    """One $3k win 12 epochs ago, then $50k/epoch of PnL-neutral churn."""
    hv, hp = honest_pool(n=10)
    av = np.zeros((N_EPOCHS, 1))
    ap = np.zeros((N_EPOCHS, 1))
    av[3, 0] = 10_000.0
    ap[3, 0] = 3_000.0
    av[4:, 0] = 50_000.0
    return np.hstack([hv, av]), np.hstack([hp, ap]), hv.shape[1]


def test_pnl_neutral_wash_volume_below_break_even():
    """Churn with no real edge must return < 1.0x fees after the boost."""
    volume, pnl, j = wash_fixture()
    r, _ = run_pool(volume, pnl)
    fees = 0.01 * volume[N_EPOCHS - 1, j]
    assert BOOST * r["tokens"][j] / fees < 1.0


def test_floor_gate_denies_wash_keeps_honest_loser():
    volume, pnl, j = wash_fixture()

    # The wash trader's stale win diluted by their own churn falls below the gate.
    edge = compute_edge(make_history(volume, pnl))
    assert 0 < edge[j] < FEE_FLOOR_MIN_ROI

    # An honest trader who loses TODAY on solid history keeps the floor.
    loser = 5
    pnl = pnl.copy()
    pnl[N_EPOCHS - 1, loser] = -1_500.0
    edge = compute_edge(make_history(volume, pnl))
    assert edge[loser] >= FEE_FLOOR_MIN_ROI

    r, _ = run_pool(volume, pnl)
    fees = 0.01 * volume[N_EPOCHS - 1, loser]
    # Floors can be scaled down by FEE_FLOOR_MAX_POOL_SHARE, hence the slack.
    assert r["tokens"][loser] >= 0.45 * fees


def test_boosted_floor_cannot_exceed_fees():
    """If this fails, wash trading on the floor alone is +EV. Never relax it."""
    assert FEE_FLOOR_MULTIPLIER * BOOST <= 1.0


def _trade(uid: int, completed_at: str, **overrides) -> dict:
    base = {
        "account_id": 1,
        "profile_id": f"0xprofile{uid}",
        "miner_id": uid,
        "miner_hotkey": f"hk{uid}",
        "is_general_pool": False,
        "is_completed": True,
        "completed_at": completed_at,
        "volume": 1_000.0,
        "expected_volume": 1_000.0,
        "expected_fees": 10.0,
        "actual_fees": 10.0,
        "pnl": 0.0,
        "is_correct": False,
        "is_reward_eligible": True,
    }
    base.update(overrides)
    return base


def test_hygiene_flagged_losses_count_against_pnl():
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime(
        "%Y-%m-%dT12:00:00Z"
    )
    all_uids = [1, 2, 3, 4]
    all_hotkeys = ["", "hk1", "hk2", "hk3", "hk4"]

    trades = [
        # Miner 1: losing trade topped up outside Almanac -> loss must count.
        _trade(1, yesterday, volume=1_500.0, expected_volume=1_000.0, pnl=-200.0),
        # Miner 2: winning trade topped up outside Almanac -> win must NOT count.
        _trade(2, yesterday, volume=1_500.0, expected_volume=1_000.0, pnl=300.0,
               is_correct=True),
        # Miner 3: arrived ineligible from the API -> untouched either way.
        _trade(3, yesterday, is_reward_eligible=False, pnl=-500.0),
        # Miner 4: clean eligible trade -> counted fully.
        _trade(4, yesterday, pnl=250.0, is_correct=True),
    ]

    hist = build_epoch_history(trades, all_uids, all_hotkeys, is_miner_pool=True)
    idx = hist["entity_map"]
    profit = hist["profit_prev"].sum(axis=0)
    volume = hist["volume_prev"].sum(axis=0)

    assert profit[idx[1]] == pytest.approx(-200.0)
    assert volume[idx[1]] == 0.0  # flagged volume stays excluded
    assert profit[idx[2]] == 0.0  # flagged win not credited
    assert 3 not in idx or profit[idx[3]] == 0.0
    assert profit[idx[4]] == pytest.approx(250.0)
    assert volume[idx[4]] == pytest.approx(1_000.0)


def test_general_pool_scoring_disabled_but_fees_reported():
    dates = [
        (datetime.now(timezone.utc) - timedelta(days=d)).strftime("%Y-%m-%dT12:00:00Z")
        for d in (1, 2, 3)
    ]
    trades = [
        _trade(0, day, is_general_pool=True, miner_id=None, miner_hotkey=None,
               pnl=50.0, is_correct=True)
        for day in dates
        for _ in range(2)
    ]

    _, gp_history, _, gp_scores, _, gp_budget = score_miners(
        all_uids=[], all_hotkeys=[], trading_history=trades
    )

    assert gp_history["n_entities"] == 1
    assert gp_budget > 0  # fees still collected and reported
    assert float(np.sum(gp_scores["tokens"])) == 0.0  # but nothing is paid


def test_project_to_budget_never_exceeds_total():
    """Caps + floors binding together must not overshoot (audit repro)."""
    from src.validator.market.scoring import _project_to_budget

    shares = np.array([1.0] * 16 + [1e-9] * 40)
    floors = np.array([0.0] * 16 + [1.0] * 40)
    alloc = _project_to_budget(shares, total=100.0, floors=floors, cap_fraction=0.06)
    assert float(alloc.sum()) <= 100.0 + 1e-9
    assert float(alloc.sum()) == pytest.approx(100.0, abs=1e-6)
    np.testing.assert_allclose(alloc[16:], 1.0, atol=1e-9)
    assert np.all(alloc >= 0.0)


def test_project_to_budget_respects_floors_and_caps():
    from src.validator.market.scoring import _project_to_budget, CAP_RELAX_FACTOR

    rng = np.random.default_rng(0)
    for n in (5, 16, 40):
        shares = rng.random(n)
        floors = rng.random(n) * 2.0
        total = 100.0
        alloc = _project_to_budget(shares, total=total, floors=floors, cap_fraction=0.06)
        n_scoring = max(int(np.sum(shares > 0)), 1)
        cap = max(0.06, CAP_RELAX_FACTOR / n_scoring) * total
        floors_fit = np.clip(floors, 0.0, cap)
        if floors_fit.sum() > total:
            floors_fit = floors_fit * (total / floors_fit.sum())
        assert float(alloc.sum()) <= total + 1e-6
        assert np.all(alloc <= cap + 1e-6)
        assert np.all(alloc + 1e-6 >= floors_fit)


def test_boost_clamps_instead_of_cliff():
    """Crossing the full-boost threshold must taper, not drop the whole boost."""
    from src.validator.market.scoring import (
        BURN_UID,
        MINER_POOL_WEIGHT_BOOST_PERCENTAGE,
        calculate_weights,
    )

    budget = 1000.0
    # Just under the full-boost cliff: distributed * 1.75 == budget.
    under = budget / (1 + MINER_POOL_WEIGHT_BOOST_PERCENTAGE) - 1.0
    over = budget / (1 + MINER_POOL_WEIGHT_BOOST_PERCENTAGE) + 1.0

    def run(tokens: float):
        miner = {"entity_ids": [1], "tokens": np.array([tokens])}
        gp = {"tokens": np.array([0.0])}
        uids = [1, BURN_UID]
        return calculate_weights(miner, gp, budget, tokens, 0.0, [], uids)

    w_under = run(under)
    w_over = run(over)
    # Both should still receive a boost factor > 1 relative to unboosted share.
    assert w_under[0] > under / budget
    assert w_over[0] > over / budget
    # Continuous: overshoot case must not collapse to the unboosted weight.
    assert w_over[0] > (over / budget) * 1.01


def test_as_of_keeps_full_trailing_window():
    """Historical as_of must score a full ROLLING window, not a truncated prefix."""
    from src.validator.market.scoring import ROLLING_HISTORY_IN_DAYS, build_epoch_history

    as_of = datetime(2026, 6, 15, tzinfo=timezone.utc)
    hist = build_epoch_history([], [1], ["", "hk1"], True, as_of=as_of)
    assert hist["n_epochs"] == ROLLING_HISTORY_IN_DAYS
    assert hist["epoch_dates"][0] == "2026-05-16"
    assert hist["epoch_dates"][-1] == "2026-06-14"


def test_burn_uid_collision_rejected():
    from src.validator.market.scoring import BURN_UID, calculate_weights

    miner = {"entity_ids": [BURN_UID], "tokens": np.array([10.0])}
    gp = {"tokens": np.array([0.0])}
    with pytest.raises(AssertionError, match="BURN_UID"):
        calculate_weights(miner, gp, 100.0, 10.0, 0.0, [], [BURN_UID])
