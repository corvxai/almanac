from __future__ import annotations

import sys
import importlib
from types import SimpleNamespace

import numpy as np
import pytest

@pytest.mark.parametrize(
    ("budget_share", "expected_budget"),
    [
        (0.25, 50.0),
        (1.5, 200.0),
        (-3.0, 0.0),
    ],
)
def test_score_market_applies_budget_share_to_epoch_budget(
    monkeypatch: pytest.MonkeyPatch,
    budget_share: float,
    expected_budget: float,
) -> None:
    seen: dict[str, float] = {}
    monkeypatch.setitem(
        sys.modules,
        "src.validator.market.metadata_manager",
        SimpleNamespace(MetadataManager=object),
    )
    market_loop = importlib.import_module("src.validator.market.loop")

    monkeypatch.setattr(market_loop, "fetch_trading_history", lambda **kw: [])
    monkeypatch.setattr(market_loop, "fetch_tao_price", lambda: 100.0)
    monkeypatch.setattr(market_loop, "compute_epoch_budget", lambda metagraph, tao_price_usd: 200.0)

    def _score_miners(all_uids, all_hotkeys, trading_history, current_epoch_budget):
        seen["score_miners_budget"] = float(current_epoch_budget)
        return (
            {},
            {},
            {"tokens": np.array([], dtype=float), "entity_ids": []},
            {"tokens": np.array([], dtype=float), "entity_ids": []},
            0.0,
            0.0,
        )

    def _calculate_weights(
        miners_scores,
        general_pool_scores,
        current_epoch_budget,
        miner_budget,
        general_pool_budget,
        miners_to_penalize,
        all_uids,
    ):
        seen["calculate_weights_budget"] = float(current_epoch_budget)
        return [1.0, 0.0]

    fake_scoring_module = SimpleNamespace(
        score_miners=_score_miners,
        calculate_weights=_calculate_weights,
        print_pool_stats=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "src.validator.market.scoring", fake_scoring_module)

    metagraph = SimpleNamespace(
        uids=SimpleNamespace(tolist=lambda: [0, 1]),
        hotkeys=["hotkey_0", "hotkey_1"],
    )

    out = market_loop.score_market(
        network="finney",
        wallet=None,
        dendrite=SimpleNamespace(),
        metagraph=metagraph,
        metadata_manager=None,
        print_stats=False,
        budget_share=budget_share,
    )

    np.testing.assert_allclose(out, np.array([1.0, 0.0]))
    assert seen["score_miners_budget"] == pytest.approx(expected_budget)
    assert seen["calculate_weights_budget"] == pytest.approx(expected_budget)
