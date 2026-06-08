from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.core.config import AppConfig
from src.validator import validator as validator_module
from src.validator.validator import Validator


def _bt_objects():
    return validator_module._BtObjects(  # noqa: SLF001 - tests use internal helper
        wallet=SimpleNamespace(hotkey=SimpleNamespace(ss58_address="hotkey_test")),
        subtensor=SimpleNamespace(),
        dendrite=SimpleNamespace(),
        metagraph=SimpleNamespace(),
        network="test",
    )


def test_score_agent_predictions_fetches_from_orchestrator_no_trace_store(monkeypatch) -> None:
    cfg = AppConfig()
    cfg.validator.orchestrator_api_url = "https://orchestrator.example.com"
    cfg.loop.rolling_window_days = 7
    v = Validator(config=cfg, store=None, bt_objects=_bt_objects(), metadata_manager=None)
    fake_loaded = SimpleNamespace(hotkey_ss58="5ValidatorHotkey", keypair=SimpleNamespace())
    monkeypatch.setattr("src.validator.validator.load_hotkey", lambda _cfg: fake_loaded)

    fetch_called = {"ok": False}
    score_called = {"ok": False}

    def _fake_fetch(**kwargs):
        fetch_called["ok"] = True
        assert kwargs["base_url"] == "https://orchestrator.example.com"
        assert kwargs["loaded_hotkey"] is fake_loaded
        assert kwargs["days"] == 7
        assert kwargs["include_trace_summary"] is False
        return [SimpleNamespace()]

    def _fake_score(**kwargs):
        score_called["ok"] = True
        assert kwargs["scored_predictions"]
        return np.array([0.1, 0.2], dtype=float)

    monkeypatch.setattr("src.validator.validator.fetch_all_scored_predictions", _fake_fetch)
    monkeypatch.setattr(
        "src.validator.validator.arcratio_scoring.score_agent_predictions",
        _fake_score,
    )

    out = v._score_agent_predictions()  # noqa: SLF001
    assert fetch_called["ok"] is True
    assert score_called["ok"] is True
    np.testing.assert_allclose(out, np.array([0.1, 0.2]))
