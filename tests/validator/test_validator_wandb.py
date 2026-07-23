from __future__ import annotations

import datetime
import sys
from types import SimpleNamespace

from src.core import constants
from src.core.config import AppConfig
from src.validator import validator as validator_module
from src.validator.validator import Validator


class _FakeRun:
    def __init__(self) -> None:
        self.finish_calls = 0

    def finish(self) -> None:
        self.finish_calls += 1


def _bt_objects():
    return validator_module._BtObjects(  # noqa: SLF001 - tests use internal helper
        wallet=SimpleNamespace(hotkey=SimpleNamespace(ss58_address="hotkey_test")),
        subtensor=SimpleNamespace(),
        dendrite=SimpleNamespace(),
        metagraph=SimpleNamespace(hotkeys=["other_hotkey", "hotkey_test"]),
        network="finney",
    )


def test_wandb_disables_itself_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    cfg = AppConfig()

    validator = Validator(config=cfg, store=None, bt_objects=_bt_objects(), metadata_manager=None)

    assert cfg.loop.wandb_enabled is False
    assert validator._wandb_run is None  # noqa: SLF001


def test_wandb_off_skips_initialization_even_with_api_key(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    init_calls: list[dict] = []
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=lambda **kwargs: init_calls.append(kwargs)),
    )
    cfg = AppConfig()
    cfg.loop.wandb_enabled = False

    Validator(config=cfg, store=None, bt_objects=_bt_objects(), metadata_manager=None)

    assert init_calls == []


def test_wandb_run_initializes_rotates_and_finishes(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    init_calls: list[dict] = []
    runs: list[_FakeRun] = []

    def _init(**kwargs):
        init_calls.append(kwargs)
        run = _FakeRun()
        runs.append(run)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=_init))
    now = [datetime.datetime(2026, 7, 23, 14, 30, tzinfo=datetime.timezone.utc)]
    cfg = AppConfig()
    validator = Validator(
        config=cfg,
        store=None,
        bt_objects=_bt_objects(),
        metadata_manager=None,
        clock=lambda: now[0],
    )

    assert init_calls[0]["name"] == "validator-1-2026-07-23_14-30-00"
    assert init_calls[0]["project"] == constants.WANDB_PROJECT
    assert init_calls[0]["entity"] == constants.WANDB_ENTITY
    assert init_calls[0]["config"] == {
        "uid": 1,
        "hotkey": "hotkey_test",
        "run_name": "2026-07-23_14-30-00",
        "type": "validator",
        "netuid": cfg.bittensor.netuid,
        "network": "finney",
    }

    now[0] += datetime.timedelta(days=1)
    validator._rotate_wandb_run_if_needed(now[0])  # noqa: SLF001

    assert runs[0].finish_calls == 1
    assert len(runs) == 2

    validator.stop()
    validator.stop()
    assert runs[1].finish_calls == 1
