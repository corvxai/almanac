"""Scoring step coverage for the production ``Validator`` class.

These tests do not touch the chain. They inject a fake ``_BtObjects`` plus a
stub ``MetadataManager`` and call ``Validator.run_scoring_step`` directly, then
assert on what ``subtensor.set_weights`` received. Three cases:

- Almanac Market only.
- Almanac Forecasting only.
- Both enabled and blended.

Almanac Market scoring is monkey-patched (the cvxpy-backed path is exercised
elsewhere by sn41's own simulator scripts; we only need to confirm the
``Validator`` wires the call + flows the result into the blender).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from src.core.config import AppConfig
from src.core.schemas import (
    EventCategory,
    EvidenceDigest,
    EventSnapshot,
    ExecutionContext,
    PredictionOutput,
    ResolutionRecord,
    SandboxEnvironment,
    TraceIntegrity,
)
from src.storage.json_store import JsonTraceStore
from src.validator import validator as validator_module
from src.validator.validator import Validator


class _StubUids:
    def __init__(self, vals: list[int]) -> None:
        self._vals = vals

    def tolist(self) -> list[int]:
        return list(self._vals)

    def __iter__(self):
        return iter(self._vals)

    def __len__(self) -> int:
        return len(self._vals)


class _StubMetagraph:
    def __init__(self, uids: list[int]) -> None:
        self.uids = _StubUids(uids)
        self.hotkeys = [f"hotkey_{u}" for u in uids]
        self.synced = 0

    def sync(self) -> None:
        self.synced += 1


class _RecordingSubtensor:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def set_weights(self, *, netuid, wallet, uids, weights, wait_for_inclusion):
        self.calls.append(
            {
                "netuid": netuid,
                "uids": list(uids),
                "weights": np.asarray(weights, dtype=float),
                "wait_for_inclusion": wait_for_inclusion,
            }
        )
        return True


def _bt_objects(uids: list[int]):
    metagraph = _StubMetagraph(uids)
    subtensor = _RecordingSubtensor()
    return validator_module._BtObjects(  # noqa: SLF001 — test reaches into module
        wallet=SimpleNamespace(hotkey=SimpleNamespace(ss58_address="hotkey_test")),
        subtensor=subtensor,
        dendrite=SimpleNamespace(keypair=SimpleNamespace(ss58_address="hotkey_test")),
        metagraph=metagraph,
        network="finney",
    )


def _config(*, market: bool, forecasting: bool, market_share=1.0, forecasting_share=1.0) -> AppConfig:
    cfg = AppConfig()
    cfg.loop.loop_enabled = True
    cfg.loop.market_enabled = market
    cfg.loop.forecasting_enabled = forecasting
    cfg.loop.market_weight_share = market_share
    cfg.loop.forecasting_weight_share = forecasting_share
    cfg.loop.wandb_enabled = False
    return cfg


def _make_resolved_trace(*, miner_uid: int, prediction: float, outcome: bool) -> EvidenceDigest:
    now = datetime.now(timezone.utc)
    digest = EvidenceDigest(
        execution_context=ExecutionContext(
            execution_id=uuid4(),
            agent_id=uuid4(),
            agent_version="0.0.1",
            event_id=uuid4(),
            validator_id=uuid4(),
            timestamp_start=now - timedelta(hours=1),
            timestamp_end=now - timedelta(hours=1) + timedelta(seconds=1),
            execution_duration_ms=1_000,
            sandbox_environment=SandboxEnvironment.IN_PROCESS,
        ),
        event_snapshot=EventSnapshot(
            event_title="t",
            event_category=EventCategory.OTHER,
            resolution_criteria="-",
            resolution_deadline=now - timedelta(hours=1),
            event_created_at=now - timedelta(days=7),
        ),
        provider_calls=[],
        reasoning_chain=[],
        prediction_output=PredictionOutput(
            final_probability=prediction,
            reasoning_summary="t",
            metadata={"miner_uid": miner_uid},
        ),
        resolution_record=ResolutionRecord(
            resolved=True,
            resolution_outcome=outcome,
            resolution_timestamp=now - timedelta(hours=1),
        ),
        trace_integrity=TraceIntegrity(
            trace_hash="", total_provider_cost=0.0, total_evidence_items=0
        ),
    )
    return digest.seal()


@pytest.fixture
def store(tmp_path: Path) -> JsonTraceStore:
    return JsonTraceStore(data_dir=tmp_path)


def test_market_only_scoring_step_sets_weights_from_market_vector(monkeypatch, store):
    bt = _bt_objects(uids=[0, 1, 2])
    cfg = _config(market=True, forecasting=False)

    monkeypatch.setattr(
        "src.validator.market.score_market",
        lambda **kw: np.array([0.1, 0.4, 0.5], dtype=float),
    )

    v = Validator(config=cfg, store=store, bt_objects=bt, metadata_manager=None)
    weights = v.run_scoring_step()

    assert bt.metagraph.synced == 1
    assert bt.subtensor.calls
    np.testing.assert_allclose(weights, np.array([0.1, 0.4, 0.5]))
    np.testing.assert_allclose(bt.subtensor.calls[0]["weights"], np.array([0.1, 0.4, 0.5]))


def test_score_market_forwards_budget_share(monkeypatch, store):
    bt = _bt_objects(uids=[0])
    cfg = _config(market=True, forecasting=False, market_share=0.37)
    seen: dict[str, object] = {}

    def _fake_score_market(**kw):
        seen.update(kw)
        return np.array([1.0], dtype=float)

    monkeypatch.setattr("src.validator.market.score_market", _fake_score_market)

    v = Validator(config=cfg, store=store, bt_objects=bt, metadata_manager=None)
    out = v._score_market()

    np.testing.assert_allclose(out, np.array([1.0]))
    assert seen["budget_share"] == pytest.approx(0.37)


def test_forecasting_only_scoring_step_scores_resolved_traces(store):
    bt = _bt_objects(uids=[0, 1])
    cfg = _config(market=False, forecasting=True)

    store.save_trace(_make_resolved_trace(miner_uid=1, prediction=1.0, outcome=True))

    v = Validator(config=cfg, store=store, bt_objects=bt, metadata_manager=None)
    weights = v.run_scoring_step()

    # Only uid=1 has a perfect prediction; uid=0 has none.
    np.testing.assert_allclose(weights, np.array([0.0, 1.0]))
    np.testing.assert_allclose(bt.subtensor.calls[0]["weights"], np.array([0.0, 1.0]))


def test_both_enabled_blends_score_vectors(monkeypatch, store):
    bt = _bt_objects(uids=[0, 1])
    cfg = _config(market=True, forecasting=True, market_share=0.5, forecasting_share=0.5)

    # Almanac Market says uid=0 gets everything.
    monkeypatch.setattr(
        "src.validator.market.score_market",
        lambda **kw: np.array([1.0, 0.0], dtype=float),
    )
    # Almanac Forecasting: uid=1 had a perfect prediction.
    store.save_trace(_make_resolved_trace(miner_uid=1, prediction=1.0, outcome=True))

    v = Validator(config=cfg, store=store, bt_objects=bt, metadata_manager=None)
    weights = v.run_scoring_step()

    np.testing.assert_allclose(weights, np.array([0.5, 0.5]))


def test_blended_scoring_step_keeps_expected_burn_weight(monkeypatch, store):
    bt = _bt_objects(uids=[0, 1, 210])
    cfg = _config(market=True, forecasting=True, market_share=0.6, forecasting_share=0.4)

    # Almanac Market vector includes burn at uid=210.
    monkeypatch.setattr(
        "src.validator.market.score_market",
        lambda **kw: np.array([0.7, 0.0, 0.3], dtype=float),
    )
    # Almanac Forecasting gives all weight to uid=1 (stubbed to keep test offline).
    monkeypatch.setattr(
        Validator,
        "_score_agent_predictions",
        lambda self: np.array([0.0, 1.0, 0.0], dtype=float),
    )

    v = Validator(config=cfg, store=store, bt_objects=bt, metadata_manager=None)
    weights = v.run_scoring_step()

    np.testing.assert_allclose(weights, np.array([0.42, 0.4, 0.18]))
    assert pytest.approx(float(weights.sum()), abs=1e-9) == 1.0


def test_market_score_failure_returns_none_and_falls_back_to_forecasting(monkeypatch, store):
    bt = _bt_objects(uids=[0, 1])
    cfg = _config(market=True, forecasting=True, market_share=1.0, forecasting_share=1.0)

    def _raises(**kw):
        raise RuntimeError("market api down")

    monkeypatch.setattr("src.validator.market.score_market", _raises)
    store.save_trace(_make_resolved_trace(miner_uid=1, prediction=1.0, outcome=True))

    v = Validator(config=cfg, store=store, bt_objects=bt, metadata_manager=None)
    weights = v.run_scoring_step()

    # Almanac Market failure -> blend reduces to forecasting (uid=1 only).
    np.testing.assert_allclose(weights, np.array([0.0, 1.0]))


def test_validator_rejects_both_mechanisms_disabled() -> None:
    cfg = _config(market=False, forecasting=False)
    bt = _bt_objects(uids=[0])

    with pytest.raises(ValueError, match="both mechanisms disabled"):
        Validator(config=cfg, store=None, bt_objects=bt, metadata_manager=None)


def test_set_weights_disabled_skips_chain_submission(monkeypatch, store):
    bt = _bt_objects(uids=[0, 1, 2])
    cfg = _config(market=True, forecasting=False)
    cfg.loop.set_weights_enabled = False

    monkeypatch.setattr(
        "src.validator.market.score_market",
        lambda **kw: np.array([0.1, 0.4, 0.5], dtype=float),
    )

    v = Validator(config=cfg, store=store, bt_objects=bt, metadata_manager=None)
    weights = v.run_scoring_step()

    assert bt.metagraph.synced == 1
    assert not bt.subtensor.calls
    np.testing.assert_allclose(weights, np.array([0.1, 0.4, 0.5]))
