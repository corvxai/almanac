"""Scoring step coverage for the production ``Validator`` class.

These tests do not touch the chain. They inject a fake ``_BtObjects`` plus a
stub ``MetadataManager`` and call ``Validator.run_scoring_step`` directly, then
assert on what ``subtensor.set_weights`` received. Three cases:

- Almanac only.
- Arcratio only.
- Both enabled and blended.

Almanac scoring is monkey-patched (the cvxpy-backed path is exercised
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


def _config(*, almanac: bool, arcratio: bool, almanac_share=1.0, arcratio_share=1.0) -> AppConfig:
    cfg = AppConfig()
    cfg.loop.loop_enabled = True
    cfg.loop.almanac_enabled = almanac
    cfg.loop.arcratio_enabled = arcratio
    cfg.loop.almanac_weight_share = almanac_share
    cfg.loop.arcratio_weight_share = arcratio_share
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


def test_almanac_only_scoring_step_sets_weights_from_almanac_vector(monkeypatch, store):
    bt = _bt_objects(uids=[0, 1, 2])
    cfg = _config(almanac=True, arcratio=False)

    monkeypatch.setattr(
        "src.validator.almanac.score_almanac",
        lambda **kw: np.array([0.1, 0.4, 0.5], dtype=float),
    )

    v = Validator(config=cfg, store=store, bt_objects=bt, metadata_manager=None)
    weights = v.run_scoring_step()

    assert bt.metagraph.synced == 1
    assert bt.subtensor.calls
    np.testing.assert_allclose(weights, np.array([0.1, 0.4, 0.5]))
    np.testing.assert_allclose(bt.subtensor.calls[0]["weights"], np.array([0.1, 0.4, 0.5]))


def test_arcratio_only_scoring_step_scores_resolved_traces(store):
    bt = _bt_objects(uids=[0, 1])
    cfg = _config(almanac=False, arcratio=True)

    store.save_trace(_make_resolved_trace(miner_uid=1, prediction=1.0, outcome=True))

    v = Validator(config=cfg, store=store, bt_objects=bt, metadata_manager=None)
    weights = v.run_scoring_step()

    # Only uid=1 has a perfect prediction; uid=0 has none.
    np.testing.assert_allclose(weights, np.array([0.0, 1.0]))
    np.testing.assert_allclose(bt.subtensor.calls[0]["weights"], np.array([0.0, 1.0]))


def test_both_enabled_blends_score_vectors(monkeypatch, store):
    bt = _bt_objects(uids=[0, 1])
    cfg = _config(almanac=True, arcratio=True, almanac_share=0.5, arcratio_share=0.5)

    # Almanac says uid=0 gets everything.
    monkeypatch.setattr(
        "src.validator.almanac.score_almanac",
        lambda **kw: np.array([1.0, 0.0], dtype=float),
    )
    # Arcratio: uid=1 had a perfect prediction.
    store.save_trace(_make_resolved_trace(miner_uid=1, prediction=1.0, outcome=True))

    v = Validator(config=cfg, store=store, bt_objects=bt, metadata_manager=None)
    weights = v.run_scoring_step()

    np.testing.assert_allclose(weights, np.array([0.5, 0.5]))


def test_almanac_score_failure_returns_none_and_falls_back_to_arcratio(monkeypatch, store):
    bt = _bt_objects(uids=[0, 1])
    cfg = _config(almanac=True, arcratio=True, almanac_share=1.0, arcratio_share=1.0)

    def _raises(**kw):
        raise RuntimeError("almanac api down")

    monkeypatch.setattr("src.validator.almanac.score_almanac", _raises)
    store.save_trace(_make_resolved_trace(miner_uid=1, prediction=1.0, outcome=True))

    v = Validator(config=cfg, store=store, bt_objects=bt, metadata_manager=None)
    weights = v.run_scoring_step()

    # Almanac failure -> blend reduces to arcratio (uid=1 only).
    np.testing.assert_allclose(weights, np.array([0.0, 1.0]))


def test_validator_rejects_both_mechanisms_disabled() -> None:
    cfg = _config(almanac=False, arcratio=False)
    bt = _bt_objects(uids=[0])

    with pytest.raises(ValueError, match="both mechanisms disabled"):
        Validator(config=cfg, store=None, bt_objects=bt, metadata_manager=None)


def test_set_weights_disabled_skips_chain_submission(monkeypatch, store):
    bt = _bt_objects(uids=[0, 1, 2])
    cfg = _config(almanac=True, arcratio=False)
    cfg.loop.set_weights_enabled = False

    monkeypatch.setattr(
        "src.validator.almanac.score_almanac",
        lambda **kw: np.array([0.1, 0.4, 0.5], dtype=float),
    )

    v = Validator(config=cfg, store=store, bt_objects=bt, metadata_manager=None)
    weights = v.run_scoring_step()

    assert bt.metagraph.synced == 1
    assert not bt.subtensor.calls
    np.testing.assert_allclose(weights, np.array([0.1, 0.4, 0.5]))
