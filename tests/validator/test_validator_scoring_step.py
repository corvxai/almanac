"""Scoring step coverage for the production ``Validator`` class.

These tests do not touch the chain. They inject a fake ``_BtObjects`` plus a
stub ``MetadataManager`` and call ``Validator.run_scoring_step`` directly, then
assert on what ``subtensor.execute`` received. Three cases:

- Almanac Market only.
- Almanac Forecasting only.
- Both enabled and blended.

Almanac Market scoring is monkey-patched (the cvxpy-backed path is exercised
elsewhere by sn41's own simulator scripts; we only need to confirm the
``Validator`` wires the call + flows the result into the blender).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from src.core import constants
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
from src.validator.market.constants import BURN_UID
from src.validator.validator import Validator


class _StubMetagraph:
    def __init__(self, uids: list[int]) -> None:
        self.neurons = [
            SimpleNamespace(uid=uid, hotkey=f"hotkey_{uid}") for uid in uids
        ]
        self.hotkeys = [f"hotkey_{u}" for u in uids]
        self.synced = 0

    def __iter__(self):
        return iter(self.neurons)

    def __len__(self) -> int:
        return len(self.neurons)


class _RecordingSubtensor:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.metagraph = None
        self.subnets = SimpleNamespace(metagraph=self._metagraph)

    def _metagraph(self, netuid):
        del netuid
        self.metagraph.synced += 1
        return self.metagraph

    def execute(self, intent, wallet, *, wait_for_inclusion):
        del wallet
        self.calls.append(
            {
                "netuid": intent.netuid,
                "uids": list(intent.uids),
                "weights": np.asarray(intent.weights, dtype=float),
                "wait_for_inclusion": wait_for_inclusion,
            }
        )
        return SimpleNamespace(success=True, error=None, message="")


def _bt_objects(uids: list[int]):
    metagraph = _StubMetagraph(uids)
    subtensor = _RecordingSubtensor()
    subtensor.metagraph = metagraph
    return validator_module._BtObjects(  # noqa: SLF001 — test reaches into module
        wallet=SimpleNamespace(hotkey=SimpleNamespace(ss58_address="hotkey_test")),
        subtensor=subtensor,
        metagraph=metagraph,
        network="finney",
    )


def _config(
    *,
    market: bool,
    forecasting: bool,
    market_share=1.0,
    forecasting_share=1.0,
    data_dir: Path | None = None,
) -> AppConfig:
    cfg = AppConfig()
    cfg.loop.loop_enabled = True
    cfg.loop.market_enabled = market
    cfg.loop.forecasting_enabled = forecasting
    cfg.loop.market_weight_share = market_share
    cfg.loop.forecasting_weight_share = forecasting_share
    cfg.loop.wandb_enabled = False
    if data_dir is not None:
        cfg.storage.data_dir = data_dir
    return cfg


def _snapshot_path(data_dir: Path, mechanism: str, netuid: int = constants.BITTENSOR.netuid) -> Path:
    return data_dir / f"weight_snapshot_{mechanism}_{netuid}.json"


def _snapshot(data_dir: Path, mechanism: str, netuid: int = constants.BITTENSOR.netuid) -> dict:
    return json.loads(_snapshot_path(data_dir, mechanism, netuid).read_text())


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


def test_market_only_scoring_step_sets_weights_from_market_vector(monkeypatch, store, caplog, tmp_path):
    bt = _bt_objects(uids=[0, 1, 2])
    cfg = _config(market=True, forecasting=False, data_dir=tmp_path)

    monkeypatch.setattr(
        "src.validator.market.score_market",
        lambda **kw: np.array([0.1, 0.4, 0.5], dtype=float),
    )

    v = Validator(config=cfg, store=store, bt_objects=bt, metadata_manager=None)
    with caplog.at_level(logging.INFO, logger="almanac.validator"):
        weights = v.run_scoring_step()

    assert f"weights_version_key={cfg.bittensor.weights_version_key}" in caplog.text
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


def test_forecasting_only_scoring_step_scores_resolved_traces(monkeypatch, store, tmp_path):
    bt = _bt_objects(uids=[0, 1])
    cfg = _config(market=False, forecasting=True, data_dir=tmp_path)

    store.save_trace(_make_resolved_trace(miner_uid=1, prediction=1.0, outcome=True))
    monkeypatch.setattr(
        Validator,
        "_score_agent_predictions",
        lambda self: np.array([0.0, 1.0], dtype=float),
    )

    v = Validator(config=cfg, store=store, bt_objects=bt, metadata_manager=None)
    weights = v.run_scoring_step()

    # Only uid=1 has a perfect prediction; uid=0 has none.
    np.testing.assert_allclose(weights, np.array([0.0, 1.0]))
    np.testing.assert_allclose(bt.subtensor.calls[0]["weights"], np.array([0.0, 1.0]))


def test_both_enabled_blends_score_vectors(monkeypatch, store, tmp_path):
    bt = _bt_objects(uids=[0, 1])
    cfg = _config(market=True, forecasting=True, market_share=0.5, forecasting_share=0.5, data_dir=tmp_path)

    # Almanac Market says uid=0 gets everything.
    monkeypatch.setattr(
        "src.validator.market.score_market",
        lambda **kw: np.array([1.0, 0.0], dtype=float),
    )
    # Almanac Forecasting: uid=1 had a perfect prediction.
    store.save_trace(_make_resolved_trace(miner_uid=1, prediction=1.0, outcome=True))
    monkeypatch.setattr(
        Validator,
        "_score_agent_predictions",
        lambda self: np.array([0.0, 1.0], dtype=float),
    )

    v = Validator(config=cfg, store=store, bt_objects=bt, metadata_manager=None)
    weights = v.run_scoring_step()

    np.testing.assert_allclose(weights, np.array([0.5, 0.5]))


def test_blended_scoring_step_keeps_expected_burn_weight(monkeypatch, store, tmp_path):
    bt = _bt_objects(uids=[0, 1, 210])
    cfg = _config(market=True, forecasting=True, market_share=0.6, forecasting_share=0.4, data_dir=tmp_path)

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


def test_market_score_failure_without_snapshot_skips_set_weights(monkeypatch, store, tmp_path, caplog):
    bt = _bt_objects(uids=[0, 1])
    cfg = _config(market=True, forecasting=True, market_share=0.9, forecasting_share=0.1, data_dir=tmp_path)

    def _raises(**kw):
        raise RuntimeError("market api down")

    monkeypatch.setattr("src.validator.market.score_market", _raises)
    monkeypatch.setattr(
        Validator,
        "_score_agent_predictions",
        lambda self: np.array([0.0, 1.0], dtype=float),
    )

    v = Validator(config=cfg, store=store, bt_objects=bt, metadata_manager=None)
    with caplog.at_level(logging.WARNING, logger="almanac.validator"):
        weights = v.run_scoring_step()

    assert weights is None
    assert not bt.subtensor.calls
    assert "market weight snapshot is missing" in caplog.text
    assert "Skipping set_weights: market snapshot unavailable" in caplog.text


def test_validator_rejects_both_mechanisms_disabled() -> None:
    cfg = _config(market=False, forecasting=False)
    bt = _bt_objects(uids=[0])

    with pytest.raises(ValueError, match="both mechanisms disabled"):
        Validator(config=cfg, store=None, bt_objects=bt, metadata_manager=None)


def test_all_zero_weights_skip_chain_submission(monkeypatch, store, caplog, tmp_path):
    bt = _bt_objects(uids=[0, 1, 2])
    cfg = _config(market=True, forecasting=False, data_dir=tmp_path)
    monkeypatch.setattr(
        "src.validator.market.score_market",
        lambda **kw: np.zeros(3, dtype=float),
    )

    v = Validator(config=cfg, store=store, bt_objects=bt, metadata_manager=None)
    weights = v.run_scoring_step()

    assert weights is None
    assert not bt.subtensor.calls
    assert "All blended weights are zero" in caplog.text
    assert [entry["weight"] for entry in _snapshot(tmp_path, "market")["entries"]] == [0.0, 0.0, 0.0]


def test_set_weights_disabled_skips_chain_submission(monkeypatch, store, tmp_path):
    bt = _bt_objects(uids=[0, 1, 2])
    cfg = _config(market=True, forecasting=False, data_dir=tmp_path)
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


def test_scoring_step_saves_each_mechanism_with_uid_hotkey_and_timestamp(monkeypatch, store, tmp_path):
    saved_at = datetime(2026, 10, 15, 16, 29, tzinfo=timezone.utc)
    bt = _bt_objects(uids=[0, 1, 210])
    cfg = _config(market=True, forecasting=True, market_share=0.9, forecasting_share=0.1, data_dir=tmp_path)
    monkeypatch.setattr(
        "src.validator.market.score_market",
        lambda **kw: np.array([0.7, 0.0, 0.3], dtype=float),
    )
    monkeypatch.setattr(
        Validator,
        "_score_agent_predictions",
        lambda self: np.array([0.0, 1.0, 0.25], dtype=float),
    )

    v = Validator(
        config=cfg,
        store=store,
        bt_objects=bt,
        metadata_manager=None,
        clock=lambda: saved_at,
    )
    v.run_scoring_step()

    market = _snapshot(tmp_path, "market")
    forecasting = _snapshot(tmp_path, "forecasting")
    assert market["saved_at"] == "2026-10-15T16:29:00+00:00"
    assert forecasting["saved_at"] == "2026-10-15T16:29:00+00:00"
    assert market["entries"] == [
        {"uid": 0, "hotkey": "hotkey_0", "weight": 0.7},
        {"uid": 1, "hotkey": "hotkey_1", "weight": 0.0},
        {"uid": 210, "hotkey": "hotkey_210", "weight": 0.3},
    ]
    assert forecasting["entries"] == [
        {"uid": 0, "hotkey": "hotkey_0", "weight": 0.0},
        {"uid": 1, "hotkey": "hotkey_1", "weight": 1.0},
        {"uid": 210, "hotkey": "hotkey_210", "weight": 0.25},
    ]
    assert not _snapshot_path(tmp_path, "blended").exists()


def test_failed_side_does_not_overwrite_its_snapshot(monkeypatch, store, tmp_path):
    now = {"t": datetime(2026, 10, 15, 16, 29, tzinfo=timezone.utc)}
    scores = {
        "market": np.array([0.7, 0.3], dtype=float),
        "forecasting": np.array([0.2, 0.8], dtype=float),
    }
    bt = _bt_objects(uids=[4, 9])
    cfg = _config(market=True, forecasting=True, data_dir=tmp_path)
    monkeypatch.setattr("src.validator.market.score_market", lambda **kw: scores["market"])
    monkeypatch.setattr(Validator, "_score_agent_predictions", lambda self: scores["forecasting"])

    v = Validator(
        config=cfg,
        store=store,
        bt_objects=bt,
        metadata_manager=None,
        clock=lambda: now["t"],
    )
    v.run_scoring_step()
    first_forecasting = _snapshot_path(tmp_path, "forecasting").read_text()

    now["t"] = datetime(2026, 10, 15, 17, 29, tzinfo=timezone.utc)
    scores["market"] = np.array([0.1, 0.9], dtype=float)
    scores["forecasting"] = None
    v.run_scoring_step()

    market = _snapshot(tmp_path, "market")
    assert market["saved_at"] == "2026-10-15T17:29:00+00:00"
    assert [entry["weight"] for entry in market["entries"]] == [0.1, 0.9]
    assert [entry["uid"] for entry in market["entries"]] == [4, 9]
    assert _snapshot_path(tmp_path, "forecasting").read_text() == first_forecasting


def test_snapshot_length_mismatch_keeps_the_previous_file(monkeypatch, store, tmp_path):
    bt = _bt_objects(uids=[0, 1])
    cfg = _config(market=True, forecasting=False, data_dir=tmp_path)
    monkeypatch.setattr(
        "src.validator.market.score_market",
        lambda **kw: np.array([0.4, 0.6], dtype=float),
    )
    v = Validator(config=cfg, store=store, bt_objects=bt, metadata_manager=None)
    v.run_scoring_step()
    original = _snapshot_path(tmp_path, "market").read_text()

    v._write_weight_snapshot("market", np.array([1.0], dtype=float))

    assert _snapshot_path(tmp_path, "market").read_text() == original


def test_snapshot_write_failure_still_sets_weights(monkeypatch, store, tmp_path):
    bt = _bt_objects(uids=[0, 1])
    cfg = _config(market=True, forecasting=False, data_dir=tmp_path)
    monkeypatch.setattr(
        "src.validator.market.score_market",
        lambda **kw: np.array([0.4, 0.6], dtype=float),
    )

    def _boom(path, text):
        raise OSError("disk full")

    monkeypatch.setattr("src.validator.validator._atomic_write_text", _boom)
    v = Validator(config=cfg, store=store, bt_objects=bt, metadata_manager=None)
    weights = v.run_scoring_step()

    np.testing.assert_allclose(weights, np.array([0.4, 0.6]))
    assert bt.subtensor.calls
    assert not _snapshot_path(tmp_path, "market").exists()


def _run_both(
    monkeypatch,
    store,
    tmp_path,
    *,
    uids: list[int],
    market: np.ndarray,
    forecasting: np.ndarray,
    when: datetime,
    market_share: float = 0.9,
    forecasting_share: float = 0.1,
):
    """Score once so both snapshots exist, then return the validator and a mutable clock."""
    bt = _bt_objects(uids=uids)
    cfg = _config(
        market=True,
        forecasting=True,
        market_share=market_share,
        forecasting_share=forecasting_share,
        data_dir=tmp_path,
    )
    scores = {"market": market, "forecasting": forecasting}
    now = {"t": when}
    monkeypatch.setattr("src.validator.market.score_market", lambda **kw: scores["market"])
    monkeypatch.setattr(Validator, "_score_agent_predictions", lambda self: scores["forecasting"])
    v = Validator(
        config=cfg,
        store=store,
        bt_objects=bt,
        metadata_manager=None,
        clock=lambda: now["t"],
    )
    v.run_scoring_step()
    bt.subtensor.calls.clear()
    return v, bt, scores, now


def test_fresh_side_blends_with_stale_snapshot(monkeypatch, store, tmp_path, caplog):
    start = datetime(2026, 10, 15, 16, 29, tzinfo=timezone.utc)
    v, bt, scores, now = _run_both(
        monkeypatch,
        store,
        tmp_path,
        uids=[0, 1],
        market=np.array([0.6, 0.4]),
        forecasting=np.array([0.0, 1.0]),
        when=start,
    )
    forecasting_saved = _snapshot_path(tmp_path, "forecasting").read_text()

    now["t"] = start + timedelta(hours=1)
    scores["market"] = np.array([1.0, 0.0])
    scores["forecasting"] = None
    with caplog.at_level(logging.INFO, logger="almanac.validator"):
        weights = v.run_scoring_step()

    np.testing.assert_allclose(weights, np.array([0.9, 0.1]))
    np.testing.assert_allclose(bt.subtensor.calls[0]["weights"], np.array([0.9, 0.1]))
    assert "forecasting=stale age=1.00h" in caplog.text
    assert "market=present" in caplog.text
    assert "effective=0.1000" in caplog.text
    assert _snapshot_path(tmp_path, "forecasting").read_text() == forecasting_saved


def test_changed_hotkey_is_dropped_from_stale_snapshot(monkeypatch, store, tmp_path, caplog):
    start = datetime(2026, 10, 15, 16, 29, tzinfo=timezone.utc)
    v, bt, scores, now = _run_both(
        monkeypatch,
        store,
        tmp_path,
        uids=[0, 1, BURN_UID],
        market=np.array([0.5, 0.3, 0.2]),
        forecasting=np.array([0.25, 0.25, 0.5]),
        when=start,
    )

    bt.metagraph.hotkeys[1] = "hotkey_recycled"
    now["t"] = start + timedelta(hours=2)
    scores["forecasting"] = None
    with caplog.at_level(logging.INFO, logger="almanac.validator"):
        weights = v.run_scoring_step()

    # uid 1 keeps its fresh market weight; its saved forecasting weight is
    # burned, so uid 0 stays at 0.25 of the forecasting slice.
    np.testing.assert_allclose(weights, np.array([0.475, 0.27, 0.255]))
    assert (
        "forecasting weight snapshot dropped 1 uid(s) whose hotkey changed or left the "
        f"metagraph; 0.2500 weight moved to BURN_UID {BURN_UID}"
    ) in caplog.text
    assert "forecasting=stale age=2.00h" in caplog.text


def test_changed_hotkey_without_burn_slot_skips_set_weights(monkeypatch, store, tmp_path, caplog):
    start = datetime(2026, 10, 15, 16, 29, tzinfo=timezone.utc)
    v, bt, scores, now = _run_both(
        monkeypatch,
        store,
        tmp_path,
        uids=[0, 1],
        market=np.array([0.5, 0.5]),
        forecasting=np.array([0.25, 0.75]),
        when=start,
    )

    bt.metagraph.hotkeys[1] = "hotkey_recycled"
    now["t"] = start + timedelta(hours=2)
    scores["forecasting"] = None
    with caplog.at_level(logging.WARNING, logger="almanac.validator"):
        weights = v.run_scoring_step()

    assert weights is None
    assert not bt.subtensor.calls
    assert "forecasting weight snapshot could not be aligned" in caplog.text


def _restore_from_file(store, tmp_path, *, uids: list[int], text: str, now: datetime):
    bt = _bt_objects(uids=uids)
    cfg = _config(market=True, forecasting=False, data_dir=tmp_path)
    v = Validator(config=cfg, store=store, bt_objects=bt, metadata_manager=None, clock=lambda: now)
    _snapshot_path(tmp_path, "market").write_text(text)
    return v._restore_weight_snapshot("market")


def test_uid_missing_from_metagraph_is_burned(store, tmp_path):
    now = datetime(2026, 10, 15, 16, 29, tzinfo=timezone.utc)
    entries = [
        {"uid": 0, "hotkey": "hotkey_0", "weight": 0.4},
        {"uid": 7, "hotkey": "hotkey_7", "weight": 0.3},
        {"uid": BURN_UID, "hotkey": f"hotkey_{BURN_UID}", "weight": 0.3},
    ]
    restored = _restore_from_file(
        store,
        tmp_path,
        uids=[0, BURN_UID],
        text=json.dumps({"saved_at": "2026-10-15T15:29:00+00:00", "entries": entries}),
        now=now,
    )

    assert restored is not None
    np.testing.assert_allclose(restored.vector, np.array([0.4, 0.6]))


def test_snapshot_dated_in_the_future_is_rejected(store, tmp_path, caplog):
    now = datetime(2026, 10, 15, 16, 29, tzinfo=timezone.utc)
    entries = [{"uid": 0, "hotkey": "hotkey_0", "weight": 1.0}]
    with caplog.at_level(logging.WARNING, logger="almanac.validator"):
        restored = _restore_from_file(
            store,
            tmp_path,
            uids=[0],
            text=json.dumps({"saved_at": "2026-10-15T16:30:00+00:00", "entries": entries}),
            now=now,
        )

    assert restored is None
    assert "after the current time" in caplog.text


@pytest.mark.parametrize("bad_weight", ["NaN", "Infinity", "-Infinity"])
def test_snapshot_with_non_finite_weight_is_rejected(store, tmp_path, bad_weight):
    now = datetime(2026, 10, 15, 16, 29, tzinfo=timezone.utc)
    text = (
        '{"saved_at": "2026-10-15T15:29:00+00:00", "entries": ['
        f'{{"uid": 0, "hotkey": "hotkey_0", "weight": {bad_weight}}}, '
        '{"uid": 1, "hotkey": "hotkey_1", "weight": 1.0}]}'
    )
    restored = _restore_from_file(store, tmp_path, uids=[0, 1], text=text, now=now)

    assert restored is None


def test_non_finite_vector_is_not_saved(store, tmp_path):
    bt = _bt_objects(uids=[0, 1])
    cfg = _config(market=True, forecasting=False, data_dir=tmp_path)
    v = Validator(config=cfg, store=store, bt_objects=bt, metadata_manager=None)

    v._write_weight_snapshot("market", np.array([np.nan, 1.0]))

    assert not _snapshot_path(tmp_path, "market").exists()


def test_both_failures_blend_fresh_snapshots(monkeypatch, store, tmp_path, caplog):
    start = datetime(2026, 10, 15, 16, 29, tzinfo=timezone.utc)
    v, bt, scores, now = _run_both(
        monkeypatch,
        store,
        tmp_path,
        uids=[0, 1],
        market=np.array([0.6, 0.4]),
        forecasting=np.array([0.0, 1.0]),
        when=start,
    )
    market_saved = _snapshot(tmp_path, "market")["saved_at"]
    forecasting_saved = _snapshot(tmp_path, "forecasting")["saved_at"]

    now["t"] = start + timedelta(hours=2)
    scores["market"] = None
    scores["forecasting"] = None
    with caplog.at_level(logging.INFO, logger="almanac.validator"):
        weights = v.run_scoring_step()

    np.testing.assert_allclose(weights, np.array([0.54, 0.46]))
    assert bt.subtensor.calls
    assert "market=stale age=2.00h" in caplog.text
    assert "forecasting=stale age=2.00h" in caplog.text
    assert _snapshot(tmp_path, "market")["saved_at"] == market_saved
    assert _snapshot(tmp_path, "forecasting")["saved_at"] == forecasting_saved


def test_snapshot_at_max_age_skips_set_weights(monkeypatch, store, tmp_path, caplog):
    start = datetime(2026, 10, 15, 16, 29, tzinfo=timezone.utc)
    v, bt, scores, now = _run_both(
        monkeypatch,
        store,
        tmp_path,
        uids=[0, 1],
        market=np.array([0.6, 0.4]),
        forecasting=np.array([0.0, 1.0]),
        when=start,
    )

    now["t"] = start + timedelta(hours=6)
    scores["forecasting"] = None
    with caplog.at_level(logging.WARNING, logger="almanac.validator"):
        weights = v.run_scoring_step()

    assert weights is None
    assert not bt.subtensor.calls
    assert "forecasting weight snapshot is 6.00h old; limit is 6h" in caplog.text
    assert "Skipping set_weights: forecasting snapshot unavailable" in caplog.text


def test_single_mechanism_failure_reuses_its_snapshot(monkeypatch, store, tmp_path, caplog):
    start = datetime(2026, 10, 15, 16, 29, tzinfo=timezone.utc)
    bt = _bt_objects(uids=[0, 1])
    cfg = _config(market=True, forecasting=False, data_dir=tmp_path)
    scores = {"market": np.array([0.2, 0.8])}
    now = {"t": start}
    monkeypatch.setattr("src.validator.market.score_market", lambda **kw: scores["market"])
    v = Validator(
        config=cfg,
        store=store,
        bt_objects=bt,
        metadata_manager=None,
        clock=lambda: now["t"],
    )
    v.run_scoring_step()
    bt.subtensor.calls.clear()

    now["t"] = start + timedelta(hours=3)
    scores["market"] = None
    with caplog.at_level(logging.INFO, logger="almanac.validator"):
        weights = v.run_scoring_step()

    np.testing.assert_allclose(weights, np.array([0.2, 0.8]))
    assert "market=stale age=3.00h" in caplog.text


def test_disabled_mechanism_is_not_restored_from_snapshot(monkeypatch, store, tmp_path):
    start = datetime(2026, 10, 15, 16, 29, tzinfo=timezone.utc)
    v, bt, scores, now = _run_both(
        monkeypatch,
        store,
        tmp_path,
        uids=[0, 1],
        market=np.array([0.6, 0.4]),
        forecasting=np.array([0.0, 1.0]),
        when=start,
        market_share=0.9,
        forecasting_share=0.1,
    )
    v._config.loop.forecasting_enabled = False
    now["t"] = start + timedelta(hours=1)
    scores["market"] = np.array([1.0, 0.0])
    scores["forecasting"] = None

    weights = v.run_scoring_step()

    np.testing.assert_allclose(weights, np.array([1.0, 0.0]))
    assert bt.subtensor.calls
