"""Tests for the v1 arcratio Brier scoring stub.

Exercises ``src.validator.scoring.score_arcratio`` against a temp trace
store with hand-constructed ``EvidenceDigest`` objects. No bittensor or
network dependencies.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

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
from src.validator.scoring import score_arcratio


class _StubMetagraph:
    """Minimal metagraph stub that satisfies ``score_arcratio`` + ``uid_map``."""

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


def _make_trace(
    *,
    miner_uid: int | None,
    final_probability: float,
    outcome: bool,
    resolved_at: datetime,
) -> EvidenceDigest:
    metadata: dict = {}
    if miner_uid is not None:
        metadata["miner_uid"] = miner_uid

    pred = PredictionOutput(
        final_probability=final_probability,
        reasoning_summary="test",
        metadata=metadata or None,
    )
    rr = ResolutionRecord(
        resolved=True,
        resolution_outcome=outcome,
        resolution_timestamp=resolved_at,
    )
    exec_ctx = ExecutionContext(
        execution_id=uuid4(),
        agent_id=uuid4(),
        agent_version="0.0.1-test",
        event_id=uuid4(),
        validator_id=uuid4(),
        timestamp_start=resolved_at - timedelta(hours=1),
        timestamp_end=resolved_at - timedelta(hours=1) + timedelta(seconds=10),
        execution_duration_ms=10_000,
        sandbox_environment=SandboxEnvironment.IN_PROCESS,
    )
    event_snap = EventSnapshot(
        event_title="test",
        event_category=EventCategory.OTHER,
        resolution_criteria="-",
        resolution_deadline=resolved_at,
        event_created_at=resolved_at - timedelta(days=7),
    )
    integrity = TraceIntegrity(
        trace_hash="",
        total_provider_cost=0.0,
        total_evidence_items=0,
    )
    digest = EvidenceDigest(
        execution_context=exec_ctx,
        event_snapshot=event_snap,
        provider_calls=[],
        reasoning_chain=[],
        prediction_output=pred,
        resolution_record=rr,
        trace_integrity=integrity,
    )
    return digest.seal()


def _store(tmp_path: Path) -> JsonTraceStore:
    return JsonTraceStore(data_dir=tmp_path)


def test_empty_trace_dir_returns_zeros(tmp_path: Path) -> None:
    store = _store(tmp_path)
    metagraph = _StubMetagraph(uids=[0, 1, 2])

    out = score_arcratio(metagraph=metagraph, trace_store=store)

    np.testing.assert_allclose(out, np.zeros(3))


def test_perfect_prediction_yields_score_of_one(tmp_path: Path) -> None:
    store = _store(tmp_path)
    metagraph = _StubMetagraph(uids=[0, 1, 2])

    now = datetime.now(timezone.utc)
    trace = _make_trace(
        miner_uid=1,
        final_probability=1.0,
        outcome=True,
        resolved_at=now - timedelta(days=1),
    )
    store.save_trace(trace)

    out = score_arcratio(metagraph=metagraph, trace_store=store, now=now)

    assert out[0] == 0.0
    assert pytest.approx(out[1]) == 1.0
    assert out[2] == 0.0


def test_worst_prediction_yields_score_of_zero(tmp_path: Path) -> None:
    store = _store(tmp_path)
    metagraph = _StubMetagraph(uids=[7])

    now = datetime.now(timezone.utc)
    store.save_trace(
        _make_trace(
            miner_uid=7,
            final_probability=0.0,
            outcome=True,
            resolved_at=now - timedelta(days=2),
        )
    )

    out = score_arcratio(metagraph=metagraph, trace_store=store, now=now)

    assert pytest.approx(out[0]) == 0.0


def test_multiple_traces_average_their_briers(tmp_path: Path) -> None:
    store = _store(tmp_path)
    metagraph = _StubMetagraph(uids=[5])

    now = datetime.now(timezone.utc)
    store.save_trace(
        _make_trace(
            miner_uid=5,
            final_probability=1.0,
            outcome=True,
            resolved_at=now - timedelta(days=1),
        )
    )
    store.save_trace(
        _make_trace(
            miner_uid=5,
            final_probability=0.0,
            outcome=True,
            resolved_at=now - timedelta(days=2),
        )
    )

    out = score_arcratio(metagraph=metagraph, trace_store=store, now=now)

    # Brier_avg = (0 + 1) / 2 = 0.5 -> score = 1 - 0.5 = 0.5
    assert pytest.approx(out[0]) == 0.5


def test_trace_outside_rolling_window_is_ignored(tmp_path: Path) -> None:
    store = _store(tmp_path)
    metagraph = _StubMetagraph(uids=[0])

    now = datetime.now(timezone.utc)
    # 100 days old, default window is 30
    store.save_trace(
        _make_trace(
            miner_uid=0,
            final_probability=1.0,
            outcome=True,
            resolved_at=now - timedelta(days=100),
        )
    )

    out = score_arcratio(metagraph=metagraph, trace_store=store, now=now)
    np.testing.assert_allclose(out, np.zeros(1))


def test_unresolved_trace_is_ignored(tmp_path: Path) -> None:
    store = _store(tmp_path)
    metagraph = _StubMetagraph(uids=[0])

    now = datetime.now(timezone.utc)
    trace = _make_trace(
        miner_uid=0,
        final_probability=1.0,
        outcome=True,
        resolved_at=now - timedelta(days=1),
    )
    # Forge an unresolved record on top.
    trace = trace.model_copy(update={"resolution_record": ResolutionRecord(resolved=False)})
    store.save_trace(trace)

    out = score_arcratio(metagraph=metagraph, trace_store=store, now=now)
    np.testing.assert_allclose(out, np.zeros(1))


def test_unmapped_uid_is_skipped_not_counted(tmp_path: Path) -> None:
    store = _store(tmp_path)
    metagraph = _StubMetagraph(uids=[0])

    now = datetime.now(timezone.utc)
    # miner_uid=None -> uid_map.resolve returns None and logs once
    store.save_trace(
        _make_trace(
            miner_uid=None,
            final_probability=1.0,
            outcome=True,
            resolved_at=now - timedelta(days=1),
        )
    )

    out = score_arcratio(metagraph=metagraph, trace_store=store, now=now)
    np.testing.assert_allclose(out, np.zeros(1))
