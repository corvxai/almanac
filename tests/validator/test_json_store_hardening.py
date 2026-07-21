"""F3/F6 hardening tests for the JSON trace store.

F3: an unparseable trace file must be logged and skipped/returned-None, never
    raise into a scoring sweep.
F6: the on-disk snake_case digest verifies its own sealed hash; the submitted
    (compacted camelCase) traceHash is a KNOWN, documented non-recomputable gap.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from uuid import uuid4

import pytest

import src.storage.json_store as json_store_mod


@contextmanager
def _capture_json_store_warnings():
    """Capture WARNING records straight off the json_store module logger.

    Deliberately does NOT use the ``caplog`` fixture: other tests in the suite
    import ``bittensor``, which reconfigures the global logging system and breaks
    root-propagation-based capture. Attaching our own handler to the module logger
    is immune to that global mutation.
    """
    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _ListHandler(level=logging.WARNING)
    logger = json_store_mod.logger
    prev_level = logger.level
    prev_disabled = logger.disabled
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    logger.disabled = False
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)
        logger.disabled = prev_disabled

from src.core.events import Event
from src.core.schemas import (
    AgentResult,
    EventCategory,
    ExecutionContext,
    ProviderCall,
    ProviderTier,
    ResponseMeta,
    SandboxEnvironment,
)
from src.agent.examples._v1_common import belief_path_single_final
from src.storage.json_store import JsonTraceStore
from src.validator.forecasting.trace_assembler import assemble_trace


def _sealed_digest():
    now = datetime(2026, 4, 24, tzinfo=timezone.utc)
    event = Event(
        event_id=uuid4(), title="Will X happen?", description="Binary event.",
        category=EventCategory.OTHER, resolution_criteria="YES iff X occurs.",
        resolution_deadline=datetime(2027, 1, 1, tzinfo=timezone.utc), created_at=now,
    )
    exec_ctx = ExecutionContext(
        execution_id=uuid4(), agent_id=uuid4(), agent_version="1.0.0", event_id=uuid4(),
        validator_id=uuid4(), timestamp_start=now, timestamp_end=now,
        execution_duration_ms=1, sandbox_environment=SandboxEnvironment.DOCKER_RUNC,
    )
    provider_call = ProviderCall(
        call_index=0, provider_id="openrouter", model="m:online",
        provider_tier=ProviderTier.SEARCH, call_type="chat_completion", query_params_summary="q",
        response_meta=ResponseMeta(response_size_bytes=100, data_freshness=now),
        extracted_evidence=[], raw_response_hash="h", latency_ms=10, cost_units=0.0,
    )
    return assemble_trace(
        execution_context=exec_ctx, event=event, provider_calls=[provider_call],
        agent_result=AgentResult(prediction=0.62, confidence=0.7, reasoning="r",
                                 beliefPath=belief_path_single_final(0.62, "r")),
    )


def test_iter_all_traces_logs_and_skips_unparseable(tmp_path) -> None:
    store = JsonTraceStore(data_dir=tmp_path)
    good = _sealed_digest()
    store.save_trace(good)
    # Write a corrupt file alongside the good one.
    (tmp_path / "traces" / "broken.json").write_text("{ not valid json ")

    with _capture_json_store_warnings() as records:
        traces = store.list_traces_by_agent(good.execution_context.agent_id)

    # the good trace is still returned
    assert len(traces) == 1
    assert traces[0].execution_context.execution_id == good.execution_context.execution_id
    # the failure was surfaced, not silently swallowed
    assert any("Skipping unparseable trace file" in r.getMessage() for r in records)


def test_get_trace_returns_none_on_unparseable_file(tmp_path) -> None:
    store = JsonTraceStore(data_dir=tmp_path)
    execution_id = uuid4()
    (tmp_path / "traces" / f"{execution_id}.json").write_text("{ garbage")

    with _capture_json_store_warnings() as records:
        result = store.get_trace(execution_id)

    assert result is None  # not a raised ValidationError
    assert any("failed to parse on read" in r.getMessage() for r in records)


def test_get_trace_roundtrip_verifies_on_disk_integrity(tmp_path) -> None:
    # F6: the on-disk snake_case digest verifies its OWN sealed hash on read.
    store = JsonTraceStore(data_dir=tmp_path)
    digest = _sealed_digest()
    store.save_trace(digest)
    loaded = store.get_trace(digest.execution_context.execution_id)
    assert loaded is not None
    assert loaded.verify_integrity() is True


@pytest.mark.xfail(
    reason="F6 KNOWN GAP (phase-2): the submitted traceHash is sealed over the "
    "pre-compaction snake_case digest and is NOT recomputable from the compacted "
    "camelCase submit payload. Documented, not exploitable today.",
    strict=True,
)
def test_submitted_trace_hash_is_recomputable_from_wire_payload() -> None:
    from src.validator.forecasting.assignment_pipeline import build_prediction_submit_payload  # noqa: F401
    # There is deliberately no server-side recompute path yet; asserting one exists
    # is expected to fail until phase-2 adds a hash over the exact wire payload.
    raise AssertionError("no server-side recompute of traceHash exists in v1")
