"""Trace assembler tests for prediction-level usage aggregates."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from src.core.events import Event
from src.core.schemas import (
    AgentResult,
    EventCategory,
    ExecutionContext,
    ProviderCall,
    ProviderTier,
    ResponseMeta,
    SandboxEnvironment,
    UsageMeta,
)
from src.validator.trace_assembler import assemble_trace


def _event() -> Event:
    return Event(
        event_id=uuid4(),
        title="Will X happen?",
        description="Binary event.",
        category=EventCategory.OTHER,
        resolution_criteria="YES iff X occurs.",
        resolution_deadline=datetime(2027, 1, 1, tzinfo=timezone.utc),
        created_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
    )


def _execution_context() -> ExecutionContext:
    now = datetime(2026, 4, 24, tzinfo=timezone.utc)
    return ExecutionContext(
        execution_id=uuid4(),
        agent_id=uuid4(),
        agent_version="0.1.0",
        event_id=uuid4(),
        validator_id=uuid4(),
        timestamp_start=now,
        timestamp_end=now,
        execution_duration_ms=1,
        sandbox_environment=SandboxEnvironment.IN_PROCESS,
    )


def _provider_call(
    idx: int,
    provider_id: str,
    usage_meta: UsageMeta | None,
    cost_units: float,
) -> ProviderCall:
    return ProviderCall(
        call_index=idx,
        provider_id=provider_id,
        provider_tier=ProviderTier.INFERENCE,
        call_type="chat_completion",
        query_params_summary=f"{provider_id}.chat_completion(model=test)",
        response_meta=ResponseMeta(
            response_size_bytes=123,
            data_freshness=datetime(2026, 4, 24, tzinfo=timezone.utc),
        ),
        usage_meta=usage_meta,
        provider_usage_raw=None,
        extracted_evidence=[],
        raw_response_hash=f"hash-{idx}",
        latency_ms=10,
        cost_units=cost_units,
    )


def test_assemble_trace_aggregates_usage_totals_across_provider_calls() -> None:
    calls = [
        _provider_call(
            idx=0,
            provider_id="openrouter",
            usage_meta=UsageMeta(
                prompt_tokens=100,
                completion_tokens=40,
                total_tokens=140,
                cached_tokens=5,
                reasoning_tokens=2,
                cost=0.0012,
            ),
            cost_units=0.0012,
        ),
        _provider_call(
            idx=1,
            provider_id="gemini",
            usage_meta=UsageMeta(
                prompt_tokens=80,
                completion_tokens=20,
                total_tokens=100,
            ),
            cost_units=0.0,
        ),
        _provider_call(
            idx=2,
            provider_id="web_search",
            usage_meta=None,
            cost_units=0.0,
        ),
    ]

    digest = assemble_trace(
        execution_context=_execution_context(),
        event=_event(),
        provider_calls=calls,
        agent_result=AgentResult(prediction=0.5, reasoning="test"),
    )

    assert digest.trace_integrity.total_provider_cost == 0.0012
    assert digest.trace_integrity.usage_totals is not None
    assert digest.trace_integrity.usage_totals.provider_calls_with_usage == 2
    assert digest.trace_integrity.usage_totals.prompt_tokens == 180
    assert digest.trace_integrity.usage_totals.completion_tokens == 60
    assert digest.trace_integrity.usage_totals.total_tokens == 240
    assert digest.trace_integrity.usage_totals.cached_tokens == 5
    assert digest.trace_integrity.usage_totals.reasoning_tokens == 2


def test_assemble_trace_omits_usage_totals_when_no_provider_usage() -> None:
    calls = [
        _provider_call(
            idx=0,
            provider_id="web_search",
            usage_meta=None,
            cost_units=0.0,
        ),
    ]

    digest = assemble_trace(
        execution_context=_execution_context(),
        event=_event(),
        provider_calls=calls,
        agent_result=AgentResult(prediction=0.5, reasoning="test"),
    )

    assert digest.trace_integrity.usage_totals is None
