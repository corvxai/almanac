"""v1 PR14 contract tests: the Forecast JSON output + the v1.0.0 trace shape.

Deterministic (no network): exercises the shared parser and the assembler so a
good run yields a valid v1.0.0 digest and a bad model output is an in-band
invalid, not a silent 0.5.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from src.core.events import Event
from src.core.schemas import (
    TRACE_SCHEMA_VERSION,
    AgentResult,
    EventCategory,
    ExecutionContext,
    ProviderCall,
    ProviderTier,
    ReasoningStepType,
    ResponseMeta,
    SandboxEnvironment,
)
from src.agent.examples._v1_common import Forecast, parse_forecast, _coerce_unit
from src.validator.assignment_pipeline import normalize_prediction_values
from src.validator.trace_assembler import assemble_trace


# --- the provider-LLM output contract ---------------------------------------

def test_parse_forecast_fixes_the_regex_bugs():
    # "62%" -> 0.62 (the old regex clamped it to 1.0)
    fc = parse_forecast('{"reasoning":"x","prediction":"62%","confidence":"80%"}')
    assert fc is not None and fc.prediction == pytest.approx(0.62) and fc.confidence == pytest.approx(0.80)
    # markdown fences and surrounding prose are tolerated
    assert parse_forecast('```json\n{"reasoning":"x","prediction":0.4,"confidence":0.5}\n```').prediction == 0.4
    assert parse_forecast('Sure!\n{"reasoning":"x","prediction":0.4,"confidence":0.5}\ndone').prediction == 0.4


@pytest.mark.parametrize("bad", ["", "I'm sorry, I can't help",
                                 '{"reasoning":"x","prediction":1.7,"confidence":0.5}',
                                 '{"reasoning":"x","prediction":0.5}'])  # confidence missing
def test_parse_forecast_rejects_never_defaults(bad):
    assert parse_forecast(bad) is None  # rejected, NOT defaulted to 0.5


def test_forecast_requires_confidence_in_range():
    with pytest.raises(Exception):
        Forecast(reasoning="x", prediction=0.5)  # confidence required
    with pytest.raises(Exception):
        Forecast(reasoning="x", prediction=1.7, confidence=0.5)  # out of range


def test_coerce_unit_percent_and_reject():
    assert _coerce_unit("62%") == pytest.approx(0.62)
    assert _coerce_unit("0.62.") == pytest.approx(0.62)
    with pytest.raises(ValueError):
        _coerce_unit("not a number")


# --- the v1.0.0 trace shape --------------------------------------------------

def _event() -> Event:
    return Event(
        event_id=uuid4(), title="Will X happen?", description="Binary event.",
        category=EventCategory.OTHER, resolution_criteria="YES iff X occurs.",
        resolution_deadline=datetime(2027, 1, 1, tzinfo=timezone.utc),
        created_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
    )


def _exec_ctx() -> ExecutionContext:
    now = datetime(2026, 4, 24, tzinfo=timezone.utc)
    return ExecutionContext(
        execution_id=uuid4(), agent_id=uuid4(), agent_version="1.0.0",
        event_id=uuid4(), validator_id=uuid4(), timestamp_start=now,
        timestamp_end=now, execution_duration_ms=1,
        sandbox_environment=SandboxEnvironment.IN_PROCESS,
    )


def _provider_call() -> ProviderCall:
    return ProviderCall(
        call_index=0, provider_id="openrouter", model="openai/gpt-4o-mini:online",
        provider_tier=ProviderTier.SEARCH, call_type="chat_completion",
        query_params_summary="openrouter.chat_completion(model=...)",
        response_meta=ResponseMeta(response_size_bytes=100,
                                   data_freshness=datetime(2026, 4, 24, tzinfo=timezone.utc)),
        extracted_evidence=[],
        sources_accessed=[{"url": "https://example.com/a", "title": "A"}],  # coerces to Source
        raw_response_hash="h0", latency_ms=10, cost_units=0.0,
    )


def test_assemble_trace_is_valid_v1():
    digest = assemble_trace(
        execution_context=_exec_ctx(), event=_event(),
        provider_calls=[_provider_call()],
        agent_result=AgentResult(prediction=0.62, confidence=0.7, reasoning="because"),
    )
    # version bumped, sealed hash valid
    assert digest.trace_integrity.trace_schema_version == TRACE_SCHEMA_VERSION == "1.0.0"
    assert digest.verify_integrity()
    # confidence carried; step types are the new vocab; sources are typed with a url
    assert digest.prediction_output.confidence == 0.7
    assert all(s.step_type in {ReasoningStepType.PRIOR, ReasoningStepType.BELIEF_UPDATE,
                               ReasoningStepType.GAP_QUERY} for s in digest.reasoning_chain)
    assert digest.provider_calls[0].sources_accessed[0].url == "https://example.com/a"
    # no positional-fake join
    assert all(s.input_evidence_refs == [] for s in digest.reasoning_chain)
    # dropped fields are gone from the serialized trace
    dump = digest.model_dump()
    assert "key_drivers" not in dump["prediction_output"]
    assert "metadata" not in dump["prediction_output"]
    for gone in ("merkle_root", "anchor_tx", "anchor_timestamp", "total_evidence_items"):
        assert gone not in dump["trace_integrity"]


def test_missing_confidence_is_invalid_in_band():
    # An agent that fails closed returns confidence=None -> the validator marks it invalid.
    digest = assemble_trace(
        execution_context=_exec_ctx(), event=_event(), provider_calls=[_provider_call()],
        agent_result=AgentResult(prediction=0.5, confidence=None, reasoning="INVALID: ..."),
    )
    _prob, _conf, is_valid, reasons, _raw = normalize_prediction_values(digest)
    assert is_valid is False
    assert "confidence_missing" in reasons
