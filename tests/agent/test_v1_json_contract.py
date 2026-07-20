"""v1 PR14 contract tests: the Forecast JSON output + the v1.1.0 trace shape.

Deterministic (no network): exercises the shared parser and the assembler so a
good run yields a valid v1.1.0 digest and a bad model output is an in-band
invalid, not a silent 0.5.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.core.events import Event
from src.core.schemas import (
    TRACE_SCHEMA_VERSION,
    AgentResult,
    BeliefStep,
    EventCategory,
    ExtractedEvidence,
    ExecutionContext,
    ProviderCall,
    ProviderTier,
    ReasoningStepType,
    ResponseMeta,
    SandboxEnvironment,
)
from src.agent.examples._v1_common import (
    Forecast,
    _coerce_unit,
    belief_path_single_final,
    belief_path_steps,
    parse_forecast,
)
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


# --- F10: scientific notation ------------------------------------------------

def test_coerce_unit_accepts_scientific_notation():
    # The old regex matched the leading "1" of "1e-5" and returned 1.0; the fix
    # parses the full mantissa+exponent.
    assert _coerce_unit("1e-5") == pytest.approx(1e-5)
    assert _coerce_unit("2.5e-1") == pytest.approx(0.25)
    assert _coerce_unit("1E-3") == pytest.approx(0.001)
    # bare integer still parses to itself (regression guard)
    assert _coerce_unit("1") == pytest.approx(1.0)
    # a string with no parseable number is still rejected
    with pytest.raises(ValueError):
        _coerce_unit("e+")


# --- F9: JSON blob extraction is string/escape-aware (raw_decode) -------------

def test_extract_json_blob_tolerates_braces_inside_strings():
    # A '}' inside a string value must NOT truncate the object (brace-counting bug).
    text = '{"reasoning": "the market closes} soon", "prediction": 0.5, "confidence": 0.6}'
    fc = parse_forecast(text)
    assert fc is not None and fc.prediction == pytest.approx(0.5)


def test_extract_json_blob_handles_nested_braces_and_trailing_prose():
    text = 'Here you go:\n{"reasoning": "note {inner} detail", "prediction": 0.4, "confidence": 0.7}\nThanks!'
    fc = parse_forecast(text)
    assert fc is not None and fc.prediction == pytest.approx(0.4) and fc.confidence == pytest.approx(0.7)


def test_extract_json_blob_fenced_json_with_prose():
    text = 'Sure.\n```json\n{"reasoning": "x", "prediction": 0.3, "confidence": 0.55}\n```\ndone'
    fc = parse_forecast(text)
    assert fc is not None and fc.prediction == pytest.approx(0.3)


# --- the v1.1.0 trace shape --------------------------------------------------

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
        agent_result=AgentResult(prediction=0.62, confidence=0.7, reasoning="because",
                                 beliefPath=belief_path_single_final(0.62, "because")),
    )
    # version bumped, sealed hash valid
    assert digest.trace_integrity.trace_schema_version == TRACE_SCHEMA_VERSION == "1.1.0"
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


def test_gap_query_references_provider_call_without_copying_evidence():
    evidence_body = "full provider response that must have one canonical home"
    provider_call = _provider_call().model_copy(
        update={
            "extracted_evidence": [
                ExtractedEvidence(
                    evidence_type="quote_summary",
                    content=evidence_body,
                    extraction_method="nlp_extraction",
                )
            ]
        }
    )
    digest = assemble_trace(
        execution_context=_exec_ctx(),
        event=_event(),
        provider_calls=[provider_call],
        agent_result=AgentResult(
            prediction=0.62,
            confidence=0.7,
            reasoning="because",
            beliefPath=belief_path_single_final(0.62, "because"),
        ),
    )

    gap_step = digest.reasoning_chain[0]
    assert gap_step.reasoning_text == (
        "[openrouter] chat_completion; evidence in providerCalls[0]"
    )
    assert evidence_body == digest.provider_calls[0].extracted_evidence[0].content
    assert evidence_body not in gap_step.reasoning_text
    assert digest.reasoning_chain[-1].reasoning_text == "because"


def test_missing_confidence_is_valid_neutral():
    # Confidence is optional (stored, unscored). An agent that fails closed to a
    # neutral forecast returns confidence=None, and that must NOT invalidate the
    # well-formed prediction or count toward the invalid gate.
    digest = assemble_trace(
        execution_context=_exec_ctx(), event=_event(), provider_calls=[_provider_call()],
        agent_result=AgentResult(prediction=0.5, confidence=None, reasoning="fail-closed neutral",
                                 beliefPath=belief_path_single_final(0.5, "fail-closed neutral")),
    )
    _prob, _conf, is_valid, reasons, _raw = normalize_prediction_values(digest)
    assert is_valid is True
    assert "confidence_missing" not in reasons


# --- the belief path -------------------------------------------------------

def _final(prob: float, text: str = "t") -> BeliefStep:
    return BeliefStep(step=0, type="final", probability=prob, text=text)


def test_belief_path_drives_multipoint_reasoning_chain():
    # A real multi-step trajectory: prior -> update -> final.
    bp = belief_path_steps([
        (0.50, "base rate, no evidence yet"),
        (0.40, "weather search: wet race favors NO"),
        (0.62, "qualifying: flips it"),
    ])
    digest = assemble_trace(
        execution_context=_exec_ctx(), event=_event(),
        provider_calls=[_provider_call()],
        agent_result=AgentResult(prediction=0.62, confidence=0.7, reasoning="r", beliefPath=bp),
    )
    chain = digest.reasoning_chain
    # provider gap_query step(s) FIRST (provider linkage preserved), THEN belief steps
    assert chain[0].step_type is ReasoningStepType.GAP_QUERY
    assert chain[0].provider_call_index == 0
    belief_steps = [s for s in chain if s.provider_call_index is None]
    assert len(belief_steps) == 3
    # vocab mapped: prior -> PRIOR; update/final -> BELIEF_UPDATE; terminal is last
    assert belief_steps[0].step_type is ReasoningStepType.PRIOR
    assert all(s.step_type is ReasoningStepType.BELIEF_UPDATE for s in belief_steps[1:])
    assert chain[-1].step_type is ReasoningStepType.BELIEF_UPDATE
    # MULTI-POINT: every belief step carries a probability, >1 DISTINCT value
    probs = [s.intermediate_probability for s in belief_steps]
    assert all(p is not None for p in probs)
    assert len({round(p, 6) for p in probs}) > 1
    # the terminal probability equals the prediction
    assert chain[-1].intermediate_probability == 0.62
    # no positional-fake evidence join yet
    assert all(s.input_evidence_refs == [] for s in chain)
    assert digest.verify_integrity()


def test_future_graph_carries_raw_belief_path():
    bp = [
        BeliefStep(step=0, type="prior", probability=0.5, text="prior"),
        BeliefStep(step=1, type="update", probability=0.4, text="searched",
                   usedCall="c1", usedSources=[0, 2]),
        BeliefStep(step=2, type="final", probability=0.62, text="final"),
    ]
    digest = assemble_trace(
        execution_context=_exec_ctx(), event=_event(),
        provider_calls=[_provider_call()],
        agent_result=AgentResult(prediction=0.62, reasoning="r", beliefPath=bp),
    )
    raw = digest.future_graph["beliefPath"]
    # the raw labels + usedCall/usedSources (which the reasoning_chain vocab cannot
    # carry) survive intact to Mongo for the phase-2 grounding hook
    assert [s["type"] for s in raw] == ["prior", "update", "final"]
    assert raw[1]["usedCall"] == "c1"
    assert raw[1]["usedSources"] == [0, 2]
    assert raw[1]["text"] == "searched"


def test_single_final_belief_path_is_valid():
    # The minimal valid path — one final step — keeps a simple agent compliant.
    ar = AgentResult(prediction=0.3, reasoning="one shot",
                     beliefPath=belief_path_single_final(0.3, "one shot"))
    assert [s.type for s in ar.beliefPath] == ["final"]
    digest = assemble_trace(
        execution_context=_exec_ctx(), event=_event(),
        provider_calls=[_provider_call()], agent_result=ar,
    )
    assert digest.verify_integrity()


@pytest.mark.parametrize("kwargs", [
    # two final steps
    dict(prediction=0.5, reasoning="r", beliefPath=[
        BeliefStep(step=0, type="final", probability=0.5, text="a"),
        BeliefStep(step=1, type="final", probability=0.5, text="b")]),
    # final is not the last step
    dict(prediction=0.5, reasoning="r", beliefPath=[
        BeliefStep(step=0, type="final", probability=0.5, text="a"),
        BeliefStep(step=1, type="update", probability=0.5, text="b")]),
    # final probability != prediction
    dict(prediction=0.5, reasoning="r", beliefPath=[_final(0.99)]),
    # empty belief path (min_length=1)
    dict(prediction=0.5, reasoning="r", beliefPath=[]),
    # empty reasoning (min_length=1)
    dict(prediction=0.5, reasoning="", beliefPath=[_final(0.5)]),
])
def test_malformed_agent_result_raises(kwargs):
    with pytest.raises(ValidationError):
        AgentResult(**kwargs)


# --- F11: belief-final vs prediction tolerance compare -----------------------

def test_belief_final_within_tolerance_of_prediction_validates():
    # ~1e-12 apart validates (was exact/4dp equality before).
    ar = AgentResult(
        prediction=0.5, reasoning="r",
        beliefPath=[BeliefStep(step=0, type="final", probability=0.5 + 1e-12, text="t")],
    )
    assert ar.prediction == 0.5


def test_belief_final_across_rounding_boundary_validates():
    # 0.12344 vs 0.12346 differ by 2e-5 (< 5e-5 tol) but round to different 4dp
    # values, so the old round(_,4) equality REJECTED this honest agent; the
    # tolerance compare accepts it.
    assert round(0.12344, 4) != round(0.12346, 4)  # the old check would have failed
    ar = AgentResult(
        prediction=0.12346, reasoning="r",
        beliefPath=[BeliefStep(step=0, type="final", probability=0.12344, text="t")],
    )
    assert ar.prediction == pytest.approx(0.12346)


def test_belief_final_exact_equal_still_validates():
    ar = AgentResult(
        prediction=0.62, reasoning="r",
        beliefPath=[BeliefStep(step=0, type="final", probability=0.62, text="t")],
    )
    assert ar.prediction == 0.62


def test_belief_final_genuine_disagreement_still_raises():
    with pytest.raises(ValidationError):
        AgentResult(
            prediction=0.7, reasoning="r",
            beliefPath=[BeliefStep(step=0, type="final", probability=0.6, text="t")],
        )


# --- F4: size caps at the docker boundary ------------------------------------

def test_agent_result_rejects_oversize_reasoning():
    with pytest.raises(ValidationError):
        AgentResult(
            prediction=0.5, reasoning="x" * 8001,
            beliefPath=[_final(0.5)],
        )


def test_belief_step_rejects_oversize_text():
    with pytest.raises(ValidationError):
        BeliefStep(step=0, type="final", probability=0.5, text="x" * 4001)


def test_belief_path_rejects_too_many_items():
    # 33 well-formed steps (prior, 31 updates, final) — only the >32 length is invalid.
    steps = [BeliefStep(step=0, type="prior", probability=0.5, text="p")]
    steps += [BeliefStep(step=i, type="update", probability=0.5, text="u") for i in range(1, 32)]
    steps.append(BeliefStep(step=32, type="final", probability=0.5, text="f"))
    assert len(steps) == 33
    with pytest.raises(ValidationError):
        AgentResult(prediction=0.5, reasoning="r", beliefPath=steps)


def test_extra_key_and_inf_prediction_raise():
    # extra='forbid' rejects unknown top-level keys
    with pytest.raises(ValidationError):
        AgentResult(prediction=0.5, reasoning="r", beliefPath=[_final(0.5)], junk=1)
    # allow_inf_nan=False rejects a non-finite prediction
    with pytest.raises(ValidationError):
        AgentResult(prediction=float("inf"), reasoning="r", beliefPath=[_final(1.0)])
