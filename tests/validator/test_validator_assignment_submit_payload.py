from __future__ import annotations

import hashlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.core.config import AppConfig
from src.validator.assignment_pipeline import (
    build_prediction_submit_payload,
    build_sandbox_assignment_agent,
    process_single_assignment,
    resolve_binary_outcome_ids,
)
from src.validator.orchestrator_api import OrchestratorAssignment


def _assignment(outcomes: list[dict]) -> OrchestratorAssignment:
    code = "class Agent:\n    pass\n"
    sha = hashlib.sha256(code.encode("utf-8")).hexdigest()
    return OrchestratorAssignment.model_validate(
        {
            "agentPredictionId": "pred_123",
            "status": "in_progress",
            "leaseExpiresAt": "2026-06-05T19:59:44.271000Z",
            "miner": {"minerHotkey": "5EqZoEKc6c8TaG4xRRHTT1uZiQF5jkjQCeUV5t77L6YbeaJ8", "minerUid": 17},
            "event": {
                "marketId": "cmpxszzto007luaaznuulufaq",
                "source": "polymarket",
                "sourceMarketId": "2411919",
                "question": "Will X happen?",
                "description": "Binary event",
                "endDate": "2026-06-06T16:00:00Z",
                "outcomes": outcomes,
                "currentOutcomePrices": {"yes_id": 0.26, "no_id": 0.74},
            },
            "agent": {
                "agentUploadId": "cmpwwsd4800000razamvsplvc",
                "sha256": sha,
                "uploadedAt": "2026-06-02T17:25:57.224000Z",
                "code": code,
            },
        }
    )


class _Digest:
    def __init__(self, probability: float, confidence: float | None = None) -> None:
        self._execution_id = str(uuid4())
        self._agent_id = str(uuid4())
        self._validator_id = str(uuid4())
        self.prediction_output = SimpleNamespace(
            final_probability=probability,
            confidence=confidence,
            reasoning_summary="summary",
        )
        self.execution_context = SimpleNamespace(
            execution_id=self._execution_id,
            agent_id=self._agent_id,
            agent_version="0.1.0",
            validator_id=self._validator_id,
            execution_duration_ms=1234,
        )
        self.provider_calls = [
            SimpleNamespace(provider_id="polymarket", model=None, call_index=0),
            SimpleNamespace(provider_id="anthropic", model="claude-sonnet-4-6", call_index=1),
        ]

    def model_dump(self, mode: str = "json"):
        assert mode == "json"
        try:
            final_probability_text = f"{float(self.prediction_output.final_probability):.4f}"
        except (TypeError, ValueError):
            final_probability_text = str(self.prediction_output.final_probability)

        return {
            "execution_context": {
                "execution_id": self._execution_id,
                "agent_id": self._agent_id,
                "agent_version": "0.1.0",
                "event_id": str(uuid4()),
                "validator_id": self._validator_id,
                "timestamp_start": "2026-06-11T19:43:53.187514Z",
                "timestamp_end": "2026-06-11T19:44:01.027491Z",
                "execution_duration_ms": 1234,
                "sandbox_environment": "docker_runc",
            },
            "provider_calls": [
                {
                    "call_index": 0,
                    "provider_id": "polymarket",
                    "model": None,
                    "provider_tier": "search",
                    "call_type": "get_market",
                    "query_params_summary": "get_market(id=2411919)",
                    "response_meta": {
                        "response_size_bytes": 123,
                        "record_count": None,
                        "data_freshness": "2026-06-11T19:43:59.000000Z",
                        "data_range_start": None,
                        "data_range_end": None,
                    },
                    "usage_meta": None,
                    "provider_usage_raw": None,
                    "extracted_evidence": [],
                    "raw_response_hash": "aa",
                    "latency_ms": 15,
                    "cost_units": 0.0,
                },
                {
                    "call_index": 1,
                    "provider_id": "anthropic",
                    "model": "claude-sonnet-4-6",
                    "provider_tier": "inference",
                    "call_type": "messages",
                    "query_params_summary": "messages(model=claude-sonnet-4-6)",
                    "response_meta": {
                        "response_size_bytes": 2482,
                        "record_count": None,
                        "data_freshness": "2026-06-11T19:44:00.901878Z",
                        "data_range_start": None,
                        "data_range_end": None,
                    },
                    "usage_meta": {
                        "prompt_tokens": 316,
                        "completion_tokens": 453,
                        "total_tokens": 769,
                        "input_tokens": None,
                        "output_tokens": None,
                        "cached_tokens": 0,
                        "reasoning_tokens": None,
                        "cost": 0.002581,
                        "is_byok": None,
                    },
                    "provider_usage_raw": {
                        "promptTokens": 316,
                        "completionTokens": 453,
                        "totalTokens": 769,
                    },
                    "extracted_evidence": [
                        {
                            "evidence_type": "probability",
                            "content": "LLM prediction: 0.8",
                            "extraction_method": "nlp_extraction",
                            "numeric_value": 0.8,
                        }
                    ],
                    "raw_response_hash": "bb",
                    "latency_ms": 5657,
                    "cost_units": 0.002581,
                },
            ],
            "reasoning_chain": [
                {
                    "step_index": 0,
                    "step_type": "evidence_gathering",
                    "provider_call_index": 1,
                    "provider_id": "anthropic",
                    "input_evidence_refs": [1],
                    "reasoning_text": "Gathered evidence from model output",
                    "agent_interpretation": None,
                    "intermediate_probability": None,
                    "confidence_delta": None,
                    "conflict_signals": None,
                    "inference_model_used": "claude-sonnet-4-6",
                },
                {
                    "step_index": 1,
                    "step_type": "synthesis",
                    "provider_call_index": 1,
                    "provider_id": "anthropic",
                    "input_evidence_refs": [1],
                    "reasoning_text": "Synthesized evidence into final estimate",
                    "agent_interpretation": None,
                    "intermediate_probability": self.prediction_output.final_probability,
                    "confidence_delta": None,
                    "conflict_signals": None,
                    "inference_model_used": "claude-sonnet-4-6",
                },
                {
                    "step_index": 2,
                    "step_type": "final_assignment",
                    "provider_call_index": 1,
                    "provider_id": "anthropic",
                    "input_evidence_refs": [],
                    "reasoning_text": f"Final probability: {final_probability_text}",
                    "agent_interpretation": None,
                    "intermediate_probability": self.prediction_output.final_probability,
                    "confidence_delta": None,
                    "conflict_signals": None,
                    "inference_model_used": "claude-sonnet-4-6",
                },
            ],
            "prediction_output": {
                "final_probability": self.prediction_output.final_probability,
                "confidence": self.prediction_output.confidence,
                "confidence_interval": [0.1, 0.9],
                "reasoning_summary": self.prediction_output.reasoning_summary,
                "key_drivers": ["LLM prediction: 0.8"],
                "key_uncertainties": ["Insufficient historical comparables"],
                "contrarian_flag": False,
                "ensemble_at_prediction": None,
                "metadata": None,
            },
            "trace_integrity": {
                "trace_hash": "trace_hash_123",
                "merkle_root": None,
                "anchor_tx": None,
                "anchor_timestamp": None,
                "trace_schema_version": "0.1.0",
                "total_provider_cost": 0.002581,
                "total_evidence_items": 1,
                "usage_totals": {
                    "provider_calls_with_usage": 1,
                    "prompt_tokens": 316,
                    "completion_tokens": 453,
                    "total_tokens": 769,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cached_tokens": 0,
                    "reasoning_tokens": 0,
                },
            },
        }


class _SparseDigest(_Digest):
    def __init__(self, probability: float, confidence: float | None = None) -> None:
        super().__init__(probability, confidence)
        self.provider_calls = []

    def model_dump(self, mode: str = "json"):
        payload = super().model_dump(mode=mode)
        payload["provider_calls"] = []
        payload["reasoning_chain"] = []
        payload["trace_integrity"]["usage_totals"] = None
        return payload


def _assert_no_snake_case_keys(payload: object) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            assert "_" not in key
            _assert_no_snake_case_keys(value)
    elif isinstance(payload, list):
        for item in payload:
            _assert_no_snake_case_keys(item)


def test_resolve_binary_outcome_ids_prefers_yes_no_names() -> None:
    assignment = _assignment(
        outcomes=[
            {"outcomeId": "oid_yes", "name": "Yes"},
            {"outcomeId": "oid_no", "name": "No"},
        ]
    )
    yes_id, no_id = resolve_binary_outcome_ids(assignment)
    assert yes_id == "oid_yes"
    assert no_id == "oid_no"


def test_build_prediction_submit_payload_binary_fields() -> None:
    assignment = _assignment(
        outcomes=[
            {"outcomeId": "oid_yes", "name": "Yes"},
            {"outcomeId": "oid_no", "name": "No"},
        ]
    )
    payload = build_prediction_submit_payload(assignment, _Digest(0.8, confidence=0.9))
    prediction = payload["prediction"]
    assert prediction["predictedOutcomeId"] == "oid_yes"
    assert prediction["outcomeProbabilities"]["oid_yes"] == 0.8
    assert prediction["outcomeProbabilities"]["oid_no"] == 0.2
    assert prediction["confidence"] == pytest.approx(0.9)
    assert prediction["executionMetadata"]["model"] == "claude-sonnet-4-6"
    assert prediction["executionMetadata"]["latencyMs"] == 1234
    assert prediction["executionMetadata"]["predictionIsInvalid"] is False
    assert prediction["executionMetadata"]["predictionInvalidReason"] is None
    assert prediction["executionMetadata"]["predictionValidation"]["isValid"] is True
    assert prediction["executionMetadata"]["predictionValidation"]["reasons"] == []
    assert payload["reasoningTrace"]["modelMetadata"]["provider"] == "anthropic"
    assert payload["reasoningTrace"]["modelMetadata"]["model"] == "claude-sonnet-4-6"
    assert payload["reasoningTrace"]["schemaVersion"] == "1.0"
    assert payload["reasoningTrace"]["traceSummary"]["strategy"] == "validator-assignment-pipeline"
    trace = payload["reasoningTrace"]["trace"]
    assert isinstance(trace["providerCalls"], list)
    assert isinstance(trace["steps"], list)
    assert isinstance(trace["evidenceDigest"], dict)

    steps = trace["steps"]
    assert all(step["origin"] == "reasoningChain" for step in steps)
    assert all(step["stage"] is None for step in steps)
    assert [step["stepIndex"] for step in steps] == list(range(len(steps)))

    provider_call_indexes = {call["callIndex"] for call in trace["providerCalls"]}
    for step in steps:
        provider_call_index = step["providerCallIndex"]
        if provider_call_index is not None:
            assert provider_call_index in provider_call_indexes
        for evidence_ref in step["inputEvidenceRefs"]:
            assert evidence_ref in provider_call_indexes

    trace_summary = payload["reasoningTrace"]["traceSummary"]
    assert trace_summary["executionId"] == trace["evidenceDigest"]["executionContext"]["executionId"]
    assert trace_summary["traceHash"] == "trace_hash_123"
    assert trace_summary["providerCallCount"] == len(trace["providerCalls"])
    assert trace_summary["reasoningStepCount"] == len(trace["steps"])
    assert trace_summary["usageTotals"]["totalTokens"] == 769

    _assert_no_snake_case_keys(payload["reasoningTrace"])


def test_build_prediction_submit_payload_handles_sparse_reasoning_chain() -> None:
    assignment = _assignment(
        outcomes=[
            {"outcomeId": "oid_yes", "name": "Yes"},
            {"outcomeId": "oid_no", "name": "No"},
        ]
    )
    payload = build_prediction_submit_payload(assignment, _SparseDigest(0.4, confidence=0.6))
    trace = payload["reasoningTrace"]["trace"]
    assert trace["providerCalls"] == []
    assert trace["steps"] == []
    assert payload["reasoningTrace"]["traceSummary"]["providerCallCount"] == 0


def test_build_sandbox_assignment_agent_sets_inline_code_attrs() -> None:
    assignment = _assignment(
        outcomes=[
            {"outcomeId": "oid_yes", "name": "Yes"},
            {"outcomeId": "oid_no", "name": "No"},
        ]
    )
    agent = build_sandbox_assignment_agent(assignment)
    assert getattr(agent, "_arcratio_agent_source_code") == assignment.agent.code
    assert str(agent.agent_id)
    assert agent.agent_version


def test_build_sandbox_assignment_agent_rejects_hash_mismatch() -> None:
    assignment = _assignment(
        outcomes=[
            {"outcomeId": "oid_yes", "name": "Yes"},
            {"outcomeId": "oid_no", "name": "No"},
        ]
    )
    assignment.agent.sha256 = "00" * 32
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        build_sandbox_assignment_agent(assignment)


def test_process_assignment_refuses_in_process_mode() -> None:
    cfg = AppConfig()
    cfg.validator.sandbox_type = "in_process"
    assignment = _assignment(
        outcomes=[
            {"outcomeId": "oid_yes", "name": "Yes"},
            {"outcomeId": "oid_no", "name": "No"},
        ]
    )
    fake_orchestrator = SimpleNamespace(
        run_agent=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("should not execute")
        )
    )
    with pytest.raises(RuntimeError, match="sandbox is not docker-based"):
        process_single_assignment(
            assignment=assignment,
            config=cfg,
            orchestrator=fake_orchestrator,  # type: ignore[arg-type]
            loaded_hotkey=None,
            submit_prediction=False,
        )


def test_build_prediction_submit_payload_marks_invalid_non_numeric_prediction() -> None:
    assignment = _assignment(
        outcomes=[
            {"outcomeId": "oid_yes", "name": "Yes"},
            {"outcomeId": "oid_no", "name": "No"},
        ]
    )
    payload = build_prediction_submit_payload(assignment, _Digest("not-a-number", confidence=0.8))
    prediction = payload["prediction"]
    assert prediction["predictedOutcomeId"] in {"oid_yes", "oid_no"}
    assert prediction["outcomeProbabilities"]["oid_yes"] == 0.0
    assert prediction["outcomeProbabilities"]["oid_no"] == 1.0
    assert prediction["executionMetadata"]["predictionIsInvalid"] is True
    assert "prediction_non_numeric" in (
        prediction["executionMetadata"]["predictionInvalidReason"] or ""
    )
    assert "prediction_non_numeric" in prediction["executionMetadata"]["predictionValidation"]["reasons"]


def test_build_prediction_submit_payload_marks_invalid_confidence_out_of_range() -> None:
    assignment = _assignment(
        outcomes=[
            {"outcomeId": "oid_yes", "name": "Yes"},
            {"outcomeId": "oid_no", "name": "No"},
        ]
    )
    payload = build_prediction_submit_payload(assignment, _Digest(0.8, confidence=5.0))
    prediction = payload["prediction"]
    assert prediction["confidence"] == pytest.approx(0.0)
    assert prediction["executionMetadata"]["predictionIsInvalid"] is True
    assert "confidence_out_of_range" in (
        prediction["executionMetadata"]["predictionInvalidReason"] or ""
    )


def test_build_prediction_submit_payload_marks_invalid_when_confidence_missing() -> None:
    assignment = _assignment(
        outcomes=[
            {"outcomeId": "oid_yes", "name": "Yes"},
            {"outcomeId": "oid_no", "name": "No"},
        ]
    )
    payload = build_prediction_submit_payload(assignment, _Digest(0.8, confidence=None))
    prediction = payload["prediction"]
    assert prediction["confidence"] == pytest.approx(0.0)
    assert prediction["executionMetadata"]["predictionIsInvalid"] is True
    assert "confidence_missing" in (
        prediction["executionMetadata"]["predictionInvalidReason"] or ""
    )
