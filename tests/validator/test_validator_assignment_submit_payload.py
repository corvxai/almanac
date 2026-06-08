from __future__ import annotations

import hashlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.core.config import AppConfig
from src.validator.orchestrator_api import OrchestratorAssignment
from src.validator import validator as validator_module
from src.validator.validator import Validator


def _bt_objects():
    return validator_module._BtObjects(  # noqa: SLF001 - tests use internal helper
        wallet=SimpleNamespace(hotkey=SimpleNamespace(ss58_address="hotkey_test")),
        subtensor=SimpleNamespace(),
        dendrite=SimpleNamespace(),
        metagraph=SimpleNamespace(),
        network="test",
    )


def _validator() -> Validator:
    cfg = AppConfig()
    return Validator(config=cfg, store=None, bt_objects=_bt_objects(), metadata_manager=None)


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
        self.prediction_output = SimpleNamespace(
            final_probability=probability,
            confidence=confidence,
            reasoning_summary="summary",
        )
        self.execution_context = SimpleNamespace(
            execution_id=uuid4(),
            agent_version="0.1.0",
            execution_duration_ms=1234,
        )
        self.provider_calls = [
            SimpleNamespace(provider_id="polymarket", model=None),
            SimpleNamespace(provider_id="claude", model="claude-sonnet-4-6"),
        ]

    def model_dump(self, mode: str = "json"):
        assert mode == "json"
        return {"prediction": self.prediction_output.final_probability}


def test_resolve_binary_outcome_ids_prefers_yes_no_names() -> None:
    v = _validator()
    assignment = _assignment(
        outcomes=[
            {"outcomeId": "oid_yes", "name": "Yes"},
            {"outcomeId": "oid_no", "name": "No"},
        ]
    )
    yes_id, no_id = v._resolve_binary_outcome_ids(assignment)  # noqa: SLF001
    assert yes_id == "oid_yes"
    assert no_id == "oid_no"


def test_build_prediction_submit_payload_binary_fields() -> None:
    v = _validator()
    assignment = _assignment(
        outcomes=[
            {"outcomeId": "oid_yes", "name": "Yes"},
            {"outcomeId": "oid_no", "name": "No"},
        ]
    )
    payload = v._build_prediction_submit_payload(assignment, _Digest(0.8, confidence=0.9))  # noqa: SLF001
    prediction = payload["prediction"]
    assert prediction["predictedOutcomeId"] == "oid_yes"
    assert prediction["outcomeProbabilities"]["oid_yes"] == 0.8
    assert prediction["outcomeProbabilities"]["oid_no"] == 0.2
    assert prediction["confidence"] == pytest.approx(0.9)
    assert prediction["executionMetadata"]["model"] == "claude-sonnet-4-6"
    assert prediction["executionMetadata"]["provider"] == "claude"
    assert prediction["executionMetadata"]["latencyMs"] == 1234
    assert prediction["executionMetadata"]["predictionIsInvalid"] is False
    assert prediction["executionMetadata"]["predictionInvalidReason"] is None
    assert prediction["executionMetadata"]["predictionValidation"]["isValid"] is True
    assert prediction["executionMetadata"]["predictionValidation"]["reasons"] == []
    assert payload["reasoningTrace"]["modelMetadata"]["provider"] == "claude"
    assert payload["reasoningTrace"]["modelMetadata"]["model"] == "claude-sonnet-4-6"
    assert payload["reasoningTrace"]["schemaVersion"] == "1.0"


def test_build_sandbox_assignment_agent_sets_inline_code_attrs() -> None:
    v = _validator()
    assignment = _assignment(
        outcomes=[
            {"outcomeId": "oid_yes", "name": "Yes"},
            {"outcomeId": "oid_no", "name": "No"},
        ]
    )
    agent = v._build_sandbox_assignment_agent(assignment)  # noqa: SLF001
    assert getattr(agent, "_arcratio_agent_source_code") == assignment.agent.code
    assert str(agent.agent_id)
    assert agent.agent_version


def test_build_sandbox_assignment_agent_rejects_hash_mismatch() -> None:
    v = _validator()
    assignment = _assignment(
        outcomes=[
            {"outcomeId": "oid_yes", "name": "Yes"},
            {"outcomeId": "oid_no", "name": "No"},
        ]
    )
    assignment.agent.sha256 = "00" * 32
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        v._build_sandbox_assignment_agent(assignment)  # noqa: SLF001


def test_handle_assignment_refuses_in_process_mode(monkeypatch) -> None:
    v = _validator()
    v._config.validator.sandbox_type = "in_process"  # noqa: SLF001
    assignment = _assignment(
        outcomes=[
            {"outcomeId": "oid_yes", "name": "Yes"},
            {"outcomeId": "oid_no", "name": "No"},
        ]
    )

    monkeypatch.setattr(
        v,
        "_get_assignment_orchestrator",
        lambda: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    v._handle_orchestrator_assignment(assignment)  # noqa: SLF001


def test_build_prediction_submit_payload_marks_invalid_non_numeric_prediction() -> None:
    v = _validator()
    assignment = _assignment(
        outcomes=[
            {"outcomeId": "oid_yes", "name": "Yes"},
            {"outcomeId": "oid_no", "name": "No"},
        ]
    )
    payload = v._build_prediction_submit_payload(assignment, _Digest("not-a-number", confidence=0.8))  # noqa: SLF001
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
    v = _validator()
    assignment = _assignment(
        outcomes=[
            {"outcomeId": "oid_yes", "name": "Yes"},
            {"outcomeId": "oid_no", "name": "No"},
        ]
    )
    payload = v._build_prediction_submit_payload(assignment, _Digest(0.8, confidence=5.0))  # noqa: SLF001
    prediction = payload["prediction"]
    assert prediction["confidence"] == pytest.approx(0.0)
    assert prediction["executionMetadata"]["predictionIsInvalid"] is True
    assert "confidence_out_of_range" in (
        prediction["executionMetadata"]["predictionInvalidReason"] or ""
    )


def test_build_prediction_submit_payload_marks_invalid_when_confidence_missing() -> None:
    v = _validator()
    assignment = _assignment(
        outcomes=[
            {"outcomeId": "oid_yes", "name": "Yes"},
            {"outcomeId": "oid_no", "name": "No"},
        ]
    )
    payload = v._build_prediction_submit_payload(assignment, _Digest(0.8, confidence=None))  # noqa: SLF001
    prediction = payload["prediction"]
    assert prediction["confidence"] == pytest.approx(0.0)
    assert prediction["executionMetadata"]["predictionIsInvalid"] is True
    assert "confidence_missing" in (
        prediction["executionMetadata"]["predictionInvalidReason"] or ""
    )
