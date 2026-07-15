"""Shared orchestrator assignment processing pipeline."""

from __future__ import annotations

import datetime
import hashlib
import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, cast
from uuid import NAMESPACE_URL, uuid5

from src.agent.base import BaseAgent
from src.core.config import AppConfig
from src.core.events import Event
from src.core.schemas import EvidenceDigest
from src.gateway.signing import LoadedKeypair
from src.validator.orchestrator import Orchestrator
from src.validator.orchestrator_api import (
    OrchestratorAssignment,
    SubmitPredictionResponse,
    submit_validator_prediction,
)
from src.validator.orchestrator_event_mapper import assignment_to_event

MAX_ASSIGNMENT_AGENT_CODE_BYTES = 2 * 1024 * 1024  # 2MB
INVALID_PREDICTION_SENTINEL = 0.0
INVALID_CONFIDENCE_SENTINEL = 0.0


def _to_camel_case(value: str) -> str:
    parts = value.split("_")
    if len(parts) == 1:
        return value
    return parts[0] + "".join(part[:1].upper() + part[1:] for part in parts[1:])


def _camelize(value: object) -> object:
    if isinstance(value, dict):
        return {_to_camel_case(str(k)): _camelize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_camelize(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    return value


def _reasoning_step_payload(raw_step: dict[str, object]) -> dict[str, object]:
    return {
        "stepType": raw_step.get("step_type"),
        "providerCallIndex": raw_step.get("provider_call_index"),
        "providerId": raw_step.get("provider_id"),
        "inputEvidenceRefs": raw_step.get("input_evidence_refs") or [],
        "reasoningText": raw_step.get("reasoning_text"),
        "intermediateProbability": raw_step.get("intermediate_probability"),
        "confidenceDelta": raw_step.get("confidence_delta"),
        "inferenceModelUsed": raw_step.get("inference_model_used"),
        "stage": None,
        "origin": "reasoningChain",
    }


def _build_reasoning_steps(raw_reasoning_chain: list[dict[str, object]]) -> list[dict[str, object]]:
    steps = [_reasoning_step_payload(step) for step in raw_reasoning_chain]
    for idx, step in enumerate(steps):
        step["stepIndex"] = idx
    return steps


def _build_provider_calls(raw_provider_calls: list[dict[str, object]]) -> list[dict[str, object]]:
    return [_camelize(call) for call in raw_provider_calls]


_REASONING_SUMMARY_REF = (
    "#/reasoningTrace/trace/evidenceDigest/predictionOutput/reasoningSummary"
)


def _compact_evidence_digest(
    digest_dump: dict[str, object],
    reasoning_summary: object,
) -> dict[str, object]:
    compact = {
        key: value
        for key, value in digest_dump.items()
        if key not in {"provider_calls", "reasoning_chain"}
    }
    future_graph = compact.get("future_graph")
    if isinstance(reasoning_summary, str) and isinstance(future_graph, dict):
        belief_path = future_graph.get("beliefPath")
        if isinstance(belief_path, list):
            for belief_step in belief_path:
                if (
                    isinstance(belief_step, dict)
                    and belief_step.get("type") == "final"
                    and belief_step.get("text") == reasoning_summary
                ):
                    belief_step.pop("text")
                    belief_step["text_ref"] = _REASONING_SUMMARY_REF
    return cast(dict[str, object], _camelize(compact))


class _InlineCodeSandboxAgent(BaseAgent):
    """Host-side placeholder; real untrusted code runs in sandbox only."""

    def predict(self, ctx):  # pragma: no cover
        raise RuntimeError("Inline orchestrator agent code must run in sandbox")


@dataclass
class AssignmentProcessingResult:
    event: Event
    digest: EvidenceDigest
    submit_payload: dict
    submit_ack: SubmitPredictionResponse | None


def build_sandbox_assignment_agent(assignment: OrchestratorAssignment) -> BaseAgent:
    code = assignment.agent.code or ""
    code_bytes = code.encode("utf-8")
    if not code_bytes:
        raise RuntimeError("assignment agent.code is empty")
    if len(code_bytes) > MAX_ASSIGNMENT_AGENT_CODE_BYTES:
        raise RuntimeError(f"assignment code exceeds {MAX_ASSIGNMENT_AGENT_CODE_BYTES} bytes")

    computed_sha256 = hashlib.sha256(code_bytes).hexdigest().lower()
    expected_sha256 = (assignment.agent.sha256 or "").strip().lower()
    if expected_sha256 and computed_sha256 != expected_sha256:
        raise RuntimeError(
            "assignment agent code sha256 mismatch: "
            f"expected={expected_sha256} computed={computed_sha256}"
        )

    agent = _InlineCodeSandboxAgent()
    agent.agent_id = uuid5(NAMESPACE_URL, f"agent-upload:{assignment.agent.agentUploadId}")
    agent.agent_version = assignment.agent.uploadedAt.isoformat()
    setattr(agent, "_arcratio_agent_source_code", code)
    return agent


def resolve_binary_outcome_ids(assignment: OrchestratorAssignment) -> tuple[str, str]:
    yes_id: str | None = None
    no_id: str | None = None
    for outcome in assignment.event.outcomes:
        if not isinstance(outcome, dict):
            continue
        outcome_id = outcome.get("outcomeId")
        name = str(outcome.get("name", "")).strip().lower()
        if not isinstance(outcome_id, str):
            continue
        if name == "yes":
            yes_id = outcome_id
        elif name == "no":
            no_id = outcome_id

    if yes_id and no_id:
        return yes_id, no_id

    if len(assignment.event.outcomes) < 2:
        raise RuntimeError("assignment has fewer than two outcomes")
    first = assignment.event.outcomes[0] if isinstance(assignment.event.outcomes[0], dict) else {}
    second = assignment.event.outcomes[1] if isinstance(assignment.event.outcomes[1], dict) else {}
    first_id = first.get("outcomeId")
    second_id = second.get("outcomeId")
    if not isinstance(first_id, str) or not isinstance(second_id, str):
        raise RuntimeError("assignment outcomes missing outcomeId")
    return first_id, second_id


def trace_primary_model_info(digest: EvidenceDigest) -> tuple[str | None, str | None]:
    for call in reversed(digest.provider_calls):
        if call.model:
            return call.provider_id, call.model
    if digest.provider_calls:
        return digest.provider_calls[-1].provider_id, None
    return None, None


def _coerce_unit_interval(
    value: object,
    field_name: str,
    reasons: list[str],
    *,
    fallback: float,
) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        reasons.append(f"{field_name}_non_numeric")
        return fallback
    if not math.isfinite(number):
        reasons.append(f"{field_name}_non_finite")
        return fallback
    if number < 0.0 or number > 1.0:
        reasons.append(f"{field_name}_out_of_range")
        return fallback
    return number


def normalize_prediction_values(
    digest: EvidenceDigest,
) -> tuple[float, float, bool, list[str], dict[str, str]]:
    reasons: list[str] = []
    raw_observed = {
        "prediction": repr(digest.prediction_output.final_probability),
        "confidence": repr(digest.prediction_output.confidence),
    }

    probability = _coerce_unit_interval(
        digest.prediction_output.final_probability,
        "prediction",
        reasons,
        fallback=INVALID_PREDICTION_SENTINEL,
    )

    confidence_raw = digest.prediction_output.confidence
    if confidence_raw is None:
        # Confidence is optional (stored, unscored). A missing value must never
        # invalidate a well-formed prediction or count toward the invalid gate, so
        # do not append an invalid reason here; only fill the numeric slot.
        confidence = INVALID_CONFIDENCE_SENTINEL
    else:
        confidence = _coerce_unit_interval(
            confidence_raw,
            "confidence",
            reasons,
            fallback=INVALID_CONFIDENCE_SENTINEL,
        )

    return probability, confidence, len(reasons) == 0, reasons, raw_observed


def log_prediction_submit_invariants(
    logger: logging.Logger,
    assignment: OrchestratorAssignment,
    digest: EvidenceDigest,
) -> None:
    prob, conf, is_valid, reasons, _raw = normalize_prediction_values(digest)
    if not is_valid:
        logger.warning(
            "Assignment id=%s prediction normalization applied: reasons=%s",
            assignment.agentPredictionId,
            ",".join(reasons),
        )
    if prob < 0.0 or prob > 1.0:
        logger.warning(
            "Assignment id=%s prediction out of range [0,1]: %s",
            assignment.agentPredictionId,
            prob,
        )

    if conf is not None and (float(conf) < 0.0 or float(conf) > 1.0):
        logger.warning(
            "Assignment id=%s confidence out of range [0,1]: %s",
            assignment.agentPredictionId,
            conf,
        )
    total = prob + (1.0 - prob)
    if abs(total - 1.0) > 1e-6:
        logger.warning(
            "Assignment id=%s outcome probabilities do not sum to 1.0 (sum=%s).",
            assignment.agentPredictionId,
            total,
        )


def build_prediction_submit_payload(
    assignment: OrchestratorAssignment,
    digest: EvidenceDigest,
) -> dict:
    digest_dump = digest.model_dump(mode="json")
    raw_provider_calls = [
        call for call in digest_dump.get("provider_calls", []) if isinstance(call, dict)
    ]
    raw_reasoning_chain = [
        step for step in digest_dump.get("reasoning_chain", []) if isinstance(step, dict)
    ]
    provider_calls = _build_provider_calls(raw_provider_calls)
    steps = _build_reasoning_steps(raw_reasoning_chain)
    execution_context = digest_dump.get("execution_context", {})
    trace_integrity = digest_dump.get("trace_integrity", {})
    prediction_output = digest_dump.get("prediction_output", {})
    reasoning_summary = (
        prediction_output.get("reasoning_summary")
        if isinstance(prediction_output, dict)
        else None
    )
    for step in reversed(steps):
        if (
            isinstance(reasoning_summary, str)
            and step.get("reasoningText") == reasoning_summary
        ):
            step.pop("reasoningText")
            step["reasoningTextRef"] = _REASONING_SUMMARY_REF
            break
    evidence_digest = _compact_evidence_digest(digest_dump, reasoning_summary)

    yes_id, no_id = resolve_binary_outcome_ids(assignment)
    (
        probability,
        confidence,
        is_valid_prediction,
        invalid_reasons,
        raw_observed,
    ) = normalize_prediction_values(digest)
    predicted_outcome_id = yes_id if probability >= 0.5 else no_id
    outcome_probs = {
        yes_id: round(probability, 6),
        no_id: round(1.0 - probability, 6),
    }
    submitted_at = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    provider_id, model = trace_primary_model_info(digest)
    invalid_reason = "; ".join(invalid_reasons) if invalid_reasons else None

    return {
        "agentPredictionId": assignment.agentPredictionId,
        "prediction": {
            "schemaVersion": "1.0",
            "submittedAt": submitted_at,
            "predictionType": "binary",
            "predictedOutcomeId": predicted_outcome_id,
            "confidence": confidence,
            "outcomeProbabilities": outcome_probs,
            "outcomePricesAtPrediction": assignment.event.currentOutcomePrices,
            "marketSnapshot": {
                "source": assignment.event.source,
                "sourceMarketId": assignment.event.sourceMarketId or assignment.event.marketId,
                "capturedAt": submitted_at,
            },
            "executionMetadata": {
                "model": model,
                "latencyMs": digest.execution_context.execution_duration_ms,
                "predictionIsInvalid": not is_valid_prediction,
                "predictionInvalidReason": invalid_reason,
                "predictionValidation": {
                    "isValid": is_valid_prediction,
                    "reasons": invalid_reasons,
                    "rawObserved": raw_observed,
                },
            },
        },
        "reasoningTrace": {
            "schemaVersion": "1.0",
            "trace": {
                "providerCalls": provider_calls,
                "steps": steps,
                "evidenceDigest": evidence_digest,
            },
            "traceSummary": {
                "strategy": "validator-assignment-pipeline",
                "executionId": execution_context.get("execution_id"),
                "traceHash": trace_integrity.get("trace_hash"),
                "providerCallCount": len(provider_calls),
                "reasoningStepCount": len(steps),
                "totalProviderCost": trace_integrity.get("total_provider_cost"),
                "usageTotals": _camelize(trace_integrity.get("usage_totals")),
                "contrarianFlag": prediction_output.get("contrarian_flag"),
            },
            "modelMetadata": {
                "provider": provider_id,
                "model": model,
                "agentId": execution_context.get("agent_id"),
                "agentVersion": execution_context.get("agent_version"),
                "validatorId": execution_context.get("validator_id"),
            },
        },
    }


def process_single_assignment(
    *,
    assignment: OrchestratorAssignment,
    config: AppConfig,
    orchestrator: Orchestrator,
    loaded_hotkey: Optional[LoadedKeypair],
    submit_prediction: bool,
    timeout_seconds: float = 15.0,
) -> AssignmentProcessingResult:
    event = assignment_to_event(assignment)

    if not config.validator.sandbox_type.startswith("docker"):
        raise RuntimeError(
            "Refusing orchestrator agent execution because sandbox is not docker-based"
        )
    if not assignment.agent.code:
        raise RuntimeError("assignment has no agent.code")

    agent = build_sandbox_assignment_agent(assignment)
    digest = orchestrator.run_agent(
        event,
        agent,
        miner_hotkey=assignment.miner.minerHotkey,
    )
    submit_payload = build_prediction_submit_payload(assignment, digest)

    submit_ack: SubmitPredictionResponse | None = None
    if submit_prediction:
        if loaded_hotkey is None:
            raise RuntimeError("validator hotkey unavailable for prediction submit")
        submit_ack = submit_validator_prediction(
            base_url=config.validator.orchestrator_api_url,
            loaded_hotkey=loaded_hotkey,
            payload=submit_payload,
            timeout_seconds=timeout_seconds,
        )

    return AssignmentProcessingResult(
        event=event,
        digest=digest,
        submit_payload=submit_payload,
        submit_ack=submit_ack,
    )
