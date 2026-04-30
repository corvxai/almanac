"""Evidence Digest schema — the canonical data structures for every forecasting trace.

All models use Pydantic v2 BaseModel. This module is the single source of truth
for the shape of data flowing through the system.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SandboxEnvironment(str, Enum):
    IN_PROCESS = "in_process"
    SUBPROCESS = "subprocess"
    DOCKER = "docker"
    DOCKER_RUNC = "docker_runc"
    DOCKER_GVISOR = "docker_gvisor"


class EventCategory(str, Enum):
    GEOPOLITICAL = "geopolitical"
    ECONOMIC = "economic"
    REGULATORY = "regulatory"
    CLIMATE = "climate"
    TECH = "tech"
    SCIENCE = "science"
    SPORTS = "sports"
    CULTURE = "culture"
    OTHER = "other"


class ProviderTier(str, Enum):
    FREE_SIGNAL = "free_signal"
    SEARCH = "search"
    DEEP_RESEARCH = "deep_research"
    VERIFICATION = "verification"
    INFERENCE = "inference"
    PROPRIETARY = "proprietary"


class EvidenceType(str, Enum):
    STATISTIC = "statistic"
    QUOTE_SUMMARY = "quote_summary"
    TREND = "trend"
    SIGNAL = "signal"
    PRICE = "price"
    PROBABILITY = "probability"
    FACT = "fact"


class ExtractionMethod(str, Enum):
    SCHEMA_MAPPING = "schema_mapping"
    STATISTICAL_SUMMARY = "statistical_summary"
    NLP_EXTRACTION = "nlp_extraction"
    DIRECT_VALUE = "direct_value"


class ReasoningStepType(str, Enum):
    EVIDENCE_GATHERING = "evidence_gathering"
    SYNTHESIS = "synthesis"
    CONFLICT_RESOLUTION = "conflict_resolution"
    PRIOR_UPDATE = "prior_update"
    CALIBRATION_CHECK = "calibration_check"
    FINAL_ASSIGNMENT = "final_assignment"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class ResponseMeta(BaseModel):
    response_size_bytes: int
    record_count: Optional[int] = None
    data_freshness: datetime
    data_range_start: Optional[datetime] = None
    data_range_end: Optional[datetime] = None


class UsageMeta(BaseModel):
    """Normalized provider usage/accounting metadata for one call."""

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    cost: Optional[float] = None
    is_byok: Optional[bool] = None


class UsageTotals(BaseModel):
    """Execution-level aggregate usage across all provider calls."""

    provider_calls_with_usage: int = Field(ge=0, default=0)
    prompt_tokens: int = Field(ge=0, default=0)
    completion_tokens: int = Field(ge=0, default=0)
    total_tokens: int = Field(ge=0, default=0)
    input_tokens: int = Field(ge=0, default=0)
    output_tokens: int = Field(ge=0, default=0)
    cached_tokens: int = Field(ge=0, default=0)
    reasoning_tokens: int = Field(ge=0, default=0)


class ExtractedEvidence(BaseModel):
    evidence_type: EvidenceType
    content: str
    extraction_method: ExtractionMethod
    numeric_value: Optional[float] = None


# ---------------------------------------------------------------------------
# Top-level trace components
# ---------------------------------------------------------------------------

class ExecutionContext(BaseModel):
    execution_id: UUID
    agent_id: UUID
    agent_version: str
    event_id: UUID
    validator_id: UUID
    timestamp_start: datetime
    timestamp_end: datetime
    execution_duration_ms: int
    sandbox_environment: SandboxEnvironment


class EventSnapshot(BaseModel):
    event_title: str
    event_category: EventCategory
    event_subcategory: Optional[str] = None
    resolution_criteria: str
    resolution_deadline: datetime
    event_created_at: datetime


class ProviderCall(BaseModel):
    """A single external data call — produced entirely by the gateway/validator."""

    call_index: int = Field(ge=0)
    provider_id: str
    model: Optional[str] = None
    provider_tier: ProviderTier
    call_type: str
    query_params_summary: str
    response_meta: ResponseMeta
    usage_meta: Optional[UsageMeta] = None
    provider_usage_raw: Optional[dict[str, Any]] = None
    extracted_evidence: list[ExtractedEvidence]
    raw_response_hash: str
    latency_ms: int = Field(ge=0)
    cost_units: float = Field(ge=0)


class ReasoningStep(BaseModel):
    """A single step in the agent's reasoning chain — validator-assembled.

    Built by the trace assembler from gateway call logs and the agent's
    free-text reasoning. Agents may optionally provide structured steps
    via the context, but it is never required.
    """

    step_index: int = Field(ge=0)
    step_type: ReasoningStepType
    provider_call_index: Optional[int] = Field(default=None, ge=0)
    provider_id: Optional[str] = None
    input_evidence_refs: list[int] = Field(default_factory=list)
    reasoning_text: str
    agent_interpretation: Optional[str] = None
    intermediate_probability: Optional[float] = Field(default=None, ge=0, le=1)
    confidence_delta: Optional[float] = None
    conflict_signals: Optional[list[str]] = None
    inference_model_used: Optional[str] = None


class PredictionOutput(BaseModel):
    """Structured prediction record — validator-assembled from the agent's result."""

    final_probability: float = Field(ge=0, le=1)
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    confidence_interval: Optional[tuple[float, float]] = None
    reasoning_summary: str = ""
    key_drivers: list[str] = Field(default_factory=list)
    key_uncertainties: list[str] = Field(default_factory=list)
    contrarian_flag: bool = False
    ensemble_at_prediction: Optional[float] = Field(default=None, ge=0, le=1)

    @field_validator("confidence_interval")
    @classmethod
    def _ci_ordered(cls, v: Optional[tuple[float, float]]) -> Optional[tuple[float, float]]:
        if v is not None:
            lo, hi = v
            if not (0 <= lo <= hi <= 1):
                raise ValueError("confidence_interval must satisfy 0 <= lo <= hi <= 1")
        return v


# ---------------------------------------------------------------------------
# Agent return type — the minimal contract agents must satisfy
# ---------------------------------------------------------------------------

class AgentResult(BaseModel):
    """What an agent returns. This is the entire agent-side contract.

    Agents return a probability and a free-text reasoning string.
    Everything else is optional. The validator/trace assembler is
    responsible for building the structured EvidenceDigest from this
    plus the gateway call logs.
    """

    prediction: float = Field(ge=0, le=1)
    reasoning: str
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    metadata: Optional[dict[str, Any]] = None


class ResolutionRecord(BaseModel):
    """Post-hoc resolution outcome — populated by the system after the event resolves."""

    resolved: bool = False
    resolution_outcome: Optional[bool] = None
    resolution_timestamp: Optional[datetime] = None
    resolution_source: Optional[str] = None
    brier_score: Optional[float] = Field(default=None, ge=0, le=1)
    composite_score: Optional[float] = None


class TraceIntegrity(BaseModel):
    """Cryptographic integrity envelope — produced by the validator."""

    trace_hash: str
    merkle_root: Optional[str] = None
    anchor_tx: Optional[str] = None
    anchor_timestamp: Optional[datetime] = None
    trace_schema_version: str = "0.1.0"
    total_provider_cost: float = Field(ge=0)
    total_evidence_items: int = Field(ge=0)
    usage_totals: Optional[UsageTotals] = None


# ---------------------------------------------------------------------------
# Top-level Evidence Digest
# ---------------------------------------------------------------------------

class EvidenceDigest(BaseModel):
    """Complete trace for a single agent execution on a single event."""

    execution_context: ExecutionContext
    event_snapshot: EventSnapshot
    provider_calls: list[ProviderCall]
    reasoning_chain: list[ReasoningStep]
    prediction_output: PredictionOutput
    resolution_record: ResolutionRecord = Field(default_factory=ResolutionRecord)
    trace_integrity: TraceIntegrity

    def compute_trace_hash(self) -> str:
        """SHA-256 of the digest with trace_hash zeroed out to avoid circularity."""
        snapshot = self.model_copy(
            update={
                "trace_integrity": self.trace_integrity.model_copy(
                    update={"trace_hash": ""}
                )
            }
        )
        payload = snapshot.model_dump_json(indent=None).encode()
        return hashlib.sha256(payload).hexdigest()

    def seal(self) -> "EvidenceDigest":
        """Return a copy with trace_hash computed and set."""
        h = self.compute_trace_hash()
        return self.model_copy(
            update={
                "trace_integrity": self.trace_integrity.model_copy(
                    update={"trace_hash": h}
                )
            }
        )

    def verify_integrity(self) -> bool:
        return self.trace_hash_matches()

    def trace_hash_matches(self) -> bool:
        return self.trace_integrity.trace_hash == self.compute_trace_hash()
