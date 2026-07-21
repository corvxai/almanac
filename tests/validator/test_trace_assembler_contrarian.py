"""F7: contrarian flag is directional disagreement vs the validator-sealed
market baseline (execution_context.market_price_at_prediction), never derived
from agent-supplied provider evidence.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from src.core.events import Event
from src.core.schemas import (
    AgentResult,
    EventCategory,
    EvidenceType,
    ExecutionContext,
    ExtractedEvidence,
    ExtractionMethod,
    ProviderCall,
    ProviderTier,
    ResponseMeta,
    SandboxEnvironment,
)
from src.agent.examples._v1_common import belief_path_single_final
from src.validator.forecasting.trace_assembler import assemble_trace


def _event() -> Event:
    now = datetime(2026, 4, 24, tzinfo=timezone.utc)
    return Event(
        event_id=uuid4(), title="Will X happen?", description="Binary event.",
        category=EventCategory.OTHER, resolution_criteria="YES iff X occurs.",
        resolution_deadline=datetime(2027, 1, 1, tzinfo=timezone.utc), created_at=now,
    )


def _exec_ctx(market_price: float | None) -> ExecutionContext:
    now = datetime(2026, 4, 24, tzinfo=timezone.utc)
    return ExecutionContext(
        execution_id=uuid4(), agent_id=uuid4(), agent_version="1.0.0", event_id=uuid4(),
        validator_id=uuid4(), timestamp_start=now, timestamp_end=now,
        execution_duration_ms=1, sandbox_environment=SandboxEnvironment.DOCKER_RUNC,
        market_price_at_prediction=market_price,
    )


def _provider_call_with_price(price: float) -> ProviderCall:
    now = datetime(2026, 4, 24, tzinfo=timezone.utc)
    return ProviderCall(
        call_index=0, provider_id="polymarket", model=None,
        provider_tier=ProviderTier.FREE_SIGNAL, call_type="get_market", query_params_summary="q",
        response_meta=ResponseMeta(response_size_bytes=100, data_freshness=now),
        extracted_evidence=[
            ExtractedEvidence(
                evidence_type=EvidenceType.PRICE,
                content=f"price {price}",
                extraction_method=ExtractionMethod.DIRECT_VALUE,
                numeric_value=price,
            )
        ],
        raw_response_hash="h", latency_ms=10, cost_units=0.0,
    )


def _assemble(prediction: float, market_price: float | None, provider_calls=None):
    return assemble_trace(
        execution_context=_exec_ctx(market_price),
        event=_event(),
        provider_calls=provider_calls if provider_calls is not None else [],
        agent_result=AgentResult(
            prediction=prediction, confidence=0.7, reasoning="r",
            beliefPath=belief_path_single_final(prediction, "r"),
        ),
    )


def test_contrarian_true_when_pred_and_market_straddle_half() -> None:
    # 0.7 vs sealed 0.3 -> opposite sides of 0.5 -> contrarian
    digest = _assemble(0.7, market_price=0.3)
    assert digest.prediction_output.contrarian_flag is True


def test_contrarian_false_when_same_side_of_half() -> None:
    # 0.6 vs 0.55 -> same side -> not contrarian
    digest = _assemble(0.6, market_price=0.55)
    assert digest.prediction_output.contrarian_flag is False


def test_contrarian_false_when_no_sealed_baseline() -> None:
    digest = _assemble(0.7, market_price=None)
    assert digest.prediction_output.contrarian_flag is False


def test_contrarian_independent_of_provider_call_price() -> None:
    # A provider 'price' evidence that would have straddled 0.5 must NOT flip the
    # flag; only the sealed baseline drives it. Sealed baseline agrees (same side)
    # while the provider price disagrees -> flag stays False.
    digest = _assemble(
        0.7,
        market_price=0.6,  # same side as 0.7 -> not contrarian
        provider_calls=[_provider_call_with_price(0.2)],  # would straddle if it were read
    )
    assert digest.prediction_output.contrarian_flag is False


@pytest.mark.parametrize("pred,market,expected", [
    (0.7, 0.3, True),
    (0.3, 0.7, True),
    (0.6, 0.55, False),
    (0.4, 0.45, False),
    (0.5, 0.3, False),   # pred exactly at 0.5 -> product is 0 -> not < 0
    (0.7, 0.5, False),   # market exactly at 0.5 -> product is 0 -> not < 0
])
def test_contrarian_directional_matrix(pred: float, market: float, expected: bool) -> None:
    digest = _assemble(pred, market_price=market)
    assert digest.prediction_output.contrarian_flag is expected
