"""Orchestrator tests for validator-side market baseline lookup behavior."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from unittest.mock import patch
from uuid import UUID, uuid4

import httpx
import pytest

from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.config import AppConfig
from src.core.events import Event
from src.core.schemas import (
    AgentResult,
    BeliefStep,
    EventCategory,
    ProviderCall,
    ProviderTier,
    ResolutionRecord,
    ResponseMeta,
)
from src.gateway.providers.base import BaseProvider
from src.storage.store import TraceStore
from src.validator.forecasting.assignment_pipeline import process_single_assignment
from src.validator.forecasting.orchestrator import Orchestrator
from src.validator.forecasting.orchestrator_api import OrchestratorAssignment

_FINAL_BP = [BeliefStep(step=0, type="final", probability=0.5, text="test")]


class _InMemoryStore(TraceStore):
    def __init__(self) -> None:
        self.saved = []

    def save_trace(self, digest) -> None:
        self.saved.append(digest)

    def get_trace(self, execution_id: UUID):
        return None

    def list_traces_by_event(self, event_id: UUID):
        return []

    def list_traces_by_agent(self, agent_id: UUID):
        return []

    def save_resolution(self, event_id: UUID, record: ResolutionRecord) -> None:
        return None

    def get_resolution(self, event_id: UUID):
        return None


class _CountingPolymarketProvider(BaseProvider):
    provider_id = "polymarket"
    provider_tier = ProviderTier.FREE_SIGNAL

    def __init__(self) -> None:
        self.market_calls = 0

    def call(self, call_type: str, params: dict) -> dict:
        if call_type == "get_market":
            self.market_calls += 1
            return {"price": 0.44, "volume_24h": 1000.0}
        raise ValueError(f"unsupported call_type: {call_type}")


class _SingleMarketAgent(BaseAgent):
    agent_id = UUID("11111111-1111-1111-1111-111111111111")
    agent_version = "0.1.0"

    def predict(self, ctx: ForecastingContext) -> AgentResult:
        ctx.call_provider("polymarket", "get_market", {"market_slug": "test-market"})
        return AgentResult(prediction=0.5, reasoning="test", beliefPath=_FINAL_BP)


def test_market_baseline_not_added_to_provider_calls() -> None:
    cfg = AppConfig.load_default()
    cfg.validator.sandbox_type = "in_process"
    store = _InMemoryStore()
    polymarket = _CountingPolymarketProvider()
    orchestrator = Orchestrator(config=cfg, store=store, providers=[polymarket])

    event = Event(
        event_id=uuid4(),
        title="Will X happen?",
        description="Binary event.",
        category=EventCategory.OTHER,
        resolution_criteria="YES iff X occurs.",
        resolution_deadline=datetime(2027, 1, 1, tzinfo=timezone.utc),
        created_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
        source="polymarket",
        source_id="485257",
        source_url="https://polymarket.com/event/test-market",
    )

    with patch(
        "src.validator.forecasting.orchestrator.fetch_polymarket_baseline",
        return_value={
            "provider": "polymarket",
            "market_slug": "test-market",
            "source_id": "485257",
            "yes_price": 0.44,
            "no_price": 0.56,
            "history_points": 1,
            "as_of": "2026-01-02T00:00:00+00:00",
        },
    ) as fetch_baseline:
        digest = orchestrator.run_agent(event, _SingleMarketAgent())
    fetch_baseline.assert_called_once_with(event)

    assert polymarket.market_calls == 1
    assert len(digest.provider_calls) == 1
    assert digest.provider_calls[0].call_type == "get_market"

    # v1: the validator baseline YES price is written to the hash-sealed
    # execution context, not to agent-writable prediction metadata.
    assert digest.execution_context.market_price_at_prediction == 0.44


_SYNC_PRICES = {"oid_yes": 0.26, "oid_no": 0.74}


def _pipeline_assignment() -> OrchestratorAssignment:
    code = "class Agent:\n    pass\n"
    sha = hashlib.sha256(code.encode("utf-8")).hexdigest()
    return OrchestratorAssignment.model_validate(
        {
            "agentPredictionId": "pred_123",
            "status": "in_progress",
            "leaseExpiresAt": "2026-06-05T19:59:44.271000Z",
            "miner": {
                "minerHotkey": "5EqZoEKc6c8TaG4xRRHTT1uZiQF5jkjQCeUV5t77L6YbeaJ8",
                "minerUid": 17,
            },
            "event": {
                "marketId": "cmpxszzto007luaaznuulufaq",
                "source": "polymarket",
                "sourceMarketId": "2411919",
                "question": "Will X happen?",
                "description": "Binary event",
                "endDate": "2026-06-06T16:00:00Z",
                "outcomes": [
                    {"outcomeId": "oid_yes", "name": "Yes"},
                    {"outcomeId": "oid_no", "name": "No"},
                ],
                "currentOutcomePrices": dict(_SYNC_PRICES),
            },
            "agent": {
                "agentUploadId": "cmpwwsd4800000razamvsplvc",
                "sha256": sha,
                "uploadedAt": "2026-06-02T17:25:57.224000Z",
                "code": code,
            },
        }
    )


class _SandboxProxy:
    def register_run(self, run_id, track="MAIN", *, miner_hotkey=None) -> None:
        return None

    def pop_calls(self, run_id):
        now = datetime(2026, 4, 24, tzinfo=timezone.utc)
        return [
            ProviderCall(
                call_index=0,
                provider_id="openrouter",
                model="m",
                provider_tier=ProviderTier.INFERENCE,
                call_type="chat_completion",
                query_params_summary="q",
                response_meta=ResponseMeta(response_size_bytes=10, data_freshness=now),
                extracted_evidence=[],
                raw_response_hash="h",
                latency_ms=5,
                cost_units=0.0,
            )
        ]


def _run_pipeline(baseline: object):
    cfg = AppConfig()
    cfg.validator.sandbox_type = "docker_runc"
    orchestrator = Orchestrator(
        config=cfg,
        store=_InMemoryStore(),
        providers=[],
        local_proxy_state=_SandboxProxy(),
    )
    captured: list[dict] = []

    def _container(agent, ctx, _cfg, run_id):
        captured.append(ctx.event.model_dump(mode="json"))
        return AgentResult(
            prediction=0.8,
            confidence=0.7,
            reasoning="because the evidence says so",
            beliefPath=[BeliefStep(step=0, type="final", probability=0.8, text="final")],
        )

    fetch_kwargs: dict[str, object] = (
        {"side_effect": baseline}
        if isinstance(baseline, BaseException)
        else {"return_value": baseline}
    )
    with (
        patch(
            "src.validator.forecasting.assignment_pipeline.fetch_polymarket_baseline",
            **fetch_kwargs,
        ) as fetch,
        patch.object(
            orchestrator,
            "_collect_market_baseline",
            side_effect=AssertionError("refetch"),
        ),
        patch(
            "src.validator.forecasting.sandbox_docker.run_agent_in_container",
            side_effect=_container,
        ),
    ):
        result = process_single_assignment(
            assignment=_pipeline_assignment(),
            config=cfg,
            orchestrator=orchestrator,
            loaded_hotkey=None,
            submit_prediction=False,
        )
    fetch.assert_called_once()
    assert len(captured) == 1
    return result, captured[0]


def test_pipeline_sandbox_event_uses_live_gamma_prices() -> None:
    result, event = _run_pipeline(
        {
            "yes_price": 0.25,
            "no_price": 0.75,
            "yes_outcome_price": 0.25,
            "no_outcome_price": 0.5,
        }
    )
    live = {"oid_yes": 0.25, "oid_no": 0.5}
    assert event["current_outcome_prices"] == live
    prediction = result.submit_payload["prediction"]
    assert prediction["outcomePricesAtPrediction"] == live
    assert prediction["executionMetadata"]["predictionIsInvalid"] is False
    assert result.digest.execution_context.market_price_at_prediction == 0.25
    assert result.digest.prediction_output.contrarian_flag is True


@pytest.mark.parametrize("baseline", [None, httpx.ConnectError("down")])
def test_pipeline_sandbox_event_keeps_sync_prices_when_fetch_fails(baseline: object) -> None:
    result, event = _run_pipeline(baseline)
    assert event["current_outcome_prices"] == _SYNC_PRICES
    prediction = result.submit_payload["prediction"]
    assert prediction["outcomePricesAtPrediction"] == {}
    assert prediction["executionMetadata"]["predictionIsInvalid"] is False
    assert result.digest.execution_context.market_price_at_prediction is None
    assert result.digest.prediction_output.contrarian_flag is False


def test_pipeline_partial_price_map_keeps_sync_prices() -> None:
    result, event = _run_pipeline(
        {"yes_price": 0.25, "no_price": 0.75, "yes_outcome_price": 0.25}
    )
    assert event["current_outcome_prices"] == _SYNC_PRICES
    prediction = result.submit_payload["prediction"]
    assert prediction["outcomePricesAtPrediction"] == {}
    assert prediction["executionMetadata"]["predictionIsInvalid"] is False
    assert result.digest.execution_context.market_price_at_prediction == 0.25
    assert result.digest.prediction_output.contrarian_flag is True
