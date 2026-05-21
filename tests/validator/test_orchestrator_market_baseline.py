"""Orchestrator tests for validator-side market baseline lookup behavior."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch
from uuid import UUID, uuid4

from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.config import AppConfig
from src.core.events import Event
from src.core.schemas import AgentResult, EventCategory, ProviderTier, ResolutionRecord
from src.gateway.providers.base import BaseProvider
from src.storage.store import TraceStore
from src.validator.orchestrator import Orchestrator


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
        return AgentResult(prediction=0.5, reasoning="test")


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
        "src.validator.orchestrator.fetch_polymarket_baseline",
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

    md = digest.prediction_output.metadata
    assert md is not None
    baseline = md.get("market_baseline")
    assert isinstance(baseline, dict)
    assert baseline["source_id"] == "485257"
    assert baseline["yes_price"] == 0.44
    assert baseline["no_price"] == 0.56
