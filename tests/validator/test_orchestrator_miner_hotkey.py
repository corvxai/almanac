"""Orchestrator must thread miner_hotkey into proxy run registration.

The local proxy derives the signed billing identity solely from the run
registration (never the agent's request body), so the orchestrator must pass a
real miner_hotkey to register_run on the docker path — including via
run_all_agents, which previously dropped it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch
from uuid import UUID, uuid4

from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.config import AppConfig
from src.core.events import Event
from src.core.schemas import AgentResult, BeliefStep, EventCategory, ResolutionRecord
from src.storage.store import TraceStore
from src.validator.forecasting.orchestrator import Orchestrator

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


class _FakeProxyState:
    """Records register_run/pop_calls the way the real LocalProxyState is used."""

    def __init__(self) -> None:
        self.registrations: list[tuple[str, str, str | None]] = []

    def register_run(self, run_id, track="MAIN", *, miner_hotkey=None) -> None:
        self.registrations.append((str(run_id), track, miner_hotkey))

    def pop_calls(self, run_id):
        return []


class _NoopAgent(BaseAgent):
    agent_id = UUID("22222222-2222-2222-2222-222222222222")
    agent_version = "0.1.0"

    def predict(self, ctx: ForecastingContext) -> AgentResult:
        return AgentResult(prediction=0.5, reasoning="test", beliefPath=_FINAL_BP)


def _event() -> Event:
    return Event(
        event_id=uuid4(),
        title="Will X happen?",
        description="Binary event.",
        category=EventCategory.OTHER,
        resolution_criteria="YES iff X occurs.",
        resolution_deadline=datetime(2027, 1, 1, tzinfo=timezone.utc),
    )


def _docker_orchestrator(proxy: _FakeProxyState) -> Orchestrator:
    cfg = AppConfig.load_default()
    cfg.validator.sandbox_type = "docker_runc"
    return Orchestrator(
        config=cfg, store=_InMemoryStore(), providers=[], local_proxy_state=proxy
    )


def test_run_agent_registers_with_miner_hotkey() -> None:
    proxy = _FakeProxyState()
    orch = _docker_orchestrator(proxy)
    with patch(
        "src.validator.forecasting.orchestrator.execute_agent",
        return_value=AgentResult(prediction=0.5, reasoning="ok", beliefPath=_FINAL_BP),
    ), patch(
        "src.validator.forecasting.orchestrator.fetch_polymarket_baseline", return_value=None
    ):
        orch.run_agent(_event(), _NoopAgent(), miner_hotkey="5RealMiner")

    assert len(proxy.registrations) == 1
    assert proxy.registrations[0][2] == "5RealMiner"


def test_run_all_agents_forwards_miner_hotkey() -> None:
    """Regression: run_all_agents must not drop miner_hotkey (would 400 every call)."""
    proxy = _FakeProxyState()
    orch = _docker_orchestrator(proxy)
    agents = [_NoopAgent(), _NoopAgent()]
    with patch(
        "src.validator.forecasting.orchestrator.execute_agent",
        return_value=AgentResult(prediction=0.5, reasoning="ok", beliefPath=_FINAL_BP),
    ), patch(
        "src.validator.forecasting.orchestrator.fetch_polymarket_baseline", return_value=None
    ):
        orch.run_all_agents(_event(), agents, miner_hotkey="5RealMiner")

    assert len(proxy.registrations) == 2
    assert all(reg[2] == "5RealMiner" for reg in proxy.registrations)
