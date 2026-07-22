from __future__ import annotations

from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from miner import cli
from src.core.events import Event
from src.core.schemas import EventCategory


_VALID_AGENT = """
from uuid import UUID
from src.agent.base import BaseAgent
from src.core.schemas import AgentResult, BeliefStep

class TestAgent(BaseAgent):
    agent_id = UUID("11111111-1111-4111-8111-111111111111")
    agent_version = "1.0.0"

    def predict(self, ctx):
        return AgentResult(
            prediction=0.6,
            reasoning="A valid local test result.",
            beliefPath=[
                BeliefStep(step=0, type="final", probability=0.6, text="Final.")
            ],
        )
"""


class _FakePortalGateway:
    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        self.available_providers = ("openrouter",)
        self.call_log = []

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:  # noqa: ANN002
        return None

    def call_provider(self, provider_id, call_type, params):  # noqa: ANN001
        raise AssertionError("the fixture agent should not make provider calls")


def _event() -> Event:
    return Event(
        title="Will the test pass?",
        description="A test event.",
        category=EventCategory.OTHER,
        resolution_criteria="Resolves yes if it passes.",
        resolution_deadline=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )


def _args(agent_file: Path) -> Namespace:
    return Namespace(
        agent_file=agent_file,
        event_id=None,
        gateway_api_key="portal-key",
        orchestrator_url="https://api.example",
        timeout_seconds=None,
    )


def test_test_agent_validates_and_prints_result(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    agent_file = tmp_path / "agent.py"
    agent_file.write_text(_VALID_AGENT)
    monkeypatch.setattr(cli, "PortalGateway", _FakePortalGateway)
    monkeypatch.setattr(cli, "fetch_portal_event", lambda **kwargs: _event())
    monkeypatch.delenv("FORECASTING_TIMEOUT_SECONDS", raising=False)

    exit_code = cli._handle_test_agent(_args(agent_file))

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Agent result validated successfully." in output
    assert "Prediction: 0.600000" in output
    assert '"beliefPath"' in output


def test_test_agent_rejects_inherited_identity(
    tmp_path: Path,
    capsys,
) -> None:
    agent_file = tmp_path / "agent.py"
    agent_file.write_text(
        "from src.agent.base import BaseAgent\n"
        "class TestAgent(BaseAgent):\n"
        "    pass\n"
    )

    exit_code = cli._handle_test_agent(_args(agent_file))

    assert exit_code == 2
    assert "must declare a stable agent_id UUID" in capsys.readouterr().out


def test_test_agent_uses_120_second_default_timeout(monkeypatch) -> None:
    monkeypatch.delenv("FORECASTING_TIMEOUT_SECONDS", raising=False)
    args = Namespace(timeout_seconds=None)
    assert cli._resolve_test_timeout(args) == 120.0


def test_balance_uses_gateway_key_and_prints_response(monkeypatch, capsys) -> None:
    seen = {}

    def fake_request(**kwargs):  # noqa: ANN003
        seen.update(kwargs)
        return SimpleNamespace(
            json=lambda: {"balanceMicro": "420000"},
            text="",
        )

    monkeypatch.setattr(cli, "_request", fake_request)
    args = Namespace(
        gateway_api_key="portal-key",
        orchestrator_url=None,
        timeout_seconds=None,
    )

    exit_code = cli._handle_balance(args)

    assert exit_code == 0
    assert seen["method"] == "GET"
    assert seen["endpoint"] == "v1/credits/balance"
    assert seen["extra_headers"]["Authorization"] == "Bearer portal-key"
    assert '"balanceMicro": "420000"' in capsys.readouterr().out


def test_balance_requires_gateway_key(monkeypatch, capsys) -> None:
    monkeypatch.delenv("GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("FORECASTING_GATEWAY_API_KEY", raising=False)
    args = Namespace(gateway_api_key=None)

    exit_code = cli._handle_balance(args)

    assert exit_code == 2
    assert "gateway API key is required" in capsys.readouterr().out
