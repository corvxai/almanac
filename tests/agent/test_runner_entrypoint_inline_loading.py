from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.agent import runner_entrypoint
from src.agent.base import BaseAgent
from src.core.schemas import AgentResult, BeliefStep

_FINAL_BP = [BeliefStep(step=0, type="final", probability=0.5, text="ok")]


class _FakeAgent(BaseAgent):
    def predict(self, ctx):  # noqa: ANN001
        return AgentResult(prediction=0.5, reasoning="ok", beliefPath=_FINAL_BP)


class _DummyClient:
    def close(self) -> None:
        return None


def _write_payload(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "payload.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _common_monkeypatch(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        runner_entrypoint,
        "Event",
        SimpleNamespace(model_validate=lambda _event: object()),
    )
    monkeypatch.setattr(runner_entrypoint, "_make_uds_client", lambda *_args, **_kw: _DummyClient())
    monkeypatch.setattr(
        runner_entrypoint,
        "SandboxGateway",
        lambda provider_factory: SimpleNamespace(provider_factory=provider_factory),
    )
    monkeypatch.setattr(
        runner_entrypoint,
        "ForecastingContext",
        lambda event, gateway: SimpleNamespace(event=event, gateway=gateway),
    )
    monkeypatch.setenv("RUN_ID", "run-123")


def test_inline_code_ignores_module_agent_class_for_inline_lookup(tmp_path: Path, monkeypatch, capsys):
    payload_path = _write_payload(
        tmp_path,
        {
            "event": {},
            "agent_module": "src.validator.forecasting.assignment_pipeline",
            "agent_class": "_InlineCodeSandboxAgent",
            "agent_code": "class Ignored:\n    pass\n",
        },
    )
    _common_monkeypatch(monkeypatch)
    monkeypatch.setenv("FORECASTING_RUNNER_INPUT_FILE", str(payload_path))

    seen = {"inline_class": "unset"}

    def _fake_inline_loader(_code: str, inline_class: str | None = None):
        seen["inline_class"] = inline_class
        return _FakeAgent

    monkeypatch.setattr(runner_entrypoint, "_load_agent_class_from_code", _fake_inline_loader)
    monkeypatch.setattr(
        runner_entrypoint,
        "_load_agent_class",
        lambda *_args, **_kw: (_ for _ in ()).throw(AssertionError("module loader should not be used")),
    )

    exit_code = runner_entrypoint.main([])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert seen["inline_class"] is None
    assert '"prediction":0.5' in captured.out


def test_inline_code_uses_inline_class_key_when_present(tmp_path: Path, monkeypatch, capsys):
    payload_path = _write_payload(
        tmp_path,
        {
            "event": {},
            "agent_module": "src.validator.forecasting.assignment_pipeline",
            "agent_class": "_InlineCodeSandboxAgent",
            "agent_code": "class Ignored:\n    pass\n",
            "inline_class": "MinerClaudeAgent",
        },
    )
    _common_monkeypatch(monkeypatch)
    monkeypatch.setenv("FORECASTING_RUNNER_INPUT_FILE", str(payload_path))

    seen = {"inline_class": None}

    def _fake_inline_loader(_code: str, inline_class: str | None = None):
        seen["inline_class"] = inline_class
        return _FakeAgent

    monkeypatch.setattr(runner_entrypoint, "_load_agent_class_from_code", _fake_inline_loader)
    monkeypatch.setattr(
        runner_entrypoint,
        "_load_agent_class",
        lambda *_args, **_kw: (_ for _ in ()).throw(AssertionError("module loader should not be used")),
    )

    exit_code = runner_entrypoint.main([])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert seen["inline_class"] == "MinerClaudeAgent"
    assert '"prediction":0.5' in captured.out
