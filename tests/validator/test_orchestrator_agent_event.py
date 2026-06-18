from __future__ import annotations

import datetime
from types import SimpleNamespace

import httpx

from src.core.config import AppConfig
from src.core.constants import ORCHESTRATOR_API_URL
from src.validator.orchestrator_api import (
    AGENT_AND_EVENT_ENDPOINT,
    AgentAndEventResponse,
    OrchestratorAssignment,
    AssignmentAgent,
    AssignmentEvent,
    AssignmentMiner,
    AUTH_DOMAIN,
    fetch_agent_event_assignment,
    build_assignment_auth_headers,
)
from src.validator import validator as validator_module
from src.validator.validator import Validator


class _FakeKeypair:
    def __init__(self) -> None:
        self.last_message: bytes | None = None

    def sign(self, message: bytes) -> bytes:
        self.last_message = message
        return bytes.fromhex("aabb")


def _bt_objects():
    return validator_module._BtObjects(  # noqa: SLF001 - tests use internal helper
        wallet=SimpleNamespace(hotkey=SimpleNamespace(ss58_address="hotkey_test")),
        subtensor=SimpleNamespace(),
        dendrite=SimpleNamespace(),
        metagraph=SimpleNamespace(),
        network="test",
    )


def test_build_assignment_auth_headers_matches_contract() -> None:
    keypair = _FakeKeypair()
    loaded_hotkey = SimpleNamespace(
        keypair=keypair,
        hotkey_ss58="5ValidatorHotkey",
        coldkey_ss58=None,
    )
    headers = build_assignment_auth_headers(
        loaded_hotkey=loaded_hotkey,  # type: ignore[arg-type]
        body=b"",
        nonce="nonce123",
        timestamp=1_700_000_000,
    )
    assert keypair.last_message is not None
    assert (
        keypair.last_message.decode("utf-8")
        == f"{AUTH_DOMAIN}\nGET\n{AGENT_AND_EVENT_ENDPOINT}\n\nnonce123\n1700000000\n"
        "e3b0c44298fc1c149afbf4c8996fb924"
        "27ae41e4649b934ca495991b7852b855"
    )
    assert headers["x-validator-hotkey"] == "5ValidatorHotkey"
    assert headers["x-validator-signature"] == "0xaabb"
    assert headers["x-validator-nonce"] == "nonce123"
    assert headers["x-validator-timestamp"] == "1700000000"


def test_fetch_agent_event_assignment_handles_none_assignment() -> None:
    keypair = _FakeKeypair()
    loaded_hotkey = SimpleNamespace(
        keypair=keypair,
        hotkey_ss58="5ValidatorHotkey",
        coldkey_ss58=None,
    )

    def _handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/validators/agent-and-event"
        assert request.headers["x-validator-hotkey"] == "5ValidatorHotkey"
        return httpx.Response(200, json={"assignment": None, "reason": "none_available"})

    client = httpx.Client(transport=httpx.MockTransport(_handler))
    got = fetch_agent_event_assignment(
        base_url=ORCHESTRATOR_API_URL,
        loaded_hotkey=loaded_hotkey,  # type: ignore[arg-type]
        http_client=client,
    )
    assert got.assignment is None
    assert got.reason == "none_available"


def test_fetch_agent_event_assignment_parses_typed_assignment() -> None:
    keypair = _FakeKeypair()
    loaded_hotkey = SimpleNamespace(
        keypair=keypair,
        hotkey_ss58="5ValidatorHotkey",
        coldkey_ss58=None,
    )

    def _handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/validators/agent-and-event"
        return httpx.Response(
            200,
            json={
                "assignment": {
                    "agentPredictionId": "ap_01JX6Q6G3V9KQ7M9V4Z2M8P1T3",
                    "status": "in_progress",
                    "leaseExpiresAt": "2026-06-05T15:10:00.000Z",
                    "miner": {
                        "minerHotkey": "5F3sa2TJAWMqDhXG6jhV4N8ko9wqXxv6cM4x2eRk9WQh7YbA",
                        "minerUid": 184,
                    },
                    "event": {
                        "marketId": "cmcyr8w2b0003v6b8p9u4n7sa",
                        "source": "polymarket",
                        "sourceMarketId": "0x8f3c2f7e4f9b3a7d2e1c6b5a4d3e2f1a9b8c7d6e",
                        "question": "Will CPI YoY be above 3.2% in June 2026?",
                        "description": "Binary market on June CPI release. Resolve YES if official CPI YoY > 3.2%.",
                        "endDate": "2026-06-12T12:30:00.000Z",
                        "currentOutcomePrices": {"yes": 0.64, "no": 0.36},
                    },
                    "agent": {
                        "agentUploadId": "cmcxr9yd1000av6b87h2nk3qw",
                        "sha256": "bd3f8a4d52f3f0f8f0b5f6d53e5d6b2d7c8f3f7a2a4b6d1e9a0b3c4d5e6f7a8",
                        "uploadedAt": "2026-06-05T13:42:11.000Z",
                        "code": "import math\n\nclass Agent:\n    def predict(self, input_data):\n        return {'yes': 0.67, 'no': 0.33}\n",
                    },
                },
                "reason": None,
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(_handler))
    got = fetch_agent_event_assignment(
        base_url=ORCHESTRATOR_API_URL,
        loaded_hotkey=loaded_hotkey,  # type: ignore[arg-type]
        http_client=client,
    )
    assert got.assignment is not None
    assert got.assignment.agentPredictionId == "ap_01JX6Q6G3V9KQ7M9V4Z2M8P1T3"
    assert got.assignment.miner.minerUid == 184
    assert got.assignment.event.currentOutcomePrices["yes"] == 0.64
    assert got.assignment.agent.agentUploadId == "cmcxr9yd1000av6b87h2nk3qw"


def test_fetch_agent_event_assignment_accepts_alias_fields_and_missing_optional_fields() -> None:
    keypair = _FakeKeypair()
    loaded_hotkey = SimpleNamespace(
        keypair=keypair,
        hotkey_ss58="5ValidatorHotkey",
        coldkey_ss58=None,
    )

    def _handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        return httpx.Response(
            200,
            json={
                "assignment": {
                    "agentPredictionId": "pred_456",
                    "leaseExpiresAt": "2026-06-05T17:26:55.379Z",
                    "miner": {
                        "hotkey": "5EqZoEKc6c8TaG4xRRHTT1uZiQF5jkjQCeUV5t77L6YbeaJ8",
                        "uid": 17,
                    },
                    "event": {
                        "marketId": "cmpxszzto0000v6b8w3v5g7xq",
                        "source": "polymarket",
                        "sourceMarketId": "0xabc",
                        "question": "Will X happen?",
                        "endDate": "2026-06-10T00:00:00.000Z",
                        "currentOutcomePrices": {
                            "440677354200730912081020726286585970197196597249853152385": 0.74
                        },
                    },
                    "agent": {
                        "uploadId": "cmpwwsd480000v6b8m3xk9t2q",
                        "sha256": "bd3f8a4d52f3f0f8f0b5f6d53e5d6b2d7c8f3f7a2a4b6d1e9a0b3c4d5e6f7a8",
                        "uploadedAt": "2026-06-02T17:25:57.224Z",
                    },
                },
                "reason": None,
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(_handler))
    got = fetch_agent_event_assignment(
        base_url=ORCHESTRATOR_API_URL,
        loaded_hotkey=loaded_hotkey,  # type: ignore[arg-type]
        http_client=client,
    )
    assert got.assignment is not None
    assert got.assignment.status is None
    assert got.assignment.miner.minerUid == 17
    assert got.assignment.miner.minerHotkey.startswith("5Eq")
    assert "440677354200730912081020726286585970197196597249853152385" in got.assignment.event.currentOutcomePrices
    assert got.assignment.agent.agentUploadId == "cmpwwsd480000v6b8m3xk9t2q"
    assert got.assignment.agent.code is None


def test_validator_polling_calls_assignment_handler(monkeypatch) -> None:
    cfg = AppConfig()
    cfg.loop.loop_enabled = False
    cfg.loop.arcratio_enabled = True

    validator = Validator(config=cfg, store=None, bt_objects=_bt_objects(), metadata_manager=None)
    cfg.validator.orchestrator_api_url = ORCHESTRATOR_API_URL

    fake_loaded = SimpleNamespace(
        keypair=_FakeKeypair(),
        hotkey_ss58="5ValidatorHotkey",
        coldkey_ss58=None,
    )
    monkeypatch.setattr("src.validator.validator.load_hotkey", lambda _cfg: fake_loaded)
    monkeypatch.setattr(
        "src.validator.validator.fetch_agent_event_assignment",
        lambda **_kw: AgentAndEventResponse(
            assignment=OrchestratorAssignment(
                agentPredictionId="pred_123",
                status="in_progress",
                leaseExpiresAt="2026-06-05T15:10:00.000Z",
                miner=AssignmentMiner(
                    minerHotkey="5F3sa2TJAWMqDhXG6jhV4N8ko9wqXxv6cM4x2eRk9WQh7YbA",
                    minerUid=184,
                ),
                event=AssignmentEvent(
                    marketId="cmcyr8w2b0003v6b8p9u4n7sa",
                    source="polymarket",
                    sourceMarketId="0x8f3c2f7e4f9b3a7d2e1c6b5a4d3e2f1a9b8c7d6e",
                    question="Will CPI YoY be above 3.2% in June 2026?",
                    description="Binary market on June CPI release. Resolve YES if official CPI YoY > 3.2%.",
                    endDate="2026-06-12T12:30:00.000Z",
                    currentOutcomePrices={"yes": 0.64, "no": 0.36},
                ),
                agent=AssignmentAgent(
                    agentUploadId="cmcxr9yd1000av6b87h2nk3qw",
                    sha256="bd3f8a4d52f3f0f8f0b5f6d53e5d6b2d7c8f3f7a2a4b6d1e9a0b3c4d5e6f7a8",
                    uploadedAt="2026-06-05T13:42:11.000Z",
                    code="class Agent:\n    pass\n",
                ),
            ),
            reason=None,
        ),
    )

    seen: list[OrchestratorAssignment] = []
    monkeypatch.setattr(validator, "_handle_orchestrator_assignment", lambda a: seen.append(a))
    validator._poll_orchestrator_assignment()
    assert len(seen) == 1
    assert seen[0].agentPredictionId == "pred_123"


def test_validator_polling_skips_during_assignment_execution_cooldown(monkeypatch) -> None:
    cfg = AppConfig()
    cfg.loop.loop_enabled = False
    cfg.loop.arcratio_enabled = True
    cfg.loop.assignment_execution_cooldown_seconds = 180

    now = datetime.datetime(2026, 6, 5, 15, 0, 0, tzinfo=datetime.timezone.utc)
    validator = Validator(
        config=cfg,
        store=None,
        bt_objects=_bt_objects(),
        metadata_manager=None,
        clock=lambda: now,
    )
    cfg.validator.orchestrator_api_url = ORCHESTRATOR_API_URL
    validator._last_assignment_execution_at = now - datetime.timedelta(seconds=60)

    fetch_calls: list[dict] = []
    monkeypatch.setattr(
        "src.validator.validator.fetch_agent_event_assignment",
        lambda **kw: fetch_calls.append(kw) or AgentAndEventResponse(
            assignment=None,
            reason="none_available",
        ),
    )

    validator._poll_orchestrator_assignment()
    assert fetch_calls == []


def test_validator_polling_runs_after_assignment_execution_cooldown(monkeypatch) -> None:
    cfg = AppConfig()
    cfg.loop.loop_enabled = False
    cfg.loop.arcratio_enabled = True
    cfg.loop.assignment_execution_cooldown_seconds = 180

    now = datetime.datetime(2026, 6, 5, 15, 0, 0, tzinfo=datetime.timezone.utc)
    validator = Validator(
        config=cfg,
        store=None,
        bt_objects=_bt_objects(),
        metadata_manager=None,
        clock=lambda: now,
    )
    cfg.validator.orchestrator_api_url = ORCHESTRATOR_API_URL
    validator._last_assignment_execution_at = now - datetime.timedelta(seconds=181)

    fake_loaded = SimpleNamespace(
        keypair=_FakeKeypair(),
        hotkey_ss58="5ValidatorHotkey",
        coldkey_ss58=None,
    )
    monkeypatch.setattr("src.validator.validator.load_hotkey", lambda _cfg: fake_loaded)

    fetch_calls: list[dict] = []
    monkeypatch.setattr(
        "src.validator.validator.fetch_agent_event_assignment",
        lambda **kw: fetch_calls.append(kw) or AgentAndEventResponse(
            assignment=None,
            reason="none_available",
        ),
    )

    validator._poll_orchestrator_assignment()
    assert len(fetch_calls) == 1
