"""Unit tests for the validator-local signing proxy.

These exercise the FastAPI app via `TestClient`, with the upstream central
gateway stubbed by an `httpx.MockTransport`. No Docker, no UDS — just the
FastAPI request/response shape and the per-run state machine.
"""

from __future__ import annotations

import json
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from src.core.config import AppConfig
from src.gateway.local_proxy import LocalProxyState, create_app


@pytest.fixture
def cfg():
    cfg = AppConfig.load_default()
    cfg.bittensor.signing_required = False  # avoid loading a real wallet
    cfg.bittensor.netuid = 99
    return cfg


@pytest.fixture
def upstream_handler():
    """Records every request the proxy forwards upstream and returns a stub
    `data` payload echoing the request shape."""
    captured = {"requests": []}

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured["requests"].append(
            {
                "headers": dict(request.headers),
                "url": str(request.url),
                "body": body,
            }
        )
        return httpx.Response(
            200,
            json={
                "provider_id": body["provider"],
                "call_type": body.get("callType") or "unknown",
                "data": {"echo": body.get("params", {})},
                "latency_ms": 0,
            },
        )

    return captured, _handler


@pytest.fixture
def proxy_client(cfg, upstream_handler):
    captured, handler = upstream_handler
    transport = httpx.MockTransport(handler)
    upstream_client = httpx.Client(transport=transport, timeout=5.0)
    app = create_app(cfg, http_client=upstream_client)
    state: LocalProxyState = app.state.proxy
    state.load_hotkey()
    with TestClient(app) as client:
        yield client, state, captured


class TestRunRegistry:
    def test_unregistered_run_id_rejected(self, proxy_client):
        client, _state, _captured = proxy_client
        resp = client.post(
            "/v1/call",
            json={"provider_id": "polymarket", "call_type": "get_market", "params": {}},
            headers={"X-Run-Id": str(uuid4())},
        )
        assert resp.status_code == 401
        assert "unknown run_id" in resp.json()["detail"]

    def test_missing_run_id_header_rejected(self, proxy_client):
        client, _state, _captured = proxy_client
        resp = client.post(
            "/v1/call",
            json={"provider_id": "polymarket", "call_type": "get_market", "params": {}},
        )
        assert resp.status_code == 401
        assert "missing X-Run-Id" in resp.json()["detail"]


class TestTrackAllowlist:
    def test_signal_track_blocks_disallowed_provider(self, proxy_client):
        client, state, _ = proxy_client
        run_id = uuid4()
        state.register_run(run_id, track="SIGNAL")
        resp = client.post(
            "/v1/call",
            json={"provider_id": "anthropic", "call_type": "messages", "params": {}},
            headers={"X-Run-Id": str(run_id)},
        )
        assert resp.status_code == 403
        assert "anthropic" in resp.json()["detail"]

    def test_main_track_allows_everything(self, proxy_client):
        client, state, _ = proxy_client
        run_id = uuid4()
        state.register_run(run_id, track="MAIN", miner_hotkey="5MinerHotkey")
        resp = client.post(
            "/v1/call",
            json={"provider_id": "anthropic", "call_type": "messages", "params": {"x": 1}},
            headers={"X-Run-Id": str(run_id)},
        )
        assert resp.status_code == 200


class TestForwardingAndRecording:
    def test_call_is_forwarded_and_recorded(self, proxy_client):
        client, state, captured = proxy_client
        run_id = uuid4()
        state.register_run(run_id, track="MAIN", miner_hotkey="5MinerHotkey")

        resp = client.post(
            "/v1/call",
            json={"provider_id": "polymarket", "call_type": "get_market", "params": {"slug": "x"}},
            headers={"X-Run-Id": str(run_id)},
        )
        assert resp.status_code == 200
        assert resp.json()["data"] == {"echo": {"slug": "x"}}

        # Upstream saw exactly one request.
        assert len(captured["requests"]) == 1
        upstream = captured["requests"][0]
        assert upstream["url"].endswith("/v1/gateway/validator/completions")
        assert upstream["body"]["provider"] == "polymarket"
        assert upstream["body"]["callType"] == "get_market"
        assert upstream["body"]["params"] == {"slug": "x"}
        assert "minerHotkey" not in upstream["body"]
        assert upstream["headers"]["x-miner-hotkey"] == "5MinerHotkey"

        # The proxy records the call against the run.
        calls = state.pop_calls(run_id)
        assert len(calls) == 1
        assert calls[0].provider_id == "polymarket"
        assert calls[0].call_type == "get_market"
        assert calls[0].call_index == 0

    def test_pop_calls_deregisters_run(self, proxy_client):
        client, state, _ = proxy_client
        run_id = uuid4()
        state.register_run(run_id, track="MAIN", miner_hotkey="5MinerHotkey")
        # Drain — re-using the run_id should now 401.
        state.pop_calls(run_id)
        resp = client.post(
            "/v1/call",
            json={"provider_id": "polymarket", "call_type": "get_market", "params": {}},
            headers={"X-Run-Id": str(run_id)},
        )
        assert resp.status_code == 401

    def test_llm_provider_payload_omits_compat_fields(self, proxy_client):
        client, state, captured = proxy_client
        run_id = uuid4()
        state.register_run(run_id, track="MAIN", miner_hotkey="5MinerHotkey")

        resp = client.post(
            "/v1/call",
            json={
                "provider_id": "anthropic",
                "call_type": "messages",
                "params": {
                    "model": "claude-sonnet-4-6",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            },
            headers={"X-Run-Id": str(run_id)},
        )
        assert resp.status_code == 200
        upstream = captured["requests"][-1]["body"]
        assert upstream["provider"] == "anthropic"
        assert "callType" not in upstream
        assert "params" not in upstream
        assert "temperature" not in upstream
        assert "topP" not in upstream
        assert "maxTokens" not in upstream

    def test_new_completions_route_forwards_and_records(self, proxy_client):
        client, state, captured = proxy_client
        run_id = uuid4()
        state.register_run(run_id, track="MAIN", miner_hotkey="5MinerHotkey")

        resp = client.post(
            "/v1/gateway/validator/completions",
            json={
                "provider": "openrouter",
                "model": "anthropic/claude-3.5-sonnet",
                "messages": [{"role": "system", "content": "Hello!"}],
                "minerHotkey": "5MinerHotkey",
            },
            headers={"X-Run-Id": str(run_id)},
        )
        assert resp.status_code == 200
        assert len(captured["requests"]) == 1
        upstream = captured["requests"][0]
        assert upstream["url"].endswith("/v1/gateway/validator/completions")
        assert upstream["body"]["provider"] == "openrouter"
        assert "minerHotkey" not in upstream["body"]
        assert upstream["headers"]["x-miner-hotkey"] == "5MinerHotkey"
        calls = state.pop_calls(run_id)
        assert len(calls) == 1
        assert calls[0].provider_id == "openrouter"

    def test_body_miner_hotkey_cannot_override_registered_identity(self, proxy_client):
        """A malicious sandbox cannot choose whose hotkey the validator signs.

        Regression for the billing/attribution spoofing hole: the signed
        `x-miner-hotkey` must come from the run registration, never the body.
        """
        client, state, captured = proxy_client
        run_id = uuid4()
        state.register_run(run_id, track="MAIN", miner_hotkey="5RealMiner")

        # /v1/call — attacker plants minerHotkey at both body and params levels.
        resp = client.post(
            "/v1/call",
            json={
                "provider_id": "anthropic",
                "call_type": "messages",
                "minerHotkey": "5AttackerTopLevel",
                "params": {"model": "claude-sonnet-4-6", "minerHotkey": "5AttackerInParams"},
            },
            headers={"X-Run-Id": str(run_id)},
        )
        assert resp.status_code == 200
        upstream = captured["requests"][-1]
        assert upstream["headers"]["x-miner-hotkey"] == "5RealMiner"
        assert "minerHotkey" not in upstream["body"]

        # /v1/gateway/validator/completions — same attempt on the other route.
        resp = client.post(
            "/v1/gateway/validator/completions",
            json={
                "provider": "openrouter",
                "model": "anthropic/claude-3.5-sonnet",
                "messages": [{"role": "system", "content": "hi"}],
                "minerHotkey": "5AttackerCompletions",
            },
            headers={"X-Run-Id": str(run_id)},
        )
        assert resp.status_code == 200
        upstream = captured["requests"][-1]
        assert upstream["headers"]["x-miner-hotkey"] == "5RealMiner"
        assert "minerHotkey" not in upstream["body"]

    def test_missing_miner_hotkey_context_rejected(self, proxy_client):
        client, state, _captured = proxy_client
        run_id = uuid4()
        state.register_run(run_id, track="MAIN")
        resp = client.post(
            "/v1/call",
            json={"provider_id": "anthropic", "call_type": "messages", "params": {"x": 1}},
            headers={"X-Run-Id": str(run_id)},
        )
        assert resp.status_code == 400
        assert "missing miner hotkey" in resp.json()["detail"]


class TestHealth:
    def test_reports_signing_disabled(self, proxy_client):
        client, _state, _ = proxy_client
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["signing_enabled"] is False
        assert body["netuid"] == 99
        assert body["hotkey"] is None
