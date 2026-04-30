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
                "body": body,
            }
        )
        return httpx.Response(
            200,
            json={
                "provider_id": body["provider_id"],
                "call_type": body["call_type"],
                "data": {"echo": body["params"]},
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
            json={"provider_id": "claude", "call_type": "messages", "params": {}},
            headers={"X-Run-Id": str(run_id)},
        )
        assert resp.status_code == 403
        assert "claude" in resp.json()["detail"]

    def test_main_track_allows_everything(self, proxy_client):
        client, state, _ = proxy_client
        run_id = uuid4()
        state.register_run(run_id, track="MAIN")
        resp = client.post(
            "/v1/call",
            json={"provider_id": "claude", "call_type": "messages", "params": {"x": 1}},
            headers={"X-Run-Id": str(run_id)},
        )
        assert resp.status_code == 200


class TestForwardingAndRecording:
    def test_call_is_forwarded_and_recorded(self, proxy_client):
        client, state, captured = proxy_client
        run_id = uuid4()
        state.register_run(run_id, track="MAIN")

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
        assert upstream["body"] == {
            "provider_id": "polymarket",
            "call_type": "get_market",
            "params": {"slug": "x"},
        }

        # The proxy records the call against the run.
        calls = state.pop_calls(run_id)
        assert len(calls) == 1
        assert calls[0].provider_id == "polymarket"
        assert calls[0].call_type == "get_market"
        assert calls[0].call_index == 0

    def test_pop_calls_deregisters_run(self, proxy_client):
        client, state, _ = proxy_client
        run_id = uuid4()
        state.register_run(run_id, track="MAIN")
        # Drain — re-using the run_id should now 401.
        state.pop_calls(run_id)
        resp = client.post(
            "/v1/call",
            json={"provider_id": "polymarket", "call_type": "get_market", "params": {}},
            headers={"X-Run-Id": str(run_id)},
        )
        assert resp.status_code == 401


class TestHealth:
    def test_reports_signing_disabled(self, proxy_client):
        client, _state, _ = proxy_client
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["signing_enabled"] is False
        assert body["netuid"] == 99
        assert body["hotkey"] is None
