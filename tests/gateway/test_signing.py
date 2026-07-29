"""Unit tests for the validator-side signing module + gateway logging path.

Skipped automatically when neither `bittensor` nor `substrateinterface` is
installed — those provide the sr25519 `Keypair`.
The runner image deliberately omits all of them, so these tests only run
on the validator/dev side.
"""

from __future__ import annotations

import base64
import time

import pytest
from fastapi.testclient import TestClient

from src.gateway import server as gateway_server
from src.gateway.providers.base import BaseProvider
from src.gateway.signing import (
    AUTH_VERSION,
    HEADER_AUTH_VERSION,
    HEADER_AUTHORIZATION,
    HEADER_HOTKEY,
    HEADER_NETUID,
    HEADER_NONCE,
    HEADER_SCHEME,
    HEADER_SIGNATURE,
    HEADER_TIMESTAMP,
    LoadedKeypair,
    SIGNATURE_SCHEME,
    sign_request,
    verify_request_headers,
)


# Runs after src.gateway.signing imports — pick whichever Keypair package is
# available. Skip the whole module if none are.
def _import_keypair():
    try:
        from bittensor.sp_core import Keypair  # type: ignore

        return Keypair
    except ImportError:
        pass
    try:
        from substrateinterface import Keypair  # type: ignore

        return Keypair
    except ImportError:
        return None


Keypair = _import_keypair()
pytestmark = pytest.mark.skipif(
    Keypair is None,
    reason="no sr25519 Keypair backend installed (bittensor / substrate-interface)",
)


@pytest.fixture
def keypair():
    """Throwaway sr25519 keypair for the test."""
    # SR25519 is the default on every supported backend.
    return Keypair.create_from_mnemonic(Keypair.generate_mnemonic())


@pytest.fixture
def loaded(keypair):
    return LoadedKeypair(
        keypair=keypair,
        hotkey_ss58=keypair.ss58_address,
        coldkey_ss58=None,
    )


# ---------------------------------------------------------------------------
# sign_request / verify_request_headers
# ---------------------------------------------------------------------------


def test_auth_version_uses_almanac_namespace():
    assert AUTH_VERSION == "almanac-v1"


class TestSignRequest:
    def test_returns_full_header_set(self, loaded):
        body = b'{"provider_id":"polymarket","call_type":"get_market","params":{}}'
        headers = sign_request(loaded, body, netuid=42)
        assert headers[HEADER_AUTH_VERSION] == AUTH_VERSION
        assert headers[HEADER_SCHEME] == SIGNATURE_SCHEME
        assert headers[HEADER_HOTKEY] == loaded.hotkey_ss58
        assert headers[HEADER_NETUID] == "42"
        assert int(headers[HEADER_TIMESTAMP]) > 0
        assert len(headers[HEADER_NONCE]) >= 8
        assert len(headers[HEADER_SIGNATURE]) > 32  # hex of 64-byte sig
        assert headers[HEADER_AUTHORIZATION].startswith("Basic ")

        token = headers[HEADER_AUTHORIZATION].split(" ", 1)[1]
        raw = base64.b64decode(token).decode("utf-8")
        username, password = raw.split(":", 1)
        assert username == loaded.hotkey_ss58
        assert password == headers[HEADER_SIGNATURE]

    def test_no_signing_returns_empty(self):
        # When signing is disabled we want a clean empty header dict — no
        # half-signed contraption that the central gateway would log as
        # `verified=False` for a hotkey we don't actually have.
        assert sign_request(None, b"{}", netuid=0) == {}


class TestVerifyRequestHeaders:
    def test_verifies_real_signature(self, loaded):
        body = b'{"hello":"world"}'
        headers = sign_request(loaded, body, netuid=1)
        result = verify_request_headers(headers, body)
        assert result.verified
        assert result.hotkey == loaded.hotkey_ss58
        assert result.netuid == 1

    def test_rejects_tampered_body(self, loaded):
        body = b'{"hello":"world"}'
        headers = sign_request(loaded, body, netuid=1)
        result = verify_request_headers(headers, b'{"hello":"WORLD"}')
        assert not result.verified
        assert result.reason == "bad_signature"

    def test_rejects_wrong_hotkey(self, loaded):
        body = b'{"hello":"world"}'
        headers = sign_request(loaded, body, netuid=1)
        # Replace hotkey ss58 with a different one — verifier will rebuild
        # canonical, see signature doesn't match the (different) public key,
        # and fail.
        other_kp = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
        headers = {**headers, HEADER_HOTKEY: other_kp.ss58_address}
        result = verify_request_headers(headers, body)
        assert not result.verified
        assert result.reason == "bad_signature"

    def test_rejects_replayed_timestamp(self, loaded):
        body = b'{"hello":"world"}'
        headers = sign_request(loaded, body, netuid=1)
        # Backdate timestamp beyond skew window. The signature was over the
        # ORIGINAL ts, so even ignoring skew the verification would fail —
        # the test name reflects the intent (replay defence) and we assert
        # the verifier flags either skew or signature.
        headers = {**headers, HEADER_TIMESTAMP: str(int(time.time()) - 3600)}
        result = verify_request_headers(headers, body, max_skew_seconds=60)
        assert not result.verified
        assert result.reason in {"timestamp_skew:3600s", "bad_signature"}

    def test_rejects_missing_signature(self):
        result = verify_request_headers({}, b"{}")
        assert not result.verified
        assert result.reason == "missing_auth_version"


# ---------------------------------------------------------------------------
# Central gateway logging path
# ---------------------------------------------------------------------------


class _StubProvider(BaseProvider):
    @property
    def provider_id(self) -> str:
        return "stub"

    @property
    def provider_tier(self):
        from src.core.schemas import ProviderTier

        return ProviderTier.INFERENCE

    def call(self, call_type, params):
        return {"echo": {"call_type": call_type, "params": params}}


@pytest.fixture
def gateway_app():
    # Reset the module-level provider registry between tests.
    gateway_server._providers.clear()
    gateway_server.register_provider(_StubProvider())
    return TestClient(gateway_server.app)


class TestGatewaySignatureLogging:
    def test_gateway_providers_endpoint_returns_capabilities(self, gateway_app):
        resp = gateway_app.get("/v1/gateway/providers")
        assert resp.status_code == 200
        body = resp.json()
        providers = body["providers"]
        assert isinstance(providers, list)
        stub = next(p for p in providers if p["id"] == "stub")
        assert "models" in stub
        assert "allowsAnyModel" in stub
        assert "defaultCallType" in stub
        assert "supportsCompletions" in stub

    def test_unsigned_request_accepted_when_not_required(self, gateway_app, monkeypatch):
        monkeypatch.delenv("REQUIRE_SIGNATURE", raising=False)
        resp = gateway_app.post(
            "/v1/call",
            json={"provider_id": "stub", "call_type": "ping", "params": {"x": 1}},
        )
        assert resp.status_code == 200
        assert resp.json()["data"] == {"echo": {"call_type": "ping", "params": {"x": 1}}}

    def test_signed_request_logs_verified_hotkey(self, gateway_app, loaded, caplog, monkeypatch):
        import json

        monkeypatch.delenv("REQUIRE_SIGNATURE", raising=False)
        body = json.dumps({"provider_id": "stub", "call_type": "ping", "params": {}}).encode()
        headers = sign_request(loaded, body, netuid=7)

        with caplog.at_level("INFO", logger="almanac.gateway"):
            resp = gateway_app.post("/v1/call", content=body, headers={**headers, "Content-Type": "application/json"})
        assert resp.status_code == 200
        # The gateway logs `... by <hotkey> (verified=True) ...`.
        log_text = " ".join(rec.getMessage() for rec in caplog.records)
        assert loaded.hotkey_ss58 in log_text
        assert "verified=True" in log_text

    def test_require_signature_rejects_unsigned(self, gateway_app, monkeypatch):
        monkeypatch.setenv("REQUIRE_SIGNATURE", "true")
        resp = gateway_app.post(
            "/v1/call",
            json={"provider_id": "stub", "call_type": "ping", "params": {}},
        )
        assert resp.status_code == 401

    def test_validator_completions_defaults_call_type_for_llm_provider(self, gateway_app, monkeypatch):
        monkeypatch.delenv("REQUIRE_SIGNATURE", raising=False)
        gateway_server._providers["openrouter"] = _StubProvider()
        resp = gateway_app.post(
            "/v1/gateway/validator/completions",
            headers={"x-miner-hotkey": "5MinerHotkey"},
            json={
                "provider": "openrouter",
                "model": "anthropic/claude-3.5-sonnet",
                "messages": [{"role": "system", "content": "Hello!"}],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["call_type"] == "chat_completion"

