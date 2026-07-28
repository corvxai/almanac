from __future__ import annotations

import pytest

from src.gateway.signing import LoadedKeypair
from src.gateway.validator_protocol import (
    HEADER_MINER_HOTKEY,
    HEADER_VALIDATOR_HOTKEY,
    HEADER_VALIDATOR_NONCE,
    HEADER_VALIDATOR_SIGNATURE,
    HEADER_VALIDATOR_TIMESTAMP,
    canonical_json,
    sign_validator_completions_request,
    verify_validator_completions_request,
)


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
    reason="no sr25519 Keypair backend installed",
)


@pytest.fixture
def loaded():
    # SR25519 is the default on every supported backend.
    kp = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    return LoadedKeypair(keypair=kp, hotkey_ss58=kp.ss58_address, coldkey_ss58=None)


def test_canonical_json_matches_js_number_formatting() -> None:
    # The gateway verifies signatures over canonicalJson(JSON.parse(body)), so our
    # canonical encoding must match JS JSON.stringify: integral floats lose the
    # decimal, object keys are sorted, non-finite -> null. Regression for the
    # confidence=0.0 -> "0" mismatch that rejected validator prediction submits.
    assert canonical_json(0.0) == "0"
    assert canonical_json(1.0) == "1"
    assert canonical_json(0.5) == "0.5"
    assert canonical_json(-0.0) == "0"
    assert canonical_json({"b": 0.0, "a": 1}) == '{"a":1,"b":0}'
    assert canonical_json([True, False, None]) == "[true,false,null]"
    assert canonical_json(float("nan")) == "null"
    assert canonical_json(float("inf")) == "null"


def test_sign_and_verify_roundtrip(loaded) -> None:
    payload = {"provider": "openrouter", "messages": [{"role": "system", "content": "Hello"}]}
    body = canonical_json(payload).encode("utf-8")
    headers = sign_validator_completions_request(
        loaded=loaded,
        miner_hotkey="5MinerHotkey",
        payload=payload,
        timestamp_ms=1_700_000_000_000,
        nonce="nonce123",
    )
    assert headers[HEADER_VALIDATOR_HOTKEY] == loaded.hotkey_ss58
    assert headers[HEADER_MINER_HOTKEY] == "5MinerHotkey"
    assert headers[HEADER_VALIDATOR_TIMESTAMP] == "1700000000000"
    assert headers[HEADER_VALIDATOR_NONCE] == "nonce123"
    assert not headers[HEADER_VALIDATOR_SIGNATURE].startswith("0x")

    got = verify_validator_completions_request(headers, body, max_skew_ms=10**15)
    assert got.verified is True
    assert got.validator_hotkey == loaded.hotkey_ss58
    assert got.miner_hotkey == "5MinerHotkey"


def test_verify_rejects_tampered_body(loaded) -> None:
    payload = {"provider": "openrouter", "messages": [{"role": "system", "content": "Hello"}]}
    body = canonical_json(payload).encode("utf-8")
    headers = sign_validator_completions_request(
        loaded=loaded,
        miner_hotkey="5MinerHotkey",
        payload=payload,
        timestamp_ms=1_700_000_000_000,
        nonce="nonce123",
    )
    got = verify_validator_completions_request(
        headers,
        b'{"provider":"openrouter","messages":[{"role":"system","content":"Hacked"}]}',
        max_skew_ms=10**15,
    )
    assert got.verified is False
    assert got.reason == "bad_signature"
