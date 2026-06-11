"""Production-like validator gateway protocol helpers."""

from __future__ import annotations

import hashlib
import json
import math
import secrets
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from src.gateway.signing import LoadedKeypair

HEADER_VALIDATOR_HOTKEY = "x-validator-hotkey"
HEADER_VALIDATOR_SIGNATURE = "x-validator-signature"
HEADER_VALIDATOR_NONCE = "x-validator-nonce"
HEADER_VALIDATOR_TIMESTAMP = "x-validator-timestamp"
HEADER_MINER_HOTKEY = "x-miner-hotkey"
COMPLETIONS_ENDPOINT = "/v1/gateway/validator/completions"


def _canonical_message(
    *,
    validator_hotkey: str,
    miner_hotkey: str,
    method: str,
    path_and_query: str,
    timestamp_ms: int,
    nonce: str,
    body_canonical_json: str,
) -> bytes:
    body_hash = hashlib.sha256(body_canonical_json.encode("utf-8")).hexdigest()
    return "\n".join(
        [
            "sub41-gateway-v1",
            method.upper(),
            path_and_query,
            miner_hotkey,
            nonce,
            str(timestamp_ms),
            body_hash,
        ]
    ).encode("utf-8")


def sign_validator_completions_request(
    *,
    loaded: Optional[LoadedKeypair],
    miner_hotkey: str,
    payload: Any,
    method: str = "POST",
    path_and_query: str = COMPLETIONS_ENDPOINT,
    timestamp_ms: int | None = None,
    nonce: str | None = None,
) -> dict[str, str]:
    if loaded is None:
        return {}
    ts = int(time.time() * 1000) if timestamp_ms is None else int(timestamp_ms)
    n = nonce or secrets.token_hex(16)
    body_canonical = canonical_json(payload)
    canonical = _canonical_message(
        validator_hotkey=loaded.hotkey_ss58,
        miner_hotkey=miner_hotkey,
        method=method,
        path_and_query=path_and_query,
        timestamp_ms=ts,
        nonce=n,
        body_canonical_json=body_canonical,
    )
    sig = loaded.keypair.sign(canonical)
    sig_hex = sig.hex() if isinstance(sig, (bytes, bytearray)) else str(sig)
    return {
        HEADER_VALIDATOR_HOTKEY: loaded.hotkey_ss58,
        HEADER_VALIDATOR_SIGNATURE: sig_hex,
        HEADER_VALIDATOR_NONCE: n,
        HEADER_VALIDATOR_TIMESTAMP: str(ts),
        HEADER_MINER_HOTKEY: miner_hotkey,
    }


@dataclass(frozen=True)
class ValidatorCompletionsVerification:
    verified: bool
    validator_hotkey: Optional[str]
    miner_hotkey: Optional[str]
    timestamp_ms: Optional[int]
    nonce: Optional[str]
    reason: Optional[str] = None


def verify_validator_completions_request(
    headers: Mapping[str, str],
    body: bytes,
    *,
    method: str = "POST",
    path_and_query: str = COMPLETIONS_ENDPOINT,
    max_skew_ms: int = 300_000,
) -> ValidatorCompletionsVerification:
    validator_hotkey = _get_header(headers, HEADER_VALIDATOR_HOTKEY)
    miner_hotkey = _get_header(headers, HEADER_MINER_HOTKEY)
    signature_hex = _get_header(headers, HEADER_VALIDATOR_SIGNATURE)
    nonce = _get_header(headers, HEADER_VALIDATOR_NONCE)
    ts_raw = _get_header(headers, HEADER_VALIDATOR_TIMESTAMP)

    if not all([validator_hotkey, miner_hotkey, signature_hex, nonce, ts_raw]):
        return ValidatorCompletionsVerification(
            verified=False,
            validator_hotkey=validator_hotkey,
            miner_hotkey=miner_hotkey,
            timestamp_ms=_safe_int(ts_raw),
            nonce=nonce,
            reason="missing_validator_header",
        )

    ts = _safe_int(ts_raw)
    if ts is None:
        return ValidatorCompletionsVerification(
            verified=False,
            validator_hotkey=validator_hotkey,
            miner_hotkey=miner_hotkey,
            timestamp_ms=None,
            nonce=nonce,
            reason="bad_timestamp",
        )

    skew = abs(int(time.time() * 1000) - ts)
    if skew > max_skew_ms:
        return ValidatorCompletionsVerification(
            verified=False,
            validator_hotkey=validator_hotkey,
            miner_hotkey=miner_hotkey,
            timestamp_ms=ts,
            nonce=nonce,
            reason=f"timestamp_skew:{skew}ms",
        )

    try:
        body_obj = json.loads(body.decode("utf-8"))
        body_canonical = canonical_json(body_obj)
    except Exception:
        return ValidatorCompletionsVerification(
            verified=False,
            validator_hotkey=validator_hotkey,
            miner_hotkey=miner_hotkey,
            timestamp_ms=ts,
            nonce=nonce,
            reason="invalid_json_body",
        )

    try:
        Keypair = _load_keypair_class()
        kp = Keypair(ss58_address=validator_hotkey, crypto_type=_sr25519_crypto_type())
        canonical = _canonical_message(
            validator_hotkey=validator_hotkey or "",
            miner_hotkey=miner_hotkey or "",
            method=method,
            path_and_query=path_and_query,
            timestamp_ms=ts,
            nonce=nonce or "",
            body_canonical_json=body_canonical,
        )
        sig = signature_hex[2:] if signature_hex.lower().startswith("0x") else signature_hex
        ok = kp.verify(canonical, bytes.fromhex(sig))
    except Exception as exc:  # noqa: BLE001
        return ValidatorCompletionsVerification(
            verified=False,
            validator_hotkey=validator_hotkey,
            miner_hotkey=miner_hotkey,
            timestamp_ms=ts,
            nonce=nonce,
            reason=f"verify_error:{type(exc).__name__}",
        )

    if not ok:
        return ValidatorCompletionsVerification(
            verified=False,
            validator_hotkey=validator_hotkey,
            miner_hotkey=miner_hotkey,
            timestamp_ms=ts,
            nonce=nonce,
            reason="bad_signature",
        )

    return ValidatorCompletionsVerification(
        verified=True,
        validator_hotkey=validator_hotkey,
        miner_hotkey=miner_hotkey,
        timestamp_ms=ts,
        nonce=nonce,
    )


def _get_header(headers: Mapping[str, str], name: str) -> Optional[str]:
    if name in headers:
        return headers[name]
    lower = name.lower()
    for k, v in headers.items():
        if k.lower() == lower:
            return v
    return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_keypair_class():
    try:
        from bittensor_wallet.keypair import Keypair  # type: ignore

        return Keypair
    except ImportError:
        pass
    try:
        from substrateinterface import Keypair  # type: ignore

        return Keypair
    except ImportError:
        pass
    from bittensor import Keypair  # type: ignore

    return Keypair


def _sr25519_crypto_type() -> int:
    return 1


def _js_number(x: float) -> str:
    """Render a float the way JS ``JSON.stringify`` does.

    The gateway verifies signatures over ``canonicalJson(JSON.parse(body))``, so
    our canonical encoding must match JS number formatting. The key divergence
    is integral floats: Python renders ``0.0`` as ``"0.0"`` while JS renders
    ``0`` (e.g. the missing-confidence sentinel ``0.0``). Non-finite values
    become ``null``, matching ``JSON.stringify``.
    """
    if not math.isfinite(x):
        return "null"
    if x.is_integer():
        return str(int(x))
    return repr(x)


def canonical_json(value: Any) -> str:
    # bool before int/float (bool is an int subclass); float before the generic
    # json.dumps so integral floats match JS. int/str/None fall through to
    # json.dumps, which already matches JS for those types.
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return _js_number(value)
    if value is None or not isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ",".join(canonical_json(v) for v in value) + "]"
    parts: list[str] = []
    for k in sorted(value.keys(), key=lambda x: str(x)):
        parts.append(f"{json.dumps(str(k), ensure_ascii=False)}:{canonical_json(value[k])}")
    return "{" + ",".join(parts) + "}"
