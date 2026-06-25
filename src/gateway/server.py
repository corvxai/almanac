"""Gateway HTTP server — SIMULATED provider proxy service (local test double).

This is **not** the production gateway service (that lives in its own repo). It
is the local stand-in used so validators and miners can exercise the full
validator → local proxy → gateway → provider path. Launched via
``scripts/run_gateway.py``, which runs it mock-by-default (no keys, no network,
no spend) unless ``--live`` is passed.

Runs as a standalone process, registers provider adapters, and exposes a single
endpoint for the validator to call through. The validator never touches external
APIs directly — everything flows through this gateway, which is the only
component that needs secrets (and only in live mode).

Signature headers (`X-Validator-Hotkey`, `X-Signature`, etc.) are logged on
every request. In v1 the gateway *does not* enforce them — verification will
be enabled (with metagraph membership checks against our netuid) in a
follow-up phase. The wire format is finalised today; only the policy is
permissive. (Real enforcement is the production gateway's responsibility.)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from src.gateway.providers.base import BaseProvider
from src.gateway.signing import verify_request_headers
from src.gateway.validator_protocol import verify_validator_completions_request

logger = logging.getLogger("arcratio.gateway")

app = FastAPI(title="Arcratio Provider Gateway (Simulated)", version="0.1.0")

# Max accepted request body for /v1/call. The body is buffered fully in
# memory before parsing, so without a cap a single large POST can OOM the
# gateway. 1 MiB is far above any legitimate provider-call payload.
MAX_BODY_BYTES = 1_048_576

_providers: dict[str, BaseProvider] = {}
_seen_validator_nonces: dict[tuple[str, str], int] = {}
_NONCE_TTL_MS = 10 * 60 * 1000

_PROVIDER_CAPABILITIES: dict[str, dict[str, Any]] = {
    "anthropic": {
        "models": ["claude-sonnet-4-6"],
        "allowsAnyModel": False,
        "defaultCallType": "messages",
        "supportsCompletions": True,
    },
    "openrouter": {
        "models": [],
        "allowsAnyModel": True,
        "defaultCallType": "chat_completion",
        "supportsCompletions": True,
    },
    "openai": {
        "models": [],
        "allowsAnyModel": True,
        "defaultCallType": "chat_completion",
        "supportsCompletions": True,
    },
    "grok": {
        "models": [],
        "allowsAnyModel": True,
        "defaultCallType": "chat_completion",
        "supportsCompletions": True,
    },
    "gemini": {
        "models": [],
        "allowsAnyModel": True,
        "defaultCallType": "generate_content",
        "supportsCompletions": True,
    },
    "perplexity": {
        "models": [],
        "allowsAnyModel": True,
        "defaultCallType": "chat_completion",
        "supportsCompletions": True,
    },
    "polymarket": {
        "models": [],
        "allowsAnyModel": False,
        "defaultCallType": "get_market",
        "supportsCompletions": False,
    },
    "web_search": {
        "models": [],
        "allowsAnyModel": False,
        "defaultCallType": "search",
        "supportsCompletions": False,
    },
}


class CallRequest(BaseModel):
    provider_id: str
    call_type: str
    params: dict[str, Any] = {}


class CallResponse(BaseModel):
    provider_id: str
    call_type: str
    data: dict[str, Any]
    latency_ms: int


class ValidatorCompletionsRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")
    provider: str
    model: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    maxTokens: int | None = None
    temperature: float | None = None
    topP: float | None = None
    stop: list[str] = Field(default_factory=list)
    callType: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


def register_provider(provider: BaseProvider) -> None:
    _providers[provider.provider_id] = provider
    logger.info("Registered provider: %s", provider.provider_id)


def _require_signature() -> bool:
    raw = os.environ.get("REQUIRE_SIGNATURE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _expected_netuid() -> int | None:
    """The netuid this gateway serves, if pinned via env.

    When set (and signatures are required), a signed request whose claimed
    netuid does not match is rejected. This binds a hotkey's authorisation to
    *our* subnet — without it, a valid signature from a hotkey on any other
    subnet would pass. Unset (the default) preserves current behaviour.
    """
    raw = os.environ.get("GATEWAY_NETUID", "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("GATEWAY_NETUID=%r is not an integer; ignoring", raw)
        return None


@app.get("/health")
def health() -> dict[str, Any]:
    # Intentionally minimal: this endpoint is unauthenticated, so it must not
    # disclose configuration (provider list, signature policy) to callers.
    return {"status": "ok"}


@app.get("/providers")
def list_providers() -> dict[str, list[str]]:
    return {"providers": sorted(_providers.keys())}


@app.get("/v1/gateway/providers")
def list_gateway_providers() -> dict[str, list[dict[str, Any]]]:
    provider_ids = sorted(_providers.keys())
    rows: list[dict[str, Any]] = []
    for provider_id in provider_ids:
        caps = _PROVIDER_CAPABILITIES.get(
            provider_id,
            {
                "models": [],
                "allowsAnyModel": True,
                "defaultCallType": "chat_completion",
                "supportsCompletions": True,
            },
        )
        rows.append(
            {
                "id": provider_id,
                "models": list(caps["models"]),
                "allowsAnyModel": bool(caps["allowsAnyModel"]),
                "defaultCallType": str(caps["defaultCallType"]),
                "supportsCompletions": bool(caps["supportsCompletions"]),
            }
        )
    return {"providers": rows}


@app.post("/v1/call", response_model=CallResponse)
async def proxy_call(request: Request) -> CallResponse:
    # Reject oversized bodies before buffering/parsing. Trust the declared
    # Content-Length when present; otherwise enforce against the read bytes.
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > MAX_BODY_BYTES:
                raise HTTPException(status_code=413, detail="request body too large")
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid Content-Length") from None

    body_bytes = await request.body()
    if len(body_bytes) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="request body too large")

    try:
        req = CallRequest.model_validate_json(body_bytes)
    except Exception as exc:  # pydantic ValidationError or JSON parse error
        # Log the detail server-side; return a generic message so we don't
        # echo schema internals / raw input back to unauthenticated callers.
        logger.info("rejected malformed /v1/call body: %s", exc)
        raise HTTPException(status_code=422, detail="invalid request body") from exc

    # Best-effort verification — logged only, never enforced in v1 unless
    # REQUIRE_SIGNATURE is set (intended for a follow-up rollout phase).
    verification = verify_request_headers(request.headers, body_bytes)
    if _require_signature():
        if not verification.verified:
            raise HTTPException(
                status_code=401,
                detail=f"signature required: {verification.reason or 'invalid'}",
            )
        expected_netuid = _expected_netuid()
        if expected_netuid is not None and verification.netuid != expected_netuid:
            logger.warning(
                "rejected signed call: netuid %s != expected %s (hotkey=%s)",
                verification.netuid,
                expected_netuid,
                verification.hotkey,
            )
            raise HTTPException(status_code=401, detail="signature netuid mismatch")

    provider = _providers.get(req.provider_id)
    if provider is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown provider '{req.provider_id}'. "
                   f"Available: {sorted(_providers.keys())}",
        )

    t0 = time.monotonic()
    try:
        data = provider.call(req.call_type, req.params)
    except Exception as exc:
        # Provider exceptions may embed raw upstream response bodies (which can
        # carry rate-limit headers, internal identifiers, or — worst case —
        # secrets). Log the full error server-side; return only the provider id.
        logger.warning(
            "provider call failed: provider=%s call_type=%s error=%s",
            req.provider_id,
            req.call_type,
            exc,
        )
        raise HTTPException(
            status_code=502,
            detail=f"upstream provider '{req.provider_id}' call failed",
        ) from exc
    latency_ms = int((time.monotonic() - t0) * 1000)

    actor = verification.hotkey or "<unsigned>"
    logger.info(
        "%s.%s by %s (verified=%s) → %d bytes, %dms",
        req.provider_id,
        req.call_type,
        actor,
        verification.verified,
        len(str(data)),
        latency_ms,
    )

    return CallResponse(
        provider_id=req.provider_id,
        call_type=req.call_type,
        data=data,
        latency_ms=latency_ms,
    )


@app.post("/v1/gateway/validator/completions")
async def validator_completions(request: Request) -> dict[str, Any]:
    body_bytes = await request.body()
    try:
        req = ValidatorCompletionsRequest.model_validate_json(body_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    miner_hotkey_header = _header(request.headers, "x-miner-hotkey")
    if not miner_hotkey_header:
        raise HTTPException(status_code=401, detail="missing x-miner-hotkey")
    if isinstance(req.params, dict) and "minerHotkey" in req.params:
        raise HTTPException(status_code=400, detail="minerHotkey is not allowed in body params")
    try:
        raw_obj = await request.json()
        if isinstance(raw_obj, dict) and "minerHotkey" in raw_obj:
            raise HTTPException(status_code=400, detail="minerHotkey is forbidden in body")
    except HTTPException:
        raise
    except Exception:
        # JSON parse already validated above through pydantic; ignore defensive parse failures.
        pass

    verification = verify_validator_completions_request(request.headers, body_bytes)
    if _require_signature() and not verification.verified:
        raise HTTPException(
            status_code=401,
            detail=verification.reason or "invalid signature",
        )
    _check_replay(verification)

    provider = _providers.get(req.provider)
    if provider is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown provider '{req.provider}'. Available: {sorted(_providers.keys())}",
        )

    caps = _PROVIDER_CAPABILITIES.get(
        req.provider,
        {
            "defaultCallType": "chat_completion",
            "supportsCompletions": True,
        },
    )
    supports_completions = bool(caps.get("supportsCompletions", True))

    params = req.params if isinstance(req.params, dict) else {}
    if not params:
        params = {
            "model": req.model,
            "messages": req.messages,
            "max_tokens": req.maxTokens,
            "temperature": req.temperature,
            "top_p": req.topP,
            "stop": req.stop,
        }
        params = {k: v for k, v in params.items() if v is not None}
    if not supports_completions and not req.callType:
        raise HTTPException(
            status_code=400,
            detail=f"provider '{req.provider}' requires callType/params in validator completions payload",
        )

    call_type = req.callType or str(caps.get("defaultCallType") or _infer_default_call_type(req.provider))
    t0 = time.monotonic()
    try:
        data = provider.call(call_type, params)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    latency_ms = int((time.monotonic() - t0) * 1000)
    return {
        "provider_id": req.provider,
        "call_type": call_type,
        "data": data,
        "latency_ms": latency_ms,
    }


def _infer_default_call_type(provider_id: str) -> str:
    defaults = {
        "anthropic": "messages",
        "openrouter": "chat_completion",
        "openai": "chat_completion",
        "grok": "chat_completion",
        "gemini": "generate_content",
        "perplexity": "chat_completion",
        "polymarket": "get_market",
        "web_search": "search",
    }
    return defaults.get(provider_id, "chat_completion")


def _check_replay(verification) -> None:  # noqa: ANN001
    if not verification.verified:
        return
    if not verification.validator_hotkey or not verification.nonce:
        return
    now_ms = int(time.time() * 1000)
    cutoff = now_ms - _NONCE_TTL_MS
    stale_keys = [k for k, ts in _seen_validator_nonces.items() if ts < cutoff]
    for key in stale_keys:
        _seen_validator_nonces.pop(key, None)
    key = (verification.validator_hotkey, verification.nonce)
    if key in _seen_validator_nonces:
        raise HTTPException(status_code=401, detail="replayed nonce")
    _seen_validator_nonces[key] = verification.timestamp_ms or now_ms


def _header(headers: Any, name: str) -> str | None:
    lower = name.lower()
    for k, v in headers.items():
        if str(k).lower() == lower:
            return str(v)
    return None
