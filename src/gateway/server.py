"""Gateway HTTP server — the provider proxy service.

Runs as a standalone process. Holds API keys, registers all provider adapters,
and exposes a single endpoint for the validator to call through.

The validator never touches external APIs directly — everything flows through
this gateway, which is the only component that needs secrets.

Signature headers (`X-Validator-Hotkey`, `X-Signature`, etc.) are logged on
every request. In v1 the gateway *does not* enforce them — verification will
be enabled (with metagraph membership checks against our netuid) in a
follow-up phase. The wire format is finalised today; only the policy is
permissive.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from src.gateway.providers.base import BaseProvider
from src.gateway.signing import verify_request_headers

logger = logging.getLogger("arcratio.gateway")

app = FastAPI(title="Arcratio Provider Gateway", version="0.1.0")

# Max accepted request body for /v1/call. The body is buffered fully in
# memory before parsing, so without a cap a single large POST can OOM the
# gateway. 1 MiB is far above any legitimate provider-call payload.
MAX_BODY_BYTES = 1_048_576

_providers: dict[str, BaseProvider] = {}


class CallRequest(BaseModel):
    provider_id: str
    call_type: str
    params: dict[str, Any] = {}


class CallResponse(BaseModel):
    provider_id: str
    call_type: str
    data: dict[str, Any]
    latency_ms: int


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
        models: list[str] = []
        allows_any_model = False
        if provider_id == "claude":
            models = ["claude-sonnet-4-6"]
        if provider_id == "openrouter":
            allows_any_model = True
        rows.append(
            {
                "id": provider_id,
                "models": models,
                "allowsAnyModel": allows_any_model,
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
