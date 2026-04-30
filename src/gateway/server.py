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


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "providers": ",".join(sorted(_providers.keys())),
        "require_signature": _require_signature(),
    }


@app.get("/providers")
def list_providers() -> dict[str, list[str]]:
    return {"providers": sorted(_providers.keys())}


@app.post("/v1/call", response_model=CallResponse)
async def proxy_call(request: Request) -> CallResponse:
    body_bytes = await request.body()
    try:
        req = CallRequest.model_validate_json(body_bytes)
    except Exception as exc:  # pydantic ValidationError or JSON parse error
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Best-effort verification — logged only, never enforced in v1 unless
    # REQUIRE_SIGNATURE is set (intended for a follow-up rollout phase).
    verification = verify_request_headers(request.headers, body_bytes)
    if _require_signature() and not verification.verified:
        raise HTTPException(
            status_code=401,
            detail=f"signature required: {verification.reason or 'invalid'}",
        )

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
        raise HTTPException(status_code=502, detail=str(exc)) from exc
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
