"""Validator-local signing proxy.

Runs inside the validator container. Listens on a UNIX domain socket; the
agent sandbox bind-mounts that socket read-only and POSTs every provider call
through it. The proxy:

1. Validates the `X-Run-Id` header against runs registered by the orchestrator.
2. Enforces the run's per-track provider allowlist (see `track_config.py`).
3. Signs the outbound request to the central gateway with the validator's
   Bittensor hotkey (sr25519, see `signing.py`).
4. Forwards the request, parses the central gateway's response, and builds a
   full `ProviderCall` record that the orchestrator can drain after the run.

The hotkey is loaded once at startup. It lives only in this process's memory
and is never serialised, logged, or forwarded into the sandbox.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Optional
from uuid import UUID

import httpx
from fastapi import FastAPI, Header, HTTPException, Request

from src.core.config import AppConfig
from src.core.schemas import ProviderCall, ProviderTier
from src.gateway.client import _DEFAULT_TIERS, optional_completions_fields
from src.gateway.constants import gateway_service_url
from src.gateway.gateway import build_provider_call_record, default_summarise_params
from src.gateway.signing import LoadedKeypair, load_hotkey
from src.gateway.track_config import is_provider_allowed
from src.gateway.validator_protocol import canonical_json, sign_validator_completions_request

logger = logging.getLogger("arcratio.local_proxy")
_NON_COMPLETIONS_PROVIDERS = {"polymarket", "web_search"}


# ---------------------------------------------------------------------------
# In-memory run registry
# ---------------------------------------------------------------------------


class _RunState:
    """Per-run state held by the proxy."""

    __slots__ = ("run_id", "track", "miner_hotkey", "calls", "call_counter", "lock")

    def __init__(self, run_id: str, track: str, miner_hotkey: str | None = None) -> None:
        self.run_id = run_id
        self.track = track
        self.miner_hotkey = miner_hotkey
        self.calls: list[ProviderCall] = []
        self.call_counter = 0
        self.lock = threading.Lock()


class LocalProxyState:
    """Mutable state shared between FastAPI handlers and the orchestrator.

    Intentionally a plain object (not module-globals) so tests can construct
    isolated instances and call `register_run` / `pop_calls` directly.
    """

    def __init__(self, cfg: AppConfig, *, http_client: Optional[httpx.Client] = None) -> None:
        self._cfg = cfg
        self._runs: dict[str, _RunState] = {}
        self._lock = threading.Lock()
        self._loaded_keypair: Optional[LoadedKeypair] = None
        self._http_client = http_client

    @property
    def cfg(self) -> AppConfig:
        return self._cfg

    @property
    def loaded_keypair(self) -> Optional[LoadedKeypair]:
        return self._loaded_keypair

    @property
    def http_client(self) -> httpx.Client:
        if self._http_client is None:
            # Lazily create on first use so unit tests can mock easily.
            self._http_client = httpx.Client(timeout=120.0)
        return self._http_client

    def load_hotkey(self) -> None:
        """Load the validator's hotkey. Called once on FastAPI startup."""
        self._loaded_keypair = load_hotkey(self._cfg.bittensor)

    def register_run(
        self,
        run_id: UUID | str,
        track: str = "MAIN",
        *,
        miner_hotkey: str | None = None,
    ) -> None:
        """Allow the proxy to accept calls tagged with this `run_id`.

        Idempotent: re-registering an existing run updates the track. The
        orchestrator must call `pop_calls` (which deregisters) after the run
        completes to avoid an unbounded registry.
        """
        rid = str(run_id)
        with self._lock:
            self._runs[rid] = _RunState(rid, track, miner_hotkey)

    def pop_calls(self, run_id: UUID | str) -> list[ProviderCall]:
        """Drain and remove the run's call log."""
        rid = str(run_id)
        with self._lock:
            state = self._runs.pop(rid, None)
        if state is None:
            return []
        return list(state.calls)

    def _get_run(self, run_id: str) -> Optional[_RunState]:
        with self._lock:
            return self._runs.get(run_id)


# ---------------------------------------------------------------------------
# Request / response shapes (must mirror src/gateway/server.py)
# ---------------------------------------------------------------------------


# We intentionally do not import server.CallRequest/CallResponse to avoid a
# heavy import; the wire shape is small and stable.


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------


def create_app(cfg: AppConfig, *, http_client: Optional[httpx.Client] = None) -> FastAPI:
    """Build a FastAPI app bound to a fresh `LocalProxyState`.

    The app exposes:
    - `GET  /health` — proxy + signing status
    - `POST /v1/call` — agent-facing call shim, signs and forwards upstream
    - `GET  /v1/runs` — debug endpoint listing registered run_ids
    """
    state = LocalProxyState(cfg, http_client=http_client)

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        state.load_hotkey()
        if state.loaded_keypair is None and cfg.bittensor.signing_required:
            raise RuntimeError("hotkey load returned None despite signing_required=True")
        if state.loaded_keypair is None:
            logger.warning("Local proxy starting WITHOUT signing (dev mode).")
        else:
            logger.info(
                "Local proxy ready: hotkey=%s netuid=%d",
                state.loaded_keypair.hotkey_ss58,
                cfg.bittensor.netuid,
            )
        yield

    app = FastAPI(title="Arcratio Validator Local Proxy", version="0.1.0", lifespan=_lifespan)
    app.state.proxy = state

    @app.get("/health")
    def _health() -> dict[str, Any]:
        kp = state.loaded_keypair
        return {
            "status": "ok",
            "signing_enabled": kp is not None,
            "hotkey": kp.hotkey_ss58 if kp else None,
            "netuid": cfg.bittensor.netuid,
            "registered_runs": _registered_count(state),
        }

    @app.get("/v1/runs")
    def _runs() -> dict[str, Any]:
        with state._lock:  # noqa: SLF001
            return {"runs": [{"run_id": r.run_id, "track": r.track, "calls": len(r.calls)} for r in state._runs.values()]}

    @app.post("/v1/call")
    async def _v1_call(
        request: Request,
        x_run_id: str = Header(default=""),
    ) -> dict[str, Any]:
        if not x_run_id:
            raise HTTPException(status_code=401, detail="missing X-Run-Id header")

        run = state._get_run(x_run_id)  # noqa: SLF001
        if run is None:
            raise HTTPException(status_code=401, detail=f"unknown run_id '{x_run_id}'")

        body_bytes = await request.body()
        try:
            body = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"invalid json body: {exc}")

        provider_id = body.get("provider_id")
        call_type = body.get("call_type")
        params = body.get("params", {}) or {}
        miner_hotkey_override = body.get("minerHotkey")
        if not isinstance(miner_hotkey_override, str) or not miner_hotkey_override.strip():
            miner_hotkey_override = (
                params.get("minerHotkey")
                if isinstance(params, dict)
                and isinstance(params.get("minerHotkey"), str)
                and str(params.get("minerHotkey")).strip()
                else None
            )

        if not isinstance(provider_id, str) or not isinstance(call_type, str):
            raise HTTPException(status_code=400, detail="provider_id and call_type are required strings")
        if not isinstance(params, dict):
            raise HTTPException(status_code=400, detail="params must be an object")

        if not is_provider_allowed(run.track, provider_id):
            logger.warning(
                "Run %s (track=%s) attempted disallowed provider '%s'",
                run.run_id,
                run.track,
                provider_id,
            )
            raise HTTPException(
                status_code=403,
                detail=f"provider '{provider_id}' not allowed under track '{run.track}'",
            )

        return _forward_and_record(
            state,
            run,
            provider_id=provider_id,
            call_type=call_type,
            params=params,
            completion_payload=None,
            miner_hotkey_override=miner_hotkey_override,
        )

    @app.post("/v1/gateway/validator/completions")
    async def _v1_gateway_validator_completions(
        request: Request,
        x_run_id: str = Header(default=""),
    ) -> dict[str, Any]:
        if not x_run_id:
            raise HTTPException(status_code=401, detail="missing X-Run-Id header")

        run = state._get_run(x_run_id)  # noqa: SLF001
        if run is None:
            raise HTTPException(status_code=401, detail=f"unknown run_id '{x_run_id}'")

        body_bytes = await request.body()
        try:
            body = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"invalid json body: {exc}")

        provider_id = body.get("provider")
        if not isinstance(provider_id, str) or not provider_id.strip():
            raise HTTPException(status_code=400, detail="provider is required")
        provider_id = provider_id.strip()
        miner_hotkey_override = body.get("minerHotkey")
        if not isinstance(miner_hotkey_override, str) or not miner_hotkey_override.strip():
            miner_hotkey_override = None
        call_type = body.get("callType")
        if not isinstance(call_type, str) or not call_type.strip():
            call_type = _default_call_type(provider_id)
        params = body.get("params", {}) or {}
        if not isinstance(params, dict):
            raise HTTPException(status_code=400, detail="params must be an object when provided")
        if not params:
            params = _params_from_completions_payload(body)

        if not is_provider_allowed(run.track, provider_id):
            logger.warning(
                "Run %s (track=%s) attempted disallowed provider '%s'",
                run.run_id,
                run.track,
                provider_id,
            )
            raise HTTPException(
                status_code=403,
                detail=f"provider '{provider_id}' not allowed under track '{run.track}'",
            )

        completion_payload = dict(body) if isinstance(body, dict) else {}
        return _forward_and_record(
            state,
            run,
            provider_id=provider_id,
            call_type=call_type,
            params=params,
            completion_payload=completion_payload,
            miner_hotkey_override=miner_hotkey_override,
        )

    return app


def _forward_and_record(
    state: LocalProxyState,
    run: _RunState,
    *,
    provider_id: str,
    call_type: str,
    params: dict[str, Any],
    completion_payload: dict[str, Any] | None,
    miner_hotkey_override: str | None = None,
) -> dict[str, Any]:
    """Sign the outbound POST, forward upstream, build a `ProviderCall`."""
    upstream = f"{gateway_service_url()}/v1/gateway/validator/completions"
    miner_hotkey = miner_hotkey_override or run.miner_hotkey
    if not isinstance(miner_hotkey, str) or not miner_hotkey.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "missing miner hotkey for gateway billing context; "
                "register run with miner_hotkey or include minerHotkey in request body"
            ),
        )
    miner_hotkey = miner_hotkey.strip()
    completion_payload = dict(completion_payload or {})
    completion_payload.setdefault("provider", provider_id)
    for key, value in optional_completions_fields(params).items():
        completion_payload.setdefault(key, value)
    completion_payload.pop("minerHotkey", None)
    if provider_id in _NON_COMPLETIONS_PROVIDERS:
        completion_payload.setdefault("callType", call_type)
        completion_payload.setdefault("params", params)
    outbound_text = canonical_json(completion_payload)
    outbound_bytes = outbound_text.encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "x-miner-hotkey": miner_hotkey,
    }
    headers.update(
        sign_validator_completions_request(
            loaded=state.loaded_keypair,
            payload=completion_payload,
            miner_hotkey=miner_hotkey,
            method="POST",
            path_and_query="/v1/gateway/validator/completions",
        )
    )

    t0 = time.monotonic()
    try:
        resp = state.http_client.post(upstream, content=outbound_bytes, headers=headers)
    except httpx.RequestError as exc:
        logger.exception("Upstream request to %s failed", upstream)
        raise HTTPException(status_code=502, detail=f"upstream unreachable: {exc}")
    latency_ms = int((time.monotonic() - t0) * 1000)

    if resp.is_error:
        detail = _extract_upstream_detail(resp)
        raise HTTPException(status_code=resp.status_code, detail=f"upstream {resp.status_code}: {detail}")

    try:
        upstream_payload = resp.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail=f"upstream non-json response: {exc}")

    _maybe_log_proxy_raw_response(
        state,
        provider_id=provider_id,
        call_type=call_type,
        value=upstream_payload,
        label="upstream_payload",
    )
    raw_response = upstream_payload.get("data", upstream_payload)
    if raw_response is not upstream_payload:
        _maybe_log_proxy_raw_response(
            state,
            provider_id=provider_id,
            call_type=call_type,
            value=raw_response,
            label="upstream_data",
        )
    if not isinstance(raw_response, dict):
        raise HTTPException(status_code=502, detail="upstream payload missing response object")

    tier = _DEFAULT_TIERS.get(provider_id, ProviderTier.INFERENCE)
    summary = default_summarise_params(call_type, params)

    with run.lock:
        call_index = run.call_counter
        run.call_counter += 1
        call_record = build_provider_call_record(
            call_index=call_index,
            provider_id=provider_id,
            provider_tier=tier,
            call_type=call_type,
            params=params,
            query_params_summary=summary,
            raw_response=raw_response,
            latency_ms=latency_ms,
        )
        run.calls.append(call_record)

    # Mirror the central gateway's response shape so RemoteProvider on the
    # agent side parses identically.
    return {
        "provider_id": provider_id,
        "call_type": call_type,
        "data": raw_response,
        "latency_ms": latency_ms,
    }


def _default_call_type(provider_id: str) -> str:
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


def _params_from_completions_payload(payload: dict[str, Any]) -> dict[str, Any]:
    params = {
        "model": payload.get("model"),
        "messages": payload.get("messages"),
        "max_tokens": payload.get("maxTokens"),
        "temperature": payload.get("temperature"),
        "top_p": payload.get("topP"),
        "stop": payload.get("stop"),
        "minerHotkey": payload.get("minerHotkey"),
    }
    return {k: v for k, v in params.items() if v is not None}


def _extract_upstream_detail(resp: httpx.Response) -> str:
    text = (resp.text or "")[:1000]
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "detail" in data:
            return str(data["detail"])
    except json.JSONDecodeError:
        pass
    return text or "(empty body)"


def _maybe_log_proxy_raw_response(
    state: LocalProxyState,
    *,
    provider_id: str,
    call_type: str,
    value: Any,
    label: str,
) -> None:
    if not state.cfg.gateway.debug_log_raw_response:
        return
    text = json.dumps(value, sort_keys=True, default=str)
    max_chars = max(0, int(state.cfg.gateway.debug_raw_response_max_chars))
    if max_chars > 0 and len(text) > max_chars:
        logger.warning(
            "Local proxy raw response debug provider=%s call_type=%s label=%s size=%d truncated=true payload=%s...(truncated %d chars)",
            provider_id,
            call_type,
            label,
            len(text),
            text[:max_chars],
            len(text) - max_chars,
        )
        return
    logger.warning(
        "Local proxy raw response debug provider=%s call_type=%s label=%s size=%d truncated=false payload=%s",
        provider_id,
        call_type,
        label,
        len(text),
        text,
    )


def _registered_count(state: LocalProxyState) -> int:
    with state._lock:  # noqa: SLF001
        return len(state._runs)
