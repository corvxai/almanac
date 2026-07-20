"""Gateway client — a provider adapter that forwards calls to the gateway HTTP service.

The validator uses RemoteProvider instances in place of real provider adapters.
All actual API calls happen on the gateway server side; the validator only sees
the raw responses flow back through HTTP.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from src.core import constants
from src.core.schemas import ProviderTier
from src.gateway.constants import gateway_service_url
from src.gateway.providers.base import BaseProvider


# Default tier mapping so the validator can still tag calls correctly
# without needing the real provider classes.
_DEFAULT_TIERS: dict[str, ProviderTier] = {
    "polymarket": ProviderTier.FREE_SIGNAL,
    "web_search": ProviderTier.SEARCH,
    "anthropic": ProviderTier.INFERENCE,
    "openai": ProviderTier.INFERENCE,
    "gemini": ProviderTier.INFERENCE,
    "grok": ProviderTier.INFERENCE,
    "perplexity": ProviderTier.DEEP_RESEARCH,
    "openrouter": ProviderTier.INFERENCE,
}


@dataclass(frozen=True)
class ProviderDescriptor:
    provider_id: str
    models: tuple[str, ...]
    allows_any_model: bool
    default_call_type: str
    supports_completions: bool


class RemoteProvider(BaseProvider):
    """Proxies calls to the gateway HTTP service.

    Two modes:

    1. Direct-to-gateway (default): builds its own ephemeral `httpx.Client`
       per call against `gateway_url`. Used by validators in in-process mode.
    2. Pre-configured client (sandbox mode): the caller supplies an
       `httpx.Client` (e.g. with a UNIX-socket transport pointing at the
       validator-local proxy) and a `headers` dict (e.g. `X-Run-Id`). The
       provider uses `client.post(...)` and adds the configured headers to
       every request.
    """

    def __init__(
        self,
        provider_id: str,
        gateway_url: str | None = None,
        provider_tier: ProviderTier | None = None,
        timeout: float = float(constants.GATEWAY.call_timeout_seconds),
        *,
        client: httpx.Client | None = None,
        headers: dict[str, str] | None = None,
    ):
        self._provider_id = provider_id
        self._gateway_url = (gateway_url or gateway_service_url()).rstrip("/")
        self._tier = provider_tier or _DEFAULT_TIERS.get(provider_id, ProviderTier.INFERENCE)
        self._timeout = timeout
        self._client = client
        self._headers = dict(headers) if headers else {}

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def provider_tier(self) -> ProviderTier:
        return self._tier

    def call(self, call_type: str, params: dict[str, Any]) -> dict[str, Any]:
        body = _build_completions_payload(self._provider_id, params)

        if self._client is not None:
            # Pre-configured client (e.g. UDS transport in the sandbox).
            url = (
                "/v1/gateway/validator/completions"
                if self._client.base_url
                else f"{self._gateway_url}/v1/gateway/validator/completions"
            )
            resp = self._client.post(url, json=body, headers=self._headers or None)
        else:
            url = f"{self._gateway_url}/v1/gateway/validator/completions"
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(url, json=body, headers=self._headers or None)
                if resp.is_error:
                    detail = _http_error_detail(resp)
                    raise RuntimeError(
                        f"Gateway {resp.status_code} for {self._provider_id}.{call_type}: {detail}",
                    )
                return resp.json()["data"]

        if resp.is_error:
            detail = _http_error_detail(resp)
            raise RuntimeError(
                f"Gateway {resp.status_code} for {self._provider_id}.{call_type}: {detail}",
            )
        return resp.json()["data"]


def _http_error_detail(resp: httpx.Response) -> str:
    text = (resp.text or "")[:2000]
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "detail" in data:
            return str(data["detail"])
    except json.JSONDecodeError:
        pass
    return text or "(no response body)"


def optional_completions_fields(params: dict[str, Any]) -> dict[str, Any]:
    """Pass through agent-set LLM fields only — no relay defaults."""
    out: dict[str, Any] = {}
    if "model" in params:
        out["model"] = params["model"]
    if "messages" in params:
        out["messages"] = params["messages"]
    if "maxTokens" in params:
        out["maxTokens"] = params["maxTokens"]
    elif "max_tokens" in params:
        out["maxTokens"] = params["max_tokens"]
    if "temperature" in params:
        out["temperature"] = params["temperature"]
    if "topP" in params:
        out["topP"] = params["topP"]
    elif "top_p" in params:
        out["topP"] = params["top_p"]
    if "stop" in params:
        stop = params["stop"]
        out["stop"] = stop if isinstance(stop, list) else [str(stop)]
    return out


def _build_completions_payload(
    provider_id: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Build a plain LLM completions payload (provider/model/messages/...).

    Every provider speaks completions; callType/params typed calls no longer
    exist, so the payload never carries them.
    """
    payload: dict[str, Any] = {"provider": provider_id}
    payload.update(optional_completions_fields(params))
    return payload


def discover_provider_descriptors(gateway_url: str | None = None) -> list[ProviderDescriptor]:
    """Ask the gateway which providers are available."""
    base = (gateway_url or gateway_service_url()).rstrip("/")
    with httpx.Client(timeout=10.0) as client:
        resp = client.get(f"{base}/v1/gateway/providers")
        resp.raise_for_status()
        providers = resp.json().get("providers", [])
        if not isinstance(providers, list):
            raise RuntimeError("invalid gateway providers response: providers must be a list")

        descriptors: list[ProviderDescriptor] = []
        for row in providers:
            if isinstance(row, str):
                descriptors.append(
                    ProviderDescriptor(
                        provider_id=row,
                        models=(),
                        allows_any_model=True,
                        default_call_type="chat_completion",
                        supports_completions=True,
                    )
                )
                continue
            if not isinstance(row, dict):
                continue
            provider_id = row.get("id")
            if isinstance(provider_id, str) and provider_id.strip():
                models = row.get("models", [])
                if not isinstance(models, list):
                    models = []
                descriptors.append(
                    ProviderDescriptor(
                        provider_id=provider_id.strip(),
                        models=tuple(str(m) for m in models if isinstance(m, str)),
                        allows_any_model=bool(row.get("allowsAnyModel", False)),
                        default_call_type=str(row.get("defaultCallType") or "chat_completion"),
                        supports_completions=bool(row.get("supportsCompletions", True)),
                    )
                )
        return descriptors


def discover_providers(gateway_url: str | None = None) -> list[str]:
    """Return provider IDs from gateway discovery."""
    return [p.provider_id for p in discover_provider_descriptors(gateway_url)]


def build_remote_providers(
    gateway_url: str | None = None,
) -> list[RemoteProvider]:
    """Discover providers from the gateway and return RemoteProvider instances."""
    base = gateway_url or gateway_service_url()
    providers = discover_provider_descriptors(base)
    return [RemoteProvider(p.provider_id, base) for p in providers]
