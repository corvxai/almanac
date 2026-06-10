"""Gateway client — a provider adapter that forwards calls to the gateway HTTP service.

The validator uses RemoteProvider instances in place of real provider adapters.
All actual API calls happen on the gateway server side; the validator only sees
the raw responses flow back through HTTP.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from src.core.schemas import ProviderTier
from src.gateway.constants import gateway_service_url
from src.gateway.providers.base import BaseProvider


# Default tier mapping so the validator can still tag calls correctly
# without needing the real provider classes.
_DEFAULT_TIERS: dict[str, ProviderTier] = {
    "polymarket": ProviderTier.FREE_SIGNAL,
    "web_search": ProviderTier.SEARCH,
    "claude": ProviderTier.INFERENCE,
    "openai": ProviderTier.INFERENCE,
    "gemini": ProviderTier.INFERENCE,
    "grok": ProviderTier.INFERENCE,
    "perplexity": ProviderTier.DEEP_RESEARCH,
    "openrouter": ProviderTier.INFERENCE,
}


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
        timeout: float = 120.0,
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
        body = {
            "provider_id": self._provider_id,
            "call_type": call_type,
            "params": params,
        }

        if self._client is not None:
            # Pre-configured client (e.g. UDS transport in the sandbox).
            url = "/v1/call" if self._client.base_url else f"{self._gateway_url}/v1/call"
            resp = self._client.post(url, json=body, headers=self._headers or None)
        else:
            url = f"{self._gateway_url}/v1/call"
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


def discover_providers(gateway_url: str | None = None) -> list[str]:
    """Ask the gateway which providers are available."""
    base = (gateway_url or gateway_service_url()).rstrip("/")
    with httpx.Client(timeout=10.0) as client:
        resp = client.get(f"{base}/v1/gateway/providers")
        resp.raise_for_status()
        providers = resp.json().get("providers", [])
        if not isinstance(providers, list):
            raise RuntimeError("invalid gateway providers response: providers must be a list")

        provider_ids: list[str] = []
        for row in providers:
            if isinstance(row, str):
                provider_ids.append(row)
                continue
            if not isinstance(row, dict):
                continue
            provider_id = row.get("id")
            if isinstance(provider_id, str) and provider_id.strip():
                provider_ids.append(provider_id.strip())
        return provider_ids


def build_remote_providers(
    gateway_url: str | None = None,
) -> list[RemoteProvider]:
    """Discover providers from the gateway and return RemoteProvider instances."""
    base = gateway_url or gateway_service_url()
    provider_ids = discover_providers(base)
    return [RemoteProvider(pid, base) for pid in provider_ids]
