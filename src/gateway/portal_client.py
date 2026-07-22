"""Miner-facing client for testing agents against the Almanac Portal API."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import httpx

from src.core.events import Event
from src.core.schemas import EventCategory
from src.gateway.client import optional_completions_fields


@dataclass(frozen=True)
class PortalCall:
    provider: str
    model: str | None
    latency_ms: int
    cost_micro: str | None
    balance_after_micro: str | None


class PortalGateway:
    """Gateway-shaped facade backed by the miner API-key completions route."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key.strip()
        if not self._api_key:
            raise ValueError("gateway API key must not be empty")
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None
        self._headers = {"Authorization": f"Bearer {self._api_key}"}
        self.call_log: list[PortalCall] = []
        self.available_providers = self._discover_providers()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "PortalGateway":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _discover_providers(self) -> tuple[str, ...]:
        try:
            response = self._client.get(
                f"{self._base_url}/v1/gateway/providers",
                headers=self._headers,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise _portal_http_error(exc.response, action="discover providers") from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"network error discovering portal providers: {exc}") from exc

        providers = response.json().get("providers")
        if not isinstance(providers, list):
            raise RuntimeError("invalid portal provider response: 'providers' must be a list")
        ids = tuple(
            provider_id.strip()
            for row in providers
            if isinstance(row, (str, dict))
            for provider_id in [row if isinstance(row, str) else row.get("id")]
            if isinstance(provider_id, str) and provider_id.strip()
        )
        if not ids:
            raise RuntimeError("portal reported no available providers")
        return ids

    def call_provider(
        self,
        provider_id: str,
        call_type: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        if provider_id not in self.available_providers:
            available = ", ".join(self.available_providers)
            raise RuntimeError(
                f"provider '{provider_id}' is not available on the portal test path; "
                f"available providers: {available}"
            )

        payload: dict[str, Any] = {"provider": provider_id}
        payload.update(optional_completions_fields(params))
        started = time.monotonic()
        try:
            response = self._client.post(
                f"{self._base_url}/v1/gateway/completions",
                headers=self._headers,
                json=payload,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise _portal_http_error(
                exc.response,
                action=f"call {provider_id}.{call_type}",
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"network error calling portal gateway: {exc}") from exc

        latency_ms = int((time.monotonic() - started) * 1000)
        body = response.json()
        if not isinstance(body, dict):
            raise RuntimeError("invalid portal completion response: expected an object")
        output = body.get("output")
        if not isinstance(output, str):
            raise RuntimeError("invalid portal completion response: missing string 'output'")

        self.call_log.append(
            PortalCall(
                provider=str(body.get("provider") or provider_id),
                model=_optional_str(body.get("model")),
                latency_ms=latency_ms,
                cost_micro=_optional_str(body.get("costMicro")),
                balance_after_micro=_optional_str(body.get("balanceAfterMicro")),
            )
        )

        almanac = {
            key: body[key]
            for key in (
                "costMicro",
                "providerCostMicro",
                "markupBps",
                "balanceAfterMicro",
            )
            if key in body
        }
        return {
            "output": output,
            "choices": [{"message": {"content": output}}],
            "model": body.get("model"),
            "usage": body.get("usage"),
            "_almanac": almanac,
        }


def fetch_portal_event(
    *,
    base_url: str,
    event_id: str | None = None,
    timeout: float = 20.0,
    client: httpx.Client | None = None,
) -> Event:
    """Fetch a random (or specific) live event and map it to the agent schema."""
    own_client = client is None
    http_client = client or httpx.Client(timeout=timeout)
    endpoint = f"/v1/events/{event_id}" if event_id else "/v1/events/random"
    try:
        try:
            response = http_client.get(f"{base_url.rstrip('/')}{endpoint}")
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise _portal_http_error(exc.response, action="fetch event") from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"network error fetching portal event: {exc}") from exc
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("invalid portal event response: expected an object")
        return portal_event_to_event(payload)
    finally:
        if own_client:
            http_client.close()


def portal_event_to_event(payload: dict[str, Any]) -> Event:
    """Map the public portal event shape to the internal agent Event."""
    _required_str(payload, "id")
    source = _required_str(payload, "source")
    source_market_id = _required_str(payload, "sourceMarketId")
    question = _required_str(payload, "question")
    description = _optional_str(payload.get("description")) or question
    end_date = _required_str(payload, "endDate")
    try:
        deadline = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(f"invalid portal event endDate: {end_date!r}") from exc
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=timezone.utc)

    outcomes = payload.get("outcomes")
    if not isinstance(outcomes, list):
        outcomes = []
    prices = payload.get("currentOutcomePrices")
    if not isinstance(prices, dict):
        prices = {}
    tags = payload.get("tagsAtAdmission")
    if not isinstance(tags, list):
        tags = []

    return Event(
        event_id=uuid5(NAMESPACE_URL, f"{source}:{source_market_id}"),
        title=question,
        description=description,
        category=EventCategory.OTHER,
        resolution_criteria=description,
        resolution_deadline=deadline,
        source=source,
        source_id=source_market_id,
        tags=[str(tag) for tag in tags],
        outcomes=[
            {str(key): str(value) for key, value in row.items() if isinstance(value, str)}
            for row in outcomes
            if isinstance(row, dict)
        ],
        current_outcome_prices={
            str(key): float(value)
            for key, value in prices.items()
        },
    )


def _portal_http_error(response: httpx.Response, *, action: str) -> RuntimeError:
    detail = _response_detail(response)
    if response.status_code == 401:
        return RuntimeError(
            f"portal gateway API key was rejected while trying to {action}; "
            "check GATEWAY_API_KEY"
        )
    if response.status_code in {402, 403} and "credit" in detail.lower():
        return RuntimeError(f"insufficient portal credits while trying to {action}: {detail}")
    return RuntimeError(
        f"portal request failed while trying to {action} "
        f"(HTTP {response.status_code}): {detail}"
    )


def _response_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text.strip() or "no response body"
    if isinstance(body, dict):
        detail = body.get("message") or body.get("detail") or body.get("error")
        if isinstance(detail, list):
            return "; ".join(str(item) for item in detail)
        if detail:
            return str(detail)
    return str(body)


def _optional_str(value: Any) -> str | None:
    return str(value) if value is not None else None


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"invalid portal event response: '{key}' is required")
    return value.strip()
