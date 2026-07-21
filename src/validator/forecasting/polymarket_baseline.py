"""Validator-owned Polymarket baseline lookup.

This path is intentionally outside the gateway stack so baseline pricing can
be fetched directly by the validator before agent execution.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

import httpx

from src.core.events import Event

POLYMARKET_GAMMA_BASE = "https://gamma-api.polymarket.com"
_DEFAULT_TIMEOUT_SECONDS = 10.0


def fetch_polymarket_baseline(
    event: Event,
    *,
    http_client: httpx.Client | None = None,
) -> dict[str, Any] | None:
    """Fetch a validator-side baseline YES/NO price for a Polymarket event."""
    if not event.source or event.source.lower() != "polymarket":
        return None

    market_slug = derive_market_slug(event)
    market = _fetch_market_snapshot(
        source_id=event.source_id,
        market_slug=market_slug,
        http_client=http_client,
    )
    if market is None:
        return None

    latest_yes = _extract_yes_price(market)
    if latest_yes is None:
        return None

    latest_yes = round(max(0.0, min(1.0, latest_yes)), 4)
    payload: dict[str, Any] = {
        "provider": "polymarket",
        "market_slug": market_slug,
        "yes_price": latest_yes,
        "no_price": round(1.0 - latest_yes, 4),
    }
    if event.source_id:
        payload["source_id"] = event.source_id

    as_of = market.get("updatedAt") or market.get("lastUpdated")
    if isinstance(as_of, str) and as_of:
        payload["as_of"] = as_of
    # This direct lookup is a spot snapshot, not a historical series.
    payload["history_points"] = 1
    return payload


def _fetch_market_snapshot(
    *,
    source_id: str | None,
    market_slug: str,
    http_client: httpx.Client | None = None,
) -> dict[str, Any] | None:
    owns_client = http_client is None
    client = http_client or httpx.Client(
        base_url=POLYMARKET_GAMMA_BASE,
        timeout=_DEFAULT_TIMEOUT_SECONDS,
    )
    try:
        if source_id:
            market = _request_first_market(client, {"id": source_id})
            if market is not None:
                return market
        return _request_first_market(client, {"slug": market_slug})
    finally:
        if owns_client:
            client.close()


def _request_first_market(client: httpx.Client, params: dict[str, Any]) -> dict[str, Any] | None:
    resp = client.get("/markets", params=params)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data[0] if data else None
    if isinstance(data, dict):
        markets = data.get("markets")
        if isinstance(markets, list) and markets:
            first = markets[0]
            return first if isinstance(first, dict) else None
    return None


def _extract_yes_price(market: dict[str, Any]) -> float | None:
    outcomes_raw = market.get("outcomes")
    prices_raw = market.get("outcomePrices")

    outcomes = _coerce_string_list(outcomes_raw)
    prices = _coerce_float_list(prices_raw)

    if prices:
        yes_idx = None
        if outcomes:
            for idx, name in enumerate(outcomes):
                if name.strip().lower() == "yes":
                    yes_idx = idx
                    break
        if yes_idx is not None and yes_idx < len(prices):
            return prices[yes_idx]
        return prices[0]

    for key in ("price", "probability"):
        value = market.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _coerce_string_list(value: Any) -> list[str] | None:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
    return None


def _coerce_float_list(value: Any) -> list[float] | None:
    raw: list[Any] | None = None
    if isinstance(value, list):
        raw = value
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, list):
            raw = parsed
    if raw is None:
        return None

    out: list[float] = []
    for item in raw:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return None
    return out


def derive_market_slug(event: Event) -> str:
    if event.source_url:
        try:
            path = urlparse(event.source_url).path.strip("/")
            parts = [p for p in path.split("/") if p]
            if len(parts) >= 2 and parts[-2] == "event":
                return parts[-1]
        except Exception:  # noqa: BLE001
            pass
    return event.title.lower().replace(" ", "-")
