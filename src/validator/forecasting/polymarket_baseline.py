"""Validator-owned Polymarket baseline lookup.

This path is intentionally outside the gateway stack so baseline pricing can
be fetched directly by the validator before agent execution.
"""

from __future__ import annotations

import json
import math
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
    """Fetch a validator-side baseline YES/NO price for a Polymarket event.

    ``yes_price`` is the rounded YES outcome. ``no_price`` remains its complement
    and is not the scored NO price. ``yes_outcome_price`` and ``no_outcome_price``
    are Gamma's ``outcomePrices`` aligned by outcome name. A missing, non-finite,
    or out-of-range side returns None.
    """
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

    named = _named_outcome_prices(market)
    if named is None:
        return None
    yes_actual, no_actual = named

    latest_yes = round(max(0.0, min(1.0, yes_actual)), 4)
    payload: dict[str, Any] = {
        "provider": "polymarket",
        "market_slug": market_slug,
        "yes_price": latest_yes,
        "no_price": round(1.0 - latest_yes, 4),
        "yes_outcome_price": yes_actual,
        "no_outcome_price": no_actual,
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


def _named_outcome_prices(market: dict[str, Any]) -> tuple[float, float] | None:
    """Return Gamma YES and NO prices aligned by outcome name, or None."""
    outcomes = _coerce_string_list(market.get("outcomes"))
    prices = _coerce_float_list(market.get("outcomePrices"))
    if not outcomes or not prices:
        return None

    yes_price: float | None = None
    no_price: float | None = None
    for name, price in zip(outcomes, prices):
        label = name.strip().lower()
        if label == "yes" and yes_price is None:
            yes_price = price
        elif label == "no" and no_price is None:
            no_price = price
    if yes_price is None or no_price is None:
        return None
    if not _in_unit_interval(yes_price) or not _in_unit_interval(no_price):
        return None
    return yes_price, no_price


def _in_unit_interval(value: float) -> bool:
    return math.isfinite(value) and 0.0 <= value <= 1.0


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
