"""Polymarket provider — mock implementation for the prototype.

Returns realistic market data without making real API calls.
Structured so swapping in real httpx calls later is straightforward.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from src.core.schemas import ProviderTier
from src.gateway.providers.base import BaseProvider


class PolymarketProvider(BaseProvider):
    provider_id: str = "polymarket"
    provider_tier: ProviderTier = ProviderTier.FREE_SIGNAL

    def call(self, call_type: str, params: dict[str, Any]) -> dict[str, Any]:
        handlers = {
            "get_market": self._mock_get_market,
            "get_price_history": self._mock_get_price_history,
        }
        handler = handlers.get(call_type)
        if handler is None:
            raise ValueError(f"Unknown call_type for polymarket: {call_type}")
        return handler(params)

    def summarise_params(self, call_type: str, params: dict[str, Any]) -> str:
        market = params.get("market_slug", params.get("event_id", "unknown"))
        return f"polymarket.{call_type}(market={market})"

    # -- mock handlers --------------------------------------------------------

    def _mock_get_market(self, params: dict[str, Any]) -> dict[str, Any]:
        base_price = random.uniform(0.15, 0.85)
        return {
            "market_slug": params.get("market_slug", "mock-market"),
            "price": round(base_price, 4),
            "probability": round(base_price, 4),
            "volume_24h": round(random.uniform(50_000, 2_000_000), 2),
            "total_volume": round(random.uniform(5_000_000, 50_000_000), 2),
            "liquidity": round(random.uniform(100_000, 5_000_000), 2),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    def _mock_get_price_history(self, params: dict[str, Any]) -> dict[str, Any]:
        days = params.get("days", 7)
        now = datetime.now(timezone.utc)
        price = random.uniform(0.25, 0.75)
        history = []
        for i in range(days):
            price += random.uniform(-0.05, 0.05)
            price = max(0.01, min(0.99, price))
            history.append({
                "timestamp": (now - timedelta(days=days - i)).isoformat(),
                "price": round(price, 4),
            })
        return {
            "market_slug": params.get("market_slug", "mock-market"),
            "price_history": history,
        }
