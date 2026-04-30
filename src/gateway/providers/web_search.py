"""Web search provider — mock implementation for the prototype.

Returns realistic search result structures without making real API calls.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any

from src.core.schemas import ProviderTier
from src.gateway.providers.base import BaseProvider

_MOCK_DOMAINS = [
    "reuters.com", "apnews.com", "bbc.co.uk", "nytimes.com",
    "bloomberg.com", "ft.com", "economist.com", "politico.com",
]


class WebSearchProvider(BaseProvider):
    provider_id: str = "web_search"
    provider_tier: ProviderTier = ProviderTier.SEARCH

    def call(self, call_type: str, params: dict[str, Any]) -> dict[str, Any]:
        if call_type != "search":
            raise ValueError(f"Unknown call_type for web_search: {call_type}")
        return self._mock_search(params)

    def summarise_params(self, call_type: str, params: dict[str, Any]) -> str:
        query = params.get("query", "")
        n = params.get("num_results", 5)
        return f'web_search.search(query="{query}", n={n})'

    def _mock_search(self, params: dict[str, Any]) -> dict[str, Any]:
        query = params.get("query", "forecasting event")
        num_results = params.get("num_results", 5)
        now = datetime.now(timezone.utc)

        results = []
        for i in range(num_results):
            domain = random.choice(_MOCK_DOMAINS)
            age_days = random.randint(0, 14)
            results.append({
                "title": f"Analysis: {query} — perspective {i + 1}",
                "url": f"https://{domain}/article/{i + 1}",
                "snippet": self._generate_snippet(query, i),
                "published": (now - timedelta(days=age_days)).isoformat(),
                "source": domain,
            })

        return {
            "query": query,
            "results": results,
            "result_count": random.randint(50_000, 500_000),
            "search_timestamp": now.isoformat(),
        }

    def _generate_snippet(self, query: str, index: int) -> str:
        templates = [
            f"Analysts suggest {query} remains uncertain, with key factors including economic indicators and policy shifts.",
            f"Recent developments in {query} point toward increased likelihood, according to multiple sources.",
            f"Experts are divided on {query}; some cite momentum while others highlight structural barriers.",
            f"Market participants have adjusted positions on {query} following recent announcements.",
            f"New data released this week could significantly impact the outlook for {query}.",
        ]
        return templates[index % len(templates)]
