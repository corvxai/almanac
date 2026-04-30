"""Perplexity provider — mock implementation.

Perplexity's Sonar API always includes web search. The response format has:
- choices[].message.content: the completion text
- citations: array of URLs used as sources
- search_results: array of {title, url, snippet} objects with full context

Visibility tier: HIGHEST — we get both citation URLs and search_results
with titles and snippets. This is the most transparent LLM provider
for evidence tracing.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any

from src.core.schemas import ProviderTier
from src.gateway.providers.base import BaseProvider

_MODELS = ["sonar-pro", "sonar-reasoning-pro", "sonar"]

_MOCK_SEARCH_RESULTS = [
    {
        "title": "Federal Reserve Interest Rate Decision Analysis",
        "url": "https://reuters.com/markets/fed-rate-analysis",
        "snippet": "The Federal Reserve is widely expected to hold rates steady, with futures markets pricing in a 65% chance of a cut by mid-year.",
    },
    {
        "title": "Polymarket: Fed Rate Cut Probability",
        "url": "https://polymarket.com/event/fed-rate-cut",
        "snippet": "Current market price implies a 58% probability of at least one rate cut before the deadline.",
    },
    {
        "title": "Economic Indicators Dashboard",
        "url": "https://bloomberg.com/economics/indicators",
        "snippet": "Latest CPI data shows inflation trending down to 2.3%, bolstering the case for rate relief.",
    },
    {
        "title": "Expert Analysis: Central Bank Policy Outlook",
        "url": "https://ft.com/central-bank-policy-outlook",
        "snippet": "Economists surveyed by the FT assign a median 55% probability to a rate cut, citing labor market softening.",
    },
    {
        "title": "Wikipedia: Federal funds rate",
        "url": "https://en.wikipedia.org/wiki/Federal_funds_rate",
        "snippet": "The federal funds rate is the interest rate at which depository institutions trade federal funds with each other overnight.",
    },
]


class PerplexityProvider(BaseProvider):
    provider_id: str = "perplexity"
    provider_tier: ProviderTier = ProviderTier.INFERENCE

    def call(self, call_type: str, params: dict[str, Any]) -> dict[str, Any]:
        if call_type != "chat_completion":
            raise ValueError(f"Unknown call_type for perplexity: {call_type}")
        return self._mock_chat_completion(params)

    def summarise_params(self, call_type: str, params: dict[str, Any]) -> str:
        model = params.get("model", _MODELS[0])
        return f"perplexity.{call_type}(model={model})"

    def _mock_chat_completion(self, params: dict[str, Any]) -> dict[str, Any]:
        model = params.get("model", _MODELS[0])
        prediction = round(random.uniform(0.15, 0.85), 4)

        num_results = random.randint(3, 5)
        search_results = random.sample(_MOCK_SEARCH_RESULTS, num_results)
        citations = [r["url"] for r in search_results]

        reasoning = (
            f"Based on search results from {len(search_results)} sources, "
            f"including market data and expert analysis, "
            f"the probability is estimated at {prediction:.1%}."
        )

        return {
            "id": f"pplx-{random.randint(100000, 999999)}",
            "model": model,
            "object": "chat.completion",
            "created": int(datetime.now(timezone.utc).timestamp()),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"PREDICTION: {prediction}\nREASONING: {reasoning}",
                },
                "finish_reason": "stop",
            }],
            "citations": citations,
            "search_results": search_results,
            "usage": {
                "prompt_tokens": random.randint(200, 500),
                "completion_tokens": random.randint(200, 600),
                "total_tokens": random.randint(500, 1100),
            },
        }
