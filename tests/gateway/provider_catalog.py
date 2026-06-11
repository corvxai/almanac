"""Inventory of gateway providers: supported ``call_type`` values and minimal ``params``.

Used by integration-style tests that invoke each adapter's ``call`` then run
``extract_evidence`` (see ``tests/gateway/harness.py``).

**Mock adapters** (no API keys): see ``MOCK_PROVIDER_IDS``.

**Live adapter**: ``anthropic`` (Anthropic Messages API) — covered only under
``pytest --live`` with ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from src.core.schemas import EvidenceType
from src.gateway.providers.base import BaseProvider
from src.gateway.providers.gemini import GeminiProvider
from src.gateway.providers.grok import GrokProvider
from src.gateway.providers.openai import OpenAIProvider
from src.gateway.providers.openrouter import OpenRouterProvider
from src.gateway.providers.perplexity import PerplexityProvider
from src.gateway.providers.polymarket import PolymarketProvider
from src.gateway.providers.web_search import WebSearchProvider

_MIN_LLM_USER = (
    "Reply with exactly two lines of text, nothing else:\n"
    "PREDICTION: 0.5\n"
    "REASONING: smoke test."
)


@dataclass(frozen=True, slots=True)
class ProviderCallExample:
    """One supported operation: minimal params + evidence types the mock should surface."""

    provider_id: str
    call_type: str
    params: dict[str, Any]
    require_evidence_types: tuple[EvidenceType, ...]


def _openrouter_mock() -> BaseProvider:
    """Always mock — avoids accidental live calls when ``OPENROUTER_API_KEY`` is in the shell."""
    return OpenRouterProvider(use_mock=True)


_MOCK_FACTORY: dict[str, Callable[[], BaseProvider]] = {
    "polymarket": PolymarketProvider,
    "web_search": WebSearchProvider,
    "grok": GrokProvider,
    "perplexity": PerplexityProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "openrouter": _openrouter_mock,
}

MOCK_PROVIDER_IDS: tuple[str, ...] = tuple(sorted(_MOCK_FACTORY.keys()))


def build_mock_provider(provider_id: str) -> BaseProvider:
    """Construct the in-repo mock (or keyless) adapter for ``provider_id``."""
    fn = _MOCK_FACTORY.get(provider_id)
    if fn is None:
        raise KeyError(f"no mock provider registered for {provider_id!r}")
    return fn()


# Every row is executed by ``test_provider_mock_calls`` (offline, in-process mocks).
MOCK_PROVIDER_CALL_EXAMPLES: tuple[ProviderCallExample, ...] = (
    ProviderCallExample(
        "polymarket",
        "get_market",
        {"market_slug": "pytest-market"},
        (EvidenceType.PRICE, EvidenceType.PROBABILITY),
    ),
    ProviderCallExample(
        "polymarket",
        "get_price_history",
        {"market_slug": "pytest-market", "days": 4},
        (EvidenceType.TREND,),
    ),
    ProviderCallExample(
        "web_search",
        "search",
        {"query": "arcratio pytest query", "num_results": 2},
        (EvidenceType.QUOTE_SUMMARY, EvidenceType.STATISTIC),
    ),
    ProviderCallExample(
        "grok",
        "chat_completion",
        {"messages": [{"role": "user", "content": _MIN_LLM_USER}], "search": True},
        (EvidenceType.PROBABILITY, EvidenceType.FACT),
    ),
    ProviderCallExample(
        "perplexity",
        "chat_completion",
        {"messages": [{"role": "user", "content": _MIN_LLM_USER}]},
        (EvidenceType.PROBABILITY, EvidenceType.QUOTE_SUMMARY),
    ),
    ProviderCallExample(
        "openai",
        "responses",
        {"model": "gpt-5-mini", "tools": [{"type": "web_search"}]},
        (EvidenceType.PROBABILITY, EvidenceType.FACT),
    ),
    ProviderCallExample(
        "openai",
        "chat_completion",
        {"model": "gpt-5-mini", "messages": [{"role": "user", "content": _MIN_LLM_USER}]},
        (EvidenceType.PROBABILITY, EvidenceType.QUOTE_SUMMARY),
    ),
    ProviderCallExample(
        "gemini",
        "generate_content",
        {"model": "gemini-2.5-flash", "grounding": True},
        (EvidenceType.PROBABILITY, EvidenceType.FACT),
    ),
    ProviderCallExample(
        "openrouter",
        "chat_completion",
        {"messages": [{"role": "user", "content": _MIN_LLM_USER}]},
        (EvidenceType.PROBABILITY, EvidenceType.QUOTE_SUMMARY),
    ),
)
