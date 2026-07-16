"""Canonical, dependency-free provider capability map.

Single source of truth for two facts about each provider:

  * whether it speaks the LLM *completions* shape (model/messages/max_tokens/...)
    or a *typed* call (callType + params, e.g. polymarket.get_market), and
  * its default call type.

Imported by the sandbox provider factory (`agent/runner_entrypoint.py`), the
validator-local proxy (`gateway/local_proxy.py`), and available to the gateway
server, so the three cannot drift. Drift here was load-bearing: data-provider
`params` (market_slug / query) were silently dropped before reaching the
gateway because the sandbox built every provider as a completions provider.
"""
from __future__ import annotations

# Providers that take a TYPED call (callType + params) rather than the LLM
# completions payload. Their `params` MUST be forwarded verbatim to the gateway.
DATA_PROVIDERS: frozenset[str] = frozenset({"polymarket", "web_search"})

_DEFAULT_CALL_TYPE: dict[str, str] = {
    "anthropic": "messages",
    "openrouter": "chat_completion",
    "openai": "chat_completion",
    "grok": "chat_completion",
    "gemini": "generate_content",
    "perplexity": "chat_completion",
    "polymarket": "get_market",
    "web_search": "search",
}


def provider_supports_completions(provider_id: str) -> bool:
    """True if the provider speaks the LLM completions payload shape.

    Data providers return False so their `params` are sent through as a typed
    call instead of being reduced to completion fields (which drops them).
    """
    return provider_id not in DATA_PROVIDERS


def provider_default_call_type(provider_id: str) -> str:
    """Default call type for a provider (fallback: chat_completion)."""
    return _DEFAULT_CALL_TYPE.get(provider_id, "chat_completion")
