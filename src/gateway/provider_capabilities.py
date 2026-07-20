"""Canonical, dependency-free provider capability map.

Single source of truth for each provider's default call type. All providers
speak the LLM *completions* shape (model/messages/max_tokens/...); agents do
their own research via those completions (e.g. OpenRouter ``:online`` models).

Imported by the sandbox provider factory (`agent/runner_entrypoint.py`), the
validator-local proxy (`gateway/local_proxy.py`), and available to the gateway
server, so the three cannot drift.
"""
from __future__ import annotations

_DEFAULT_CALL_TYPE: dict[str, str] = {
    "anthropic": "messages",
    "openrouter": "chat_completion",
    "openai": "chat_completion",
    "grok": "chat_completion",
    "gemini": "generate_content",
    "perplexity": "chat_completion",
}


def provider_default_call_type(provider_id: str) -> str:
    """Default call type for a provider (fallback: chat_completion)."""
    return _DEFAULT_CALL_TYPE.get(provider_id, "chat_completion")
