"""Gateway tests for usage/cost metadata normalization."""

from __future__ import annotations

import pytest

from src.core.schemas import ProviderTier
from src.gateway.gateway import ProviderGateway

from tests.gateway.harness import StubProvider, load_raw_fixture


def test_gateway_records_openrouter_usage_metadata() -> None:
    raw = load_raw_fixture("openrouter", "chat_completion")
    provider = StubProvider(
        provider_id="openrouter",
        call_type="chat_completion",
        raw=raw,
        tier=ProviderTier.INFERENCE,
    )
    gateway = ProviderGateway(providers={"openrouter": provider})

    out = gateway.call_provider("openrouter", "chat_completion", {})
    assert out == raw

    call = gateway.call_log[0]
    assert call.usage_meta is not None
    assert call.usage_meta.prompt_tokens == raw["usage"]["prompt_tokens"]
    assert call.usage_meta.completion_tokens == raw["usage"]["completion_tokens"]
    assert call.usage_meta.total_tokens == raw["usage"]["total_tokens"]
    assert call.usage_meta.cost == raw["usage"]["cost"]
    assert call.usage_meta.is_byok == raw["usage"]["is_byok"]
    assert call.usage_meta.cached_tokens == raw["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert (
        call.usage_meta.reasoning_tokens
        == raw["usage"]["completion_tokens_details"]["reasoning_tokens"]
    )
    assert call.provider_usage_raw == raw["usage"]
    assert call.cost_units == raw["usage"]["cost"]


def test_gateway_leaves_usage_empty_when_provider_has_no_usage() -> None:
    raw = load_raw_fixture("web_search", "search")
    provider = StubProvider(
        provider_id="web_search",
        call_type="search",
        raw=raw,
        tier=ProviderTier.SEARCH,
    )
    gateway = ProviderGateway(providers={"web_search": provider})

    gateway.call_provider("web_search", "search", {"query": "test"})
    call = gateway.call_log[0]
    assert call.usage_meta is None
    assert call.provider_usage_raw is None
    assert call.cost_units == 0.0


def test_gateway_normalizes_claude_style_usage_tokens() -> None:
    raw = {
        "id": "msg_test",
        "usage": {
            "input_tokens": 1200,
            "output_tokens": 300,
            "cache_read_input_tokens": 100,
        },
        "content": [{"type": "text", "text": "PREDICTION: 0.5\nREASONING: test"}],
    }
    provider = StubProvider(
        provider_id="anthropic",
        call_type="messages",
        raw=raw,
        tier=ProviderTier.INFERENCE,
    )
    gateway = ProviderGateway(providers={"anthropic": provider})

    gateway.call_provider("anthropic", "messages", {"model": "claude-sonnet-4-6"})
    call = gateway.call_log[0]
    assert call.usage_meta is not None
    assert call.usage_meta.input_tokens == 1200
    assert call.usage_meta.output_tokens == 300
    # Backfilled for consistent cross-provider accounting fields.
    assert call.usage_meta.prompt_tokens == 1200
    assert call.usage_meta.completion_tokens == 300
    assert call.usage_meta.total_tokens == 1500
    assert call.usage_meta.cached_tokens == 100
    # Estimated from configured per-model rates when provider omits cost.
    assert call.usage_meta.cost == pytest.approx(0.00783, rel=1e-9)
    assert call.cost_units == pytest.approx(0.00783, rel=1e-9)


def test_gateway_normalizes_openrouter_camelcase_usage_tokens() -> None:
    raw = {
        "provider": "openrouter",
        "output": "PREDICTION: 0.6\nCONVICTION: 0.7\nREASONING: test",
        "usage": {
            "promptTokens": 280,
            "completionTokens": 405,
            "totalTokens": 685,
            "cacheReadInputTokens": 12,
            "costUsd": 0.000285,
        },
    }
    provider = StubProvider(
        provider_id="openrouter",
        call_type="chat_completion",
        raw=raw,
        tier=ProviderTier.INFERENCE,
    )
    gateway = ProviderGateway(providers={"openrouter": provider})

    gateway.call_provider("openrouter", "chat_completion", {})
    call = gateway.call_log[0]
    assert call.usage_meta is not None
    assert call.usage_meta.prompt_tokens == 280
    assert call.usage_meta.completion_tokens == 405
    assert call.usage_meta.total_tokens == 685
    assert call.usage_meta.cached_tokens == 12
    assert call.usage_meta.cost == pytest.approx(0.000285, rel=1e-9)
