"""ClaudeProvider mock mode — the simulated gateway runs mock-by-default.

Verifier/miner test runs must work with no ANTHROPIC_API_KEY, no network, and
no spend, while still returning a payload shaped like the real Messages API so
the validator's recorder/extractor treats it like a live call.
"""

from __future__ import annotations

import pytest

from src.gateway.providers.claude import ClaudeProvider


def test_use_mock_forces_mock_even_with_key():
    p = ClaudeProvider(api_key="sk-real", use_mock=True)
    assert p.mode == "mock"


def test_no_key_defaults_to_mock(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    p = ClaudeProvider()
    assert p.mode == "mock"


def test_key_present_is_live(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    p = ClaudeProvider(api_key="sk-real")
    assert p.mode == "live"


def test_mock_messages_shape_matches_anthropic(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    p = ClaudeProvider(use_mock=True)
    out = p.call("messages", {"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": "hi"}]})

    assert out["type"] == "message"
    assert out["role"] == "assistant"
    assert out["content"][0]["type"] == "text"
    assert "PREDICTION" in out["content"][0]["text"]
    # usage uses Anthropic's input_tokens/output_tokens keys (not prompt/completion)
    assert "input_tokens" in out["usage"] and "output_tokens" in out["usage"]


def test_mock_makes_no_network_call(monkeypatch):
    """If the mock path accidentally hit httpx, this would raise."""
    import src.gateway.providers.claude as claude_mod

    def _boom(*a, **k):  # pragma: no cover - only fires on regression
        raise AssertionError("mock path must not construct an httpx.Client")

    monkeypatch.setattr(claude_mod.httpx, "Client", _boom)
    p = ClaudeProvider(use_mock=True)
    out = p.call("messages", {"messages": []})
    assert out["type"] == "message"
