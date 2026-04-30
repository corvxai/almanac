"""Optional live call against Anthropic (``pytest --live`` + ``ANTHROPIC_API_KEY``)."""

from __future__ import annotations

import pytest

from src.gateway.providers.claude import ClaudeProvider

from tests.gateway.harness import assert_evidence_nonempty, call_and_extract, require_env_vars


@pytest.mark.live
@pytest.mark.provider("claude")
def test_claude_messages_live_smoke(maybe_pretty_print_raw) -> None:
    require_env_vars("ANTHROPIC_API_KEY")
    params = {
        "model": "claude-haiku-4-5-20250414",
        "max_tokens": 128,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Reply with exactly two lines, nothing else:\n"
                    "PREDICTION: 0.5\n"
                    "REASONING: live gateway smoke test."
                ),
            },
        ],
    }
    raw, evidence = call_and_extract(ClaudeProvider(), "messages", params)
    maybe_pretty_print_raw(raw, banner="claude.messages (live API)")
    assert isinstance(raw, dict)
    assert raw.get("content"), "expected Claude Messages API content blocks"
    assert_evidence_nonempty(evidence)
