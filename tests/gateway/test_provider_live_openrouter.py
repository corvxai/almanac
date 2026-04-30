"""Optional live call to OpenRouter (``pytest --live`` + ``OPENROUTER_API_KEY``).

Uses the same system prompt, user-message shape, and chat parameters as
``OpenRouterAgent`` in ``run_forecast.py`` so live smoke matches production inference.
"""

from __future__ import annotations

import pytest

from src.agent.examples.openrouter_agent import MODELS, SYSTEM_PROMPT, _build_user_prompt
from src.gateway.providers.openrouter import OpenRouterProvider

from tests.gateway.harness import assert_evidence_nonempty, call_and_extract, require_env_vars

# Fixed sample event + synthetic research block (mirrors agent phase-1 context string).
_SAMPLE_TITLE = "Sample binary event: major US tech IPO above $20B valuation in Q3 2026?"
_SAMPLE_DESCRIPTION = (
    "Resolves YES if at least one US-listed technology company completes an IPO "
    "with initial market cap over USD 20B before 2026-10-01."
)
_SAMPLE_RESEARCH = """\
Polymarket: price=0.38, 24h_volume=1,200,000, implied_probability=0.3800
[ft.com] Late-stage private markets remain selective; several large listings postponed into H2.
[reuters.com] Analysts note a narrow window for jumbo IPOs if macro conditions stabilise."""


@pytest.mark.live
@pytest.mark.provider("openrouter")
def test_openrouter_chat_completion_live_smoke(maybe_pretty_print_raw) -> None:
    require_env_vars("OPENROUTER_API_KEY")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _build_user_prompt(
                _SAMPLE_TITLE,
                _SAMPLE_DESCRIPTION,
                _SAMPLE_RESEARCH,
            ),
        },
    ]
    params = {
        "model": MODELS[0],
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 1024,
    }
    raw, evidence = call_and_extract(OpenRouterProvider(), "chat_completion", params)
    maybe_pretty_print_raw(raw, banner="openrouter.chat_completion (live API, OpenRouterAgent-style)")
    assert isinstance(raw, dict)
    assert raw.get("choices"), "expected OpenRouter chat completion choices"
    assert_evidence_nonempty(evidence)
