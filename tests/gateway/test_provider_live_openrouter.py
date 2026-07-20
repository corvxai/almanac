"""Optional live call to OpenRouter (``pytest --live`` + ``OPENROUTER_API_KEY``).

Uses a forecasting system prompt, user-message shape, and chat parameters that
mirror an OpenRouter-driven agent so live smoke matches production inference.
"""

from __future__ import annotations

import pytest

from src.gateway.providers.openrouter import OpenRouterProvider

from tests.gateway.harness import assert_evidence_nonempty, call_and_extract, require_env_vars

MODELS = [
    "anthropic/claude-haiku-4-5",
    "openai/gpt-4o-mini",
    "meta-llama/llama-4-maverick",
    "deepseek/deepseek-r1",
]

SYSTEM_PROMPT = """\
You are an expert forecaster specialising in probabilistic predictions on \
binary events (YES/NO outcomes).

Key principles:
- Consider base rates and historical precedents
- Weigh evidence quality and recency
- Account for uncertainty and missing information
- Avoid extreme predictions (0 or 1) unless evidence is overwhelming
- Use the full probability range: 0.0 (impossible) to 1.0 (certain)"""


def _build_user_prompt(title: str, description: str, context: str) -> str:
    return f"""\
**Event to Forecast:**
{title}

**Full Description:**
{description}

**Research Context:**
{context if context else "No additional research context available."}

**Your Task:**
Estimate the probability (0.0 to 1.0) that this event will resolve YES.

**Required Output Format:**
PREDICTION: [number between 0.0 and 1.0]
CONVICTION: [number between 0.0 and 1.0 indicating your confidence in the prediction value]
REASONING: [2-4 sentences explaining your probability estimate]"""

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
