"""Shared helpers for the v1 capability-ladder agents (agents 1-5).

These five agents demonstrate increasing forecasting capability on the SAME
event so their traces can be compared apples-to-apples. They all share the
same PREDICTION/CONVICTION/REASONING output contract and parser, copied from
``openrouter_agent2.py``. The only thing that changes across the ladder is
*how much real external evidence* each one gathers before it reasons.

All external calls go through ``ctx.call_provider("openrouter",
"chat_completion", ...)``. A "web search call" is a model id with the
``:online`` suffix or a web-native model (e.g. ``perplexity/sonar-pro``); a
"reasoning call" is a plain model id.
"""

from __future__ import annotations

import re

# Model ladder (use exactly these — chosen for the POC) -----------------------
BASIC_LLM = "openai/gpt-4o-mini"
GOOD_LLM = "anthropic/claude-sonnet-4-6"  # reliable output-contract follower; deepseek-r1 buries the final number in prose
BASIC_SEARCH = "openai/gpt-4o-mini:online"
AMAZING_SEARCH = "perplexity/sonar-pro"

SYSTEM_PROMPT = """\
You are an expert forecaster specialising in probabilistic predictions on \
binary events (YES/NO outcomes).

Key principles:
- Consider base rates and historical precedents
- Weigh evidence quality and recency
- Account for uncertainty and missing information
- Avoid extreme predictions (0 or 1) unless evidence is overwhelming
- Use the full probability range: 0.0 (impossible) to 1.0 (certain)"""


def build_user_prompt(title: str, description: str, context: str | None) -> str:
    """Final-synthesis prompt — asks for the strict output contract."""
    return f"""\
**Event to Forecast:**
{title}

**Full Description:**
{description}

**Research Context:**
{context if context else "No additional research context available."}

**Your Task:**
Estimate the probability (0.0 to 1.0) that this event will resolve YES.

Consider:
1. What is the base rate for similar events?
2. What specific evidence supports or contradicts this outcome?
3. What uncertainties or unknowns remain?

**Required Output Format:**
PREDICTION: [number between 0.0 and 1.0]
CONVICTION: [number between 0.0 and 1.0 indicating your confidence in the prediction value]
REASONING: [2-4 sentences explaining your probability estimate, key factors considered, and main uncertainties]"""


def parse_llm_output(text: str) -> tuple[float, float | None, str]:
    """Parse PREDICTION / CONVICTION / REASONING from model output text."""
    prediction = 0.5
    confidence: float | None = None
    reasoning = "No reasoning provided."
    for line in text.strip().split("\n"):
        cleaned = line.strip().strip("*`_ ").strip()
        if cleaned.startswith("PREDICTION:"):
            try:
                prediction = max(0.0, min(1.0, float(cleaned.replace("PREDICTION:", "").strip())))
            except ValueError:
                pass
        elif cleaned.startswith("CONVICTION:"):
            try:
                confidence = max(0.0, min(1.0, float(cleaned.replace("CONVICTION:", "").strip())))
            except ValueError:
                pass
        elif cleaned.startswith("REASONING:"):
            reasoning = cleaned.replace("REASONING:", "").strip()
    if prediction == 0.5:
        match = re.search(r"(?im)\bPREDICTION\b\s*:\s*([0-9]*\.?[0-9]+)", text)
        if match:
            try:
                prediction = max(0.0, min(1.0, float(match.group(1))))
            except ValueError:
                pass
    if confidence is None:
        match = re.search(r"(?im)\b(?:CONVICTION|CONFIDENCE)\b\s*:\s*([0-9]*\.?[0-9]+)", text)
        if match:
            try:
                confidence = max(0.0, min(1.0, float(match.group(1))))
            except ValueError:
                pass
    if reasoning == "No reasoning provided.":
        match = re.search(r"(?is)\bREASONING\b\s*:\s*(.+)$", text)
        if match:
            reasoning = match.group(1).strip()
    return prediction, confidence, reasoning


def extract_text(raw: dict) -> str:
    """Pull the assistant text out of a raw OpenRouter chat response dict."""
    if not isinstance(raw, dict):
        return ""
    output = raw.get("output")
    if isinstance(output, str) and output.strip():
        return output
    choices = raw.get("choices", [])
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message", {})
        if isinstance(message, dict):
            content = message.get("content", "")
            content = content if isinstance(content, str) else ""
            # Reasoning models (e.g. deepseek-r1) often return their answer in a
            # separate `reasoning`/`reasoning_content` field with `content` empty
            # or thin. Fall back to / append the reasoning so the PREDICTION line
            # is never lost to a parse default.
            reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
            reasoning = reasoning if isinstance(reasoning, str) else ""
            if content.strip():
                return content if "PREDICTION" in content.upper() else (content + "\n" + reasoning)
            return reasoning
    return ""


def chat(ctx, model: str, messages: list[dict], *, max_tokens: int = 1024,
         temperature: float = 0.2) -> str:
    """One chat_completion call through the gateway → assistant text.

    Returns "" on failure so callers can degrade gracefully rather than crash
    the whole run.
    """
    try:
        raw = ctx.call_provider("openrouter", "chat_completion", {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
    except Exception:
        return ""
    return extract_text(raw)


def web_search(ctx, model: str, query: str, *, max_tokens: int = 700) -> str:
    """One web-search call (online/web-native model) → answer text.

    The query is phrased as a research instruction so the web-native model
    returns current, cited facts rather than a refusal.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant. Answer with current, factual, "
                "dated information and cite concrete sources where possible. "
                "Be concise."
            ),
        },
        {"role": "user", "content": query},
    ]
    return chat(ctx, model, messages, max_tokens=max_tokens, temperature=0.1)
