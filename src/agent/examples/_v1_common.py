"""Shared helpers for the v1 capability-ladder agents (agents 1-6).

These agents demonstrate increasing forecasting capability on the SAME event so
their traces can be compared apples-to-apples. They all share the same
**validated JSON** output contract (``Forecast``) and parser. The only thing
that changes across the ladder is *how much real external evidence* each one
gathers before it reasons.

Output contract (replaces the old NL ``PREDICTION/CONVICTION/REASONING`` template
+ regex, which silently returned 0.5 on a parse miss and clamped "62%" to 1.0):

* The provider LLM is asked for a single JSON object ``{reasoning, prediction,
  confidence}`` — ``reasoning`` FIRST so the model reasons before committing the
  number (field-generation order follows schema order).
* ``prediction``/``confidence`` are REQUIRED floats in [0,1]. Range is enforced in
  pydantic (``Forecast``), not a JSON schema, because OpenAI/Anthropic ignore
  JSON-Schema min/max. ``"62%"`` is coerced to 0.62; unparseable values are
  REJECTED, never defaulted.
* ``request_forecast`` retries once (Instructor-style, appends the validation
  error) and raises ``ValueError`` if it still can't get a valid object. Agents
  catch that and fail closed IN-BAND (return ``AgentResult`` with
  ``confidence=None`` → the validator marks it invalid) rather than crashing or
  emitting a silent 0.5.

All external calls go through ``ctx.call_provider("openrouter", "chat_completion",
...)``. A "web search call" is a model id with the ``:online`` suffix or a
web-native model (e.g. ``perplexity/sonar-pro``); a "reasoning call" is a plain
model id.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, ValidationError, field_validator

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

# System prompt for the final-synthesis call: demands the strict JSON contract.
JSON_SYSTEM_PROMPT = SYSTEM_PROMPT + """

Return your answer as a SINGLE JSON object and NOTHING else — no prose before or
after, no markdown fences. The object MUST have exactly these keys, in this order:
  "reasoning":  string, 2-4 sentences. Reason FIRST — base rates, the specific
                evidence for and against, and the main uncertainties.
  "prediction": number in [0.0, 1.0] — P(event resolves YES). A decimal, NOT a
                percentage. This is the CONCLUSION of your reasoning.
  "confidence": number in [0.0, 1.0] — how sure you are of that probability."""


def build_json_prompt(title: str, description: str, context: str | None) -> str:
    """Final-synthesis prompt — asks for the strict JSON output contract."""
    return f"""\
**Event to Forecast:** {title}

**Full Description:**
{description}

**Research Context:**
{context if context else "No additional research context available."}

Return the JSON object now (reasoning first, then prediction, then confidence)."""


# --- the provider-LLM output contract ---------------------------------------

def _coerce_unit(v: object) -> float:
    """Coerce a model-supplied probability to a float BEFORE range validation.
    ``"62%" -> 0.62``, ``"0.62." -> 0.62``, numeric strings -> float. Raises on
    anything with no parseable number so it is REJECTED, never silently defaulted.
    """
    if isinstance(v, bool):  # bool is an int subclass — reject explicitly
        raise ValueError("boolean is not a probability")
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().rstrip(".").strip()
        is_pct = s.endswith("%")
        s = s.rstrip("%").strip()
        m = re.search(r"-?\d*\.?\d+", s)
        if not m:
            raise ValueError(f"no number in {v!r}")
        num = float(m.group(0))
        return num / 100.0 if is_pct else num
    raise ValueError(f"unparseable probability {v!r}")


class Forecast(BaseModel):
    """Validated final forecast. ``reasoning`` first — see module docstring."""

    model_config = {"extra": "forbid"}

    reasoning: str = Field(min_length=1)
    prediction: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("prediction", "confidence", mode="before")
    @classmethod
    def _unit(cls, v: object) -> float:
        return _coerce_unit(v)


# --- robust JSON parsing -----------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json_blob(text: str) -> str | None:
    """Pull the first balanced ``{...}`` object out of model text, tolerating
    markdown fences and surrounding prose."""
    if not text or not text.strip():
        return None
    t = text.strip()
    fence = _FENCE.search(t)
    if fence:
        t = fence.group(1).strip()
    start = t.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                return t[start:i + 1]
    return None


def parse_forecast(text: str) -> Forecast | None:
    """Return a validated ``Forecast`` or ``None`` — never a default."""
    blob = _extract_json_blob(text)
    if blob is None:
        return None
    try:
        return Forecast.model_validate_json(blob)
    except ValidationError:
        return None


def request_json(
    ctx,
    model: str,
    messages: list[dict],
    *,
    max_retries: int = 1,
    max_tokens: int = 900,
) -> Forecast:
    """Run a chat that must return a ``Forecast`` JSON object. Retries once with
    the failure appended (Instructor-style self-correction). Raises ``ValueError``
    if no valid object is produced — callers fail closed IN-BAND (see module
    docstring). ``messages`` must already demand the {reasoning, prediction,
    confidence} contract."""
    messages = list(messages)
    last = ""
    for _ in range(max_retries + 1):
        text = chat(ctx, model, messages, max_tokens=max_tokens, temperature=0.2)
        forecast = parse_forecast(text)
        if forecast is not None:
            return forecast
        last = text
        messages.append({"role": "assistant", "content": text or "(empty response)"})
        messages.append({
            "role": "user",
            "content": (
                "That was not a valid JSON object with keys reasoning, "
                "prediction (0..1 decimal) and confidence (0..1 decimal). "
                "Return ONLY that JSON object now."
            ),
        })
    raise ValueError(
        f"no valid forecast after {max_retries + 1} attempts; "
        f"last output: {last[:200]!r}"
    )


def request_forecast(
    ctx,
    model: str,
    title: str,
    description: str,
    context: str | None,
    *,
    max_retries: int = 1,
    max_tokens: int = 900,
) -> Forecast:
    """Standard forecast: ask ``model`` for a validated ``Forecast`` on the event."""
    messages = [
        {"role": "system", "content": JSON_SYSTEM_PROMPT},
        {"role": "user", "content": build_json_prompt(title, description, context)},
    ]
    return request_json(ctx, model, messages, max_retries=max_retries, max_tokens=max_tokens)


# --- provider plumbing (unchanged) -------------------------------------------

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
            # or thin. Fall back to / append the reasoning so the JSON object is
            # never lost to a parse default.
            reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
            reasoning = reasoning if isinstance(reasoning, str) else ""
            if content.strip():
                return content if "{" in content else (content + "\n" + reasoning)
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
