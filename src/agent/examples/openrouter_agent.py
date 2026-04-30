"""OpenRouter agent — gathers market data + news, then calls an LLM via
OpenRouter to synthesise a prediction.

Two-phase flow:
1. Research: pull Polymarket data + web search
2. Inference: feed gathered context to an LLM via OpenRouter with a
   model fallback chain
"""

from __future__ import annotations

from uuid import UUID

from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.schemas import AgentResult

MODELS = [
    #"google/gemini-2.5-flash",
    #"anthropic/claude-sonnet-4-6",
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

Consider:
1. What is the base rate for similar events?
2. What specific evidence supports or contradicts this outcome?
3. What uncertainties or unknowns remain?

**Required Output Format:**
PREDICTION: [number between 0.0 and 1.0]
CONVICTION: [number between 0.0 and 1.0 indicating your confidence in the prediction value]
REASONING: [2-4 sentences explaining your probability estimate, key factors considered, and main uncertainties]"""


def _parse_llm_output(text: str) -> tuple[float, float | None, str]:
    prediction = 0.5
    confidence = None
    reasoning = "No reasoning provided."
    for line in text.strip().split("\n"):
        if line.startswith("PREDICTION:"):
            try:
                prediction = max(0.0, min(1.0, float(line.replace("PREDICTION:", "").strip())))
            except ValueError:
                pass
        elif line.startswith("CONVICTION:"):
            try:
                confidence = max(0.0, min(1.0, float(line.replace("CONVICTION:", "").strip())))
            except ValueError:
                pass
        elif line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()
    return prediction, confidence, reasoning


class OpenRouterAgent(BaseAgent):
    agent_id = UUID("b2c3d4e5-f6a7-8901-bcde-f12345678901")
    agent_version = "0.1.0"

    def predict(self, ctx: ForecastingContext) -> AgentResult:
        # --- Phase 1: Research ---
        context_parts: list[str] = []

        market = ctx.call_provider("polymarket", "get_market", {
            "market_slug": ctx.event_title.lower().replace(" ", "-"),
        })
        price = market.get("price", 0.5)
        volume = market.get("volume_24h", 0)
        context_parts.append(
            f"Polymarket: price={price:.4f}, 24h_volume={volume:,.0f}, "
            f"implied_probability={market.get('probability', price):.4f}"
        )

        news = ctx.call_provider("web_search", "search", {
            "query": ctx.event_title,
            "num_results": 5,
        })
        for r in news.get("results", []):
            snippet = r.get("snippet", "")
            source = r.get("source", r.get("url", ""))
            if snippet:
                context_parts.append(f"[{source}] {snippet}")

        research_context = "\n".join(context_parts)

        # --- Phase 2: LLM inference via OpenRouter ---
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(
                ctx.event_title, ctx.event_description, research_context,
            )},
        ]

        llm_response = None
        for model in MODELS:
            try:
                llm_response = ctx.call_provider("openrouter", "chat_completion", {
                    "model": model,
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": 1024,
                })
                break
            except Exception:
                continue

        if llm_response is None:
            return AgentResult(
                prediction=price,
                reasoning=f"LLM unavailable. Falling back to market price: {price:.4f}",
            )

        content = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        prediction, confidence, reasoning = _parse_llm_output(content)

        return AgentResult(
            prediction=prediction,
            confidence=confidence,
            reasoning=reasoning,
        )
