"""Claude agent — uses Anthropic's Messages API with the web_search server
tool for grounded, citation-backed predictions.

Two-phase flow:
1. Research: pull Polymarket data for a quantitative anchor
2. Inference: call Claude with web_search enabled so the model itself
   retrieves and cites current information. The gateway extracts
   structured citations from Claude's response.

This agent benefits from Claude's HIGH visibility tier — the gateway
gets full search results and per-sentence citation links.
"""

from __future__ import annotations

import logging
import re
from uuid import UUID

from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.schemas import AgentResult

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """\
You are an expert forecaster making calibrated probabilistic predictions.

You have access to a web_search tool — use it to find recent, relevant \
information about the event before forming your estimate.

Key principles:
- Ground predictions in concrete evidence, not speculation
- Consider base rates and reference classes
- Weigh source credibility and recency
- Be explicit about what shifts your probability away from a prior
- Avoid extremes unless evidence is overwhelming"""


def _build_user_prompt(title: str, description: str, market_context: str) -> str:
    return f"""\
**Event to Forecast:**
{title}

**Full Description:**
{description}

**Market Data:**
{market_context if market_context else "No market data available."}

**Instructions:**
1. Use your web_search tool to find the latest information about this event
2. Synthesise market data and search results
3. Produce a calibrated probability estimate

**Required Output Format:**
PREDICTION: [number between 0.0 and 1.0]
CONVICTION: [number between 0.0 and 1.0 indicating your confidence in the prediction value]
REASONING: [2-4 sentences citing the key evidence, noting areas of agreement \
and disagreement between sources, and explaining your final estimate]"""


def _parse_llm_output(text: str) -> tuple[float, float | None, str]:
    prediction = 0.5
    confidence = None
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


def _extract_text_from_claude_response(raw: dict) -> str:
    """Pull text from Claude's structured content blocks or output text."""
    content = raw.get("content", [])
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    if parts:
        return "\n".join(parts)
    output = raw.get("output")
    if isinstance(output, str):
        return output
    return ""


class ClaudeAgent(BaseAgent):
    agent_id = UUID("c3d4e5f6-a7b8-9012-cdef-123456789012")
    agent_version = "0.1.0"

    def predict(self, ctx: ForecastingContext) -> AgentResult:
        # --- Phase 1: Market anchor ---
        market_context = ""
        try:
            market = ctx.call_provider("polymarket", "get_market", {
                "market_slug": ctx.event_title.lower().replace(" ", "-"),
            })
            price = market.get("price", 0.5)
            volume = market.get("volume_24h", 0)
            market_context = (
                f"Polymarket implied probability: {price:.4f} "
                f"(24h volume: ${volume:,.0f})"
            )
        except Exception:
            pass

        # --- Phase 2: Claude with web search ---
        messages = [
            {"role": "user", "content": _build_user_prompt(
                ctx.event_title, ctx.event_description, market_context,
            )},
        ]

        try:
            llm_response = ctx.call_provider("anthropic", "messages", {
                "model": MODEL,
                "system": SYSTEM_PROMPT,
                "messages": messages,
                "max_tokens": 1024,
                # Server tool type must match Anthropic docs (not plain "web_search").
                "tools": [{
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 3,
                }],
            })
        except Exception as exc:
            logger.warning("Claude messages failed (%s): %s", MODEL, exc)
            return AgentResult(
                prediction=0.5,
                reasoning="Claude API unavailable. Returning neutral prediction.",
            )

        text = _extract_text_from_claude_response(llm_response)
        prediction, confidence, reasoning = _parse_llm_output(text)

        return AgentResult(
            prediction=prediction,
            confidence=confidence,
            reasoning=reasoning,
        )
