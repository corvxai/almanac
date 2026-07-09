"""v1 Agent 6 - market reader.

Not a forecaster. It searches for the market-implied probability of the event and
reports that number. Two uses: it brings the real market price into the trace via a
witnessed web search (instead of a mocked market provider), and it is the copy-the-market
control, market-relative scoring should pay it close to nothing because it only echoes
what the market already says.
"""

from __future__ import annotations

from uuid import UUID

from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.schemas import AgentResult, ReasoningStepType

from src.agent.examples._v1_common import (
    BASIC_LLM,
    BASIC_SEARCH,
    request_json,
    web_search,
)


class V1Agent6MarketReader(BaseAgent):
    agent_id = UUID("a0000006-0000-4000-8000-000000000006")
    agent_version = "1.0.0"

    def predict(self, ctx: ForecastingContext) -> AgentResult:
        # --- Search specifically for the market-implied probability ---
        query = (
            "What is the current prediction-market implied probability that this resolves YES? "
            "Check Polymarket, Kalshi, and CME FedWatch and report the percentage. "
            f"Event: {ctx.event_title}. {ctx.event_description}"
        )
        market = web_search(ctx, BASIC_SEARCH, query, max_tokens=600)
        ctx.record_reasoning_step(
            ReasoningStepType.GAP_QUERY,
            reasoning_text=f"Market-price search via the basic online model.\n{market[:500]}",
            provider_id="openrouter",
            inference_model_used=BASIC_SEARCH,
        )

        # --- Extract the implied probability as the Forecast JSON contract ---
        # Custom prompt: report the MARKET number, not an independent forecast.
        messages = [
            {"role": "system", "content": (
                "You report prediction-market-implied probabilities as a strict JSON "
                "object and NOTHING else. Keys, in order: \"reasoning\" (one sentence "
                "stating the market number and its source), \"prediction\" (0.0-1.0 "
                "decimal, the market-implied probability — convert any percentage), "
                "\"confidence\" (0.0-1.0, how clearly the market data stated it). Do not "
                "add your own forecast."
            )},
            {"role": "user", "content": (
                f"Event: {ctx.event_title}\n\nMarket data:\n{market}\n\n"
                "Return the JSON object now."
            )},
        ]
        try:
            fc = request_json(ctx, BASIC_LLM, messages, max_tokens=400)
        except ValueError as exc:
            return AgentResult(prediction=0.5, confidence=None,
                               reasoning=f"INVALID: {exc}")

        ctx.record_reasoning_step(
            ReasoningStepType.BELIEF_UPDATE,
            reasoning_text=fc.reasoning,
            intermediate_probability=fc.prediction,
            inference_model_used=BASIC_LLM,
        )
        return AgentResult(prediction=fc.prediction, confidence=fc.confidence,
                           reasoning=fc.reasoning)
