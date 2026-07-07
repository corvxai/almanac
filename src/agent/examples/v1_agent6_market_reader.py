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
    SYSTEM_PROMPT,
    chat,
    parse_llm_output,
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
            ReasoningStepType.EVIDENCE_GATHERING,
            reasoning_text=f"Market-price search via the basic online model.\n{market[:500]}",
            provider_id="openrouter",
            inference_model_used=BASIC_SEARCH,
        )

        # --- Extract the implied probability as a single number ---
        user = (
            "From the market data below, report the market-implied probability that the event "
            "resolves YES. Convert any percentage to a 0.0-1.0 number. Do not add your own forecast.\n\n"
            f"Event: {ctx.event_title}\n\nMarket data:\n{market}\n\n"
            "Reply in exactly this format:\n"
            "PREDICTION: <number 0.0-1.0, the market-implied probability>\n"
            "CONVICTION: <number 0.0-1.0>\n"
            "REASONING: <one sentence stating the market number and its source>"
        )
        text = chat(
            ctx, BASIC_LLM,
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            max_tokens=300, temperature=0.1,
        )
        prediction, confidence, reasoning = parse_llm_output(text)
        ctx.record_reasoning_step(
            ReasoningStepType.FINAL_ASSIGNMENT,
            reasoning_text=reasoning,
            intermediate_probability=prediction,
            inference_model_used=BASIC_LLM,
        )
        return AgentResult(prediction=prediction, confidence=confidence, reasoning=reasoning)
