"""v1 Agent 2 — BASIC search + BASIC LLM.

One single generic web-search call (basic search model, one broad query built
from the event title), then one basic-LLM call to reason over the returned
text. Demonstrates the value of a single piece of witnessed external evidence.
"""

from __future__ import annotations

from uuid import UUID

from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.schemas import AgentResult, ReasoningStepType

from src.agent.examples._v1_common import (
    BASIC_LLM,
    BASIC_SEARCH,
    belief_path_from_forecast,
    belief_path_single_final,
    request_forecast,
    web_search,
)


class V1Agent2Basic(BaseAgent):
    agent_id = UUID("a0000002-0000-4000-8000-000000000002")
    agent_version = "1.0.0"

    def predict(self, ctx: ForecastingContext) -> AgentResult:
        # --- Phase 1: one broad web search ---
        query = (
            f"Latest news and current status relevant to this question: "
            f"{ctx.event_title}. {ctx.event_description}"
        )
        research = web_search(ctx, BASIC_SEARCH, query, max_tokens=700)

        ctx.record_reasoning_step(
            ReasoningStepType.GAP_QUERY,
            reasoning_text=(
                "Ran one broad web search via the basic search model.\n"
                f"Findings: {research[:600] if research else '(no results)'}"
            ),
            provider_id="openrouter",
            inference_model_used=BASIC_SEARCH,
        )

        # --- Phase 2: basic-LLM synthesis (validated JSON contract) ---
        try:
            fc = request_forecast(
                ctx, BASIC_LLM, ctx.event_title, ctx.event_description, context=research,
            )
        except ValueError as exc:
            reason = f"fail-closed neutral forecast (no valid model output): {exc}"
            return AgentResult(prediction=0.5, confidence=None, reasoning=reason,
                               beliefPath=belief_path_single_final(0.5, reason))

        ctx.record_reasoning_step(
            ReasoningStepType.BELIEF_UPDATE,
            reasoning_text=fc.reasoning,
            intermediate_probability=fc.prediction,
            inference_model_used=BASIC_LLM,
        )

        return AgentResult(prediction=fc.prediction, confidence=fc.confidence,
                           reasoning=fc.reasoning, beliefPath=belief_path_from_forecast(fc))
