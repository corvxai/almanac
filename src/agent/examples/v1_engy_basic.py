"""v1 Engy basic — BASIC search (OpenRouter) + Engy LLM.

Same search path as Agent 2 (one broad web-search via the basic search model),
then synthesis on the Engy provider with ``qwen3.6-35b-a3b`` instead of the
OpenRouter basic LLM.
"""

from __future__ import annotations

from uuid import UUID

from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.schemas import AgentResult, ReasoningStepType

from src.agent.examples._v1_common import (
    BASIC_SEARCH,
    belief_path_from_forecast,
    belief_path_single_final,
    request_forecast,
    web_search,
)

ENGY_LLM = "qwen3.6-35b-a3b"
ENGY_PROVIDER = "engy"


class V1EngyBasic(BaseAgent):
    agent_id = UUID("e0000008-0000-4000-8000-000000000008")
    agent_version = "1.0.0"

    def predict(self, ctx: ForecastingContext) -> AgentResult:
        # --- Phase 1: one broad web search (same as Agent 2) ---
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

        # --- Phase 2: Engy LLM synthesis (validated JSON contract) ---
        try:
            fc = request_forecast(
                ctx, ENGY_LLM, ctx.event_title, ctx.event_description,
                context=research, provider_id=ENGY_PROVIDER, max_tokens=2000,
            )
        except ValueError as exc:
            reason = f"fail-closed neutral forecast (no valid model output): {exc}"
            return AgentResult(prediction=0.5, confidence=None, reasoning=reason,
                               beliefPath=belief_path_single_final(0.5, reason))

        ctx.record_reasoning_step(
            ReasoningStepType.BELIEF_UPDATE,
            reasoning_text=fc.reasoning,
            intermediate_probability=fc.prediction,
            provider_id=ENGY_PROVIDER,
            inference_model_used=ENGY_LLM,
        )

        return AgentResult(prediction=fc.prediction, confidence=fc.confidence,
                           reasoning=fc.reasoning, beliefPath=belief_path_from_forecast(fc))
