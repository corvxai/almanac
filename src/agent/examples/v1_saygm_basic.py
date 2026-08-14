"""v1 SayGM basic — BASIC search (OpenRouter) + SayGM LLM.

Same search path as Agent 2 (one broad web-search via the basic search model),
then synthesis on the SayGM provider with ``glm-5.2`` instead of the
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

SAYGM_LLM = "glm-5.2"
SAYGM_PROVIDER = "saygm"


class V1SaygmBasic(BaseAgent):
    agent_id = UUID("e000000a-0000-4000-8000-00000000000a")
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

        # --- Phase 2: SayGM LLM synthesis (validated JSON contract) ---
        try:
            fc = request_forecast(
                ctx, SAYGM_LLM, ctx.event_title, ctx.event_description,
                context=research, provider_id=SAYGM_PROVIDER, max_tokens=2000,
            )
        except ValueError as exc:
            reason = f"fail-closed neutral forecast (no valid model output): {exc}"
            return AgentResult(prediction=0.5, confidence=None, reasoning=reason,
                               beliefPath=belief_path_single_final(0.5, reason))

        ctx.record_reasoning_step(
            ReasoningStepType.BELIEF_UPDATE,
            reasoning_text=fc.reasoning,
            intermediate_probability=fc.prediction,
            provider_id=SAYGM_PROVIDER,
            inference_model_used=SAYGM_LLM,
        )

        return AgentResult(prediction=fc.prediction, confidence=fc.confidence,
                           reasoning=fc.reasoning, beliefPath=belief_path_from_forecast(fc))
