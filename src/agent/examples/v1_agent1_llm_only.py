"""v1 Agent 1 — LLM ONLY (bottom of the capability ladder).

Makes NO web search. A single plain basic-LLM call using only the model's own
parametric knowledge. This is the control case: a prediction with zero
witnessed external evidence in the trace.
"""

from __future__ import annotations

from uuid import UUID

from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.schemas import AgentResult, ReasoningStepType

from src.agent.examples._v1_common import (
    BASIC_LLM,
    belief_path_from_forecast,
    belief_path_single_final,
    request_forecast,
)


class V1Agent1LLMOnly(BaseAgent):
    agent_id = UUID("a0000001-0000-4000-8000-000000000001")
    agent_version = "1.0.0"

    def predict(self, ctx: ForecastingContext) -> AgentResult:
        try:
            fc = request_forecast(
                ctx, BASIC_LLM, ctx.event_title, ctx.event_description, context=None,
            )
        except ValueError as exc:
            reason = f"fail-closed neutral forecast (no valid model output): {exc}"
            return AgentResult(prediction=0.5, confidence=None, reasoning=reason,
                               beliefPath=belief_path_single_final(0.5, reason))

        ctx.record_reasoning_step(
            ReasoningStepType.BELIEF_UPDATE,
            reasoning_text=(
                "No external evidence gathered. Probability assigned purely "
                f"from the model's parametric knowledge. {fc.reasoning}"
            ),
            intermediate_probability=fc.prediction,
            inference_model_used=BASIC_LLM,
        )

        return AgentResult(prediction=fc.prediction, confidence=fc.confidence,
                           reasoning=fc.reasoning, beliefPath=belief_path_from_forecast(fc))
