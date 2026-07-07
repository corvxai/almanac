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
    SYSTEM_PROMPT,
    build_user_prompt,
    chat,
    parse_llm_output,
)


class V1Agent1LLMOnly(BaseAgent):
    agent_id = UUID("a0000001-0000-4000-8000-000000000001")
    agent_version = "1.0.0"

    def predict(self, ctx: ForecastingContext) -> AgentResult:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(
                ctx.event_title, ctx.event_description, context=None,
            )},
        ]

        content = chat(ctx, BASIC_LLM, messages, max_tokens=1024, temperature=0.2)
        if not content:
            return AgentResult(prediction=0.5, confidence=0.0,
                               reasoning="LLM unavailable; defaulting to 0.5.")

        prediction, confidence, reasoning = parse_llm_output(content)

        ctx.record_reasoning_step(
            ReasoningStepType.FINAL_ASSIGNMENT,
            reasoning_text=(
                "No external evidence gathered. Probability assigned purely "
                f"from the model's parametric knowledge. {reasoning}"
            ),
            intermediate_probability=prediction,
            inference_model_used=BASIC_LLM,
        )

        return AgentResult(prediction=prediction, confidence=confidence,
                           reasoning=reasoning)
