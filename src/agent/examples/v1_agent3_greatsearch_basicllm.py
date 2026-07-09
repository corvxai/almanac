"""v1 Agent 3 — AMAZING search + BASIC LLM.

Three TARGETED web-search calls against a strong web-native search model
(distinct sub-questions: current state, recent signals, base rate), then one
basic-LLM call to synthesize. Better evidence, same reasoning engine as
Agent 2 — isolates the value of search quality.
"""

from __future__ import annotations

from uuid import UUID

from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.schemas import AgentResult, ReasoningStepType

from src.agent.examples._v1_common import (
    AMAZING_SEARCH,
    BASIC_LLM,
    request_forecast,
    web_search,
)


def _subqueries(title: str, description: str) -> list[tuple[str, str]]:
    return [
        ("current_state",
         f"What is the CURRENT state of the world relevant to: {title}? "
         f"({description}) Give the latest concrete facts and dates."),
        ("recent_signals",
         f"What are the most RECENT signals, statements, or data points in the "
         f"last few weeks that bear on whether this resolves YES: {title}?"),
        ("base_rate",
         f"What is the historical BASE RATE for outcomes like this: {title}? "
         f"Give frequencies / precedents."),
    ]


class V1Agent3GreatSearchBasicLLM(BaseAgent):
    agent_id = UUID("a0000003-0000-4000-8000-000000000003")
    agent_version = "1.0.0"

    def predict(self, ctx: ForecastingContext) -> AgentResult:
        # --- Phase 1: three targeted web searches ---
        findings: list[str] = []
        for label, q in _subqueries(ctx.event_title, ctx.event_description):
            answer = web_search(ctx, AMAZING_SEARCH, q, max_tokens=600)
            findings.append(f"[{label}]\n{answer}" if answer else f"[{label}]\n(no results)")
            ctx.record_reasoning_step(
                ReasoningStepType.GAP_QUERY,
                reasoning_text=(
                    f"Targeted search ({label}) via amazing search model.\n"
                    f"{answer[:500] if answer else '(no results)'}"
                ),
                provider_id="openrouter",
                inference_model_used=AMAZING_SEARCH,
            )

        research = "\n\n".join(findings)

        # --- Phase 2: basic-LLM synthesis (validated JSON contract) ---
        try:
            fc = request_forecast(
                ctx, BASIC_LLM, ctx.event_title, ctx.event_description, context=research,
            )
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
