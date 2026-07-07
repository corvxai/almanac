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
    SYSTEM_PROMPT,
    build_user_prompt,
    chat,
    parse_llm_output,
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
                ReasoningStepType.EVIDENCE_GATHERING,
                reasoning_text=(
                    f"Targeted search ({label}) via amazing search model.\n"
                    f"{answer[:500] if answer else '(no results)'}"
                ),
                provider_id="openrouter",
                inference_model_used=AMAZING_SEARCH,
            )

        research = "\n\n".join(findings)

        # --- Phase 2: basic-LLM synthesis ---
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(
                ctx.event_title, ctx.event_description, context=research,
            )},
        ]
        content = chat(ctx, BASIC_LLM, messages, max_tokens=1024, temperature=0.2)
        if not content:
            return AgentResult(prediction=0.5, confidence=0.0,
                               reasoning="LLM unavailable; defaulting to 0.5.")

        prediction, confidence, reasoning = parse_llm_output(content)

        ctx.record_reasoning_step(
            ReasoningStepType.FINAL_ASSIGNMENT,
            reasoning_text=reasoning,
            intermediate_probability=prediction,
            inference_model_used=BASIC_LLM,
        )

        return AgentResult(prediction=prediction, confidence=confidence,
                           reasoning=reasoning)
