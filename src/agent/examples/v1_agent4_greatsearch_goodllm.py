"""v1 Agent 4 — AMAZING search + GOOD LLM.

Same strong multi-query search as Agent 3, then one GOOD reasoning-model call
(deepseek-r1) to synthesize. Isolates the value of a stronger reasoning engine
on top of identical evidence.
"""

from __future__ import annotations

from uuid import UUID

from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.schemas import AgentResult, ReasoningStepType

from src.agent.examples._v1_common import (
    AMAZING_SEARCH,
    GOOD_LLM,
    SYSTEM_PROMPT,
    build_user_prompt,
    chat,
    parse_llm_output,
    web_search,
)
from src.agent.examples.v1_agent3_greatsearch_basicllm import _subqueries


class V1Agent4GreatSearchGoodLLM(BaseAgent):
    agent_id = UUID("a0000004-0000-4000-8000-000000000004")
    agent_version = "1.0.0"

    def predict(self, ctx: ForecastingContext) -> AgentResult:
        # --- Phase 1: three targeted web searches (same as Agent 3) ---
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

        # --- Phase 2: GOOD reasoning-model synthesis ---
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(
                ctx.event_title, ctx.event_description, context=research,
            )},
        ]
        content = chat(ctx, GOOD_LLM, messages, max_tokens=6000, temperature=0.2)
        if not content:
            return AgentResult(prediction=0.5, confidence=0.0,
                               reasoning="Reasoning model unavailable; defaulting to 0.5.")

        prediction, confidence, reasoning = parse_llm_output(content)

        ctx.record_reasoning_step(
            ReasoningStepType.FINAL_ASSIGNMENT,
            reasoning_text=reasoning,
            intermediate_probability=prediction,
            inference_model_used=GOOD_LLM,
        )

        return AgentResult(prediction=prediction, confidence=confidence,
                           reasoning=reasoning)
