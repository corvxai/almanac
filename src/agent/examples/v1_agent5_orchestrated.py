"""v1 Agent 5 — ORCHESTRATED (top of the capability ladder).

An iterative belief loop that mirrors the "Colombia" belief-path pattern:

  prior  ->  search  ->  update  ->  identify gap  ->  search gap  ->  update
         ->  final good-LLM synthesis  ->  final belief

Every update is recorded with an ``intermediate_probability`` so the trace
carries an explicit, timeblocked belief path (multiple updates), not just a
single endpoint. This is the agent whose trace is most useful to score *and*
to ablate.
"""

from __future__ import annotations

import re
from uuid import UUID

from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.schemas import AgentResult, ReasoningStepType

from src.agent.examples._v1_common import (
    AMAZING_SEARCH,
    BASIC_LLM,
    GOOD_LLM,
    SYSTEM_PROMPT,
    build_user_prompt,
    chat,
    parse_llm_output,
    web_search,
)


def _interim_probability(ctx: ForecastingContext, title: str, description: str,
                         context: str, prior: float) -> tuple[float, str]:
    """One cheap basic-LLM belief update → (probability, one-line rationale)."""
    user = (
        f"Question: {title}\n{description}\n\n"
        f"Your current prior probability of YES is {prior:.2f}.\n\n"
        f"New evidence:\n{context if context else '(none)'}\n\n"
        "Update your belief. Reply in exactly this format:\n"
        "PROB: <number 0.0-1.0>\n"
        "WHY: <one sentence>"
    )
    text = chat(ctx, BASIC_LLM, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ], max_tokens=300, temperature=0.1)

    prob = prior
    m = re.search(r"(?im)\bPROB\b\s*:\s*([0-9]*\.?[0-9]+)", text)
    if m:
        try:
            prob = max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            pass
    why = ""
    m2 = re.search(r"(?is)\bWHY\b\s*:\s*(.+)$", text)
    if m2:
        why = m2.group(1).strip().split("\n")[0]
    return prob, why


class V1Agent5Orchestrated(BaseAgent):
    agent_id = UUID("a0000005-0000-4000-8000-000000000005")
    agent_version = "1.0.0"

    def predict(self, ctx: ForecastingContext) -> AgentResult:
        title = ctx.event_title
        description = ctx.event_description

        # --- Step 0: prior belief from parametric knowledge (no search) ---
        prior, prior_why = _interim_probability(ctx, title, description, "", 0.5)
        ctx.record_reasoning_step(
            ReasoningStepType.PRIOR_UPDATE,
            reasoning_text=f"Initial prior from base knowledge. {prior_why}",
            intermediate_probability=prior,
            inference_model_used=BASIC_LLM,
        )

        # --- Step 1: search current state, then update ---
        s1 = web_search(
            ctx, AMAZING_SEARCH,
            f"Current state and latest developments for: {title}. {description}",
            max_tokens=600,
        )
        ctx.record_reasoning_step(
            ReasoningStepType.EVIDENCE_GATHERING,
            reasoning_text=f"Searched current state.\n{s1[:500] if s1 else '(no results)'}",
            provider_id="openrouter",
            inference_model_used=AMAZING_SEARCH,
        )
        belief, why1 = _interim_probability(ctx, title, description, s1, prior)
        ctx.record_reasoning_step(
            ReasoningStepType.PRIOR_UPDATE,
            reasoning_text=f"Updated on current-state evidence. {why1}",
            intermediate_probability=belief,
            confidence_delta=belief - prior,
            inference_model_used=BASIC_LLM,
        )

        # --- Step 2: identify the biggest remaining gap ---
        gap = chat(ctx, BASIC_LLM, [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Question: {title}\n{description}\n\nWhat we know so far:\n{s1}\n\n"
                f"Current belief: {belief:.2f}. What is the SINGLE most important "
                "missing piece of information that would most change this estimate? "
                "Reply with one specific search query only."
            )},
        ], max_tokens=120, temperature=0.2).strip()
        gap_query = (gap.split("\n")[0] if gap else f"key risk factors for {title}")
        ctx.record_reasoning_step(
            ReasoningStepType.SYNTHESIS,
            reasoning_text=f"Identified information gap; next query: {gap_query}",
            intermediate_probability=belief,
            inference_model_used=BASIC_LLM,
        )

        # --- Step 3: search the gap, then update again ---
        s2 = web_search(ctx, AMAZING_SEARCH, gap_query, max_tokens=600)
        ctx.record_reasoning_step(
            ReasoningStepType.EVIDENCE_GATHERING,
            reasoning_text=f"Searched the gap.\n{s2[:500] if s2 else '(no results)'}",
            provider_id="openrouter",
            inference_model_used=AMAZING_SEARCH,
        )
        belief2, why2 = _interim_probability(
            ctx, title, description, f"{s1}\n\n{s2}", belief,
        )
        ctx.record_reasoning_step(
            ReasoningStepType.PRIOR_UPDATE,
            reasoning_text=f"Updated on gap-filling evidence. {why2}",
            intermediate_probability=belief2,
            confidence_delta=belief2 - belief,
            inference_model_used=BASIC_LLM,
        )

        # --- Step 4: final good-LLM synthesis over the full belief path ---
        full_context = (
            f"[current_state]\n{s1}\n\n[gap: {gap_query}]\n{s2}\n\n"
            f"[belief path] prior={prior:.2f} -> after_search1={belief:.2f} "
            f"-> after_search2={belief2:.2f}"
        )
        content = chat(ctx, GOOD_LLM, [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(
                title, description, context=full_context,
            )},
        ], max_tokens=6000, temperature=0.2)

        if not content:
            # degrade to the last interim belief rather than crash
            return AgentResult(
                prediction=belief2, confidence=0.4,
                reasoning=("Final reasoning model unavailable; returning the "
                           f"last updated belief {belief2:.2f}."),
            )

        prediction, confidence, reasoning = parse_llm_output(content)
        ctx.record_reasoning_step(
            ReasoningStepType.FINAL_ASSIGNMENT,
            reasoning_text=reasoning,
            intermediate_probability=prediction,
            confidence_delta=prediction - belief2,
            inference_model_used=GOOD_LLM,
        )

        return AgentResult(prediction=prediction, confidence=confidence,
                           reasoning=reasoning)
