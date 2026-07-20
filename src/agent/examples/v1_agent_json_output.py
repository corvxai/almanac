"""v1 JSON-output reference agent — the minimal validated-JSON contract.

The shared contract (``Forecast`` model, ``build_json_prompt``, ``parse_forecast``,
``request_forecast``) lives in ``_v1_common``; this file is the smallest agent that
uses it, kept as the canonical reference for miners.

Why it exists: the old contract asked the provider LLM for a
``PREDICTION/CONVICTION/REASONING`` text template and scraped it with a regex that
silently returned 0.5 on a miss and clamped "62%" to 1.0. This agent instead asks
for a single JSON object ``{reasoning, prediction, confidence}`` (reasoning first,
so the model reasons before committing the number) and validates it with pydantic.

Contract alignment (verified against base + validator):
* ``confidence`` is optional (stored, unscored); the success path sets it from the
  forecast, and a missing value never invalidates the prediction.
* Fail closed IN-BAND: on unrecoverable model output, return a valid neutral
  forecast — a structurally-valid ``AgentResult`` with ``prediction=0.5`` and a
  single-``final`` belief path — rather than ``raise`` (a sandbox crash path) or a
  silent, unvalidated 0.5. It is scored normally.
* No ``record_reasoning_step``: the Docker sandbox discards agent-recorded steps
  (``orchestrator``: agent_steps=None); the reasoning_chain is validator-assembled.
"""

from __future__ import annotations

from uuid import UUID

from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.schemas import AgentResult
from src.agent.examples._v1_common import (
    BASIC_LLM,
    belief_path_from_forecast,
    belief_path_single_final,
    parse_forecast,
    request_forecast,
)


class V1AgentJsonOutput(BaseAgent):
    agent_id = UUID("a0000009-0000-4000-8000-000000000009")
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
        return AgentResult(prediction=fc.prediction, confidence=fc.confidence,
                           reasoning=fc.reasoning, beliefPath=belief_path_from_forecast(fc))


if __name__ == "__main__":
    # Offline smoke test — no gateway needed. Shows the shared parser fixing the
    # cases the regex got wrong, and rejecting (not defaulting) on garbage.
    cases = {
        "clean":        '{"reasoning": "Base rate low.", "prediction": 0.12, "confidence": 0.6}',
        "fenced":       '```json\n{"reasoning": "x", "prediction": 0.62, "confidence": 0.7}\n```',
        "percent":      '{"reasoning": "x", "prediction": "62%", "confidence": "80%"}',
        "prose_around": 'Sure!\n{"reasoning": "x", "prediction": 0.4, "confidence": 0.5}\nHope that helps.',
        "empty":        '',
        "refusal":      "I'm sorry, I can't help with that.",
        "out_of_range": '{"reasoning": "x", "prediction": 1.7, "confidence": 0.5}',
    }
    for name, raw in cases.items():
        fc = parse_forecast(raw)
        if fc is None:
            print(f"{name:14s} -> INVALID (rejected, not defaulted)")
        else:
            print(f"{name:14s} -> prediction={fc.prediction:.3f} confidence={fc.confidence:.3f}")
