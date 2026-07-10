# v1 implementation plan

Status of the v1 trace + scoring work on this branch: what has landed, what is in flight, what is left, and the order for the rest. Companion to `roadmap.md` (areas A-G) and `trace-schema.md`.

## Decisions locked (from review + schema sync)

1. **Agents return structured JSON, no regex.** The agent's own code produces a Pydantic-validated object. If it does not validate, the prediction is marked invalid and scores 0. The burden is on the miner; we ship base classes and examples so valid output takes minutes.
2. **Scoring = PR #15 as the MVP baseline.** The scoring sketches in `scoring-idea.md` move to phase-2 notes; the edge pillar in #15 already covers the Brier + PnL direction. Schema validity stays a boolean gate in MVP; graded schema scoring is a later phase.
3. **Scored now vs stored now.** Scored: outcome probabilities, predicted outcome, resolved outcome, market price at prediction, valid/invalid. Stored for the data collection (unscored in MVP): belief path, witnessed provider calls, sources/evidence, confidence, reasoning.
4. **Belief path is miner-written, stored from day one, scored later.** Validation now, grading in phase 2.

## Done on this branch

### Docs
- `design/roadmap.md` — the seven work areas (A-G) and build order.
- `design/trace-schema.md` — the v1 trace shape and assembler contract, with a worked example.
- `design/workflow-before-after.html` — miner and validator flows, before vs after.

### Example agents (capability ladder)
`src/agent/examples/v1_agent1..6` + `_v1_common.py`: six agents on the same event, from a single bare LLM call up to an orchestrated search-update-reason loop plus a market reader, so traces can be compared apples to apples. `scripts/run_five_v1_agents.py` runs the ladder.

### Area A: v1.0.0 schema + JSON agent output contract
The data-JSON changes, in one commit (`feat: v1 JSON agent output contract + v1.0.0 trace schema`):

**Agent output contract** (replaces the NL `PREDICTION/CONVICTION/...` template + regex parse, which silently defaulted to 0.5 on a parse miss and clamped "62%" to 1.0):
- The provider LLM is asked for a single JSON object `{reasoning, prediction, confidence}` — reasoning first, so the model reasons before committing the number.
- `prediction`/`confidence` are required floats in [0,1], enforced in Pydantic. `"62%"` coerces to 0.62; unparseable values are rejected, never defaulted.
- One retry that feeds the validation error back to the model; if it still fails, the agent fails closed in-band (`confidence=None` → validator marks the prediction invalid). No silent 0.5, no crash.

**Trace schema v1.0.0** (`src/core/schemas.py`, `trace_assembler.py`):
- Reasoning-step vocabulary collapsed to `prior` / `belief_update` / `gap_query`.
- Typed `Source` (url required) in `sources_accessed[]`, with `counts_toward_grounding` as the gateway's admissibility verdict; the duplicate evidence array is gone.
- Dropped the fields that were null across the corpus: `key_drivers`, `key_uncertainties`, `merkle_root`, `anchor_*`, `total_evidence_items`, `agent_interpretation`, `conflict_signals`.
- Added `market_price_at_prediction` to `execution_context`, so the number a miner is graded against sits inside the hashed trace.
- Added a `future_graph` seam reserving space for the phase-2 verification machinery (join, ablation), so enabling it later is a reprocess, not a schema break.
- The assembler no longer fills evidence refs positionally (the fake join); it emits an honest empty list until the real join (area D) exists.
- `TRACE_SCHEMA_VERSION = "1.0.0"`; contract tests in `tests/agent/test_v1_json_contract.py`.

## Next: extend the agent contract with the belief path

Agreed direction: the structured object the agent returns grows a `beliefPath` — an ordered list of steps, each `{step, type (prior/update/final), probability [0,1], text, usedCall?, usedSources?}`, ending in exactly one `final` whose probability equals `prediction`. Well-formedness is part of the validity gate (misaligned → invalid → 0); the path itself is stored, not scored, in MVP. A single-`final` path is valid, so the simplest agent stays a couple of lines.

Open points to confirm before this lands:
- `beliefPath` required, or optional-but-validated-when-present?
- Binary events only for MVP (the shape extends to a per-outcome map later)?
- Rounding tolerance for the final-step probability == `prediction` check.

## Remaining work areas (build order)

| Area | What | Status |
|---|---|---|
| A. Schema | v1.0.0 + agent JSON contract | **done** (belief-path extension pending the confirms above) |
| B. Witness spine | keep request bytes + stamp a correlation key per call | next |
| C. Source home + grounding | one typed source list for every provider; grounding computed, not rubber-stamped | **started** — OpenRouter citations/search results now parsed into `sources_accessed`; other providers + computed grounding (needs B's request bytes) remain |
| D. Evidence join + witnessed path | verify declared step→call links against the witness; parse per-step probability | after A-C, the big one |
| E. Model catalog gate | safe constants now, allow-list behind a flag | parallel with D |
| F. Belief path onto the scoring path | carry path + refs + market price through the collapse to the scorer | after D |
| G. Scoring signals | phase-2 only: early-move reward on the witnessed path, behind an off-switch | last, needs sign-off |

## Gated / needs sign-off (unchanged from roadmap)

- End-to-end belief-path scoring needs the server to round-trip the path and market price on scored results; F lands the validator side only.
- Freeze the model allow-list before enforcing it.
- Any scoring-weight change shifts emissions and ships behind an off-switch defaulting to current behaviour.
