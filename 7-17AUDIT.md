# Audit — PR 14 (`v1-trace-scoring`) — 2026-07-17

**Scope:** full diff of `v1-trace-scoring` against `origin/main` (52 files, ~2,910 insertions), reviewed for security and code-quality issues that could impact users. Ingestion cross-checked against the Sub41 gateway (`gateway/apps/api`, read-only). No code changes were made.

**Method:** manual line-by-line review of all runtime code in the diff (trust boundary: `src/gateway/local_proxy.py` + docker sandbox), followed by a multi-agent workflow review.

**Cross-branch check:** the only other open branches (`scoring-tweaks`, `leos-refactor`) touch scoring/weight-conversion code only. **None of the findings below are addressed in main or any other branch.** Note: `scoring-tweaks` consumes `EvidenceDigest` and will need reconciliation with the v1 schema when merged.

---

## Verdict summary

The PR is a net security improvement in its intent — the market baseline moves out of agent-writable metadata into the hash-sealed execution context, `AgentResult` becomes `extra="forbid"` + `allow_inf_nan=False` at the docker boundary, the sandbox→proxy trust boundary is intact, and the validated-JSON forecast contract removes the silent-0.5 regex parser. But verification surfaced one finding that blocks the headline feature in production (F1), two storage-compatibility breaks (F3, F15), and a scoring-integrity hole (F5).

**Pre-lock priority:** F1 → F3/F15 → F4 → F5. The rest can be batched as hardening.

---

## Confirmed findings (ranked most severe first)

### F1 — Typed data-provider calls are rejected by the production Sub41 gateway

`src/agent/runner_entrypoint.py:196`, `src/gateway/provider_capabilities.py:19` — **correctness / feature-breaking in prod**

Typed calls (polymarket `get_market`, web_search `search`) are now sent as `{provider, callType, params}` with no `model`/`messages`. The Sub41 gateway's `GatewayCompletionDto` (`apps/api/src/gateway/dto/gateway-request.dto.ts`) runs under `forbidNonWhitelisted` with required `model` + `messages` and no `callType`/`params` properties, so the NestJS ValidationPipe returns 400 (`property callType should not exist`, `model must be a string`, `messages must be an array`). The params-forwarding fix (commit b1ac0ab) therefore only works against the local simulator (`src/gateway` server). In a docker/production run every polymarket/web_search call fails and `v1_agent6_market_reader` always degrades to its 0.5 fallback.

**Resolution:** either extend the Sub41 DTO to accept typed calls, or explicitly scope typed data providers as simulator-only.

### F2 — Citation extractor crashes on malformed-but-2xx upstream JSON inside the trust boundary

`src/gateway/extractor.py:531` — **robustness at trust boundary**

`_openrouter_url_citations` does not isinstance-guard upstream JSON (`choices[0]`, `message`, annotation items, `url_citation`, `search_results` items), unlike the guarded pre-existing `_extract_llm_completion_text`. It runs inside the validator local proxy via `build_provider_call_record`. A response with `"message": null` (note: `dict.get("message", {})` returns `None` when the key is present-but-null), a non-dict `choices[0]`, or a string in `annotations`/`search_results` raises `AttributeError` inside `run.lock`: the agent's call returns HTTP 500 despite the upstream call succeeding and being billed, the `ProviderCall` record is never appended (evidence lost, `call_index` gap from the pre-incremented counter), and the trace is missing a witnessed call.

**Resolution:** isinstance-guard every level, matching the style of `_extract_llm_completion_text`.

### F3 — ReasoningStepType enum rewrite breaks all stored 0.1.0 traces

`src/core/schemas.py:71` + `src/storage/json_store.py:98` — **backward-compat / silent data loss**

The six v0.1 values (`evidence_gathering`, `synthesis`, `conflict_resolution`, `prior_update`, `calibration_check`, `final_assignment`) were deleted with no migration. `json_store.get_trace` calls `EvidenceDigest.model_validate_json` with no exception handling — direct lookups of pre-upgrade traces raise `ValidationError`. Worse, `list_traces_by_event`/`list_traces_by_agent` go through `_iter_all_traces`, whose bare `except Exception: continue` **silently drops every pre-upgrade trace with no log line** — historical traces vanish from listings.

**Resolution:** purge old local traces at deploy (if disposable POC data) or add legacy enum values / a migration shim; at minimum log skipped traces in `_iter_all_traces`.

### F4 — Unbounded miner-authored text vs. the gateway's 1 MB body limit

`src/core/schemas.py:252,275,277` — **DoS / lost predictions after paid execution**

`BeliefStep.text`, `AgentResult.reasoning`, and `beliefPath` have `min_length` only — no `max_length`/`max_items`. The docker sandbox permits 8 MiB of agent stdout (`sandbox_docker.py:66`), and the submit payload embeds the same text up to **three times** (`steps[].reasoningText`, `futureGraph.beliefPath[].text`, `predictionOutput.reasoningSummary`). The Sub41 gateway caps JSON bodies at 1 MB (`apps/api/src/main.ts:45`). A miner returning a few hundred KB of belief text produces a valid, sealed trace whose submit then 413s — the prediction is never recorded even though the validator already paid for the agent's provider calls. Sub-limit bloat also lands verbatim in both Mongo stores (`reasoningTrace.trace` is stored as unvalidated `unknown` — `post-validator-prediction.dto.ts:144`).

**Resolution:** add `max_length` on `text`/`reasoning` (~8k chars) and `max_length` on `beliefPath` (~64 steps), sized so worst-case payload stays under 1 MB.

### F5 — Self-reported citations are granted grounding credit

`src/gateway/extractor.py:517` — **scoring integrity / gameable**

`counts_toward_grounding=True` is granted to citation evidence purely because the upstream response body self-reports `url_citation`/`search_results`/`citations` fields; no verification the URL was fetched, and `witness_call_index` is left `None`. A miner can route `chat_completion` calls to a roleplay-capable OpenRouter marketplace model whose output includes fabricated `url_citation` annotations; the validator records them as admissible grounded evidence in the sealed trace, letting an ungrounded prediction pass the fact-gate / invalid-rate gate the field exists to enforce.

**Resolution:** restrict grounding credit to an allowlist of genuinely web-native models/endpoints, or defer setting the flag until a real verification step exists.

### F6 — traceHash is unrecomputable from the submitted payload and never verified server-side

`src/validator/forecasting/assignment_pipeline.py:297` (also :80, :99) — **integrity envelope is decorative**

The hash is sealed over the intact snake_case digest (full `provider_calls`, `reasoning_chain`, original texts). `_compact_evidence_digest` then strips those arrays from the submitted `evidenceDigest` **and mutates content** (final belief `text` → `textRef`, one step's `reasoningText` → `reasoningTextRef`), and the retained copies at `trace.providerCalls`/`trace.steps` are camelized. No recipient can recompute the pre-image, and nothing in `apps/api` references `traceHash` at all — a tampered payload submitted alongside the original hash is accepted and stored as if verified.

**Resolution:** document the exact reconstruction procedure, or additionally hash the compacted camelCase form, and add server-side verification when scoring starts consuming traces.

### F7 — contrarianFlag ignores the sealed baseline this PR introduced

`src/validator/forecasting/trace_assembler.py:119` — **agent-controllable signal**

`contrarian_flag` is still derived from agent-controlled provider evidence via `_find_market_price` (first `price` evidence item), ignoring the validator-sealed `execution_context.market_price_at_prediction` added by this PR for exactly this purpose. All six v1 example agents (and any agent that never calls polymarket) always emit `contrarianFlag=False` regardless of deviation from the sealed baseline; conversely a miner can suppress or trigger the flag by choosing which provider calls to make.

**Resolution:** derive the flag from `execution_context.market_price_at_prediction`, falling back to evidence only when the baseline is absent.

### F8 — Missing confidence is submitted as declared 0.0 with isValid=true

`src/validator/forecasting/assignment_pipeline.py:222` — **data quality for downstream consumers**

With `confidence_missing` no longer an invalid reason, `confidence=None` is filled with `INVALID_CONFIDENCE_SENTINEL = 0.0` and submitted as `prediction.confidence: 0.0` (the Sub41 DTO requires a number 0..1; null cannot be sent). Downstream Sub41 scoring/portal consumers see a _declared_ extreme zero-confidence for every confidence-omitting miner; only the free-text `rawObserved.confidence` repr (`'None'`) distinguishes the cases.

**Resolution:** make confidence nullable in the Sub41 DTO, or carry an explicit `confidenceOmitted` boolean in `executionMetadata`.

### F9 — JSON blob extractor breaks on braces inside string values

`src/agent/examples/_v1_common.py:144` — **valid forecasts discarded**

`_extract_json_blob`'s balanced-brace scan counts braces inside JSON string values (no string-context or escape tracking). A valid response whose `reasoning` contains a lone `}` truncates the blob; the retry hits the same behavior; the agent fails closed to a neutral 0.5 — discarding a valid, paid-for forecast.

**Resolution:** track in-string/escape state during the scan, or try `json.JSONDecoder().raw_decode` from the first `{`.

### F10 — Scientific-notation probability `"1e-5"` coerces to 1.0 and validates

`src/agent/examples/_v1_common.py:104` — **silently wrong prediction**

`_coerce_unit`'s regex `-?\d*\.?\d+` cannot parse exponents: `"1e-5"` matches `1` → `1.0`, which passes the [0,1] gate (venv-verified). A model emitting a tiny probability as `"1e-5"` yields a validated forecast of certain-YES — the exact silent wrongness the JSON contract was introduced to prevent. Other mantissas (`"5e-2"` → 5.0) are luckily rejected by range.

**Resolution:** accept scientific notation in the regex (e.g. `-?\d*\.?\d+(?:[eE][-+]?\d+)?`) or reject strings containing `e`/`E` exponents outright.

### F11 — round(,4) float equality rejects boundary-equal belief/prediction pairs

`src/core/schemas.py:285` — **spurious run rejection at the docker boundary**

`round(x, 4) != round(y, 4)` rejects semantically equal values at `.xxxx5` rounding boundaries (venv-verified: `round(0.12345, 4) == 0.1235` vs `round(0.123449999999, 4) == 0.1234`, values 1e-12 apart) and asymmetrically accepts values up to ~1e-4 apart. A miner whose final belief and prediction differ by 1 ulp near a midpoint has the whole run rejected.

**Resolution:** `abs(a - b) <= 5e-5` or `math.isclose`.

### F12 — 0.5 sentinel lets an earlier prose match override the model's real answer

`src/agent/examples/openrouter_agent.py:83` (same code in `claude_agent.py:83`, `claude_agent2.py:83`, `openrouter_agent2.py:83`) — **silently wrong prediction (legacy agents)**

`if prediction == 0.5` is used as a parse-failed sentinel; the fallback regex takes the FIRST `PREDICTION:` match while the line parser took the LAST. A model that discusses `PREDICTION: 0.85` in prose and concludes `PREDICTION: 0.5` submits 0.85.

**Resolution:** use a `None` sentinel for parse failure; make the fallback regex take the last match.

### F13 — Removed public kwargs break out-of-tree v0 miner agents mid-run

`src/agent/context.py:80` — **API break without deprecation**

`agent_interpretation` and `conflict_signals` were deleted from `record_reasoning_step` with no `**kwargs` shim. An out-of-tree agent written against v0 raises `TypeError` mid-predict (in-process mode) after provider calls were already made and billed — spend lost, no trace/prediction produced.

**Resolution:** accept-and-ignore the removed kwargs for one deprecation cycle, or document the hard break in the miner guide.

### F14 — sources_accessed contradicts the grounding verdict on the same citation

`src/gateway/extractor.py:553` — **phase-2 seam inconsistency**

`extract_sources` returns dicts without `counts_toward_grounding`, so every `Source` in `sources_accessed` defaults to `False` — while the `ExtractedEvidence` built from the _same_ citation sets `counts_toward_grounding=True`. A phase-2 fact-gate reading `sources_accessed` (documented as the fact-gate substrate) will judge every web source inadmissible, zeroing grounding credit for honestly-grounded miners.

**Resolution:** set the flag consistently in `extract_sources` (subject to the F5 policy decision), or document which field is authoritative.

### F15 — reasoningTrace.schemaVersion still "1.0" despite a materially changed Mongo shape

`src/validator/forecasting/assignment_pipeline.py:344` — **versioning / mixed-shape store**

The wrapper `reasoningTrace.schemaVersion` stays `"1.0"` while the stored shape changes materially (evidenceDigest compacted, steps gain `stage`/`origin`/`reasoningTextRef`, `futureGraph.beliefPath` with `textRef`; internal `trace_integrity` version bumped to 1.1.0). After validators upgrade, Mongo holds two incompatible shapes under the same version marker; a scorer or migration dispatching on `schemaVersion` will process new documents with the old reader and drop or misread reasoning text.

**Resolution:** bump the wrapper version (e.g. `"1.1"`) in the same commit as the shape change.

---

## Refuted during verification

- `src/validator/forecasting/orchestrator.py:165` — claim that an out-of-range/NaN live Polymarket `yes_price` fed into `ExecutionContext.market_price_at_prediction` (ge=0, le=1) would raise post-execution and abort the run. Refuted by the verifier pass.

---

## Positive security changes worth preserving

- Market baseline moved from agent-writable `prediction_output.metadata` into the hash-sealed `ExecutionContext.market_price_at_prediction` — closes a baseline-spoofing vector (but see F7: the contrarian flag doesn't use it yet).
- `AgentResult` now `extra="forbid"` + `allow_inf_nan=False`, required non-empty reasoning, required validated `beliefPath` — hardened docker-boundary contract, well tested.
- Trust boundary intact in `local_proxy.py`: sandbox cannot choose the signed hotkey; track allowlist enforced on typed calls; `_NON_COMPLETIONS_PROVIDERS = DATA_PROVIDERS` is set-identical centralization.
- Validated-JSON forecast contract replaces a regex that silently defaulted to 0.5 and clamped `"62%"` to 1.0; failures are rejected or become explicit in-band neutral forecasts (modulo F9/F10 parser gaps).
- Deterministic per-slug mock polymarket pricing keeps multi-miner comparisons legible.
- 8 MiB capped, streamed sandbox-stdout read contains hostile stdout flooding.

---
