"""Trace Assembler — the validator builds the complete EvidenceDigest.

Takes three independently-produced inputs:
1. ProviderCalls — from the gateway (validator-produced, agent can't influence)
2. AgentResult  — from the agent (just a probability + reasoning string)
3. Agent reasoning steps — optionally provided by the agent via the context

The assembler constructs the structured PredictionOutput and ReasoningChain
from these inputs. If the agent didn't provide structured steps, the assembler
synthesises them from the gateway call log and the agent's free-text reasoning.
"""

from __future__ import annotations

from src.core.events import Event
from src.core.schemas import (
    TRACE_SCHEMA_VERSION,
    AgentResult,
    EvidenceDigest,
    EventSnapshot,
    ExecutionContext,
    PredictionOutput,
    ProviderCall,
    ReasoningStep,
    ReasoningStepType,
    TraceIntegrity,
    UsageTotals,
)


def assemble_trace(
    execution_context: ExecutionContext,
    event: Event,
    provider_calls: list[ProviderCall],
    agent_result: AgentResult,
    agent_reasoning_steps: list[ReasoningStep] | None = None,
) -> EvidenceDigest:
    """Build the complete EvidenceDigest and seal it.

    If the agent provided structured reasoning steps (via context), those
    are used directly. Otherwise, the assembler constructs a trace from
    the gateway logs and the agent's free-text reasoning.
    """

    event_snapshot = EventSnapshot(
        event_title=event.title,
        event_category=event.category,
        event_subcategory=event.subcategory,
        resolution_criteria=event.resolution_criteria,
        resolution_deadline=event.resolution_deadline,
        event_created_at=event.created_at,
    )

    prediction_output = _build_prediction_output(agent_result, provider_calls)

    # beliefPath is the one canonical source that survives the docker sandbox
    # boundary (orchestrator drops ctx.reasoning_chain there), so it drives the
    # chain for both the docker and in-process paths. The agent_reasoning_steps
    # and synth branches are legacy fallbacks for beliefPath-absent traces only.
    if agent_result.beliefPath:
        reasoning_chain = _belief_path_to_reasoning_chain(agent_result, provider_calls)
    elif agent_reasoning_steps:
        reasoning_chain = agent_reasoning_steps
    else:
        reasoning_chain = _synthesise_reasoning_chain(agent_result, provider_calls)

    # Stash the raw beliefPath (labels + usedCall/usedSources, which the
    # reasoning_chain vocab cannot carry) into the reserved phase-2 seam so it
    # reaches Mongo intact via save_trace. Part of the sealed digest.
    future_graph = (
        {"beliefPath": [bs.model_dump() for bs in agent_result.beliefPath]}
        if agent_result.beliefPath
        else None
    )

    total_cost = sum(pc.cost_units for pc in provider_calls)
    usage_totals = _aggregate_usage(provider_calls)

    integrity = TraceIntegrity(
        trace_hash="",
        trace_schema_version=TRACE_SCHEMA_VERSION,
        total_provider_cost=total_cost,
        usage_totals=usage_totals,
    )

    digest = EvidenceDigest(
        execution_context=execution_context,
        event_snapshot=event_snapshot,
        provider_calls=provider_calls,
        reasoning_chain=reasoning_chain,
        prediction_output=prediction_output,
        trace_integrity=integrity,
        future_graph=future_graph,
    )

    return digest.seal()


def _build_prediction_output(
    result: AgentResult,
    provider_calls: list[ProviderCall],
) -> PredictionOutput:
    """Construct the structured PredictionOutput from the agent's simple result.

    The agent provides the probability, confidence and reasoning text; the
    assembler derives the confidence interval and the contrarian flag.
    """
    ci = None
    if result.confidence is not None:
        half_width = (1.0 - result.confidence) / 2
        ci = (
            max(0.0, result.prediction - half_width),
            min(1.0, result.prediction + half_width),
        )

    market_price = _find_market_price(provider_calls)
    contrarian = False
    if market_price is not None:
        contrarian = abs(result.prediction - market_price) > 0.10

    return PredictionOutput(
        final_probability=result.prediction,
        confidence=result.confidence,
        confidence_interval=ci,
        reasoning_summary=result.reasoning,
        contrarian_flag=contrarian,
    )


def _provider_gap_query_steps(provider_calls: list[ProviderCall]) -> list[ReasoningStep]:
    """One gap_query step per provider call (from the gateway log).

    ``input_evidence_refs`` is emitted as [] — NOT a positional fake. The real
    declared-then-verified evidence→step join is a phase-2 hook (see schema).
    """
    steps: list[ReasoningStep] = []
    for idx, pc in enumerate(provider_calls):
        evidence_summary = "; ".join(
            e.content for e in pc.extracted_evidence
        ) or f"{pc.provider_id}.{pc.call_type} returned {pc.response_meta.response_size_bytes}B"

        steps.append(ReasoningStep(
            step_index=idx,
            step_type=ReasoningStepType.GAP_QUERY,
            provider_call_index=pc.call_index,
            provider_id=pc.provider_id,
            input_evidence_refs=[],
            reasoning_text=f"[{pc.provider_id}] {evidence_summary}",
            inference_model_used=pc.model,
        ))
    return steps


def _belief_path_to_reasoning_chain(
    result: AgentResult,
    provider_calls: list[ProviderCall],
) -> list[ReasoningStep]:
    """Map the agent's miner-written beliefPath into the validator reasoning chain.

    Emits the per-provider-call gap_query steps first (provider linkage preserved),
    then one belief step per BeliefStep carrying its ``intermediate_probability`` — so
    the chain holds the full multi-point trajectory, not a single endpoint. The two
    vocabularies differ (BeliefStep.type is prior/update/final; ReasoningStepType is
    prior/belief_update/gap_query), so 'update' and 'final' both map to BELIEF_UPDATE,
    with the terminal step last. usedCall/usedSources stay agent-declared (kept raw in
    future_graph), never lifted into the verified provider_call_index —
    ``input_evidence_refs`` is [] (no declared→verified join yet).
    """
    steps = _provider_gap_query_steps(provider_calls)
    idx = len(steps)
    for bs in result.beliefPath:
        step_type = (
            ReasoningStepType.PRIOR
            if bs.type == "prior"
            else ReasoningStepType.BELIEF_UPDATE
        )
        steps.append(ReasoningStep(
            step_index=idx,
            step_type=step_type,
            provider_call_index=None,
            input_evidence_refs=[],
            reasoning_text=bs.text,
            intermediate_probability=bs.probability,
        ))
        idx += 1
    return steps


def _synthesise_reasoning_chain(
    result: AgentResult,
    provider_calls: list[ProviderCall],
) -> list[ReasoningStep]:
    """Legacy fallback for beliefPath-absent traces: build a chain from gateway
    logs + the agent's free-text reasoning. One gap_query step per provider call,
    then a terminal belief_update step with the agent's reasoning and probability.
    Unreachable for any valid (required-beliefPath) AgentResult.
    """
    steps = _provider_gap_query_steps(provider_calls)
    inference_call = _find_inference_call(provider_calls)
    steps.append(ReasoningStep(
        step_index=len(steps),
        step_type=ReasoningStepType.BELIEF_UPDATE,
        provider_call_index=inference_call.call_index if inference_call else None,
        provider_id=inference_call.provider_id if inference_call else None,
        input_evidence_refs=[],
        reasoning_text=result.reasoning,
        intermediate_probability=result.prediction,
        inference_model_used=inference_call.model if inference_call else None,
    ))
    return steps


def _find_market_price(provider_calls: list[ProviderCall]) -> float | None:
    """Look for a market price in the provider evidence for contrarian detection."""
    for pc in provider_calls:
        for ev in pc.extracted_evidence:
            if ev.evidence_type.value == "price" and ev.numeric_value is not None:
                return ev.numeric_value
    return None


def _find_inference_call(provider_calls: list[ProviderCall]) -> ProviderCall | None:
    """Pick the most likely inference provider call for synthesis/final steps."""
    for pc in reversed(provider_calls):
        if pc.model:
            return pc
    return None


def _aggregate_usage(provider_calls: list[ProviderCall]) -> UsageTotals | None:
    totals = UsageTotals()
    saw_usage = False

    for pc in provider_calls:
        usage = pc.usage_meta
        if usage is None:
            continue
        saw_usage = True
        totals.provider_calls_with_usage += 1
        totals.prompt_tokens += _value_or_zero(usage.prompt_tokens)
        totals.completion_tokens += _value_or_zero(usage.completion_tokens)
        totals.total_tokens += _value_or_zero(usage.total_tokens)
        totals.input_tokens += _value_or_zero(usage.input_tokens)
        totals.output_tokens += _value_or_zero(usage.output_tokens)
        totals.cached_tokens += _value_or_zero(usage.cached_tokens)
        totals.reasoning_tokens += _value_or_zero(usage.reasoning_tokens)

    return totals if saw_usage else None


def _value_or_zero(value: int | None) -> int:
    return value if value is not None else 0
