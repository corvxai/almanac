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

    if agent_reasoning_steps:
        reasoning_chain = agent_reasoning_steps
    else:
        reasoning_chain = _synthesise_reasoning_chain(
            agent_result, provider_calls,
        )

    total_cost = sum(pc.cost_units for pc in provider_calls)
    total_evidence = sum(len(pc.extracted_evidence) for pc in provider_calls)
    usage_totals = _aggregate_usage(provider_calls)

    integrity = TraceIntegrity(
        trace_hash="",
        trace_schema_version="0.1.0",
        total_provider_cost=total_cost,
        total_evidence_items=total_evidence,
        usage_totals=usage_totals,
    )

    digest = EvidenceDigest(
        execution_context=execution_context,
        event_snapshot=event_snapshot,
        provider_calls=provider_calls,
        reasoning_chain=reasoning_chain,
        prediction_output=prediction_output,
        trace_integrity=integrity,
    )

    return digest.seal()


def _build_prediction_output(
    result: AgentResult,
    provider_calls: list[ProviderCall],
) -> PredictionOutput:
    """Construct the structured PredictionOutput from the agent's simple result.

    The agent provides the probability and reasoning text. The assembler
    enriches it with key drivers extracted from the gateway evidence.
    """
    key_drivers = _extract_key_drivers(provider_calls)
    key_uncertainties = ["Agent-reported reasoning only — no structured decomposition"]

    if result.metadata and "key_drivers" in result.metadata:
        key_drivers = result.metadata["key_drivers"]
    if result.metadata and "key_uncertainties" in result.metadata:
        key_uncertainties = result.metadata["key_uncertainties"]

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
        key_drivers=key_drivers,
        key_uncertainties=key_uncertainties,
        contrarian_flag=contrarian,
    )


def _synthesise_reasoning_chain(
    result: AgentResult,
    provider_calls: list[ProviderCall],
) -> list[ReasoningStep]:
    """Build a reasoning chain from gateway logs + agent's free-text reasoning.

    Produces one evidence_gathering step per provider call (from the gateway
    log), then a final synthesis step containing the agent's reasoning.
    """
    steps: list[ReasoningStep] = []
    step_idx = 0
    inference_call = _find_inference_call(provider_calls)

    for pc in provider_calls:
        evidence_summary = "; ".join(
            e.content for e in pc.extracted_evidence
        ) or f"{pc.provider_id}.{pc.call_type} returned {pc.response_meta.response_size_bytes}B"

        steps.append(ReasoningStep(
            step_index=step_idx,
            step_type=ReasoningStepType.EVIDENCE_GATHERING,
            provider_call_index=pc.call_index,
            provider_id=pc.provider_id,
            input_evidence_refs=[pc.call_index],
            reasoning_text=f"[{pc.provider_id}] {evidence_summary}",
            inference_model_used=pc.model,
        ))
        step_idx += 1

    steps.append(ReasoningStep(
        step_index=step_idx,
        step_type=ReasoningStepType.SYNTHESIS,
        provider_call_index=inference_call.call_index if inference_call else None,
        provider_id=inference_call.provider_id if inference_call else None,
        input_evidence_refs=[pc.call_index for pc in provider_calls],
        reasoning_text=result.reasoning,
        intermediate_probability=result.prediction,
        inference_model_used=inference_call.model if inference_call else None,
    ))
    step_idx += 1

    steps.append(ReasoningStep(
        step_index=step_idx,
        step_type=ReasoningStepType.FINAL_ASSIGNMENT,
        provider_call_index=inference_call.call_index if inference_call else None,
        provider_id=inference_call.provider_id if inference_call else None,
        reasoning_text=f"Final probability: {result.prediction:.4f}",
        intermediate_probability=result.prediction,
        inference_model_used=inference_call.model if inference_call else None,
    ))

    return steps


def _extract_key_drivers(provider_calls: list[ProviderCall]) -> list[str]:
    """Pull the most salient evidence items from provider calls as key drivers."""
    drivers: list[str] = []
    for pc in provider_calls:
        for ev in pc.extracted_evidence:
            if ev.numeric_value is not None:
                drivers.append(ev.content)
            if len(drivers) >= 5:
                return drivers
    return drivers or ["No structured evidence extracted"]


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
