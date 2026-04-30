"""ForecastingContext — the SDK object passed to every agent at execution time.

Provides:
- Event information (title, description, metadata)
- Data provider access via call_provider() (routed through a gateway-shaped facade)
- Optional structured reasoning step recording (agents don't have to use this)

The `gateway` argument is duck-typed: any object with
``call_provider(provider_id, call_type, params) -> dict`` works. Validators
running agents in-process pass a heavyweight `ProviderGateway`; agents running
inside the Docker sandbox receive a thin `SandboxGateway` whose `call_provider`
forwards to the validator-local signing proxy over a UNIX socket. The
`ForecastingContext` does not care which one it has.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol

from src.core.events import Event
from src.core.schemas import ReasoningStep, ReasoningStepType


class GatewayLike(Protocol):
    """Minimal duck-typed contract for `ForecastingContext`."""

    def call_provider(
        self,
        provider_id: str,
        call_type: str,
        params: dict[str, Any],
    ) -> dict[str, Any]: ...


class ForecastingContext:
    """The agent's view of the world during execution.

    The only method agents MUST use is call_provider() to access external data.
    record_reasoning_step() is available for agents that want to provide
    structured traces, but the validator will build a trace regardless.
    """

    def __init__(self, event: Event, gateway: GatewayLike):
        self._event = event
        self._gateway = gateway
        self._reasoning_chain: list[ReasoningStep] = []
        self._step_counter: int = 0

    @property
    def event(self) -> Event:
        return self._event

    @property
    def event_title(self) -> str:
        return self._event.title

    @property
    def event_description(self) -> str:
        return self._event.description

    @property
    def reasoning_chain(self) -> list[ReasoningStep]:
        """Any structured steps the agent chose to record. May be empty."""
        return list(self._reasoning_chain)

    def call_provider(
        self,
        provider_id: str,
        call_type: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fetch data from an external provider via the gateway.

        The agent gets the raw response data. The gateway independently
        records, hashes, and extracts evidence — the agent cannot influence
        that process.
        """
        return self._gateway.call_provider(provider_id, call_type, params or {})

    def record_reasoning_step(
        self,
        step_type: ReasoningStepType,
        reasoning_text: str,
        provider_call_index: Optional[int] = None,
        provider_id: Optional[str] = None,
        input_evidence_refs: list[int] | None = None,
        agent_interpretation: Optional[str] = None,
        intermediate_probability: Optional[float] = None,
        confidence_delta: Optional[float] = None,
        conflict_signals: Optional[list[str]] = None,
        inference_model_used: Optional[str] = None,
    ) -> None:
        """Optionally record a structured reasoning step.

        Agents that want finer-grained trace visibility can call this.
        If not called, the validator will still build a reasoning trace
        from the agent's free-text reasoning and gateway call logs.
        """
        step = ReasoningStep(
            step_index=self._step_counter,
            step_type=step_type,
            provider_call_index=provider_call_index,
            provider_id=provider_id,
            input_evidence_refs=input_evidence_refs or [],
            reasoning_text=reasoning_text,
            agent_interpretation=agent_interpretation,
            intermediate_probability=intermediate_probability,
            confidence_delta=confidence_delta,
            conflict_signals=conflict_signals,
            inference_model_used=inference_model_used,
        )
        self._reasoning_chain.append(step)
        self._step_counter += 1
