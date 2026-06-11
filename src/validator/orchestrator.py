"""Validator Orchestrator — the main execution loop.

Receives an event, runs agents through the gateway, assembles traces, stores them.
This is the top-level coordination point for the entire prediction cycle.

Two execution paths share the same trace shape:

- `in_process` mode: agents call providers through the orchestrator's local
  `ProviderGateway`. The call log is `self._gateway.call_log`.
- `docker_*` mode: agents run in a sibling container; their provider calls
  flow through the validator-local signing proxy. The orchestrator drains
  the proxy's per-run log to assemble the trace.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.config import AppConfig
from src.core.events import Event
from src.core.schemas import (
    EvidenceDigest,
    ExecutionContext,
    ProviderCall,
    SandboxEnvironment,
)
from src.gateway.gateway import ProviderGateway
from src.gateway.providers.base import BaseProvider
from src.storage.store import TraceStore
from src.validator.polymarket_baseline import fetch_polymarket_baseline
from src.validator.sandbox import execute_agent
from src.validator.trace_assembler import assemble_trace

logger = logging.getLogger("arcratio.orchestrator")


class Orchestrator:
    def __init__(
        self,
        config: AppConfig,
        store: TraceStore,
        providers: list[BaseProvider],
        local_proxy_state: Optional["object"] = None,
        track: str = "MAIN",
    ):
        """Build the orchestrator.

        Args:
            config: Loaded `AppConfig`.
            store: Trace persistence backend.
            providers: Provider adapters used in `in_process` mode. In
                `docker_*` mode these are unused — the local proxy talks to
                the central gateway, which holds the real adapters.
            local_proxy_state: A `LocalProxyState` from
                `src.gateway.local_proxy`. Required for `docker_*` modes; the
                orchestrator calls `register_run` / `pop_calls` on it.
            track: Track name used to register every run with the proxy.
                Mirrors Numinous's track-based allowlist concept; v1 hardcodes
                to "MAIN" but real subnet configs may dispatch on event class.
        """
        self._config = config
        self._store = store
        self._gateway = ProviderGateway()
        for p in providers:
            self._gateway.register_provider(p)
        self._local_proxy_state = local_proxy_state
        self._track = track

    @property
    def validator_id(self) -> UUID:
        return self._config.validator.validator_id

    def run_agent(
        self,
        event: Event,
        agent: BaseAgent,
        *,
        miner_hotkey: str | None = None,
    ) -> EvidenceDigest:
        """Execute a single agent on a single event and return the sealed trace.

        The agent is a black box — it returns AgentResult (probability + reasoning).
        The validator independently records all gateway calls and assembles
        the complete structured trace.
        """
        execution_id = uuid4()
        sandbox_type = self._config.validator.sandbox_type
        is_docker = sandbox_type.startswith("docker")

        if is_docker and self._local_proxy_state is None:
            raise RuntimeError(
                f"sandbox_type='{sandbox_type}' requires a LocalProxyState; "
                f"none was provided to the Orchestrator."
            )

        # In-process mode: reset the local gateway so a stale call log from
        # a previous run does not bleed into this trace. In docker mode the
        # call log lives on the proxy; we drain it post-run instead.
        if not is_docker:
            self._gateway.reset()

        # Baseline metadata for scoring: pull the latest market signal up-front
        # in a validator-controlled lookup, independent of agent behavior.
        market_baseline = self._collect_market_baseline(event)

        if is_docker:
            self._local_proxy_state.register_run(
                execution_id,
                self._track,
                miner_hotkey=miner_hotkey,
            )

        ts_start = datetime.now(timezone.utc)
        t0 = time.monotonic()

        ctx = ForecastingContext(event=event, gateway=self._gateway)
        try:
            agent_result = execute_agent(
                agent,
                ctx,
                self._config.validator if is_docker else None,
                execution_id if is_docker else None,
            )
        except Exception:
            if is_docker:
                # Drain to free the proxy's per-run state.
                try:
                    self._local_proxy_state.pop_calls(execution_id)  # type: ignore[union-attr]
                except Exception:  # noqa: BLE001
                    pass
            raise

        t1 = time.monotonic()
        ts_end = datetime.now(timezone.utc)
        duration_ms = int((t1 - t0) * 1000)
        duration_s = duration_ms / 1000.0

        logger.info(
            "Agent execution finished: execution_id=%s agent_id=%s agent_version=%s "
            "sandbox=%s duration_ms=%s",
            execution_id,
            agent.agent_id,
            agent.agent_version,
            sandbox_type,
            duration_ms,
        )
        print(
            f"[arcratio] agent execution finished: sandbox={sandbox_type} "
            f"duration={duration_s:.2f}s ({duration_ms} ms) "
            f"agent={agent.agent_id!r} execution_id={execution_id}",
            flush=True,
        )

        sandbox_env = SandboxEnvironment(sandbox_type)

        exec_ctx = ExecutionContext(
            execution_id=execution_id,
            agent_id=agent.agent_id,
            agent_version=agent.agent_version,
            event_id=event.event_id,
            validator_id=self.validator_id,
            timestamp_start=ts_start,
            timestamp_end=ts_end,
            execution_duration_ms=duration_ms,
            sandbox_environment=sandbox_env,
        )

        if is_docker:
            provider_calls: list[ProviderCall] = (
                self._local_proxy_state.pop_calls(execution_id)  # type: ignore[union-attr]
            )
            agent_steps = None  # sandbox runners don't surface structured steps
        else:
            provider_calls = self._gateway.call_log
            agent_steps = ctx.reasoning_chain or None

        if market_baseline is not None:
            metadata = dict(agent_result.metadata or {})
            metadata["market_baseline"] = market_baseline
            agent_result = agent_result.model_copy(update={"metadata": metadata})

        digest = assemble_trace(
            execution_context=exec_ctx,
            event=event,
            provider_calls=provider_calls,
            agent_result=agent_result,
            agent_reasoning_steps=agent_steps,
        )

        self._store.save_trace(digest)
        return digest

    def run_all_agents(
        self,
        event: Event,
        agents: list[BaseAgent],
    ) -> list[EvidenceDigest]:
        """Run multiple agents on the same event, returning all traces."""
        return [self.run_agent(event, agent) for agent in agents]

    def _collect_market_baseline(self, event: Event) -> dict[str, Any] | None:
        """Fetch latest YES/NO baseline prices from direct Polymarket lookup."""
        try:
            return fetch_polymarket_baseline(event)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "market baseline fetch failed for event=%s source_id=%s: %s",
                event.event_id,
                event.source_id,
                exc,
            )
            return None
