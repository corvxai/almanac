"""Agent execution sandbox.

Two modes:

- `in_process`: the agent's `predict` runs directly in this process. Used
  for unit tests and quick local development. No isolation.
- `docker_runc` (default) / `docker_gvisor`: a sibling Docker container is
  spawned per execution with `--network=none`, a read-only rootfs, dropped
  capabilities, resource limits, and a single bind-mounted UDS pointing at
  the validator-local signing proxy. The agent's only outbound channel is
  through that socket.

Phase 1 (subprocess isolation) was skipped — Docker provides strictly more
guarantees and is the production path. `SUBPROCESS` remains in the enum for
backwards compatibility with any historical traces.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.config import ValidatorConfig
from src.core.schemas import AgentResult


def execute_agent(
    agent: BaseAgent,
    ctx: ForecastingContext,
    cfg: Optional[ValidatorConfig] = None,
    run_id: Optional[UUID] = None,
) -> AgentResult:
    """Run an agent's prediction logic.

    Args:
        agent: The agent instance to execute.
        ctx: A `ForecastingContext` over a gateway-shaped facade. For
            `in_process` mode the gateway is the validator's own
            `ProviderGateway`. For `docker_*` modes `ctx` is *not* used by the
            sandbox itself — the runner inside the container builds its own
            `ForecastingContext` from the JSON payload — but the orchestrator
            passes a pre-built `ctx` so it can pluck the `Event` out.
        cfg: `ValidatorConfig`. When omitted the agent runs in-process for
            backwards compatibility with older callers.
        run_id: Per-execution UUID. Required for `docker_*` modes; the
            validator-local proxy uses it to attribute calls to a run.

    Returns:
        The agent's `AgentResult`. Exceptions propagate to the caller.
    """
    if cfg is None or cfg.sandbox_type == "in_process":
        return agent.predict(ctx)

    if run_id is None:
        raise ValueError("run_id is required for docker sandbox modes")

    # Imported lazily so the validator's main config path doesn't pay for the
    # docker SDK on dev machines that only run in-process tests.
    from src.validator.forecasting.sandbox_docker import run_agent_in_container

    return run_agent_in_container(agent, ctx, cfg, run_id)
