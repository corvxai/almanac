"""Attack: fork-bomb until pids_limit kicks in. The sandbox sets
`--pids-limit=256` so the kernel refuses new clones long before the host
gets unhappy.
"""

from __future__ import annotations

import os
from uuid import UUID

from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.schemas import AgentResult


class ForkBombAttack(BaseAgent):
    agent_id = UUID("00000000-0000-0000-0000-0000000a7406")
    agent_version = "attack/fork_bomb/1"

    def predict(self, ctx: ForecastingContext) -> AgentResult:
        for _ in range(10_000):
            try:
                os.fork()
            except BlockingIOError:
                # pids_limit reached — kernel refused. The validator should
                # see at most ~256 processes before the agent dies (timeout
                # or oom-kill); we still raise so the test sees a non-zero
                # exit.
                raise RuntimeError("fork blocked by pids_limit (good)")
            except OSError as exc:
                raise RuntimeError(f"fork failed: {exc}") from exc
        return AgentResult(prediction=0.5, reasoning="forked successfully")
