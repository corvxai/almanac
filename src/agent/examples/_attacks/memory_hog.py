"""Attack: allocate 2 GB. Memory limit (`--memory=1024m`) makes the kernel
oom-kill the agent process before it succeeds.
"""

from __future__ import annotations

from uuid import UUID

from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.schemas import AgentResult, BeliefStep


class MemoryHogAttack(BaseAgent):
    agent_id = UUID("00000000-0000-0000-0000-0000000a7407")
    agent_version = "attack/memory_hog/1"

    def predict(self, ctx: ForecastingContext) -> AgentResult:
        chunks: list[bytes] = []
        for _ in range(2048):
            chunks.append(b"x" * (1024 * 1024))  # 1 MiB
        total = sum(len(c) for c in chunks)
        reason = f"allocated {total} bytes"
        belief = [BeliefStep(step=0, type="final", probability=0.5, text=reason)]
        return AgentResult(prediction=0.5, reasoning=reason, beliefPath=belief)
