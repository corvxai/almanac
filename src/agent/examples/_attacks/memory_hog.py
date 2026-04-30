"""Attack: allocate 2 GB. Memory limit (`--memory=1024m`) makes the kernel
oom-kill the agent process before it succeeds.
"""

from __future__ import annotations

from uuid import UUID

from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.schemas import AgentResult


class MemoryHogAttack(BaseAgent):
    agent_id = UUID("00000000-0000-0000-0000-0000000a7407")
    agent_version = "attack/memory_hog/1"

    def predict(self, ctx: ForecastingContext) -> AgentResult:
        chunks: list[bytes] = []
        for _ in range(2048):
            chunks.append(b"x" * (1024 * 1024))  # 1 MiB
        total = sum(len(c) for c in chunks)
        return AgentResult(prediction=0.5, reasoning=f"allocated {total} bytes")
