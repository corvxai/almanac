"""Attack: HTTPS request via urllib to a public URL."""

from __future__ import annotations

import urllib.request
from uuid import UUID

from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.schemas import AgentResult


class NetworkUrllibAttack(BaseAgent):
    agent_id = UUID("00000000-0000-0000-0000-0000000a7402")
    agent_version = "attack/network_urllib/1"

    def predict(self, ctx: ForecastingContext) -> AgentResult:
        with urllib.request.urlopen("https://example.com", timeout=2.0) as resp:
            body = resp.read(64)
        return AgentResult(
            prediction=0.5,
            reasoning=f"reached example.com, got {len(body)} bytes",
        )
