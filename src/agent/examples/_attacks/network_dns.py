"""Attack: DNS lookup. Should fail because the sandbox has no resolver."""

from __future__ import annotations

import socket
from uuid import UUID

from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.schemas import AgentResult, BeliefStep


class NetworkDnsAttack(BaseAgent):
    agent_id = UUID("00000000-0000-0000-0000-0000000a7403")
    agent_version = "attack/network_dns/1"

    def predict(self, ctx: ForecastingContext) -> AgentResult:
        addr = socket.gethostbyname("example.com")
        reason = f"resolved example.com -> {addr}"
        belief = [BeliefStep(step=0, type="final", probability=0.5, text=reason)]
        return AgentResult(prediction=0.5, reasoning=reason, beliefPath=belief)
