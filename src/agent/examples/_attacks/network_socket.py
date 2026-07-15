"""Attack: open a raw TCP socket to an external IP. Should fail under
`--network=none` because the sandbox has no IP stack at all.
"""

from __future__ import annotations

import socket
from uuid import UUID

from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.schemas import AgentResult, BeliefStep


class NetworkSocketAttack(BaseAgent):
    agent_id = UUID("00000000-0000-0000-0000-0000000a7401")
    agent_version = "attack/network_socket/1"

    def predict(self, ctx: ForecastingContext) -> AgentResult:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect(("1.1.1.1", 443))
        s.close()
        reason = "connected to 1.1.1.1:443"
        belief = [BeliefStep(step=0, type="final", probability=0.5, text=reason)]
        return AgentResult(prediction=0.5, reasoning=reason, beliefPath=belief)
