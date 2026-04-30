"""Attack: bypass the local proxy by hitting the central gateway URL directly.

This requires a working DNS + TCP stack — both of which `--network=none`
removes. Even with the URL hardcoded the connection cannot leave the netns.
"""

from __future__ import annotations

from uuid import UUID

import httpx

from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.schemas import AgentResult


class DirectGatewayAttack(BaseAgent):
    agent_id = UUID("00000000-0000-0000-0000-0000000a7409")
    agent_version = "attack/direct_gateway/1"

    def predict(self, ctx: ForecastingContext) -> AgentResult:
        with httpx.Client(timeout=2.0) as client:
            resp = client.post(
                "https://gateway.arcratio.ai/v1/call",
                json={"provider_id": "polymarket", "call_type": "get_market", "params": {}},
            )
        return AgentResult(prediction=0.5, reasoning=f"gateway returned {resp.status_code}")
