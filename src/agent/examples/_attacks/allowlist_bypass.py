"""Attack: call a provider that the run's track does not allow.

The run is registered with track="SIGNAL" which only permits
`polymarket` + `web_search`; this agent calls `anthropic.messages` and the
local proxy must respond with HTTP 403.
"""

from __future__ import annotations

from uuid import UUID

from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.schemas import AgentResult, BeliefStep


class AllowlistBypassAttack(BaseAgent):
    agent_id = UUID("00000000-0000-0000-0000-0000000a7408")
    agent_version = "attack/allowlist_bypass/1"

    def predict(self, ctx: ForecastingContext) -> AgentResult:
        ctx.call_provider("anthropic", "messages", {"messages": []})
        reason = "bypassed allowlist"
        belief = [BeliefStep(step=0, type="final", probability=0.5, text=reason)]
        return AgentResult(prediction=0.5, reasoning=reason, beliefPath=belief)
