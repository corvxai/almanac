"""Base agent — the minimal interface every forecasting agent implements.

Agents are black boxes. They receive event info and a context for calling
data providers, and they return a probability + reasoning string.
How they get there is their business.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from src.agent.context import ForecastingContext
    from src.core.schemas import AgentResult


class BaseAgent:
    """A forecasting agent that produces predictions on binary events.

    Subclass this and implement predict(). The only hard requirements are
    agent_id, agent_version, and returning an AgentResult from predict().
    """

    agent_id: UUID = uuid4()
    agent_version: str = "0.1.0"

    def predict(self, ctx: "ForecastingContext") -> "AgentResult":
        """Run the agent's forecasting logic and return a prediction.

        Args:
            ctx: Provides event info and call_provider() for data access.

        Returns:
            AgentResult with at minimum a prediction (float) and reasoning (str).
        """
        raise NotImplementedError
