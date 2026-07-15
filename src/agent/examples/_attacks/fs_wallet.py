"""Attack: read the validator's Bittensor wallet from inside the sandbox.

The validator container mounts `~/.bittensor:/root/.bittensor:ro`, but the
sandbox container's mount namespace explicitly does NOT include that path.
Either the directory does not exist or it is empty — the attack must not be
able to read any keypair material.
"""

from __future__ import annotations

import os
from uuid import UUID

from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.schemas import AgentResult, BeliefStep


class FsWalletAttack(BaseAgent):
    agent_id = UUID("00000000-0000-0000-0000-0000000a7405")
    agent_version = "attack/fs_wallet/1"

    def predict(self, ctx: ForecastingContext) -> AgentResult:
        targets = ["/root/.bittensor", "/root/.bittensor/wallets", "/home/runner/.bittensor"]
        found: list[str] = []
        for path in targets:
            if os.path.exists(path):
                try:
                    found.extend(
                        os.path.join(path, name) for name in os.listdir(path)
                    )
                except OSError:
                    pass
        reason = "found: " + ",".join(found) if found else "wallet not visible"
        belief = [BeliefStep(step=0, type="final", probability=0.5, text=reason)]
        return AgentResult(prediction=0.5, reasoning=reason, beliefPath=belief)
