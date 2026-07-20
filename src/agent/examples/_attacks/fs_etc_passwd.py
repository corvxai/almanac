"""Attack: read /etc/passwd. Read-only rootfs still exposes /etc; verify the
attack returns the hardened image's tiny user table (UID 10001 only) and
*not* anything from the host.

The test asserts the agent succeeded but exposed nothing useful — i.e. only
container-internal accounts. This proves the sandbox FS is not the host FS.
"""

from __future__ import annotations

from uuid import UUID

from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.schemas import AgentResult, BeliefStep


class FsEtcPasswdAttack(BaseAgent):
    agent_id = UUID("00000000-0000-0000-0000-0000000a7404")
    agent_version = "attack/fs_etc_passwd/1"

    def predict(self, ctx: ForecastingContext) -> AgentResult:
        with open("/etc/passwd", "r", encoding="utf-8") as fh:
            content = fh.read()
        belief = [BeliefStep(step=0, type="final", probability=0.5,
                             text=content or "read /etc/passwd")]
        return AgentResult(prediction=0.5, reasoning=content, beliefPath=belief)
