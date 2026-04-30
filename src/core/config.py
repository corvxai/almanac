"""Configuration management for the forecasting prototype."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from dotenv import load_dotenv
from pydantic import BaseModel, Field


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _PROJECT_ROOT / ".env"


# Sandbox type literal — kept as a str so legacy values like "in_process" still
# round-trip through ValidatorConfig and the SandboxEnvironment enum.
SandboxType = Literal[
    "in_process",
    "subprocess",
    "docker",
    "docker_runc",
    "docker_gvisor",
]


class ValidatorConfig(BaseModel):
    validator_id: UUID = Field(default_factory=uuid4)
    available_providers: list[str] = Field(default_factory=lambda: ["polymarket", "web_search"])

    sandbox_type: SandboxType = "docker_runc"
    sandbox_image: str = "arcratio/agent-runner:latest"
    sandbox_timeout_seconds: int = 240
    sandbox_max_concurrent: int = 8
    sandbox_memory_mb: int = 1024
    sandbox_cpus: float = 1.0
    sandbox_pids_limit: int = 256
    # Default under the repo so host runs (e.g. ``run_forecast.py``) need no
    # root; Docker Compose still sets ``SANDBOX_SOCKET_DIR=/var/run/arcratio``
    # inside the validator container where that path is volume-mounted.
    sandbox_socket_dir: Path = Field(
        default_factory=lambda: _PROJECT_ROOT / "var/run/arcratio",
    )
    # Host path for the proxy UDS bind into *sibling* agent containers. When the
    # validator runs inside Docker, the Docker API resolves bind sources on the
    # host — not inside this container — so in-container paths are wrong unless
    # this is set or auto-resolved (see sandbox_docker.mountinfo helper).
    sandbox_socket_host_bind: str | None = None


class BittensorConfig(BaseModel):
    """Validator-side Bittensor wallet configuration.

    Only the validator container reads this. The runner image must NOT depend
    on or read this config. Wallet files are mounted read-only into the
    validator container only — the agent sandbox never sees them.
    """

    wallet_path: Path = Path("/root/.bittensor/wallets")
    wallet_name: str = "default"
    wallet_hotkey: str = "default"
    netuid: int = 0
    subtensor_network: str = "finney"
    signing_required: bool = True


class StorageConfig(BaseModel):
    backend: str = "json"
    data_dir: Path = _PROJECT_ROOT / "data"


class GatewayConfig(BaseModel):
    default_timeout_ms: int = 30_000
    max_retries: int = 2


class AppConfig(BaseModel):
    validator: ValidatorConfig = Field(default_factory=ValidatorConfig)
    bittensor: BittensorConfig = Field(default_factory=BittensorConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    log_level: str = "INFO"

    @classmethod
    def load_default(cls) -> "AppConfig":
        """Build a config with .env-aware Bittensor + sandbox overrides applied.

        Mirrors how `src.gateway.constants` loads `.env` with `override=False`,
        so a shell-set variable always wins over the `.env` file.
        """
        if _ENV_FILE.is_file():
            load_dotenv(_ENV_FILE, override=False)

        cfg = cls()

        # Bittensor wallet/netuid overrides
        if (v := os.environ.get("BITTENSOR_WALLET_PATH", "").strip()):
            cfg.bittensor.wallet_path = Path(v)
        if (v := os.environ.get("BITTENSOR_WALLET_NAME", "").strip()):
            cfg.bittensor.wallet_name = v
        if (v := os.environ.get("BITTENSOR_WALLET_HOTKEY", "").strip()):
            cfg.bittensor.wallet_hotkey = v
        if (v := os.environ.get("BITTENSOR_NETUID", "").strip()):
            try:
                cfg.bittensor.netuid = int(v)
            except ValueError:
                pass
        if (v := os.environ.get("BITTENSOR_NETWORK", "").strip()):
            cfg.bittensor.subtensor_network = v
        if (v := os.environ.get("BITTENSOR_SIGNING_REQUIRED", "").strip()):
            cfg.bittensor.signing_required = v.lower() in {"1", "true", "yes", "on"}

        # Sandbox overrides
        if (v := os.environ.get("SANDBOX_TYPE", "").strip()):
            cfg.validator.sandbox_type = v  # type: ignore[assignment]
        if (v := os.environ.get("SANDBOX_IMAGE", "").strip()):
            cfg.validator.sandbox_image = v
        if (v := os.environ.get("SANDBOX_SOCKET_DIR", "").strip()):
            cfg.validator.sandbox_socket_dir = Path(v)
        if (v := os.environ.get("SANDBOX_BIND_SRC", "").strip()):
            cfg.validator.sandbox_socket_host_bind = v

        return cfg
