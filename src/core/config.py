"""Configuration models for the forecasting prototype."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from src.core import constants
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
    available_providers: list[str] = Field(
        default_factory=lambda: list(constants.VALIDATOR.available_providers)
    )
    orchestrator_api_url: str = constants.ORCHESTRATOR_API_URL

    sandbox_type: SandboxType = constants.VALIDATOR.sandbox_type  # type: ignore[assignment]
    sandbox_image: str = constants.VALIDATOR.sandbox_image
    sandbox_timeout_seconds: int = constants.VALIDATOR.sandbox_timeout_seconds
    sandbox_max_concurrent: int = constants.VALIDATOR.sandbox_max_concurrent
    sandbox_memory_mb: int = constants.VALIDATOR.sandbox_memory_mb
    sandbox_cpus: float = constants.VALIDATOR.sandbox_cpus
    sandbox_pids_limit: int = constants.VALIDATOR.sandbox_pids_limit
    # Default under the repo so host runs (e.g. ``run_forecast.py``) need no
    # root privileges.
    sandbox_socket_dir: Path = Field(
        default_factory=lambda: constants.VALIDATOR.sandbox_socket_dir,
    )
    # Host path for the proxy UDS bind into *sibling* agent containers. When the
    # validator runs inside Docker, the Docker API resolves bind sources on the
    # host — not inside this container — so in-container paths are wrong unless
    # this is set or auto-resolved (see sandbox_docker.mountinfo helper).
    sandbox_socket_host_bind: str | None = constants.VALIDATOR.sandbox_socket_host_bind


class BittensorConfig(BaseModel):
    """Validator-side Bittensor wallet configuration.

    Only the validator container reads this. The runner image must NOT depend
    on or read this config. Wallet files are mounted read-only into the
    validator container only — the agent sandbox never sees them.
    """

    wallet_path: Path = constants.BITTENSOR.wallet_path
    wallet_name: str = constants.BITTENSOR.wallet_name
    wallet_hotkey: str = constants.BITTENSOR.wallet_hotkey
    netuid: int = constants.BITTENSOR.netuid
    subtensor_network: str = constants.BITTENSOR.subtensor_network
    signing_required: bool = constants.BITTENSOR.signing_required


class StorageConfig(BaseModel):
    backend: str = constants.STORAGE.backend
    data_dir: Path = constants.STORAGE.data_dir
    persist_traces: bool = constants.STORAGE.persist_traces


class GatewayConfig(BaseModel):
    default_timeout_ms: int = constants.GATEWAY.default_timeout_ms
    max_retries: int = constants.GATEWAY.max_retries
    debug_log_raw_response: bool = constants.GATEWAY.debug_log_raw_response
    debug_raw_response_max_chars: int = constants.GATEWAY.debug_raw_response_max_chars


class ValidatorLoopConfig(BaseModel):
    """Long-running Bittensor validator configuration.

    Controls the per-mechanism enable toggles and weight shares that the
    main ``Validator`` reads at each epoch. When ``loop_enabled`` is false,
    the entrypoint runs only the local signing proxy + agent orchestrator
    daemon for dev use.
    """

    loop_enabled: bool = constants.VALIDATOR_LOOP.loop_enabled
    almanac_enabled: bool = constants.VALIDATOR_LOOP.almanac_enabled
    arcratio_enabled: bool = constants.VALIDATOR_LOOP.arcratio_enabled
    metadata_manager_enabled: bool = constants.VALIDATOR_LOOP.metadata_manager_enabled
    almanac_weight_share: float = constants.VALIDATOR_LOOP.almanac_weight_share
    arcratio_weight_share: float = constants.VALIDATOR_LOOP.arcratio_weight_share
    almanac_api_url: Optional[str] = constants.VALIDATOR_LOOP.almanac_api_url
    use_synthetic_trading_data: bool = constants.VALIDATOR_LOOP.use_synthetic_trading_data
    write_trading_history: bool = constants.VALIDATOR_LOOP.write_trading_history
    db_score_logging: bool = constants.VALIDATOR_LOOP.db_score_logging
    wandb_enabled: bool = constants.VALIDATOR_LOOP.wandb_enabled
    rolling_window_days: int = constants.VALIDATOR_LOOP.rolling_window_days
    metadata_update_interval_seconds: int = constants.VALIDATOR_LOOP.metadata_update_interval_seconds
    metadata_batch_size: int = constants.VALIDATOR_LOOP.metadata_batch_size
    metadata_batch_delay_seconds: float = constants.VALIDATOR_LOOP.metadata_batch_delay_seconds
    metadata_per_uid_delay_seconds: float = constants.VALIDATOR_LOOP.metadata_per_uid_delay_seconds
    metadata_use_bulk_commitments: bool = constants.VALIDATOR_LOOP.metadata_use_bulk_commitments
    set_weights_enabled: bool = constants.VALIDATOR_LOOP.set_weights_enabled


class AppConfig(BaseModel):
    validator: ValidatorConfig = Field(default_factory=ValidatorConfig)
    bittensor: BittensorConfig = Field(default_factory=BittensorConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    loop: ValidatorLoopConfig = Field(default_factory=ValidatorLoopConfig)
    log_level: str = constants.LOG_LEVEL

    @classmethod
    def load_default(cls) -> "AppConfig":
        """Build config from code defaults in :mod:`src.core.constants`.

        Runtime settings are intentionally code-driven. Environment variables are
        reserved for sensitive provider credentials and are read in gateway
        provider clients, not here.
        """
        return cls()
