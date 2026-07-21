"""Project-wide non-sensitive runtime defaults.

This module is the canonical source of truth for validator/runtime settings.
Operators should tune these values in code and redeploy, rather than
depending on large `.env` files.

Environment variables are reserved for sensitive or host-specific overrides
(for example API keys in gateway provider clients). The core validator loop,
sandbox, and blend configuration should come from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ValidatorDefaults:
    available_providers: list[str] = field(default_factory=lambda: ["polymarket", "web_search"])
    sandbox_type: str = "docker_runc"
    sandbox_image: str = "almanac/agent-runner:latest"
    sandbox_timeout_seconds: int = 240
    sandbox_max_concurrent: int = 8
    sandbox_memory_mb: int = 1024
    sandbox_cpus: float = 1.0
    sandbox_pids_limit: int = 256
    sandbox_socket_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "var/run/almanac")
    sandbox_socket_host_bind: str | None = None


@dataclass(frozen=True)
class BittensorDefaults:
    # Use the current runtime user's home so host runs resolve to
    # /home/<user>/.bittensor/wallets while container runs as root still
    # resolve to /root/.bittensor/wallets.
    wallet_path: Path = field(default_factory=lambda: Path.home() / ".bittensor" / "wallets")
    wallet_name: str = "default"
    wallet_hotkey: str = "default"
    netuid: int = 0
    subtensor_network: str = "finney"
    signing_required: bool = True


@dataclass(frozen=True)
class StorageDefaults:
    backend: str = "json"
    data_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "data")
    # When False, traces are not written under data/traces;
    persist_traces: bool = False


@dataclass(frozen=True)
class GatewayDefaults:
    # HTTP timeout for every provider-call hop (agent → proxy → gateway → upstream).
    # Keep aligned with ``ValidatorDefaults.sandbox_timeout_seconds``.
    call_timeout_seconds: int = 240
    # Full raw response dumps are off by default; enable via constant or
    # GATEWAY_DEBUG_RAW_RESPONSE=1. Successful calls still emit a short INFO line.
    debug_log_raw_response: bool = False
    # 0 means "no truncation" when debug logging is enabled.
    debug_raw_response_max_chars: int = 0


@dataclass(frozen=True)
class ValidatorLoopDefaults:
    loop_enabled: bool = True
    market_enabled: bool = True
    forecasting_enabled: bool = True
    metadata_manager_enabled: bool = True
    market_weight_share: float = 0.95
    forecasting_weight_share: float = 0.05
    market_api_url: str | None = None
    use_synthetic_trading_data: bool = False
    write_trading_history: bool = False
    db_score_logging: bool = False
    wandb_enabled: bool = False
    rolling_window_days: int = 30
    metadata_update_interval_seconds: int = 3600
    metadata_batch_size: int = 10
    metadata_batch_delay_seconds: float = 2.0
    metadata_per_uid_delay_seconds: float = 0.1
    metadata_use_bulk_commitments: bool = True
    set_weights_enabled: bool = True
    assignment_execution_cooldown_seconds: int = 300


VALIDATOR = ValidatorDefaults()
BITTENSOR = BittensorDefaults()
STORAGE = StorageDefaults()
GATEWAY = GatewayDefaults()
VALIDATOR_LOOP = ValidatorLoopDefaults()
ORCHESTRATOR_API_URL = "https://portal.almnc.ai"
GATEWAY_SERVICE_URL = "https://portal.almnc.ai"
LOG_LEVEL = "INFO"
