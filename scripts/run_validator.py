#!/usr/bin/env python3
"""Entrypoint for the arcratio Bittensor validator.

Starts the local signing proxy + agent orchestrator on a daemon
thread (so existing tooling that spawns agent containers keeps working),
then runs the main Validator loop — which scores each enabled
incentivemechanism (Almanac, arcratio), blends the score vectors, and emits one
``set_weights`` call per epoch.

For ad-hoc / dev runs of just the agent orchestrator (no chain weight
setting), use ``scripts/run_forecast.py`` instead.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import AppConfig
from src.storage.json_store import JsonTraceStore
from src.validator.validator import Validator, start_local_proxy


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(name)-28s  %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SN41 Bittensor validator")
    parser.add_argument(
        "--netuid",
        type=int,
        default=None,
        help="Override subnet netuid for this run.",
    )
    parser.add_argument(
        "--wallet.name",
        dest="wallet_name",
        default=None,
        help="Override bittensor wallet name for this run.",
    )
    parser.add_argument(
        "--wallet.hotkey",
        dest="wallet_hotkey",
        default=None,
        help="Override bittensor wallet hotkey for this run.",
    )
    parser.add_argument(
        "--wallet.path",
        dest="wallet_path",
        type=Path,
        default=None,
        help="Override bittensor wallet path for this run.",
    )
    parser.add_argument(
        "--subtensor.network",
        dest="subtensor_network",
        default=None,
        help="Override subtensor network (for example: finney, test, local).",
    )
    parser.add_argument(
        "--wandb.off",
        dest="wandb_off",
        action="store_true",
        help="Disable wandb integration for this run.",
    )
    parser.add_argument(
        "--db_score_logging",
        dest="db_score_logging",
        action="store_true",
        help="Enable Almanac postgres score logging for this run.",
    )
    parser.add_argument(
        "--write_trading_history",
        dest="write_trading_history",
        action="store_true",
        help="Write fetched Almanac trading history to <data-dir>/trading_history.json each scoring epoch.",
    )
    parser.add_argument(
        "--metadata_manager.off",
        dest="metadata_manager_off",
        action="store_true",
        help="Disable Almanac metadata manager thread for this run.",
    )
    logging_group = parser.add_mutually_exclusive_group()
    logging_group.add_argument(
        "--logging.debug",
        dest="logging_debug",
        action="store_true",
        help="Set validator log level to DEBUG for this run.",
    )
    logging_group.add_argument(
        "--logging.info",
        dest="logging_info",
        action="store_true",
        help="Set validator log level to INFO for this run.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Path to the data directory (default: <repo>/data)",
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help=(
            "Skip starting the local signing proxy + orchestrator daemon. "
            "Only useful when this process is run alongside a separately-launched proxy."
        ),
    )
    parser.add_argument(
        "--unsafe-no-signing",
        action="store_true",
        help=(
            "Disable Bittensor request signing on the validator-local proxy. "
            "Dev/contributor use only — never run production this way."
        ),
    )
    args = parser.parse_args()

    config = AppConfig.load_default()

    # Per-run CLI overrides (constants stay the default source of truth).
    if args.netuid is not None:
        config.bittensor.netuid = args.netuid
    if args.wallet_name:
        config.bittensor.wallet_name = args.wallet_name
    if args.wallet_hotkey:
        config.bittensor.wallet_hotkey = args.wallet_hotkey
    if args.wallet_path is not None:
        config.bittensor.wallet_path = args.wallet_path
    if args.subtensor_network:
        config.bittensor.subtensor_network = args.subtensor_network
    if args.logging_debug:
        config.log_level = "DEBUG"
    elif args.logging_info:
        config.log_level = "INFO"
    if args.wandb_off:
        config.loop.wandb_enabled = False
    if args.db_score_logging:
        config.loop.db_score_logging = True
    if args.write_trading_history:
        config.loop.write_trading_history = True
    if args.metadata_manager_off:
        config.loop.metadata_manager_enabled = False

    config.storage.data_dir = args.data_dir
    if args.unsafe_no_signing:
        config.bittensor.signing_required = False

    _setup_logging(config.log_level)
    log = logging.getLogger("arcratio.run_validator")

    log.info("Bittensor netuid=%d network=%s", config.bittensor.netuid, config.bittensor.subtensor_network)
    log.info(
        "Validator loop: enabled=%s almanac=%s (share=%.2f) arcratio=%s (share=%.2f)",
        config.loop.loop_enabled,
        config.loop.almanac_enabled,
        config.loop.almanac_weight_share,
        config.loop.arcratio_enabled,
        config.loop.arcratio_weight_share,
    )

    if not args.no_proxy:
        state = start_local_proxy(config)
        kp = state.loaded_keypair
        log.info(
            "Local signing proxy ready: %s",
            "ENABLED hotkey=" + kp.hotkey_ss58 if kp else "DISABLED (dev)",
        )

    store = JsonTraceStore(data_dir=config.storage.data_dir)

    if not config.loop.loop_enabled:
        log.warning(
            "Validator loop is disabled in config constants; running proxy + orchestrator daemon only. "
            "Set src/core/constants.py -> VALIDATOR_LOOP.loop_enabled=True to start the chain-facing loop."
        )
        # Block forever so the daemon thread keeps serving the proxy.
        import threading
        threading.Event().wait()
        return

    validator = Validator(config=config, store=store)
    try:
        validator.run()
    except KeyboardInterrupt:
        log.info("Keyboard interrupt; shutting down.")
    finally:
        validator.stop()


if __name__ == "__main__":
    main()
