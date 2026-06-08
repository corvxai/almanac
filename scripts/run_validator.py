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

from tabulate import tabulate

from src.core.config import AppConfig
from src.storage.json_store import JsonTraceStore
from src.validator.validator import Validator, start_local_proxy


_ASCII_BANNER = r"""

    ________________________________________________________________________________________________________________________________________

_____╱╲╲╲╲╲╲╲╲╲_____╱╲╲╲______________╱╲╲╲╲____________╱╲╲╲╲_____╱╲╲╲╲╲╲╲╲╲_____╱╲╲╲╲╲_____╱╲╲╲_____╱╲╲╲╲╲╲╲╲╲___________╱╲╲╲╲╲╲╲╲╲_        
 ___╱╲╲╲╲╲╲╲╲╲╲╲╲╲__╲╱╲╲╲_____________╲╱╲╲╲╲╲╲________╱╲╲╲╲╲╲___╱╲╲╲╲╲╲╲╲╲╲╲╲╲__╲╱╲╲╲╲╲╲___╲╱╲╲╲___╱╲╲╲╲╲╲╲╲╲╲╲╲╲______╱╲╲╲╱╱╱╱╱╱╱╱__       
  __╱╲╲╲╱╱╱╱╱╱╱╱╱╲╲╲_╲╱╲╲╲_____________╲╱╲╲╲╱╱╲╲╲____╱╲╲╲╱╱╲╲╲__╱╲╲╲╱╱╱╱╱╱╱╱╱╲╲╲_╲╱╲╲╲╱╲╲╲__╲╱╲╲╲__╱╲╲╲╱╱╱╱╱╱╱╱╱╲╲╲___╱╲╲╲╱___________      
   _╲╱╲╲╲_______╲╱╲╲╲_╲╱╲╲╲_____________╲╱╲╲╲╲╱╱╱╲╲╲╱╲╲╲╱_╲╱╲╲╲_╲╱╲╲╲_______╲╱╲╲╲_╲╱╲╲╲╱╱╲╲╲_╲╱╲╲╲_╲╱╲╲╲_______╲╱╲╲╲__╱╲╲╲_____________     
    _╲╱╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲_╲╱╲╲╲_____________╲╱╲╲╲__╲╱╱╱╲╲╲╱___╲╱╲╲╲_╲╱╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲_╲╱╲╲╲╲╱╱╲╲╲╲╱╲╲╲_╲╱╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲_╲╱╲╲╲_____________    
     _╲╱╲╲╲╱╱╱╱╱╱╱╱╱╲╲╲_╲╱╲╲╲_____________╲╱╲╲╲____╲╱╱╱_____╲╱╲╲╲_╲╱╲╲╲╱╱╱╱╱╱╱╱╱╲╲╲_╲╱╲╲╲_╲╱╱╲╲╲╱╲╲╲_╲╱╲╲╲╱╱╱╱╱╱╱╱╱╲╲╲_╲╱╱╲╲╲____________   
      _╲╱╲╲╲_______╲╱╲╲╲_╲╱╲╲╲_____________╲╱╲╲╲_____________╲╱╲╲╲_╲╱╲╲╲_______╲╱╲╲╲_╲╱╲╲╲__╲╱╱╲╲╲╲╲╲_╲╱╲╲╲_______╲╱╲╲╲__╲╱╱╱╲╲╲__________  
       _╲╱╲╲╲_______╲╱╲╲╲_╲╱╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲_╲╱╲╲╲_____________╲╱╲╲╲_╲╱╲╲╲_______╲╱╲╲╲_╲╱╲╲╲___╲╱╱╲╲╲╲╲_╲╱╲╲╲_______╲╱╲╲╲____╲╱╱╱╱╲╲╲╲╲╲╲╲╲_ 
        _╲╱╱╱________╲╱╱╱__╲╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱__╲╱╱╱______________╲╱╱╱__╲╱╱╱________╲╱╱╱__╲╱╱╱_____╲╱╱╱╱╱__╲╱╱╱________╲╱╱╱________╲╱╱╱╱╱╱╱╱╱__

    ________________________________________________________________________________________________________________________________________
"""


class _ColumnFormatter(logging.Formatter):
    """Formatter with compact, centered level/logger columns."""

    LEVEL_COL_WIDTH = 8
    LOGGER_COL_WIDTH = 24

    @staticmethod
    def _center(value: str, width: int) -> str:
        if len(value) >= width:
            return value
        return value.center(width)

    def format(self, record: logging.LogRecord) -> str:
        record.asctime_col = self.formatTime(record, self.datefmt)
        record.level_col = self._center(record.levelname, self.LEVEL_COL_WIDTH)
        record.name_col = self._center(record.name, self.LOGGER_COL_WIDTH)
        return super().format(record)


class _ColorColumnFormatter(_ColumnFormatter):
    """Colorized variant:
    - timestamp always blue
    - levelname only is colorized (INFO = bold white)
    """

    _RESET = "\033[0m"
    _BLUE = "\033[34m"
    _LEVEL_COLORS = {
        logging.DEBUG: "\033[36m",    # cyan
        logging.INFO: "\033[1;37m",   # bold white
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",    # red
        logging.CRITICAL: "\033[35m", # magenta
    }

    def format(self, record: logging.LogRecord) -> str:
        record.asctime_col = f"{self._BLUE}{self.formatTime(record, self.datefmt)}{self._RESET}"
        centered_level = self._center(record.levelname, self.LEVEL_COL_WIDTH)
        level_color = self._LEVEL_COLORS.get(record.levelno, "")
        if level_color:
            record.level_col = f"{level_color}{centered_level}{self._RESET}"
        else:
            record.level_col = centered_level
        record.name_col = self._center(record.name, self.LOGGER_COL_WIDTH)
        return logging.Formatter.format(self, record)


def _setup_logging(level: str, *, wire_debug: bool = False, color: bool = False) -> None:
    root_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(root_level)

    stream = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime_col)s | %(level_col)s | %(name_col)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter: logging.Formatter
    if color and hasattr(stream.stream, "isatty") and stream.stream.isatty():
        formatter = _ColorColumnFormatter(fmt=fmt, datefmt=datefmt)
    else:
        formatter = _ColumnFormatter(fmt=fmt, datefmt=datefmt)
    stream.setFormatter(formatter)
    root_logger.addHandler(stream)

    # Keep bittensor's internal listener active (it manages state transitions),
    # but prevent duplicate propagation into root and align formatter/style.
    bt_logger = logging.getLogger("bittensor")
    bt_logger.setLevel(root_level)
    bt_logger.propagate = False
    try:
        import bittensor as bt  # type: ignore

        listener = getattr(bt.logging, "_listener", None)
        if listener is not None:
            for h in getattr(listener, "handlers", ()):
                try:
                    h.setFormatter(formatter)
                    h.setLevel(root_level)
                except Exception:
                    pass
    except Exception:
        # If bittensor is not importable yet, we still have sane root logging.
        pass

    wire_level = logging.DEBUG if wire_debug else logging.WARNING
    for noisy in (
        "async_substrate_interface",
        "websockets",
        "websockets.client",
        "websockets.server",
        "httpx",
        "httpcore",
    ):
        logging.getLogger(noisy).setLevel(wire_level)


def _log_startup_config_table(log: logging.Logger, config: AppConfig, args: argparse.Namespace) -> None:
    """Emit one concise startup configuration table (no duplicated params)."""
    logging_flags = {
        "trace": bool(args.logging_trace),
        "debug": bool(args.logging_debug or args.logging_trace),
        "info": bool(args.logging_info or (not args.logging_debug and not args.logging_trace)),
        "color": bool(args.logging_color),
    }
    rows = [
        ("bittensor", "netuid", config.bittensor.netuid),
        ("bittensor", "subtensor.network", config.bittensor.subtensor_network),
        ("bittensor", "wallet.name", config.bittensor.wallet_name),
        ("bittensor", "wallet.hotkey", config.bittensor.wallet_hotkey),
        ("bittensor", "wallet.path", str(config.bittensor.wallet_path)),
        ("bittensor", "signing_required", config.bittensor.signing_required),
        ("logging", "trace", logging_flags["trace"]),
        ("logging", "debug", logging_flags["debug"]),
        ("logging", "info", logging_flags["info"]),
        ("logging", "color", logging_flags["color"]),
        ("logging", "wire.debug", bool(args.wire_debug)),
        ("validator_loop", "enabled", config.loop.loop_enabled),
        ("validator_loop", "almanac_enabled", config.loop.almanac_enabled),
        ("validator_loop", "arcratio_enabled", config.loop.arcratio_enabled),
        ("validator_loop", "metadata_manager_enabled", config.loop.metadata_manager_enabled),
        ("validator_loop", "almanac_weight_share", config.loop.almanac_weight_share),
        ("validator_loop", "arcratio_weight_share", config.loop.arcratio_weight_share),
        ("validator_loop", "rolling_window_days", config.loop.rolling_window_days),
        ("validator_loop", "metadata_update_interval_seconds", config.loop.metadata_update_interval_seconds),
        ("validator_loop", "metadata_batch_size", config.loop.metadata_batch_size),
        ("validator_loop", "metadata_batch_delay_seconds", config.loop.metadata_batch_delay_seconds),
        ("validator_loop", "metadata_per_uid_delay_seconds", config.loop.metadata_per_uid_delay_seconds),
        ("validator_loop", "metadata_use_bulk_commitments", config.loop.metadata_use_bulk_commitments),
        ("validator_loop", "write_trading_history", config.loop.write_trading_history),
        ("validator_loop", "db_score_logging", config.loop.db_score_logging),
        ("validator_loop", "wandb_enabled", config.loop.wandb_enabled),
        ("runtime", "proxy_enabled", not args.no_proxy),
        ("runtime", "data_dir", str(config.storage.data_dir)),
    ]
    table = tabulate(rows, headers=("section", "key", "value"), tablefmt="simple_outline")
    log.info("Startup config:\n%s", table)


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
    parser.add_argument(
        "--wire.debug",
        dest="wire_debug",
        action="store_true",
        help="Enable DEBUG logs for substrate/websocket/http transport internals.",
    )
    parser.add_argument(
        "--logging.color",
        dest="logging_color",
        action="store_true",
        help="Enable colorized log output for terminal runs.",
    )
    logging_group = parser.add_mutually_exclusive_group()
    logging_group.add_argument(
        "--logging.trace",
        dest="logging_trace",
        action="store_true",
        help="Set validator log level to TRACE-equivalent (DEBUG) and enable wire debug logs.",
    )
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
    if args.logging_trace:
        config.log_level = "DEBUG"
        args.wire_debug = True
    elif args.logging_debug:
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

    _setup_logging(
        config.log_level,
        wire_debug=args.wire_debug,
        color=args.logging_color,
    )
    print(_ASCII_BANNER, flush=True)
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
    _log_startup_config_table(log, config, args)

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
