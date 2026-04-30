#!/usr/bin/env python3
"""Start the validator-local signing proxy.

Listens on a UNIX domain socket inside the validator container. Agent
sandboxes bind-mount that socket and POST every provider call through it.

The proxy loads the validator's Bittensor hotkey on startup, signs every
outbound request to the central provider gateway with sr25519, and enforces
per-track allowlists.

Usage:
    python scripts/run_local_proxy.py [--socket /var/run/arcratio/proxy.sock]
                                       [--unsafe-no-signing]

For local development without a Bittensor wallet, pass --unsafe-no-signing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn  # noqa: E402

from src.core.config import AppConfig  # noqa: E402
from src.gateway.local_proxy import create_app  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("arcratio.local_proxy")


def main() -> None:
    parser = argparse.ArgumentParser(description="Arcratio Validator Local Proxy")
    parser.add_argument(
        "--socket",
        default=None,
        help=(
            "Path to the UNIX domain socket to bind. Defaults to "
            "<validator.sandbox_socket_dir>/proxy.sock from config."
        ),
    )
    parser.add_argument(
        "--unsafe-no-signing",
        action="store_true",
        help=(
            "Disable Bittensor signing — proxy will forward unsigned requests. "
            "Dev/contributor use only."
        ),
    )
    args = parser.parse_args()

    cfg = AppConfig.load_default()
    if args.unsafe_no_signing:
        cfg.bittensor.signing_required = False

    socket_path = Path(args.socket) if args.socket else cfg.validator.sandbox_socket_dir / "proxy.sock"
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        socket_path.unlink()

    app = create_app(cfg)

    print()
    print("=" * 60)
    print("  ARCRATIO VALIDATOR LOCAL PROXY")
    print(f"  Socket:        {socket_path}")
    print(f"  Signing:       {'DISABLED (dev)' if not cfg.bittensor.signing_required else 'ENABLED'}")
    print(f"  Wallet:        {cfg.bittensor.wallet_path}/{cfg.bittensor.wallet_name}/{cfg.bittensor.wallet_hotkey}")
    print(f"  Netuid:        {cfg.bittensor.netuid}")
    print("=" * 60)
    print()

    uvicorn.run(app, uds=str(socket_path), log_level="info")


if __name__ == "__main__":
    main()
