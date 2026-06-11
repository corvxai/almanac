#!/usr/bin/env python3
"""Start the Provider Gateway HTTP service.

Loads .env for API keys, registers all available providers (real + mock),
and serves on the configured port. Run this in its own terminal tab.

Usage:
    python scripts/run_gateway.py [--port 8077] [--host 0.0.0.0]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import uvicorn

from src.gateway.server import app, register_provider

from src.gateway.providers.polymarket import PolymarketProvider
from src.gateway.providers.web_search import WebSearchProvider
from src.gateway.providers.claude import ClaudeProvider
from src.gateway.providers.openrouter import OpenRouterProvider


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("arcratio.gateway")


def main() -> None:
    parser = argparse.ArgumentParser(description="Arcratio Provider Gateway")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address (0.0.0.0 exposes to other hosts/containers)",
    )
    parser.add_argument("--port", "-p", type=int, default=8077, help="Port")
    args = parser.parse_args()

    # The gateway holds upstream provider API keys. Binding to a non-loopback
    # address without signature enforcement leaves an open, unauthenticated
    # proxy to those keys. The default stays 0.0.0.0 so the documented Docker
    # dev flow (containers reaching the host via host.docker.internal) keeps
    # working, but surface the risk loudly.
    require_signature = os.environ.get("REQUIRE_SIGNATURE", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not require_signature:
        log.warning(
            "Gateway bound to %s with REQUIRE_SIGNATURE off: this is an OPEN, "
            "unauthenticated proxy to your provider API keys. Bind to 127.0.0.1 "
            "or set REQUIRE_SIGNATURE=true before exposing it.",
            args.host,
        )

    # Real providers (need API keys)
    try:
        register_provider(ClaudeProvider())
        log.info("Claude provider: LIVE")
    except Exception as exc:
        log.warning("Claude provider failed to init: %s", exc)

    try:
        openrouter = OpenRouterProvider()
        register_provider(openrouter)
        log.info("OpenRouter provider: %s", openrouter.mode.upper())
    except Exception as exc:
        log.warning("OpenRouter provider failed to init: %s", exc)

    # Mock providers (no keys needed)
    register_provider(PolymarketProvider())
    log.info("Polymarket provider: MOCK")

    register_provider(WebSearchProvider())
    log.info("Web Search provider: MOCK")

    print()
    print("=" * 60)
    print("  ARCRATIO PROVIDER GATEWAY")
    print(f"  Listening on http://{args.host}:{args.port}")
    print(f"  Health:     http://localhost:{args.port}/health")
    print(f"  Providers:  http://localhost:{args.port}/v1/gateway/providers")
    print("=" * 60)
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
