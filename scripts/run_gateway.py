#!/usr/bin/env python3
"""Start the SIMULATED Provider Gateway — a local test double.

This is NOT the production gateway service (that lives in its own repo). It
exists so validators and miners can exercise the full validator → local proxy →
gateway → provider path locally. It runs **mock-by-default**: deterministic,
offline, keyless, no spend. Pass --live to load real provider API keys from
.env and hit upstream APIs (only do this intentionally).

Usage:
    python scripts/run_gateway.py [--port 8077] [--host 127.0.0.1] [--live]
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

from src.gateway.providers.claude import ClaudeProvider
from src.gateway.providers.openrouter import OpenRouterProvider


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("arcratio.gateway")


_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _bind_policy_error(host: str, *, require_signature: bool, allow_open: bool) -> str | None:
    """Return an error message if this (host, auth) combination is unsafe to serve.

    The gateway holds upstream provider API keys. Binding to a non-loopback
    address without signature enforcement leaves an OPEN, unauthenticated proxy
    to those keys, so we refuse rather than only warn. Callers can opt in
    explicitly (REQUIRE_SIGNATURE, or --allow-open-unauthenticated /
    ALLOW_OPEN_GATEWAY=1) when they understand the exposure.
    """
    if host in _LOOPBACK_HOSTS:
        return None
    if require_signature or allow_open:
        return None
    return (
        f"refusing to bind {host} with REQUIRE_SIGNATURE off: this would be an "
        "OPEN, unauthenticated proxy to your provider API keys. Bind to "
        "127.0.0.1 (default), set REQUIRE_SIGNATURE=true, or pass "
        "--allow-open-unauthenticated if you truly intend an open relay."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Arcratio Provider Gateway")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Bind address (default 127.0.0.1, loopback-only). A non-loopback "
            "address such as 0.0.0.0 exposes the gateway — and your provider API "
            "keys — to other hosts/containers and requires REQUIRE_SIGNATURE=true "
            "or --allow-open-unauthenticated."
        ),
    )
    parser.add_argument("--port", "-p", type=int, default=8077, help="Port")
    parser.add_argument(
        "--allow-open-unauthenticated",
        action="store_true",
        help=(
            "Permit binding a non-loopback address without signature enforcement. "
            "This is an open, unauthenticated relay to your provider keys — only "
            "use it on a trusted/isolated network."
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Load real provider API keys from .env and call upstream APIs. "
            "Default is mock (deterministic, offline, keyless, no spend). "
            "Can also be enabled with GATEWAY_LIVE=1."
        ),
    )
    args = parser.parse_args()
    live = args.live or _env_flag("GATEWAY_LIVE")

    require_signature = _env_flag("REQUIRE_SIGNATURE")
    allow_open = args.allow_open_unauthenticated or _env_flag("ALLOW_OPEN_GATEWAY")
    policy_error = _bind_policy_error(
        args.host, require_signature=require_signature, allow_open=allow_open
    )
    if policy_error:
        log.error("%s", policy_error)
        raise SystemExit(2)
    if args.host not in _LOOPBACK_HOSTS and not require_signature:
        log.warning(
            "Gateway bound to %s as an OPEN, unauthenticated proxy to your "
            "provider API keys (explicitly allowed). Ensure the network is trusted.",
            args.host,
        )

    # LLM providers — mock-by-default. In live mode they read API keys from .env
    # and call upstream; otherwise they return deterministic offline payloads, so
    # validator/miner test runs need no keys, no network, and incur no spend.
    use_mock = not live
    try:
        claude = ClaudeProvider(use_mock=use_mock)
        register_provider(claude)
        log.info("Claude provider: %s", claude.mode.upper())
    except Exception as exc:
        log.warning("Claude provider failed to init: %s", exc)

    try:
        openrouter = OpenRouterProvider(use_mock=use_mock)
        register_provider(openrouter)
        log.info("OpenRouter provider: %s", openrouter.mode.upper())
    except Exception as exc:
        log.warning("OpenRouter provider failed to init: %s", exc)

    print()
    print("=" * 60)
    print("  ARCRATIO PROVIDER GATEWAY  [SIMULATED — not for production]")
    print(f"  Mode:       {'LIVE (real API keys, real spend)' if live else 'MOCK (offline, keyless)'}")
    print(f"  Listening on http://{args.host}:{args.port}")
    print(f"  Health:     http://localhost:{args.port}/health")
    print(f"  Providers:  http://localhost:{args.port}/v1/gateway/providers")
    if not live:
        print("  Tip:        pass --live to use real providers/keys from .env")
    print("=" * 60)
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
