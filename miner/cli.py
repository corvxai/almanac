#!/usr/bin/env python3
"""Miner CLI for orchestrator interactions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
load_dotenv(_PROJECT_ROOT / ".env")

from src.core.constants import ORCHESTRATOR_API_URL
from src.agent.context import ForecastingContext
from src.agent.loader import load_agent_class_from_code
from src.core.schemas import AgentResult
from src.gateway.portal_client import PortalGateway, fetch_portal_event

DEFAULT_ORCHESTRATOR_URL = ORCHESTRATOR_API_URL
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_TEST_TIMEOUT_SECONDS = 120.0
MAX_AGENT_FILE_BYTES = 2 * 1024 * 1024  # 2MB
# The API does not expose this endpoint yet; keep the command disabled below.
LIST_AGENTS_ENDPOINT = "v1/agents/list-agents"
UPLOAD_AGENT_ENDPOINT = "v1/agents/submit-agent"
AUTH_DOMAIN = "sub41-agent-v1"


@dataclass(frozen=True)
class SignedHeaders:
    headers: dict[str, str]
    path_and_query: str


def _resolve_orchestrator_url(args: argparse.Namespace) -> str:
    return (
        args.orchestrator_url
        or os.getenv("FORECASTING_ORCHESTRATOR_URL")
        or DEFAULT_ORCHESTRATOR_URL
    ).rstrip("/")


def _resolve_timeout(args: argparse.Namespace) -> float:
    if args.timeout_seconds is not None:
        return args.timeout_seconds
    raw = os.getenv("FORECASTING_TIMEOUT_SECONDS")
    if raw is None:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        return float(raw)
    except ValueError:
        print("warning: FORECASTING_TIMEOUT_SECONDS is invalid; using default timeout.")
        return DEFAULT_TIMEOUT_SECONDS


def _url(base_url: str, endpoint: str) -> str:
    return f"{base_url}/{endpoint.lstrip('/')}"


def _resolve_wallet_path(args: argparse.Namespace) -> Path:
    value = args.wallet_path or os.getenv("FORECASTING_WALLET_PATH") or "~/.bittensor/wallets"
    return Path(value).expanduser()


def _resolve_wallet_name(args: argparse.Namespace) -> str:
    return args.wallet_name or os.getenv("FORECASTING_WALLET_NAME", "default")


def _resolve_wallet_hotkey_name(args: argparse.Namespace) -> str:
    return args.wallet_hotkey_name or os.getenv("FORECASTING_WALLET_HOTKEY", "default")


def _resolve_gateway_api_key(args: argparse.Namespace) -> str | None:
    return (
        args.gateway_api_key
        or os.getenv("GATEWAY_API_KEY")
        or os.getenv("FORECASTING_GATEWAY_API_KEY")
    )


def _canonical_message(
    *,
    method: str,
    path_and_query: str,
    subject_hotkey: str,
    nonce: str,
    timestamp_ms: int,
    body: bytes,
) -> bytes:
    body_sha256 = hashlib.sha256(body).hexdigest()
    message = "\n".join(
        [
            AUTH_DOMAIN,
            method.upper(),
            path_and_query,
            subject_hotkey,
            nonce,
            str(timestamp_ms),
            body_sha256,
        ]
    )
    return message.encode("utf-8")


def _sign_headers(
    *,
    role: str,
    keypair: Any,
    method: str,
    path_and_query: str,
    subject_hotkey: str,
    body: bytes,
) -> SignedHeaders:
    nonce = secrets.token_hex(12)
    timestamp_ms = int(time.time() * 1000)
    message = _canonical_message(
        method=method,
        path_and_query=path_and_query,
        subject_hotkey=subject_hotkey,
        nonce=nonce,
        timestamp_ms=timestamp_ms,
        body=body,
    )
    signature_hex = "0x" + keypair.sign(message).hex()
    return SignedHeaders(
        path_and_query=path_and_query,
        headers={
            f"x-{role}-hotkey": keypair.ss58_address,
            f"x-{role}-signature": signature_hex,
            f"x-{role}-nonce": nonce,
            f"x-{role}-timestamp": str(timestamp_ms),
        },
    )


def _load_hotkey_keypair(args: argparse.Namespace):
    wallet_path = _resolve_wallet_path(args)
    wallet_name = _resolve_wallet_name(args)
    hotkey_name = _resolve_wallet_hotkey_name(args)

    try:
        bittensor = __import__("bittensor")
    except Exception:
        # Fall back to the lightweight `bittensor_wallet` package, which exposes
        # the same `Wallet`/`wallet` constructors and is sufficient for signing
        # (mirrors the fallback already used in src/gateway/signing.py).
        try:
            bittensor = __import__("bittensor_wallet")
        except Exception as exc:
            print(
                "bittensor or bittensor_wallet is required for wallet signature "
                f"auth but neither is available: {exc}"
            )
            return None

    try:
        # Prefer a callable constructor. The full `bittensor` SDK exposes
        # `wallet`/`Wallet` as classes, but in `bittensor_wallet` `wallet` is a
        # submodule (not callable) while `Wallet` is the class.
        wallet_ctor = next(
            (
                c
                for c in (getattr(bittensor, "Wallet", None), getattr(bittensor, "wallet", None))
                if callable(c)
            ),
            None,
        )
        if wallet_ctor is None:
            print("failed to load bittensor wallet: module has no wallet constructor.")
            return None
        wallet = wallet_ctor(name=wallet_name, hotkey=hotkey_name, path=str(wallet_path))
        return wallet.hotkey
    except Exception as exc:
        hotkeys_dir = wallet_path / wallet_name / "hotkeys"
        available_hotkeys: list[str] = []
        if hotkeys_dir.exists():
            available_hotkeys = sorted(
                p.name
                for p in hotkeys_dir.iterdir()
                if p.is_file() and not p.name.endswith("pub.txt")
            )
        print(
            "failed to load bittensor hotkey "
            f"({wallet_path}/{wallet_name}/{hotkey_name}): {exc}"
        )
        if available_hotkeys:
            print("available hotkeys:", ", ".join(available_hotkeys))
            print("rerun with --wallet-hotkey-name <one_of_above>.")
        return None


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _request(
    *,
    method: str,
    endpoint: str,
    args: argparse.Namespace,
    params: dict[str, object] | None = None,
    data: object | None = None,
    extra_headers: dict[str, str] | None = None,
) -> httpx.Response | None:
    base_url = _resolve_orchestrator_url(args)
    timeout = _resolve_timeout(args)
    target = _url(base_url, endpoint)
    headers = extra_headers or {}
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.request(
                method=method,
                url=target,
                params=params,
                data=data,
                headers=headers,
            )
        response.raise_for_status()
        return response
    except httpx.HTTPStatusError as exc:
        print(f"request failed with status {exc.response.status_code}: {exc.request.url}")
        body = exc.response.text.strip()
        if body:
            try:
                _print_json(exc.response.json())
            except ValueError:
                print(body)
        return None
    except httpx.HTTPError as exc:
        print(f"network error calling orchestrator: {exc}")
        return None


def _handle_upload_agent(args: argparse.Namespace) -> int:
    if not args.agent_file.exists() or not args.agent_file.is_file():
        print(f"agent file does not exist or is not a file: {args.agent_file}")
        return 2

    if args.agent_file.suffix.lower() != ".py":
        print(f"agent file must be a .py file: {args.agent_file}")
        return 2

    file_size = args.agent_file.stat().st_size
    if file_size > MAX_AGENT_FILE_BYTES:
        print(
            "agent file exceeds max size "
            f"({file_size} bytes > {MAX_AGENT_FILE_BYTES} bytes)"
        )
        return 2

    body = args.agent_file.read_bytes()
    miner = _load_hotkey_keypair(args)
    if miner is None:
        return 2
    gateway_api_key = _resolve_gateway_api_key(args)
    if not gateway_api_key:
        print("gateway API key is required; pass --gateway-api-key or set GATEWAY_API_KEY.")
        return 2
    miner_hotkey = args.miner_hotkey or miner.ss58_address
    signed = _sign_headers(
        role="miner",
        keypair=miner,
        method="POST",
        path_and_query=f"/{UPLOAD_AGENT_ENDPOINT}",
        subject_hotkey=miner_hotkey,
        body=body,
    )
    headers = {
        **signed.headers,
        "Authorization": f"Bearer {gateway_api_key}",
        "content-type": "application/octet-stream",
        "x-agent-filename": args.agent_file.name or "agent.py",
    }

    response = _request(
        method="POST",
        endpoint=UPLOAD_AGENT_ENDPOINT,
        args=args,
        data=body,
        extra_headers=headers,
    )

    if response is None:
        return 1

    print(f"uploaded agent file: {args.agent_file}")
    print(f"minerHotkey: {miner_hotkey}")
    try:
        _print_json(response.json())
    except ValueError:
        print(response.text)
    return 0


def _handle_test_agent(args: argparse.Namespace) -> int:
    if not args.agent_file.exists() or not args.agent_file.is_file():
        print(f"agent file does not exist or is not a file: {args.agent_file}")
        return 2
    if args.agent_file.suffix.lower() != ".py":
        print(f"agent file must be a .py file: {args.agent_file}")
        return 2
    if args.agent_file.stat().st_size > MAX_AGENT_FILE_BYTES:
        print(f"agent file exceeds max size of {MAX_AGENT_FILE_BYTES} bytes")
        return 2

    gateway_api_key = _resolve_gateway_api_key(args)
    if not gateway_api_key:
        print("gateway API key is required; pass --gateway-api-key or set GATEWAY_API_KEY.")
        return 2

    try:
        source = args.agent_file.read_text(encoding="utf-8")
        agent_class = load_agent_class_from_code(source, filename=str(args.agent_file))
        _validate_agent_identity(agent_class)
        agent = agent_class()
    except Exception as exc:
        print(f"failed to load agent: {exc}")
        return 2

    base_url = _resolve_orchestrator_url(args)
    timeout = _resolve_test_timeout(args)
    try:
        event = fetch_portal_event(
            base_url=base_url,
            event_id=args.event_id,
            timeout=timeout,
        )
        with PortalGateway(
            base_url=base_url,
            api_key=gateway_api_key,
            timeout=timeout,
        ) as gateway:
            print(f"Event: {event.title}")
            print(f"Available providers: {', '.join(gateway.available_providers)}")
            started = time.monotonic()
            raw_result = agent.predict(ForecastingContext(event, gateway))
            duration_ms = int((time.monotonic() - started) * 1000)
            result = AgentResult.model_validate(raw_result)
            result = AgentResult.model_validate_json(result.model_dump_json())
            _print_test_result(result, gateway, duration_ms)
    except Exception as exc:
        print(f"agent test failed: {exc}")
        return 1
    return 0


def _resolve_test_timeout(args: argparse.Namespace) -> float:
    if args.timeout_seconds is not None or os.getenv("FORECASTING_TIMEOUT_SECONDS") is not None:
        return _resolve_timeout(args)
    return DEFAULT_TEST_TIMEOUT_SECONDS


def _validate_agent_identity(agent_class: type) -> None:
    agent_id = agent_class.__dict__.get("agent_id")
    if not isinstance(agent_id, UUID):
        raise RuntimeError("agent class must declare a stable agent_id UUID")
    agent_version = agent_class.__dict__.get("agent_version")
    if not isinstance(agent_version, str) or not agent_version.strip():
        raise RuntimeError("agent class must declare a non-empty agent_version")


def _print_test_result(
    result: AgentResult,
    gateway: PortalGateway,
    duration_ms: int,
) -> None:
    print()
    print("Agent result validated successfully.")
    print(f"Prediction: {result.prediction:.6f}")
    print(f"Duration: {duration_ms}ms")
    print(f"Belief path steps: {len(result.beliefPath)}")
    print(f"Reasoning: {result.reasoning[:500]}")
    print()
    print(f"Provider calls: {len(gateway.call_log)}")
    for index, call in enumerate(gateway.call_log):
        model = f" model={call.model}" if call.model else ""
        cost = f" costMicro={call.cost_micro}" if call.cost_micro is not None else ""
        print(f"  [{index}] {call.provider}{model} {call.latency_ms}ms{cost}")
    if gateway.call_log:
        balance = gateway.call_log[-1].balance_after_micro
        if balance is not None:
            print(f"Portal balanceAfterMicro: {balance}")
    print()
    print("Validated AgentResult JSON:")
    _print_json(result.model_dump(mode="json"))


def _handle_list_agents(args: argparse.Namespace) -> int:
    response = _request(
        method="GET",
        endpoint=LIST_AGENTS_ENDPOINT,
        args=args,
        params={"limit": args.limit, "offset": args.offset},
    )
    if response is None:
        return 1
    try:
        payload = response.json()
    except ValueError:
        print(response.text)
        return 0
    _print_json(payload)
    return 0


def _handle_buy_credits(args: argparse.Namespace) -> int:
    print("buy-credits: not implemented yet.")
    print("  credits flow and wallet-signature auth are deferred.")
    print(f"  amount: {args.amount}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--orchestrator-url",
        default=None,
        help=(
            "Override orchestrator base URL. Defaults to "
            f"{DEFAULT_ORCHESTRATOR_URL}."
        ),
    )
    common.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help=(
            "HTTP timeout in seconds. Defaults to FORECASTING_TIMEOUT_SECONDS, "
            f"otherwise {DEFAULT_TIMEOUT_SECONDS} ({DEFAULT_TEST_TIMEOUT_SECONDS} "
            "for test-agent)."
        ),
    )
    common.add_argument(
        "--gateway-api-key",
        default=None,
        help=(
            "Portal gateway API key for agent testing and submit auth. "
            "Defaults to GATEWAY_API_KEY or FORECASTING_GATEWAY_API_KEY."
        ),
    )

    parser = argparse.ArgumentParser(
        description="Almanac Forecasting miner CLI",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    upload_parser = subparsers.add_parser(
        "submit-agent",
        aliases=["upload-agent"],
        parents=[common],
        help="Submit a miner agent .py file to the orchestrator.",
    )
    upload_parser.add_argument("agent_file", type=Path, help="Path to the .py agent file.")
    upload_parser.add_argument(
        "--miner-hotkey",
        default=None,
        help="Miner hotkey address (defaults to wallet hotkey ss58).",
    )
    upload_parser.add_argument(
        "--miner-uid",
        type=int,
        default=None,
        help="Miner UID (retained for CLI compatibility; ignored by submit-agent API).",
    )
    upload_parser.add_argument(
        "--wallet-name",
        default=None,
        help="Bittensor wallet name (defaults to FORECASTING_WALLET_NAME or 'default').",
    )
    upload_parser.add_argument(
        "--wallet-hotkey-name",
        default=None,
        help="Bittensor wallet hotkey name (defaults to FORECASTING_WALLET_HOTKEY or 'default').",
    )
    upload_parser.add_argument(
        "--wallet-path",
        type=Path,
        default=None,
        help="Bittensor wallet path (defaults to FORECASTING_WALLET_PATH or ~/.bittensor/wallets).",
    )
    upload_parser.add_argument(
        "--subtensor-network",
        default=None,
        help="Retained for CLI compatibility; not used by submit-agent API.",
    )
    upload_parser.add_argument(
        "--netuid",
        type=int,
        default=None,
        help="Retained for CLI compatibility; not used by submit-agent API.",
    )
    upload_parser.set_defaults(handler=_handle_upload_agent)

    test_parser = subparsers.add_parser(
        "test-agent",
        parents=[common],
        help="Run an agent locally against live portal events and providers.",
    )
    test_parser.add_argument("agent_file", type=Path, help="Path to the .py agent file.")
    test_parser.add_argument(
        "--event-id",
        default=None,
        help="Portal event id to test (defaults to a random live event).",
    )
    test_parser.set_defaults(handler=_handle_test_agent)

    # Disabled until the gateway exposes LIST_AGENTS_ENDPOINT.
    # list_parser = subparsers.add_parser(
    #     "list-agents",
    #     parents=[common],
    #     help="List published agents from the orchestrator.",
    # )
    # list_parser.add_argument("--limit", type=int, default=25, help="Max results to return.")
    # list_parser.add_argument("--offset", type=int, default=0, help="Pagination offset.")
    # list_parser.set_defaults(handler=_handle_list_agents)

    credits_parser = subparsers.add_parser(
        "buy-credits",
        parents=[common],
        help="Buy credits (stub command for future implementation).",
    )
    credits_parser.add_argument("amount", type=float, help="Requested credit amount.")
    credits_parser.set_defaults(handler=_handle_buy_credits)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
