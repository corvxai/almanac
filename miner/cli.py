#!/usr/bin/env python3
"""Miner CLI for orchestrator interactions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

DEFAULT_ORCHESTRATOR_URL = "http://localhost:4000"
DEFAULT_TIMEOUT_SECONDS = 20.0
MAX_AGENT_FILE_BYTES = 2 * 1024 * 1024  # 2MB
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
        or os.getenv("ARCRATIO_ORCHESTRATOR_URL")
        or DEFAULT_ORCHESTRATOR_URL
    ).rstrip("/")


def _resolve_timeout(args: argparse.Namespace) -> float:
    if args.timeout_seconds is not None:
        return args.timeout_seconds
    raw = os.getenv("ARCRATIO_TIMEOUT_SECONDS")
    if raw is None:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        return float(raw)
    except ValueError:
        print("warning: ARCRATIO_TIMEOUT_SECONDS is invalid; using default timeout.")
        return DEFAULT_TIMEOUT_SECONDS


def _url(base_url: str, endpoint: str) -> str:
    return f"{base_url}/{endpoint.lstrip('/')}"


def _resolve_wallet_path(args: argparse.Namespace) -> Path:
    value = args.wallet_path or os.getenv("ARCRATIO_WALLET_PATH") or "~/.bittensor/wallets"
    return Path(value).expanduser()


def _resolve_wallet_name(args: argparse.Namespace) -> str:
    return args.wallet_name or os.getenv("ARCRATIO_WALLET_NAME", "default")


def _resolve_wallet_hotkey_name(args: argparse.Namespace) -> str:
    return args.wallet_hotkey_name or os.getenv("ARCRATIO_WALLET_HOTKEY", "default")


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
    except Exception as exc:
        print(f"bittensor is required for wallet signature auth but is not available: {exc}")
        return None

    try:
        wallet_ctor = getattr(bittensor, "wallet", None) or getattr(bittensor, "Wallet", None)
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
    parser = argparse.ArgumentParser(description="Arcratio miner CLI")
    parser.add_argument(
        "--orchestrator-url",
        default=None,
        help=(
            "Override orchestrator base URL. Defaults to ARCRATIO_ORCHESTRATOR_URL "
            f"or {DEFAULT_ORCHESTRATOR_URL}."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help=(
            "HTTP timeout in seconds. Defaults to ARCRATIO_TIMEOUT_SECONDS "
            f"or {DEFAULT_TIMEOUT_SECONDS}."
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    upload_parser = subparsers.add_parser(
        "submit-agent",
        aliases=["upload-agent"],
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
        help="Bittensor wallet name (defaults to ARCRATIO_WALLET_NAME or 'default').",
    )
    upload_parser.add_argument(
        "--wallet-hotkey-name",
        default=None,
        help="Bittensor wallet hotkey name (defaults to ARCRATIO_WALLET_HOTKEY or 'default').",
    )
    upload_parser.add_argument(
        "--wallet-path",
        type=Path,
        default=None,
        help="Bittensor wallet path (defaults to ARCRATIO_WALLET_PATH or ~/.bittensor/wallets).",
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

    list_parser = subparsers.add_parser(
        "list-agents",
        help="List published agents from the orchestrator.",
    )
    list_parser.add_argument("--limit", type=int, default=25, help="Max results to return.")
    list_parser.add_argument("--offset", type=int, default=0, help="Pagination offset.")
    list_parser.set_defaults(handler=_handle_list_agents)

    credits_parser = subparsers.add_parser(
        "buy-credits",
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
