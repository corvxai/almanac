#!/usr/bin/env python3
"""Miner CLI for orchestrator interactions."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path

import httpx

DEFAULT_ORCHESTRATOR_URL = "http://localhost:4000"
DEFAULT_TIMEOUT_SECONDS = 20.0
MAX_AGENT_FILE_BYTES = 2 * 1024 * 1024  # 2MB
LIST_AGENTS_ENDPOINT = "v1/agents/list-agents"
UPLOAD_AGENT_ENDPOINT = "v1/agents/submit-agent"


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


def _resolve_token(args: argparse.Namespace) -> str | None:
    return args.api_token or os.getenv("ARCRATIO_API_TOKEN")


def _url(base_url: str, endpoint: str) -> str:
    return f"{base_url}/{endpoint.lstrip('/')}"


def _headers(token: str | None) -> dict[str, str]:
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _resolve_wallet_path(args: argparse.Namespace) -> Path:
    return args.wallet_path or Path(os.getenv("ARCRATIO_WALLET_PATH", "~/.bittensor/wallets")).expanduser()


def _resolve_wallet_name(args: argparse.Namespace) -> str:
    return args.wallet_name or os.getenv("ARCRATIO_WALLET_NAME", "default")


def _resolve_wallet_hotkey_name(args: argparse.Namespace) -> str:
    return args.wallet_hotkey_name or os.getenv("ARCRATIO_WALLET_HOTKEY", "default")


def _resolve_subtensor_network(args: argparse.Namespace) -> str:
    return args.subtensor_network or os.getenv("ARCRATIO_SUBTENSOR_NETWORK", "finney")


def _resolve_netuid(args: argparse.Namespace) -> int | None:
    if args.netuid is not None:
        return args.netuid
    raw = os.getenv("ARCRATIO_NETUID")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _load_hotkey_keypair(args: argparse.Namespace):
    try:
        import bittensor  # type: ignore
    except Exception:
        print("bittensor is required for wallet signature auth but is not available.")
        return None

    wallet_path = _resolve_wallet_path(args)
    wallet_name = _resolve_wallet_name(args)
    hotkey_name = _resolve_wallet_hotkey_name(args)

    try:
        wallet_ctor = getattr(bittensor, "wallet", None) or getattr(bittensor, "Wallet", None)
        if wallet_ctor is None:
            print("failed to load bittensor wallet: module has no wallet constructor.")
            return None
        wallet = wallet_ctor(name=wallet_name, hotkey=hotkey_name, path=str(wallet_path))
        # Accessing hotkey may throw if keyfile is missing/unreadable.
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


def _build_upload_signature_headers(keypair, file_bytes: bytes) -> dict[str, str]:
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    payload = f"{keypair.ss58_address}:{file_hash}"
    signature = keypair.sign(payload.encode("utf-8"))
    token = base64.b64encode(signature).decode("utf-8")
    return {
        "Authorization": f"Bearer {token}",
        "Miner-Public-Key": keypair.public_key.hex(),
        "Miner": keypair.ss58_address,
        "X-Payload": payload,
    }


def _resolve_miner_uid(args: argparse.Namespace, miner_hotkey: str) -> int | None:
    if args.miner_uid is not None:
        return args.miner_uid

    netuid = _resolve_netuid(args)
    if netuid is None:
        print("miner UID was not provided and ARCRATIO_NETUID/--netuid is not set.")
        print("pass --miner-uid explicitly or configure --netuid for auto lookup.")
        return None

    try:
        import bittensor  # type: ignore
    except Exception:
        print("bittensor is required for miner UID lookup but is not available.")
        return None

    network = _resolve_subtensor_network(args)
    try:
        subtensor_ctor = getattr(bittensor, "subtensor", None) or getattr(bittensor, "Subtensor", None)
        if subtensor_ctor is None:
            print("failed to initialize subtensor: module has no subtensor constructor.")
            return None
        subtensor = subtensor_ctor(network=network)
        metagraph = subtensor.metagraph(netuid=netuid)
        hotkeys = list(metagraph.hotkeys)
    except Exception as exc:
        print(f"failed to query metagraph for network={network} netuid={netuid}: {exc}")
        return None

    if miner_hotkey not in hotkeys:
        print(
            f"hotkey {miner_hotkey} is not registered on network={network} netuid={netuid}."
        )
        return None
    return hotkeys.index(miner_hotkey)


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _request(
    *,
    method: str,
    endpoint: str,
    args: argparse.Namespace,
    params: dict[str, object] | None = None,
    data: dict[str, object] | None = None,
    files: dict[str, tuple[str, object, str]] | None = None,
    extra_headers: dict[str, str] | None = None,
) -> httpx.Response | None:
    base_url = _resolve_orchestrator_url(args)
    token = _resolve_token(args)
    timeout = _resolve_timeout(args)
    target = _url(base_url, endpoint)
    headers = _headers(token)
    if extra_headers:
        headers.update(extra_headers)
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.request(
                method=method,
                url=target,
                params=params,
                data=data,
                files=files,
                headers=headers,
            )
        response.raise_for_status()
        return response
    except httpx.HTTPStatusError as exc:
        print(f"request failed with status {exc.response.status_code}: {exc.request.url}")
        body = exc.response.text.strip()
        if body:
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

    with args.agent_file.open("rb") as handle:
        file_bytes = handle.read()

    keypair = _load_hotkey_keypair(args)
    if keypair is None:
        return 2

    miner_hotkey = args.miner_hotkey or keypair.ss58_address
    miner_uid = _resolve_miner_uid(args, miner_hotkey)
    if miner_uid is None:
        return 2

    signature_headers = _build_upload_signature_headers(keypair, file_bytes)

    with args.agent_file.open("rb") as handle:
        files = {"agentFile": (args.agent_file.name, handle, "text/x-python")}
        response = _request(
            method="POST",
            endpoint=UPLOAD_AGENT_ENDPOINT,
            args=args,
            data={"minerHotkey": miner_hotkey, "minerUid": str(miner_uid)},
            files=files,
            extra_headers=signature_headers,
        )

    if response is None:
        return 1

    print(f"uploaded agent file: {args.agent_file}")
    print(f"minerHotkey: {miner_hotkey}")
    print(f"minerUid: {miner_uid}")
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
        "--api-token",
        default=None,
        help="Optional bearer token for endpoints that use token auth.",
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
        "upload-agent",
        help="Upload a miner agent artifact to the orchestrator.",
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
        help="Miner UID. If omitted, looked up via metagraph from hotkey.",
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
        help="Bittensor subtensor network for UID lookup (default: ARCRATIO_SUBTENSOR_NETWORK or finney).",
    )
    upload_parser.add_argument(
        "--netuid",
        type=int,
        default=None,
        help="Netuid for UID lookup (default: ARCRATIO_NETUID). Required when --miner-uid is omitted.",
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
