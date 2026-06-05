#!/usr/bin/env python3
"""Manual smoke test for GET /v1/validators/agent-and-event.

Run from repo root:

    python3 scripts/test_orchestrator_agent_event.py \\
      --wallet.name <wallet-name> \\
      --wallet.hotkey <wallet-hotkey> \\
      --wallet.path <wallet-path>

Optional:
    --base-url <orchestrator-url>
    --timeout-seconds <seconds>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import AppConfig
from src.gateway.signing import load_hotkey
from src.validator.orchestrator_api import fetch_agent_event_assignment


def _code_preview(code: str, *, preview_chars: int = 50) -> str:
    compact = code.replace("\n", "\\n")
    snippet = compact[:preview_chars]
    if len(compact) > preview_chars:
        snippet += "..."
    size_kb = len(code.encode("utf-8")) / 1024.0
    lines = code.count("\n") + (1 if code else 0)
    return f"{snippet} ({size_kb:.1f} KB, {lines} lines)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch one validator agent/event assignment.")
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override orchestrator base URL. Defaults to validator.orchestrator_api_url.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=15.0,
        help="HTTP timeout for this request.",
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
    args = parser.parse_args()

    cfg = AppConfig.load_default()
    if args.wallet_name:
        cfg.bittensor.wallet_name = args.wallet_name
    if args.wallet_hotkey:
        cfg.bittensor.wallet_hotkey = args.wallet_hotkey
    if args.wallet_path is not None:
        cfg.bittensor.wallet_path = args.wallet_path

    base_url = (args.base_url or cfg.validator.orchestrator_api_url).strip()
    if not base_url:
        print("missing orchestrator API URL in validator config")
        return 2

    loaded_hotkey = load_hotkey(cfg.bittensor)
    if loaded_hotkey is None:
        print("validator hotkey signing is unavailable (signing_required is false).")
        return 2

    try:
        response = fetch_agent_event_assignment(
            base_url=base_url,
            loaded_hotkey=loaded_hotkey,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"request failed: {exc}")
        return 1

    if response.assignment is None:
        print(f"No assignment: {response.reason or 'none_available'}")
        return 0

    print(f"Got assignment: {response.assignment.agentPredictionId}")
    assignment_payload = response.assignment.model_dump(mode="json")
    agent_payload = assignment_payload.get("agent")
    if isinstance(agent_payload, dict):
        code = agent_payload.get("code")
        if isinstance(code, str):
            agent_payload["code"] = _code_preview(code)
    print(json.dumps(assignment_payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
