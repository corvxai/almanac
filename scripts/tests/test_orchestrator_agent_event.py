#!/usr/bin/env python3
"""Manual orchestrator single-job harness.

Run from repo root:

    python3 scripts/tests/test_orchestrator_agent_event.py \\
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import AppConfig
from src.gateway.client import build_remote_providers
from src.gateway.constants import gateway_service_url
from src.gateway.signing import load_hotkey
from src.storage.json_store import JsonTraceStore
from src.validator.assignment_pipeline import process_single_assignment
from src.validator.orchestrator import Orchestrator
from src.validator.orchestrator_api import (
    PREDICTION_ENDPOINT,
    fetch_agent_event_assignment,
)
from src.validator.validator import start_local_proxy


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
    parser.add_argument(
        "--submit-prediction",
        action="store_true",
        help="Execute the assignment and prepare prediction submit payload.",
    )
    parser.add_argument(
        "--i-understand-this-submits",
        action="store_true",
        help="Required with --submit-prediction to actually POST to the API.",
    )
    parser.add_argument(
        "--execute-agent",
        action="store_true",
        help="Execute assignment agent code in sandbox without submitting.",
    )
    parser.add_argument(
        "--gateway-url",
        default=None,
        help="Gateway service URL (defaults to configured gateway URL).",
    )
    parser.add_argument(
        "--sandbox",
        choices=["docker_runc", "docker_gvisor"],
        default=None,
        help="Override sandbox type for this run (docker-only for orchestrator assignments).",
    )
    parser.add_argument(
        "--unsafe-no-signing",
        action="store_true",
        help="Disable wallet signing for local proxy requests (dev only).",
    )
    args = parser.parse_args()

    cfg = AppConfig.load_default()
    if args.wallet_name:
        cfg.bittensor.wallet_name = args.wallet_name
    if args.wallet_hotkey:
        cfg.bittensor.wallet_hotkey = args.wallet_hotkey
    if args.wallet_path is not None:
        cfg.bittensor.wallet_path = args.wallet_path
    if args.sandbox is not None:
        cfg.validator.sandbox_type = args.sandbox
    if args.unsafe_no_signing:
        cfg.bittensor.signing_required = False

    base_url = (args.base_url or cfg.validator.orchestrator_api_url).strip()
    if not base_url:
        print("missing orchestrator API URL in validator config")
        return 2
    cfg.validator.orchestrator_api_url = base_url

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

    should_execute = args.execute_agent or args.submit_prediction
    if not should_execute:
        return 0

    gateway_url = args.gateway_url or gateway_service_url()
    store = JsonTraceStore(data_dir=cfg.storage.data_dir)
    try:
        providers = build_remote_providers(gateway_url)
    except Exception as exc:  # noqa: BLE001
        print(f"failed to connect to gateway ({gateway_url}): {exc}")
        return 1

    local_proxy_state = None
    if cfg.validator.sandbox_type.startswith("docker"):
        try:
            local_proxy_state = start_local_proxy(cfg)
        except Exception as exc:  # noqa: BLE001
            print(f"failed to start local proxy: {exc}")
            return 1

    orchestrator = Orchestrator(
        config=cfg,
        store=store,
        providers=providers,
        local_proxy_state=local_proxy_state,
    )
    submit_live = args.submit_prediction and args.i_understand_this_submits
    try:
        result = process_single_assignment(
            assignment=response.assignment,
            config=cfg,
            orchestrator=orchestrator,
            loaded_hotkey=loaded_hotkey if submit_live else None,
            submit_prediction=submit_live,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"assignment processing failed: {exc}")
        return 1

    print()
    print(
        "Execution completed:"
        f" event_id={result.event.event_id}"
        f" execution_id={result.digest.execution_context.execution_id}"
    )
    if not args.submit_prediction:
        return 0

    submit_url = f"{base_url.rstrip('/')}{PREDICTION_ENDPOINT}"
    print()
    print(f"Submit endpoint: {submit_url}")
    print("Prediction payload preview:")
    print(json.dumps(result.submit_payload, indent=2, sort_keys=True))

    if not args.i_understand_this_submits:
        print()
        print(
            "Dry run only. Re-run with both --submit-prediction and "
            "--i-understand-this-submits to actually POST."
        )
        return 0

    print()
    print("Submit response:")
    if result.submit_ack is None:
        print("missing submit acknowledgement")
        return 1
    print(json.dumps(result.submit_ack.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
