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
from datetime import datetime, timezone
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import AppConfig
from src.gateway.signing import load_hotkey
from src.validator.orchestrator_api import (
    PREDICTION_ENDPOINT,
    OrchestratorAssignment,
    fetch_agent_event_assignment,
    submit_validator_prediction,
)


def _code_preview(code: str, *, preview_chars: int = 50) -> str:
    compact = code.replace("\n", "\\n")
    snippet = compact[:preview_chars]
    if len(compact) > preview_chars:
        snippet += "..."
    size_kb = len(code.encode("utf-8")) / 1024.0
    lines = code.count("\n") + (1 if code else 0)
    return f"{snippet} ({size_kb:.1f} KB, {lines} lines)"


def _build_submit_payload(assignment: OrchestratorAssignment) -> dict:
    yes_id, no_id = _resolve_binary_outcomes(assignment)
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "agentPredictionId": assignment.agentPredictionId,
        "prediction": {
            "schemaVersion": "1.0",
            "submittedAt": now_iso,
            "predictionType": "binary",
            "predictedOutcomeId": yes_id,
            "confidence": 0.5,
            "outcomeProbabilities": {
                yes_id: 0.5,
                no_id: 0.5,
            },
            "outcomePricesAtPrediction": assignment.event.currentOutcomePrices,
            "marketSnapshot": {
                "source": assignment.event.source,
                "sourceMarketId": assignment.event.sourceMarketId or assignment.event.marketId,
                "capturedAt": now_iso,
            },
            "executionMetadata": {
                "predictionIsInvalid": False,
                "predictionInvalidReason": None,
                "predictionValidation": {
                    "isValid": True,
                    "reasons": [],
                    "rawObserved": {
                        "prediction": "0.5",
                        "confidence": "0.5",
                    },
                },
                "model": "manual-smoke-test",
                "latencyMs": 0,
            },
        },
        "reasoningTrace": {
            "schemaVersion": "1.0",
            "trace": {
                "steps": [
                    "Fetched assignment via validator auth",
                    "Built manual smoke-test prediction payload",
                    "Submitted payload to orchestrator API",
                ]
            },
            "traceSummary": {
                "strategy": "manual-smoke-test",
            },
            "modelMetadata": {
                "provider": "manual",
                "model": "manual-smoke-test",
            },
        },
    }


def _resolve_binary_outcomes(assignment: OrchestratorAssignment) -> tuple[str, str]:
    yes_id: str | None = None
    no_id: str | None = None
    for outcome in assignment.event.outcomes:
        if not isinstance(outcome, dict):
            continue
        oid = outcome.get("outcomeId")
        name = str(outcome.get("name", "")).strip().lower()
        if not isinstance(oid, str):
            continue
        if name == "yes":
            yes_id = oid
        elif name == "no":
            no_id = oid
    if yes_id and no_id:
        return yes_id, no_id

    if len(assignment.event.outcomes) < 2:
        raise RuntimeError("assignment has fewer than two outcomes")
    first = assignment.event.outcomes[0] if isinstance(assignment.event.outcomes[0], dict) else {}
    second = assignment.event.outcomes[1] if isinstance(assignment.event.outcomes[1], dict) else {}
    first_id = first.get("outcomeId")
    second_id = second.get("outcomeId")
    if not isinstance(first_id, str) or not isinstance(second_id, str):
        raise RuntimeError("assignment outcomes missing outcomeId")
    return first_id, second_id


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
        help="Prepare and optionally submit a prediction payload for the fetched assignment.",
    )
    parser.add_argument(
        "--i-understand-this-submits",
        action="store_true",
        help="Required with --submit-prediction to actually POST to the API.",
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

    if not args.submit_prediction:
        return 0

    submit_url = f"{base_url.rstrip('/')}{PREDICTION_ENDPOINT}"
    submit_payload = _build_submit_payload(response.assignment)
    print()
    print(f"Submit endpoint: {submit_url}")
    print("Prediction payload preview:")
    print(json.dumps(submit_payload, indent=2, sort_keys=True))

    if not args.i_understand_this_submits:
        print()
        print(
            "Dry run only. Re-run with both --submit-prediction and "
            "--i-understand-this-submits to actually POST."
        )
        return 0

    try:
        ack = submit_validator_prediction(
            base_url=base_url,
            loaded_hotkey=loaded_hotkey,
            payload=submit_payload,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"prediction submit failed: {exc}")
        return 1

    print()
    print("Submit response:")
    print(json.dumps(ack.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
