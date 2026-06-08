#!/usr/bin/env python3
"""Simulate validator scoring from local orchestrator agent predictions.

Run from repo root:

    python3 scripts/tests/sim_agent_predictions_scoring.py
    python3 scripts/tests/sim_agent_predictions_scoring.py --generate-mock-data

By default, this script reads/writes:

    scripts/tests/agent_predictions.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.validator.orchestrator_api import ScoredPredictionsPage
from src.validator.scoring import DEFAULT_ROLLING_WINDOW_DAYS, score_agent_predictions

DEFAULT_AGENT_PREDICTIONS_FILE = (
    PROJECT_ROOT / "scripts" / "tests" / "agent_predictions.json"
)

# Mock data generation constants.
MOCK_RANDOM_SEED = 41
MOCK_MINER_COUNT = 25
MOCK_MIN_PREDICTIONS_PER_MINER = 5
MOCK_MAX_PREDICTIONS_PER_MINER = 100
MOCK_MAX_AGE_DAYS = 30
MOCK_START_UID = 1


def _load_scored_predictions(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        page_payload = {"items": payload, "nextCursor": None}
    elif isinstance(payload, dict):
        page_payload = payload
    else:
        raise RuntimeError("agent_predictions.json must contain a JSON object or list")
    page = ScoredPredictionsPage.model_validate(page_payload)
    return page.items


def _mock_hotkey(uid: int) -> str:
    return f"mock_hotkey_{uid:04d}"


def _mock_agent_prediction_id(uid: int, idx: int) -> str:
    return f"ap_mock_uid{uid:04d}_{idx:05d}"


def _mock_market_id(uid: int, idx: int) -> str:
    return f"mkt_{uid:04d}_{idx:05d}"


def _generate_mock_predictions(now: datetime) -> list[dict]:
    rng = random.Random(MOCK_RANDOM_SEED)
    rows: list[dict] = []

    max_age_seconds = MOCK_MAX_AGE_DAYS * 24 * 60 * 60
    global_idx = 0

    for uid in range(MOCK_START_UID, MOCK_START_UID + MOCK_MINER_COUNT):
        count = rng.randint(MOCK_MIN_PREDICTIONS_PER_MINER, MOCK_MAX_PREDICTIONS_PER_MINER)
        for miner_idx in range(count):
            global_idx += 1
            age_seconds = rng.randint(0, max_age_seconds)
            scored_at = now - timedelta(seconds=age_seconds)

            p_yes = round(rng.uniform(0.01, 0.99), 4)
            p_no = round(1.0 - p_yes, 4)
            resolved_outcome = "yes" if rng.random() < 0.5 else "no"
            predicted_outcome = "yes" if p_yes >= 0.5 else "no"
            confidence = round(max(p_yes, p_no), 4)

            final_prices = {"yes": 1.0, "no": 0.0}
            if resolved_outcome == "no":
                final_prices = {"yes": 0.0, "no": 1.0}

            rows.append(
                {
                    "agentPredictionId": _mock_agent_prediction_id(uid, global_idx),
                    "minerHotkey": _mock_hotkey(uid),
                    "minerUid": uid,
                    "marketId": _mock_market_id(uid, miner_idx),
                    "sourceMarketId": f"src_{uid:04d}_{miner_idx:05d}",
                    "predictedOutcomeId": predicted_outcome,
                    "confidence": confidence,
                    "outcomeProbabilities": {
                        "yes": p_yes,
                        "no": p_no,
                    },
                    "resolvedOutcomeId": resolved_outcome,
                    "finalOutcomePrices": final_prices,
                    "predictionIsInvalid": False,
                    "predictionInvalidReason": None,
                    "predictionValidation": {
                        "isValid": True,
                        "reasons": [],
                        "rawObserved": {
                            "prediction": predicted_outcome.upper(),
                            "confidence": str(confidence),
                        },
                    },
                    "scoredAt": scored_at.isoformat().replace("+00:00", "Z"),
                    "resolutionStatus": "resolved",
                    "traceSummary": {
                        "strategy": "mock-sim",
                    },
                }
            )
    return rows


def _write_mock_predictions(path: Path) -> int:
    now = datetime.now(timezone.utc)
    rows = _generate_mock_predictions(now)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote {len(rows)} mock rows to {path}")
    print(
        "Config:"
        f" miners={MOCK_MINER_COUNT},"
        f" predictions_per_miner={MOCK_MIN_PREDICTIONS_PER_MINER}-{MOCK_MAX_PREDICTIONS_PER_MINER},"
        f" max_age_days={MOCK_MAX_AGE_DAYS},"
        f" seed={MOCK_RANDOM_SEED}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simulate agent-prediction scoring from local JSON."
    )
    parser.add_argument(
        "--generate-mock-data",
        action="store_true",
        help="Generate mock agent_predictions.json (combined rows list) and exit.",
    )
    parser.add_argument(
        "--input-file",
        default=str(DEFAULT_AGENT_PREDICTIONS_FILE),
        help=f"Path to input JSON file (default: {DEFAULT_AGENT_PREDICTIONS_FILE}).",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if args.generate_mock_data:
        return _write_mock_predictions(input_path)

    if not input_path.exists():
        print(f"missing file: {input_path}")
        print(f"Create {input_path} and rerun, or use --generate-mock-data.")
        return 2

    try:
        rows = _load_scored_predictions(input_path)
    except Exception as exc:  # noqa: BLE001
        print(f"failed to load {input_path}: {exc}")
        return 1

    metagraph_uids = sorted(
        {int(row.minerUid) for row in rows if row.minerUid is not None}
    )
    metagraph = SimpleNamespace(uids=metagraph_uids)
    scores = score_agent_predictions(
        metagraph=metagraph,
        scored_predictions=rows,
        rolling_window_days=DEFAULT_ROLLING_WINDOW_DAYS,
        now=datetime.now(timezone.utc),
    )

    print(f"Loaded {len(rows)} scored prediction rows from {input_path}")
    print(f"Miner UIDs in payload: {len(metagraph_uids)}")
    if not metagraph_uids:
        print("No minerUid values found; no scores generated.")
        return 0

    print()
    print("UID scores:")
    for idx, uid in enumerate(metagraph_uids):
        print(f"  uid={uid}: score={float(scores[idx]):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
