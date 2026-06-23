#!/usr/bin/env python3
"""Simulate validator scoring from local orchestrator agent predictions.

Run from repo root:

    python3 scripts/tests/sim_agent_predictions_scoring.py
    python3 scripts/tests/sim_agent_predictions_scoring.py --generate-mock-data
    python3 scripts/tests/sim_agent_predictions_scoring.py --fetch-live
    python3 scripts/tests/sim_agent_predictions_scoring.py --fetch-live --wallet.name <vali-cold> --wallet.hotkey <vali-hot>

By default, this script reads/writes:

    scripts/tests/agent_predictions.json

A normal scoring run also writes:

    scripts/tests/pareto_scores.png

``--fetch-live`` pulls scored predictions from the orchestrator API (same path
the validator uses during its loop) and saves them locally. A normal run then
scores that snapshot.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import logging
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import AppConfig
from src.gateway.signing import load_hotkey
from src.validator.orchestrator_api import (
    ScoredPredictionItem,
    ScoredPredictionsPage,
    fetch_all_scored_predictions,
)
from src.validator import scoring as arcratio_scoring

DEFAULT_AGENT_PREDICTIONS_FILE = (
    PROJECT_ROOT / "scripts" / "tests" / "agent_predictions.json"
)
DEFAULT_GRAPH_OUTPUT_DIR = PROJECT_ROOT / "scripts" / "tests"
GRAPH_SCORE_THRESHOLD = 0.0
SCORE_TIER_COLORS = ("#FF5733", "#FFC300", "#33FFF5", "#33FF57")
SCORE_TIER_LABELS = ("Bottom quartile", "Lower-mid quartile", "Upper-mid quartile", "Top quartile")

# Mock data generation constants.
MOCK_RANDOM_SEED = 29
MOCK_MINER_COUNT = 100
MOCK_MIN_PREDICTIONS_PER_MINER = 5
MOCK_MAX_PREDICTIONS_PER_MINER = 100
MOCK_MAX_AGE_DAYS = 30
MOCK_START_UID = 1
# Fraction of predictions that reuse an existing market (enables ensemble signal).
MOCK_SHARED_MARKET_RATE = 0.75


logger = logging.getLogger("arcratio.sim_scoring")


def _clamp_prob(value: float, *, low: float = 0.08, high: float = 0.92) -> float:
    return max(low, min(high, value))


def _logit(prob: float) -> float:
    p = _clamp_prob(prob, low=1e-4, high=1.0 - 1e-4)
    return math.log(p / (1.0 - p))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _miner_profile(uid: int) -> dict[str, float]:
    """Stable per-miner traits: skill (edge toward truth), noise, invalid-rate."""
    profile_rng = random.Random(MOCK_RANDOM_SEED * 10_000 + uid)
    return {
        "skill": profile_rng.uniform(-1.1, 1.1),
        "noise": profile_rng.uniform(0.20, 0.65),
        "invalid_rate": profile_rng.uniform(0.0, 0.07),
    }


def _sample_market_event(rng: random.Random) -> tuple[float, float, str]:
    """Return (true_yes_prob, market_yes_price, resolved_outcome_id)."""
    true_yes = _clamp_prob(rng.betavariate(2.4, 2.4))
    # Market is efficient-ish: price tracks latent truth with small noise.
    market_yes = _clamp_prob(true_yes + rng.gauss(0.0, 0.05))
    resolved_outcome = "yes" if rng.random() < true_yes else "no"
    return true_yes, market_yes, resolved_outcome


def _agent_yes_probability(
    *,
    true_yes: float,
    market_yes: float,
    skill: float,
    noise: float,
    rng: random.Random,
) -> float:
    """Skilled miners lean toward truth; unskilled miners stay near market + noise."""
    logit_market = _logit(market_yes)
    logit_truth = _logit(true_yes)
    skill_weight = max(0.0, min(1.0, abs(skill) / 1.1))
    direction = 1.0 if skill >= 0.0 else -1.0
    target = logit_market + direction * skill_weight * (logit_truth - logit_market)
    noisy = target + rng.gauss(0.0, noise)
    return round(_clamp_prob(_sigmoid(noisy)), 4)


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
    market_events: dict[str, tuple[float, float, str]] = {}
    market_pool: list[str] = []

    for uid in range(MOCK_START_UID, MOCK_START_UID + MOCK_MINER_COUNT):
        profile = _miner_profile(uid)
        count = rng.randint(MOCK_MIN_PREDICTIONS_PER_MINER, MOCK_MAX_PREDICTIONS_PER_MINER)
        for miner_idx in range(count):
            global_idx += 1
            age_seconds = rng.randint(0, max_age_seconds)
            scored_at = now - timedelta(seconds=age_seconds)
            predicted_at = scored_at - timedelta(minutes=rng.randint(1, 180))

            reuse_market = market_pool and rng.random() < MOCK_SHARED_MARKET_RATE
            if reuse_market:
                market_id = rng.choice(market_pool)
                true_yes, market_yes, resolved_outcome = market_events[market_id]
            else:
                market_id = _mock_market_id(uid, miner_idx)
                true_yes, market_yes, resolved_outcome = _sample_market_event(rng)
                market_events[market_id] = (true_yes, market_yes, resolved_outcome)
                market_pool.append(market_id)

            market_yes = round(market_yes, 4)
            market_no = round(1.0 - market_yes, 4)
            p_yes = _agent_yes_probability(
                true_yes=true_yes,
                market_yes=market_yes,
                skill=profile["skill"],
                noise=profile["noise"],
                rng=rng,
            )
            p_no = round(1.0 - p_yes, 4)
            predicted_outcome = "yes" if p_yes >= 0.5 else "no"
            confidence = round(max(p_yes, p_no), 4)

            is_invalid = rng.random() < profile["invalid_rate"]
            invalid_reason = "confidence_missing" if is_invalid else None
            if is_invalid:
                confidence = 0.0

            final_prices = {"yes": 1.0, "no": 0.0}
            if resolved_outcome == "no":
                final_prices = {"yes": 0.0, "no": 1.0}

            rows.append(
                {
                    "agentPredictionId": _mock_agent_prediction_id(uid, global_idx),
                    "minerHotkey": _mock_hotkey(uid),
                    "minerUid": uid,
                    "marketId": market_id,
                    "sourceMarketId": f"src_{market_id}",
                    "predictedOutcomeId": predicted_outcome,
                    "predictedAt": predicted_at.isoformat().replace("+00:00", "Z"),
                    "confidence": confidence,
                    "outcomePricesAtPrediction": {
                        "yes": market_yes,
                        "no": market_no,
                    },
                    "outcomeProbabilities": {
                        "yes": p_yes,
                        "no": p_no,
                    },
                    "resolvedOutcomeId": resolved_outcome,
                    "finalOutcomePrices": final_prices,
                    "predictionIsInvalid": is_invalid,
                    "predictionInvalidReason": invalid_reason,
                    "predictionValidation": {
                        "isValid": not is_invalid,
                        "reasons": [invalid_reason] if invalid_reason else [],
                        "rawObserved": {
                            "prediction": predicted_outcome.upper(),
                            "confidence": str(confidence),
                        },
                    },
                    "scoredAt": scored_at.isoformat().replace("+00:00", "Z"),
                    "resolutionStatus": "resolved",
                    "traceSummary": {
                        "strategy": "mock-sim-latent-skill",
                    },
                }
            )
    return rows


def _serialize_scored_predictions(rows: list[ScoredPredictionItem]) -> list[dict]:
    return [row.model_dump(mode="json") for row in rows]


def _write_scored_predictions(path: Path, rows: list[ScoredPredictionItem]) -> int:
    payload = _serialize_scored_predictions(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return len(payload)


def _fetch_live_predictions(
    *,
    base_url: str,
    days: int,
    timeout_seconds: float,
    wallet_name: str | None,
    wallet_hotkey: str | None,
    wallet_path: Path | None,
) -> list[ScoredPredictionItem]:
    cfg = AppConfig.load_default()
    if wallet_name:
        cfg.bittensor.wallet_name = wallet_name
    if wallet_hotkey:
        cfg.bittensor.wallet_hotkey = wallet_hotkey
    if wallet_path is not None:
        cfg.bittensor.wallet_path = wallet_path

    loaded_hotkey = load_hotkey(cfg.bittensor)
    if loaded_hotkey is None:
        raise RuntimeError(
            "validator hotkey signing is unavailable; configure wallet settings "
            "or enable signing_required."
        )

    return fetch_all_scored_predictions(
        base_url=base_url,
        loaded_hotkey=loaded_hotkey,
        days=days,
        include_trace_summary=False,
        timeout_seconds=timeout_seconds,
    )


def _normalise_weights(vec: np.ndarray) -> np.ndarray:
    """Match validator ``combine_weights`` normalization for arcratio-only scores."""
    clipped = np.clip(np.asarray(vec, dtype=float), 0.0, None)
    total = float(clipped.sum())
    if total <= 0.0:
        return np.zeros_like(clipped)
    return clipped / total


def _score_tier(index: int, count: int) -> int:
    if count <= 1:
        return len(SCORE_TIER_COLORS) - 1
    quartile = int((index / (count - 1)) * len(SCORE_TIER_COLORS))
    return min(quartile, len(SCORE_TIER_COLORS) - 1)


def graph_results(
    uids: list[int],
    pre_pareto_scores: np.ndarray,
    post_pareto_scores: np.ndarray,
    *,
    output_dir: Path = DEFAULT_GRAPH_OUTPUT_DIR,
    score_threshold: float = GRAPH_SCORE_THRESHOLD,
) -> Path:
    """Graph pre-Pareto composite and normalized metagraph weights.

    Miners are sorted by post-Pareto score (ascending, matching the sportstensor
    sim). Zero/low scores below ``score_threshold`` are omitted from the plot.
    """
    uid_array = np.asarray(uids, dtype=int)
    pre_scores = np.asarray(pre_pareto_scores, dtype=float)
    post_scores = np.asarray(post_pareto_scores, dtype=float)
    weights = _normalise_weights(post_scores)

    sorted_indices = np.argsort(post_scores)
    sorted_uids = uid_array[sorted_indices]
    sorted_pre = pre_scores[sorted_indices]
    sorted_post = post_scores[sorted_indices]
    sorted_weights = weights[sorted_indices]

    keep = sorted_post > score_threshold
    sorted_uids = sorted_uids[keep]
    sorted_pre = sorted_pre[keep]
    sorted_post = sorted_post[keep]
    sorted_weights = sorted_weights[keep]

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    tier_counts = [0] * len(SCORE_TIER_COLORS)
    for idx in range(len(sorted_uids)):
        tier_counts[_score_tier(idx, len(sorted_uids))] += 1

    for ax, y_values, title, ylabel in (
        (axes[0], sorted_pre, "Pre-Pareto Composite Scores", "Composite score"),
        (axes[1], sorted_weights, "Normalized Metagraph Weights", "Weight"),
    ):
        for tier, color in enumerate(SCORE_TIER_COLORS):
            tier_indices = [
                i for i in range(len(sorted_uids)) if _score_tier(i, len(sorted_uids)) == tier
            ]
            if not tier_indices:
                continue
            ax.scatter(
                np.array(tier_indices),
                np.array(y_values)[tier_indices],
                label=f"{SCORE_TIER_LABELS[tier]} ({tier_counts[tier]} UIDs)",
                color=color,
                s=12,
                alpha=0.85,
            )
        ax.set_ylabel(ylabel, fontsize=12, color="white")
        ax.set_title(title, fontsize=14, color="white")
        ax.grid(True, linestyle="--", alpha=0.3, color="gray")
        ax.legend(title="Score rank", fontsize=9, facecolor="gray", edgecolor="white")

    axes[1].set_xlabel("Miners (sorted by post-Pareto score)", fontsize=12, color="white")
    fig.suptitle(
        "Arcratio scoring simulation: pre-Pareto composite vs metagraph weights",
        fontsize=15,
        color="white",
        y=0.98,
    )
    fig.tight_layout()

    output_path = output_dir / "pareto_scores.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


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
        "--fetch-live",
        action="store_true",
        help=(
            "Fetch scored predictions from orchestrator API and save to --input-file, "
            "then exit."
        ),
    )
    parser.add_argument(
        "--input-file",
        default=str(DEFAULT_AGENT_PREDICTIONS_FILE),
        help=f"Path to input/output JSON file (default: {DEFAULT_AGENT_PREDICTIONS_FILE}).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override orchestrator base URL. Defaults to validator.orchestrator_api_url.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Rolling window days for live fetch. Defaults to loop.rolling_window_days.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=15.0,
        help="HTTP timeout for live orchestrator fetch.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python log level for sim run output.",
    )
    parser.add_argument(
        "--wallet.name",
        dest="wallet_name",
        default=None,
        help="Override bittensor wallet name for live fetch signing.",
    )
    parser.add_argument(
        "--wallet.hotkey",
        dest="wallet_hotkey",
        default=None,
        help="Override bittensor wallet hotkey for live fetch signing.",
    )
    parser.add_argument(
        "--wallet.path",
        dest="wallet_path",
        type=Path,
        default=None,
        help="Override bittensor wallet path for live fetch signing.",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Skip generating the scoring visualization PNG.",
    )
    parser.add_argument(
        "--graph-output-dir",
        type=Path,
        default=DEFAULT_GRAPH_OUTPUT_DIR,
        help=f"Directory for pareto_scores.png (default: {DEFAULT_GRAPH_OUTPUT_DIR}).",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    input_path = Path(args.input_file)
    if args.generate_mock_data and args.fetch_live:
        print("choose only one of --generate-mock-data or --fetch-live")
        return 2

    if args.fetch_live:
        cfg = AppConfig.load_default()
        base_url = (args.base_url or cfg.validator.orchestrator_api_url).strip()
        if not base_url:
            print("missing orchestrator API URL in validator config")
            return 2
        days = args.days if args.days is not None else cfg.loop.rolling_window_days
        try:
            rows = _fetch_live_predictions(
                base_url=base_url,
                days=days,
                timeout_seconds=args.timeout_seconds,
                wallet_name=args.wallet_name,
                wallet_hotkey=args.wallet_hotkey,
                wallet_path=args.wallet_path,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"failed to fetch scored predictions from {base_url}: {exc}")
            return 1

        count = _write_scored_predictions(input_path, rows)
        print(f"Fetched {count} scored prediction rows from {base_url}")
        print(f"Wrote snapshot to {input_path}")
        print(f"Config: days={days}, include_trace_summary=False")
        return 0

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
    now_utc = datetime.now(timezone.utc)
    scoring_result = arcratio_scoring.score_agent_predictions(
        metagraph=metagraph,
        scored_predictions=rows,
        rolling_window_days=arcratio_scoring.DEFAULT_ROLLING_WINDOW_DAYS,
        now=now_utc,
        return_pre_pareto=True,
    )
    scores, pre_pareto_scores = scoring_result
    logger.info(
        "scoring simulation summary: rows=%d miners=%d (non-zero scores=%d)",
        len(rows),
        len(metagraph_uids),
        int((scores > 0).sum()),
    )

    print(f"Loaded {len(rows)} scored prediction rows from {input_path}")
    print(f"Miner UIDs in payload: {len(metagraph_uids)}")
    if not metagraph_uids:
        print("No minerUid values found; no scores generated.")
        return 0

    print("Per-miner diagnostics table is emitted by arcratio.scoring INFO logs.")
    if not args.no_graph:
        graph_path = graph_results(
            metagraph_uids,
            pre_pareto_scores,
            scores,
            output_dir=args.graph_output_dir,
        )
        print(f"Wrote scoring visualization to {graph_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
