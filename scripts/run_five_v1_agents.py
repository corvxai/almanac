#!/usr/bin/env python3
"""Run the five v1 capability-ladder agents on a single event and log results.

The five agents climb a capability ladder on the SAME event so their traces
can be compared apples-to-apples:

  1  LLM only            (no search)
  2  basic search  + basic LLM
  3  amazing search + basic LLM
  4  amazing search + good LLM
  5  orchestrated        (iterative belief path)

Wiring (gateway/orchestrator/providers) mirrors scripts/run_forecast.py:
remote providers against the local gateway, the dev miner-hotkey header
injected on each provider, in_process sandbox, signing disabled.

Usage:
  python scripts/run_five_v1_agents.py --agent 1 --event fed-rate-cut-2026
  python scripts/run_five_v1_agents.py --agent all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import AppConfig
from src.core.events import Event
from src.gateway.client import build_remote_providers
from src.storage.json_store import build_trace_store
from src.validator.orchestrator import Orchestrator

from src.agent.examples.v1_agent1_llm_only import V1Agent1LLMOnly
from src.agent.examples.v1_agent2_basic import V1Agent2Basic
from src.agent.examples.v1_agent3_greatsearch_basicllm import V1Agent3GreatSearchBasicLLM
from src.agent.examples.v1_agent4_greatsearch_goodllm import V1Agent4GreatSearchGoodLLM
from src.agent.examples.v1_agent5_orchestrated import V1Agent5Orchestrated
from src.agent.examples.v1_agent6_market_reader import V1Agent6MarketReader

DEV_MINER_HOTKEY = "5Exgnozdp8BfVXq6cuzmE2B8CXpQAqr7q7St2zE9RNcdzdyH"
DEFAULT_GATEWAY_URL = "http://localhost:8077"

AGENTS = {
    1: ("Agent 1 — LLM only (no search)", V1Agent1LLMOnly),
    2: ("Agent 2 — basic search + basic LLM", V1Agent2Basic),
    3: ("Agent 3 — amazing search + basic LLM", V1Agent3GreatSearchBasicLLM),
    4: ("Agent 4 — amazing search + good LLM", V1Agent4GreatSearchGoodLLM),
    5: ("Agent 5 — orchestrated belief path", V1Agent5Orchestrated),
    6: ("Agent 6 — market reader (reads the market-implied probability)", V1Agent6MarketReader),
}


def load_event(data_dir: Path, stem: str) -> Event:
    path = data_dir / "events" / f"{stem}.json"
    if not path.exists():
        # allow substring match
        matches = sorted((data_dir / "events").glob("*.json"))
        matches = [m for m in matches if stem.lower() in m.stem.lower()]
        if not matches:
            raise SystemExit(f"No event matching '{stem}' in {data_dir / 'events'}")
        path = matches[0]
    return Event.model_validate(json.loads(path.read_text()))


def format_block(agent_num: int, agent_name: str, digest) -> str:
    lines: list[str] = []
    ec = digest.execution_context
    pred = digest.prediction_output
    integrity = digest.trace_integrity

    lines.append("=" * 72)
    lines.append(f"  {agent_name}")
    lines.append("=" * 72)
    lines.append(f"  Run at:      {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"  Event:       {digest.event_snapshot.event_title}")
    lines.append(f"  Agent:       {ec.agent_id} v{ec.agent_version}")
    lines.append(f"  Execution:   {ec.execution_id}")
    lines.append(f"  Duration:    {ec.execution_duration_ms}ms")
    lines.append(f"  Sandbox:     {ec.sandbox_environment.value}")
    lines.append("")

    lines.append(f"  Provider Calls: {len(digest.provider_calls)}")
    for pc in digest.provider_calls:
        ev_count = len(pc.extracted_evidence)
        model = pc.model or "-"
        lines.append(
            f"    [{pc.call_index}] {pc.provider_id}.{pc.call_type} "
            f"model={model} evidence={ev_count} latency={pc.latency_ms}ms "
            f"cost={pc.cost_units}"
        )
    lines.append("")

    lines.append(f"  Reasoning Chain: {len(digest.reasoning_chain)} steps")
    for step in digest.reasoning_chain:
        prob = (f" p={step.intermediate_probability:.3f}"
                if step.intermediate_probability is not None else "")
        model = f" [{step.inference_model_used}]" if step.inference_model_used else ""
        lines.append(f"    [{step.step_index}] {step.step_type.value}{prob}{model}")
        snippet = step.reasoning_text.replace("\n", " ")[:120]
        lines.append(f"        {snippet}")
    lines.append("")

    conf = f"{pred.confidence:.4f}" if pred.confidence is not None else "n/a"
    lines.append("  ┌─ PREDICTION ─────────────────────────────────")
    lines.append(f"  │  Probability:  {pred.final_probability:.4f}")
    lines.append(f"  │  Confidence:   {conf}")
    lines.append("  └──────────────────────────────────────────────")
    lines.append(f"  Reasoning:   {pred.reasoning_summary[:300]}")
    lines.append("")
    lines.append(f"  Total provider cost: {integrity.total_provider_cost}")
    lines.append(f"  Trace hash: {integrity.trace_hash[:16]}...  valid={digest.verify_integrity()}")
    lines.append("=" * 72)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the five v1 ladder agents")
    parser.add_argument("--agent", "-a", default="all",
                        help="1-5 or 'all' (default: all)")
    parser.add_argument("--event", "-e", default="fed-rate-cut-2026",
                        help="Event filename stem (default: fed-rate-cut-2026)")
    parser.add_argument("--data-dir", "-d", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--gateway-url", "-g", default=None)
    args = parser.parse_args()

    if args.agent == "all":
        selected = [1, 2, 3, 4, 5, 6]
    else:
        try:
            n = int(args.agent)
        except ValueError:
            raise SystemExit("--agent must be 1-6 or 'all'")
        if n not in AGENTS:
            raise SystemExit("--agent must be 1-6 or 'all'")
        selected = [n]

    # The live POC gateway runs at 8077. The repo .env points
    # GATEWAY_SERVICE_URL at 4000 (a different/stale endpoint), so we default
    # to 8077 explicitly and only honour an override passed on the CLI.
    gateway_url = args.gateway_url or DEFAULT_GATEWAY_URL

    config = AppConfig.load_default()
    config.storage.data_dir = args.data_dir
    config.validator.sandbox_type = "in_process"
    config.bittensor.signing_required = False

    store = build_trace_store(config.storage)

    print(f"Connecting to gateway at {gateway_url} ...")
    try:
        providers = build_remote_providers(gateway_url)
    except Exception as exc:
        raise SystemExit(f"ERROR: cannot reach gateway at {gateway_url}: {exc}")

    miner_hk = os.environ.get("DEV_MINER_HOTKEY", DEV_MINER_HOTKEY)
    for _p in providers:
        _p._headers = {**getattr(_p, "_headers", {}), "x-miner-hotkey": miner_hk}
    print(f"  Gateway providers: {[p.provider_id for p in providers]}")

    orchestrator = Orchestrator(
        config=config,
        store=store,
        providers=providers,
        local_proxy_state=None,
    )

    event = load_event(args.data_dir, args.event)

    log_dir = PROJECT_ROOT / "poc" / "agent-logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning agent(s) {selected} on event '{event.title}'")
    print(f"Validator: {orchestrator.validator_id}\n")

    for n in selected:
        agent_name, agent_cls = AGENTS[n]
        print(f"--- running {agent_name} ---")
        digest = orchestrator.run_agent(event, agent_cls())
        block = format_block(n, agent_name, digest)
        print(block)
        log_path = log_dir / f"agent{n}.log"
        with log_path.open("a") as fh:
            fh.write(block + "\n\n")
        print(f"  (appended to {log_path})\n")

    print(f"Done. Traces saved under {config.storage.data_dir / 'traces'}")


if __name__ == "__main__":
    main()
