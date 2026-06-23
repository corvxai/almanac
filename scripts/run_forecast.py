#!/usr/bin/env python3
"""CLI entry point — pick an event, run an agent, store and display traces.

Connects to the Provider Gateway HTTP service for all external data access.
The gateway must be running first (see scripts/run_gateway.py).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import AppConfig
from src.core.events import Event
from src.gateway.client import build_remote_providers
from src.gateway.constants import DEFAULT_GATEWAY_SERVICE_URL, gateway_service_url
from src.agent.examples.simple_agent import SimpleAgent
from src.agent.examples.openrouter_agent import OpenRouterAgent
from src.agent.examples.claude_agent import ClaudeAgent
from src.storage.json_store import build_trace_store
from src.validator.orchestrator import Orchestrator
from src.validator.validator import start_local_proxy as _start_local_proxy


@dataclass(frozen=True)
class LoadedEvent:
    filename_stem: str
    event: Event


def load_events(data_dir: Path) -> list[LoadedEvent]:
    events_dir = data_dir / "events"
    events: list[LoadedEvent] = []
    for path in sorted(events_dir.glob("*.json")):
        raw = json.loads(path.read_text())
        events.append(LoadedEvent(filename_stem=path.stem, event=Event.model_validate(raw)))
    return events


def print_summary(digest, idx: int) -> None:
    ec = digest.execution_context
    pred = digest.prediction_output
    integrity = digest.trace_integrity
    event = digest.event_snapshot

    print(f"\n{'='*70}")
    print(f"  TRACE #{idx + 1}")
    print(f"{'='*70}")
    print(f"  Event:       {event.event_title}")
    print(f"  Category:    {event.event_category.value}")
    print(f"  Agent:       {ec.agent_id} v{ec.agent_version}")
    print(f"  Execution:   {ec.execution_id}")
    print(f"  Duration:    {ec.execution_duration_ms}ms")
    print(f"  Sandbox:     {ec.sandbox_environment.value}")
    print()

    print(f"  Provider Calls: {len(digest.provider_calls)}")
    for pc in digest.provider_calls:
        ev_count = len(pc.extracted_evidence)
        model_suffix = f" model={pc.model}" if pc.model else ""
        print(f"    [{pc.call_index}] {pc.provider_id}.{pc.call_type} "
              f"→ {ev_count} evidence items, {pc.latency_ms}ms{model_suffix}")
    print()

    print(f"  Reasoning Chain: {len(digest.reasoning_chain)} steps")
    for step in digest.reasoning_chain:
        prob = f" p={step.intermediate_probability:.3f}" if step.intermediate_probability is not None else ""
        print(f"    [{step.step_index}] {step.step_type.value}{prob}")
        print(f"        {step.reasoning_text[:100]}...")
    print()

    print(f"  ┌─ PREDICTION ─────────────────────────────────")
    print(f"  │  Probability:    {pred.final_probability:.4f}")
    confidence_text = f"{pred.confidence:.4f}" if pred.confidence is not None else "n/a"
    print(f"  │  Confidence:     {confidence_text}")
    if pred.confidence_interval:
        lo, hi = pred.confidence_interval
        print(f"  │  95% CI:         [{lo:.3f}, {hi:.3f}]")
    print(f"  │  Contrarian:     {pred.contrarian_flag}")
    if pred.key_drivers:
        print(f"  │  Key drivers:")
        for d in pred.key_drivers:
            print(f"  │    • {d}")
    if pred.key_uncertainties:
        print(f"  │  Key uncertainties:")
        for u in pred.key_uncertainties:
            print(f"  │    • {u}")
    print(f"  └────────────────────────────────────────────────")
    print()

    print(f"  Integrity:")
    print(f"    Schema version:    {integrity.trace_schema_version}")
    print(f"    Trace hash:        {integrity.trace_hash[:16]}...")
    print(f"    Hash valid:        {digest.verify_integrity()}")
    print(f"    Evidence items:    {integrity.total_evidence_items}")
    print(f"    Provider cost:     {integrity.total_provider_cost}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run forecasting agents on events")
    parser.add_argument(
        "--event", "-e",
        help="Event JSON filename substring. If omitted, picks one at random.",
    )
    parser.add_argument(
        "--agent", "-a",
        choices=["simple", "openrouter", "anthropic", "all"],
        default="openrouter",
        help="Which agent to run (default: openrouter)",
    )
    parser.add_argument(
        "--all-events",
        action="store_true",
        help="Run on all available events instead of picking one.",
    )
    parser.add_argument(
        "--data-dir", "-d",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Path to the data directory",
    )
    parser.add_argument(
        "--gateway-url", "-g",
        default=None,
        help=(
            "Gateway service URL (overrides GATEWAY_SERVICE_URL env; "
            f"default env or {DEFAULT_GATEWAY_SERVICE_URL})"
        ),
    )
    parser.add_argument(
        "--sandbox",
        choices=["in_process", "docker_runc", "docker_gvisor"],
        default=None,
        help=(
            "Override the sandbox type. Default is taken from config "
            "(currently 'docker_runc'). Use 'in_process' for dev/tests where "
            "the agent runs directly in this Python process."
        ),
    )
    parser.add_argument(
        "--unsafe-no-signing",
        action="store_true",
        help=(
            "Disable Bittensor request signing on the validator-local proxy. "
            "Required for contributors without a Bittensor wallet on disk. "
            "DO NOT use this in production."
        ),
    )
    args = parser.parse_args()

    gateway_url = args.gateway_url or gateway_service_url()

    config = AppConfig.load_default()
    config.storage.data_dir = args.data_dir
    if args.sandbox is not None:
        config.validator.sandbox_type = args.sandbox
    if args.unsafe_no_signing:
        config.bittensor.signing_required = False
        print(
            "  WARNING: --unsafe-no-signing is set. The validator will NOT "
            "sign outbound gateway requests with a Bittensor hotkey. This is "
            "intended for local development only."
        )

    store = build_trace_store(config.storage)

    print(f"Connecting to gateway at {gateway_url} ...")
    try:
        providers = build_remote_providers(gateway_url)
    except Exception as exc:
        print(f"ERROR: Cannot reach gateway at {gateway_url}: {exc}")
        print("Start the gateway first:  python scripts/run_gateway.py")
        sys.exit(1)

    print(f"  Gateway providers: {[p.provider_id for p in providers]}")

    local_proxy_state = None
    if config.validator.sandbox_type.startswith("docker"):
        local_proxy_state = _start_local_proxy(config)
        print(f"  Local proxy:       UDS at {config.validator.sandbox_socket_dir / 'proxy.sock'}")
        kp = local_proxy_state.loaded_keypair
        print(f"  Signing:           {'ENABLED hotkey=' + kp.hotkey_ss58 if kp else 'DISABLED (dev)'}")

    orchestrator = Orchestrator(
        config=config,
        store=store,
        providers=providers,
        local_proxy_state=local_proxy_state,
    )

    agent_map = {
        "simple": SimpleAgent,
        "openrouter": OpenRouterAgent,
        "anthropic": ClaudeAgent,
    }
    if args.agent == "all":
        agents = [cls() for cls in agent_map.values()]
    else:
        agents = [agent_map[args.agent]()]

    all_events = load_events(args.data_dir)
    if not all_events:
        print("No events found in", args.data_dir / "events")
        sys.exit(1)

    if args.event:
        needle = args.event.lower()
        all_events = [e for e in all_events if needle in e.filename_stem.lower()]
        if not all_events:
            print(f"No events matching '{args.event}'")
            sys.exit(1)
    elif not args.all_events:
        chosen = random.choice(all_events)
        print(f"  Randomly selected event: {chosen.event.title}")
        all_events = [chosen]

    print()
    print(f"Running {len(agents)} agent(s) on {len(all_events)} event(s)...")
    print(f"Validator: {orchestrator.validator_id}")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

    trace_idx = 0
    for loaded_event in all_events:
        event = loaded_event.event
        for agent in agents:
            print(
                "  Baseline: gathering validator-side market signal "
                f"(source={event.source or 'n/a'}, source_id={event.source_id or 'n/a'}) ..."
            )
            digest = orchestrator.run_agent(event, agent)
            baseline = None
            if digest.prediction_output.metadata:
                baseline = digest.prediction_output.metadata.get("market_baseline")
            if baseline is None:
                print("  Baseline: unavailable for this run.")
            else:
                print("  Baseline: attached to trace metadata.")
            print_summary(digest, trace_idx)
            trace_idx += 1

    print(f"\n{'='*70}")
    if config.storage.persist_traces:
        print(f"  Completed {trace_idx} trace(s). Saved to {config.storage.data_dir / 'traces'}")
    else:
        print(f"  Completed {trace_idx} trace(s). Local trace persistence is disabled.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
