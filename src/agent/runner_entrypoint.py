"""Entry point baked into the agent-runner Docker image.

Runs as PID 1 inside the agent sandbox. The validator's `sandbox_docker.py`
spawns the runner image with:

- `--network=none` (no IP stack of any kind)
- `--read-only` rootfs + small tmpfs for /tmp
- `cap_drop=ALL`, `no-new-privileges`, non-root UID 10001
- a single read-only bind mount: the UNIX domain socket exposed by the
  validator-local signing proxy at `/run/arcratio/proxy.sock`
- env: `SANDBOX_PROXY_URL=http+unix:///run/arcratio/proxy.sock`,
  `RUN_ID=<uuid>`, and `ARCRATIO_RUNNER_INPUT_FILE=/run/arcratio/inputs/.arcratio_stdin_<uuid>.json`
  (JSON payload on the shared bind mount; stdin is not used because Docker
  stdin EOF is unreliable.)

The entrypoint:

1. Reads the input JSON from ``ARCRATIO_RUNNER_INPUT_FILE`` (preferred), from
   ``--input-file``, or from stdin for manual runs.
2. Builds a `RemoteProvider` factory that posts over the UDS to the local
   proxy, attaching `X-Run-Id` so the proxy can attribute calls.
3. Imports the requested agent class.
4. Calls `agent.predict(ctx)` over a `ForecastingContext` backed by a
   `SandboxGateway`.
5. Writes `AgentResult.model_dump_json()` to stdout. Exits non-zero on any
   exception.

The runner image deliberately omits `bittensor`, the validator code, and the
gateway server — a compromised agent has no chain library to import and no
secret key material to exfiltrate.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
from typing import Any

import httpx

from src.core import constants
from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.agent.sandbox_gateway import SandboxGateway
from src.core.events import Event
from src.gateway.client import RemoteProvider
from src.gateway.provider_capabilities import (
    provider_default_call_type,
    provider_supports_completions,
)


# httpx URL the RemoteProvider's preconfigured client points at. The host is
# arbitrary because we replace transport with a UDS-bound one — but FastAPI
# still needs *some* host header.
_BASE_URL = "http://proxy.local"


def _make_uds_client(
    socket_path: str,
    run_id: str,
    timeout: float = float(constants.GATEWAY.call_timeout_seconds),
) -> httpx.Client:
    transport = httpx.HTTPTransport(uds=socket_path)
    return httpx.Client(
        base_url=_BASE_URL,
        transport=transport,
        timeout=timeout,
        headers={"X-Run-Id": run_id},
    )


def _socket_path_from_env() -> str:
    """Resolve the UDS path from `SANDBOX_PROXY_URL`.

    Accepts either a plain filesystem path or `http+unix:///abs/path/to.sock`.
    """
    raw = os.environ.get("SANDBOX_PROXY_URL", "").strip()
    if not raw:
        return "/run/arcratio/proxy.sock"
    if raw.startswith("http+unix://"):
        return raw[len("http+unix://"):]
    return raw


def _load_agent_class(module_name: str, class_name: str) -> type[BaseAgent]:
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise RuntimeError(f"agent class {module_name}.{class_name} not found")
    if not isinstance(cls, type) or not issubclass(cls, BaseAgent):
        raise RuntimeError(f"{module_name}.{class_name} is not a BaseAgent subclass")
    return cls


def _load_agent_class_from_code(code: str, class_name: str | None = None) -> type[BaseAgent]:
    scope: dict[str, Any] = {}
    exec(compile(code, "<orchestrator-agent>", "exec"), scope)  # noqa: S102
    if class_name:
        cls = scope.get(class_name)
        if not isinstance(cls, type) or not issubclass(cls, BaseAgent):
            raise RuntimeError(f"class '{class_name}' is not a BaseAgent subclass")
        return cls
    candidates = []
    for obj in scope.values():
        if isinstance(obj, type) and issubclass(obj, BaseAgent) and obj is not BaseAgent:
            candidates.append(obj)
    if not candidates:
        raise RuntimeError("no BaseAgent subclass found in inline agent_code")
    if len(candidates) > 1:
        names = ", ".join(sorted(c.__name__ for c in candidates))
        raise RuntimeError(
            f"multiple BaseAgent subclasses found in inline agent_code ({names})"
        )
    return candidates[0]


def _read_stdin_payload() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        raise RuntimeError("no input on stdin (expected JSON event payload)")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid stdin JSON: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    # Early stderr line so `docker logs` shows life before payload/agent work.
    sys.stderr.write("[arcratio-runner] entering main\n")
    sys.stderr.flush()

    parser = argparse.ArgumentParser(description="Arcratio agent sandbox runner")
    parser.add_argument(
        "--input-file",
        default=None,
        help=(
            "Optional path to a JSON file with the "
            "{event, agent_module, agent_class} payload or "
            "{event, agent_code, inline_class} payload. "
            "payload. Defaults to stdin."
        ),
    )
    args = parser.parse_args(argv)

    env_input = os.environ.get("ARCRATIO_RUNNER_INPUT_FILE", "").strip()
    if env_input:
        with open(env_input, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    elif args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    else:
        payload = _read_stdin_payload()

    try:
        event_dict = payload["event"]
    except KeyError as exc:
        raise RuntimeError(f"missing required payload key: {exc}") from exc

    inline_code = payload.get("agent_code")
    inline_class = payload.get("inline_class")
    agent_module = payload.get("agent_module")
    agent_class = payload.get("agent_class")

    run_id = os.environ.get("RUN_ID", "").strip()
    if not run_id:
        raise RuntimeError("RUN_ID env var is required")

    socket_path = _socket_path_from_env()

    event = Event.model_validate(event_dict)
    if isinstance(inline_code, str) and inline_code.strip():
        cls = _load_agent_class_from_code(inline_code, inline_class)
    else:
        if not isinstance(agent_module, str) or not isinstance(agent_class, str):
            raise RuntimeError("missing agent loader keys: expected inline agent_code or agent_module+agent_class")
        cls = _load_agent_class(agent_module, agent_class)
    agent = cls()

    client = _make_uds_client(socket_path, run_id)

    def _provider_factory(provider_id: str) -> RemoteProvider:
        # Pass the provider's real capabilities so typed data-provider calls
        # (polymarket.get_market, web_search.search) forward their `params`
        # instead of being coerced to an empty completions payload — otherwise
        # market_slug/query never leave the sandbox (agent can't pick a market).
        return RemoteProvider(
            provider_id=provider_id,
            client=client,
            headers={"X-Run-Id": run_id},
            default_call_type=provider_default_call_type(provider_id),
            supports_completions=provider_supports_completions(provider_id),
        )

    gateway = SandboxGateway(provider_factory=_provider_factory)
    ctx = ForecastingContext(event=event, gateway=gateway)

    try:
        result = agent.predict(ctx)
    finally:
        client.close()

    sys.stdout.write(result.model_dump_json())
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"runner_entrypoint failed: {exc}\n")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
