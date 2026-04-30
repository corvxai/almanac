# Decentralised AI Forecasting — Prototype

Phase 1 prototype of the validator-agent execution loop. Proves the basic prediction cycle: event ingestion, agent execution via gateway-proxied data access, evidence digest assembly, and cryptographic trace sealing.

## Quick Start

```bash
# Create a virtualenv and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run all example events
python3 scripts/run_forecast.py

# Run a specific event
python3 scripts/run_forecast.py --event fed
```

Traces are saved to `data/traces/` as JSON files conforming to the Evidence Digest schema.

## Cost estimation

Provider usage/cost handling is documented in `docs/pricing_cards.md`.
Pricing cards can be synced with discovered Claude model strings via
`python scripts/sync_pricing_cards.py --write`.

## Testing

Install dev dependencies (includes pytest), then run from the repo root:

```bash
pip install -r requirements-dev.txt
pytest
```

**Offline runs (default)** — No API keys required. Tests that call real providers are skipped unless you opt in.

**Live / network tests** — Pass `--live` and set the relevant API keys (e.g. `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`). See `.env.example`. Pytest loads the repo-root **`.env`** at startup (via `python-dotenv`, same idea as `scripts/run_gateway.py`), so variables defined there are visible to live tests. Shell exports still win over `.env` when both are set (`override=False`).

```bash
pytest --live
pytest --live tests/gateway/test_provider_live_openrouter.py
```

If a live test is **skipped** at runtime, it is usually missing an API key (the test calls `pytest.skip`). Show skip reasons with:

```bash
pytest -rs
```

**Filter by gateway provider id** — Only runs tests tagged for that provider (`@pytest.mark.provider`). Case-insensitive.

```bash
pytest --provider openrouter
pytest --provider claude
pytest --provider polymarket
```

**`--live` + `--provider`** still runs **every** test tagged for that provider: golden JSON / mock adapter tests **and** the `@pytest.mark.live` smoke test. That is intentional so one command can exercise the full matrix (and `--pretty-print` can show fixture vs mock vs live side by side).

When you **only** want the network smoke test (no golden, no mock), use **`--live-only`** (you can omit `--live`; `--live-only` implies live tests are collected and not skipped):

```bash
pytest --live-only --provider openrouter --pretty-print -s
pytest --live-only --provider claude --pretty-print -s
```

Combine **`--live`** (without `--live-only`) when you want the full provider slice **including** live:

```bash
pytest --live --provider openrouter
pytest --live --provider claude
```

**Substring filter (built-in pytest)** — Useful when you do not want to use `--provider`:

```bash
pytest -k openrouter
```

Use `pytest --help` for flags; `--provider`, `--live`, `--live-only`, and `--pretty-print` are defined in `tests/conftest.py`.

### Pretty-print raw responses (fixture / API checks)

Pytest does **not** show stdout unless you pass **`-s`**. With **`--pretty-print`**, gateway tests that already call providers (or load golden fixtures) also emit **indented JSON** for the raw `dict` they used, so you can compare against `tests/fixtures/raw/` or spot schema drift.

**Live API (real response shape only — no golden / mock):**

```bash
pytest --live-only --provider openrouter --pretty-print -s
pytest --live-only --provider claude --pretty-print -s
```

**Live + full provider slice (golden + mock + live):**

```bash
pytest --live --provider openrouter --pretty-print -s
```

**Mock adapter only (no keys; simulated shape):**

```bash
pytest --provider polymarket --pretty-print -s
```

**Golden JSON files loaded in offline tests:**

```bash
pytest --provider openrouter --pretty-print -s
```

Copy printed output into `tests/fixtures/raw/<provider>/<call_type>.json` (trim secrets) when refreshing goldens. Provenance notes live in `tests/fixtures/raw/README.md`.

## Architecture

Three actors cooperate in every prediction cycle:

**Validator (orchestrator)** — Receives events, manages agent execution, proxies all data access through the gateway, assembles the complete evidence digest trace, and stores it.

**Agent** — A Python function that accepts an event, calls data providers (via the gateway), builds a reasoning chain, and returns a probability prediction.

**Provider Gateway** — Proxy layer between agents and external data. Logs every call, hashes raw responses, and runs an independent evidence extraction pipeline. The agent cannot influence what gets recorded.

## Project Structure

```
src/
├── core/              Shared types, schemas, config
│   ├── schemas.py     Pydantic models for the Evidence Digest
│   ├── config.py      Configuration management
│   └── events.py      Event data structures
├── validator/         Validator orchestration
│   ├── orchestrator.py    Main execution loop
│   ├── sandbox.py         Agent execution sandbox (in-process for now)
│   └── trace_assembler.py Merges gateway evidence + agent reasoning
├── gateway/           Provider Gateway
│   ├── gateway.py     Core proxy — intercepts, logs, hashes, extracts
│   ├── extractor.py   Evidence extraction pipeline
│   └── providers/     Provider adapters
│       ├── base.py        Abstract base provider
│       ├── polymarket.py  Polymarket (mock)
│       └── web_search.py  Web search (mock)
├── agent/             Agent framework
│   ├── base.py        Abstract base agent
│   ├── context.py     ForecastingContext (agent SDK)
│   └── examples/
│       └── simple_agent.py
└── storage/           Persistence layer
    ├── store.py       Abstract storage interface
    └── json_store.py  JSON file implementation
```

## Evidence Digest Schema

Every trace is a sealed `EvidenceDigest` containing:

| Component | Source | Description |
|---|---|---|
| `execution_context` | Validator | IDs, timestamps, sandbox info |
| `event_snapshot` | System | Frozen copy of the event at execution time |
| `provider_calls` | Gateway | Every external data call with extracted evidence |
| `reasoning_chain` | Agent | Step-by-step reasoning with intermediate probabilities |
| `prediction_output` | Agent | Final probability, drivers, uncertainties |
| `resolution_record` | System | Post-hoc outcome (populated later) |
| `trace_integrity` | Validator | SHA-256 hash, schema version, cost totals |

The separation between gateway-produced evidence and agent-produced reasoning is enforced at the architecture level.

## Writing a New Agent

Implement `BaseAgent` and define `predict()`:

```python
from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.schemas import PredictionOutput, ReasoningStepType

class MyAgent(BaseAgent):
    agent_id = UUID("...")
    agent_version = "0.1.0"

    def predict(self, ctx: ForecastingContext) -> PredictionOutput:
        data = ctx.call_provider("polymarket", "get_market", {"market_slug": "..."})
        ctx.record_reasoning_step(
            step_type=ReasoningStepType.EVIDENCE_GATHERING,
            reasoning_text="...",
            input_evidence_refs=[0],
        )
        return PredictionOutput(
            final_probability=0.65,
            reasoning_summary="...",
            key_drivers=["..."],
            key_uncertainties=["..."],
        )
```

## Adding a New Provider

Implement `BaseProvider`:

```python
from src.gateway.providers.base import BaseProvider
from src.core.schemas import ProviderTier

class MyProvider(BaseProvider):
    provider_id = "my_source"
    provider_tier = ProviderTier.SEARCH

    def call(self, call_type: str, params: dict) -> dict:
        # fetch and return raw data
        ...
```

Register it with the gateway and add extraction rules in `extractor.py`.

## Docker sandbox + Bittensor signing

Production deployments run agents in a hardened Docker sandbox and sign every
outbound call to the central gateway with the validator's Bittensor hotkey.

**Two images, one stack:**

```bash
# 1. Build the agent runner once. Validators spawn sibling containers from this tag.
docker compose build agent-runner

# 2. Bring up the validator (orchestrator + signing proxy).
docker compose up validator
```

**Local gateway on the host:** If the provider gateway is running on your machine (`python scripts/run_gateway.py` in another terminal), the validator container must **not** use `http://localhost:8077` — inside the container, `localhost` is the container itself. Point at the Docker host instead (Compose adds `host.docker.internal` via `extra_hosts` in `docker-compose.yaml`):

```bash
GATEWAY_SERVICE_URL=http://host.docker.internal:8077 docker compose up validator
```

Shell-set `GATEWAY_SERVICE_URL` overrides the value from project `.env` for that command, so you can keep `localhost` in `.env` for host-only tools like `scripts/run_forecast.py`.

**Sibling agent containers:** The validator process bind-mounts the proxy UDS directory into each agent-runner container using the **Docker host** path for the source (the daemon does not interpret paths relative to the validator container). The code resolves that host path from `/proc/self/mountinfo` by default. If sibling runs still cannot reach the proxy, set an absolute host path explicitly: `SANDBOX_BIND_SRC` in `.env` (same directory Compose maps onto `/var/run/arcratio` in the validator).

**Debugging agent-runner:** By default, `docker compose logs validator` only shows the validator process. Each agent run is a **separate** container. The validator waits with **`docker wait`** (CLI subprocess), resolving the binary at **`/usr/bin/docker`** / **`/bin/docker`** before ``PATH`` (slim images often lack ``docker`` on ``PATH``). The agent JSON is passed as a **file on the shared UDS bind mount** (`ARCRATIO_RUNNER_INPUT_FILE`), not stdin, because Docker stdin EOF is unreliable. After each run finishes, the validator prints **log tails** unless `ARCRATIO_SANDBOX_RUNNER_LOGS_QUIET=1`. Follow a live run on the host: `docker logs -f <container_id>` from the spawn line. Provider calls from the agent go **UDS → validator local proxy → gateway**, so the gateway tab shows upstream HTTP from the **validator** (or your host), not a separate “agent” client identity.

**Wallet mount:** `docker-compose.yaml` bind-mounts `~/.bittensor` into the
validator container read-only. Set the wallet name / hotkey / netuid in `.env`
(see `.env.example`). The wallet is reachable **only** inside the validator
container — agent sandboxes never see it.

**No wallet on disk?** Pass `--unsafe-no-signing` to either entry point
(`scripts/run_forecast.py` or `scripts/run_local_proxy.py`), or set
`BITTENSOR_SIGNING_REQUIRED=false` in `.env`. The proxy will forward
unsigned requests; the central gateway logs the absence. Dev/contributor use
only — never run production this way.

**In-process dev mode:** `python scripts/run_forecast.py --sandbox in_process`
keeps the existing fast feedback loop. Agents run directly in the orchestrator
process, no Docker required.

## What's Stubbed (Phase 1)

- Provider adapters return mock data (no real API calls)
- Evidence extraction is rule-based (no NLP/LLM)
- Storage is flat-file JSON (no database)
- Central gateway logs Bittensor signature headers but does not yet enforce
  them (deferred to a follow-up phase along with metagraph membership checks)

## Dependencies

- `pydantic>=2.0` — Schema validation
- `httpx>=0.27` — HTTP client (wired in when real providers are added)
- Python 3.10+
