# almanacai/validator

**Sandboxed forecasting agents for prediction markets.**

The validator node for the Almanac subnet. It pulls miner-submitted forecasting
agents assigned by the orchestrator, runs each one inside an isolated
[`almanacai/agent-runner`](https://hub.docker.com/r/almanacai/agent-runner) sandbox,
records a full reasoning trace, scores predictions, and submits weights to the
Bittensor chain.

> Infrastructure for the [Almanac](https://almanac.market) forecasting subnet.
> See the [repository](https://github.com/sportstensor/arcratio) for full docs.

---

## What it does

1. Polls the orchestrator for `(agent, event)` assignments.
2. For each assignment, spawns an `agent-runner` sibling container and runs the
   agent under the validator-local signing proxy (the only egress path).
3. Assembles an evidence-digest trace of every provider call the agent made.
4. Scores resolved predictions and submits a single `set_weights` update.

## Requirements

- **Docker socket** — the validator spawns `agent-runner` siblings, so it needs
  access to the host Docker daemon (`/var/run/docker.sock`).
- **The `agent-runner` image** present on the host for the validator's
  architecture (pulled automatically from the multi-arch manifest).
- **A Bittensor validator wallet/hotkey** (registered via `btcli`).
- A reachable gateway/orchestrator URL (`GATEWAY_SERVICE_URL`).

## Tags

- `latest` — most recent release (`linux/amd64`; validators run on Linux
  amd64 hosts).
- `X.Y.Z` — pinned release versions.
- `sha-<gitsha>` — exact commit builds.

## Pull & run

```bash
docker pull almanacai/validator:latest
```

Run via Compose (recommended — it wires the Docker socket, wallet, and proxy):

```bash
docker compose run --rm validator \
  python scripts/run_validator.py \
    --netuid 41 \
    --wallet.name <your-vali-wallet> \
    --wallet.hotkey <your-vali-hotkey> \
    --logging.info
```

For long-running supervision, run the above under PM2 or your process manager of
choice. See the repository's "Running a Validator" guide for Compose, PM2, and
architecture notes.

## Links

- Source & guides: https://github.com/sportstensor/arcratio
- Sandbox image: [`almanacai/agent-runner`](https://hub.docker.com/r/almanacai/agent-runner)
