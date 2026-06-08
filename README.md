<div align="center">

```
   CORVUS LABS PRESENTS
--------------------------------------------------------------------------------------------------
   █████████   █████       ██████   ██████   █████████   ██████   █████   █████████     █████████ 
  ███▒▒▒▒▒███ ▒▒███       ▒▒██████ ██████   ███▒▒▒▒▒███ ▒▒██████ ▒▒███   ███▒▒▒▒▒███   ███▒▒▒▒▒███
 ▒███    ▒███  ▒███        ▒███▒█████▒███  ▒███    ▒███  ▒███▒███ ▒███  ▒███    ▒███  ███     ▒▒▒ 
 ▒███████████  ▒███        ▒███▒▒███ ▒███  ▒███████████  ▒███▒▒███▒███  ▒███████████ ▒███         
 ▒███▒▒▒▒▒███  ▒███        ▒███ ▒▒▒  ▒███  ▒███▒▒▒▒▒███  ▒███ ▒▒██████  ▒███▒▒▒▒▒███ ▒███         
 ▒███    ▒███  ▒███      █ ▒███      ▒███  ▒███    ▒███  ▒███  ▒▒█████  ▒███    ▒███ ▒▒███     ███
 █████   █████ ███████████ █████     █████ █████   █████ █████  ▒▒█████ █████   █████ ▒▒█████████ 
▒▒▒▒▒   ▒▒▒▒▒ ▒▒▒▒▒▒▒▒▒▒▒ ▒▒▒▒▒     ▒▒▒▒▒ ▒▒▒▒▒   ▒▒▒▒▒ ▒▒▒▒▒    ▒▒▒▒▒ ▒▒▒▒▒   ▒▒▒▒▒   ▒▒▒▒▒▒▒▒▒  
```                                                                                                                                         

</div>

Almanac is a validator stack for decentralized forecasting on Bittensor. It combines two incentive mechanisms, both geared towards improving prediction accuracy.

IM1: Almanac.market is the front end, a prediction market terminal that makes competing and submitting predictions by trading simpler and much more accessible.

IM2: Miners submit forecasting agents that predict events while validators build reasoning traces for our data pipeline.

Both incentive mechanisms generate scores and publish blended weights on-chain.

- [Introduction](#introduction)
- [How It Works](#how-it-works)
- [Miner and Validator Functionality](#miner-and-validator-functionality)
  - [Miner](#miner)
  - [Validator](#validator)
- [Running a Validator](#running-a-validator)
- [For Miners](#for-miners)
- [Developer and Test Docs](#developer-and-test-docs)
- [Repository Structure](#repository-structure)
- [Status](#status)
- [License](#license)

## Introduction

This repository powers the validator side of the Almanac SN41 workflow:

- Sync miner metadata and market activity.
- Score miners under multiple mechanisms.
- Blend scores into final validator weights.
- Run an auditable agent/evidence execution path for signals.

The goal is production-safe validator behavior with transparent scoring and reproducible traces.

## How It Works

Each scoring cycle, the validator:

1. Syncs the subnet metagraph and miner metadata.
2. Computes Market incentive scores (when enabled).
3. Computes Forecasting scores (when enabled).
4. Blends score vectors and submits a single `set_weights` update.

Mechanism toggles and blend shares are configured in `src/core/constants.py` under `VALIDATOR_LOOP`.

## Miner and Validator Functionality

### Miner

- Miners participate through Almanac trading and metadata registration.
- Miner onboarding and CLI flows are documented in `miner/README.md`.
- Miner metadata can be registered with:

```bash
pip install -r requirements-miner.txt
python3 scripts/run_almanac_miner.py
```

### Validator

- `scripts/run_validator.py` is the long-running validator entrypoint.
- `scripts/run_forecast.py` is a developer harness for agent orchestration and trace generation (no chain interaction).
- Traces are written to `data/traces/` as evidence-digest JSON artifacts.

## Running a Validator

### Requirements

- Python 3.10+
- `pip`
- Docker Engine + Docker Compose plugin
- A registered Bittensor validator wallet/hotkey setup (via btcli)

### Production: Docker required

When Forecasting IM is enabled, the validator must spawn sandboxed agent-runner
child containers. In this repo, that means running the validator with Docker
access (`docker.sock`) via Compose.

```bash
# Build both images once (runner + validator)
docker compose build agent-runner validator
```

```bash
docker compose run --rm validator \
  python scripts/run_validator.py \
  --netuid 41 \
  --wallet.name almanac-vali \
  --wallet.hotkey almanac-vali-hot \
  --logging.info
```

### PM2 (commonly used to continuously run/supervise a validator)

```bash
# Recommended
pm2 start --name almanac-validator --interpreter bash -- -lc '
docker compose run --rm validator \
  python scripts/run_validator.py \
    --netuid 41 \
    --wallet.name almanac-vali \
    --wallet.hotkey almanac-vali-hot \
    --logging.info
'

# Alternative: supervise the compose service directly (best when args are wired in compose command/env)
pm2 start "docker compose up validator" --name almanac-validator-compose-up
```

Tip: build images separately on deploy/update (avoid `--build` in PM2 commands
so restart loops stay fast and predictable):

```bash
docker compose build agent-runner validator
```

Persist and enable restart on reboot:

```bash
pm2 save
pm2 startup
pm2 logs almanac-validator
```

## For Miners

Miner documentation lives in `miner/README.md`, including:

- prerequisites,
- agent packaging/upload flows,
- CLI command reference and troubleshooting.

## Developer and Test Docs

Public docs are intentionally concise.  
Detailed contributor guidance (tests, live-provider workflows, Docker sandboxing, signing/proxy behavior, and advanced local ops) is in `tests/README.md`.

Additional references:

- Pricing and provider usage notes: `docs/pricing_cards.md`
- Pricing card sync helper: `python scripts/sync_pricing_cards.py --write`

## Repository Structure

```text
src/
  core/        Shared schemas, config, and constants
  validator/   Scoring and orchestrator pipeline
  gateway/     Provider proxy and evidence extraction
  agent/       Agent framework and examples
  storage/     Trace persistence backends
scripts/       Runtime entrypoints and utilities
tests/         Unit/integration/live test suites and contributor docs
miner/         Miner-facing CLI and onboarding docs
```

## Environments

| Network | Netuid |
| ----------- | -----: |
| Mainnet     |     41 |
| Testnet     |    172 |

## Community

Join the vibrant Bittensor community and find our channel `#פ • almanac • 41` on [Discord](https://discord.gg/bittensor).

## License

The SN41 Almanac subnet is released under the [MIT License](./LICENSE).
