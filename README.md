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

- [Introduction](#introduction)
- [Miner and Validator Functionality](#miner-and-validator-functionality)
  - [Miner](#miner)
  - [Validator](#validator)
- [Running a Validator](#running-a-validator)
- [For Miners](#for-miners)
- [Repository Structure](#repository-structure)
- [Status](#status)
- [License](#license)

## Introduction

Almanac is a validator stack for decentralized agentic forecasting and market prediction performance on Bittensor. It combines two incentive mechanisms, both geared toward improving prediction accuracy.

IM1 — **Almanac Market** (`almanac.market`): a prediction-market terminal that makes competing and submitting predictions through trading more accessible.

IM2 — **Almanac Forecasting**: miners submit forecasting agents that predict events while validators build reasoning traces for our data pipeline.

Both incentive mechanisms generate scores and publish blended weights on-chain.

## Miner and Validator Functionality

### Miner

There are two tracks for miners to participate in:
1. Almanac Market 
- Participate through trading on `almanac.market`.
- Requires metadata registration and connecting UID/hotkey to Almanac Market account
- 1% fee is collected from every buy trade and used towards the daily reward pool
- Scoring mechanism largely favors sustained edge, substantial volume, and winning over the competition

2. Almanac Forecasting
- Participate by creating an Almanac Portal account and funding credits on `portal.almnc.ai`
- Build and submit forecasting agents through `miner/cli.py`.
- Validators will execute agent's to forecast event predictions.
- Scoring mechanism largely favors sustained accuracy, calibrated confidence, and beating the market
- Miner onboarding and CLI flows are documented in `miner/README.md`.

### Validator

The validator's high-level, continuously looping execution:

- Sync miner metadata and market activity.
- Execute miner agents in locked down docker environment using Almanac's portal
- Posts agent predictions and reasoning chain metadata to Almanac's portal
- Score miners under multiple mechanisms.
- Blend scores into final validator weights.

## Running a Validator

### Requirements

- Python 3.10+
- `pip`
- Docker Engine + Docker Compose plugin
```bash
sudo apt update
sudo apt install docker.io docker-compose-v2
```
- A registered Bittensor validator wallet/hotkey setup (via btcli)
- `WANDB_API_KEY` in `.env` for W&B logging (or use `--wandb.off`)

### Production: Docker required

When Almanac Forecasting is enabled, the validator must spawn sandboxed agent-runner
child containers. In this repo, that means running the validator with Docker
access (`docker.sock`) via Compose.

```bash
# Build both images once (runner + validator)
docker compose build agent-runner validator
```

> **Build for the architecture the validator will run on.** The validator
> spawns the `agent-runner` image as sibling containers, and container images
> are architecture-specific. An image built on an Apple Silicon (arm64) laptop
> will not start on an amd64 Linux server, and vice versa. `docker compose
> build` only builds for the builder's native arch. To target a specific
> platform (e.g. building on a Mac for an amd64 server), use the arch-aware
> wrappers, which set `--platform` explicitly and warn on cross-arch builds:
>
> ```bash
> # Linux / macOS / WSL / Git-Bash
> PLATFORM=linux/amd64 scripts/build_images.sh all
> # Windows PowerShell
> ./scripts/build_images.ps1 -Target all -Platform linux/amd64
> ```
>
> Or pin Compose builds with `export DOCKER_DEFAULT_PLATFORM=linux/amd64`.

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

Two tracks:

| Track | Install | Entrypoint | Docs |
|---|---|---|---|
| **Almanac Market** | `pip install -r requirements.txt` | `python3 scripts/run_market_miner.py` | [almanac.market](https://almanac.market), `miner/market/` |
| **Almanac Forecasting** | `pip install -r requirements.txt` | `python3 miner/cli.py` | **`miner/README.md`** (numbered quick start) |

Forecasting miners: 
1. Get a gateway API key at [portal.almnc.ai](https://portal.almnc.ai)
2. Put `GATEWAY_API_KEY` in `.env`
3. Build an agent from `src/agent/examples/`
4. Test it on a live random event with
   `python3 miner/cli.py test-agent path/to/agent.py`
5. Submit it using `python3 miner/cli.py submit-agent path/to/agent.py ...`

Full steps, sandbox allowlist, and belief-path contract are in `miner/README.md`.

## Repository Structure

```text
src/
  core/        Shared schemas, config, and constants
  validator/
    validator.py   Shared blend loop and on-chain weight submission
    market/        Almanac Market scoring and metadata
    forecasting/   Almanac Forecasting scoring, orchestration, and sandboxing
  gateway/     Provider proxy and evidence extraction
  agent/       Agent framework and examples
  storage/     Trace persistence backends
scripts/       Runtime entrypoints and utilities
tests/         Unit/integration/live test suites and contributor docs
miner/
  cli.py       Almanac Forecasting agent submission CLI
  market/      Almanac Market miner registration and trading integration
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
