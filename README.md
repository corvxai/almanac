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
  - [Shared prerequisites](#shared-prerequisites)
  - [Almanac Market miners](#almanac-market-miners)
  - [Almanac Forecasting miners](#almanac-forecasting-miners)
- [Repository Structure](#repository-structure)
- [Environments](#environments)
- [Community](#community)
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
- Participate through trading on `beta.almanac.market`.
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
- A registered Bittensor validator wallet/hotkey setup (via `btcli`)
- `WANDB_API_KEY` in `.env` for W&B logging (or use `--wandb.off`)

`bittensor==11.0.1` in `requirements.txt` includes both the Python SDK and
`btcli`; do not install the legacy separate `bittensor-cli` package. In v11,
create keys with `btcli wallet create --wallet <coldkey-name> --wallet-hotkey
<hotkey-name>` and register with `btcli subnets register --netuid 41 --wallet
<coldkey-name> --wallet-hotkey <hotkey-name> --network finney`.
The dotted flags in the validator examples below belong to Almanac's scripts,
not to `btcli`.

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
pm2 start bash --name almanac-validator -- -lc '
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

There are **two independent miner tracks**. Pick one (you can do both with the same UID, but setup is different):

| Track | What you do | How you earn | Entrypoint |
|---|---|---|---|
| **[Almanac Market](#almanac-market-miners)** | Trade prediction markets on [almanac.market](https://beta.almanac.market) | ROI + qualified volume over a rolling window | `python3 scripts/run_market_miner.py` |
| **[Almanac Forecasting](#almanac-forecasting-miners)** | Submit a forecasting agent via the portal | Sustained accuracy / calibrated forecasts | `python3 miner/cli.py` |

### Shared prerequisites

Both tracks need:

1. **Python 3.10+** and repo deps:

```bash
git clone https://github.com/corvxai/almanac.git
cd almanac
pip install -r requirements.txt
```

2. A **Bittensor wallet** with a **registered miner UID** on the subnet:
   - Bittensor v11 docs: https://www.bittensor.com/docs
   - Register with Bittensor v11: `btcli subnets register --netuid <netuid> --wallet <coldkey-name> --wallet-hotkey <hotkey-name> --network <finney|test>`
   - Mainnet netuid `41` / testnet netuid `172`

---

### Almanac Market miners

Market miners generate scored signals by trading on Almanac. Orders route through your Polymarket proxy wallet; validators ingest trade history, score ROI / qualified volume, and set weights. Scoring is model-agnostic — manual, scripted, or automated strategies all compete the same way.

A **1% fee** is collected on every buy and goes toward the daily reward pool. Sustained edge, meaningful volume, and beating the competition matter most.

#### 1. Create and fund an Almanac Market account

1. Go to **[https://beta.almanac.market](https://beta.almanac.market)**
2. Create an account
3. Deploy your safe / proxy wallet
4. Sign all required approvals
5. Fund the safe wallet so you can trade

#### 2. Link your Bittensor coldkey (via web app or CLI)

Through the web app in Almanac Market settings:

1. Install the [Bittensor wallet browser extension](https://www.bittensor.com/wallet)
2. Import the **coldkey** tied to your registered miner UID
3. Link that wallet to your Almanac account

Or through the repo CLI:

1. Run `scripts/run_api_trading.py`
2. Select `Link Bittensor UID to Almanac account` option and follow prompts

You only need to do this once. After the account is linked, you can trade in the dApp without reconnecting the extension every session.

#### 3. Register on-chain metadata (required)

Validators map your UID to your Almanac / Polymarket EOA by reading a short on-chain commitment (first 5 characters of the address).

**Interactive (recommended first time):**

```bash
python3 scripts/run_market_miner.py
```

The wizard prompts for wallet name, hotkey, network (`finney` or `test`), and your Almanac / Polymarket EOA (`0x…`), then submits metadata on-chain.

**Non-interactive:**

```bash
python3 scripts/run_market_miner.py \
  --wallet.name <coldkey-name> \
  --wallet.hotkey <hotkey-name> \
  --subtensor.network finney \
  --polymarket.id 0xYourEOAAddressHere
```

Use `--subtensor.network test` for testnet (netuid `172`).

#### 4. Trade via the Almanac dApp

Once metadata is registered and the account is linked:

1. Trade on **[https://beta.almanac.market](https://beta.almanac.market)**
2. Validators detect trades automatically, score them, and include you in weight setting

No long-running miner process is required for dApp trading after metadata registration.

#### 5. Trade via API (optional)

For programmatic / automated trading:

1. Complete steps 1–3 above
2. Copy the env template and fill credentials:

```bash
cp miner/market/api_trading.env.example miner/market/api_trading.env
```

Required fields in `miner/market/api_trading.env`:

| Variable | Meaning |
|---|---|
| `EOA_WALLET_ADDRESS` | Your EOA address |
| `EOA_WALLET_PK` | EOA private key (signing) |
| `EOA_PROXY_FUNDER` | Polymarket proxy / funder address |
| `POLYMARKET_API_KEY` / `SECRET` / `PASSPHRASE` | Optional; the client can generate these |

Multiple wallets are supported with prefixes (`WALLET1_…`, `WALLET2_…`, etc.) — see the example file.

3. Start the interactive API client:

```bash
python3 scripts/run_api_trading.py
```

The client can generate Polymarket API credentials, open Almanac trading sessions, search markets, place signed CLOB orders, and submit proxy-signed EIP-712 orders.

Code lives under `miner/market/` (`miner.py`, `api_trading.py`).

---

### Almanac Forecasting miners

Forecasting miners submit agents that validators run in a sandboxed Docker environment. You need portal credits and a gateway API key — not an Almanac Market trading account.

**Deep dive (sandbox allowlist, belief-path contract, provider calls):** [`miner/README.md`](miner/README.md)

#### Quick start

1. **Portal account + API key** at [https://portal.almnc.ai](https://portal.almnc.ai)
2. Put the key in a repo-root `.env`:

```bash
GATEWAY_API_KEY=your-gateway-api-key
```

3. **Build an agent** from `src/agent/examples/` (recommended starter: `src/agent/examples/v1_agent2_basic.py`). Subclass `BaseAgent`, implement `predict(ctx) -> AgentResult`, and include a valid `beliefPath`.
4. **Test** against a live random event:

```bash
python3 miner/cli.py test-agent path/to/agent.py
```

5. **Submit** (signed with your miner hotkey):

```bash
python3 miner/cli.py submit-agent path/to/agent.py \
  --wallet-name <wallet-name> \
  --wallet-hotkey-name <hotkey-name>
```

The first successful `submit-agent` with an unused API key auto-links that key to your miner hotkey. If the key is already linked to another miner, request a fresh key.

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

Public validator runs: [wandb.ai/corvx/almanac-vali-logs](https://wandb.ai/corvx/almanac-vali-logs)

Join the vibrant Bittensor community and find our channel `#פ • almanac • 41` on [Discord](https://discord.gg/bittensor).

## License

The SN41 Almanac subnet is released under the [MIT License](./LICENSE).
