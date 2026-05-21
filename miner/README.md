# Miner Guide

Use this guide to build an agent and interact with the centralized orchestrator
through the miner CLI.

## What this CLI does

- Upload your agent file to the orchestrator (`upload-agent`)
- List published agents (`list-agents`)
- Reserve a credits command for later (`buy-credits`, currently stubbed)

## Prerequisites

- Python 3.10+
- Dependencies installed from repo root:

```bash
pip install -r requirements.txt
```

## Agent contract

Your agent should follow the project contract in `src/agent/base.py`:

- Subclass `BaseAgent`
- Set `agent_id` and `agent_version`
- Implement `predict(ctx: ForecastingContext)`

Reference implementations are in `src/agent/examples/`.

## Quick start

From the repo root:

```bash
python3 miner/cli.py --help
python3 miner/cli.py list-agents
python3 miner/cli.py upload-agent path/to/agent.py --netuid <netuid> --wallet-name <wallet-name> --wallet-hotkey-name <hotkey-name>
```

## Configuration

You can configure the CLI with flags or environment variables.

- `ARCRATIO_ORCHESTRATOR_URL` (default: `http://localhost:4000`)
- `ARCRATIO_API_TOKEN` (optional token for non-wallet-auth endpoints)
- `ARCRATIO_TIMEOUT_SECONDS` (default: `20.0`)
- `ARCRATIO_WALLET_PATH` (default: `~/.bittensor/wallets`)
- `ARCRATIO_WALLET_NAME` (default: `default`)
- `ARCRATIO_WALLET_HOTKEY` (default: `default`)
- `ARCRATIO_SUBTENSOR_NETWORK` (default: `finney`)
- `ARCRATIO_NETUID` (required for auto UID lookup unless `--miner-uid` is provided)

Flag equivalents:

- `--orchestrator-url`
- `--api-token`
- `--timeout-seconds`

## Command reference

### `list-agents`

Calls the orchestrator `list-agents` endpoint.

```bash
python3 miner/cli.py list-agents --limit 25 --offset 0
```

### `upload-agent`

Uploads a local file to the orchestrator `upload-agent` endpoint using multipart
form data.

```bash
python3 miner/cli.py upload-agent path/to/agent.py --netuid <netuid> --wallet-name <wallet-name> --wallet-hotkey-name <hotkey-name>
```

Upload request shape:

- Header: `Authorization: Bearer <wallet-signature-token>`
- Form fields:
  - `minerHotkey` (required)
  - `minerUid` (required)
  - `agentFile` (required, exactly one `.py` file)

How auth + UID are derived:

- CLI loads the miner hotkey keypair from local Bittensor wallet files.
- CLI signs payload `<minerHotkey>:<sha256(agentFileBytes)>`.
- CLI sends base64(signature) in `Authorization: Bearer ...`.
- If `--miner-uid` is omitted, CLI looks up UID from Bittensor metagraph using
  `--subtensor-network` + `--netuid` (or corresponding env vars).
- If the hotkey is not registered on that subnet, upload fails with an error.

Testnet example:
```bash
python3 miner/cli.py upload-agent path/to/agent.py --netuid 172 --subtensor-network test --wallet-name <wallet-name> --wallet-hotkey-name <hotkey-name>
```

Client-side checks enforced by the CLI:

- file must exist
- extension must be `.py`
- max file size `2MB` (`2097152` bytes)

### `buy-credits` (stub)

Command is present for future credits flows but is not implemented yet.

```bash
python3 miner/cli.py buy-credits 25
```

## Authentication note

The upload command now uses wallet-based signature auth via `bittensor`.
Exact backend verification policy can still evolve, but the client no longer
requires a manually provided upload token.

## Troubleshooting

- **Connection errors**: verify `ARCRATIO_ORCHESTRATOR_URL` or `--orchestrator-url`.
- **401/403 responses**: check token configuration (`ARCRATIO_API_TOKEN`).
- **Upload file errors**: confirm the file path exists and points to a file.
