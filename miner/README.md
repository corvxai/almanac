# Miner Guide

Use this guide to build an agent and interact with gateway orchestrator miner endpoints via `miner/cli.py`.

## What this CLI does

- Submit your agent file to gateway (`submit-agent`, alias `upload-agent`)
- List published agents (`list-agents`)
- Reserve a credits command for later (`buy-credits`, currently stubbed)

`get-agent` is intentionally not part of this miner CLI because that route is validator-facing.

## Prerequisites

- Python 3.10+
- Dependencies installed from repo root:

```bash
pip install -r requirements.txt
```

## Creating agents

Your agent should follow the project contract in `src/agent/base.py`:

- Subclass `BaseAgent`
- Set `agent_id` and `agent_version`
- Implement `predict(ctx: ForecastingContext)`

Reference implementations are in `src/agent/examples/`.

### Calling providers

Use `ctx.call_provider(provider_id, call_type, params)` to reach external data and LLM APIs. Every key in `params` is forwarded as-is through the validator gateway — **the platform does not inject defaults** for LLM sampling or token limits. If you omit a param, it is not sent upstream.

Example:

```python
ctx.call_provider("openrouter", "chat_completion", {
    "model": "anthropic/claude-sonnet-4-6",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 1024,
    "temperature": 0.2,
})
```

Common `provider_id` / `call_type` pairs:

| Provider | `call_type` | Typical use |
|---|---|---|
| `anthropic` | `messages` | Claude Messages API (tools, web search) |
| `openrouter` | `chat_completion` | OpenAI-compatible chat via OpenRouter |
| `openai` | `chat_completion` | OpenAI chat completions |
| `gemini` | `generate_content` | Google Gemini |
| `perplexity` | `chat_completion` | Perplexity Sonar |
| `polymarket` | `get_market` | Market odds and metadata |
| `web_search` | `search` | Web search results |

Provider-specific params (for example `system`, `tools`, `grounding`) are also pass-through — include them in `params` when your chosen provider supports them. See the example agents and `src/gateway/providers/` for what each adapter accepts.

### LLM params you must define

These are the most common params for inference providers. **Only params you set in your agent code are forwarded.** Calibrate them per provider and model — some models reject certain sampling params entirely (for example newer Claude models may 400 on `temperature` or `top_p`).

| Param | Type | Description |
|---|---|---|
| `model` | `str` | Model ID for the provider (required for LLM calls). |
| `messages` | `list[dict]` | Chat messages, typically `[{"role": "user", "content": "..."}]`. |
| `max_tokens` | `int` | Maximum tokens in the completion. Omit only if you accept the provider adapter's own fallback. |
| `temperature` | `float` | Randomness (lower = more deterministic). Often `0.1`–`0.3` for forecasting. Omit on models that reject sampling params. |
| `top_p` | `float` | Nucleus sampling alternative to temperature. Use one or the other, not both, on Claude 4+ models. |
| `stop` | `list[str]` | Sequences that stop generation when encountered. |

Anthropic-only params (pass-through when using `anthropic` / `messages`):

| Param | Type | Description |
|---|---|---|
| `system` | `str` | System prompt. |
| `tools` | `list[dict]` | Tool definitions (for example web search). |

You are responsible for choosing values that work with your target model, testing them locally, and handling provider errors when a model rejects a param combination.

## Quick start

From the repo root:

```bash
python3 miner/cli.py --help
python3 miner/cli.py list-agents
python3 miner/cli.py submit-agent path/to/agent.py --wallet-name <wallet-name> --wallet-hotkey-name <hotkey-name>
```
Optionally run with explicit gateway API key defined, but MORE SECURE to define GATEWAY_API_KEY in .env:
```bash
python3 miner/cli.py submit-agent path/to/agent.py --wallet-name <wallet-name> --wallet-hotkey-name <hotkey-name> --gateway-api-key <gateway-api-key>
```

## Configuration

The CLI uses the shared orchestrator constant from `src/core/constants.py`:
`ORCHESTRATOR_API_URL` (default: `http://localhost:4000`).

You can still override the URL per command with `--orchestrator-url`.
The CLI also auto-loads repo-root `.env` (`python-dotenv`) before resolving env vars.

- `ARCRATIO_TIMEOUT_SECONDS` (default: `20.0`)
- `ARCRATIO_WALLET_PATH` (default: `~/.bittensor/wallets`)
- `ARCRATIO_WALLET_NAME` (default: `default`)
- `ARCRATIO_WALLET_HOTKEY` (default: `default`)
- `GATEWAY_API_KEY` or `ARCRATIO_GATEWAY_API_KEY`

Flag equivalents:

- `--orchestrator-url`
- `--timeout-seconds`
- `--wallet-path`
- `--wallet-name`
- `--wallet-hotkey-name`
- `--gateway-api-key`

## Command reference

### `list-agents`

Calls `GET /v1/agents/list-agents`.

```bash
python3 miner/cli.py list-agents --limit 25 --offset 0
```

### `submit-agent` (alias: `upload-agent`)

Calls `POST /v1/agents/submit-agent` with the raw `.py` file bytes as the request body.

```bash
python3 miner/cli.py submit-agent path/to/agent.py --wallet-name <wallet-name> --wallet-hotkey-name <hotkey-name> --gateway-api-key <gateway-api-key>
```

Submit auth + request shape:

- Content type: `application/octet-stream`
- Header: `x-agent-filename: <agent file name>`
- Header: `Authorization: Bearer <gateway_api_key>`
- Signed headers:
  - `x-miner-hotkey`
  - `x-miner-signature`
  - `x-miner-nonce`
  - `x-miner-timestamp`

How signing works:

- CLI loads miner hotkey from local Bittensor wallet files.
- CLI builds canonical message domain `sub41-agent-v1`:
  - method
  - path (and query when present)
  - subject hotkey
  - nonce
  - timestamp (ms)
  - sha256(body)
- CLI signs with the wallet hotkey and sends hex signature as `x-miner-signature`.
- CLI does not include org or gateway account id in payload headers or signature.

Compatibility flags retained on `submit-agent`:

- `--miner-hotkey` (optional override; defaults to wallet hotkey ss58)
- `--gateway-api-key` (required by gateway upload auth unless env var is set)
- `--miner-uid` (currently ignored by submit endpoint)
- `--subtensor-network` (currently ignored by submit endpoint)
- `--netuid` (currently ignored by submit endpoint)

Gateway key linking behavior:

- First valid upload with an unoccupied API key auto-links miner to that key/org.
- If an API key is already linked to a miner, upload fails; use a fresh key.

Client-side checks enforced by the CLI:

- file must exist
- extension must be `.py`
- max file size `2MB` (`2097152` bytes)

### `buy-credits` (stub)

Command is present for future credits flows but is not implemented yet.

```bash
python3 miner/cli.py buy-credits 25
```

## Troubleshooting

- **Connection errors**: verify `ORCHESTRATOR_API_URL` in `src/core/constants.py` or use `--orchestrator-url`.
- **Wallet load errors**: verify wallet path/name/hotkey name and that hotkey files exist.
- **Upload file errors**: confirm the file path exists and points to a `.py` file.
