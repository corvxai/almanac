# Almanac Forecasting Miner Guide

Use this guide to build an Almanac Forecasting agent, test it locally, and submit it to the Almanac gateway
with `miner/cli.py`.


## Quick start

### 1. Install Forecasting deps

From the repo root (Python 3.10+):

```bash
pip install -r requirements.txt
```

### 2. Get a gateway API key

1. Open the Almanac portal: [https://portal.almnc.ai](https://portal.almnc.ai)
2. Create an account / org and generate a **gateway API key**.
3. Put it in a repo-root `.env` (preferred) or pass `--gateway-api-key` on the CLI:

```bash
# repo-root .env
GATEWAY_API_KEY=your-gateway-api-key
```

You also need a registered Bittensor miner wallet/hotkey on the subnet (mainnet
netuid `41`, testnet `172`). The CLI signs uploads with that hotkey.

The CLI already talks to the production orchestrator by default — no URL config
needed unless you are pointing at a non-prod environment.

**Key linking:** the first successful `submit-agent` with an unused API key
auto-links that key to your miner hotkey/org. If the key is already linked to
another miner, upload fails — request a fresh key.

### 3. Create an agent

Start from a reference implementation and adapt it:

| Starter | Path | Notes |
|---|---|---|
| **Recommended** | `src/agent/examples/v1_agent2_basic.py` | One web search + one LLM call |
| Minimal LLM-only | `src/agent/examples/v1_agent1_llm_only.py` | No external search |
| JSON contract | `src/agent/examples/v1_agent_json_output.py` | Validated forecast JSON shape |
| Capability ladder | `v1_agent3` … `v1_agent6` | Search / orchestration / markets |

Contract (`src/agent/base.py`):

- Subclass `BaseAgent`
- Set stable `agent_id` (UUID) and `agent_version`
- Implement `predict(ctx) -> AgentResult`
- Include a valid `beliefPath` (required; see [Belief path](#belief-path-required))

Submit a **single `.py` file** (max 2MB). Prefer one `BaseAgent` subclass per
file. Reach external data and LLMs only through
`ctx.call_provider(...)` — see [Allowed dependencies](#allowed-dependencies)
and [Calling providers](#calling-providers).

### 4. Test locally before submit

Run your agent against a random live event and the production portal gateway:

```bash
python3 miner/cli.py test-agent path/to/agent.py
```

The command loads `GATEWAY_API_KEY` from the repo-root `.env`, fetches
`GET /v1/events/random`, executes your agent in-process, and sends provider
calls to the live `POST /v1/gateway/completions` endpoint. It validates the
returned `AgentResult` and performs the same JSON round-trip used at the
production sandbox boundary. No OpenRouter or Anthropic API key is needed.

To repeat a test against the same portal event:

```bash
python3 miner/cli.py test-agent path/to/agent.py --event-id <portal-event-id>
```

The portal test endpoint currently exposes the provider IDs returned by
`GET /v1/gateway/providers` (currently `openrouter`). For search, use an
OpenRouter `:online` model or a `perplexity/*` model slug through the
`openrouter` provider. The CLI reports a clear error if your agent requests a
provider unavailable on the portal test path.

Portal completions provide normalized assistant text, usage, and cost data.
They do not currently expose provider-native citation/annotation fields, so an
agent whose logic depends on those raw fields cannot fully exercise that logic
through this test command.

Before uploading your own file, check:

- [ ] Subclasses `BaseAgent`, returns `AgentResult` with `prediction` + `reasoning`
- [ ] `beliefPath` present and well-formed (final step matches `prediction`)
- [ ] No direct provider SDKs / raw sockets / HTTP except via `ctx.call_provider`
- [ ] Imports stay within the runner allowlist (stdlib + packages below)

Contributor-oriented gateway and sandbox details: `tests/README.md`.

### 5. Submit

With `GATEWAY_API_KEY` in `.env`:

```bash
python3 miner/cli.py submit-agent path/to/agent.py \
  --wallet-name <wallet-name> \
  --wallet-hotkey-name <hotkey-name>
```

Prefer `.env` for the API key over `--gateway-api-key` (shell history).
---

## Allowed dependencies

Validators run miner agents inside the `almanacai/agent-runner` image with
`--network=none`. The agent can only import what that image ships.

Curated third-party packages (`docker/runner-requirements.txt`):

- `httpx`
- `pydantic`
- `python-dotenv`

Plus the Almanac agent/runtime packages baked into the image (`src.agent`,
`src.core`, gateway client helpers used by the runner).

**Do not** import:

- Provider SDKs (`anthropic`, `openai`, `google-generativeai`, …) — call
  `ctx.call_provider` instead; the gateway holds API keys.
- Chain libraries (`bittensor`, `web3`, …) — not present in the sandbox.
- Arbitrary network clients or OS process tools aimed at escaping the sandbox.

If an import is not on the runner image, the agent will fail at runtime even if
it works on your laptop.

## Belief path (required)

Every `AgentResult` must include a `beliefPath`: the ordered trajectory of your
agent's probability as it reasons, from an optional prior through updates to a
final belief. It is **required** and validated at the sandbox boundary. It is
stored in the trace but **not scored in MVP** (phase-2 grounding signal). An
agent that omits it, or returns an ill-formed one, is **rejected**.

`BeliefStep` fields (no extra fields allowed):

| Field | Type | Required | Notes |
|---|---|---|---|
| `step` | `int` (≥ 0) | yes | Position in the path, 0-based. |
| `type` | `"prior" \| "update" \| "final"` | yes | The path must end with exactly one `final`. |
| `probability` | `float` (0.0–1.0) | yes | Your belief at this step. |
| `text` | `str` (non-empty) | yes | Why the belief is where it is. |
| `usedCall` | `str \| null` | no | Provider-call id this step used. Phase-2; not verified. |
| `usedSources` | `list[int] \| null` | no | Source indices this step used. Phase-2; not verified. |

Validation rules (enforced by Pydantic; agent rejected on failure):

- `beliefPath` has at least one step.
- Exactly one step is `type: "final"`, and it is the last step.
- The final step's `probability` equals `prediction` (to 4 decimal places).

Minimal valid agent — a single `final` step is the trivial valid path:

```python
from src.core.schemas import AgentResult, BeliefStep

return AgentResult(
    prediction=0.42,
    reasoning="Base rate ~0.4, no strong signal either way.",
    beliefPath=[BeliefStep(step=0, type="final", probability=0.42, text="Base rate, held.")],
)
```

Evidence-linked path:

```python
return AgentResult(
    prediction=0.19,
    reasoning="Sonar search showed the strait open and traffic normal; moved down from 0.5.",
    beliefPath=[
        BeliefStep(step=0, type="prior",  probability=0.50, text="No evidence yet; even odds."),
        BeliefStep(step=1, type="update", probability=0.19, text="Strait open, traffic normal.",
                   usedCall="c2", usedSources=[0, 3]),
        BeliefStep(step=2, type="final",  probability=0.19, text="Hold at 0.19."),
    ],
)
```

Links are stored **as declared**; they are not verified in MVP.

## Calling providers

Use `ctx.call_provider(provider_id, call_type, params)` for external data and
LLM APIs. Every key in `params` is forwarded as-is through the validator
gateway — **the platform does not inject defaults** for LLM sampling or token
limits. If you omit a param, it is not sent upstream.

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

Production validators support the provider IDs above. The miner `test-agent`
route currently exposes only the IDs announced by the portal provider catalog;
see [Test locally before submit](#4-test-locally-before-submit).

Provider-specific params (`system`, `tools`, `grounding`, …) are also
pass-through when the adapter supports them. See example agents and
`src/gateway/providers/`.

### LLM params you must define

Only params you set in agent code are forwarded. Calibrate per provider/model —
some models reject certain sampling params (for example newer Claude models may
400 on `temperature` or `top_p`).

| Param | Type | Description |
|---|---|---|
| `model` | `str` | Model ID for the provider (required for LLM calls). |
| `messages` | `list[dict]` | Chat messages, typically `[{"role": "user", "content": "..."}]`. |
| `max_tokens` | `int` | Maximum tokens in the completion. |
| `temperature` | `float` | Randomness; often `0.1`–`0.3` for forecasting. Omit if the model rejects it. |
| `top_p` | `float` | Nucleus sampling. Prefer one of temperature / top_p on Claude 4+. |
| `stop` | `list[str]` | Stop sequences. |

Anthropic-only (with `anthropic` / `messages`):

| Param | Type | Description |
|---|---|---|
| `system` | `str` | System prompt. |
| `tools` | `list[dict]` | Tool definitions (for example web search). |

---

## Configuration

Defaults target production. Override only when you intentionally need something else.

| Setting | Default | Override (optional) |
|---|---|---|
| Orchestrator base URL | production (`ORCHESTRATOR_API_URL` in `src/core/constants.py`) | `FORECASTING_ORCHESTRATOR_URL` or `--orchestrator-url` |
| Gateway API key | — (required for test and submit) | `GATEWAY_API_KEY` (or `FORECASTING_GATEWAY_API_KEY` / `--gateway-api-key`) |
| Timeout | `20.0` s (`120.0` s for `test-agent`) | `FORECASTING_TIMEOUT_SECONDS` / `--timeout-seconds` |
| Wallet path | `~/.bittensor/wallets` | `FORECASTING_WALLET_PATH` / `--wallet-path` |
| Wallet name | `default` | `FORECASTING_WALLET_NAME` / `--wallet-name` |
| Hotkey name | `default` | `FORECASTING_WALLET_HOTKEY` / `--wallet-hotkey-name` |

The CLI auto-loads repo-root `.env` (`python-dotenv`) before resolving env vars.

## Command reference

### `test-agent`

Fetch a live event, run one local agent file through the portal gateway, and
validate its `AgentResult`:

```bash
python3 miner/cli.py test-agent path/to/agent.py
python3 miner/cli.py test-agent path/to/agent.py --event-id <portal-event-id>
```

This command incurs normal portal provider usage and reports per-call cost and
the remaining portal balance.

### `submit-agent` (alias: `upload-agent`)

`POST /v1/agents/submit-agent` with the raw `.py` file bytes as the body.

```bash
python3 miner/cli.py submit-agent path/to/agent.py \
  --wallet-name <wallet-name> \
  --wallet-hotkey-name <hotkey-name>
```

Request shape:

- Content type: `application/octet-stream`
- Header: `x-agent-filename: <agent file name>`
- Header: `Authorization: Bearer <gateway_api_key>`
- Signed headers: `x-miner-hotkey`, `x-miner-signature`, `x-miner-nonce`,
  `x-miner-timestamp`

Signing:

- CLI loads the miner hotkey from local Bittensor wallet files.
- Canonical message domain `sub41-agent-v1`: method, path (+ query), subject
  hotkey, nonce, timestamp (ms), sha256(body).
- Hex signature sent as `x-miner-signature`.
- Org / gateway account id are not part of the signed payload.

Compatibility flags (retained; currently ignored by the submit API):

- `--miner-uid`
- `--subtensor-network`
- `--netuid`

Optional override: `--miner-hotkey` (defaults to wallet hotkey ss58).

Client-side checks: file exists, `.py` extension, max size `2MB` (`2097152` bytes).

`get-agent` is intentionally not in this CLI (validator-facing route).

### `buy-credits` (stub)

Reserved for a future credits flow; not implemented yet.

```bash
python3 miner/cli.py buy-credits 25
```

## Troubleshooting

- **Connection errors**: confirm network access to the default production
  orchestrator. Only set `FORECASTING_ORCHESTRATOR_URL` / `--orchestrator-url`
  if you are intentionally using a non-default host.
- **Missing API key**: set `GATEWAY_API_KEY` in `.env` (or pass
  `--gateway-api-key`). Obtain a key at [portal.almnc.ai](https://portal.almnc.ai).
- **Key already linked**: request a fresh gateway API key for this miner.
- **Wallet load errors**: check wallet path / name / hotkey name and that hotkey
  files exist under `~/.bittensor/wallets`.
- **Upload file errors**: path must exist and end in `.py`; size ≤ 2MB.
- **Provider unavailable during `test-agent`**: use a provider returned by the
  portal catalog. Search-capable models can be called through `openrouter`.
- **Import / sandbox failures after submit**: remove disallowed deps; use only
  `ctx.call_provider` for network I/O (see [Allowed dependencies](#allowed-dependencies)).

# Almanac Market Miner Guide

For **Almanac Market** (trading / on-chain metadata), use a different track:

```bash
python3 scripts/run_market_miner.py
```

See `miner/market/miner.py --help` and [almanac.market](https://almanac.market)
for Market account setup. The rest of this guide is **Forecasting only**.