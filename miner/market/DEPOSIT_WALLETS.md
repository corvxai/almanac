# CLI Update — Deposit Wallets (V2)

_Branch `v2-deposit-wallets` (extends `mkt_api_trading_updates`). Applies to `miner/market/api_trading.py`, launched via `python3 scripts/run_api_trading.py`._

Polymarket has retired legacy Gnosis Safe proxy wallets. Almanac is moving every account to a **deposit wallet** — a per-user smart-contract wallet (ERC-1967) that validates your existing EOA key via ERC-1271. This is a **hard cutover**: after the cutoff date, order placement from a Safe wallet returns `403 MIGRATION_REQUIRED`. Cancels, withdrawals, and claims keep working indefinitely, so funds are never locked.

**Your EOA private key does not change.** It still signs every order and every relayer batch — only the wallet that holds funds (and the order struct fields) changes.

## What you must do (once)

1. **Pause your bot / exit the CLI.** Almanac allows one active session per address; the web login below kills your CLI session.
2. **Migrate in the web app** at [almanac.market](https://almanac.market) — log in with the same EOA key the CLI uses and click **Upgrade wallet**. The guided flow cancels open orders, deploys the deposit wallet (gasless), sets approvals, and moves your pUSD + positions in one atomic transaction. There is no headless migration API. Your trade history, scoring, and leaderboard standing move with you.
3. **Restart the CLI** and create/refresh the trading session. That's it — everything below happens automatically.
4. Optional: update `EOA_PROXY_FUNDER` in `api_trading.env` to your deposit-wallet address (or leave it blank; the session value is authoritative).

## What the CLI now does automatically

### 1. Session tells you where you stand

Session creation/refresh prints the resolved wallet kind and warns if migration is pending:

```
Trading session created successfully.
Deposit wallet: 0xf07D...04E9 (signatureType 3)
```

or, before you migrate:

```
Safe wallet (legacy): 0x2D1c...966a (signatureType 2)
⚠️  Safe → deposit-wallet migration required before 2026-09-01T00:00:00.000Z.
   Migrate once in the web app at https://almanac.market. The web login ends this
   CLI session — use 'Refresh Trading Session' afterwards to pick up the deposit wallet.
```

The session response now carries `walletKind`, `signatureType`, and a `migration` object — the CLI reads all three (`_session_wallet_kind()`, `_print_session_wallet_summary()`).

### 2. Orders sign with signatureType 3

For a deposit session, orders are built with `signature_type=3` (POLY_1271) and `funder=<deposit address>`; the SDK then sets **`maker` = `signer` = deposit-wallet address** while your EOA key produces the signature (validated on-chain via ERC-1271):

| Field | Safe (before) | Deposit wallet (now) |
|---|---|---|
| `maker` | Safe address | deposit-wallet address |
| `signer` | your EOA | deposit-wallet address |
| `signatureType` | `2` | `3` |

Safe sessions keep signing exactly as before until you migrate (the sig-1/sig-2 retry survives for legacy backends). After the cutoff, a Safe order gets a clear message instead of a raw error dump:

```
Order rejected: Safe-wallet trading has ended (MIGRATION_REQUIRED).
  Cutoff: 2026-09-01T00:00:00.000Z
  Migrate once in the web app at https://almanac.market, then use
  'Refresh Trading Session'. Cancels, withdrawals, and claims still work.
```

### 3. Per-side builder codes from the API

Builder codes are now fetched at runtime from the public endpoint `GET /api/v1/trading/config` (cached 60s) instead of a hardcoded constant:

- **BUY** orders sign `builderCodes.buy` (the 1% fee profile — Polymarket deducts at fill)
- **SELL** orders sign `builderCodes.sell` (zero-fee profile, attribution only)

The backend rejects mismatched codes (`INVALID_BUILDER_CODE` / `BUILDER_CODE_NOT_ALLOWED_ON_SELL`), so this is required for sells to keep working after the release. Against a pre-cutover backend (endpoint not deployed) the CLI falls back to the legacy constant on both sides — identical to old behavior.

### 4. Claims route through the deposit batch

`Claim Polymarket Proceeds` still does: data-api plan → `POST /v1/redeem` → execute returned calldata via the Polymarket relayer. What changed:

- **Deposit sessions** submit ONE `execute_deposit_wallet_batch` (single EOA signature over the whole batch). The backend routes deposit claims through Polymarket's **CtfCollateralAdapter**, which redeems + wraps + pays **pUSD in one call** — so `selectedCollateral` is pUSD and the old USDC.e auto-wrap correctly self-disables.
- **Safe sessions** keep the byte-identical legacy Safe multicall path.
- The auto-wrap fallback (for any USDC.e residue) is now a **single combined [approve + wrap] batch**. Split approve→wrap batches were the documented deposit-wallet failure mode: the follow-up wrap-only batch dies at the relayer without broadcasting.

### 5. Withdrawals and deposits are kind-aware

- **Withdraw pUSD** executes through the Safe multicall or the deposit batch depending on the session's wallet kind. Deposit sessions trust the server-derived `proxyWallet` (no client-side Safe derivation).
- **Deposit funds** already targets the session's trading wallet, which after migration IS the deposit address — menu labels now say "trading wallet" and print the kind so you always know where funds are going.

## Requirements

```bash
pip install -r requirements.txt   # adds: web3, py-builder-relayer-client, eth-account
```

`py_clob_client_v2 >= 1.0.1` (POLY_1271 support) and `py-builder-relayer-client >= 0.0.2` (deposit-wallet batch + Polygon deposit contract config) — both satisfied by a fresh install.

## Example session (migrated wallet)

```
$ python3 scripts/run_api_trading.py
...
No active trading session detected. Creating one now...
Trading session created successfully.
Deposit wallet: 0xf07DA1B02CD0B6bF87d9cC4609Ceb8D570c704E9 (signatureType 3)

Trading Menu:
  1) Search and Trade Markets        ← BUY signs builderCodes.buy, SELL signs builderCodes.sell
  2) See Positions
  3) See Orders
  4) Claim Polymarket Proceeds      ← one deposit batch, pUSD lands directly
  5) Funds (balance / deposit / withdraw)
  6) Refresh Trading Session
  7) Back to Main Menu
```

Placing a BUY from a deposit session produces a signed order with:

```json
{
  "maker":  "0xf07DA1B02CD0B6bF87d9cC4609Ceb8D570c704E9",
  "signer": "0xf07DA1B02CD0B6bF87d9cC4609Ceb8D570c704E9",
  "signatureType": 3,
  "builder": "<builderCodes.buy from /v1/trading/config>"
}
```

Claiming proceeds:

```
Checking for claimable proceeds...
Amount available to claim: $12.34
...
Request claim transactions from Almanac? [y/N]: y

Submitting redemption transaction(s)...
  (deposit-wallet batch: 2 call(s), one EOA signature)
Waiting for confirmation...
Polygon transaction hash: 0x…
Claim succeeded (relayer confirmed on-chain).
```

## For bot authors using the API directly

The same rules apply outside this CLI — the full contract (order-struct diff, error codes, checklist) is in the Almanac docs: **Guide → Migrating from Safe Wallets to Deposit Wallets**. Short version:

```python
import requests

API = "https://api.almanac.market/api"

# 1) Session — read the wallet kind, never cache the old proxyWallet
data = requests.post(f"{API}/v1/trading/sessions", json={...}).json()["data"]
funder = data["proxyWallet"]            # deposit wallet after migration
assert data["walletKind"] == "deposit"  # else: migrate on the web first

# 2) Runtime config — builder codes + cutoff schedule (public, 60s cache)
cfg = requests.get(f"{API}/v1/trading/config").json()["data"]
buy_code, sell_code = cfg["builderCodes"]["buy"], cfg["builderCodes"]["sell"]

# 3) Orders — py_clob_client_v2 does the maker/signer swap for you
from py_clob_client_v2 import ClobClient, OrderArgs
client = ClobClient(host="https://clob.polymarket.com", chain_id=137,
                    key=EOA_PRIVATE_KEY, funder=funder, signature_type=3)
order = client.create_order(OrderArgs(token_id=token_id, price=0.55, size=10,
                                      side="BUY", builder_code=buy_code), options)
# → maker = signer = funder (deposit address), signatureType = 3, EOA-signed
```

## Files changed on this branch

| File | Change |
|---|---|
| `miner/market/api_trading.py` | session wallet-kind helpers + migration warnings; runtime `/v1/trading/config` fetch (60s cache); side-aware builder codes; sig-type 3 order path; `MIGRATION_REQUIRED` handling; `_execute_relay_batch` (Safe multicall vs deposit batch, delegatecall guard); claim/withdraw/auto-wrap routed through it; combined single-batch auto-wrap; kind-aware menu labels |
| `miner/market/api_trading.env.example` | `EOA_PROXY_FUNDER` documented as offline fallback (deposit address after migration) |
| `miner/market/DEPOSIT_WALLETS.md` | this document |
