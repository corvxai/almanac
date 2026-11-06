# Building Almanac Order Payloads Without the Python SDK

For traders writing their own client (Go, Rust, Java, …). `api_trading.py` uses `polymarket-client` to sign; this page is what that SDK does, written down so you can do it in any language. The Almanac API receives plain JSON and never sees your private key. Full field reference: [almanac.market docs → Trading API](https://docs.almanac.market/api/v1/trading).

Every value on this page was checked three ways: `polymarket-client` 0.16.0's own signer (byte-identical to 0.14.0's), an independent re-implementation from the raw EIP-712 type strings, and the Almanac backend's order-hash code.

## Flow

| Step | Call | You sign |
|---|---|---|
| 1 | `GET /api/wallet/proxy/:eoa` → `proxyAddress`, `walletKind`, `signatureType`, `deployed` | — |
| 2 | CLOB API key (once): EIP-712 `ClobAuth` → `POST https://clob.polymarket.com/auth/api-key` | EIP-712, EOA |
| 3 | `POST /api/v1/trading/sessions` with the five-line message | EIP-191 `personal_sign`, EOA |
| 4 | `GET /api/v1/trading/config` (builder codes, `protocolV2.tradingEnabled`); read the market (`GET /api/markets/:id`) for `version` and the id arrays; CLOB `GET /tick-size?token_id=`, and `GET /neg-risk?token_id=` on a v1 market | — |
| 5 | Build and sign the `Order` struct | EIP-712, EOA (see Signing) |
| 6 | `POST /api/v1/trading/orders`; reconcile on `GET /orders/:hash` when told to | — |

Chain id **137**. Base URL `https://api.almanac.market/api`. Your wallet must be `deployed: true` — deploy once by logging in at almanac.market.

**Which id you sign.** Almanac serves markets from two Polymarket ledgers. Read the market's `version`: `"v2"` means a Protocol V2 market and the `tokenId` is `position_ids[outcomeIndex]`; anything else (`"v1"`, missing, blank) is a CTF market and the `tokenId` is `clob_token_ids[outcomeIndex]` (an array, or a JSON string of one — parse before indexing). `version` is the only selector: many v1 markets also carry `position_ids` (not tradable), and on V2 markets Polymarket currently copies `position_ids` into `clob_token_ids`, so neither presence nor shape tells you anything. See [Protocol V2 markets](#protocol-v2-markets) for what else differs.

**Step 2 detail.** Domain `{ name: "ClobAuthDomain", version: "1", chainId: 137 }` (no `verifyingContract`); type `ClobAuth(address address,string timestamp,uint256 nonce,string message)`; message `{ address: <EOA>, timestamp: "<unix seconds>", nonce: 0, message: "This message attests that I control the given wallet" }`. Send headers `POLY_ADDRESS`, `POLY_SIGNATURE`, `POLY_TIMESTAMP`, `POLY_NONCE: 0`. Response `{ apiKey, secret, passphrase }`; `GET /auth/derive-api-key` with the same headers returns the same key later.

**Step 3 detail.** Sign exactly, lines joined by `\n`, no trailing newline:

```
Create Almanac trading session
wallet:<EOA>
chainId:137
nonce:<16–64 hex chars, fresh>
timestamp:<unix seconds, ±300 s of server time>
```

Body: `{ signature, message, walletAddress: <EOA>, apiCredentials: { apiKey, secret, passphrase } }`. Use `data.sessionId` / `data.walletAddress` as `x-session-id` / `x-wallet-address` headers afterwards.

## The `Order` struct

Type string (exact, no spaces):

```
Order(uint256 salt,address maker,address signer,uint256 tokenId,uint256 makerAmount,uint256 takerAmount,uint8 side,uint8 signatureType,uint256 timestamp,bytes32 metadata,bytes32 builder)
```

Domain: `name = "Polymarket CTF Exchange"`, `chainId = 137`, and then by market `version`:

| Market `version` | `version` (domain) | `verifyingContract` |
|---|---|---|
| v1, `neg_risk` false | `"2"` | CTF Exchange `0xE111180000d2663C0091e4f400237545B87B996B` |
| v1, `neg_risk` true | `"2"` | NegRisk CTF Exchange `0xe2222d279d744050d28e00520010520000310F59` |
| v2 (any `neg_risk`) | `"3"` | Exchange V3 `0xe3333700cA9d93003F00f0F71f8515005F6c00Aa` |

The `Order` type string, field encoding, amounts, and both signing schemes below are the same on every row; only the domain changes.

| Field | Value |
|---|---|
| `salt` | random integer `0 ≤ salt < 2^53` (never reuse) |
| `maker` | trading wallet (`proxyAddress`) |
| `signer` | deposit wallet: `proxyAddress`; Safe: your EOA |
| `tokenId` | the outcome's trading id (uint256): `clob_token_ids[i]` on a v1 market, `position_ids[i]` on a v2 market |
| `makerAmount` | BUY: USDC paid; SELL: shares given — 6-decimal integer units |
| `takerAmount` | BUY: shares received; SELL: USDC received — 6-decimal integer units |
| `side` | `0` BUY, `1` SELL |
| `signatureType` | `3` deposit wallet, `2` Safe (nothing else is accepted) |
| `timestamp` | unix **milliseconds** |
| `metadata` | 32 zero bytes |
| `builder` | `builderCodes.buy` for BUY; `builderCodes.sell` or 32 zero bytes for SELL |

`expiration` (`0`, or unix seconds for GTD) and `orderType` (`GTC` `GTD` `FOK` `FAK`) are sent in the JSON but are **not** in the signed struct.

**Amounts (limit order), decimal arithmetic only.** Per tick size: `0.1` → price 1 dp / amount 3 dp; `0.01` → 2 / 4; `0.005` and `0.001` → 3 / 5; `0.0025` and `0.0001` → 4 / 6; shares always 2 dp.
`shares = round_down(size, 2)`; `usdc = shares × price`, and if it has more than *amount dp*: round up to *amount dp + 4*, then if still too long round down to *amount dp*. BUY: `makerAmount = usdc·10⁶`, `takerAmount = shares·10⁶`; SELL swapped. Example, tick 0.01, BUY 17.36 @ 0.58 → `10068800` / `17360000`. BUYs pay a 1% builder fee at fill on top of `makerAmount`.

FOK/FAK use the same struct; size them from the limit math with your price cap as `price`, and test small — the SDKs add book- and fee-aware rounding there that is not pinned by the vectors below.

## Signing

**Safe (`signatureType` 2).** Plain EIP-712: sign `keccak256(0x1901 ‖ domainSeparator ‖ hashStruct(Order))`; 65-byte `r‖s‖v`, `v ∈ {27,28}`.

**Deposit wallet (`signatureType` 3) — ERC-7739 nested typed data.** The EOA signs a wrapper, not the order digest, and the wire signature carries a trailer.

Typed data: domain = the **exchange** domain above; `primaryType = "TypedDataSign"`; types `Order` (as above) and

```
TypedDataSign(Order contents,string name,string version,uint256 chainId,address verifyingContract,bytes32 salt)
```

message `{ contents: <Order>, name: "DepositWallet", version: "1", chainId: 137, verifyingContract: <deposit wallet>, salt: 0x00…00 }`. Any standard EIP-712 library (go-ethereum `apitypes`, ethers, viem, eth_account) hashes this correctly given the nested struct.

Wire `signature` = 317 bytes:

| Offset | Len | Content |
|---|---|---|
| 0 | 65 | EOA signature of the `TypedDataSign` digest |
| 65 | 32 | exchange domain separator |
| 97 | 32 | `hashStruct(Order)` |
| 129 | 186 | the `Order(...)` type string, ASCII |
| 315 | 2 | `0x00ba` (186, big-endian uint16) |

## Request

```http
POST /api/v1/trading/orders
x-session-id: <sessionId>
x-wallet-address: <walletAddress from the session>
Content-Type: application/json

{
  "marketId": "<Almanac market id>",
  "orderType": "GTC",
  "signedOrder": {
    "salt": "1234567890123",
    "maker": "0xBc0fF067b7740Eff76C1ca93c875Ba6B890d6B50",
    "signer": "0xBc0fF067b7740Eff76C1ca93c875Ba6B890d6B50",
    "tokenId": "71321045679252212594626385532706912750332728571942532289631379312455583992563",
    "makerAmount": "10068800",
    "takerAmount": "17360000",
    "side": "BUY",
    "signatureType": 3,
    "timestamp": "1762000000000",
    "expiration": 0,
    "metadata": "0x0000000000000000000000000000000000000000000000000000000000000000",
    "builder": "<builderCodes.buy from /config>",
    "signature": "0x<317 bytes>",
    "orderType": "GTC"
  }
}
```

JSON types: `salt`/amounts/`timestamp` decimal strings; `tokenId` string only; `side` `"BUY"`/`"SELL"`; `signatureType` and `expiration` numbers; `orderType` on the order and/or beside it (equal if both); omit `postOnly` unless `true` (GTC/GTD); no `taker`. `maker` must be the session's trading wallet, `signer` that wallet or the session EOA (`403 ORDER_WALLET_MISMATCH`).

Success: `{ "success": true, "data": { "orderId": "0x…", "orderHash": "0x…", "status": "live" } }`. `orderId` is the `Order` hash — `keccak256(0x1901 ‖ domainSeparator ‖ hashStruct(Order))` under the exchange domain (for deposit wallets: the inner `Order`, not the wrapper) — so you can compute it before sending.

## Responses that are not a plain accept/reject

| Response | Do |
|---|---|
| `200` with `status: "delayed"` | poll `GET /api/v1/trading/orders/:hash` until `live` / `matched` / `cancelled` |
| `503 SUBMIT_TIMEOUT` / `503 SUBMIT_UPSTREAM_ERROR` + `orderHash` | **do not resubmit**; poll `GET /orders/:hash` (session headers, honour `Retry-After`; a `503` on the lookup means retry, not missing) for ~60 s, then check open orders/positions |
| `503 ORDER_NOT_SUBMITTED`, `400 ORDER_UNMATCHED` | nothing was placed; safe to resubmit |
| `400 INVALID_BUILDER_CODE` / `BUILDER_CODE_NOT_ALLOWED_ON_SELL` / `VALIDATION_ERROR` | fix the struct and re-sign (the builder code is inside the signature) |
| `403 MIGRATION_REQUIRED` | a Safe wallet. `details.reason` absent: Safe trading has ended (`details.cutoffDate`). `details.reason: "protocol_v2"`: a Safe wallet on a V2 market, returned before the cutoff too and in both `tradingEnabled` states. Either way: migrate to a deposit wallet on the web once |
| `403 PROTOCOL_V2_DISABLED` | a V2 order while `protocolV2.tradingEnabled` is `false` on `GET /v1/trading/config`. Refused before any CLOB call; nothing was placed. Treat the market as not yet tradable; v1 markets are unaffected |
| `400 INSUFFICIENT_BALANCE` on a V2 order | the CLOB reports a missing Exchange V3 allowance the same way as a missing balance. Check the four V2 approvals (below) before assuming funds |

## Protocol V2 markets

Polymarket Protocol V2 is a second position ledger beside the CTF. From its launch, new markets arrive as `version: "v2"`; existing CTF markets stay `"v1"` and nothing about them changes. For an order, V2 is one row in the domain table above plus a different id. Everything else on this page — the `Order` struct, amounts, both signature schemes, the request body, builder codes, order types, session and CLOB auth — is identical.

**Market fields.** `GET /api/markets/:id`, `GET /api/markets/events/:id`, and `GET /api/markets/search` serve `version` (`"v1"` / `"v2"`), `position_ids` (array of decimal strings, the tradable ids when `version` is `"v2"`), `resolution_status` (`"inactive"` / `"active"` / `"resolved"` on V2 markets, `null` on v1), and `clob_token_ids` as before. Select the id by `version` and nothing else:

```
tokenId = market.version == "v2" ? market.position_ids[i] : parse(market.clob_token_ids)[i]
```

**Domain.** Every V2 order, neg-risk or not, is signed under `version = "3"` with `verifyingContract` = Exchange V3 `0xe3333700cA9d93003F00f0F71f8515005F6c00Aa`. The `/neg-risk` lookup is irrelevant for V2 (the flag does not pick the exchange). For a deposit wallet the ERC-7739 wrapper is unchanged in shape; because its outer domain is the exchange domain, the 32-byte domain separator in the trailer is Exchange V3's (`0x466c6391…095b`, vector C). `orderId` is still the `Order` hash under that domain, so it is computable before sending. A structurally valid position id has bits 40–103 clear (`(x & (((1 << 64) − 1) << 40)) == 0`); the SDKs use that test to pick the domain, but you should pick it from `version` and only use the shape as a sanity check.

**Gate.** `GET /api/v1/trading/config` carries `protocolV2.tradingEnabled`. While it is `false`, any V2 order is answered `403 PROTOCOL_V2_DISABLED` before reaching the CLOB. Poll the config (it is public, cached 60 s) rather than signing and retrying.

**Approvals.** A deposit wallet needs four approvals beyond the seven CTF ones, or the CLOB rejects V2 orders for allowance (surfaced as `400 INSUFFICIENT_BALANCE`):

| Token | Call | Spender / operator |
|---|---|---|
| pUSD `0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB` | `approve(spender, max)` | Exchange V3 `0xe3333700cA9d93003F00f0F71f8515005F6c00Aa` |
| pUSD | `approve(spender, max)` | Router `0x12121212006e4CD160D18e3f00711DA5c3372600` |
| PositionManager `0x006F54F7f9A22e0000CC2AB60031000000ae9fEF` | `setApprovalForAll(operator, true)` | Exchange V3 |
| PositionManager | `setApprovalForAll(operator, true)` | Router |

Grant them once by logging in at almanac.market (the approvals panel sends the missing ones in one gasless batch). Check with the public `GET /api/wallet/approvals/<EOA>`: eleven entries for a deposit wallet, each `{ spender, label, approved, ledger }` with `ledger` `pusd` / `ctf` / `positionManager`; the four V2 entries are the ones whose `spender` is Exchange V3 or the Router. A Safe wallet gets seven entries and cannot trade V2 (`MIGRATION_REQUIRED`, reason `protocol_v2`).

**Balances.** V2 shares live in PositionManager (ERC-1155, id = position id), not in the CTF. If you refresh the CLOB balance-allowance cache yourself, use `asset_type=CONDITIONAL-V2` with `token_id=<position id>` for a V2 SELL; `COLLATERAL` for BUYs is unchanged. Almanac's `GET /v1/trading/positions` returns the position id in `token_id` with no `version` field — keep the market's `version` next to it.

**Claims.** `POST /v1/redeem` decides the ledger per `conditionId` from the market's `version`. For V2 it prepares Router `redeem(bytes31 conditionId, uint256 outcomeIndex, uint256 amount)` calls (`to` = the Router) that pay pUSD; they go through the same relayer batch as CTF claims. Conditions it cannot prepare come back in `skippedRedemptions[]` with a `code`: `not_resolved` (V2 market not resolved on-chain yet), `router_not_approved` (grant the approvals above), `version_unknown` (ledger undecided; refused rather than guessed), `no_balance`. When nothing can be prepared the response is `200` with `success: false` and a `userMessage`.

## Relayer calls: builder headers from Almanac

Orders never touch this. It is only for the gasless path — claims, withdrawals, wraps, deploys — which goes through Polymarket's relayer (`https://relayer-v2.polymarket.com`) and needs **builder HMAC headers** on `POST /submit`. The builder key is Almanac's, so you do not compute the HMAC; you ask Almanac for the four headers per request and forward them. `api_trading.py` does this in `AlmanacRelayBuilderConfig`.

```http
POST /api/v1/auth/builder-sign
x-session-id: <sessionId>
x-wallet-address: <walletAddress from the session>
Content-Type: application/json

{ "method": "POST", "path": "/submit", "body": "<the exact JSON string you will POST to the relayer>" }
```

Response (200): `{ "POLY_BUILDER_API_KEY", "POLY_BUILDER_TIMESTAMP", "POLY_BUILDER_PASSPHRASE", "POLY_BUILDER_SIGNATURE" }` — set all four as HTTP headers on the relayer request, with the body bytes **identical** to the `body` string you signed (`Content-Type: application/json`). The HMAC is over `timestamp ‖ method ‖ path ‖ body` and the relayer checks it against the raw body it receives, so serialise first, sign that string, send that string. One call per relayer request, immediately before sending; the relayer's timestamp window is short.

Rules (anything else is `400 INVALID_REQUEST`, a pair not listed is `403 FORBIDDEN`, no session is `401`):

| Field | Accepted |
|---|---|
| `method` | `GET` or `POST`, exact case |
| `path` | relayer path only — no host, no query string, no trailing `/`; allowed pairs: `POST /submit`, `GET /deployed`, `GET /v1/account/transactions/params`, `GET /v1/account/transactions/<lowercase uuid>` |
| `body` | `POST /submit`: a **string** containing one JSON object, ≤ 64 KiB. GET pairs: omit the key (or `""`) |
| `timestamp` | do not send; it is ignored. The server signs with its own clock and returns it as `POLY_BUILDER_TIMESTAMP` — your clock does not matter |

In practice only `POST /submit` needs headers: the relayer answers `GET /deployed`, `/nonce`, `/transaction?id=` without them (that is how the Python CLI works — a query string would be rejected by the rule above anyway). The GET pairs are allowlisted for clients that mirror Polymarket's JS `RelayClient`, which signs them.

The `/submit` envelope itself (Safe or deposit-wallet batch, signed by your EOA) is Polymarket's format, not Almanac's; take it from `py-builder-relayer-client` (`builder/safe.py`, `builder/deposit_wallet.py`) or `@polymarket/client`. Almanac does not inspect it beyond the rules above.

## Test vectors

Hardhat/Anvil account #0 (public key material — never fund it), fixed salt/timestamp, placeholder builder `0x11…11`.

```
EOA key        0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
EOA            0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266
Deposit wallet 0xBc0fF067b7740Eff76C1ca93c875Ba6B890d6B50
Safe           0xd93B25cb943D14d0d34FBaF01Fc93a0f8b5F6E47
Exchange       0xE111180000d2663C0091e4f400237545B87B996B   (neg_risk false)

Order: salt 1234567890123 · tokenId 71321045679252212594626385532706912750332728571942532289631379312455583992563
       makerAmount 10068800 · takerAmount 17360000 · side 0 · timestamp 1762000000000
       metadata 0x00…00 · builder 0x1111111111111111111111111111111111111111111111111111111111111111

Exchange domain separator
  0x3264e159346253e26a64e00b69032db0e7d32f94628de3e6eecb50304d7af3d2
```

**A — deposit wallet, type 3** (`maker = signer = 0xBc0f…6B50`)

```
hashStruct(Order)              0x6678b0478d37813a337e1f17ae61605c637a4a16187d5f000556b7e11a6e0d61
Order hash = orderId           0x7a7ead30f5d3060ce6dc3e7fd32eacbf7b393596e99fed9bf56dd63d35237d7c
hashStruct(TypedDataSign)      0x68b2fcaa6a8589a4362082df5f29368f0f380c5c6311d4d3dbf004f7013ef704
Digest the EOA signs           0x030a619295d56dd55b0e37c15b2eaa2a55db834db6b43964a595dd6132324672
EOA signature (65 B)
  0x5668f90b71b38ca603d7e660744b20648aa73e657bba40a6d1961b73d36f4ca95e69565991f3443f51649f357fb8a5f6b74d26cd654c0c7d638bcc9eadb2af8b1c
Wire signature (317 B) = EOA sig ‖ 3264e1…f3d2 ‖ 6678b0…0d61 ‖ hex("Order(…)") ‖ 00ba
  0x5668f90b71b38ca603d7e660744b20648aa73e657bba40a6d1961b73d36f4ca95e69565991f3443f51649f357fb8a5f6b74d26cd654c0c7d638bcc9eadb2af8b1c3264e159346253e26a64e00b69032db0e7d32f94628de3e6eecb50304d7af3d26678b0478d37813a337e1f17ae61605c637a4a16187d5f000556b7e11a6e0d614f726465722875696e743235362073616c742c61646472657373206d616b65722c61646472657373207369676e65722c75696e7432353620746f6b656e49642c75696e74323536206d616b6572416d6f756e742c75696e743235362074616b6572416d6f756e742c75696e743820736964652c75696e7438207369676e6174757265547970652c75696e743235362074696d657374616d702c62797465733332206d657461646174612c62797465733332206275696c6465722900ba
```

**B — Safe, type 2** (`maker = 0xd93B…6E47`, `signer = 0xf39F…2266`)

```
Order hash = orderId           0x3cafab089125e8fb1c323bd3b605e571388191976f41fd694b25bb2a16475a55
Wire signature (65 B)
  0x291ca1a9469634926c2a18a01d0076a49b4dc0a49a31b4f0703804b7d26a8a8c44e28eaed4ba46ce81ef1eb3c2e15c9333bcd311a6cae03da42209e1ca220ec91b
```

**C — Protocol V2, deposit wallet, type 3** (`maker = signer = 0xBc0f…6B50`). Same order fields as A except `tokenId` = position id `496434284537833646782127667851621953528514251545071522230068564777834119168` (a real binary V2 market's YES id; `0x0118…` module 1, outcome byte `00`), domain `version "3"`, `verifyingContract` Exchange V3 `0xe3333700cA9d93003F00f0F71f8515005F6c00Aa`.

```
Exchange V3 domain separator  0x466c63910185bbd55e8679264200c4e0abdcbb0c6264eb3d41d13326022e095b
hashStruct(Order)             0x98dec4bbde276f7d9dd33192fa759f2cb758644d0d2d8d4d3f0c9d63763435b8
Order hash = orderId          0x6128bd509443ccc73f95f0ec5ed20c159b7d1352728f805bac6260702cceb761
EOA signature (65 B)
  0xe44385e6ef04e3e5068f1212650d9e09e4c0c17b7df08622d2642f398bf32ef257491d87af06e871a0785508d0619ff01ca2b071693f00853ac07458bd457a311b
Wire signature (317 B) = EOA sig ‖ 466c63…095b ‖ 98dec4…35b8 ‖ hex("Order(…)") ‖ 00ba
  0xe44385e6ef04e3e5068f1212650d9e09e4c0c17b7df08622d2642f398bf32ef257491d87af06e871a0785508d0619ff01ca2b071693f00853ac07458bd457a311b466c63910185bbd55e8679264200c4e0abdcbb0c6264eb3d41d13326022e095b98dec4bbde276f7d9dd33192fa759f2cb758644d0d2d8d4d3f0c9d63763435b84f726465722875696e743235362073616c742c61646472657373206d616b65722c61646472657373207369676e65722c75696e7432353620746f6b656e49642c75696e74323536206d616b6572416d6f756e742c75696e743235362074616b6572416d6f756e742c75696e743820736964652c75696e7438207369676e6174757265547970652c75696e743235362074696d657374616d702c62797465733332206d657461646174612c62797465733332206275696c6465722900ba
```

**D — Protocol V2, Safe, type 2** (`maker = 0xd93B…6E47`, `signer = 0xf39F…2266`, same V2 fields as C). Listed so a Safe implementation can check its V3 domain; the backend refuses this order with `MIGRATION_REQUIRED` (reason `protocol_v2`).

```
Order hash = orderId           0x223ef62f358d96fe63a801b60a3a0c1f693a5e9bdf686e7333e95f9e1d55d530
Wire signature (65 B)
  0xed38742eecf2ba5d7536f19ef0644354a1f7ab92c05f6d19e5fe2307525d4c28646c1eff4a1c10e4aa832c850aee5b1b90c5b4bfd42ccd8cfcf2d7aab28c99961c
```

**Auth** (same EOA)

```
ClobAuth, timestamp "1762000000", nonce 0
  0x21795045165099dc0c4eb96f6f31ed950af458623635f061009dbb90d2a9776711d984eaf63252c2ad53ae01e6db07160da647abb95d14a03d345a2773b743971b
Session message "Create Almanac trading session\nwallet:0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266\nchainId:137\nnonce:0123456789abcdef0123456789abcdef\ntimestamp:1762000000"
  0xa956365c91f23560c82c2fcf3e104eefc959ed4bcc62f57cc0c18574bd17bd133dce3216c66ddc0d3623bf896c1484bab0714da63bbb96b228c0032c83a4bccd1b
```

Reproduce vector A's wire signature from the key and the order fields and every step is right; reproduce C's and your Protocol V2 domain is right too (the only difference between A and C is the `tokenId` and the domain row).

## Go hints

go-ethereum `signer/core/apitypes`: put `EIP712Domain`, `Order`, `TypedDataSign` in `Types`, `PrimaryType: "TypedDataSign"`, nested `contents` map for the order; `TypedDataAndHash` gives the digest. `uint256` fields accept `*big.Int` or decimal strings; `bytes32` are `0x` hex strings. Sign with `crypto.Sign` and add 27 to byte 64. Append the trailer as raw bytes (`binary.BigEndian.PutUint16` for the length). Use `math/big` or a decimal package for amounts — `float64` produces off-grid prices.

Not covered: session-key signers; the relayer batch envelopes for gasless deploy/approve/claim/withdraw (only the Almanac builder-header step is documented above; the web app does these once, including the four Protocol V2 approvals).
