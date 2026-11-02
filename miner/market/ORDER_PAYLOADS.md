# Building Almanac Order Payloads Without the Python SDK

For traders writing their own client (Go, Rust, Java, …). `api_trading.py` uses `polymarket-client` to sign; this page is what that SDK does, written down so you can do it in any language. The Almanac API receives plain JSON and never sees your private key. Full field reference: [almanac.market docs → Trading API](https://docs.almanac.market/api/v1/trading).

Every value on this page was checked three ways: `polymarket-client` 0.14.0's own signer, an independent re-implementation from the raw EIP-712 type strings, and the Almanac backend's order-hash code.

## Flow

| Step | Call | You sign |
|---|---|---|
| 1 | `GET /api/wallet/proxy/:eoa` → `proxyAddress`, `walletKind`, `signatureType`, `deployed` | — |
| 2 | CLOB API key (once): EIP-712 `ClobAuth` → `POST https://clob.polymarket.com/auth/api-key` | EIP-712, EOA |
| 3 | `POST /api/v1/trading/sessions` with the five-line message | EIP-191 `personal_sign`, EOA |
| 4 | `GET /api/v1/trading/config` (builder codes); CLOB `GET /tick-size?token_id=`, `GET /neg-risk?token_id=` | — |
| 5 | Build and sign the `Order` struct | EIP-712, EOA (see Signing) |
| 6 | `POST /api/v1/trading/orders`; reconcile on `GET /orders/:hash` when told to | — |

Chain id **137**. Base URL `https://api.almanac.market/api`. Your wallet must be `deployed: true` — deploy once by logging in at almanac.market.

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

Domain: `name = "Polymarket CTF Exchange"`, `version = "2"`, `chainId = 137`, `verifyingContract` =
`0xE111180000d2663C0091e4f400237545B87B996B` when `neg_risk` is `false`, `0xe2222d279d744050d28e00520010520000310F59` when `true`.

| Field | Value |
|---|---|
| `salt` | random integer `0 ≤ salt < 2^53` (never reuse) |
| `maker` | trading wallet (`proxyAddress`) |
| `signer` | deposit wallet: `proxyAddress`; Safe: your EOA |
| `tokenId` | outcome token id (uint256) |
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
| `403 MIGRATION_REQUIRED` | Safe trading has ended; migrate on the web once |

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

**Auth** (same EOA)

```
ClobAuth, timestamp "1762000000", nonce 0
  0x21795045165099dc0c4eb96f6f31ed950af458623635f061009dbb90d2a9776711d984eaf63252c2ad53ae01e6db07160da647abb95d14a03d345a2773b743971b
Session message "Create Almanac trading session\nwallet:0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266\nchainId:137\nnonce:0123456789abcdef0123456789abcdef\ntimestamp:1762000000"
  0xa956365c91f23560c82c2fcf3e104eefc959ed4bcc62f57cc0c18574bd17bd133dce3216c66ddc0d3623bf896c1484bab0714da63bbb96b228c0032c83a4bccd1b
```

Reproduce vector A's wire signature from the key and the order fields and every step is right.

## Go hints

go-ethereum `signer/core/apitypes`: put `EIP712Domain`, `Order`, `TypedDataSign` in `Types`, `PrimaryType: "TypedDataSign"`, nested `contents` map for the order; `TypedDataAndHash` gives the digest. `uint256` fields accept `*big.Int` or decimal strings; `bytes32` are `0x` hex strings. Sign with `crypto.Sign` and add 27 to byte 64. Append the trailer as raw bytes (`binary.BigEndian.PutUint16` for the length). Use `math/big` or a decimal package for amounts — `float64` produces off-grid prices.

Not covered: Protocol V2 position ids (domain version `"3"`, Exchange V3) — unused by Almanac markets today; session-key signers; gasless deploy/approve/claim/withdraw (done once in the web app).
