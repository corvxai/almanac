# Arcratio — Architecture & Production-Readiness Audit

A high-level audit of the current codebase against the goal of moving from
prototype to a robust foundation for a commercial service. Findings are
prioritised; each cites file:line so it can be expanded into a ticket. No
code changes were made for this audit.

---

## 1. Executive summary

Arcratio is split into two distinct pillars; only one of them lives in this
repo.

- **Pillar 1 (this repo + a separate scoring service):** Subnet ingestion
  and incentive mechanism. External operators run validators that execute
  agent code through the central provider gateway and produce sealed
  Evidence Digest traces. A separate service (not in this repo) ingests
  those traces, scores them, and pays validators from the IM. The
  "tenants" of pillar 1 are validators, not customers; the Bittensor
  hotkey is the right identity primitive. The Evidence Digest is internal
  training/scoring data, not a customer artefact.
- **Pillar 2 (separate codebase, not yet built):** Customer-facing API
  serving live probability predictions and calibrated forecasts with
  uncertainty. Consumes the outputs of pillar 1. Auth, multi-tenancy,
  billing, public schema versioning, SDKs all live in pillar 2.

The single biggest undecided architectural question is **how sealed traces
flow from validators back to the operator's ingestion side**. Today, traces
land only on each validator's local disk and never leave. Until that path
is designed, every other pillar 1 robustness question is premature.

The strongest parts of the codebase today are the three-actor decomposition
(validator / agent / gateway), the architectural enforcement that "the
agent cannot influence what gets recorded" in Docker-sandbox mode, and the
attack-suite-as-tests under `tests/security/`. Those should be preserved
through any refactor.

The weakest parts are: storage that is not crash- or concurrency-safe; a
trace integrity model that is half-realised (sealed but mutable, never
verified on read, with a split resolution story); a gateway that is
unauthenticated by default and has no per-validator metering; and
sandbox baseline flags that depend on `:latest` image trust and Docker's
default seccomp profile.

> **Second-pass update (2026-06-08).** This audit was re-run after the
> `main` + `origin/integration_updates` merges, which added two large
> surfaces the first pass did not cover: an **orchestrator API
> integration** (validator now polls assignments and submits predictions
> + full traces to a central orchestrator) and the **Almanac incentive
> mechanism** (a second IM merged from the sn41 codebase, with Postgres
> storage and dual-IM weight blending). The most consequential changes
> from this pass:
>
> - **Trace egress now exists.** C7 ("traces never leave the validator")
>   is reframed, not closed — validators now POST the full `EvidenceDigest`
>   to the orchestrator, but the path is lossy and runs over an unenforced
>   transport. See the revised C7, plus H9/H10.
> - **A plaintext DB credential is committed to the repo** —
>   `src/validator/almanac/storage/storage.env`. This is the single
>   highest-urgency item in the document. See **C9**.
> - **The Almanac Postgres store repeats C1's mistakes** for the new IM's
>   payout data (single reused connection, no pool, no rollback). See **C10**.
>
> A finding-by-finding status table for this pass is in **§12**.

> **Third-pass update (2026-06-10).** A further review was run after the
> `origin/assignment_pipeline` sync (which refactored the orchestrator
> assignment flow into `src/validator/assignment_pipeline.py`). It found a
> set of **new findings the first two passes missed** — including two
> payout-correctness bugs in the Almanac scoring path — and several were
> remediated in a non-breaking patch this pass. The most consequential:
>
> - **Two latent payout bugs in Almanac scoring (now fixed).** A
>   case-sensitivity bug falsely flagged legitimate miners as multi-profile
>   and zeroed their weight; an unguarded division crashed the entire
>   weight-setting step on a zero-fee epoch. See **N1** and **N2**.
> - **Gateway error responses leaked upstream provider bodies** verbatim to
>   callers (potential secret/identifier exposure), and validation errors
>   leaked schema internals. Both sanitised this pass. See **N3**.
> - **The leaked DB credential (C9) is confirmed still reachable in git
>   history** (`git show 67a949a:.../storage.env`); the scrub remains
>   outstanding.
>
> New findings and the remediation log for this pass are in **§13**.

---

## 2. Architecture as understood

### 2.1 Pillar 1 — what this repo contains

Three actors, one execution cycle per event:

- **Validator** (`src/validator/orchestrator.py`): receives an event,
  spawns the agent in a sibling Docker container, drains gateway-side
  call logs, assembles the trace, seals it, and writes it locally.
- **Agent** (`src/agent/`): a `BaseAgent` subclass running inside the
  sandbox container. Sees only a `ForecastingContext` whose only egress
  is a UDS socket to the validator-local signing proxy.
- **Provider gateway** (`src/gateway/`): in production, a central HTTP
  service holding upstream provider API keys. Validators reach it
  through a per-validator local proxy (`src/gateway/local_proxy.py`)
  that signs every call with the validator's Bittensor hotkey.

The sealed artefact is the `EvidenceDigest`
(`src/core/schemas.py:257-296`), containing execution context, event
snapshot, every provider call (with raw-response hash, extracted
evidence, and cost), the agent's reasoning chain, the prediction output,
and a SHA-256 trace hash.

### 2.2 Pillar 2 — what is intended but not in this repo

A separate service consumes pillar 1 outputs (scored predictions,
calibration history, possibly a derived model) and exposes a
customer-facing API serving live probability predictions and calibrated
forecasts with uncertainty. None of pillar 2 exists in the current
codebase.

### 2.3 The connection between pillars (partially built since first pass)

Three artefacts are needed to bridge pillar 1 → pillar 2:

1. **Trace egress** — a path for sealed traces to leave validator-local
   disk and reach the operator's ingestion side. **Now partially built**
   via the orchestrator API: the validator polls
   `GET /v1/validators/agent-and-event` for an assignment, runs the agent,
   and `POST`s the prediction plus the full `EvidenceDigest` to
   `/v1/validators/prediction` (`src/validator/orchestrator_api.py:17-19`).
   The path is real but lossy and transport-insecure — see the revised C7,
   H9, H10.
2. **Scoring contract** — the interface between the trace producer (this
   repo) and the scoring/IM consumer. **Now implicit** in the submit
   payload (`validator.py` `_build_prediction_submit_payload`, ~line 547+)
   and the `GET /v1/validators/scored-predictions` reader, but still
   undocumented and unversioned beyond an inline `schemaVersion: "1.0"`.
3. **Prediction publication** — once scored, predictions need to land
   somewhere pillar 2 can read with low latency. Still not designed.

Items 1 and 2 are no longer greenfield decisions — they are an existing
path to **harden** (idempotency, durable retry, TLS, a versioned contract)
rather than design from scratch. See Q1/Q2 below for the revised framing.

---

## 3. What's working well (preserve through any refactor)

1. **Three-actor decomposition is sound.** Validator orchestrates, agent
   predicts, gateway records. Boundaries are clean in the in-process
   case and architecturally enforced in Docker mode.
2. **"Agent cannot influence what gets recorded" is enforced in Docker
   mode.** The agent has only a UDS socket to the local proxy
   (`src/agent/sandbox_gateway.py`, `src/gateway/local_proxy.py:176-216`).
   The proxy is the sole producer of `ProviderCall` records; the agent
   return path (`AgentResult` in `src/core/schemas.py:214-227`) is
   deliberately minimal so it cannot inject evidence.
3. **Sandbox baseline flags are correct.** `network_mode="none"`,
   `read_only=True`, `cap_drop=["ALL"]`, `no-new-privileges`, non-root
   UID 10001, mem/cpu/pids limits, tmpfs `/tmp`
   (`src/validator/sandbox_docker.py:266-296`).
4. **Attack-suite-as-tests.** `src/agent/examples/_attacks/*.py` runs
   the sandbox against fork-bombs, memory hogs, raw sockets, DNS,
   urllib, /etc/passwd reads, wallet reads, and allowlist bypass; each
   is asserted as contained in
   `tests/security/test_sandbox_lockdown.py`. This is a real moat;
   keep growing it.
5. **Signing wire format is finalised.**
   `arcratio:v1:{netuid}:{ts}:{nonce}:sha256(body)` in
   `src/gateway/signing.py` is a clean canonical form that can be
   turned on without changing clients.
6. **Cost estimation is deterministic and externalised.**
   `config/pricing_cards.json` + `src/gateway/cost_estimator.py`
   separate pricing data from code, including cache-read /
   cache-creation rates.

---

## 4. Top-priority undecided design questions

These are paper decisions before they become tickets.

### Q1. How do sealed traces flow from validators to the operator's ingestion side?

Today there is no egress at all — traces land on validator-local disk
and stop. Options to pick between (each implies different threat
models, durability stories, and code shapes):

- New endpoint on the existing central gateway (`POST /v1/traces`)
  that validators sign and call after each run.
- A separate ingestion service with its own queue/storage, leaving
  the gateway as a call-time LLM proxy only — better blast-radius
  separation if the gateway is compromised.
- Validators anchor a trace hash on-chain; full digest is fetched
  off-chain. Makes the existing `merkle_root` / `anchor_tx` fields on
  `TraceIntegrity` (`src/core/schemas.py:244-246`) live.

### Q2. What is the contract with the separate scoring/IM service?

The scoring service consumes traces and emits scores. The interface
is now an internal-public API. Open questions:

- Format on the wire: full `EvidenceDigest`, or a normalised subset?
- Where does `ResolutionRecord` get populated, and by whom? Today
  it's split between the digest field
  (`src/core/schemas.py:265`) and the separate
  `results/{event_id}.json` file (`src/storage/json_store.py:53`)
  with no link between them.
- Does the scoring service produce scores into a separate table, or
  re-seal an enriched trace?
- Versioning: how does pillar 1 evolve the trace schema without
  breaking pillar 2?

Q2 should be answered before Q1 is implemented, because the contract
determines what the egress path needs to carry.

### Q3. What does the customer-facing API actually serve at query time?

The user has indicated **live probability predictions** and **calibrated
forecasts with uncertainty**. Two architectural shapes:

- **(a) On-demand subnet run per query.** Latency 10-60 s,
  full-LLM-cost per query. Premium / low-volume only.
- **(b) Cached prediction from a curated event registry.** Low
  latency, but only works for events the subnet has already
  predicted. Requires an event registry, a refresh policy, and a
  published-prediction store.

This repo today is shaped for (a) (orchestrator runs an agent on
demand). For (b), which is almost certainly the cheaper and more
scalable shape, there is no event registry, no prediction cache, and
no refresh scheduler. This affects what pillar 1 needs to *produce*: a
steady stream of predictions on a curated event set, not just ad-hoc
runs.

---

## 5. Critical findings

### C1. Storage layer is not crash-safe or concurrency-safe

**Status (2026-06-08): torn-write half addressed this pass.**
`save_trace`/`save_resolution` now write atomically via
`_atomic_write_text` (`src/storage/json_store.py`): temp file in the same
directory → `flush` + `os.fsync` → `os.replace`, so a power loss mid-write
leaves either the old or the new file, never a half-written JSON. Verified
by a functional round-trip + overwrite test. Still open: there is no
file lock / per-key serialisation, so two writers on the same
`execution_id` still race (last writer wins, but each write is now whole),
and `_iter_all_traces` still swallows corruption via
`except Exception: continue`.

Original finding (for the record): `path.write_text(...)` with no
`os.fsync`, no atomic `os.replace`, no lock. Two concurrent writers on the
same `execution_id` race; a power loss mid-write yielded a half-written
JSON file that `model_validate_json` rejects and that `_iter_all_traces`
silently skips. The "swappable interface" comment at the top
is true at the type level, but list operations
(`list_traces_by_event`, `list_traces_by_agent`) glob the entire
`traces/` directory and parse every file in memory — that's not swappable
in shape, it's a re-architecture.

Under the corrected framing this is **payout integrity**: if torn or
silently-dropped traces feed the IM scoring, validators are paid against
incomplete data.

### C2. Trace integrity model is half-realised

**Status (2026-06-08): read-time verification added this pass.**
`JsonTraceStore.get_trace` now calls `digest.verify_integrity()` on read
and logs a warning when a loaded trace's sealed hash does not match its
contents (tampered on disk or mutated after seal). It is intentionally
non-fatal — the digest is still returned so callers choose how to react —
so it is a detection signal, not yet enforcement. Verified by a
tamper-on-disk test. Remaining open items below.

`EvidenceDigest.seal()` (`src/core/schemas.py:281-290`) computes a SHA-256
over the digest with `trace_hash` zeroed, but:

- The model is **not frozen** (`model_config` omitted) — `resolution_record`
  and any other field can be mutated in memory after seal, silently
  invalidating the hash. (Freezing is deferred: resolution is written to
  the digest after seal in places, so `frozen=True` needs a resolution-flow
  change to avoid breakage — see the resolution-duality point below.)
- Read-time verification now *detects* mismatches (above) but does not
  *reject* them; a tampered file still loads (with a warning).
- `resolution_record` is **inside** the sealed digest with a default
  factory, but resolution is persisted **separately** at
  `results/{event_id}.json` (`src/storage/json_store.py:53`). Two sources
  of truth for the same fact, with no link between them and no story
  for "trace is sealed-once but resolution arrives later". This is
  exactly the spot where the scoring contract (Q2) needs to land.
- `merkle_root` and `anchor_tx` fields exist on `TraceIntegrity`
  (`src/core/schemas.py:244-246`) but nothing in the codebase populates
  them. Either ship the chain anchor or remove the fields.

### C3. Gateway is unauthenticated by default

`src/gateway/server.py:54-88`: signature verification is gated on
`REQUIRE_SIGNATURE` env var which defaults to **off**. There is no API
key, OAuth, mTLS, rate limit, quota, or per-caller usage cap on
`/v1/call`. The README ("What's Stubbed", lines 261-265) acknowledges the
gap.

Under the corrected framing the threat model is **rogue or compromised
validator burning the operator's upstream LLM bill** — the gateway is
the operator's spending choke point, not a customer-facing API.

### C4. Signing has no replay protection

Each `sign_request_headers` call generates a fresh nonce
(`src/gateway/signing.py:153`) but `verify_request_headers`
(lines 186-273) does **not** maintain a seen-nonce store. The 5-minute
timestamp window (`max_skew_seconds=300`, line 244) is the only replay
barrier; within it, identical signed bodies are valid as many times as
replayed. Combined with C3, even after signing is enforced a captured
signed request can be reused for 5 minutes — at the operator's
upstream-API expense.

### C5. Validator container is effectively host-root

**Status (2026-06-08): one defence-in-depth flag added; core risk unchanged.**
`security_opt: ["no-new-privileges:true"]` was added to the validator
service in `docker-compose.yaml` (validated with `docker compose config`).
This blocks setuid-based escalation inside the container but does **not**
close the load-bearing risk: the read-write `docker.sock` mount still gives
the validator process full control of the host Docker daemon. The
remaining items below (userns-remap, read-only rootfs, AppArmor) are
unchanged and still recommended.


`docker-compose.yaml:56` mounts `/var/run/docker.sock` into the validator
read-write so it can spawn sibling agent containers. This is the
documented design, but the validator process itself has **no secondary
containment** — a single RCE in the validator (FastAPI input parsing, a
dependency CVE, or a bug in `src/validator/sandbox_docker.py` payload
handling) gives the attacker full control of the host *and* the
in-memory hotkey loaded at `src/gateway/local_proxy.py:144`. No
user-namespace remap, no AppArmor profile, no read-only validator
rootfs.

### C6. Per-validator metering on the gateway is missing

The central gateway is the choke point where validators spend the
operator's upstream LLM budget. Today there is no per-hotkey quota, no
cost cap, no rate limit, no signing enforcement. Pillar 1's economics
depend on this being controlled before any external validators are
onboarded. Concretely needed:

- Enforced signature verification with metagraph membership check
- Per-hotkey daily/monthly cost cap
- Per-hotkey rate limit
- Replay nonce store (closes C4)
- Audit log of every authorised call keyed by hotkey for reconciliation

### C7. Trace egress now exists, but the path is lossy and transport-insecure

**Status (2026-06-08): partially addressed — downgraded from "does not exist".**
Sealed traces now leave the validator: after each run the validator POSTs
the prediction plus the full `EvidenceDigest` (in `reasoningTrace.trace`)
to the orchestrator at `/v1/validators/prediction`
(`src/validator/orchestrator_api.py:18,208`; payload built in
`validator.py` `_build_prediction_submit_payload`). Authentication to the
orchestrator is Bittensor hotkey signing — the right primitive. The
remaining gaps that keep this from being production-grade:

- **No durability on failure.** If the submit raises, it is caught and the
  call simply `return`s (`src/validator/validator.py:484-490`) — the scored
  prediction is dropped. There is no queue, no retry, no dead-letter. A
  single network blip loses a trace. (See H9.)
- **No idempotency key.** The payload carries no `submissionId` /
  content hash, so any retry the orchestrator *does* receive is an
  uncontrolled duplicate. This is H8, now concrete on a live path.
- **Transport is not enforced TLS.** Default URL is
  `http://localhost:4000` (`src/core/constants.py:86`); nothing rejects a
  plaintext `http://` orchestrator URL in production. (See H10.)

The egress *shape* is sound; treat C7 now as a hardening task, not a
design task. See Q1.

### C8. Scoring contract is implicit and unversioned

**Status (2026-06-08): partially addressed.** A wire format now exists in
practice — the orchestrator consumes the submit payload (full
`EvidenceDigest` under `reasoningTrace.trace`) and the validator reads
back `GET /v1/validators/scored-predictions`. But the contract is defined
only by the code that emits it, pinned to an inline `schemaVersion: "1.0"`
string with no validation on either side, and the resolution-record
duality described in C2 still applies. Promote this to an explicit,
versioned, documented interface before external validators or pillar 2
depend on it. See Q2.

### C9. A plaintext database credential was committed to the repo

**New finding (2026-06-08). Tracking fixed this pass; history scrub still
outstanding.** `src/validator/almanac/storage/storage.env` was
**git-tracked** and contained a plaintext Postgres password
(`DB_PASSWORD=...`, line 6) alongside `DB_USER`, `DB_NAME`, host and port.
Per the maintainer this is a **local/test credential** (`almanac_market_test`
DB) with low direct exposure — but committing any secret is the wrong
default, and it is present in committed history (commits `4a02092`,
`67a949a`). The original `.gitignore` `.env` rule did **not** match
`storage.env`, so nothing prevented the commit.

Status of remediation:

- ✅ **Stopped tracking the file** (`git rm --cached`); the local copy is
  preserved and the loader (`postgres_validator_storage.py:69-88`) still
  reads it, falling back to OS env vars when absent.
- ✅ **`.gitignore` tightened** to cover `*.env` / `storage.env` while
  keeping `*.env.example` tracked; a placeholder `storage.env.example`
  already exists for operators.
- ⚠️ **Still outstanding:** rotate the credential wherever it is reused,
  and **scrub git history** (e.g. `git filter-repo`) + force-push with
  coordination so the secret leaves all refs. Low urgency given it is a
  test credential, but it should not live in history indefinitely.
- ⚠️ **Recommended:** add a pre-commit secret scanner so this class of
  leak cannot recur.

### C10. Almanac Postgres store is not concurrency- or crash-safe

**New finding (2026-06-08). This is C1 repeated for the new IM's data.**
`src/validator/almanac/storage/postgres_validator_storage.py` is a
singleton holding a single long-lived connection literally named
`continuous_connection_do_not_reuse` (line 55, created at 148, returned at
293) guarded by a process-local `RLock`. Queries are correctly
**parameterised** (`%s` placeholders — no SQL-injection exposure), but:

- **No connection pool.** Every write also opens an ad-hoc connection via
  `contextlib.closing(self._create_connection())` (lines 153, 338, 392),
  so the "continuous" connection and per-call connections coexist with no
  coherent lifecycle. A dropped DB connection has no recovery path.
- **No rollback on error.** `connection.commit()` is called only on the
  success path (lines 271, 368, 469); there is no `try/except` issuing
  `ROLLBACK`, so a mid-transaction failure leaves the session in an
  aborted/inconsistent state.
- **No schema migrations.** Schema is created inline with raw DDL; there
  is no versioning or migration framework, so any schema change is a
  manual, unversioned operation.

Because this store holds epoch scores and miner-pool weights that feed the
IM, the same "payout integrity" framing as C1 applies: torn or lost writes
here translate directly into mis-weighting on chain.

**Status (2026-06-08): deliberately deferred, not fixed this pass.** The
write methods already use `contextlib.closing(self._create_connection())`,
and psycopg2 rolls back an uncommitted transaction on connection close, so
the immediate torn-commit risk is partially mitigated. The real fixes
(connection pool, explicit rollback, migrations) touch payout-critical
code that **cannot be runtime-tested in this environment** (the module
imports `bittensor` and needs a live Postgres). Rather than ship a
compile-only-verified re-indent of large SQL blocks, this is left for a
dedicated change with a Postgres integration harness.

---

## 6. High findings

### H1. Agent image is referenced by floating tag, not digest

`src/core/config.py:34` and `docker-compose.yaml:45` both use
`arcratio/agent-runner:latest`. `latest` is resolved at pull time on the
host daemon. There is no `--digest` pin, no signature check, no
`docker pull --policy` discipline. A registry compromise (or a developer
accidentally pushing a debug image to that tag) replaces the sandbox
runtime for every subsequent run with no audit signal.

### H2. No seccomp/AppArmor profile on the agent sandbox

`src/validator/sandbox_docker.py:266-296` relies on Docker's *default*
seccomp profile plus `cap_drop=ALL`. The default profile still permits
roughly 300 syscalls, including `ptrace`, `process_vm_readv`, and
`madvise(MADV_DONTNEED)`. For a Bittensor-style untrusted-agent threat
model this is the bare minimum, not robust. gVisor is supported
(`docker_gvisor` runtime) but is opt-in and not the default.

### H3. In-process sandbox mode is the path of least resistance for users

`scripts/run_forecast.py` defaults to whatever `SANDBOX_TYPE` is in env;
the README (line 255) actively recommends `--sandbox in_process` for
dev. In that mode the agent gets a direct Python reference to
`ProviderGateway` via `ForecastingContext`, and the in-process gateway
does not run the proxy's track allowlist (`src/gateway/track_config.py`).
If this mode ships to a validator operator "for trying out an agent
locally" it is a footgun: agents can call any provider, on any track,
with no per-tenant policy. Either gate it behind an explicit "I am
developing my own agent" flag or remove it from `run_forecast.py`'s
default path.

### H4. Observability is essentially absent

`logger.info` calls in the gateway (`src/gateway/server.py:106-114`) and
`print` calls in the orchestrator
(`src/validator/orchestrator.py:129-143`) are the entirety of the
operational signal. No structured logs (JSON), no correlation ID
joining validator → local proxy → central gateway → provider, no
metrics endpoint, no OpenTelemetry. You cannot run a paid service SLA
or diagnose a misbehaving validator without this.

### H5. List operations don't scale and don't filter

`JsonTraceStore.list_traces_by_event` and `list_traces_by_agent`
(`src/storage/json_store.py:41-51`) load *every* trace in the directory
into memory before filtering. There are no by-date, by-cost,
by-resolution-status, or pagination variants. Whatever durable
ingestion store replaces this on the operator side needs a query model
designed up front.

### H6. Failure semantics drop data silently

`Orchestrator.run_agent` saves the trace only if `assemble_trace`
succeeds. If the gateway logged 5 calls and the agent crashed on the
6th, those 5 calls and their cost are not persisted anywhere. The
`_iter_all_traces` `except Exception: continue`
(`src/storage/json_store.py:68-69`) also hides corruption from
operators. For payout integrity, "we logged the spend even if we
couldn't ship the trace" must be invariant; today it isn't.

### H7. Hotkey is held in plaintext in process memory for the validator's lifetime

`src/gateway/local_proxy.py:89-91` loads the keypair on startup and
keeps it for the life of the process. No rotation, no revocation hook,
no in-memory zeroisation. Combined with C5 (validator = host-root) the
blast radius of any validator compromise includes the validator's
identity on the subnet.

### H8. No idempotency for re-ingestion

A validator that retries submitting the same trace (network blip,
process restart) will be processed multiple times by whatever ingestion
service consumes traces. There is no `request_id` / `submission_id` /
content-derived key to dedupe. This must be designed into the egress
path (Q1) from the start. **(2026-06-08: now concrete — the live submit
payload in `validator.py` carries no idempotency key. See C7, H9.)**

### H9. Prediction submission is lossy — no retry, queue, or durability

**New finding (2026-06-08).** The validator's submit path
(`src/validator/validator.py:469-490`) builds the payload, calls
`submit_validator_prediction`, and on any exception logs and `return`s.
The digest is persisted locally, but the *scored prediction* — the thing
the IM pays against — is silently dropped on the first network error,
timeout (default 15 s, `orchestrator_api.py:20`), or 5xx. The same lossy
pattern repeats on the read side: assignment fetch and scored-prediction
fetch both `except Exception: return` and simply skip the epoch
(`validator.py:389-391`, `~827-842`). For payout integrity this must
become a durable, retried, idempotent submission (a local outbox/queue
with backoff), not a fire-and-forget POST. Couples with H8.

### H10. Orchestrator path has no enforced TLS and no miner code-signature chain

**New finding (2026-06-08).** Two transport-trust gaps on the new
orchestrator integration:

- **No TLS enforcement.** The orchestrator base URL defaults to
  `http://localhost:4000` (`src/core/constants.py:86`) and nothing rejects
  a plaintext `http://` URL in production. Agent **code**, predictions, and
  full traces all traverse this URL; on a plaintext link they are
  MITM-able.
- **Code integrity rests on a hash, not a signature.** When the validator
  pulls agent code it verifies the SHA-256 against the value the
  orchestrator itself supplied (`validator.py:515-521`) — that detects
  transport corruption but **not** substitution by a compromised or
  spoofed orchestrator, because the attacker controls both the code and
  the hash. There is no miner hotkey signature over the code that the
  validator can independently verify. The Docker sandbox (good) is then
  the *only* thing standing between an injected payload and the validator
  host (see C5). Mitigate by enforcing HTTPS (reject `http://` for
  non-loopback) and adding a miner-signature-over-code check before
  execution.

### H11. Almanac ships production footguns

**New finding (2026-06-08).** The merged Almanac IM carries several
dev-only behaviours into production code paths:

- **Toggleable synthetic/mock trading data.** `use_synthetic_data`
  (`src/validator/almanac/loop.py:100`) swaps real trading history for an
  sn41-shipped mock dataset. Left true (or defaulted true in a config),
  the entire subnet rewards miners against static fixture data.
- **Hardcoded localhost test endpoint.**
  `_DEFAULT_TRADING_HISTORY_ENDPOINT_TEST = "http://localhost:3001/..."`
  (`loop.py:34`) is selected whenever `network == "test"`; a
  misconfiguration silently points scoring at a non-existent local API.
- **Substring miner-id matching.** A miner profile is accepted if the
  on-chain commitment id is a **substring** of the stored profile id
  (`loop.py:246-247`). This is intentional ("chain stores only a partial
  id for privacy"), but substring matching is weaker than a prefix/exact
  match and widens the space for id collision/spoofing; worth tightening
  to an explicit prefix match with a minimum length.
- **Dual-IM blend edge cases.** The Almanac/Arcratio weight blend can
  silently misbehave: a non-zero `share` on a *disabled* mechanism is
  ignored with no warning, and a total share of zero falls back to equal
  weighting rather than erroring. An operator can believe they are running
  a 50/50 blend while emitting 100%/0%. Add validation that enabled
  mechanisms' shares are positive and sum to 1.0, and fail loudly otherwise.

---

## 7. Medium findings

### M1. Configuration has four sources of truth

`.env` (loaded twice — `src/core/config.py:93` and
`src/gateway/constants.py:21`), `AppConfig` Pydantic models,
`config/pricing_cards.json`, and the hardcoded `TRACK_ALLOWED_PROVIDERS`
dict in `src/gateway/track_config.py:14`. There is no precedence
documented and no schema check. Operators running multiple environments
(dev/staging/prod) will fight this.

### M2. `track_config.py` allowlist is code-not-data

Adding a new track or provider requires a code change and a re-deploy.
For a multi-validator subnet where tracks may evolve, this should be a
database table or a hot-reloadable config file with audit history.

### M3. The README is doing the work of documentation

The Docker section (lines 218-258) carries operational knowledge about
sibling-container bind paths, `host.docker.internal`, `docker wait`
fallback, and `mountinfo` parsing. This is institutional knowledge that
should live next to the code (a `docs/operations.md` plus inline
docstrings on `_sibling_socket_host_bind` and the wait-CLI logic).

### M4. `EvidenceDigest` is mutable

No `model_config = ConfigDict(frozen=True)` anywhere in
`src/core/schemas.py`. After `seal()`, any caller with a reference can
silently mutate fields and re-save without re-sealing. Trivial fix; high
value for the integrity story.

### M5. No schema-version validation on read

`TraceIntegrity.trace_schema_version` is set to `"0.1.0"`
(`src/core/schemas.py:247`) but `JsonTraceStore.get_trace`
(`src/storage/json_store.py:35`) does not check it. Future schema bumps
will silently load old data and fail Pydantic validation in unhelpful
ways. The scoring contract (Q2) needs to pin this down.

### M6. `pyproject.toml` still says `forecasting-prototype` v0.1.0

Cosmetic, but a paid product cannot ship from a
`name = "forecasting-prototype"` package. Worth doing alongside any
rename of imports from `src.*` to a proper package namespace.

### M7. Tests don't cover the central gateway in concurrent / adversarial mode

`tests/gateway/` is mostly per-provider extractor and signing-shape
tests. There's no test asserting "two clients cannot interleave traces"
or "unsigned request is rejected when REQUIRE_SIGNATURE=true". The
shape is right; the coverage isn't.

### M8. Heavy new dependencies are unpinned and one is undeclared

**New finding (2026-06-08).** The Almanac merge added a scientific stack
to `requirements.txt` (lines 11-17) with **no version pins** —
`cvxpy`, `ecos`, `numpy`, `scipy`, `pandas`. `cvxpy`/`ecos` pull compiled
conic solvers; an unpinned resolve can silently break the validator on a
new release and is a supply-chain surface. Separately,
`postgres_validator_storage.py` imports `psycopg2` but `psycopg2-binary`
is **not in any requirements file** (only mentioned in the module
docstring) — Postgres storage degrades to disabled silently if the
operator doesn't hand-install it. `bittensor>=9.0,<11` is also a wide
range across a major version. Pin these (ideally via a lockfile /
`pip-compile`) and declare `psycopg2-binary` as an extra.

**Status (2026-06-08): `psycopg2-binary` now declared** in a dedicated
optional `requirements-almanac-postgres.txt` (additive — keeps it off the
default install since the Postgres path is import-guarded). Pinning the
`cvxpy`/`ecos`/`scipy` stack is still open: it needs a full resolve + smoke
test that this environment cannot run, so it is left for a lockfile pass.

---

## 8. Cross-cutting risks

- **The trust boundary the system most wants to assert (agents are
  contained) is the boundary with the most undocumented sharp edges.**
  The Docker mode is correct in spirit but depends on `:latest` image
  trust, default seccomp, runc-not-gVisor, and a validator process with
  host-root powers (C5, H1, H2). Each of those is a single-point-of-
  failure for the entire isolation claim.
- **The trust boundary the operator most needs for spending control
  (gateway authenticates the validator and meters per-hotkey) does not
  exist yet** (C3, C4, C6). The wire format is finalised; the policy is
  permissive.
- **The integrity story for payouts is half-realised** (C1, C2, M4,
  M5). Sealed but mutable, written but not crash-safe, verified-on-write
  but not verified-on-read, with a split resolution story. Whatever IM
  scoring depends on this needs the gaps closed first.
- **The whole pipeline is broken at step 1** (C7). Sealed traces never
  leave the validator. Nothing else in the operator's plan can be
  validated end-to-end until egress exists.

---

## 9. Recommended priority order

0. **Leaked DB credential (C9) — mostly contained.** Untracking +
   `.gitignore` done this pass; remaining: rotate the credential and scrub
   git history (low urgency — it is a test credential), plus add a secret
   scanner. Not a roadmap item; close it out.
1. **Harden the now-existing egress path, don't redesign it.** Document
   and version the scoring contract (Q2/C8), then make the submit durable
   and idempotent (H9, H8) and enforce TLS + a code-signature check (H10).
2. **Lock down the gateway as the operator's spending choke point**
   (C3, C4, C6 together; H7 alongside).
3. **Fix storage and seal-integrity** wherever traces and IM scores
   durably land (C1, C2, M4, M5, H6), **including the new Almanac Postgres
   store (C10)** — connection pool, rollback, migrations.
4. **Continue tightening the agent sandbox** (C5, H1, H2, H3). These
   stay the same regardless of framing.
5. **De-footgun the Almanac IM** (H11): default off the synthetic-data
   path, validate blend shares, pin deps (M8).
6. **Add observability** before external validators onboard (H4).
7. **Pillar 2 design** is its own exercise; do not start it before Q2
   is fixed.

---

## 10. Verification steps

The audit is read-only. To independently reproduce the load-bearing
findings:

1. **Storage atomicity (C1):** start two `scripts/run_forecast.py` runs
   against the same event id and inspect `data/traces/` for a torn
   write under `kill -9` of one process.
2. **Seal mutability (C2, M4):** in a Python REPL, load a sealed trace
   with `JsonTraceStore.get_trace`, mutate
   `digest.resolution_record.resolved = True`, save it back, and call
   `verify_integrity()` — it returns False and nothing in the read path
   catches it.
3. **Gateway auth (C3):** start `scripts/run_gateway.py` without
   `REQUIRE_SIGNATURE=true`, `curl -X POST /v1/call` from any host with
   no headers — call succeeds.
4. **Replay window (C4):** capture a signed request with `tcpdump` or a
   logging proxy, replay it within 5 minutes — second call succeeds.
5. **Sandbox baseline (H2):** `docker inspect` a live agent container
   and confirm `SecurityOpt` lacks any `seccomp=` and `apparmor=`
   entries.
6. **Image pinning (H1):** `docker image inspect arcratio/agent-runner:latest`
   — the digest is whatever was pulled last; nothing in the codebase
   pins it.

Each check reproduces the finding without modifying the system.

---

## 11. File index

Files that any remediation pass will touch most:

- `src/storage/json_store.py` — entire file is a critical-path rewrite
  (C1, H5, H6)
- `src/core/schemas.py:240-296` — sealing model + integrity envelope
  (C2, M4, M5)
- `src/gateway/server.py:54-121` — gateway auth gap (C3, C6)
- `src/gateway/signing.py` — signature wire format and verification
  (C4)
- `src/validator/sandbox_docker.py:262-298` — sandbox flags and
  runtime (C5, H1, H2)
- `src/validator/orchestrator.py` — failure semantics, identity
  propagation (H4, H6)
- `src/gateway/local_proxy.py:89-91, 176-216` — hotkey lifetime
  (H7), proxy recording
- `src/gateway/track_config.py` — track allowlist (M2)
- `docker-compose.yaml:55-58` — docker.sock + wallet mount (C5)
- `config/pricing_cards.json` + `src/gateway/cost_estimator.py` —
  billing data flow

New hot files added by the orchestrator + Almanac merges (2026-06-08):

- `src/validator/orchestrator_api.py` — assignment poll + prediction
  submit + scored-prediction read (C7, C8, H9, H10)
- `src/validator/validator.py` — submit payload build, code SHA check,
  lossy submit/return paths (C7, H9, H10)
- `src/validator/almanac/storage/storage.env` — committed DB credential
  (C9); untracked + gitignored this pass, history scrub still pending
- `src/validator/almanac/storage/postgres_validator_storage.py` —
  connection reuse, no pool/rollback/migrations (C10)
- `src/validator/almanac/loop.py` — synthetic-data toggle, localhost
  test endpoint, substring miner-id match (H11)
- `src/validator/almanac/scoring.py` + `src/validator/scoring.py` +
  `src/validator/uid_map.py` — dual-IM scoring + blend (H11)
- `requirements.txt` — unpinned `cvxpy`/`ecos`, undeclared
  `psycopg2-binary` (M8)

Files that don't yet exist but should be designed before further code
work:

- A versioned, documented scoring-service interface schema / OpenAPI doc
  formalising the now-implicit submit + scored-prediction contract (Q2)
- A durable, idempotent submission outbox on the validator (H8, H9)
- An ingestion-side durable trace store with by-validator,
  by-event, by-date queries (replaces `JsonTraceStore`)

---

## 12. Re-verification status (second pass, 2026-06-08)

This pass re-checked the first-pass findings against the current tree
(after the `main` + `origin/integration_updates` merges) and added
findings for the orchestrator API and Almanac IM. Line numbers below are
current.

| # | Finding | Status | Current anchor |
|---|---------|--------|----------------|
| C1 | Storage not crash/concurrency-safe | PARTIALLY FIXED — atomic writes added | `json_store.py` `_atomic_write_text` (locking still open) |
| C2 | Trace integrity half-realised | IMPROVED — read-time verify added | `get_trace` now warns on hash mismatch; still unfrozen/non-enforcing |
| C3 | Gateway unauthenticated by default | STILL VALID | `server.py:54-56` (`REQUIRE_SIGNATURE` off by default) |
| C4 | Signing has no replay protection | STILL VALID | `signing.py:155,194` (no nonce store; 300 s skew) |
| C5 | Validator container = host-root | IMPROVED — `no-new-privileges` added; sock risk unchanged | `docker-compose.yaml` (docker.sock rw, no userns/apparmor) |
| C6 | No per-validator gateway metering | STILL VALID | `server.py:73-121` |
| C7 | Trace egress | REFRAMED — now exists, lossy/insecure | `orchestrator_api.py:18,208` |
| C8 | Scoring contract | REFRAMED — now implicit, unversioned | submit payload in `validator.py` |
| **C9** | **Committed DB credential** | **FIXED (tracking) — untracked + gitignored; history scrub pending** | `almanac/storage/storage.env:6` |
| **C10** | **Almanac Postgres not crash/concurrency-safe** | **NEW — deferred (needs PG harness)** | `postgres_validator_storage.py:55,148,293` |
| H1 | Agent image floating `:latest` | STILL VALID (cross-arch build now scripted) | `core/constants.py`, `docker-compose.yaml` |
| H2 | No seccomp/AppArmor on sandbox | STILL VALID | `sandbox_docker.py` (`no-new-privileges` only) |
| H3 | In-process sandbox footgun | IMPROVED (partial) | default now `docker_runc`; `--sandbox in_process` still available |
| H6 | Failure semantics drop data | STILL VALID | `orchestrator.py` (spend lost if assemble fails) |
| H7 | Hotkey plaintext in memory | STILL VALID (by design) | `local_proxy.py` |
| H8 | No idempotency for re-ingestion | STILL VALID — now concrete | submit payload has no key |
| **H9** | **Prediction submit is lossy** | **NEW (high)** | `validator.py:469-490` |
| **H10** | **No TLS / no code-signature on orch path** | **NEW (high)** | `constants.py:86`, `validator.py:515-521` |
| **H11** | **Almanac production footguns** | **NEW (high)** | `almanac/loop.py:34,100,246` |
| **M8** | **Unpinned/undeclared deps** | **PARTIALLY FIXED — `psycopg2-binary` declared** | `requirements-almanac-postgres.txt` (cvxpy/ecos pins open) |

### 12.1 Remediation applied in this pass (non-breaking, verified)

These changes were made and committed alongside this audit update. Each was
verified without breaking the codebase (the full test suite needs
`bittensor`/`cvxpy`, absent in this environment, so verification was scoped
to what could be exercised):

- **C9 — leaked credential contained.** `storage.env` untracked
  (`git rm --cached`, local copy preserved), `.gitignore` tightened to
  `*.env`/`storage.env` while keeping `*.env.example`. *Pending:* rotate +
  history scrub (your call, low urgency — test credential).
- **C1 — atomic trace writes.** `_atomic_write_text` (tempfile → fsync →
  `os.replace`) in `json_store.py`. Verified: round-trip, overwrite, and a
  no-torn-file check in an isolated pydantic venv.
- **C2 — read-time integrity check.** `get_trace` now warns on a sealed-hash
  mismatch (non-fatal). Verified: tamper-on-disk emits the warning and still
  returns the digest.
- **C5 — validator hardening.** `no-new-privileges:true` on the validator
  service. Verified with `docker compose config`.
- **Docker cross-arch (H1-adjacent, the multi-OS ask).** Added
  `scripts/build_images.sh` (+ `.ps1` for Windows) that sets `--platform`
  explicitly and warns on cross-arch builds; README + compose comments
  document the footgun. Verified end-to-end by cross-building `agent-runner`
  for `linux/amd64` on an arm64 host and confirming `Architecture=amd64`.
- **M8 — optional Postgres dep declared** in
  `requirements-almanac-postgres.txt`.

Deferred deliberately (need an environment this pass lacks): C10 (Postgres
pool/rollback/migrations — needs a live PG + bittensor), `cvxpy`/`ecos`
pinning (needs a full resolve), and the larger gateway/orchestrator work
(C3/C4/C6, H9/H10) which is feature work, not a non-breaking patch.

---

## 13. Third pass — new findings + remediation (2026-06-10)

Run after the `origin/assignment_pipeline` sync. That sync refactored the
orchestrator-assignment flow out of `validator.py` into
`src/validator/assignment_pipeline.py`; the load-bearing safety guard was
re-verified intact (`assignment_pipeline.py:274` still refuses to execute
orchestrator-supplied agent code unless `sandbox_type` is docker-based, and
the untrusted `exec()` only ever runs inside the sandbox container, never the
validator process). The sync also renamed the inline payload key
`agent_class` → `inline_class` and added the `/v1/gateway/providers` catalog
endpoint used by the miner upload flow.

This pass found new issues the first two passes missed and remediated the
non-breaking subset. Findings are numbered N1–N8 to avoid colliding with the
existing C/H/M scheme.

### 13.1 New findings

#### N1. Almanac profile-id matching had a case-sensitivity bug that zeroed legitimate miners (PAYOUT)

**Severity: high — fixed this pass.** `scoring.py` (~line 324) stored the
first eligible trade's profile id lowercased
(`miner_profiles[miner_id] = trade["profile_id"].lower()`) but compared
subsequent trades against the **raw, non-lowercased** value
(`elif miner_profiles[miner_id] != trade["profile_id"]`). Any miner whose
profile id contained an uppercase character had every trade after the first
"mismatch", get comma-appended, and then be flagged as multi-profile by the
comma check in `loop.py` — which sets that miner's weight to **0**. A
legitimate miner with a mixed-case profile id and ≥2 trades was silently
zeroed every epoch. Fixed: compare lowercase-to-lowercase and dedupe against
the already-recorded ids, so genuine multi-profile is still detected but the
false positive is gone.

#### N2. Unguarded division crashes weight-setting on a zero-fee epoch (PAYOUT)

**Severity: critical — fixed this pass.** `calculate_weights` in `scoring.py`
divided each miner's tokens and the dynamic general-pool tokens by
`total_epoch_budget` (~lines 1340, 1367) with no zero guard. An epoch with no
qualifying fees (`total_epoch_budget == 0`) raised `ZeroDivisionError` and
crashed the entire weight-setting step, so **no weights were committed on
chain** that epoch. Fixed: guard both divisions, falling back to `0.0` weight
when the budget is non-positive. (The `scale_factor` divisions at ~1236/1294,
flagged by an automated reviewer, were verified **already safe** — their
enclosing `if total_tokens > budget` guarantees `total_tokens > 0`.)

#### N3. Gateway leaked upstream provider bodies and schema internals to callers

**Severity: high — fixed this pass.** `src/gateway/server.py` returned
`detail=str(exc)` on the 502 (provider-call-failed) path; the Claude and
OpenRouter adapters embed `resp.text[:2000]` from the upstream response in
that exception (`providers/claude.py:83`, `providers/openrouter.py:99`), so a
raw upstream error body — potentially carrying rate-limit headers, internal
ids, or secrets — was relayed verbatim to the (currently unauthenticated)
caller. The 422 path likewise leaked full Pydantic validation detail. Fixed:
both paths now log the full error server-side and return a generic message
(`upstream provider '<id>' call failed` / `invalid request body`). Verified a
synthetic `sk-…` secret in an upstream error reaches only the server log,
never the HTTP response.

#### N4. Gateway had no request-body size limit (DoS)

**Severity: medium — fixed this pass.** `/v1/call` buffered the full request
body in memory before parsing with no cap. Fixed: reject bodies over 1 MiB
(via declared `Content-Length` and actual read length) with a 413.

#### N5. Gateway netuid claim was signed but never enforced

**Severity: high — partially addressed this pass.** The signing canonical
form includes `netuid`, but even with `REQUIRE_SIGNATURE=true` the server
never checked the claimed netuid against the operator's subnet — a valid
signature from a hotkey on *any* subnet would pass (the metagraph-membership
check itself remains C6 feature work). Added an optional `GATEWAY_NETUID`
pin: when set and signatures are required, a signed request whose netuid does
not match is rejected. Default-off, so no behaviour change until configured.

#### N6. Unauthenticated `/health` disclosed gateway configuration

**Severity: low — fixed this pass.** `/health` returned the provider list and
the `require_signature` policy to any caller. Trimmed to `{"status": "ok"}`.

#### N7. Agent stdout was buffered and JSON-scanned without a size cap (DoS)

**Severity: medium — fixed this pass.** `sandbox_docker.py` read the
untrusted agent container's entire stdout into memory and ran an O(n)
backwards brace-scan (`_last_json_object`) plus a pydantic parse over it, with
no ceiling (`mem_limit` bounds the agent, not the validator's read). A hostile
agent could flood stdout to pressure the validator. Fixed: reject stdout over
8 MiB before decoding/parsing. (The deeper fix — a streamed, capped read so
the bytes are never fully buffered — is left as a follow-up.)

#### N8. Proxy-socket directory created with umask-default permissions

**Severity: medium — fixed this pass.** `validator.py`
`_prepare_proxy_socket_path` created the UDS directory (and its `/tmp`
fallback) with the process umask (typically `0o755`). That UDS lets its
holder drive provider calls, so a world-traversable parent dir widens local
access. Fixed: explicit `chmod 0o700` on both the primary and fallback
directories.

**Also surfaced, deferred to a deliberate decision (run_gateway bind):** the
gateway still defaults to `--host 0.0.0.0`, which — combined with C3 (signing
off by default) — is an open, unauthenticated proxy to the operator's
provider keys on all interfaces. The default was **intentionally left as-is**
this pass because the documented Docker dev flow relies on containers
reaching the host gateway via `host.docker.internal`; flipping to loopback
would silently break it. As a non-breaking interim, `run_gateway.py` now logs
a loud warning when bound to a non-loopback address with `REQUIRE_SIGNATURE`
off. Flipping the default to `127.0.0.1` (with `--host 0.0.0.0` documented for
the container flow) remains a recommended follow-up.

### 13.2 Remediation applied this pass (non-breaking, verified)

Changes were made to the working tree (uncommitted — the maintainer handles
commit/push). Each edited file byte-compiles; the gateway test suite still
passes (32 passed / 13 skipped, unchanged from baseline); the gateway
hardening and the scoring logic were verified with targeted checks. As in
prior passes, the full validator/Almanac suites need `bittensor`+`cvxpy`
(absent here), so those changes are compile- and logic-verified only.

- **N1, N2 — payout bugs fixed.** `scoring.py`: lowercase-to-lowercase
  profile-id compare with dedupe; zero-budget guards on both weight
  divisions. `almanac/constants.py`: import-time range validation
  (`[0, 1]`) on `EXCESS_MINER_TAKE_PERCENTAGE` and
  `GENERAL_POOL_WEIGHT_PERCENTAGE` to prevent negative/over-unity burn
  weights. Verified pass + reject paths in isolation.
- **N3 — gateway error sanitisation.** 502 and 422 return generic messages;
  full detail logged server-side. Verified no upstream secret reaches the
  response body.
- **N4 — gateway body cap.** 1 MiB limit → 413. Verified.
- **N5 — optional netuid pin** (`GATEWAY_NETUID`). Verified it still rejects
  unsigned requests and does not regress the existing signing test.
- **N6 — `/health` trimmed** to `{"status": "ok"}`. Verified.
- **N7 — sandbox stdout cap** (8 MiB) in `sandbox_docker.py`.
- **N8 — `chmod 0o700`** on the proxy-socket dir + fallback in `validator.py`.
- **run_gateway bind warning** (see N8 note above).

### 13.3 Still outstanding after this pass

- **C9 history scrub — confirmed still reachable.**
  `git show 67a949a:src/validator/almanac/storage/storage.env` still returns
  the plaintext password. Rotate the credential and `git filter-repo` +
  coordinated force-push. Unchanged from §12.
- **Dockerfile hardening + dependency lockfile** (validator image runs as
  root; base images pinned by tag not digest; `cvxpy`/`ecos`/`requests`/
  `fastapi`/`uvicorn` unpinned). These change the build and need a full
  resolve/rebuild to validate, so they were not patched blind this pass.
- **C3/C4/C6, H9/H10, C10** remain feature work as described in §5–§6.
