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

### 2.3 The connection between pillars (currently absent)

Three artefacts are needed to bridge pillar 1 → pillar 2:

1. **Trace egress** — a path for sealed traces to leave validator-local
   disk and reach the operator's ingestion side. Not designed.
2. **Scoring contract** — the interface between the trace producer (this
   repo) and the scoring/IM consumer (separate repo). Not defined.
3. **Prediction publication** — once scored, predictions need to land
   somewhere pillar 2 can read with low latency. Not designed.

Items 1 and 2 are the two highest-priority undecided architectural
decisions. They should be resolved before any remediation work begins on
the existing code.

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

`src/storage/json_store.py:31-33` writes traces with bare
`path.write_text(digest.model_dump_json(...))`. No `os.fsync`, no atomic
`os.replace` from a tempfile, no file lock, no per-key serialisation. Two
concurrent writers on the same `execution_id` race; a power loss mid-write
yields a half-written JSON file that `model_validate_json` rejects and
that `_iter_all_traces` (line 65) silently skips via
`except Exception: continue`. The "swappable interface" comment at the top
is true at the type level, but list operations
(`list_traces_by_event`, `list_traces_by_agent`) glob the entire
`traces/` directory and parse every file in memory — that's not swappable
in shape, it's a re-architecture.

Under the corrected framing this is **payout integrity**: if torn or
silently-dropped traces feed the IM scoring, validators are paid against
incomplete data.

### C2. Trace integrity model is half-realised

`EvidenceDigest.seal()` (`src/core/schemas.py:280-289`) computes a SHA-256
over the digest with `trace_hash` zeroed, but:

- The model is **not frozen** (`model_config` omitted) — `resolution_record`
  and any other field can be mutated in memory after seal, silently
  invalidating the hash.
- Verification is never automatic. `JsonTraceStore.get_trace`
  (`src/storage/json_store.py:35`) does not call `verify_integrity()` on
  read, so a tampered file on disk loads cleanly.
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

### C7. Trace egress / ingest path does not exist

Sealed traces sit on validator-local disk and never leave. There is no
HTTP endpoint, no queue producer, no chain anchor producer, no S3
uploader. This is the load-bearing missing piece for pillar 1; until it
is designed and implemented, the rest of the operator's pipeline is
theoretical. See Q1.

### C8. Scoring contract is undefined

Whatever scoring service consumes traces and emits scores needs an
explicit, versioned interface. Today there is none, and the
resolution-record duality described in C2 will make any naïve
implementation inconsistent. See Q2.

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
path (Q1) from the start.

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

1. **Decide Q2 (scoring contract), then Q1 (trace egress).** Paper
   decisions. They unblock everything else.
2. **Lock down the gateway as the operator's spending choke point**
   (C3, C4, C6 together; H7 alongside).
3. **Fix storage and seal-integrity** on the *ingestion* side wherever
   traces durably land for scoring (C1, C2, M4, M5, H6).
4. **Continue tightening the agent sandbox** (C5, H1, H2, H3). These
   stay the same regardless of framing.
5. **Add observability** before external validators onboard (H4).
6. **Pillar 2 design** is its own exercise; do not start it before Q2
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

Files that don't yet exist but should be designed before further code
work:

- A trace egress producer (Q1)
- A scoring-service interface schema or OpenAPI doc (Q2)
- An ingestion-side durable trace store with by-validator,
  by-event, by-date queries (replaces `JsonTraceStore`)
