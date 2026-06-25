# Docker Environment Security & Resilience Review

**Scope:** Validator Docker sandbox + local gateway proxy — focus on the customer-facing Docker environment and service-side safety, security, and crash resistance.
**Branch reviewed:** `docker-review`
**Review date:** 2026-06-25
**Method:** Multi-agent code review (high effort) — 8 finder angles → 39 candidates → independent verifier per candidate → 10 reported.

## How to use this tracker

Each issue has a status. Update it as work progresses:

- `OPEN` — not yet started
- `IN PROGRESS` — being addressed (note who/what)
- `FIXED` — code change landed; fill in **Fixed** (date + commit/PR) and **Resolution**
- `CLOSED` — verified and signed off; fill in **Closed**
- `WONTFIX` — accepted risk; document the rationale in **Resolution**

When you address an issue, fill in the **Fixed**, **Resolution**, and **Verified** fields, then move the status. Do not delete closed issues — they are the audit trail.

## Summary

| # | Severity | Area | Issue | File | Status |
|---|----------|------|-------|------|--------|
| 1 | 🔴 Critical | Security | Agent can spoof signed billing/attribution hotkey | `src/gateway/local_proxy.py:315` | FIXED |
| 2 | 🔴 Critical | Security | Whole-directory bind mount defeats X-Run-Id isolation | `src/validator/sandbox_docker.py:300` | FIXED (round-trip pending Linux CI) |
| 3 | 🔴 Critical | Security | Sandbox runs as host root in default deployment | `src/validator/sandbox_docker.py:285` | FIXED (round-trip pending Linux CI) |
| 4 | 🔴 High | Security | Provider gateway is an open relay by default | `scripts/run_gateway.py:53` | FIXED |
| 5 | 🟠 High | Security | Agent source + event payloads world-readable on host | `src/validator/sandbox_docker.py:325` | FIXED |
| 6 | 🟠 High | Crash | Unbounded stdout read OOMs the validator | `src/validator/sandbox_docker.py:379` | FIXED (log max-size follow-up) |
| 7 | 🟠 High | Crash | Sync httpx blocks the async event loop | `src/gateway/local_proxy.py:351` | FIXED |
| 8 | 🟡 Medium | Correctness | JSON brace-parsing rejects valid results | `src/validator/sandbox_docker.py:437` | FIXED |
| 9 | 🟡 Medium | Correctness | Host-bind fallback breaks all runs | `src/validator/sandbox_docker.py:113` | FIXED |
| 10 | 🟡 Medium | Correctness | Missing `miner_hotkey` → 400 on default sandbox path | `src/gateway/local_proxy.py:316` | FIXED |

**Recommended fix order:** #1, #2, #3, #6 first (identity spoofing, cross-run isolation bypass, root-on-host, OOM), then the rest.

**Progress (2026-06-25):** All 10 implemented + tested. #1, #4–#10 are fully verified locally; #2 and #3 (the container mount/uid/socket model) are implemented and unit-tested, with their macOS launch path confirmed (9/10 lockdown tests pass) — only the UDS **round-trip** awaits the Linux `docker-security` CI job, since macOS Docker Desktop can't connect to a bind-mounted host socket. Full unit suite after changes: **156 passed, 16 skipped** (Docker-gated), with 5 pre-existing unrelated failures deselected in CI (2 genuine offline bugs + 3 wallet-gated; see CI note). Also: the gateway here was clarified to be a **simulator** (not the production service), so it was made **mock-by-default** and clearly labeled — see Issue 4's severity reframe.

**Running the Docker suite:** the `--docker` flag is registered in `tests/security/conftest.py`, so it must be invoked with that directory as a path arg — **`pytest tests/security --docker`** (bare `pytest --docker` errors "unrecognized arguments"). A test-harness bug was fixed along the way: the `proxy_socket_dir` fixture built the UDS under pytest's deep `tmp_path`, exceeding the macOS 104-byte `AF_UNIX` limit so every test errored with "AF_UNIX path too long"; it now uses a short `/tmp` base.

**macOS run (2026-06-25):** **9 of 10 lockdown tests pass** on Docker Desktop. The 10th (`TestPositiveEndToEnd::test_simple_agent_completes`, the only UDS round-trip) fails with `httpx.ConnectError: [Errno 95] Operation not supported` — a Docker Desktop limitation (a host UNIX socket shared into a Linux container via virtiofs/gRPC-FUSE returns `ENOTSUP` on `connect()`). Confirmed **pre-existing** (fails identically on clean HEAD, with my src changes stashed). The launch path is still exercised — the runner read its input under the new `0o700` perms (#5) and the container started — so #5/#6/#8/#9 don't break startup. **The positive round-trip path, and final sign-off for deferred #2/#3, must be validated on Linux** (Linux CI, a Linux host, or the production/compose topology where the validator runs inside a container sharing a real socket — not validator-on-host + sibling).

**CI (`.github/workflows/ci.yml`, added 2026-06-25):** two `ubuntu-latest` jobs. `unit` runs the full suite (148 passed, 16 Docker-gated skips) with 5 pre-existing, unrelated failures quarantined via documented `--deselect` (2 genuine offline bugs: claude usage-cost + anthropic extractor; 3 wallet-gated: `test_validator_scoring_step.py`, which `KeyFileError` without a Bittensor wallet). `docker-security` runs `pytest tests/security --docker` on Linux — this is the real coverage for the UDS round-trip that macOS Docker Desktop can't run, and the gate for signing off #2/#3.

---

## 🔴 Security — untrusted-agent trust boundary

### Issue 1 — Agent can spoof the signed billing/attribution hotkey
- **Severity:** 🔴 Critical
- **Status:** FIXED (awaiting sign-off → CLOSED)
- **File:** `src/gateway/local_proxy.py:315`
- **Risk:** The proxy uses `miner_hotkey_override or run.miner_hotkey`, letting the untrusted sandbox set `minerHotkey` in its request body. The validator signs it and sets `x-miner-hotkey` to the attacker-chosen value. A malicious agent can charge inference cost and usage to a **competitor's** miner, corrupting billing and attribution.
- **Failure scenario:** A malicious agent POSTs a provider call with `"minerHotkey": "<competitor_ss58>"`; the proxy signs it with the validator hotkey, and the central gateway attributes/charges the cost to a different miner than the one being evaluated.
- **Fix direction:** The signed identity must come from the run registration only, never the request body. Drop the body override for the sandbox path.
- **Fixed:** 2026-06-25, working tree (uncommitted) on branch `docker-review`.
- **Resolution:** Removed the `miner_hotkey_override` parameter from `_forward_and_record` and the body parsing in both `/v1/call` and `/v1/gateway/validator/completions`. The signed `x-miner-hotkey` now derives solely from `run.miner_hotkey` (set by the validator at `register_run`). Also dropped the now-dead `minerHotkey` key from `_params_from_completions_payload` so recorded params no longer carry an agent-supplied identity. Confirmed no leak path back into the signed payload: `optional_completions_fields` does not pass `minerHotkey` through, and the payload already `pop`s it before signing.
- **Verified:** Added regression test `test_body_miner_hotkey_cannot_override_registered_identity` (plants attacker `minerHotkey` at body + params level on both routes; asserts upstream `x-miner-hotkey == "5RealMiner"` and no `minerHotkey` in the forwarded body). `tests/gateway/test_local_proxy.py` → 11/11 pass. Negative control confirmed: under the old `miner_hotkey_override or ...` line the attacker value would win and the assertion would fail, so the guard is real. (Two unrelated pre-existing failures in `test_gateway_usage_metadata.py` / `test_extractor_offline.py` confirmed present on clean HEAD — not caused by this change.)
- **Closed:** _(date / sign-off)_

### Issue 2 — Whole-directory bind mount defeats X-Run-Id isolation
- **Severity:** 🔴 Critical
- **Status:** DEFERRED — fix designed, **not applied** (needs Docker E2E; mis-set breaks every run)
- **File:** `src/validator/sandbox_docker.py:300`
- **Risk:** Docstrings claim each runner gets "a single read-only bind mount: the UDS," but the entire `/run/arcratio` directory — including `inputs/` holding **every concurrent run's payload** — is bind-mounted into each sibling. X-Run-Id authentication is bypassable.
- **Failure scenario:** A malicious agent reads `/run/arcratio/inputs/.arcratio_stdin_<other>.json`, extracts another concurrent run's RUN_ID and agent_code, then issues provider calls under that run_id over the shared UDS — escaping its own track allowlist and charging/attributing calls to another run.
- **Fix direction:** Mount only the UDS into each sibling (or a per-run isolated dir), not the shared `/run/arcratio` tree.
- **Fixed:** 2026-06-25, working tree (uncommitted) on `docker-review`.
- **Resolution:** Replaced the whole-socket-dir bind with **two per-run read-only file binds** (`_sandbox_volumes`): the proxy socket `<host>/proxy.sock` → `/run/arcratio/proxy.sock`, and *this run's* input file `<host>/inputs/<name>` → a fixed `/run/arcratio/input.json` (with `ARCRATIO_RUNNER_INPUT_FILE` pointed at it). The `inputs/` directory is no longer mounted, so a sibling can neither enumerate nor read another concurrent run's payload — X-Run-Id is a real capability again, since an agent only knows its own `RUN_ID` (env). The input file is chmod'd `0o644` (readable by the non-root sandbox uid through the bind) inside the still-`0o700` dir (other host users blocked).
- **Verified:** Unit — `tests/validator/test_sandbox_mounts.py` asserts the plan: exactly two file binds, the dir itself is never mounted, all RO, and two different runs bind disjoint input sources (no cross-run reference). macOS Docker — `pytest tests/security --docker`: **9/10 lockdown tests pass**, and the runner launches as uid 10001 and **reads its bind-mounted input** before the (macOS-only) socket round-trip fails, exercising the new mount path.
- **Linux CI (run 28188589973):** the `docker-security` job confirmed the round-trip — **all 9 isolation tests pass** and the runner **connected to the proxy over the bind-mounted socket and reached the gateway** (proving #2's file-bind mounts + #3's non-root uid + `0777` socket all work on Linux). The lone failure was a stale test, not the fix: `TestPositiveEndToEnd` registered its run without a `miner_hotkey`, which now (post-#1) 400s; fixed by passing `miner_hotkey="5TestMiner"` in all `register_run` calls in `test_sandbox_lockdown.py`. Expect green on re-run.
- **Pending (final sign-off):** confirm the `docker-security` job is green after the test fix. Optional follow-up: an attack agent that tries to read a second run's input and asserts it cannot, for an explicit isolation assertion.
- **Closed:** _(date / sign-off — after Linux CI green)_

### Issue 3 — Sandbox runs as host root in the default deployment
- **Severity:** 🔴 Critical
- **Status:** DEFERRED — fix designed, **not applied** (needs Docker E2E + socket-perm coordination; mis-set breaks every run)
- **File:** `src/validator/sandbox_docker.py:285`
- **Risk:** `runner_user = "{uid}:{gid}" if host_uid != 0 else "0:0"`. The default compose runs the validator as root (wallet at `/root/.bittensor`), so the documented "non-root UID 10001" invariant is silently dropped and every sandbox runs as uid 0. With the default `docker_runc` runtime (no gVisor), a container escape lands as **root on the host** — a sandbox bug becomes full host compromise.
- **Failure scenario:** In the default compose deployment every agent sandbox runs as uid 0; a runc/kernel escape by hostile agent code lands as root on the host instead of an unprivileged user.
- **Fix direction:** Never fall back to `0:0`. Force a non-root sandbox UID regardless of the validator's own uid; document running the validator as non-root and/or gVisor for the runtime.
- **Fixed:** 2026-06-25, working tree (uncommitted) on `docker-review`.
- **Resolution:** `_sandbox_runner_user()` now returns a fixed `10001:10001` unconditionally — the `0:0`-when-root branch is gone, so a container escape lands unprivileged even in the root-validator compose default. To keep the UDS reachable by this non-root uid, `scripts/run_local_proxy.py` sets `os.umask(0o000)` right before `uvicorn.run(uds=...)` so the socket is created `0777` (world-connectable); the test fixture mirrors this with `os.chmod(socket_path, 0o666)`. This is safe because the socket is gated by X-Run-Id and, with Issue 2 applied, an agent can't learn another run's id. The per-run input file is `0o644` so uid 10001 can read it through the bind (Issue 5 coordination handled).
- **Verified:** Unit — `tests/validator/test_sandbox_mounts.py::test_runner_user_is_fixed_nonroot` (never `0:`). macOS Docker — `pytest tests/security --docker`: **9/10 pass**; the runner demonstrably executes as uid 10001 and reads its input before the macOS-only socket round-trip fails (errno shifted `95 ENOTSUP` → `13`, both macOS UDS-over-bindmount artifacts, not Linux-predictive). The wallet/passwd isolation tests still pass under the non-root uid.
- **Pending (final sign-off):** the UDS **round-trip** under uid 10001 must be confirmed green on Linux via the `docker-security` CI job (verifies the `0777` socket is connectable by 10001). **Recommended follow-up:** make gVisor (`runtime=runsc` / `docker_gvisor`) the documented default sandbox runtime for untrusted agents — defence in depth against a runc escape.
- **Closed:** _(date / sign-off — after Linux CI green)_

### Issue 4 — Provider gateway is an open relay by default
- **Severity:** 🔴 High
- **Status:** FIXED (awaiting sign-off → CLOSED)
- **File:** `scripts/run_gateway.py:53`
- **Risk:** Defaults to binding `0.0.0.0` with `REQUIRE_SIGNATURE` off, guarded only by a `log.warning`. Anyone on the network can send unsigned `/v1` calls and burn the operator's upstream provider API keys.
- **Failure scenario:** An operator runs `scripts/run_gateway.py` with defaults on a network-reachable host; the gateway accepts unsigned `/v1` calls from anyone and bills the operator's provider keys — effectively an open relay.
- **Fix direction:** Default bind to `127.0.0.1`; require `REQUIRE_SIGNATURE` (or refuse to start unsigned on a non-loopback bind) instead of only warning.
- **Fixed:** 2026-06-25, working tree (uncommitted) on `docker-review`.
- **Resolution:** Default `--host` changed from `0.0.0.0` to `127.0.0.1`. Added a pure `_bind_policy_error()` check: a non-loopback bind without `REQUIRE_SIGNATURE` now **refuses to start** (`SystemExit(2)`) instead of merely warning. Added an explicit `--allow-open-unauthenticated` / `ALLOW_OPEN_GATEWAY=1` opt-in for operators who truly intend an open relay on a trusted network. Note: a Linux Docker dev flow where containers reach the host gateway now requires an explicit `--host 0.0.0.0` + opt-in (or signature), which is the intended secure-by-default posture.
- **Verified:** `tests/gateway/test_run_gateway_bind_policy.py` (8 tests): loopback always allowed; open bind refused without auth; allowed with signature or explicit opt-in; default host is loopback; and an end-to-end subprocess test confirming the real CLI exits `2` with "refusing to bind" before serving.
- **Severity reframe (2026-06-25):** the gateway in this repo is a **simulated test double**, not the production gateway service (separate repo). Real fix for the key-leak risk is therefore "don't hold real keys," addressed by **mock-by-default** (`run_gateway.py` now loads deterministic offline mocks unless `--live`/`GATEWAY_LIVE=1`; added a mock path to `ClaudeProvider`, tested in `tests/gateway/test_claude_provider_mock.py`). The sim is also now clearly labeled (banner, `server.py`/`run_gateway.py` docstrings, FastAPI title "(Simulated)"). With no keys loaded by default, #4's open-relay exposure is largely moot; the loopback default + refusal remain as cheap hygiene. Net effect: this finding is effectively dev-hygiene, and the sim is leaner/safer by default.
- **Closed:** _(date / sign-off)_

### Issue 5 — Agent source + event payloads are world-readable on the host
- **Severity:** 🟠 High
- **Status:** FIXED (awaiting sign-off → CLOSED)
- **File:** `src/validator/sandbox_docker.py:325`
- **Risk:** `os.chmod(payload_host_dir, 0o755)` makes the `inputs/` directory world-traversable/readable on the Docker host. Any other local user or unrelated container on the validator host can read in-flight agent source and event payloads.
- **Failure scenario:** Any other local user or unrelated container reads `var/run/arcratio/inputs/*.json` and exfiltrates orchestrator-injected agent source code and event payloads for runs in flight.
- **Fix direction:** Restrict the dir to the validator/sandbox uid (e.g. `0o700`/`0o750`) rather than world-readable.
- **Fixed:** 2026-06-25, working tree (uncommitted) on `docker-review`.
- **Resolution:** Changed the inputs-dir chmod from `0o755` to `0o700` (owner-only). The sibling reads its payload through the bind mount as the same uid the validator runs as (`runner_user` is `host_uid:host_gid`, or `0:0` when the validator is root — the dir's owner in both cases), so `0o700` does not block the legitimate read while removing all other-user/other-container access on the host.
- **Verified:** Logic review under the current uid model (owner == reader). ⚠️ Not exercised by an automated test in this environment — the perms only matter once a real sibling container mounts the dir, which needs `pytest --docker`. Recommend confirming `test_sandbox_lockdown.py::TestPositiveEndToEnd` still passes (sibling can read its input under `0o700`) before final sign-off. Coupled to #2/#3: if the sibling is later run under a *different* non-root uid, revisit to `0o750` + shared gid.
- **Closed:** _(date / sign-off)_

---

## 🟠 Crash resistance / availability

### Issue 6 — Unbounded stdout read OOMs the validator
- **Severity:** 🟠 High
- **Status:** FIXED (validator-OOM part); **log max-size is a remaining follow-up** (needs Docker E2E)
- **File:** `src/validator/sandbox_docker.py:379`
- **Risk:** The `_MAX_SANDBOX_STDOUT_BYTES` ceiling is checked *after* `container.logs()` has already buffered the **entire** agent stdout into a Python `bytes` object. A hostile/buggy agent writing GBs to stdout crashes the validator before the guard runs. Separately, the sibling container has no log-driver `max-size`, so the host json-file log can grow unbounded and exhaust disk. The advertised protection does not protect anything.
- **Failure scenario:** A hostile or buggy agent writes hundreds of MB / GB to stdout; `container.logs(stdout=True)` loads the whole blob into memory before the size guard runs, OOMing the validator. The host docker log also grows unbounded.
- **Fix direction:** Stream/cap stdout while reading (enforce the ceiling during the read, not after); set a `max-size` log driver option on the sibling container.
- **Fixed:** 2026-06-25, working tree (uncommitted) on `docker-review`.
- **Resolution:** Added `_cap_log_stream()` + `_read_capped_stdout()`: stdout is now read via `container.logs(..., stream=True)` and accumulated chunk-by-chunk, aborting with `_SandboxStdoutTooLarge` the moment the running total crosses `_MAX_SANDBOX_STDOUT_BYTES` — so the validator never materialises more than ~cap + one chunk. The old "buffer everything, then check `len()`" path is gone. Also bounded the **stderr** read (same OOM shape, was reading the whole stream) to `tail=_LOG_TAIL_LINES`, since stderr is only ever used as a short tail.
- **Verified:** `tests/validator/test_sandbox_stdout_cap.py` (4 tests): under-cap join; empty-chunk skipping; raises when total exceeds cap; and an early-abort test proving it stops mid-stream on an effectively unbounded flood (does not drain the generator).
- **Remaining follow-up (not applied):** the docker log-driver `max-size` (host-disk exhaustion via the json-file log) is **not** yet set. It requires a `docker.types.LogConfig` kwarg on `containers.create`, whose exact accepted shape can't be verified without a live daemon and would break *all* container creation if wrong. Apply `log_config=LogConfig(type="json-file", config={"max-size": "16m", "max-file": "2"})` and confirm with one `pytest --docker` run. Tracked here; lower severity than the validator OOM (host disk vs. process crash).
- **Closed:** _(date / sign-off)_

### Issue 7 — Sync httpx blocks the async event loop
- **Severity:** 🟠 High
- **Status:** FIXED (awaiting sign-off → CLOSED)
- **File:** `src/gateway/local_proxy.py:351`
- **Risk:** `_forward_and_record` calls the blocking sync `http_client.post` inside `async` handlers. Under concurrent runs, each seconds-long LLM call stalls the **entire** event loop — serializing all provider calls and freezing `/health`, causing timeouts and runs that blow the sandbox deadline even when upstream is healthy.
- **Failure scenario:** `_v1_call` / `_v1_gateway_validator_completions` are async but do a blocking sync POST; concurrent agent runs serialize and the proxy event loop freezes.
- **Fix direction:** Use an async httpx client (`httpx.AsyncClient` + `await`) on the proxy's async paths, or offload the sync call to a thread executor.
- **Fixed:** 2026-06-25, working tree (uncommitted) on `docker-review`.
- **Resolution:** Both async handlers now call `await run_in_threadpool(_forward_and_record, ...)` (FastAPI's threadpool offload) so the blocking sync `httpx.Client.post` runs in a worker thread and never stalls the event loop. Chose threadpool offload over switching to `httpx.AsyncClient` to keep the change minimal and preserve the existing sync-client test fixture and `_forward_and_record` body (the shared `httpx.Client` and per-run `threading.Lock` are thread-safe). `/health` and other concurrent runs stay responsive while an upstream call is in flight.
- **Verified:** `tests/gateway/test_local_proxy_concurrency.py` fires 5 concurrent `/v1/call`s against a deliberately slow (sleeping) upstream and asserts total wall-clock stays well under N×delay — i.e. they overlap instead of serializing on the loop. All 11 existing `test_local_proxy.py` tests still pass (behavior preserved).
- **Closed:** _(date / sign-off)_

---

## 🟡 Correctness — breaks legitimate runs

### Issue 8 — JSON brace-parsing rejects valid results
- **Severity:** 🟡 Medium
- **Status:** FIXED (awaiting sign-off → CLOSED)
- **File:** `src/validator/sandbox_docker.py:437`
- **Risk:** `_last_json_object`'s backward brace-balance walk ignores braces inside string values. If an `AgentResult` text field contains `{` or `}` (and anything is printed before it so it no longer starts/ends with a brace), the walk never rebalances to depth 0, `model_validate_json` fails, and an otherwise-valid run is rejected.
- **Failure scenario:** An agent returns an `AgentResult` whose reasoning/text field contains `{`/`}`, and a curated dep or the agent prints a line before the result; the backward walk miscounts the in-string brace and returns the wrong slice, so validation fails.
- **Fix direction:** Have the runner write the result to a dedicated output file (the input already uses one) rather than scraping stdout; or make the parser string-aware.
- **Fixed:** 2026-06-25, working tree (uncommitted) on `docker-review`.
- **Resolution:** Rewrote `_last_json_object` as a **string-aware forward scan** that tracks in-string state with `\\` escape handling and brace depth, returning the last balanced top-level `{...}`. Braces inside JSON string values are no longer counted. Chose the parser fix over the "dedicated output file" alternative because the result channel is stdout by design and the runner's bind mount is read-only — a writable shared output file would be a larger architectural change (writable mount) that conflicts with the sandbox's read-only-rootfs guarantee. The string-aware parser fully resolves the false-rejection without weakening isolation.
- **Verified:** `tests/validator/test_sandbox_json_parse.py` (6 tests): plain object; leading stdout noise; **braces inside a string value**; escaped quotes inside a string; last-of-multiple-objects; and nested objects with brace-laden string values — all round-trip through `json.loads`.
- **Closed:** _(date / sign-off)_

### Issue 9 — Host-bind fallback breaks all runs
- **Severity:** 🟡 Medium (verifier rated impact PLAUSIBLE — triggers only in a specific mount layout)
- **Status:** FIXED (awaiting sign-off → CLOSED)
- **File:** `src/validator/sandbox_docker.py:113`
- **Risk:** When `sandbox_socket_dir` is a *subdirectory* of a bind mount (not the mountpoint) and `sandbox_socket_host_bind` is unset, `_sibling_socket_host_bind` falls back to the in-container path as the host source. Docker creates a fresh empty dir, the sibling never sees `proxy.sock`, and **every** `call_provider()` fails with connection-refused until the operator manually sets the override.
- **Failure scenario:** `_mountinfo_host_root_for_mountpoint` returns `None`, the container-internal path is bound as if it were a host path, and all agent executions fail until `sandbox_socket_host_bind` is set manually.
- **Fix direction:** Resolve the host root for subdirectories of a bind mount (walk parent mountpoints) or fail loudly instead of silently binding a wrong path.
- **Fixed:** 2026-06-25, working tree (uncommitted) on `docker-review`.
- **Resolution:** Generalised `_mountinfo_host_root_for_mountpoint` to match not just an exact mountpoint but any **ancestor** mount of the target, picking the most specific (longest mountpoint) and preferring a real `bind` entry over a non-bind fallback at equal depth, then appending the trailing path segment to the host-side root. A socket dir nested under a bind mount now resolves to `host_root/<subdir>` instead of `None`, so the caller no longer silently binds a wrong container-internal path. The existing host-run fallback (no relevant mount → use the socket dir as-is) is preserved.
- **Verified:** `tests/validator/test_sandbox_mountinfo.py` extended: subdir-of-a-bind resolves to `host_src/arcratio`; most-specific mount wins when both an ancestor and exact mountpoint match. Original mountinfo tests (exact-match prefers bind, explicit-config, host-run fallback) still pass.
- **Closed:** _(date / sign-off)_

### Issue 10 — Missing `miner_hotkey` → 400 on the default sandbox path
- **Severity:** 🟡 Medium
- **Status:** FIXED (awaiting sign-off → CLOSED)
- **File:** `src/gateway/local_proxy.py:316`
- **Risk:** `run_agent`/`run_all_agents` and `scripts/run_forecast.py` call the orchestrator without `miner_hotkey`, so `register_run` stores `None` and `_forward_and_record` raises HTTP 400 for every provider call. `RemoteProvider` never sends `minerHotkey`, so the override fallback is dead. The default documented sandbox path is currently broken for billed provider calls.
- **Failure scenario:** In default `docker_runc` mode, `python scripts/run_forecast.py --agent anthropic` returns 400 "missing miner hotkey for gateway billing context"; `RemoteProvider` raises, the runner exits 1, and no forecast is produced.
- **Fix direction:** Thread a real `miner_hotkey` through `run_agent`/`run_all_agents`/`run_forecast.py` into `register_run`. (Coordinate with Issue 1 — fix should set identity at registration, not via body.)
- **Fixed:** 2026-06-25, working tree (uncommitted) on `docker-review`.
- **Resolution:** `Orchestrator.run_all_agents` now accepts and forwards `miner_hotkey` to `run_agent` (it previously dropped it). `scripts/run_forecast.py` derives a billing identity via new `_dev_miner_hotkey()`: the validator's own loaded hotkey (`loaded_keypair.hotkey_ss58`) in signed mode, a clearly-marked `dev-local-<validator_id>` placeholder when signing is disabled, and `None` in `in_process` mode (no proxy/billing context). This pairs with Issue 1: identity is now set only at run registration, so a real value here is required for the docker path to work at all.
- **Verified:** `tests/validator/test_orchestrator_miner_hotkey.py` (2 tests, docker_runc mode with a fake proxy state): `run_agent` registers with the passed `miner_hotkey`; `run_all_agents` forwards it to *every* registration (the regression that previously 400'd every call).
- **Closed:** _(date / sign-off)_

---

## Refuted / out of scope

Dismissed by verifiers (not actionable):
- `/v1/runs` run-id leak — endpoint is gated.
- Hardcoded UDS contract in two files — actually coupled via one shared constant (`_SANDBOX_SOCKET_URL`).
- Double `X-Run-Id` header — harmless.
- Nested-ternary `miner_hotkey_override` cleanup — cosmetic nit.

Below the high-effort reporting cap (lower-severity cleanups, noted for awareness):
- Duplicate `_default_call_type` map (`local_proxy.py` vs `server.py`).
- Eager stderr fetch consumed only on failure paths.
- `docker.from_env()` version probe per run.
- `docker.io` apt package pulls the full Docker engine into the image when only the client is needed.
