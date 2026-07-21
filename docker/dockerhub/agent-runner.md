# almanacai/agent-runner

**Sandboxed forecasting agents for prediction markets.**

The hardened sandbox that executes a single, untrusted forecasting agent for the
Almanac subnet. Validators spawn this image as short-lived sibling containers to
run miner-submitted agent code safely; miners pull the same image to test their
agent in the exact environment a validator will use.

> This is infrastructure for the [Almanac](https://almanac.market) forecasting
> subnet. It does **not** hold any keys and makes **no** outbound network calls.

---

## What it does

1. Reads one agent + event payload (bind-mounted read-only as a single file).
2. Runs the agent's `predict(ctx)` with the network disabled.
3. Routes every provider/LLM call through the validator-local signing proxy over
   a UNIX domain socket — the only channel out of the sandbox.
4. Writes the resulting `AgentResult` to stdout for the validator to record.

## Security posture

This container is the trust boundary for untrusted miner code. It runs:

- `--network=none` — no IP stack, no DNS, no loopback.
- Read-only root filesystem; a small `tmpfs` at `/tmp` is the only writable area.
- Non-root `USER 10001:10001` (never root).
- `cap_drop=ALL`, `no-new-privileges`, plus `mem`/`cpu`/`pids` ceilings.
- Exactly two read-only mounts per run: the proxy socket and that run's input
  file — so one agent can never read another run's payload.
- Optional gVisor (`runtime=runsc`) for syscall-level isolation.

## Tags

- `latest` — most recent release (multi-arch: `linux/amd64`, `linux/arm64`).
- `X.Y.Z` — pinned release versions.
- `sha-<gitsha>` — exact commit builds.

A multi-arch manifest means `docker pull` returns the right architecture
automatically — amd64 for Linux validator hosts, arm64 for Apple-Silicon dev
machines.

## Pull

```bash
docker pull almanacai/agent-runner:latest
```

**Pin by digest for production.** Validators should reference an immutable digest
rather than a moving tag so the sandbox image can't be silently swapped:

```bash
docker pull almanacai/agent-runner@sha256:<digest>
```

Point the validator at it via `cfg.sandbox_image` (or the matching env/config),
e.g. `almanacai/agent-runner@sha256:<digest>`.

## How it's used

You normally don't run this image by hand — the validator launches it per agent
execution. To test an agent locally in this sandbox, use the validator's docker
sandbox mode (see the repository's miner guide / `scripts/run_forecast.py`).

> **Architecture must match the validator host.** Thanks to the multi-arch
> manifest this is automatic on pull; only override `DOCKER_DEFAULT_PLATFORM`
> if you are intentionally cross-running.

## Links

- Source & guides: https://github.com/corvxai/almanac
- Companion image: [`almanacai/validator`](https://hub.docker.com/r/almanacai/validator)
