"""Docker-based agent sandbox executor.

The validator container spawns a sibling agent-runner container per agent
execution via the `docker.sock` mount. The runner is launched with the
strictest set of flags Docker offers:

- `network_mode="none"` — no IP stack, no DNS, no loopback.
- `read_only=True` — root filesystem is immutable.
- `tmpfs={"/tmp": "size=64m,mode=1777"}` — sole writable area.
- `mem_limit`, `nano_cpus`, `pids_limit` — fail-closed resource ceilings.
- `cap_drop=["ALL"]`, `security_opt=["no-new-privileges:true"]` — kernel
  surface area minimised.
- `user="10001:10001"` — a fixed unprivileged uid, **never root**, even when
  the validator itself runs as root (the proxy UDS is made connectable by this
  uid, see `scripts/run_local_proxy.py`), so a container escape lands
  unprivileged on the host.
- Two read-only **file** binds: the proxy UDS and *this run's* input JSON. The
  whole socket dir is deliberately NOT mounted — that would expose every
  concurrent run's payload and defeat X-Run-Id isolation.
- Optional gVisor (`runtime="runsc"`) for syscall-level isolation.

Stdin: not used for the validator-spawned path (payload is a JSON file bind-
mounted read-only). Manual ``docker run -i`` may still pipe JSON to stdin.
Stdout: `AgentResult.model_dump_json()`.
Stderr: free-form diagnostics (never trusted for the trace).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from uuid import UUID

from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.config import ValidatorConfig
from src.core.schemas import AgentResult

logger = logging.getLogger("forecasting.sandbox_docker")


# Where the runner image expects the validator-local proxy's UDS to be
# mounted (read-only). Pairs with `runner_entrypoint._socket_path_from_env`.
_SANDBOX_SOCKET_DIR = "/run/almanac"
_SANDBOX_SOCKET_URL = f"http+unix://{_SANDBOX_SOCKET_DIR}/proxy.sock"
# Per-run mounts are bound as individual files (not the whole dir) so a sibling
# never sees another run's payload — these are the in-container target paths.
_SANDBOX_SOCKET_FILE = f"{_SANDBOX_SOCKET_DIR}/proxy.sock"
_SANDBOX_INPUT_FILE = f"{_SANDBOX_SOCKET_DIR}/input.json"

# Fixed unprivileged identity for the sandbox. Never root, regardless of the
# validator's own uid — a container escape must not land as host root.
_SANDBOX_RUNNER_UID = 10001
_SANDBOX_RUNNER_GID = 10001

# Upper bound on agent stdout we are willing to buffer/parse. A real
# AgentResult JSON is a few KB; this is a generous ceiling to contain a
# hostile agent flooding stdout (see read path below).
_MAX_SANDBOX_STDOUT_BYTES = 8 * 1024 * 1024


def _decode_proc_mount_token(token: str) -> str:
    """Undo mountinfo(5) octal escapes in a single path token."""
    return (
        token.replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\040", " ")
        .replace("\\134", "\\")
    )


def _mountinfo_host_root_for_mountpoint(
    mountpoint: Path,
    *,
    _mountinfo_path: Path | None = None,
) -> str | None:
    """Return the host-side root path for a bind mount visible at `mountpoint`.

    When the validator runs in Docker, sibling containers are created by the
    host daemon. Binds must use the **host** path of the proxy socket directory
    (the same inode the validator sees via compose), not e.g. ``/var/run/almanac``
    resolved on the host filesystem.

    Handles the case where `mountpoint` is a *subdirectory* of a bind mount
    rather than the mountpoint itself: the closest ancestor mount is matched and
    the trailing path segment is appended to its host-side root. Without this,
    a socket dir nested under a bind mount would resolve to ``None`` and the
    caller would silently bind a wrong (container-internal) path.
    """
    target = str(mountpoint.resolve())

    def _is_ancestor(mp: str, child: str) -> bool:
        return child == mp or child.startswith(mp.rstrip("/") + "/")

    # Among all mounts whose mountpoint is `target` or an ancestor dir of it,
    # pick the most specific (longest mountpoint); prefer a real bind entry over
    # a non-bind fallback (e.g. the overlay root) at equal specificity.
    best_root: str | None = None
    best_mp: str | None = None
    best_is_bind = False
    mi_path = _mountinfo_path or Path("/proc/self/mountinfo")
    try:
        with mi_path.open(encoding="utf-8") as fh:
            for line in fh:
                if " - " not in line:
                    continue
                left, right = line.split(" - ", 1)
                rparts = right.split()
                if len(rparts) < 2:
                    continue
                fstype = rparts[0]
                parts = left.split()
                if len(parts) < 5:
                    continue
                root = _decode_proc_mount_token(parts[3])
                mp = _decode_proc_mount_token(parts[4])
                if not _is_ancestor(mp, target):
                    continue
                is_bind = fstype == "bind"
                better = (
                    best_mp is None
                    or len(mp) > len(best_mp)
                    or (len(mp) == len(best_mp) and is_bind and not best_is_bind)
                )
                if better:
                    best_root, best_mp, best_is_bind = root, mp, is_bind
    except OSError:
        return None
    if best_root is None or best_mp is None:
        return None
    rel = target[len(best_mp):].lstrip("/")
    if not rel:
        return best_root
    return str(Path(best_root) / rel)


# A mountinfo root landing inside a named volume's data dir. The root token is
# relative to the backing device's filesystem root, so when the daemon keeps
# /var/lib/docker on its own device (e.g. Docker Desktop's VM) the resolved
# path is missing that prefix — the daemon's Mountpoint for the volume is the
# authoritative host path.
_VOLUME_DATA_ROOT_RE = re.compile(r"(?:^|/)docker/volumes/([^/]+)/_data(/.*)?$")


def _sibling_socket_host_bind(cfg: ValidatorConfig, client: Any | None = None) -> str:
    """Filesystem path on the Docker *host* to bind into the agent container."""
    if cfg.sandbox_socket_host_bind:
        return cfg.sandbox_socket_host_bind
    resolved = _mountinfo_host_root_for_mountpoint(cfg.sandbox_socket_dir)
    if resolved:
        volume_match = _VOLUME_DATA_ROOT_RE.search(resolved)
        if volume_match and client is not None:
            try:
                volume = client.volumes.get(volume_match.group(1))
                mountpoint = (volume.attrs or {}).get("Mountpoint")
            except Exception:  # noqa: BLE001 — fall back to the raw path
                mountpoint = None
            if mountpoint:
                return mountpoint.rstrip("/") + (volume_match.group(2) or "")
        return resolved
    return str(cfg.sandbox_socket_dir)


def _sandbox_runner_user() -> str:
    """uid:gid for the sandbox — always a fixed non-root identity.

    The previous logic dropped to ``0:0`` whenever the validator ran as root
    (the default compose deployment), silently running every untrusted agent as
    in-container root so a runc/kernel escape would land as root on the host.
    We pin an unprivileged uid instead; socket access no longer requires root
    because the proxy UDS is made world-connectable (see run_local_proxy).
    """
    return f"{_SANDBOX_RUNNER_UID}:{_SANDBOX_RUNNER_GID}"


def _sandbox_volumes(bind_src: str, payload_name: str) -> dict[str, dict[str, str]]:
    """Read-only mount plan for a sibling: ONLY the proxy socket and this run's input.

    `bind_src` is the Docker *host* path of the proxy socket directory. Binding
    that whole directory would expose ``inputs/`` — every concurrent run's
    payload — into each sibling, letting a hostile agent read a neighbour's
    RUN_ID and agent_code and issue calls under it (defeating X-Run-Id). Instead
    bind the two files individually so no sibling can enumerate or read another
    run. The input file lives at ``<socket_dir>/inputs/<name>`` on the host and
    is presented to the runner at a fixed in-container path.
    """
    return {
        f"{bind_src}/proxy.sock": {"bind": _SANDBOX_SOCKET_FILE, "mode": "ro"},
        f"{bind_src}/inputs/{payload_name}": {"bind": _SANDBOX_INPUT_FILE, "mode": "ro"},
    }


_LOG_TAIL_LINES = 400
_LOG_PRINT_BYTES = 12_000


def _sandbox_runner_logs_verbose() -> bool:
    """When true, print agent-runner log tails after a normal exit (quiet by default)."""
    v = os.environ.get("FORECASTING_SANDBOX_RUNNER_LOGS_VERBOSE", "").strip().lower()
    return v in {"1", "true", "yes", "on"}


def _docker_cli_binary() -> str | None:
    """Return the Docker CLI path.

    ``shutil.which("docker")`` is often wrong inside minimal containers (empty
    or reduced ``PATH``). Prefer well-known install locations.
    """
    for p in ("/usr/bin/docker", "/bin/docker"):
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return shutil.which("docker")


def _wait_container_exit_docker_cli_or_sdk(container: Any, timeout_sec: int) -> dict[str, Any]:
    """Block until the sibling exits.

    Prefer ``docker wait`` subprocess: docker-py's ``container.wait()`` holds an
    HTTP connection open for the whole run and often hits urllib3 ReadTimeout
    (~60s) even when the sandbox timeout is much larger (docker/docker-py#1950).
    The CLI uses the engine client and avoids that failure mode.
    """
    cid = getattr(container, "id", None) or ""
    if not cid:
        raise RuntimeError("container has no id before wait")
    docker_bin = _docker_cli_binary()
    if docker_bin:
        try:
            proc = subprocess.run(
                [docker_bin, "wait", cid],
                capture_output=True,
                text=True,
                timeout=float(timeout_sec),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"docker wait exceeded {timeout_sec}s (agent still running?)"
            ) from exc
        if proc.returncode != 0:
            err = (proc.stderr or "").strip()
            out = (proc.stdout or "").strip()
            raise RuntimeError(
                f"docker wait failed (CLI exit {proc.returncode}): {err or out!r}"
            )
        lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
        if not lines:
            raise RuntimeError(
                f"docker wait produced no stdout (stderr={proc.stderr!r})"
            )
        try:
            code = int(lines[-1])
        except ValueError as ve:
            raise RuntimeError(
                f"docker wait unexpected stdout: {proc.stdout!r}"
            ) from ve
        return {"StatusCode": code}
    return container.wait(timeout=timeout_sec)


def _emit_sibling_container_logs(container: Any, reason: str) -> None:
    """Print agent-runner stdout/stderr tails to the validator process stdout.

    Sibling containers do not stream into ``docker compose`` logs unless we
    pull ``docker logs`` here or you run ``docker logs`` yourself (see README).
    """
    cid = getattr(container, "id", "") or ""
    short = f"{cid[:12]}…" if len(cid) > 12 else (cid or "?")
    bar = "-" * 72
    print(f"\n{bar}\n[almanac] agent-runner logs  {reason}\n  container: {short}\n{bar}", flush=True)
    try:
        so = container.logs(
            stdout=True, stderr=False, tail=_LOG_TAIL_LINES, timestamps=False
        )
        se = container.logs(
            stdout=False, stderr=True, tail=_LOG_TAIL_LINES, timestamps=False
        )
    except Exception as exc:
        print(f"(could not read container logs: {exc!r})", flush=True)
        return
    out_t = (so or b"").decode("utf-8", errors="replace")
    err_t = (se or b"").decode("utf-8", errors="replace")
    if out_t.strip():
        print("--- stdout (tail) ---", flush=True)
        print(out_t[-_LOG_PRINT_BYTES:], flush=True)
    if err_t.strip():
        print("--- stderr (tail) ---", flush=True)
        print(err_t[-_LOG_PRINT_BYTES:], flush=True)
    if not out_t.strip() and not err_t.strip():
        print("(no log output captured yet)", flush=True)
    print(bar + "\n", flush=True)


def run_agent_in_container(
    agent: BaseAgent,
    ctx: ForecastingContext,
    cfg: ValidatorConfig,
    run_id: UUID,
) -> AgentResult:
    """Launch a sibling container running the agent and return its result."""
    try:
        import docker  # type: ignore
        from docker.errors import DockerException  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "The 'docker' Python SDK is required to use docker_runc / "
            "docker_gvisor sandbox modes. Install it via requirements.txt."
        ) from exc

    agent_module = type(agent).__module__
    agent_class = type(agent).__qualname__

    payload = {
        "event": ctx.event.model_dump(mode="json"),
        "agent_module": agent_module,
        "agent_class": agent_class,
    }
    inline_code = getattr(agent, "_almanac_agent_source_code", None)
    inline_class = getattr(agent, "_almanac_agent_source_class", None)
    if isinstance(inline_code, str) and inline_code.strip():
        payload["agent_code"] = inline_code
        if isinstance(inline_class, str) and inline_class.strip():
            payload["inline_class"] = inline_class
    # JSON payload is written to the shared UDS bind mount so the sibling can
    # read it via FORECASTING_RUNNER_INPUT_FILE. Docker stdin attach + EOF is
    # unreliable; ``sys.stdin.read()`` in the runner would otherwise hang.
    payload_name = f".almanac_stdin_{run_id}.json"
    payload_host_dir = cfg.sandbox_socket_dir / "inputs"
    payload_host_path = payload_host_dir / payload_name

    # docker.from_env() defaults to a 60s HTTP read timeout. Raise the client
    # timeout so long ``container.wait()`` calls are less likely to hit
    # ReadTimeout before the sandbox deadline (docker/docker-py#1950 / #1966).
    _sandbox_wait = int(cfg.sandbox_timeout_seconds)
    _client_timeout = max(120, _sandbox_wait + 120)
    try:
        client = docker.from_env(timeout=_client_timeout)
    except DockerException as exc:
        err_l = str(exc).lower()
        if "permission denied" in err_l or (
            getattr(exc.__cause__, "errno", None) == 13
        ):
            raise RuntimeError(
                "Cannot connect to the Docker daemon (permission denied on the "
                "API socket). On Linux, add your user to the 'docker' group and "
                "start a new session, or run with --sandbox in_process to skip "
                "containers for local development."
            ) from exc
        raise

    runtime = "runsc" if cfg.sandbox_type == "docker_gvisor" else None

    bind_src = _sibling_socket_host_bind(cfg, client)
    runner_user = _sandbox_runner_user()

    container_kwargs: dict[str, Any] = dict(
        image=cfg.sandbox_image,
        stdin_open=False,
        detach=True,
        network_mode="none",
        read_only=True,
        tmpfs={"/tmp": "size=64m,mode=1777"},
        mem_limit=f"{cfg.sandbox_memory_mb}m",
        nano_cpus=int(cfg.sandbox_cpus * 1e9),
        pids_limit=cfg.sandbox_pids_limit,
        cap_drop=["ALL"],
        security_opt=["no-new-privileges:true"],
        user=runner_user,
        volumes=_sandbox_volumes(bind_src, payload_name),
        environment={
            "SANDBOX_PROXY_URL": _SANDBOX_SOCKET_URL,
            "RUN_ID": str(run_id),
            "PYTHONUNBUFFERED": "1",
            "FORECASTING_RUNNER_INPUT_FILE": _SANDBOX_INPUT_FILE,
        },
        labels={
            "almanac.run_id": str(run_id),
            "almanac.agent_module": agent_module,
            "almanac.agent_class": agent_class,
        },
    )
    if runtime is not None:
        container_kwargs["runtime"] = runtime

    container: Any | None = None
    try:
        payload_host_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Owner-only dir: it holds orchestrator-injected agent source and
            # event payloads for in-flight runs. World-readable (0o755) let any
            # other local user or container on the host read them. 0o700 keeps
            # other host users out (they cannot traverse it).
            os.chmod(payload_host_dir, 0o700)
        except OSError:
            pass
        payload_host_path.write_text(
            json.dumps(payload, separators=(",", ":")),
            encoding="utf-8",
        )
        try:
            # The file itself is bind-mounted read-only into the sandbox, which
            # runs as the unprivileged uid 10001 — distinct from the validator's
            # uid — so it must be world-readable to be read through the bind. It
            # stays protected from other host users by the 0o700 parent dir, and
            # from other runs by per-file mounting (see _sandbox_volumes).
            os.chmod(payload_host_path, 0o644)
        except OSError:
            pass
        container = client.containers.create(**container_kwargs)
        cid = (container.id or "")[:12]
        print(
            f"[almanac] spawning agent-runner sibling: container={cid}… "
            f"image={cfg.sandbox_image} agent={agent_module}:{agent_class} "
            f"host_bind={bind_src!r} -> container:{_SANDBOX_SOCKET_DIR}\n"
            f"  (host: docker logs -f {container.id})",
            flush=True,
        )
        logger.info(
            "Launching sandbox: container=%s image=%s runtime=%s run_id=%s agent=%s.%s host_bind=%s user=%s",
            container.id,
            cfg.sandbox_image,
            runtime or "runc",
            run_id,
            agent_module,
            agent_class,
            bind_src,
            runner_user,
        )

        container.start()

        try:
            result = _wait_container_exit_docker_cli_or_sdk(container, _sandbox_wait)
        except TimeoutError as exc:
            _emit_sibling_container_logs(container, "docker wait subprocess timed out")
            _force_kill(container)
            raise RuntimeError(
                f"Agent sandbox timed out after {_sandbox_wait}s"
            ) from exc
        except Exception as exc:
            _emit_sibling_container_logs(
                container, "wait failed (docker CLI or SDK)"
            )
            _force_kill(container)
            raise RuntimeError(
                "Agent sandbox wait failed: "
                f"{exc!r}"
            ) from exc

        exit_code = result.get("StatusCode", -1)
        if _sandbox_runner_logs_verbose():
            _emit_sibling_container_logs(
                container, f"runner finished (exit_code={exit_code})"
            )

        # The agent is untrusted: a hostile (or buggy) agent can flood stdout to
        # force the validator to buffer, decode, and JSON-scan an arbitrarily
        # large string. Stream stdout and abort the moment it crosses the ceiling
        # so we never materialise the whole blob in memory. A legitimate
        # AgentResult is a few KB. stderr is only ever used as a bounded tail, so
        # read it with `tail` rather than pulling the whole (also untrusted) log.
        try:
            stdout_raw = _read_capped_stdout(container, _MAX_SANDBOX_STDOUT_BYTES)
        except _SandboxStdoutTooLarge as exc:
            raise RuntimeError(
                f"Agent sandbox stdout exceeded {_MAX_SANDBOX_STDOUT_BYTES} bytes "
                f"({exc.total_bytes}+ bytes); refusing to buffer/parse."
            ) from exc
        stderr = container.logs(stdout=False, stderr=True, tail=_LOG_TAIL_LINES)

        if exit_code != 0:
            stderr_text = (stderr or b"").decode("utf-8", errors="replace")[-4000:]
            raise RuntimeError(
                f"Agent sandbox exited with code {exit_code}. stderr tail:\n{stderr_text}"
            )

        stdout_text = stdout_raw.decode("utf-8", errors="replace").strip()
        if not stdout_text:
            stderr_text = (stderr or b"").decode("utf-8", errors="replace")[-1000:]
            raise RuntimeError(
                f"Agent sandbox produced no stdout. stderr tail:\n{stderr_text}"
            )

        try:
            return AgentResult.model_validate_json(_last_json_object(stdout_text))
        except Exception as exc:
            _emit_sibling_container_logs(container, "failed to parse AgentResult JSON from stdout")
            raise RuntimeError(
                f"Failed to parse AgentResult from sandbox stdout: {exc}\n"
                f"stdout tail: {stdout_text[-1000:]}"
            ) from exc
    finally:
        try:
            payload_host_path.unlink(missing_ok=True)
        except OSError:
            pass
        if container is not None:
            _safe_remove(container)


class _SandboxStdoutTooLarge(RuntimeError):
    """Raised when agent stdout exceeds the buffer ceiling mid-stream."""

    def __init__(self, total_bytes: int) -> None:
        self.total_bytes = total_bytes
        super().__init__(f"sandbox stdout exceeded cap at {total_bytes}+ bytes")


def _cap_log_stream(chunks: Iterable[bytes], max_bytes: int) -> bytes:
    """Accumulate `chunks` but abort as soon as the total exceeds `max_bytes`.

    Pulling ``container.logs(stdout=True)`` in one shot buffers the agent's
    entire (possibly hostile) stdout into memory before any size check — an
    untrusted agent can flood it to OOM the validator. Streaming chunk-by-chunk
    and bailing early bounds the buffer to ~max_bytes + one chunk.
    """
    out: list[bytes] = []
    total = 0
    for chunk in chunks:
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            raise _SandboxStdoutTooLarge(total)
        out.append(chunk)
    return b"".join(out)


def _read_capped_stdout(container: Any, max_bytes: int) -> bytes:
    """Read the sibling's stdout without buffering more than `max_bytes`."""
    try:
        stream = container.logs(stdout=True, stderr=False, stream=True, follow=False)
    except TypeError:
        # docker-py too old for stream kw: fall back to a bounded one-shot read.
        data = container.logs(stdout=True, stderr=False) or b""
        if len(data) > max_bytes:
            raise _SandboxStdoutTooLarge(len(data))
        return data
    return _cap_log_stream(stream, max_bytes)


def _force_kill(container: Any) -> None:
    try:
        container.kill(signal="SIGKILL")
    except Exception:
        pass


def _safe_remove(container: Any) -> None:
    try:
        container.remove(force=True)
    except Exception:
        logger.debug("container.remove failed for %s", getattr(container, "id", "?"), exc_info=True)


def _last_json_object(text: str) -> str:
    """Return the last balanced `{...}` JSON object in `text`.

    The runner's ENTRYPOINT writes a single JSON payload to stdout, but the
    agent's own code (or `print()` debug from any of the curated runner deps
    like httpx) may also write to stdout before that. The real payload is
    always the trailing balanced `{...}`. Be tolerant.

    Brace counting is **string-aware**: braces appearing inside JSON string
    values (e.g. a reasoning field containing ``"use {curly} braces"``) are
    ignored, with standard ``\\`` escape handling. A naive brace walk would
    miscount those and reject an otherwise-valid AgentResult.
    """
    text = text.strip()
    start: int | None = None
    last: str | None = None
    depth = 0
    in_str = False
    escaped = False
    for i, ch in enumerate(text):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    last = text[start : i + 1]
    return last if last is not None else text  # last-ditch — let the parser error
