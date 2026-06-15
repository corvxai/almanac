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
- `user="<host_uid>:<host_gid>"` when the validator runs unprivileged
  (fallback `10001:10001` when host uid is root).
- Single read-only bind mount: the validator-local proxy's UDS dir.
- Optional gVisor (`runtime="runsc"`) for syscall-level isolation.

Stdin: not used for the validator-spawned path (payload is a JSON file on the
shared bind mount). Manual ``docker run -i`` may still pipe JSON to stdin.
Stdout: `AgentResult.model_dump_json()`.
Stderr: free-form diagnostics (never trusted for the trace).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any
from uuid import UUID

from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.config import ValidatorConfig
from src.core.schemas import AgentResult

logger = logging.getLogger("arcratio.sandbox_docker")


# Where the runner image expects the validator-local proxy's UDS to be
# mounted (read-only). Pairs with `runner_entrypoint._socket_path_from_env`.
_SANDBOX_SOCKET_DIR = "/run/arcratio"
_SANDBOX_SOCKET_URL = f"http+unix://{_SANDBOX_SOCKET_DIR}/proxy.sock"
_SANDBOX_INPUT_DIR = f"{_SANDBOX_SOCKET_DIR}/inputs"

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
    (the same inode the validator sees via compose), not e.g. ``/var/run/arcratio``
    resolved on the host filesystem.
    """
    target = str(mountpoint.resolve())
    bind_match: str | None = None
    fallback: str | None = None
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
                if mp != target:
                    continue
                if fstype == "bind":
                    bind_match = root
                else:
                    fallback = root
    except OSError:
        return None
    return bind_match or fallback


def _sibling_socket_host_bind(cfg: ValidatorConfig) -> str:
    """Filesystem path on the Docker *host* to bind into the agent container."""
    if cfg.sandbox_socket_host_bind:
        return cfg.sandbox_socket_host_bind
    resolved = _mountinfo_host_root_for_mountpoint(cfg.sandbox_socket_dir)
    if resolved:
        return resolved
    return str(cfg.sandbox_socket_dir)


_LOG_TAIL_LINES = 400
_LOG_PRINT_BYTES = 12_000


def _sandbox_runner_logs_quiet() -> bool:
    """When true, do not print agent-runner log tails after a normal exit."""
    v = os.environ.get("ARCRATIO_SANDBOX_RUNNER_LOGS_QUIET", "").strip().lower()
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
    print(f"\n{bar}\n[arcratio] agent-runner logs  {reason}\n  container: {short}\n{bar}", flush=True)
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
    inline_code = getattr(agent, "_arcratio_agent_source_code", None)
    inline_class = getattr(agent, "_arcratio_agent_source_class", None)
    if isinstance(inline_code, str) and inline_code.strip():
        payload["agent_code"] = inline_code
        if isinstance(inline_class, str) and inline_class.strip():
            payload["inline_class"] = inline_class
    # JSON payload is written to the shared UDS bind mount so the sibling can
    # read it via ARCRATIO_RUNNER_INPUT_FILE. Docker stdin attach + EOF is
    # unreliable; ``sys.stdin.read()`` in the runner would otherwise hang.
    payload_name = f".arcratio_stdin_{run_id}.json"
    payload_host_dir = cfg.sandbox_socket_dir / "inputs"
    payload_host_path = payload_host_dir / payload_name
    payload_runner_path = f"{_SANDBOX_INPUT_DIR}/{payload_name}"

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

    bind_src = _sibling_socket_host_bind(cfg)

    host_uid = os.getuid()
    host_gid = os.getgid()
    runner_user = f"{host_uid}:{host_gid}" if host_uid != 0 else "10001:10001"

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
        volumes={
            bind_src: {
                "bind": _SANDBOX_SOCKET_DIR,
                "mode": "ro",
            }
        },
        environment={
            "SANDBOX_PROXY_URL": _SANDBOX_SOCKET_URL,
            "RUN_ID": str(run_id),
            "PYTHONUNBUFFERED": "1",
            "ARCRATIO_RUNNER_INPUT_FILE": payload_runner_path,
        },
        labels={
            "arcratio.run_id": str(run_id),
            "arcratio.agent_module": agent_module,
            "arcratio.agent_class": agent_class,
        },
    )
    if runtime is not None:
        container_kwargs["runtime"] = runtime

    container: Any | None = None
    try:
        payload_host_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(payload_host_dir, 0o755)
        except OSError:
            pass
        payload_host_path.write_text(
            json.dumps(payload, separators=(",", ":")),
            encoding="utf-8",
        )
        container = client.containers.create(**container_kwargs)
        cid = (container.id or "")[:12]
        print(
            f"[arcratio] spawning agent-runner sibling: container={cid}… "
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
        if not _sandbox_runner_logs_quiet():
            _emit_sibling_container_logs(
                container, f"runner finished (exit_code={exit_code})"
            )

        stdout = container.logs(stdout=True, stderr=False)
        stderr = container.logs(stdout=False, stderr=True)

        if exit_code != 0:
            stderr_text = (stderr or b"").decode("utf-8", errors="replace")[-4000:]
            raise RuntimeError(
                f"Agent sandbox exited with code {exit_code}. stderr tail:\n{stderr_text}"
            )

        # The agent is untrusted: a hostile (or buggy) agent can flood stdout
        # to force the validator to buffer, decode, and JSON-scan an arbitrarily
        # large string (the O(n) `_last_json_object` walk + pydantic parse). A
        # legitimate AgentResult is small, so reject anything implausibly large.
        stdout_raw = stdout or b""
        if len(stdout_raw) > _MAX_SANDBOX_STDOUT_BYTES:
            raise RuntimeError(
                f"Agent sandbox stdout exceeded {_MAX_SANDBOX_STDOUT_BYTES} bytes "
                f"({len(stdout_raw)} bytes); refusing to parse."
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
    """Return the last `{...}` JSON object in `text`.

    The runner's ENTRYPOINT writes a single JSON payload to stdout, but the
    agent's own code (or `print()` debug from any of the curated runner deps
    like httpx) may also write to stdout before that. The real payload is
    always the trailing balanced `{...}`. Be tolerant.
    """
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    # Walk backwards looking for a balanced `{...}` at the tail.
    depth = 0
    end = len(text)
    for i in range(len(text) - 1, -1, -1):
        ch = text[i]
        if ch == "}":
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0:
                return text[i:end]
    return text  # last-ditch — let the JSON parser produce the error
