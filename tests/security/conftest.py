"""Shared fixtures for the Docker sandbox lockdown suite.

These tests build the runner image once per session and run sibling
containers per case. Skipped automatically when:

- The Docker SDK is not installed.
- The Docker daemon is unreachable.
- The user did not pass `--docker` (we do not want the build cost on every
  developer's `pytest` invocation).
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--docker",
        action="store_true",
        default=False,
        help="Run the security suite against a real Docker daemon (slow).",
    )


def _docker_available() -> tuple[bool, str]:
    try:
        import docker  # type: ignore
    except ImportError as exc:
        return False, f"docker SDK not installed ({exc})"
    try:
        client = docker.from_env()
        client.ping()
    except Exception as exc:  # noqa: BLE001
        return False, f"docker daemon unreachable ({exc})"
    return True, ""


@pytest.fixture(scope="session")
def docker_enabled(pytestconfig: pytest.Config) -> bool:
    if not pytestconfig.getoption("--docker"):
        pytest.skip("docker security suite disabled (pass --docker to enable)")
    ok, reason = _docker_available()
    if not ok:
        pytest.skip(reason)
    return True


@pytest.fixture(scope="session")
def runner_image(docker_enabled) -> str:
    """Build the agent-runner image once per session, return the tag."""
    import docker  # type: ignore

    client = docker.from_env()
    tag = "almanac/agent-runner:test"
    dockerfile = PROJECT_ROOT / "docker" / "agent-runner.Dockerfile"
    client.images.build(
        path=str(PROJECT_ROOT),
        dockerfile=str(dockerfile.relative_to(PROJECT_ROOT)),
        tag=tag,
        rm=True,
        forcerm=True,
    )
    return tag


@pytest.fixture
def proxy_socket_dir() -> Path:
    # The proxy binds a UNIX domain socket inside this dir. macOS (BSD) caps
    # AF_UNIX paths at 104 bytes, and pytest's tmp_path under
    # /private/var/folders/.../pytest-of-<user>/pytest-NN/<test>/ blows past
    # that, so the bind fails with "AF_UNIX path too long" and every test in
    # this suite errors in the local_proxy fixture. Use a short base dir.
    base = Path(tempfile.mkdtemp(prefix="arc", dir="/tmp"))
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


@pytest.fixture
def upstream_capture():
    """Captures every request the proxy forwards upstream + lets the test
    pre-program the response."""
    state: dict[str, Any] = {"requests": [], "response": None}

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        state["requests"].append({"headers": dict(request.headers), "body": body})
        if state["response"] is not None:
            return state["response"]
        return httpx.Response(
            200,
            json={
                "provider_id": body.get("provider_id"),
                "call_type": body.get("call_type"),
                "data": {"echo": body.get("params", {})},
                "latency_ms": 0,
            },
        )

    return state, _handler


@pytest.fixture
def local_proxy(proxy_socket_dir: Path, upstream_capture):
    """Start the local-proxy FastAPI app on a UDS in `proxy_socket_dir`,
    backed by a stubbed upstream `httpx.Client`."""
    import uvicorn

    from src.core.config import AppConfig
    from src.gateway.local_proxy import create_app

    state, handler = upstream_capture
    transport = httpx.MockTransport(handler)
    upstream_client = httpx.Client(transport=transport, timeout=5.0)

    cfg = AppConfig.load_default()
    cfg.bittensor.signing_required = False  # tests don't need a real wallet
    cfg.validator.sandbox_socket_dir = proxy_socket_dir

    app = create_app(cfg, http_client=upstream_client)
    proxy_state = app.state.proxy
    proxy_state.load_hotkey()

    socket_path = proxy_socket_dir / "proxy.sock"
    uconfig = uvicorn.Config(app, uds=str(socket_path), log_level="warning", lifespan="off")
    server = uvicorn.Server(uconfig)
    thread = threading.Thread(target=server.run, name="proxy-test", daemon=True)
    thread.start()

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if socket_path.exists():
            break
        time.sleep(0.05)
    assert socket_path.exists(), "proxy UDS failed to start"
    from src.validator.forecasting.proxy_socket import make_proxy_socket_connectable

    make_proxy_socket_connectable(socket_path)

    yield proxy_state, state

    server.should_exit = True
    upstream_client.close()


def _docker_socket_reachable() -> bool:
    sock = "/var/run/docker.sock"
    if not os.path.exists(sock):
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            s.connect(sock)
        return True
    except OSError:
        return False
