"""Adversarial smoke tests for the Docker sandbox.

Each case launches a real sibling container running an attack agent (see
`src/agent/examples/_attacks/`) and asserts the failure mode.

Skipped unless `pytest --docker` is passed and the Docker daemon is up.
The runner image is built once per session via the `runner_image` fixture.

Mirrors the network/FS/resource attacks from the Numinous security suite,
with one arcratio-specific addition: the read-only Bittensor wallet boundary.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.config import AppConfig
from src.core.events import Event
from src.core.schemas import EventCategory


def _sample_event() -> Event:
    return Event(
        title="Test event",
        description="Used by sandbox lockdown tests; never resolved.",
        category=EventCategory.OTHER,
        resolution_criteria="N/A",
        resolution_deadline=datetime(2030, 1, 1, tzinfo=timezone.utc),
    )


def _build_cfg(socket_dir: Path, sandbox_image: str) -> AppConfig:
    cfg = AppConfig.load_default()
    cfg.validator.sandbox_type = "docker_runc"
    cfg.validator.sandbox_image = sandbox_image
    cfg.validator.sandbox_socket_dir = socket_dir
    cfg.validator.sandbox_timeout_seconds = 30
    cfg.validator.sandbox_memory_mb = 256
    cfg.validator.sandbox_cpus = 0.5
    cfg.validator.sandbox_pids_limit = 64
    cfg.bittensor.signing_required = False
    return cfg


def _run_attack_in_sandbox(
    cfg: AppConfig,
    proxy_state,
    agent: BaseAgent,
    *,
    track: str = "MAIN",
):
    """Register a run, launch the sandbox with the attack agent, return the
    raised exception (or `None` on success).
    """
    from src.validator.sandbox_docker import run_agent_in_container

    event = _sample_event()
    ctx = ForecastingContext(event=event, gateway=None)  # type: ignore[arg-type]
    run_id = uuid4()
    # Register with a miner_hotkey: the proxy derives the signed billing identity
    # solely from the run registration (the agent body can no longer override it),
    # so a provider call with no registered hotkey is rejected with a 400.
    proxy_state.register_run(run_id, track, miner_hotkey="5TestMiner")
    try:
        run_agent_in_container(agent, ctx, cfg.validator, run_id)
    except Exception as exc:  # noqa: BLE001
        return exc
    finally:
        proxy_state.pop_calls(run_id)
    return None


# ---------------------------------------------------------------------------
# Network attacks
# ---------------------------------------------------------------------------


class TestNetworkLockdown:
    def test_raw_socket_blocked(self, runner_image, local_proxy, proxy_socket_dir):
        from src.agent.examples._attacks.network_socket import NetworkSocketAttack

        proxy_state, _ = local_proxy
        cfg = _build_cfg(proxy_socket_dir, runner_image)
        exc = _run_attack_in_sandbox(cfg, proxy_state, NetworkSocketAttack())
        assert exc is not None, "raw socket should fail under --network=none"

    def test_urllib_blocked(self, runner_image, local_proxy, proxy_socket_dir):
        from src.agent.examples._attacks.network_urllib import NetworkUrllibAttack

        proxy_state, _ = local_proxy
        cfg = _build_cfg(proxy_socket_dir, runner_image)
        exc = _run_attack_in_sandbox(cfg, proxy_state, NetworkUrllibAttack())
        assert exc is not None

    def test_dns_blocked(self, runner_image, local_proxy, proxy_socket_dir):
        from src.agent.examples._attacks.network_dns import NetworkDnsAttack

        proxy_state, _ = local_proxy
        cfg = _build_cfg(proxy_socket_dir, runner_image)
        exc = _run_attack_in_sandbox(cfg, proxy_state, NetworkDnsAttack())
        assert exc is not None


# ---------------------------------------------------------------------------
# Filesystem attacks
# ---------------------------------------------------------------------------


class TestFilesystemLockdown:
    def test_etc_passwd_is_container_only(self, runner_image, local_proxy, proxy_socket_dir):
        """`/etc/passwd` is readable but only contains the runner image's
        UIDs — never any host user names. Asserts mount-namespace isolation.
        """
        from src.agent.examples._attacks.fs_etc_passwd import FsEtcPasswdAttack

        proxy_state, _ = local_proxy
        cfg = _build_cfg(proxy_socket_dir, runner_image)
        from src.validator.sandbox_docker import run_agent_in_container

        event = _sample_event()
        ctx = ForecastingContext(event=event, gateway=None)  # type: ignore[arg-type]
        run_id = uuid4()
        proxy_state.register_run(run_id, "MAIN", miner_hotkey="5TestMiner")
        try:
            result = run_agent_in_container(FsEtcPasswdAttack(), ctx, cfg.validator, run_id)
        finally:
            proxy_state.pop_calls(run_id)

        passwd = result.reasoning
        assert "root:x:0:0:" in passwd
        # Sanity: host-only accounts must not appear.
        for forbidden in ("andy", "ubuntu", "ec2-user"):
            assert forbidden not in passwd

    def test_wallet_not_visible(self, runner_image, local_proxy, proxy_socket_dir):
        from src.agent.examples._attacks.fs_wallet import FsWalletAttack

        proxy_state, _ = local_proxy
        cfg = _build_cfg(proxy_socket_dir, runner_image)
        from src.validator.sandbox_docker import run_agent_in_container

        event = _sample_event()
        ctx = ForecastingContext(event=event, gateway=None)  # type: ignore[arg-type]
        run_id = uuid4()
        proxy_state.register_run(run_id, "MAIN", miner_hotkey="5TestMiner")
        try:
            result = run_agent_in_container(FsWalletAttack(), ctx, cfg.validator, run_id)
        finally:
            proxy_state.pop_calls(run_id)
        assert "wallet not visible" in result.reasoning


# ---------------------------------------------------------------------------
# Resource attacks
# ---------------------------------------------------------------------------


class TestResourceLockdown:
    def test_fork_bomb_blocked_by_pids_limit(self, runner_image, local_proxy, proxy_socket_dir):
        from src.agent.examples._attacks.fork_bomb import ForkBombAttack

        proxy_state, _ = local_proxy
        cfg = _build_cfg(proxy_socket_dir, runner_image)
        exc = _run_attack_in_sandbox(cfg, proxy_state, ForkBombAttack())
        assert exc is not None

    def test_memory_hog_oom_killed(self, runner_image, local_proxy, proxy_socket_dir):
        from src.agent.examples._attacks.memory_hog import MemoryHogAttack

        proxy_state, _ = local_proxy
        cfg = _build_cfg(proxy_socket_dir, runner_image)
        exc = _run_attack_in_sandbox(cfg, proxy_state, MemoryHogAttack())
        assert exc is not None


# ---------------------------------------------------------------------------
# Positive end-to-end run
# ---------------------------------------------------------------------------


class TestPositiveEndToEnd:
    def test_simple_agent_completes(self, runner_image, local_proxy, proxy_socket_dir):
        """A non-attack agent runs through Docker → UDS → local proxy →
        (stubbed) central gateway and produces a parseable AgentResult."""
        from src.agent.examples.v1_agent1_llm_only import V1Agent1LLMOnly

        proxy_state, _captured = local_proxy
        cfg = _build_cfg(proxy_socket_dir, runner_image)
        from src.validator.sandbox_docker import run_agent_in_container

        event = _sample_event()
        ctx = ForecastingContext(event=event, gateway=None)  # type: ignore[arg-type]
        run_id = uuid4()
        proxy_state.register_run(run_id, "MAIN", miner_hotkey="5TestMiner")
        try:
            result = run_agent_in_container(V1Agent1LLMOnly(), ctx, cfg.validator, run_id)
            calls = proxy_state.pop_calls(run_id)
        finally:
            # In case the test failed before pop_calls.
            try:
                proxy_state.pop_calls(run_id)
            except Exception:
                pass

        assert 0.0 <= result.prediction <= 1.0
        assert calls, "proxy should have recorded at least one provider call"
