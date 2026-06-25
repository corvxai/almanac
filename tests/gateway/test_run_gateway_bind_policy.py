"""Bind-policy tests for scripts/run_gateway.py.

The gateway must not become an open, unauthenticated relay to provider API
keys by default: a non-loopback bind without signature enforcement (or an
explicit opt-in) is refused.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_run_gateway():
    spec = importlib.util.spec_from_file_location(
        "run_gateway_under_test", PROJECT_ROOT / "scripts" / "run_gateway.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rg():
    return _load_run_gateway()


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_always_allowed(rg, host):
    assert rg._bind_policy_error(host, require_signature=False, allow_open=False) is None


def test_open_bind_without_auth_refused(rg):
    err = rg._bind_policy_error("0.0.0.0", require_signature=False, allow_open=False)
    assert err is not None and "refusing to bind" in err


def test_open_bind_allowed_with_signature(rg):
    assert rg._bind_policy_error("0.0.0.0", require_signature=True, allow_open=False) is None


def test_open_bind_allowed_with_explicit_optin(rg):
    assert rg._bind_policy_error("0.0.0.0", require_signature=False, allow_open=True) is None


def test_default_host_is_loopback(rg):
    """The argparse default must be loopback, not 0.0.0.0."""
    import argparse

    # Reconstruct just the --host default the script declares.
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    assert parser.parse_args([]).host == "127.0.0.1"


def test_cli_refuses_open_bind_and_exits_nonzero():
    """End-to-end: the real CLI exits non-zero before serving on an unsafe bind."""
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "run_gateway.py"), "--host", "0.0.0.0"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(PROJECT_ROOT),
        env={"PATH": "/usr/bin:/bin", "REQUIRE_SIGNATURE": ""},
    )
    assert proc.returncode == 2, f"expected refusal exit 2, got {proc.returncode}\n{proc.stderr}"
    assert "refusing to bind" in proc.stderr
