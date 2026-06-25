"""Unit tests for the sandbox mount plan and runner identity (#2, #3).

These assert the *plan* the validator hands to Docker. The live container
behaviour (sibling actually reads its input and connects to the proxy over the
UDS) is covered by the Linux `pytest tests/security --docker` job — that round
trip can't run on macOS Docker Desktop (host UDS bind returns ENOTSUP).
"""

from __future__ import annotations

from src.validator import sandbox_docker as sd


def test_runner_user_is_fixed_nonroot():
    # Never 0:0 — a container escape must not land as host root, even when the
    # validator itself runs as root (the default compose deployment).
    assert sd._sandbox_runner_user() == "10001:10001"
    assert not sd._sandbox_runner_user().startswith("0:")


def test_volumes_bind_only_socket_and_own_input():
    vols = sd._sandbox_volumes("/host/run/arcratio", ".arcratio_stdin_RUN_A.json")

    # Exactly two file binds — never the whole socket dir.
    assert len(vols) == 2
    assert "/host/run/arcratio" not in vols  # the dir itself is NOT mounted

    assert vols["/host/run/arcratio/proxy.sock"] == {
        "bind": "/run/arcratio/proxy.sock",
        "mode": "ro",
    }
    assert vols["/host/run/arcratio/inputs/.arcratio_stdin_RUN_A.json"] == {
        "bind": "/run/arcratio/input.json",
        "mode": "ro",
    }
    # All mounts are read-only.
    assert all(spec["mode"] == "ro" for spec in vols.values())


def test_volumes_are_per_run_isolated():
    """Two different runs must bind two different input sources — no sibling can
    reach another run's payload because only its own file is mounted."""
    a = sd._sandbox_volumes("/host/run/arcratio", ".arcratio_stdin_RUN_A.json")
    b = sd._sandbox_volumes("/host/run/arcratio", ".arcratio_stdin_RUN_B.json")

    a_inputs = [src for src in a if "/inputs/" in src]
    b_inputs = [src for src in b if "/inputs/" in src]
    assert a_inputs != b_inputs
    # Neither run's mount set references the other's payload file.
    assert not any("RUN_B" in src for src in a)
    assert not any("RUN_A" in src for src in b)
