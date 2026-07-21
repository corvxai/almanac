"""Tests for docker-CLI-based sibling container wait."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.validator.forecasting import sandbox_docker as sd


def test_wait_cli_uses_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    docker_fake = tmp_path / "docker"
    docker_fake.write_text("#!/bin/sh\necho 42\n", encoding="utf-8")
    docker_fake.chmod(0o755)
    monkeypatch.setattr(sd, "_docker_cli_binary", lambda: str(docker_fake))

    class _C:
        id = "deadbeef"

        def wait(self, **_kw: object) -> dict[str, int]:
            raise AssertionError("SDK wait should not run when docker CLI is resolved")

    got = sd._wait_container_exit_docker_cli_or_sdk(_C(), 30)
    assert got == {"StatusCode": 42}


def test_wait_cli_falls_back_to_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sd, "_docker_cli_binary", lambda: None)

    class _C:
        id = "abc"

        def wait(self, timeout: int | None = None) -> dict[str, int]:
            assert timeout == 5
            return {"StatusCode": 0}

    got = sd._wait_container_exit_docker_cli_or_sdk(_C(), 5)
    assert got == {"StatusCode": 0}
