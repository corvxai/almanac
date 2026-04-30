"""Unit tests for sibling-container bind path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import ValidatorConfig
from src.validator import sandbox_docker as sd


def test_decode_proc_mount_token() -> None:
    assert sd._decode_proc_mount_token(r"a\040b") == "a b"
    assert sd._decode_proc_mount_token(r"a\134b") == r"a\b"


def test_mountinfo_host_root_prefers_bind(tmp_path: Path) -> None:
    mp = tmp_path / "sockdir"
    mp.mkdir()
    host_src = "/host/proj/var/run/arcratio"
    mi = (
        f"1 0 252:0 / / rw,relatime - overlay overlay rw\n"
        f"2 1 252:0 {host_src} {mp} rw,nosuid - bind {host_src} rw\n"
    )
    mi_path = tmp_path / "mountinfo"
    mi_path.write_text(mi, encoding="utf-8")

    got = sd._mountinfo_host_root_for_mountpoint(mp, _mountinfo_path=mi_path)
    assert got == host_src


def test_sibling_socket_host_bind_explicit_config(tmp_path: Path) -> None:
    cfg = ValidatorConfig(
        sandbox_socket_dir=tmp_path,
        sandbox_socket_host_bind="/explicit/host/path",
    )
    assert sd._sibling_socket_host_bind(cfg) == "/explicit/host/path"


def test_sibling_socket_host_bind_falls_back_to_socket_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sd, "_mountinfo_host_root_for_mountpoint", lambda _p, **_kw: None)
    cfg = ValidatorConfig(sandbox_socket_dir=Path("/var/run/arcratio"))
    assert sd._sibling_socket_host_bind(cfg) == "/var/run/arcratio"
