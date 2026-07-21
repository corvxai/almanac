"""Unit tests for the streaming stdout cap.

A hostile agent can flood stdout to OOM the validator. `_cap_log_stream` must
bound the buffered bytes and abort *early* rather than draining the whole
(possibly unbounded) stream.
"""

from __future__ import annotations

import pytest

from src.validator.forecasting import sandbox_docker as sd


def test_under_cap_returns_joined_bytes() -> None:
    chunks = [b"hello ", b"world"]
    assert sd._cap_log_stream(iter(chunks), max_bytes=1024) == b"hello world"


def test_skips_empty_chunks() -> None:
    chunks = [b"", b"a", b"", b"b"]
    assert sd._cap_log_stream(iter(chunks), max_bytes=1024) == b"ab"


def test_raises_when_total_exceeds_cap() -> None:
    chunks = [b"x" * 10, b"y" * 10]
    with pytest.raises(sd._SandboxStdoutTooLarge):
        sd._cap_log_stream(iter(chunks), max_bytes=15)


def test_aborts_early_without_draining_stream() -> None:
    """The flood must stop mid-stream, not after consuming everything."""
    consumed = {"chunks": 0}

    def flood():
        # An effectively unbounded generator; if the cap drained it, this hangs
        # the test rather than raising promptly.
        while True:
            consumed["chunks"] += 1
            yield b"x" * 1024

    with pytest.raises(sd._SandboxStdoutTooLarge):
        sd._cap_log_stream(flood(), max_bytes=4096)

    # 4096/1024 = 4 chunks fit; the 5th trips the cap. Must not run away.
    assert consumed["chunks"] <= 6
