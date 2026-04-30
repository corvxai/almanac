"""Sanity check for ``@pytest.mark.live`` gating (see tests/conftest.py)."""

from __future__ import annotations

import pytest


@pytest.mark.live
def test_live_marker_only_runs_with_flag() -> None:
    """When this runs, ``pytest`` was invoked with ``--live``."""
    assert True
