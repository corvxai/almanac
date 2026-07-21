"""Sanity test for the vendored sportstensor/sn41 quarantine.

This is a cheap regression net against bad relative-import edits during
re-vendoring. It only asserts that the modules import and the public symbols
exist; it does *not* exercise any logic.
"""

from __future__ import annotations

import pytest


def test_constants_module_loads_known_values() -> None:
    from src.validator.market import constants

    assert constants.ROLLING_HISTORY_IN_DAYS == 30
    assert constants.VOLUME_FEE == 0.01
    assert constants.TOTAL_MINER_ALPHA_PER_DAY == 2952
    assert constants.BURN_UID == 210
    assert isinstance(constants.POLY_BUILDER_CODE, str)


def test_scoring_module_exposes_public_entries() -> None:
    pytest.importorskip("cvxpy")
    pytest.importorskip("scipy")
    pytest.importorskip("tabulate")
    pytest.importorskip("bittensor")

    from src.validator.market import scoring

    for name in ("score_miners", "calculate_weights", "print_pool_stats"):
        assert callable(getattr(scoring, name)), f"missing {name}"


def test_metadata_manager_imports() -> None:
    pytest.importorskip("bittensor")

    from src.validator.market.metadata_manager import MetadataManager

    assert MetadataManager.__init__.__qualname__.startswith("MetadataManager")


def test_postgres_storage_imports() -> None:
    pytest.importorskip("bittensor")

    from src.validator.market.storage import postgres_validator_storage  # noqa: F401
