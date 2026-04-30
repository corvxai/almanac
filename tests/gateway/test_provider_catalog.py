"""Catalog invariants — mock factory and call examples stay aligned."""

from __future__ import annotations

from tests.gateway.provider_catalog import (
    MOCK_PROVIDER_CALL_EXAMPLES,
    MOCK_PROVIDER_IDS,
    build_mock_provider,
)


def test_mock_catalog_covers_every_registered_mock_provider() -> None:
    covered = {e.provider_id for e in MOCK_PROVIDER_CALL_EXAMPLES}
    assert covered == set(MOCK_PROVIDER_IDS)


def test_build_mock_provider_for_each_id() -> None:
    for pid in MOCK_PROVIDER_IDS:
        p = build_mock_provider(pid)
        assert p.provider_id == pid
