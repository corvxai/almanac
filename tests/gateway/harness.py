"""Helpers for provider calls and evidence extraction (used in offline and live tests)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from src.core.schemas import EvidenceType, ExtractedEvidence, ProviderTier
from src.gateway.extractor import extract_evidence
from src.gateway.providers.base import BaseProvider

FIXTURES_RAW_ROOT = Path(__file__).resolve().parent.parent / "fixtures" / "raw"


def load_raw_fixture(provider_id: str, call_type: str) -> dict[str, Any]:
    """Load a golden raw response from ``tests/fixtures/raw/<provider>/<call_type>.json``."""
    path = FIXTURES_RAW_ROOT / provider_id / f"{call_type}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing fixture: {path}")
    with path.open(encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data


def extract_from_raw(provider_id: str, call_type: str, raw: dict[str, Any]) -> list[ExtractedEvidence]:
    """Run the same extraction pipeline the gateway uses on a raw provider dict."""
    return extract_evidence(provider_id, call_type, raw)


def call_and_extract(
    provider: BaseProvider,
    call_type: str,
    params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[ExtractedEvidence]]:
    """Execute ``provider.call`` then ``extract_evidence`` for that provider."""
    params = params or {}
    raw = provider.call(call_type, params)
    evidence = extract_evidence(provider.provider_id, call_type, raw)
    return raw, evidence


def assert_evidence_nonempty(evidence: list[ExtractedEvidence]) -> None:
    assert evidence, "expected at least one ExtractedEvidence item"


def assert_evidence_covers_types(
    evidence: list[ExtractedEvidence],
    *required: EvidenceType,
) -> None:
    """Assert every listed ``EvidenceType`` appears at least once."""
    found = {e.evidence_type for e in evidence}
    missing = [t for t in required if t not in found]
    assert not missing, f"missing evidence types {missing}; have {found}"


def require_env_vars(*names: str) -> None:
    """Skip the current test unless all named environment variables are set (for live tests)."""
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        pytest.skip(f"missing env for live test: {', '.join(missing)}")


class StubProvider(BaseProvider):
    """Returns a fixed dict from ``call`` — useful for deterministic harness checks."""

    def __init__(self, provider_id: str, call_type: str, raw: dict[str, Any], tier: ProviderTier):
        self._provider_id = provider_id
        self._tier = tier
        self._call_type = call_type
        self._raw = raw

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def provider_tier(self) -> ProviderTier:
        return self._tier

    def call(self, call_type: str, params: dict[str, Any]) -> dict[str, Any]:
        if call_type != self._call_type:
            raise ValueError(f"stub expected call_type {self._call_type!r}, got {call_type!r}")
        return dict(self._raw)
