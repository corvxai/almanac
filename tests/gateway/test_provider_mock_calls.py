"""In-process mock providers: ``call`` + ``extract_evidence`` for every catalogued operation."""

from __future__ import annotations

import pytest

from tests.gateway.harness import assert_evidence_covers_types, assert_evidence_nonempty, call_and_extract
from tests.gateway.provider_catalog import MOCK_PROVIDER_CALL_EXAMPLES, ProviderCallExample, build_mock_provider

_MOCK_CALL_PARAMS = [
    pytest.param(
        ex,
        id=f"{ex.provider_id}.{ex.call_type}",
        marks=pytest.mark.provider(ex.provider_id),
    )
    for ex in MOCK_PROVIDER_CALL_EXAMPLES
]


@pytest.mark.parametrize("example", _MOCK_CALL_PARAMS)
def test_mock_provider_call_and_extract(
    example: ProviderCallExample,
    maybe_pretty_print_raw,
) -> None:
    provider = build_mock_provider(example.provider_id)
    assert provider.provider_id == example.provider_id
    raw, evidence = call_and_extract(provider, example.call_type, example.params)
    maybe_pretty_print_raw(
        raw,
        banner=f"{example.provider_id}.{example.call_type} (mock adapter)",
    )
    assert_evidence_nonempty(evidence)
    assert_evidence_covers_types(evidence, *example.require_evidence_types)
