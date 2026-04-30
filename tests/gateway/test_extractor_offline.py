"""Offline tests: golden raw fixtures → ``extract_evidence`` (no network)."""

from __future__ import annotations

import pytest

from src.core.schemas import EvidenceType, ProviderTier

from tests.gateway.harness import (
    StubProvider,
    assert_evidence_covers_types,
    assert_evidence_nonempty,
    call_and_extract,
    extract_from_raw,
    load_raw_fixture,
)

# (provider_id, call_type, evidence types that must appear at least once)
OFFLINE_FIXTURE_CASES: list[tuple[str, str, tuple[EvidenceType, ...]]] = [
    ("polymarket", "get_market", (EvidenceType.PRICE, EvidenceType.PROBABILITY)),
    ("polymarket", "get_price_history", (EvidenceType.TREND,)),
    ("web_search", "search", (EvidenceType.QUOTE_SUMMARY, EvidenceType.STATISTIC)),
    ("grok", "chat_completion", (EvidenceType.PROBABILITY, EvidenceType.FACT)),
    ("perplexity", "chat_completion", (EvidenceType.PROBABILITY, EvidenceType.QUOTE_SUMMARY)),
    ("claude", "messages", (EvidenceType.PROBABILITY, EvidenceType.FACT)),
    ("openai", "responses", (EvidenceType.PROBABILITY, EvidenceType.FACT)),
    ("gemini", "generate_content", (EvidenceType.PROBABILITY, EvidenceType.FACT)),
    ("openrouter", "chat_completion", (EvidenceType.PROBABILITY, EvidenceType.QUOTE_SUMMARY)),
    ("vendor_x", "snapshot", (EvidenceType.STATISTIC,)),
]

_OFFLINE_PARAMS = [
    pytest.param(p, c, rt, id=f"{p}.{c}", marks=pytest.mark.provider(p))
    for p, c, rt in OFFLINE_FIXTURE_CASES
]


@pytest.mark.parametrize("provider_id,call_type,required_types", _OFFLINE_PARAMS)
def test_extract_evidence_from_golden_fixture(
    provider_id: str,
    call_type: str,
    required_types: tuple[EvidenceType, ...],
    maybe_pretty_print_raw,
) -> None:
    raw = load_raw_fixture(provider_id, call_type)
    maybe_pretty_print_raw(raw, banner=f"{provider_id}.{call_type} (golden fixture)")
    evidence = extract_from_raw(provider_id, call_type, raw)
    assert_evidence_nonempty(evidence)
    assert_evidence_covers_types(evidence, *required_types)


@pytest.mark.provider("polymarket")
def test_call_and_extract_stub_matches_pipeline(maybe_pretty_print_raw) -> None:
    """``call_and_extract`` should mirror ``extract_from_raw`` for the same payload."""
    provider_id, call_type = "polymarket", "get_market"
    raw = load_raw_fixture(provider_id, call_type)
    stub = StubProvider(provider_id, call_type, raw, ProviderTier.FREE_SIGNAL)
    out_raw, evidence = call_and_extract(stub, call_type, {})
    maybe_pretty_print_raw(out_raw, banner="polymarket.get_market (stub / golden)")
    assert out_raw == raw
    expected = extract_from_raw(provider_id, call_type, raw)
    assert [e.model_dump() for e in evidence] == [e.model_dump() for e in expected]
