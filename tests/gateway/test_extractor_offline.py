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
    ("grok", "chat_completion", (EvidenceType.PROBABILITY, EvidenceType.FACT)),
    ("perplexity", "chat_completion", (EvidenceType.PROBABILITY, EvidenceType.QUOTE_SUMMARY)),
    ("anthropic", "messages", (EvidenceType.PROBABILITY, EvidenceType.FACT)),
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


@pytest.mark.provider("openrouter")
def test_call_and_extract_stub_matches_pipeline(maybe_pretty_print_raw) -> None:
    """``call_and_extract`` should mirror ``extract_from_raw`` for the same payload."""
    provider_id, call_type = "openrouter", "chat_completion"
    raw = load_raw_fixture(provider_id, call_type)
    stub = StubProvider(provider_id, call_type, raw, ProviderTier.INFERENCE)
    out_raw, evidence = call_and_extract(stub, call_type, {})
    maybe_pretty_print_raw(out_raw, banner="openrouter.chat_completion (stub / golden)")
    assert out_raw == raw
    expected = extract_from_raw(provider_id, call_type, raw)
    assert [e.model_dump() for e in evidence] == [e.model_dump() for e in expected]


@pytest.mark.provider("openrouter")
def test_openrouter_string_output_shape_extracts_without_crashing() -> None:
    raw = {
        "provider": "openrouter",
        "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "output": "PREDICTION: 0.6\nCONVICTION: 0.7\nREASONING: short rationale.",
        "usage": {"promptTokens": 280, "completionTokens": 405, "totalTokens": 685},
    }
    evidence = extract_from_raw("openrouter", "chat_completion", raw)
    assert_evidence_nonempty(evidence)
    assert_evidence_covers_types(evidence, EvidenceType.PROBABILITY, EvidenceType.QUOTE_SUMMARY)


@pytest.mark.provider("openrouter")
def test_openrouter_markdown_output_lines_parse_prediction_and_confidence() -> None:
    raw = {
        "provider": "openrouter",
        "model": "anthropic/claude-4.5-haiku-20251001",
        "output": (
            "**PREDICTION: 0.15**\n\n"
            "**CONVICTION: 0.72**\n\n"
            "**REASONING:** Tail risk remains possible."
        ),
    }
    evidence = extract_from_raw("openrouter", "chat_completion", raw)
    prob_values = [
        e.numeric_value
        for e in evidence
        if e.evidence_type == EvidenceType.PROBABILITY and e.numeric_value is not None
    ]
    assert 0.15 in prob_values
    assert 0.72 in prob_values
