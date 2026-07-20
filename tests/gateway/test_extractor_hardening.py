"""Hardening tests for the OpenRouter citation extractor.

F2: a 2xx body with unexpected shapes must never raise out of the extractor.
F5/F14: web citations are UNSCORED in MVP — the per-evidence grounding flag must be
False and consistent with the Source default returned by ``extract_sources``.
"""

from __future__ import annotations

import pytest

from src.core.schemas import EvidenceType, ProviderTier, Source
from src.gateway.extractor import (
    _extract_openrouter,
    _openrouter_url_citations,
    extract_evidence,
    extract_sources,
)
from src.gateway.gateway import build_provider_call_record


# --- F2: malformed-but-2xx bodies never raise -------------------------------

_MALFORMED_BODIES = [
    pytest.param({"choices": [{"message": None}]}, id="message-null"),
    pytest.param({"choices": ["str"]}, id="choices-list-of-str"),
    pytest.param({"choices": [{"message": {"annotations": ["str"]}}]}, id="annotations-list-of-str"),
    pytest.param(
        {"choices": [{"message": {"annotations": [{"type": "url_citation", "url_citation": "str"}]}}]},
        id="url_citation-str",
    ),
    pytest.param({"search_results": ["str"]}, id="search_results-list-of-str"),
    pytest.param({"citations": [123]}, id="citations-list-of-int"),
    pytest.param({"choices": "not-a-list"}, id="choices-str"),
    pytest.param({"annotations": "top-level-ignored"}, id="no-choices"),
]


@pytest.mark.parametrize("raw", _MALFORMED_BODIES)
def test_openrouter_url_citations_never_raises_on_2xx_body(raw: dict) -> None:
    # Arrange/Act
    out = _openrouter_url_citations(raw)
    # Assert: no URL survives any of these shapes, and nothing raised.
    assert out == []


@pytest.mark.parametrize("raw", _MALFORMED_BODIES)
def test_extract_openrouter_does_not_raise_on_malformed_body(raw: dict) -> None:
    # A malformed body may still carry no completion text; the extractor must
    # simply return (possibly empty) evidence without raising.
    items = _extract_openrouter("chat_completion", raw)
    assert isinstance(items, list)


def test_build_provider_call_record_survives_malformed_2xx_body() -> None:
    # E2e: the gateway record builder appends a ProviderCall with no call_index gap
    # even when annotations are malformed on an otherwise-good completion.
    raw = {
        "choices": [
            {
                "message": {
                    "content": "PREDICTION: 0.6\nCONVICTION: 0.7\nREASONING: ok.",
                    "annotations": ["garbage", {"type": "url_citation", "url_citation": "str"}],
                }
            }
        ],
        "citations": [123, "https://example.com/valid"],
    }
    record = build_provider_call_record(
        call_index=3,
        provider_id="openrouter",
        provider_tier=ProviderTier.SEARCH,
        call_type="chat_completion",
        params={"model": "openai/gpt-4o-mini:online"},
        query_params_summary="q",
        raw_response=raw,
        latency_ms=12,
    )
    assert record.call_index == 3
    # the one valid string citation is captured; the int and malformed annotation are skipped
    urls = [ev.source_url for ev in record.extracted_evidence if ev.source_url]
    assert "https://example.com/valid" in urls


# --- F5 / F14: grounding flag is False and consistent -----------------------

def _online_body_with_citation() -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": "PREDICTION: 0.6\nCONVICTION: 0.7\nREASONING: grounded.",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "url_citation": {
                                "url": "https://example.com/a",
                                "title": "A",
                                "content": "excerpt text",
                            },
                        }
                    ],
                }
            }
        ]
    }


def test_openrouter_citation_evidence_is_not_grounding() -> None:
    # F5: citation evidence carries the audit trail but is NOT admissible grounding.
    items = _extract_openrouter("chat_completion", _online_body_with_citation())
    fact_items = [e for e in items if e.evidence_type == EvidenceType.FACT and e.source_url]
    assert fact_items, "expected a cited FACT evidence item"
    for e in fact_items:
        assert e.counts_toward_grounding in (False, None)
        # audit trail preserved
        assert e.source_url == "https://example.com/a"
        assert e.excerpt == "excerpt text"


def test_evidence_flag_matches_source_flag_for_same_citation() -> None:
    # F14: the per-evidence grounding flag equals the Source default from
    # extract_sources for the same citation (both False / not admissible in MVP).
    raw = _online_body_with_citation()
    evidence = extract_evidence("openrouter", "chat_completion", raw)
    sources = extract_sources("openrouter", raw)
    assert sources, "expected extract_sources to return the citation"

    ev_flag = next(
        e.counts_toward_grounding
        for e in evidence
        if e.source_url == "https://example.com/a"
    )
    source_flag = Source(**{k: v for k, v in sources[0].items() if k in {"url", "title", "excerpt"}}).counts_toward_grounding
    # normalize None -> False for the comparison (both mean "not admissible")
    assert bool(ev_flag) == source_flag == False  # noqa: E712
