"""Trace validity checks (Training-Grade Traces, Proposal 3).

Computed on the validator from the prompt and response text it holds in
memory for the duration of a run. Only the resulting flags, one count and a
list of domain labels leave this module; no prompt or response text does.

Logging only: the flags are sent to the gateway but do not yet change
``predictionIsInvalid`` or scoring.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

# A research call is "about the question" when its prompt contains at least
# this share of the question's distinct content words.
QUESTION_WORD_COVERAGE_MIN = 0.6
# Reasoning is supported when it shares at least this many distinct word
# 3-grams with the union of provider responses. Start value; tune on honest
# traces while the flags are logging only.
REASONING_MIN_SHARED_TRIGRAMS = 8

_WORD_RE = re.compile(r"[a-z0-9]+")
_URL_RE = re.compile(r"https?://[^\s\"'<>\\)\]]+", re.IGNORECASE)

# Function words that carry nothing about which event a question is about.
_STOPWORDS = frozenset(
    """
    a an and any are as at be been before by can could did do does for from
    had has have how if in into is it its may might more most of on or over
    shall should than that the their then there these this those to under
    was were what when where which who whom whose why will with would
    """.split()
)

# Substrings that mark a fetch of the source market's own page or API.
_MARKET_SOURCE_MARKERS = (
    "polymarket.com",
    "gamma-api",
    "clob.polymarket",
    "kalshi.com/markets",
)


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _content_words(text: str) -> set[str]:
    return {w for w in _words(text) if w not in _STOPWORDS}


def _trigrams(text: str) -> set[tuple[str, str, str]]:
    words = _words(text)
    return set(zip(words, words[1:], words[2:]))


def _text(call: Any, name: str) -> str:
    value = getattr(call, name, "")
    return value if isinstance(value, str) else ""


# The gateway accepts at most this many labels, each matching [a-z0-9./-]{1,253}.
# A label outside that shape would fail the whole submit, so both are enforced here.
MAX_MARKET_LOOKUP_DOMAINS = 16


def _url_host(url: str) -> str:
    authority = re.sub(r"^https?://", "", url, flags=re.IGNORECASE).split("/")[0]
    host = authority.rsplit("@", 1)[-1].split(":")[0].lower()
    return re.sub(r"[^a-z0-9.-]", "", host)[:253]


def _has_research_call(question: str, provider_calls: Iterable[Any]) -> bool:
    question_words = _content_words(question)
    for call in provider_calls:
        if not _text(call, "response_text").strip():
            continue
        if not question_words:
            return True
        prompt_words = set(_words(_text(call, "prompt_text")))
        covered = len(question_words & prompt_words) / len(question_words)
        if covered >= QUESTION_WORD_COVERAGE_MIN:
            return True
    return False


def _market_lookup_domains(source_market_id: str | None, provider_calls: Iterable[Any]) -> list[str]:
    market_id = (source_market_id or "").strip().lower()
    markers: set[str] = set()
    hosts: set[str] = set()
    for call in provider_calls:
        prompt = _text(call, "prompt_text").lower()
        for marker in _MARKET_SOURCE_MARKERS:
            if marker in prompt:
                markers.add(marker)
        if market_id:
            # The id must sit in a URL as a whole path/query token, so a bare
            # number in prose or a longer id that merely contains it is ignored.
            id_re = re.compile(rf"(?<![a-z0-9]){re.escape(market_id)}(?![a-z0-9])")
            for url in _URL_RE.findall(prompt):
                if id_re.search(url):
                    host = _url_host(url)
                    if host:
                        hosts.add(host)
    # Fixed markers first so the cap never drops them.
    return (sorted(markers) + sorted(hosts - markers))[:MAX_MARKET_LOOKUP_DOMAINS]


def compute_trace_checks(
    *,
    question: str,
    source_market_id: str | None,
    provider_calls: Iterable[Any],
    reasoning: str | None,
) -> dict[str, Any]:
    """Return the three Proposal 3 flags in the submit payload's camelCase shape."""
    calls = list(provider_calls)

    response_trigrams: set[tuple[str, str, str]] = set()
    for call in calls:
        response_trigrams |= _trigrams(_text(call, "response_text"))
    overlap = len(_trigrams(reasoning or "") & response_trigrams)

    domains = _market_lookup_domains(source_market_id, calls)
    return {
        "researchCallMissing": not _has_research_call(question, calls),
        "reasoningUnsupported": overlap < REASONING_MIN_SHARED_TRIGRAMS,
        "reasoningOverlap": overlap,
        "marketLookup": bool(domains),
        "marketLookupDomains": domains,
    }
