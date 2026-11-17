"""Trace validity checks (Training-Grade Traces, Proposal 3).

Computed on the validator from the prompt and response text it holds in
memory for the duration of a run. Only the resulting flags, one count and a
list of domain labels leave this module; no prompt or response text does.

Logging only: the flags are sent to the gateway but do not yet change
``predictionIsInvalid`` or scoring.
"""

from __future__ import annotations

import re
import time
from typing import Any, Iterable

# Bump whenever the same trace could produce different values: tokeniser,
# stopwords, thresholds, market markers, or what counts as prompt/response text.
CHECKS_VERSION = 1

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


def _question_coverage(question: str, provider_calls: Iterable[Any]) -> float:
    """Best share of the question's content words found in the prompt of any
    call that got a non-empty response. 0.0 when there is no such call."""
    question_words = _content_words(question)
    best = 0.0
    for call in provider_calls:
        if not _text(call, "response_text").strip():
            continue
        if not question_words:
            return 1.0
        prompt_words = set(_words(_text(call, "prompt_text")))
        best = max(best, len(question_words & prompt_words) / len(question_words))
    return best


# Agent-controlled text decides how much work the checks do, so they run under
# a wall-clock budget and examine a bounded number of market references.
CHECK_TIME_BUDGET_SECONDS = 5.0
MAX_MARKET_REFERENCES = 200
_REFERENCE_RE = re.compile(r"[^\s\"'<>\\()\[\]{}]+")


class TraceCheckBudgetExceeded(Exception):
    """The checks ran out of time; the caller submits without them."""


def _check_budget(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise TraceCheckBudgetExceeded()


def _market_lookup_domains(
    source_market_id: str | None,
    provider_calls: Iterable[Any],
    supplied_text: str,
    deadline: float,
) -> list[str]:
    """Market-source references the agent itself put in a prompt.

    A reference is one whitespace-delimited token: a URL or a bare domain. It
    is ignored when the same token already appeared in text the agent was
    given: the event's own question and description, or an earlier response
    in the same run (a search result citing the market, passed along). Calls
    are read in the order they were made.
    """
    market_id = (source_market_id or "").strip().lower()
    # The id must sit in a URL as a whole path/query token, so a bare number in
    # prose or a longer id that merely contains it is ignored.
    id_re = re.compile(rf"(?<![a-z0-9]){re.escape(market_id)}(?![a-z0-9])") if market_id else None
    markers: set[str] = set()
    hosts: set[str] = set()
    given: list[str] = [supplied_text.lower()]
    judged: set[str] = set()

    for call in provider_calls:
        _check_budget(deadline)
        prompt = _text(call, "prompt_text").lower()
        for token in set(_REFERENCE_RE.findall(prompt)):
            token = token.rstrip(".,;:!?")
            found_markers = [m for m in _MARKET_SOURCE_MARKERS if m in token]
            is_id_url = (
                id_re is not None
                and token.startswith(("http://", "https://"))
                and id_re.search(token) is not None
            )
            if not (found_markers or is_id_url) or token in judged:
                continue
            if len(judged) >= MAX_MARKET_REFERENCES:
                break
            judged.add(token)
            _check_budget(deadline)
            if any(token in text for text in given):
                continue
            markers.update(found_markers)
            if is_id_url:
                host = _url_host(token)
                if host:
                    hosts.add(host)
        given.append(_text(call, "response_text").lower())
    # Fixed markers first so the cap never drops them.
    return (sorted(markers) + sorted(hosts - markers))[:MAX_MARKET_LOOKUP_DOMAINS]


def compute_trace_checks(
    *,
    question: str,
    source_market_id: str | None,
    provider_calls: Iterable[Any],
    reasoning: str | None,
    supplied_text: str = "",
) -> dict[str, Any]:
    """Return the Proposal 3 flags and their supporting numbers in the submit
    payload's camelCase shape.

    ``supplied_text`` is the event text the validator gave the agent (question
    and description). Raises ``TraceCheckBudgetExceeded`` when the time budget
    runs out.
    """
    calls = list(provider_calls)
    deadline = time.monotonic() + CHECK_TIME_BUDGET_SECONDS

    reasoning_trigrams = _trigrams(reasoning or "")
    # Per call: the reasoning 3-grams that also appear in that call's response.
    shared = []
    for call in calls:
        _check_budget(deadline)
        shared.append(reasoning_trigrams & _trigrams(_text(call, "response_text")))
    overlap = len(set().union(*shared))
    # Same count without the call that shares most with the reasoning, which
    # is the call that wrote it when the reasoning is a model's own output.
    other_overlap = 0
    if len(shared) > 1:
        writer = max(range(len(shared)), key=lambda i: len(shared[i]))
        other_overlap = len(set().union(*(s for i, s in enumerate(shared) if i != writer)))

    coverage = _question_coverage(question, calls)
    _check_budget(deadline)
    domains = _market_lookup_domains(source_market_id, calls, supplied_text, deadline)
    sourced_calls = 0
    for call in calls:
        count = getattr(call, "source_url_count", 0)
        if isinstance(count, int) and count > 0:
            sourced_calls += 1
    return {
        "checksVersion": CHECKS_VERSION,
        "researchCallMissing": coverage < QUESTION_WORD_COVERAGE_MIN,
        "questionCoverage": round(coverage, 4),
        "reasoningUnsupported": overlap < REASONING_MIN_SHARED_TRIGRAMS,
        "reasoningOverlap": overlap,
        "reasoningOverlapOtherCalls": other_overlap,
        "sourcedCallCount": sourced_calls,
        "marketLookup": bool(domains),
        "marketLookupDomains": domains,
    }
