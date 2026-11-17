from __future__ import annotations

from types import SimpleNamespace

from src.core.schemas import ProviderTier
from src.gateway.gateway import build_provider_call_record
from src.validator.forecasting.trace_validity import (
    REASONING_MIN_SHARED_TRIGRAMS,
    compute_trace_checks,
)

QUESTION = "Will the Federal Reserve cut interest rates in December 2026?"
RESPONSE = (
    "Fed officials signalled at the November meeting that inflation has cooled faster "
    "than projected and that a quarter point reduction is on the table for December."
)
REASONING = (
    "Officials signalled at the November meeting that inflation has cooled faster than "
    "projected, so a quarter point reduction is on the table and I put this at 0.62."
)


def _call(prompt: str, response: str = RESPONSE) -> SimpleNamespace:
    return SimpleNamespace(prompt_text=prompt, response_text=response)


def _checks(calls, reasoning: str = REASONING, market_id: str | None = "2411919") -> dict:
    return compute_trace_checks(
        question=QUESTION,
        source_market_id=market_id,
        provider_calls=calls,
        reasoning=reasoning,
    )


def test_honest_trace_passes_all_three() -> None:
    checks = _checks([_call(f"Research this question: {QUESTION}")])
    assert checks["researchCallMissing"] is False
    assert checks["reasoningUnsupported"] is False
    assert checks["reasoningOverlap"] >= REASONING_MIN_SHARED_TRIGRAMS
    assert checks["marketLookup"] is False
    assert checks["marketLookupDomains"] == []


def test_research_call_missing_when_no_prompt_is_about_the_question() -> None:
    checks = _checks([_call("What is the weather in Paris tomorrow?")])
    assert checks["researchCallMissing"] is True


def test_research_call_missing_when_on_topic_call_has_empty_response() -> None:
    checks = _checks([_call(QUESTION, response="  ")])
    assert checks["researchCallMissing"] is True


def test_research_call_coverage_threshold_is_sixty_percent() -> None:
    # Content words: federal reserve cut interest rates december 2026 (7).
    four_of_seven = _call("federal reserve cut interest")
    five_of_seven = _call("federal reserve cut interest rates")
    assert _checks([four_of_seven])["researchCallMissing"] is True
    assert _checks([five_of_seven])["researchCallMissing"] is False


def test_research_call_missing_with_no_calls() -> None:
    checks = _checks([])
    assert checks["researchCallMissing"] is True
    assert checks["reasoningUnsupported"] is True
    assert checks["reasoningOverlap"] == 0


def test_reasoning_unsupported_when_it_ignores_the_responses() -> None:
    checks = _checks(
        [_call(QUESTION)],
        reasoning="Base rates for central bank decisions of this kind suggest roughly even odds here.",
    )
    assert checks["reasoningUnsupported"] is True
    assert checks["reasoningOverlap"] < REASONING_MIN_SHARED_TRIGRAMS


def test_reasoning_overlap_counts_distinct_trigrams_across_all_responses() -> None:
    half = len(RESPONSE) // 2
    split = RESPONSE.rfind(" ", 0, half)
    one = _checks([_call(QUESTION)])
    # A repeated response adds nothing; a second on-topic call still counts.
    repeated = _checks([_call(QUESTION), _call(QUESTION)])
    assert repeated["reasoningOverlap"] == one["reasoningOverlap"]
    partial = _checks([_call(QUESTION, response=RESPONSE[:split])])
    assert partial["reasoningOverlap"] < one["reasoningOverlap"]


def test_market_lookup_flags_source_domains_without_invalidating() -> None:
    checks = _checks(
        [
            _call(QUESTION),
            _call("Open https://Gamma-API.polymarket.com/markets/2411919 and return the YES price"),
        ]
    )
    assert checks["marketLookup"] is True
    assert checks["marketLookupDomains"] == ["gamma-api", "polymarket.com", "gamma-api.polymarket.com"]
    assert checks["researchCallMissing"] is False
    assert checks["reasoningUnsupported"] is False


def test_market_lookup_flags_market_id_only_inside_a_url() -> None:
    in_url = _checks([_call("fetch https://example.org/api/markets/2411919?fmt=json")])
    assert in_url["marketLookup"] is True
    assert in_url["marketLookupDomains"] == ["example.org"]

    in_prose = _checks([_call(f"{QUESTION} (market 2411919)")])
    assert in_prose["marketLookup"] is False

    longer_id = _checks([_call("fetch https://example.org/item/924119190")])
    assert longer_id["marketLookup"] is False


def test_market_lookup_domains_always_fit_the_gateway_contract() -> None:
    import re

    urls = " ".join(f"https://h{i}.example/m/2411919" for i in range(40))
    odd = "https://user:pw@Ex_ample.ORG:8443/m/2411919 https://[::1]/2411919"
    checks = _checks([_call(f"{urls} {odd} https://polymarket.com/event/x")])
    domains = checks["marketLookupDomains"]
    assert checks["marketLookup"] is True
    assert 0 < len(domains) <= 16
    assert all(re.fullmatch(r"[a-z0-9./-]{1,253}", d) for d in domains)
    assert domains[0] == "polymarket.com"

    only_odd = _checks([_call(odd)])["marketLookupDomains"]
    assert "example.org" in only_odd


def test_market_lookup_kalshi_needs_the_markets_path() -> None:
    assert _checks([_call("see https://kalshi.com/markets/kxfed")])["marketLookupDomains"] == [
        "kalshi.com/markets"
    ]
    assert _checks([_call("see https://kalshi.com/blog/fed")])["marketLookup"] is False


def test_calls_without_captured_text_are_handled() -> None:
    checks = _checks([SimpleNamespace(provider_id="anthropic")])
    assert checks["researchCallMissing"] is True
    assert checks["marketLookup"] is False


def _record(params: dict, raw_response: dict):
    return build_provider_call_record(
        call_index=0,
        provider_id="openrouter",
        provider_tier=ProviderTier.INFERENCE,
        call_type="chat",
        params=params,
        query_params_summary="chat(model=x)",
        raw_response=raw_response,
        latency_ms=5,
    )


def test_provider_call_record_keeps_text_in_memory_but_never_dumps_it() -> None:
    record = _record(
        {"model": "x", "messages": [{"role": "user", "content": [{"type": "text", "text": QUESTION}]}]},
        {"choices": [{"message": {"content": RESPONSE}}]},
    )
    assert QUESTION in record.prompt_text
    assert record.response_text == RESPONSE

    dumped = record.model_dump(mode="json")
    assert "prompt_text" not in dumped and "response_text" not in dumped
    assert "response_text" not in record.model_dump_json()
    # The question reaches the record only through prompt_text here.
    assert QUESTION not in record.model_dump_json()
    assert QUESTION not in repr(record)

    checks = _checks([record])
    assert checks["researchCallMissing"] is False
    assert checks["reasoningUnsupported"] is False


def test_provider_call_record_empty_llm_completion_is_an_empty_response() -> None:
    record = _record(
        {"messages": [{"role": "user", "content": QUESTION}]},
        {"id": "gen-1", "model": "x", "choices": [{"message": {"content": ""}}]},
    )
    assert record.response_text == ""
    assert _checks([record])["researchCallMissing"] is True


def test_provider_call_record_uses_payload_strings_for_non_llm_responses() -> None:
    record = _record({"query": QUESTION}, {"results": [{"title": "Fed", "snippet": RESPONSE}]})
    assert RESPONSE in record.response_text
    assert _checks([record])["reasoningUnsupported"] is False


def _supported(raw_response: dict) -> dict:
    return _checks([_record({"messages": [{"role": "user", "content": QUESTION}]}, raw_response)])


def test_response_text_covers_every_llm_response_shape() -> None:
    shapes = {
        "claude web search": {
            "content": [
                {"type": "text", "text": "I'll search for the latest news."},
                {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "fed"}},
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_1",
                    "content": [{"type": "web_search_result", "title": "Fed", "encrypted_content": "QUJD"}],
                },
                {"type": "text", "text": RESPONSE},
            ]
        },
        "gemini grounded": {"candidates": [{"content": {"parts": [{"text": "Searching."}, {"text": RESPONSE}]}}]},
        "openai responses": {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "Summary:"},
                        {"type": "output_text", "text": RESPONSE},
                    ],
                }
            ]
        },
        "chat content parts": {"choices": [{"message": {"content": [{"type": "text", "text": RESPONSE}]}}]},
        "grok reasoning_content": {"choices": [{"message": {"content": "", "reasoning_content": RESPONSE}}]},
    }
    for name, raw in shapes.items():
        checks = _supported(raw)
        assert checks["researchCallMissing"] is False, name
        assert checks["reasoningUnsupported"] is False, name


def test_response_with_only_metadata_is_empty() -> None:
    record = _record(
        {"messages": [{"role": "user", "content": QUESTION}]},
        {
            "id": "gen-1",
            "model": "x",
            "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": None}}],
        },
    )
    assert record.response_text == ""


def test_deeply_nested_params_do_not_break_the_call_record() -> None:
    nested: dict = {"q": QUESTION}
    for _ in range(5000):
        nested = {"n": nested}
    record = _record({"messages": [{"role": "user", "content": QUESTION}], "extra": nested}, {"choices": []})
    assert QUESTION in record.prompt_text


def test_retained_text_is_capped_per_call() -> None:
    record = _record({"prompt": "a " * 500_000}, {"results": [{"snippet": "b " * 500_000}]})
    assert len(record.prompt_text) == 200_000
    assert len(record.response_text) == 200_000


def test_trace_check_failure_never_blocks_the_submit(monkeypatch) -> None:
    from src.validator.forecasting import assignment_pipeline
    from tests.validator.test_validator_assignment_submit_payload import _assignment, _Digest

    def boom(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(assignment_pipeline, "compute_trace_checks", boom)
    payload = assignment_pipeline.build_prediction_submit_payload(
        _assignment(outcomes=[{"outcomeId": "oid_yes", "name": "Yes"}, {"outcomeId": "oid_no", "name": "No"}]),
        _Digest(0.7, confidence=0.8),
        {},
    )
    validation = payload["prediction"]["executionMetadata"]["predictionValidation"]
    assert "traceChecks" not in validation
    assert validation["isValid"] is True


def test_checks_version_is_reported() -> None:
    assert _checks([_call(QUESTION)])["checksVersion"] == 1


def test_question_coverage_is_the_best_call_and_drives_the_flag() -> None:
    checks = _checks([_call("federal reserve"), _call("federal reserve cut interest"), _call(QUESTION, response="")])
    # 4 of 7 content words; the full-question call has no response so it does not count.
    assert checks["questionCoverage"] == round(4 / 7, 4)
    assert checks["researchCallMissing"] is True
    assert _checks([_call(QUESTION)])["questionCoverage"] == 1.0
    assert _checks([])["questionCoverage"] == 0.0


def test_overlap_other_calls_excludes_the_call_that_wrote_the_reasoning() -> None:
    # One call: the reasoning overlaps only with its own source.
    single = _checks([_call(QUESTION)])
    assert single["reasoningOverlap"] >= REASONING_MIN_SHARED_TRIGRAMS
    assert single["reasoningOverlapOtherCalls"] == 0

    # A synthesis call that wrote the reasoning, plus an unrelated call: still 0.
    writer = _call(QUESTION, response=REASONING)
    unrelated = _call(QUESTION, response="The weather in Paris will be mild and dry all week long.")
    assert _checks([unrelated, writer])["reasoningOverlapOtherCalls"] == 0

    # A research call the reasoning really draws on is still counted.
    drawn = _checks([_call(QUESTION), writer])
    assert drawn["reasoningOverlap"] > drawn["reasoningOverlapOtherCalls"] >= REASONING_MIN_SHARED_TRIGRAMS


def test_sourced_call_count_uses_structured_source_links_only() -> None:
    prompt = {"messages": [{"role": "user", "content": QUESTION}]}
    prose_only = _record(prompt, {"choices": [{"message": {"content": "See https://example.org/a for details."}}]})
    annotated = _record(
        prompt,
        {
            "choices": [
                {
                    "message": {
                        "content": RESPONSE,
                        "annotations": [
                            {"type": "url_citation", "url_citation": {"url": "https://example.org/a", "title": "A"}}
                        ],
                    }
                }
            ]
        },
    )
    cited = _record(prompt, {"choices": [{"message": {"content": RESPONSE}}], "citations": ["https://example.org/b"]})
    grounded = _record(
        prompt,
        {"candidates": [{"content": {"parts": [{"text": RESPONSE}]}, "groundingMetadata": {"groundingChunks": [{"web": {"uri": "https://example.org/c"}}]}}]},
    )
    assert prose_only.source_url_count == 0
    assert (annotated.source_url_count, cited.source_url_count, grounded.source_url_count) == (1, 1, 1)
    assert "source_url_count" not in annotated.model_dump(mode="json")
    assert _checks([prose_only, annotated, cited, grounded])["sourcedCallCount"] == 3
    assert _checks([_call(QUESTION)])["sourcedCallCount"] == 0


def _gateway_completion(output: str) -> dict:
    # Shape the Sub41 gateway returns to the validator (GatewayCompletionResponse).
    return {
        "provider": "openrouter",
        "model": "openai/gpt-4o-mini:online",
        "output": output,
        "usage": {"promptTokens": 10, "completionTokens": 0, "totalTokens": 10},
        "costMicro": "1234",
        "providerCostMicro": "1028",
        "markupBps": 2000,
        "balanceAfterMicro": "998766",
        "finishReason": "length",
        "upstreamId": "gen-1763398552-abcdef",
    }


def test_gateway_completion_with_empty_output_is_an_empty_response() -> None:
    prompt = {"model": "openai/gpt-4o-mini:online", "messages": [{"role": "user", "content": QUESTION}]}
    empty = _record(prompt, _gateway_completion(""))
    assert empty.response_text == ""
    assert _checks([empty])["researchCallMissing"] is True

    filled = _record(prompt, _gateway_completion(RESPONSE))
    assert filled.response_text == RESPONSE
    checks = _checks([filled])
    assert checks["researchCallMissing"] is False
    assert checks["reasoningUnsupported"] is False


def test_sourced_call_count_reads_the_gateway_citation_count() -> None:
    prompt = {"model": "perplexity/sonar", "messages": [{"role": "user", "content": QUESTION}]}
    cited = _record(prompt, {**_gateway_completion(RESPONSE), "citationCount": 28})
    uncited = _record(prompt, {**_gateway_completion(RESPONSE), "citationCount": 0})
    malformed = _record(prompt, {**_gateway_completion(RESPONSE), "citationCount": "many"})
    assert (cited.source_url_count, uncited.source_url_count, malformed.source_url_count) == (28, 0, 0)
    assert _checks([cited, uncited, malformed])["sourcedCallCount"] == 1
    # The count is a number, not text: it must not leak into the response text.
    assert cited.response_text == RESPONSE
