from __future__ import annotations

import json
from datetime import timezone

import httpx
import pytest

from src.gateway.portal_client import PortalGateway, fetch_portal_event, portal_event_to_event


def _portal_event() -> dict:
    return {
        "id": "portal-event-1",
        "source": "polymarket",
        "sourceMarketId": "market-42",
        "question": "Will the test pass?",
        "description": "Resolves yes if the test passes.",
        "endDate": "2026-08-01T00:00:00.000Z",
        "outcomes": [
            {"name": "Yes", "outcomeId": "yes-id"},
            {"name": "No", "outcomeId": "no-id"},
        ],
        "currentOutcomePrices": {"yes-id": "0.61", "no-id": 0.39},
        "tagsAtAdmission": ["testing"],
    }


def test_portal_gateway_uses_bearer_auth_and_normalizes_output() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/gateway/providers":
            return httpx.Response(
                200,
                json={"providers": [{"id": "openrouter", "allowsAnyModel": True}]},
            )
        assert request.url.path == "/v1/gateway/completions"
        payload = json.loads(request.content)
        assert payload["maxTokens"] == 12
        return httpx.Response(
            200,
            json={
                "provider": "openrouter",
                "model": "openai/gpt-4o-mini",
                "output": "hello",
                "usage": {"totalTokens": 3},
                "costMicro": "4",
                "providerCostMicro": "4",
                "markupBps": 0,
                "balanceAfterMicro": "99",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gateway = PortalGateway(base_url="https://api.example", api_key="secret", client=client)
    result = gateway.call_provider(
        "openrouter",
        "chat_completion",
        {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 12,
        },
    )

    assert all(request.headers["authorization"] == "Bearer secret" for request in requests)
    assert result["output"] == "hello"
    assert result["choices"][0]["message"]["content"] == "hello"
    assert result["_almanac"]["costMicro"] == "4"
    assert gateway.call_log[0].balance_after_micro == "99"


def test_portal_gateway_rejects_provider_outside_catalog() -> None:
    post_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal post_count
        if request.method == "POST":
            post_count += 1
        return httpx.Response(200, json={"providers": [{"id": "openrouter"}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gateway = PortalGateway(base_url="https://api.example", api_key="secret", client=client)

    with pytest.raises(RuntimeError, match="available providers: openrouter"):
        gateway.call_provider("web_search", "search", {"query": "news"})
    assert post_count == 0


@pytest.mark.parametrize(
    ("status", "body", "message"),
    [
        (401, {"message": "missing api key"}, "GATEWAY_API_KEY"),
        (402, {"message": "insufficient credits"}, "insufficient portal credits"),
    ],
)
def test_portal_gateway_reports_auth_and_credit_errors(
    status: int,
    body: dict,
    message: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/providers"):
            return httpx.Response(200, json={"providers": ["openrouter"]})
        return httpx.Response(status, json=body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gateway = PortalGateway(base_url="https://api.example", api_key="secret", client=client)
    with pytest.raises(RuntimeError, match=message):
        gateway.call_provider("openrouter", "chat_completion", {})


def test_portal_event_mapping_matches_internal_event_shape() -> None:
    event = portal_event_to_event(_portal_event())

    assert event.title == "Will the test pass?"
    assert event.source_id == "market-42"
    assert event.resolution_deadline.tzinfo == timezone.utc
    assert event.current_outcome_prices == {"yes-id": 0.61, "no-id": 0.39}
    assert event.outcomes[0]["outcomeId"] == "yes-id"
    assert event.tags == ["testing"]


def test_fetch_portal_event_uses_specific_event_endpoint() -> None:
    seen_path = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_path
        seen_path = request.url.path
        return httpx.Response(200, json=_portal_event())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    event = fetch_portal_event(
        base_url="https://api.example",
        event_id="portal-event-1",
        client=client,
    )

    assert seen_path == "/v1/events/portal-event-1"
    assert event.source_id == "market-42"
