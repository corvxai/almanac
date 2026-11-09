from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import httpx
import pytest

from src.core.events import Event
from src.core.schemas import EventCategory
from src.validator.forecasting.polymarket_baseline import fetch_polymarket_baseline


def _event(**overrides: Any) -> Event:
    base = dict(
        event_id=uuid4(),
        title="Will X happen?",
        description="Binary event.",
        category=EventCategory.OTHER,
        resolution_criteria="YES iff X occurs.",
        resolution_deadline=datetime(2027, 1, 1, tzinfo=timezone.utc),
        created_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
        source="polymarket",
        source_id="485257",
        source_url="https://polymarket.com/event/test-market",
    )
    base.update(overrides)
    return Event(**base)


def test_fetch_baseline_returns_none_for_non_polymarket() -> None:
    event = _event(source="manual")
    assert fetch_polymarket_baseline(event) is None


def test_fetch_baseline_uses_source_id_lookup_first() -> None:
    calls: list[dict[str, str]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        calls.append(dict(request.url.params))
        assert request.url.path == "/markets"
        return httpx.Response(
            status_code=200,
            json=[{
                "slug": "test-market",
                "outcomes": ["Yes", "No"],
                "outcomePrices": [0.44, 0.56],
                "updatedAt": "2026-01-02T00:00:00+00:00",
            }],
        )

    event = _event()
    transport = httpx.MockTransport(_handler)
    with httpx.Client(base_url="https://gamma-api.polymarket.com", transport=transport, timeout=5.0) as client:
        baseline = fetch_polymarket_baseline(event, http_client=client)

    assert baseline is not None
    assert baseline["source_id"] == "485257"
    assert baseline["yes_price"] == 0.44
    assert baseline["no_price"] == 0.56
    assert baseline["yes_outcome_price"] == 0.44
    assert baseline["no_outcome_price"] == 0.56
    assert baseline["history_points"] == 1
    assert calls[0].get("id") == "485257"


def test_fetch_baseline_falls_back_to_slug_lookup() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("id") == "485257":
            return httpx.Response(status_code=200, json=[])
        assert request.url.params.get("slug") == "test-market"
        return httpx.Response(
            status_code=200,
            json=[{
                "slug": "test-market",
                "outcomes": '["Yes","No"]',
                "outcomePrices": "[0.31,0.62]",
            }],
        )

    event = _event()
    transport = httpx.MockTransport(_handler)
    with httpx.Client(base_url="https://gamma-api.polymarket.com", transport=transport, timeout=5.0) as client:
        baseline = fetch_polymarket_baseline(event, http_client=client)

    assert baseline is not None
    assert baseline["yes_price"] == 0.31
    assert baseline["no_price"] == 0.69
    assert baseline["yes_outcome_price"] == 0.31
    assert baseline["no_outcome_price"] == 0.62


def test_fetch_baseline_carries_named_gamma_prices_not_the_complement() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/markets"
        return httpx.Response(
            status_code=200,
            json=[{
                "slug": "test-market",
                "outcomes": ["No", "Yes"],
                "outcomePrices": [0.5, 0.25],
            }],
        )

    event = _event()
    transport = httpx.MockTransport(_handler)
    with httpx.Client(
        base_url="https://gamma-api.polymarket.com",
        transport=transport,
        timeout=5.0,
    ) as client:
        baseline = fetch_polymarket_baseline(event, http_client=client)

    assert baseline is not None
    assert baseline["yes_price"] == 0.25
    assert baseline["no_price"] == 0.75
    assert baseline["yes_outcome_price"] == 0.25
    assert baseline["no_outcome_price"] == 0.5


@pytest.mark.parametrize(
    "market",
    [
        pytest.param(
            {"outcomes": ["Yes", "No"], "outcomePrices": [0.4, 1.2]},
            id="out_of_range",
        ),
        pytest.param(
            {"outcomes": ["Yes", "No"], "outcomePrices": [0.4, float("nan")]},
            id="non_finite",
        ),
        pytest.param(
            {"outcomes": ["Yes"], "outcomePrices": [0.4, 0.6]},
            id="missing_side",
        ),
    ],
)
def test_fetch_baseline_returns_none_when_a_side_is_unusable(market: dict[str, Any]) -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.dumps([market], allow_nan=True)
        return httpx.Response(status_code=200, content=body.encode("utf-8"))

    event = _event()
    transport = httpx.MockTransport(_handler)
    with httpx.Client(
        base_url="https://gamma-api.polymarket.com",
        transport=transport,
        timeout=5.0,
    ) as client:
        baseline = fetch_polymarket_baseline(event, http_client=client)

    assert baseline is None
