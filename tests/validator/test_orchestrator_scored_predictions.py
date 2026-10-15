from __future__ import annotations

import logging
from types import SimpleNamespace

import httpx
import pytest

from src.core.constants import ORCHESTRATOR_API_URL
from src.validator.forecasting.orchestrator_api import (
    SCORED_PREDICTIONS_CONNECT_TIMEOUT_SECONDS,
    SCORED_PREDICTIONS_ENDPOINT,
    SCORED_PREDICTIONS_PAGE_LIMIT,
    SCORED_PREDICTIONS_TIMEOUT_SECONDS,
    AUTH_DOMAIN,
    fetch_all_scored_predictions,
    fetch_scored_predictions_page,
)


class _FakeKeypair:
    def __init__(self) -> None:
        self.last_message: bytes | None = None
        self.messages: list[bytes] = []

    def sign(self, message: bytes) -> bytes:
        self.last_message = message
        self.messages.append(message)
        return bytes.fromhex("aabb")


def _loaded():
    return SimpleNamespace(
        keypair=_FakeKeypair(),
        hotkey_ss58="5ValidatorHotkey",
        coldkey_ss58=None,
    )


def test_fetch_scored_predictions_page_parses_payload_and_signs_path_query() -> None:
    loaded_hotkey = _loaded()

    def _handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == SCORED_PREDICTIONS_ENDPOINT
        assert request.url.params["days"] == "30"
        assert request.url.params["limit"] == "1000"
        assert request.url.params["includeTraceSummary"] == "1"
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "agentPredictionId": "ap_01JX9F8KQ4R2M7N6T3V1W5Y8Z",
                        "minerHotkey": "5F3sa2TJAWMqDhXG6jhV4N8ko9wqXxv6cM4x2eRk9WQh7YbA",
                        "minerUid": 184,
                        "marketId": "cmcz0b6h90002v6b8r5n1k3qf",
                        "sourceMarketId": "0xabc123...",
                        "predictedOutcomeId": "yes",
                        "predictedAt": "2026-06-08T15:10:00.000Z",
                        "minerLastPredictedAt": "2026-06-08T16:00:00.000Z",
                        "confidence": 0.71,
                        "outcomePricesAtPrediction": {"yes": 0.64, "no": 0.36},
                        "outcomeProbabilities": {"yes": 0.71, "no": 0.29},
                        "resolvedOutcomeId": "yes",
                        "finalOutcomePrices": {"yes": 1, "no": 0},
                        "predictionIsInvalid": False,
                        "predictionInvalidReason": None,
                        "predictionValidation": {
                            "isValid": True,
                            "reasons": [],
                            "rawObserved": {"prediction": "YES", "confidence": "0.71"},
                        },
                        "scoredAt": "2026-06-08T15:59:40.000Z",
                        "resolutionStatus": "resolved",
                        "traceSummary": {"strategy": "trend + priors"},
                    }
                ],
                "nextCursor": "abc",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(_handler))
    page = fetch_scored_predictions_page(
        base_url=ORCHESTRATOR_API_URL,
        loaded_hotkey=loaded_hotkey,  # type: ignore[arg-type]
        http_client=client,
    )
    assert len(page.items) == 1
    assert page.items[0].agentPredictionId == "ap_01JX9F8KQ4R2M7N6T3V1W5Y8Z"
    assert page.items[0].predictedAt is not None
    assert page.items[0].minerLastPredictedAt is not None
    assert page.items[0].outcomePricesAtPrediction == {"yes": 0.64, "no": 0.36}
    assert page.items[0].predictionValidation is not None
    assert page.items[0].predictionValidation.isValid is True
    assert page.nextCursor == "abc"

    expected = (
        f"{AUTH_DOMAIN}\nGET\n{SCORED_PREDICTIONS_ENDPOINT}?days=30&limit=1000&includeTraceSummary=1\n\n"
    )
    assert loaded_hotkey.keypair.last_message is not None
    assert loaded_hotkey.keypair.last_message.decode("utf-8").startswith(expected)


def test_fetch_all_scored_predictions_paginates_to_completion() -> None:
    loaded_hotkey = _loaded()
    seen_cursors: list[str | None] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        cursor = request.url.params.get("cursor")
        seen_cursors.append(cursor)
        if cursor is None:
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "agentPredictionId": "ap_1",
                            "minerHotkey": "5F3sa2TJAWMqDhXG6jhV4N8ko9wqXxv6cM4x2eRk9WQh7YbA",
                            "minerUid": 184,
                            "marketId": "m1",
                            "sourceMarketId": "s1",
                            "predictedOutcomeId": "yes",
                            "predictedAt": "2026-06-08T10:00:00.000Z",
                            "confidence": 0.8,
                            "outcomePricesAtPrediction": {"yes": 0.58, "no": 0.42},
                            "outcomeProbabilities": {"yes": 0.8, "no": 0.2},
                            "resolvedOutcomeId": "yes",
                            "finalOutcomePrices": {"yes": 1, "no": 0},
                            "scoredAt": "2026-06-08T15:59:40.000Z",
                            "resolutionStatus": "resolved",
                        }
                    ],
                    "nextCursor": "next1",
                },
            )
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "agentPredictionId": "ap_2",
                        "minerHotkey": "5F3sa2TJAWMqDhXG6jhV4N8ko9wqXxv6cM4x2eRk9WQh7YbA",
                        "minerUid": 184,
                        "marketId": "m2",
                        "sourceMarketId": "s2",
                        "predictedOutcomeId": "no",
                        "predictedAt": "2026-06-08T12:30:00.000Z",
                        "confidence": 0.6,
                        "outcomePricesAtPrediction": {"yes": 0.4, "no": 0.6},
                        "outcomeProbabilities": {"yes": 0.4, "no": 0.6},
                        "resolvedOutcomeId": "yes",
                        "finalOutcomePrices": {"yes": 1, "no": 0},
                        "scoredAt": "2026-06-08T15:59:40.000Z",
                        "resolutionStatus": "resolved",
                    }
                ],
                "nextCursor": None,
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(_handler))
    items = fetch_all_scored_predictions(
        base_url=ORCHESTRATOR_API_URL,
        loaded_hotkey=loaded_hotkey,  # type: ignore[arg-type]
        http_client=client,
    )
    assert [it.agentPredictionId for it in items] == ["ap_1", "ap_2"]
    assert all(it.predictedAt is not None for it in items)
    assert items[0].outcomePricesAtPrediction == {"yes": 0.58, "no": 0.42}
    assert items[1].outcomePricesAtPrediction == {"yes": 0.4, "no": 0.6}
    assert seen_cursors == [None, "next1"]
    assert not client.is_closed


def _scored_item(agent_prediction_id: str) -> dict[str, object]:
    return {
        "agentPredictionId": agent_prediction_id,
        "minerHotkey": "5F3sa2TJAWMqDhXG6jhV4N8ko9wqXxv6cM4x2eRk9WQh7YbA",
        "minerUid": 184,
        "marketId": "m1",
        "sourceMarketId": "s1",
        "predictedOutcomeId": "yes",
        "predictedAt": "2026-06-08T10:00:00.000Z",
        "confidence": 0.8,
        "outcomePricesAtPrediction": {"yes": 0.58, "no": 0.42},
        "outcomeProbabilities": {"yes": 0.8, "no": 0.2},
        "resolvedOutcomeId": "yes",
        "finalOutcomePrices": {"yes": 1, "no": 0},
        "scoredAt": "2026-06-08T15:59:40.000Z",
        "resolutionStatus": "resolved",
    }


def _page(agent_prediction_id: str, next_cursor: str | None) -> httpx.Response:
    return httpx.Response(
        200,
        json={"items": [_scored_item(agent_prediction_id)], "nextCursor": next_cursor},
    )


def test_fetch_all_scored_predictions_retries_same_cursor_with_a_new_signature(
    caplog: pytest.LogCaptureFixture,
) -> None:
    loaded_hotkey = _loaded()
    seen_cursors: list[str | None] = []
    seen_limits: list[str] = []
    timeouts: list[object] = []
    attempts = {"n": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        seen_cursors.append(request.url.params.get("cursor"))
        seen_limits.append(request.url.params["limit"])
        if attempts["n"] == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        if attempts["n"] == 2:
            return httpx.Response(401, json={"detail": "unauthorized"})
        return _page("ap_1", None)

    class _RecordingClient(httpx.Client):
        def get(self, *args, **kwargs):
            timeouts.append(kwargs.get("timeout"))
            return super().get(*args, **kwargs)

    client = _RecordingClient(transport=httpx.MockTransport(_handler))
    with caplog.at_level(logging.INFO, logger="forecasting.orchestrator"):
        items = fetch_all_scored_predictions(
            base_url=ORCHESTRATOR_API_URL,
            loaded_hotkey=loaded_hotkey,  # type: ignore[arg-type]
            http_client=client,
            retry_backoff_seconds=0,
        )

    assert [it.agentPredictionId for it in items] == ["ap_1"]
    assert seen_cursors == [None, None, None]
    assert seen_limits == ["200", "200", "200"]
    assert timeouts == [
        httpx.Timeout(
            SCORED_PREDICTIONS_TIMEOUT_SECONDS,
            connect=SCORED_PREDICTIONS_CONNECT_TIMEOUT_SECONDS,
        )
    ] * 3
    assert len(loaded_hotkey.keypair.messages) == 3
    assert len(set(loaded_hotkey.keypair.messages)) == 3
    assert "page 1 attempt 1/3 failed (ReadTimeout)" in caplog.text
    assert "page 1 attempt 2/3 failed (HTTP 401)" in caplog.text
    assert "scored predictions page 1: 1 rows in" in caplog.text
    assert not client.is_closed


def test_fetch_all_scored_predictions_does_not_retry_client_errors() -> None:
    loaded_hotkey = _loaded()
    attempts = {"n": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(400, json={"detail": "bad request"})

    client = httpx.Client(transport=httpx.MockTransport(_handler))
    with pytest.raises(httpx.HTTPStatusError):
        fetch_all_scored_predictions(
            base_url=ORCHESTRATOR_API_URL,
            loaded_hotkey=loaded_hotkey,  # type: ignore[arg-type]
            http_client=client,
            retry_backoff_seconds=0,
        )
    assert attempts["n"] == 1


def test_fetch_all_scored_predictions_failure_discards_earlier_pages() -> None:
    loaded_hotkey = _loaded()
    attempts = {"n": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if request.url.params.get("cursor") is None:
            return _page("ap_1", "next1")
        return httpx.Response(503, json={"detail": "unavailable"})

    client = httpx.Client(transport=httpx.MockTransport(_handler))
    with pytest.raises(httpx.HTTPStatusError):
        fetch_all_scored_predictions(
            base_url=ORCHESTRATOR_API_URL,
            loaded_hotkey=loaded_hotkey,  # type: ignore[arg-type]
            http_client=client,
            max_attempts=2,
            retry_backoff_seconds=0,
        )
    # Page 1 once, then page 2 twice. Nothing is returned to score.
    assert attempts["n"] == 3


def test_fetch_all_scored_predictions_opens_one_client_for_the_walk(monkeypatch) -> None:
    loaded_hotkey = _loaded()
    opened: list[httpx.Client] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("cursor") is None:
            return _page("ap_1", "next1")
        return _page("ap_2", None)

    class _TrackingClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, transport=httpx.MockTransport(_handler), **kwargs)
            opened.append(self)

    monkeypatch.setattr(
        "src.validator.forecasting.orchestrator_api.httpx.Client",
        _TrackingClient,
    )
    items = fetch_all_scored_predictions(
        base_url=ORCHESTRATOR_API_URL,
        loaded_hotkey=loaded_hotkey,  # type: ignore[arg-type]
        retry_backoff_seconds=0,
    )
    assert [it.agentPredictionId for it in items] == ["ap_1", "ap_2"]
    assert len(opened) == 1
    assert opened[0].timeout.read == SCORED_PREDICTIONS_TIMEOUT_SECONDS
    assert opened[0].timeout.connect == SCORED_PREDICTIONS_CONNECT_TIMEOUT_SECONDS
    assert opened[0].is_closed
    assert SCORED_PREDICTIONS_PAGE_LIMIT == 200
