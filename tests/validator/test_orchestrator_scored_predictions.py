from __future__ import annotations

from types import SimpleNamespace

import httpx

from src.core.constants import ORCHESTRATOR_API_URL
from src.validator.orchestrator_api import (
    SCORED_PREDICTIONS_ENDPOINT,
    AUTH_DOMAIN,
    fetch_all_scored_predictions,
    fetch_scored_predictions_page,
)


class _FakeKeypair:
    def __init__(self) -> None:
        self.last_message: bytes | None = None

    def sign(self, message: bytes) -> bytes:
        self.last_message = message
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
                        "confidence": 0.71,
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
                            "confidence": 0.8,
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
                        "confidence": 0.6,
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
    assert seen_cursors == [None, "next1"]
