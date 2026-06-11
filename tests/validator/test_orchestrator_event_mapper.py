from __future__ import annotations

from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid5

import pytest

from src.core.schemas import EventCategory
from src.validator.orchestrator_api import OrchestratorAssignment
from src.validator.orchestrator_event_mapper import assignment_to_event


def _assignment(*, description: str | None = "Binary event", source_market_id: str | None = "2411919"):
    return OrchestratorAssignment.model_validate(
        {
            "agentPredictionId": "pred_123",
            "status": "in_progress",
            "leaseExpiresAt": "2026-06-05T19:59:44.271000Z",
            "miner": {
                "minerHotkey": "5EqZoEKc6c8TaG4xRRHTT1uZiQF5jkjQCeUV5t77L6YbeaJ8",
                "minerUid": 17,
            },
            "event": {
                "marketId": "cmpxszzto007luaaznuulufaq",
                "source": "polymarket",
                "sourceMarketId": source_market_id,
                "question": "Will the White House call a full lid by 6:30PM on June 3?",
                "description": description,
                "endDate": "2026-06-06T16:00:00Z",
                "currentOutcomePrices": {
                    "44067735420073091208102132584206540755478475924816968380795164046445880451653": 0.26,
                    "50021132755437548153440115538135773388691694036974767046675729697249853152385": 0.74,
                },
            },
            "agent": {
                "agentUploadId": "cmpwwsd4800000razamvsplvc",
                "sha256": "72a9f703c400d35fecbc30be7d789aa820115e7a736af49d36d5b41b9ccb274a",
                "uploadedAt": "2026-06-02T17:25:57.224000Z",
                "code": "class Agent:\n    pass\n",
            },
        }
    )


def test_assignment_to_event_maps_core_fields() -> None:
    fixed_now = datetime(2026, 6, 5, 20, 0, tzinfo=timezone.utc)
    assignment = _assignment()
    event = assignment_to_event(assignment, now=fixed_now)

    assert event.title == assignment.event.question
    assert event.description == "Binary event"
    assert event.resolution_criteria == "Binary event"
    assert event.resolution_deadline == datetime(2026, 6, 6, 16, 0, tzinfo=timezone.utc)
    assert event.created_at == fixed_now
    assert event.category == EventCategory.OTHER
    assert event.source == "polymarket"
    assert event.source_id == "2411919"
    assert event.event_id == uuid5(NAMESPACE_URL, "polymarket:2411919")


def test_assignment_to_event_fallbacks_description_and_source_id() -> None:
    assignment = _assignment(description=None, source_market_id=None)
    event = assignment_to_event(assignment)

    assert event.description == assignment.event.question
    assert event.resolution_criteria == assignment.event.question
    assert event.source_id == assignment.event.marketId
    assert event.event_id == uuid5(
        NAMESPACE_URL, f"{assignment.event.source}:{assignment.event.marketId}"
    )


def test_assignment_to_event_rejects_blank_required_fields() -> None:
    assignment = _assignment()
    assignment.event.question = "   "
    with pytest.raises(ValueError, match="question"):
        assignment_to_event(assignment)
