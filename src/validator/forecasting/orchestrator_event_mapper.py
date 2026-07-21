"""Map orchestrator assignment payloads to internal Event objects."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid5

from src.core.events import Event
from src.core.schemas import EventCategory
from src.validator.forecasting.orchestrator_api import OrchestratorAssignment


def assignment_to_event(
    assignment: OrchestratorAssignment,
    *,
    now: datetime | None = None,
) -> Event:
    """Convert an orchestrator assignment into the internal Event schema."""
    title = assignment.event.question.strip()
    if not title:
        raise ValueError("assignment.event.question is required")

    source = assignment.event.source.strip()
    if not source:
        raise ValueError("assignment.event.source is required")

    source_id = (assignment.event.sourceMarketId or "").strip() or assignment.event.marketId
    description = (assignment.event.description or "").strip() or title

    resolution_deadline = _ensure_aware_utc(assignment.event.endDate)
    created_at = _ensure_aware_utc(now or datetime.now(timezone.utc))
    event_id = uuid5(NAMESPACE_URL, f"{source}:{source_id}")

    return Event(
        event_id=event_id,
        title=title,
        description=description,
        category=EventCategory.OTHER,
        subcategory=None,
        resolution_criteria=description,
        resolution_deadline=resolution_deadline,
        created_at=created_at,
        source=source,
        source_id=source_id,
        source_url=None,
        tags=[],
        outcomes=[
            {str(k): str(v) for k, v in o.items() if isinstance(v, str)}
            for o in assignment.event.outcomes
            if isinstance(o, dict)
        ],
        current_outcome_prices=dict(assignment.event.currentOutcomePrices),
    )


def _ensure_aware_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
