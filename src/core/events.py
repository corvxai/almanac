"""Event data structures — the questions the system is forecasting on."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from src.core.schemas import EventCategory


class Event(BaseModel):
    """A binary forecasting event, typically sourced from prediction markets."""

    event_id: UUID = Field(default_factory=uuid4)
    title: str
    description: str
    category: EventCategory
    subcategory: Optional[str] = None
    # Optional; orchestrator assignments leave this unset so market description
    # is agent-only and does not land in sealed EventSnapshot traces.
    resolution_criteria: Optional[str] = None
    resolution_deadline: datetime
    created_at: datetime = Field(default_factory=lambda: datetime.utcnow())
    source: Optional[str] = None
    source_id: Optional[str] = None
    source_url: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    # Outcome ids/names from the source market (agent-visible). Live prices are
    # kept on the assignment for submit/scoring and are not populated on the
    # agent-facing Event — `current_outcome_prices` stays empty in production
    # runs so agents cannot copy the quote at prediction time.
    outcomes: list[dict[str, str]] = Field(default_factory=list)
    current_outcome_prices: dict[str, float] = Field(default_factory=dict)
