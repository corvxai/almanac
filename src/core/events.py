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
    resolution_criteria: str
    resolution_deadline: datetime
    created_at: datetime = Field(default_factory=lambda: datetime.utcnow())
    source: Optional[str] = None
    source_id: Optional[str] = None
    source_url: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
