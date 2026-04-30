"""Abstract storage interface — swap implementations without touching consumers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from src.core.schemas import EvidenceDigest, ResolutionRecord


class TraceStore(ABC):
    """Persistence layer for evidence digest traces and resolution records."""

    @abstractmethod
    def save_trace(self, digest: EvidenceDigest) -> None:
        """Persist a complete evidence digest."""
        ...

    @abstractmethod
    def get_trace(self, execution_id: UUID) -> EvidenceDigest | None:
        """Retrieve a trace by its execution ID, or None if not found."""
        ...

    @abstractmethod
    def list_traces_by_event(self, event_id: UUID) -> list[EvidenceDigest]:
        """Return all traces for a given event."""
        ...

    @abstractmethod
    def list_traces_by_agent(self, agent_id: UUID) -> list[EvidenceDigest]:
        """Return all traces produced by a given agent."""
        ...

    @abstractmethod
    def save_resolution(self, event_id: UUID, record: ResolutionRecord) -> None:
        """Store a resolution outcome for an event."""
        ...

    @abstractmethod
    def get_resolution(self, event_id: UUID) -> ResolutionRecord | None:
        """Retrieve the resolution record for an event, or None."""
        ...
