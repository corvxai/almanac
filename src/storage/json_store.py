"""JSON file storage implementation — flat-file persistence for the prototype.

Each trace is stored as a single JSON file keyed by execution_id.
Resolution records are stored per-event. Designed to be trivially swappable
to a database backend later via the TraceStore interface.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

from src.core.schemas import EvidenceDigest, ResolutionRecord
from src.storage.store import TraceStore


class JsonTraceStore(TraceStore):
    def __init__(self, data_dir: Path):
        self._traces_dir = data_dir / "traces"
        self._results_dir = data_dir / "results"
        self._traces_dir.mkdir(parents=True, exist_ok=True)
        self._results_dir.mkdir(parents=True, exist_ok=True)

    def _trace_path(self, execution_id: UUID) -> Path:
        return self._traces_dir / f"{execution_id}.json"

    def _resolution_path(self, event_id: UUID) -> Path:
        return self._results_dir / f"{event_id}.json"

    def save_trace(self, digest: EvidenceDigest) -> None:
        path = self._trace_path(digest.execution_context.execution_id)
        path.write_text(digest.model_dump_json(indent=2))

    def get_trace(self, execution_id: UUID) -> EvidenceDigest | None:
        path = self._trace_path(execution_id)
        if not path.exists():
            return None
        return EvidenceDigest.model_validate_json(path.read_text())

    def list_traces_by_event(self, event_id: UUID) -> list[EvidenceDigest]:
        return [
            t for t in self._iter_all_traces()
            if t.execution_context.event_id == event_id
        ]

    def list_traces_by_agent(self, agent_id: UUID) -> list[EvidenceDigest]:
        return [
            t for t in self._iter_all_traces()
            if t.execution_context.agent_id == agent_id
        ]

    def save_resolution(self, event_id: UUID, record: ResolutionRecord) -> None:
        path = self._resolution_path(event_id)
        path.write_text(record.model_dump_json(indent=2))

    def get_resolution(self, event_id: UUID) -> ResolutionRecord | None:
        path = self._resolution_path(event_id)
        if not path.exists():
            return None
        return ResolutionRecord.model_validate_json(path.read_text())

    def _iter_all_traces(self) -> list[EvidenceDigest]:
        traces = []
        for path in self._traces_dir.glob("*.json"):
            try:
                traces.append(EvidenceDigest.model_validate_json(path.read_text()))
            except Exception:
                continue
        return traces
