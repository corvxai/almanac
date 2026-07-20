"""JSON file storage implementation — flat-file persistence for the prototype.

Each trace is stored as a single JSON file keyed by execution_id.
Resolution records are stored per-event. Designed to be trivially swappable
to a database backend later via the TraceStore interface.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from uuid import UUID

from pydantic import ValidationError

from src.core.config import StorageConfig
from src.core.schemas import EvidenceDigest, ResolutionRecord
from src.storage.store import TraceStore

logger = logging.getLogger(__name__)


def _atomic_write_text(path: Path, text: str) -> None:
    """Durably write ``text`` to ``path`` without exposing a torn file.

    Writes to a temp file in the same directory (so ``os.replace`` is a same-
    filesystem atomic rename), fsyncs the file contents, then renames over the
    destination. A power loss mid-write leaves either the old file or the new
    file fully written — never a half-written JSON that downstream scoring would
    silently skip.
    """
    directory = path.parent
    fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        # Best-effort cleanup of the temp file; never mask the original error.
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


class NullTraceStore(TraceStore):
    """No-op trace store for production runs that submit to the orchestrator only."""

    def save_trace(self, digest: EvidenceDigest) -> None:
        return None

    def get_trace(self, execution_id: UUID) -> EvidenceDigest | None:
        return None

    def list_traces_by_event(self, event_id: UUID) -> list[EvidenceDigest]:
        return []

    def list_traces_by_agent(self, agent_id: UUID) -> list[EvidenceDigest]:
        return []

    def save_resolution(self, event_id: UUID, record: ResolutionRecord) -> None:
        return None

    def get_resolution(self, event_id: UUID) -> ResolutionRecord | None:
        return None


def build_trace_store(config: StorageConfig) -> TraceStore:
    if config.persist_traces:
        return JsonTraceStore(data_dir=config.data_dir)
    return NullTraceStore()


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
        _atomic_write_text(path, digest.model_dump_json(indent=2))

    def get_trace(self, execution_id: UUID) -> EvidenceDigest | None:
        path = self._trace_path(execution_id)
        if not path.exists():
            return None
        try:
            digest = EvidenceDigest.model_validate_json(path.read_text())
        except ValidationError as exc:
            # F3: a file that fails to parse must not raise into the caller. Log and
            # return None (same not-found contract) so one corrupt trace never breaks
            # a scoring sweep. Integrity-mismatch is handled below (parses, warns).
            logger.warning(
                "Trace %s failed to parse on read (path=%s): %s",
                execution_id,
                path,
                exc,
            )
            return None
        if not digest.verify_integrity():
            # The file parsed but its sealed hash does not match its contents —
            # i.e. it was mutated after sealing or corrupted on disk. Surface it
            # rather than feeding a tampered trace into scoring. Non-fatal: the
            # caller still receives the digest and can decide how to handle it.
            logger.warning(
                "Trace %s failed integrity verification on read (path=%s).",
                execution_id,
                path,
            )
        return digest

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
        _atomic_write_text(path, record.model_dump_json(indent=2))

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
            except Exception as exc:
                # F3: surface parse failures instead of silently swallowing them, so a
                # corrupt/legacy-shape trace file is visible in logs. Still skip it and
                # keep returning the parseable traces.
                logger.warning(
                    "Skipping unparseable trace file %s: %s", path, exc
                )
                continue
        return traces
