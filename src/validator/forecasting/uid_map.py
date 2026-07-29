"""Resolve a miner UID from a trace.

v1 fallback chain — honest about what's actually present in traces today:

1. ``prediction_output.metadata["miner_uid"]`` — explicit, takes precedence.
2. ``prediction_output.metadata["hotkey_ss58"]`` — looked up against the
   current metagraph's ``hotkeys`` list.
3. ``execution_context`` does not yet carry a hotkey field. If neither
   metadata path is present, return ``None`` and log once per
   ``(execution_id, reason)`` so a misconfigured miner pipeline surfaces
   without spamming the validator log every epoch.

Returning ``None`` is a hard skip: the scorer drops the trace, and that
miner gets zero forecasting score until the trace pipeline starts attaching
identity to its prediction output.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.core.schemas import EvidenceDigest

logger = logging.getLogger("forecasting.uid_map")


_seen_misses: set[tuple[str, str]] = set()


def _log_once(execution_id: str, reason: str) -> None:
    key = (execution_id, reason)
    if key in _seen_misses:
        return
    _seen_misses.add(key)
    logger.warning("unmapped_trace execution_id=%s reason=%s", execution_id, reason)


def resolve(trace: EvidenceDigest, metagraph) -> Optional[int]:
    """Return the miner UID for ``trace`` against ``metagraph``, or ``None``.

    ``metagraph`` is duck-typed: any object exposing a list-like
    ``hotkeys`` attribute and a list/array-like ``uids`` attribute works
    (this lets tests pass a tiny stub without bittensor installed).
    """
    metadata = trace.prediction_output.metadata or {}
    execution_id_str = str(trace.execution_context.execution_id)

    explicit_uid = metadata.get("miner_uid")
    if explicit_uid is not None:
        try:
            uid = int(explicit_uid)
        except (TypeError, ValueError):
            _log_once(execution_id_str, "miner_uid_not_int")
            return None
        if not _uid_in_metagraph(uid, metagraph):
            _log_once(execution_id_str, f"miner_uid_{uid}_not_in_metagraph")
            return None
        return uid

    hotkey = metadata.get("hotkey_ss58") or metadata.get("miner_hotkey")
    if isinstance(hotkey, str) and hotkey:
        hotkeys = list(getattr(metagraph, "hotkeys", []) or [])
        try:
            return hotkeys.index(hotkey)
        except ValueError:
            _log_once(execution_id_str, "hotkey_not_in_metagraph")
            return None

    _log_once(execution_id_str, "no_uid_or_hotkey_in_metadata")
    return None


def _uid_in_metagraph(uid: int, metagraph) -> bool:
    uids = getattr(metagraph, "uids", None)
    if uids is None:
        return any(int(neuron.uid) == uid for neuron in metagraph)
    try:
        uid_list = uids.tolist()
    except AttributeError:
        uid_list = list(uids)
    return uid in uid_list
