"""Per-track endpoint allowlist for the validator-local signing proxy.

Inspired by Numinous's `track_config.py`. Each "track" is a class of agent
execution; the local proxy only forwards calls whose `provider_id` is in the
list for that track. The orchestrator picks a track when it registers a run.

Use `*` to allow every provider. Otherwise the entry is an explicit list of
`provider_id` values that match the gateway's registered providers
(see `src/gateway/server.py::register_provider`).
"""

from __future__ import annotations

TRACK_ALLOWED_PROVIDERS: dict[str, list[str]] = {
    "MAIN": ["*"],
    "SIGNAL": ["polymarket", "web_search"],
}


def is_provider_allowed(track: str, provider_id: str) -> bool:
    """Return True if `provider_id` is permitted under `track`.

    Unknown tracks are permissive in v1 — we treat them as "MAIN" so a misnamed
    config doesn't silently lock everything out. The proxy's per-run registry
    is the authoritative gate; this is a second-line policy check.
    """
    allowlist = TRACK_ALLOWED_PROVIDERS.get(track)
    if allowlist is None:
        return True
    if "*" in allowlist:
        return True
    return provider_id in allowlist
