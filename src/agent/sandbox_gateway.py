"""Thin gateway-shaped facade used inside the Docker sandbox.

Agents running in-process get a real `ProviderGateway` that records calls and
extracts evidence. Agents running inside the sandbox get this `SandboxGateway`
instead — the canonical call log lives on the validator-local signing proxy,
not in-process, so this object has no bookkeeping responsibilities. It just
forwards calls through `RemoteProvider` instances configured by the runner
entrypoint to use the UDS transport + the per-run `X-Run-Id` header.
"""

from __future__ import annotations

from typing import Any, Callable

from src.gateway.client import RemoteProvider


class SandboxGateway:
    """`ProviderGateway`-shaped object that lazily creates RemoteProviders.

    The validator-local proxy is the authoritative recorder of provider calls
    when an agent runs inside the sandbox. This class therefore intentionally
    does not build `ProviderCall` records — duplicating that work inside the
    sandbox would only enlarge the runner image's attack surface.
    """

    def __init__(
        self,
        provider_factory: Callable[[str], RemoteProvider],
    ) -> None:
        """`provider_factory(provider_id)` must return a fully configured
        `RemoteProvider` (UDS client + `X-Run-Id` header attached). The
        runner entrypoint provides this; tests can substitute a stub.
        """
        self._provider_factory = provider_factory
        self._providers: dict[str, RemoteProvider] = {}

    def call_provider(
        self,
        provider_id: str,
        call_type: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        provider = self._providers.get(provider_id)
        if provider is None:
            provider = self._provider_factory(provider_id)
            self._providers[provider_id] = provider
        return provider.call(call_type, params or {})
