"""Abstract base provider — all data providers implement this interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.core.schemas import ProviderTier


class BaseProvider(ABC):
    """Interface that every data provider adapter must implement.

    Providers are stateless adapters responsible for fetching data from an
    external source and returning it in a standardised dict structure.
    Evidence extraction is NOT the provider's job — that happens in the
    gateway's extraction pipeline.
    """

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique identifier for this provider (e.g. 'polymarket')."""
        ...

    @property
    @abstractmethod
    def provider_tier(self) -> ProviderTier:
        ...

    @abstractmethod
    def call(self, call_type: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a data call and return the raw response as a dict.

        Args:
            call_type: The specific endpoint or operation (e.g. 'get_market').
            params: Arbitrary parameters for the call.

        Returns:
            Raw response data as a dictionary. The gateway will hash and
            extract evidence from this independently.
        """
        ...

    def summarise_params(self, call_type: str, params: dict[str, Any]) -> str:
        """Return a human-readable summary of the query parameters.

        Default implementation joins key=value pairs. Providers can override
        for more meaningful descriptions.
        """
        parts = [f"{k}={v}" for k, v in sorted(params.items())]
        return f"{call_type}({', '.join(parts)})"
