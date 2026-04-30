"""Token-based cost estimation for providers that do not return per-call cost.

This is intentionally deterministic (usage tokens x configured rates) rather
than heuristic. For billing-critical systems, reconcile periodically against
provider-side billing exports/APIs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.core.schemas import UsageMeta

_PER_MILLION = 1_000_000.0
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PRICE_FILE = _PROJECT_ROOT / "config" / "pricing_cards.json"


@dataclass(frozen=True, slots=True)
class PriceCard:
    """USD list pricing per 1M tokens."""

    input_per_mtoken: float
    output_per_mtoken: float
    cache_read_input_per_mtoken: float | None = None
    cache_creation_input_per_mtoken: float | None = None


def estimate_cost_usd(
    provider_id: str,
    model: str | None,
    usage_meta: UsageMeta | None,
    provider_usage_raw: dict | None,
) -> float | None:
    """Estimate request cost in USD when provider payload omits direct cost."""
    if usage_meta is None:
        return None

    price_card = _resolve_price_card(provider_id, model)
    if price_card is None:
        return None

    input_tokens = _nz(usage_meta.input_tokens if usage_meta.input_tokens is not None else usage_meta.prompt_tokens)
    output_tokens = _nz(usage_meta.output_tokens if usage_meta.output_tokens is not None else usage_meta.completion_tokens)

    cache_read_tokens = 0
    cache_creation_tokens = 0
    if isinstance(provider_usage_raw, dict):
        cache_read_tokens = _nz(_as_int(provider_usage_raw.get("cache_read_input_tokens")))
        cache_creation_tokens = _nz(_as_int(provider_usage_raw.get("cache_creation_input_tokens")))
        # Claude sometimes nests cache-creation buckets instead of top-level totals.
        if cache_creation_tokens == 0 and isinstance(provider_usage_raw.get("cache_creation"), dict):
            cc = provider_usage_raw["cache_creation"]
            cache_creation_tokens = (
                _nz(_as_int(cc.get("ephemeral_5m_input_tokens")))
                + _nz(_as_int(cc.get("ephemeral_1h_input_tokens")))
            )

    # Treat input_tokens as total input; split cached read portion if present.
    uncached_input_tokens = max(input_tokens - cache_read_tokens, 0)

    cost = (
        (uncached_input_tokens / _PER_MILLION) * price_card.input_per_mtoken
        + (output_tokens / _PER_MILLION) * price_card.output_per_mtoken
    )
    if price_card.cache_read_input_per_mtoken is not None:
        cost += (cache_read_tokens / _PER_MILLION) * price_card.cache_read_input_per_mtoken
    if price_card.cache_creation_input_per_mtoken is not None:
        cost += (cache_creation_tokens / _PER_MILLION) * price_card.cache_creation_input_per_mtoken

    return round(cost, 10)


def _resolve_price_card(provider_id: str, model: str | None) -> PriceCard | None:
    cards = _price_cards()
    if model:
        exact = cards.get(f"{provider_id}/{model}")
        if exact is not None:
            return exact
        fam = _family_fallback_key(provider_id, model)
        if fam is not None and cards.get(fam) is not None:
            return cards[fam]

    # Provider-specific fallback key, configured in the cards file.
    fallback = cards.get(f"{provider_id}/__default__")
    if fallback is not None:
        return fallback
    return None


@lru_cache(maxsize=1)
def _price_cards() -> dict[str, PriceCard]:
    cards = _load_price_cards_from_file()
    return _merge_env_override(cards)


def _load_price_cards_from_file() -> dict[str, PriceCard]:
    path = Path(os.environ.get("ARCRATIO_MODEL_PRICING_FILE", str(_DEFAULT_PRICE_FILE)))
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}

    cards: dict[str, PriceCard] = {}
    for key, val in parsed.items():
        card = _parse_price_card(key, val)
        if card is not None:
            cards[key] = card
    return cards


def _merge_env_override(base_cards: dict[str, PriceCard]) -> dict[str, PriceCard]:
    raw = os.environ.get("ARCRATIO_MODEL_PRICING_JSON", "").strip()
    if not raw:
        return base_cards

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return base_cards
    if not isinstance(parsed, dict):
        return base_cards

    cards = dict(base_cards)
    for key, val in parsed.items():
        card = _parse_price_card(key, val)
        if card is not None:
            cards[key] = card
    return cards


def _parse_price_card(key: object, value: object) -> PriceCard | None:
    if not isinstance(key, str) or not isinstance(value, dict):
        return None
    try:
        return PriceCard(
            input_per_mtoken=float(value["input_per_mtoken"]),
            output_per_mtoken=float(value["output_per_mtoken"]),
            cache_read_input_per_mtoken=_as_float(value.get("cache_read_input_per_mtoken")),
            cache_creation_input_per_mtoken=_as_float(value.get("cache_creation_input_per_mtoken")),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _nz(value: int | None) -> int:
    return value if value is not None else 0


def _family_fallback_key(provider_id: str, model: str) -> str | None:
    m = model.lower()
    if provider_id != "claude":
        return None
    if "opus" in m:
        return "claude/__default_opus__"
    if "sonnet" in m:
        return "claude/__default_sonnet__"
    if "haiku" in m:
        return "claude/__default_haiku__"
    return None
