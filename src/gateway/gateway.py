"""Provider Gateway — the validator's proxy layer for all agent data access.

Every call an agent makes flows through here. The gateway:
1. Dispatches to the correct provider adapter
2. Hashes the raw response (SHA-256)
3. Runs the evidence extraction pipeline
4. Returns data to the agent
5. Logs the complete ProviderCall record

The agent cannot influence what gets recorded.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from src.core import constants
from src.core.schemas import ProviderCall, ProviderTier, ResponseMeta, UsageMeta
from src.gateway.cost_estimator import estimate_cost_usd
from src.gateway.extractor import extract_evidence, extract_sources
from src.gateway.providers.base import BaseProvider

logger = logging.getLogger("arcratio.gateway")


def default_summarise_params(call_type: str, params: dict[str, Any]) -> str:
    """Same default `BaseProvider.summarise_params` produces, but standalone.

    The validator-local proxy does not hold a `BaseProvider` instance — it only
    forwards POSTs to the central gateway — so it needs an off-provider
    fallback to fill `ProviderCall.query_params_summary`.
    """
    parts = [f"{k}={v}" for k, v in sorted(params.items())]
    return f"{call_type}({', '.join(parts)})"


def build_provider_call_record(
    *,
    call_index: int,
    provider_id: str,
    provider_tier: ProviderTier,
    call_type: str,
    params: dict[str, Any],
    query_params_summary: str,
    raw_response: dict[str, Any],
    latency_ms: int,
) -> ProviderCall:
    """Build a fully-populated `ProviderCall` from a raw response.

    Pure function — does not mutate any caller state. Used by both
    `ProviderGateway.call_provider` (validator in-process path) and
    `local_proxy` (validator-side proxy path) so the trace shape is identical
    regardless of whether the agent ran in-process or in a sandbox.
    """
    _maybe_log_raw_response(provider_id=provider_id, call_type=call_type, raw_response=raw_response)
    raw_json = json.dumps(raw_response, sort_keys=True, default=str)
    raw_hash = hashlib.sha256(raw_json.encode()).hexdigest()

    evidence = extract_evidence(provider_id, call_type, raw_response)
    sources_accessed = extract_sources(provider_id, raw_response)

    response_bytes = len(raw_json.encode())
    record_count = None
    if isinstance(raw_response.get("results"), list):
        record_count = len(raw_response["results"])
    elif isinstance(raw_response.get("price_history"), list):
        record_count = len(raw_response["price_history"])

    now = datetime.now(timezone.utc)
    meta = ResponseMeta(
        response_size_bytes=response_bytes,
        record_count=record_count,
        data_freshness=now,
    )
    usage_meta, provider_usage_raw = ProviderGateway._normalise_usage(raw_response)
    model = _extract_model(params, raw_response)
    if usage_meta is not None and usage_meta.cost is None:
        est = estimate_cost_usd(provider_id, model, usage_meta, provider_usage_raw)
        if est is not None:
            usage_meta.cost = est
    cost_units = usage_meta.cost if usage_meta is not None and usage_meta.cost is not None else 0.0

    return ProviderCall(
        call_index=call_index,
        provider_id=provider_id,
        model=model,
        provider_tier=provider_tier,
        call_type=call_type,
        query_params_summary=query_params_summary,
        response_meta=meta,
        usage_meta=usage_meta,
        provider_usage_raw=provider_usage_raw,
        extracted_evidence=evidence,
        sources_accessed=sources_accessed,
        raw_response_hash=raw_hash,
        latency_ms=latency_ms,
        cost_units=cost_units,
    )


def _maybe_log_raw_response(*, provider_id: str, call_type: str, raw_response: dict[str, Any]) -> None:
    if not constants.GATEWAY.debug_log_raw_response:
        return
    raw_json = json.dumps(raw_response, sort_keys=True, default=str)
    max_chars = max(0, int(constants.GATEWAY.debug_raw_response_max_chars))
    if max_chars > 0 and len(raw_json) > max_chars:
        logger.warning(
            "Gateway raw response debug provider=%s call_type=%s size=%d truncated=true payload=%s...(truncated %d chars)",
            provider_id,
            call_type,
            len(raw_json),
            raw_json[:max_chars],
            len(raw_json) - max_chars,
        )
        return
    logger.warning(
        "Gateway raw response debug provider=%s call_type=%s size=%d truncated=false payload=%s",
        provider_id,
        call_type,
        len(raw_json),
        raw_json,
    )


class ProviderGateway:
    """Central proxy that mediates all agent ↔ provider communication."""

    def __init__(self, providers: dict[str, BaseProvider] | None = None):
        self._providers: dict[str, BaseProvider] = providers or {}
        self._call_log: list[ProviderCall] = []
        self._call_counter: int = 0

    def register_provider(self, provider: BaseProvider) -> None:
        self._providers[provider.provider_id] = provider

    @property
    def call_log(self) -> list[ProviderCall]:
        """Read-only access to the accumulated call records."""
        return list(self._call_log)

    def reset(self) -> None:
        """Clear call log for a fresh execution."""
        self._call_log.clear()
        self._call_counter = 0

    def call_provider(
        self,
        provider_id: str,
        call_type: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a provider call, record everything, return data to caller.

        This is the ONLY way agents should access external data.
        """
        provider = self._providers.get(provider_id)
        if provider is None:
            raise ValueError(
                f"Unknown provider '{provider_id}'. "
                f"Available: {list(self._providers.keys())}"
            )

        call_index = self._call_counter
        self._call_counter += 1

        t0 = time.monotonic()
        raw_response = provider.call(call_type, params)
        latency_ms = int((time.monotonic() - t0) * 1000)

        provider_call = build_provider_call_record(
            call_index=call_index,
            provider_id=provider_id,
            provider_tier=provider.provider_tier,
            call_type=call_type,
            params=params,
            query_params_summary=provider.summarise_params(call_type, params),
            raw_response=raw_response,
            latency_ms=latency_ms,
        )
        self._call_log.append(provider_call)

        return raw_response

    @staticmethod
    def _normalise_usage(raw_response: dict[str, Any]) -> tuple[UsageMeta | None, dict[str, Any] | None]:
        usage = raw_response.get("usage")
        usage_meta = raw_response.get("usageMetadata")

        if isinstance(usage, dict):
            prompt_tokens = _safe_int(usage.get("prompt_tokens"))
            completion_tokens = _safe_int(usage.get("completion_tokens"))
            input_tokens = _safe_int(usage.get("input_tokens"))
            output_tokens = _safe_int(usage.get("output_tokens"))
            # OpenRouter/other providers often use camelCase usage keys.
            if prompt_tokens is None:
                prompt_tokens = _safe_int(usage.get("promptTokens"))
            if completion_tokens is None:
                completion_tokens = _safe_int(usage.get("completionTokens"))
            if input_tokens is None:
                input_tokens = _safe_int(usage.get("inputTokens"))
            if output_tokens is None:
                output_tokens = _safe_int(usage.get("outputTokens"))
            if prompt_tokens is None and input_tokens is not None:
                prompt_tokens = input_tokens
            if completion_tokens is None and output_tokens is not None:
                completion_tokens = output_tokens
            total_tokens = _safe_int(usage.get("total_tokens"))
            if total_tokens is None:
                total_tokens = _safe_int(usage.get("totalTokens"))
            if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens

            prompt_details = usage.get("prompt_tokens_details")
            cost_details = usage.get("cost_details")
            completion_details = usage.get("completion_tokens_details")
            cached_tokens = None
            if isinstance(prompt_details, dict):
                cached_tokens = _safe_int(prompt_details.get("cached_tokens"))
            if cached_tokens is None:
                cached_tokens = _safe_int(usage.get("cache_read_input_tokens"))
            if cached_tokens is None:
                cached_tokens = _safe_int(usage.get("cacheReadInputTokens"))
            reasoning_tokens = None
            if isinstance(completion_details, dict):
                reasoning_tokens = _safe_int(completion_details.get("reasoning_tokens"))
            cost = _safe_float(usage.get("cost"))
            if cost is None:
                cost = _safe_float(usage.get("costUsd"))
            if cost is None and isinstance(cost_details, dict):
                cost = _safe_float(cost_details.get("upstream_inference_cost"))

            return UsageMeta(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
                reasoning_tokens=reasoning_tokens,
                cost=cost,
                is_byok=_safe_bool(usage.get("is_byok")),
            ), usage

        if isinstance(usage_meta, dict):
            prompt_tokens = _safe_int(usage_meta.get("promptTokenCount"))
            completion_tokens = _safe_int(usage_meta.get("candidatesTokenCount"))
            total_tokens = _safe_int(usage_meta.get("totalTokenCount"))
            if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens

            return UsageMeta(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ), usage_meta

        return None, None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes"}:
            return True
        if v in {"false", "0", "no"}:
            return False
    return None


def _extract_model(params: dict[str, Any], raw_response: dict[str, Any]) -> str | None:
    model = params.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    raw_model = raw_response.get("model")
    if isinstance(raw_model, str) and raw_model.strip():
        return raw_model.strip()
    raw_model_version = raw_response.get("modelVersion")
    if isinstance(raw_model_version, str) and raw_model_version.strip():
        return raw_model_version.strip()
    return None
