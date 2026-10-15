"""Validator client for orchestrator agent/event assignment polling."""

from __future__ import annotations

import json
import hashlib
import logging
import secrets
import time
from datetime import datetime
from typing import Any

import httpx
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError

from src.gateway.signing import LoadedKeypair
from src.gateway.validator_protocol import canonical_json as _canonical_json

AGENT_AND_EVENT_ENDPOINT = "/v1/validators/agent-and-event"
PREDICTION_ENDPOINT = "/v1/validators/prediction"
SCORED_PREDICTIONS_ENDPOINT = "/v1/validators/scored-predictions"
DEFAULT_TIMEOUT_SECONDS = 15.0
# Scored-predictions pages are much heavier than assignment polls. Keep the
# assignment timeout at DEFAULT_TIMEOUT_SECONDS; this walk has its own budget.
# The walk runs on the validator's main loop, so a hung page blocks assignment
# polling for up to max_attempts * read timeout. Connect fails fast because a
# slow connect means the host is down, not that the page is large.
SCORED_PREDICTIONS_TIMEOUT_SECONDS = 30.0
SCORED_PREDICTIONS_CONNECT_TIMEOUT_SECONDS = 5.0
SCORED_PREDICTIONS_PAGE_LIMIT = 200
SCORED_PREDICTIONS_MAX_ATTEMPTS = 3
AUTH_DOMAIN = "sub41-gateway-v1"

logger = logging.getLogger("forecasting.orchestrator")


class OrchestratorRequestError(RuntimeError):
    """Orchestrator HTTP failure; details are already logged at the call site."""


class AssignmentMiner(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")
    minerHotkey: str = Field(validation_alias=AliasChoices("minerHotkey", "hotkey"))
    minerUid: int = Field(validation_alias=AliasChoices("minerUid", "uid"))


class AssignmentEvent(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")
    marketId: str
    source: str
    sourceMarketId: str | None = None
    question: str
    description: str | None = None
    endDate: datetime
    outcomes: list[dict[str, Any]] = Field(default_factory=list)
    currentOutcomePrices: dict[str, float] = Field(default_factory=dict)


class AssignmentAgent(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")
    agentUploadId: str = Field(validation_alias=AliasChoices("agentUploadId", "uploadId"))
    sha256: str
    uploadedAt: datetime
    code: str | None = None


class OrchestratorAssignment(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")
    agentPredictionId: str
    status: str | None = None
    leaseExpiresAt: datetime
    miner: AssignmentMiner
    event: AssignmentEvent
    agent: AssignmentAgent


class AgentAndEventResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")
    assignment: OrchestratorAssignment | None
    reason: str | None = None


class SubmitPredictionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")
    ok: bool
    agentPredictionId: str
    status: str


class PredictionValidation(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")
    isValid: bool
    reasons: list[str] = Field(default_factory=list)
    rawObserved: dict[str, Any] = Field(default_factory=dict)


class ScoredPredictionItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")
    agentPredictionId: str
    minerHotkey: str
    minerUid: int | None = None
    marketId: str
    sourceMarketId: str
    predictedOutcomeId: str | None = None
    predictedAt: datetime
    minerLastPredictedAt: datetime | None = None
    confidence: float | None = None
    outcomePricesAtPrediction: dict[str, float] | None = None
    outcomeProbabilities: dict[str, float] | None = None
    resolvedOutcomeId: str | None = None
    finalOutcomePrices: dict[str, float] | None = None
    predictionIsInvalid: bool | None = None
    predictionInvalidReason: str | None = None
    predictionValidation: PredictionValidation | None = None
    scoredAt: datetime
    resolutionStatus: str
    traceSummary: dict[str, Any] | None = None


class ScoredPredictionsPage(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")
    items: list[ScoredPredictionItem]
    nextCursor: str | None = None


def build_assignment_auth_headers(
    *,
    loaded_hotkey: LoadedKeypair,
    method: str = "GET",
    path_and_query: str = AGENT_AND_EVENT_ENDPOINT,
    body: bytes = b"",
    nonce: str | None = None,
    timestamp: int | None = None,
) -> dict[str, str]:
    """Build auth headers for ``GET /v1/validators/agent-and-event``."""
    ts = int(time.time() * 1000) if timestamp is None else int(timestamp)
    request_nonce = nonce or secrets.token_hex(16)
    body_hash = hashlib.sha256(body).hexdigest()
    # validator-only routes sign with empty miner field by contract.
    canonical = "\n".join(
        [
            AUTH_DOMAIN,
            method.upper(),
            path_and_query,
            "",
            request_nonce,
            str(ts),
            body_hash,
        ]
    )
    signature = loaded_hotkey.keypair.sign(canonical.encode("utf-8"))
    signature_hex = (
        signature.hex() if isinstance(signature, (bytes, bytearray)) else str(signature)
    )
    return {
        "x-validator-hotkey": loaded_hotkey.hotkey_ss58,
        "x-validator-signature": f"0x{signature_hex}" if not str(signature_hex).lower().startswith("0x") else str(signature_hex),
        "x-validator-nonce": request_nonce,
        "x-validator-timestamp": str(ts),
        "accept": "application/json",
    }


def fetch_agent_event_assignment(
    *,
    base_url: str,
    loaded_hotkey: LoadedKeypair,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    http_client: httpx.Client | None = None,
) -> AgentAndEventResponse:
    """Fetch one assignment from orchestrator and parse assignment/reason."""
    # Example response payloads from GET /v1/validators/agent-and-event:
    #
    # {
    #   "assignment": {
    #     "agentPredictionId": "ap_01JX6Q6G3V9KQ7M9V4Z2M8P1T3",
    #     "status": "in_progress",
    #     "leaseExpiresAt": "2026-06-05T15:10:00.000Z",
    #     "miner": {
    #       "minerHotkey": "5F3sa2TJAWMqDhXG6jhV4N8ko9wqXxv6cM4x2eRk9WQh7YbA",
    #       "minerUid": 184
    #     },
    #     "event": {
    #       "marketId": "cmcyr8w2b0003v6b8p9u4n7sa",
    #       "source": "polymarket",
    #       "sourceMarketId": "0x8f3c2f7e4f9b3a7d2e1c6b5a4d3e2f1a9b8c7d6e",
    #       "question": "Will CPI YoY be above 3.2% in June 2026?",
    #       "description": "Binary market on June CPI release. Resolve YES if official CPI YoY > 3.2%.",
    #       "endDate": "2026-06-12T12:30:00.000Z",
    #       "currentOutcomePrices": {"yes": 0.64, "no": 0.36}
    #     },
    #     "agent": {
    #       "agentUploadId": "cmcxr9yd1000av6b87h2nk3qw",
    #       "sha256": "bd3f8a4d52f3f0f8f0b5f6d53e5d6b2d7c8f3f7a2a4b6d1e9a0b3c4d5e6f7a8",
    #       "uploadedAt": "2026-06-05T13:42:11.000Z",
    #       "code": "import math\\n\\nclass Agent:\\n    def predict(self, input_data):\\n        return {\\"yes\\": 0.67, \\"no\\": 0.33}\\n"
    #     }
    #   },
    #   "reason": null
    # }
    #
    # Empty queue:
    # {"assignment": null, "reason": "none_available"}
    headers = build_assignment_auth_headers(
        loaded_hotkey=loaded_hotkey,
        method="GET",
        path_and_query=AGENT_AND_EVENT_ENDPOINT,
        body=b"",
    )
    target = f"{base_url.rstrip('/')}{AGENT_AND_EVENT_ENDPOINT}"

    if http_client is not None:
        resp = http_client.get(target, headers=headers, timeout=timeout_seconds)
    else:
        with httpx.Client(timeout=timeout_seconds) as client:
            resp = client.get(target, headers=headers)
    if resp.is_error:
        detail = _http_error_detail(resp)
        logger.error(
            "Orchestrator agent-and-event request failed (%s): %s",
            resp.status_code,
            detail,
        )
        raise OrchestratorRequestError(
            f"orchestrator agent-and-event request failed ({resp.status_code}): {detail}"
        )

    payload = resp.json()
    if not isinstance(payload, dict):
        raise RuntimeError("orchestrator assignment response must be a JSON object")
    try:
        response = AgentAndEventResponse.model_validate(payload)
    except ValidationError as exc:
        raise RuntimeError(f"invalid orchestrator assignment payload: {exc}") from exc

    if response.assignment is None:
        logger.info(
            "No agent-and-event assignments available to predict on (reason=%s).",
            response.reason or "none_available",
        )

    return response


def _abbreviated_prediction_submit_log(
    payload: dict[str, Any],
    *,
    assignment: OrchestratorAssignment | None = None,
    payload_bytes: int | None = None,
) -> dict[str, Any]:
    """Build a privacy-safe at-a-glance summary of a prediction submit payload.

    Omits probabilities, confidence, reasoning text, and other agent-private
    fields. Surfaces identifiers, public market context, and structural sizes.
    """
    prediction = payload.get("prediction") if isinstance(payload.get("prediction"), dict) else {}
    exec_meta = (
        prediction.get("executionMetadata")
        if isinstance(prediction.get("executionMetadata"), dict)
        else {}
    )
    market_snap = (
        prediction.get("marketSnapshot")
        if isinstance(prediction.get("marketSnapshot"), dict)
        else {}
    )
    reasoning_trace = (
        payload.get("reasoningTrace") if isinstance(payload.get("reasoningTrace"), dict) else {}
    )
    trace = (
        reasoning_trace.get("trace") if isinstance(reasoning_trace.get("trace"), dict) else {}
    )
    trace_summary = (
        reasoning_trace.get("traceSummary")
        if isinstance(reasoning_trace.get("traceSummary"), dict)
        else {}
    )
    evidence = (
        trace.get("evidenceDigest") if isinstance(trace.get("evidenceDigest"), dict) else {}
    )
    future_graph = (
        evidence.get("futureGraph") if isinstance(evidence.get("futureGraph"), dict) else {}
    )
    belief_path = future_graph.get("beliefPath")
    provider_calls = trace.get("providerCalls")
    steps = trace.get("steps")

    question = None
    miner_uid = None
    if assignment is not None:
        miner_uid = assignment.miner.minerUid
        question = (assignment.event.question or "").strip() or None
        if question and len(question) > 160:
            question = question[:157] + "..."

    return {
        "agentPredictionId": payload.get("agentPredictionId"),
        "minerUid": miner_uid,
        "question": question,
        "source": market_snap.get("source") or (assignment.event.source if assignment else None),
        "sourceMarketId": market_snap.get("sourceMarketId")
        or (
            (assignment.event.sourceMarketId or assignment.event.marketId)
            if assignment
            else None
        ),
        "prediction": "[REDACTED]",
        "predictionIsInvalid": exec_meta.get("predictionIsInvalid"),
        "confidenceOmitted": exec_meta.get("confidenceOmitted"),
        "latencyMs": exec_meta.get("latencyMs"),
        "providerCallCount": trace_summary.get("providerCallCount")
        if trace_summary.get("providerCallCount") is not None
        else (len(provider_calls) if isinstance(provider_calls, list) else None),
        "reasoningStepCount": trace_summary.get("reasoningStepCount")
        if trace_summary.get("reasoningStepCount") is not None
        else (len(steps) if isinstance(steps, list) else None),
        "beliefPathStepCount": len(belief_path) if isinstance(belief_path, list) else None,
        "evidenceDigestKeyCount": len(evidence) if evidence else 0,
        "payloadBytes": payload_bytes,
    }


def submit_validator_prediction(
    *,
    base_url: str,
    loaded_hotkey: LoadedKeypair,
    payload: dict[str, Any],
    assignment: OrchestratorAssignment | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    http_client: httpx.Client | None = None,
) -> SubmitPredictionResponse:
    """Submit prediction payload to ``POST /v1/validators/prediction``."""
    body_text = _canonical_json(payload)
    body_bytes = body_text.encode("utf-8")
    summary = _abbreviated_prediction_submit_log(
        payload,
        assignment=assignment,
        payload_bytes=len(body_bytes),
    )
    logger.info(
        "Submitting prediction %s",
        json.dumps(summary, ensure_ascii=False, separators=(",", ":")),
    )

    headers = build_assignment_auth_headers(
        loaded_hotkey=loaded_hotkey,
        method="POST",
        path_and_query=PREDICTION_ENDPOINT,
        body=body_bytes,
    )
    headers["content-type"] = "application/json"
    target = f"{base_url.rstrip('/')}{PREDICTION_ENDPOINT}"

    if http_client is not None:
        resp = http_client.post(target, headers=headers, content=body_text, timeout=timeout_seconds)
    else:
        with httpx.Client(timeout=timeout_seconds) as client:
            resp = client.post(target, headers=headers, content=body_text)
    if resp.is_error:
        raise RuntimeError(
            f"validator prediction submit failed ({resp.status_code}): {_http_error_detail(resp)}"
        )
    raw = resp.json()
    if not isinstance(raw, dict):
        raise RuntimeError("validator prediction submit response must be a JSON object")
    try:
        return SubmitPredictionResponse.model_validate(raw)
    except ValidationError as exc:
        raise RuntimeError(f"invalid validator prediction submit response: {exc}") from exc


def _http_error_detail(resp: httpx.Response) -> str:
    text = (resp.text or "")[:4000]
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            if "detail" in data:
                return str(data["detail"])
            if "message" in data:
                return str(data["message"])
            if "reason" in data:
                return str(data["reason"])
            return json.dumps(data, ensure_ascii=False)
    except json.JSONDecodeError:
        pass
    return text or "(empty body)"




def fetch_scored_predictions_page(
    *,
    base_url: str,
    loaded_hotkey: LoadedKeypair,
    days: int = 30,
    limit: int = 1000,
    cursor: str | None = None,
    include_trace_summary: bool = True,
    timeout_seconds: float | httpx.Timeout = DEFAULT_TIMEOUT_SECONDS,
    http_client: httpx.Client | None = None,
) -> ScoredPredictionsPage:
    """Fetch one page from ``GET /v1/validators/scored-predictions``."""
    query_items: list[tuple[str, str]] = [("days", str(days)), ("limit", str(limit))]
    if cursor:
        query_items.append(("cursor", cursor))
    if include_trace_summary:
        query_items.append(("includeTraceSummary", "1"))
    query_str = "&".join(f"{k}={v}" for k, v in query_items)
    path_and_query = (
        f"{SCORED_PREDICTIONS_ENDPOINT}?{query_str}"
        if query_str
        else SCORED_PREDICTIONS_ENDPOINT
    )
    headers = build_assignment_auth_headers(
        loaded_hotkey=loaded_hotkey,
        method="GET",
        path_and_query=path_and_query,
        body=b"",
    )
    target = f"{base_url.rstrip('/')}{SCORED_PREDICTIONS_ENDPOINT}"
    params = {k: v for k, v in query_items}

    if http_client is not None:
        resp = http_client.get(target, headers=headers, params=params, timeout=timeout_seconds)
    else:
        with httpx.Client(timeout=timeout_seconds) as client:
            resp = client.get(target, headers=headers, params=params)
    resp.raise_for_status()
    raw = resp.json()
    if not isinstance(raw, dict):
        raise RuntimeError("scored predictions page response must be a JSON object")
    try:
        return ScoredPredictionsPage.model_validate(raw)
    except ValidationError as exc:
        raise RuntimeError(f"invalid scored predictions page payload: {exc}") from exc


def _scored_predictions_error_is_retryable(exc: Exception) -> bool:
    """Timeouts, connection failures, 401, 429, and 5xx are worth another try."""
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status in (401, 429) or status >= 500
    return False


def _scored_predictions_error_label(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return type(exc).__name__


def _fetch_scored_predictions_page_retrying(
    *,
    base_url: str,
    loaded_hotkey: LoadedKeypair,
    days: int,
    limit: int,
    cursor: str | None,
    include_trace_summary: bool,
    timeout: httpx.Timeout,
    http_client: httpx.Client,
    page_index: int,
    max_attempts: int,
    retry_backoff_seconds: float,
) -> ScoredPredictionsPage:
    """Fetch one page, retrying the same cursor with a freshly signed request."""
    for attempt in range(1, max_attempts + 1):
        started = time.perf_counter()
        try:
            page = fetch_scored_predictions_page(
                base_url=base_url,
                loaded_hotkey=loaded_hotkey,
                days=days,
                limit=limit,
                cursor=cursor,
                include_trace_summary=include_trace_summary,
                timeout_seconds=timeout,
                http_client=http_client,
            )
        except Exception as exc:
            if attempt >= max_attempts or not _scored_predictions_error_is_retryable(exc):
                raise
            logger.warning(
                "scored predictions page %d attempt %d/%d failed (%s); retrying same cursor",
                page_index,
                attempt,
                max_attempts,
                _scored_predictions_error_label(exc),
            )
            if retry_backoff_seconds > 0:
                time.sleep(retry_backoff_seconds)
            continue
        elapsed = time.perf_counter() - started
        logger.info(
            "scored predictions page %d: %d rows in %.2fs",
            page_index,
            len(page.items),
            elapsed,
        )
        return page
    raise RuntimeError("scored predictions retry loop exited without a page")


def fetch_all_scored_predictions(
    *,
    base_url: str,
    loaded_hotkey: LoadedKeypair,
    days: int = 30,
    page_limit: int = SCORED_PREDICTIONS_PAGE_LIMIT,
    include_trace_summary: bool = True,
    timeout_seconds: float = SCORED_PREDICTIONS_TIMEOUT_SECONDS,
    http_client: httpx.Client | None = None,
    max_attempts: int = SCORED_PREDICTIONS_MAX_ATTEMPTS,
    retry_backoff_seconds: float = 1.0,
) -> list[ScoredPredictionItem]:
    """Fetch all pages from ``GET /v1/validators/scored-predictions``.

    Pages share one HTTP client. A failed page is retried with a new signature
    and the same cursor. If that page still fails, this raises and the caller
    scores none of the pages already read.

    ``timeout_seconds`` bounds reads, writes, and pool waits. Connecting is
    capped at ``SCORED_PREDICTIONS_CONNECT_TIMEOUT_SECONDS``.
    """
    timeout = httpx.Timeout(
        timeout_seconds,
        connect=min(SCORED_PREDICTIONS_CONNECT_TIMEOUT_SECONDS, timeout_seconds),
    )
    owns_client = http_client is None
    client = http_client or httpx.Client(timeout=timeout)
    all_items: list[ScoredPredictionItem] = []
    cursor: str | None = None
    page_index = 0
    try:
        while True:
            page_index += 1
            page = _fetch_scored_predictions_page_retrying(
                base_url=base_url,
                loaded_hotkey=loaded_hotkey,
                days=days,
                limit=page_limit,
                cursor=cursor,
                include_trace_summary=include_trace_summary,
                timeout=timeout,
                http_client=client,
                page_index=page_index,
                max_attempts=max_attempts,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            all_items.extend(page.items)
            if not page.nextCursor:
                break
            cursor = page.nextCursor
        logger.info(
            "scored predictions fetch complete: %d rows across %d pages",
            len(all_items),
            page_index,
        )
        return all_items
    finally:
        if owns_client:
            client.close()
