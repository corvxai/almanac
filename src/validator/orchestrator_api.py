"""Validator client for orchestrator agent/event assignment polling."""

from __future__ import annotations

import json
import hashlib
import secrets
import time
from datetime import datetime
from typing import Any

import httpx
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError

from src.gateway.signing import LoadedKeypair

AGENT_AND_EVENT_ENDPOINT = "/v1/validators/agent-and-event"
PREDICTION_ENDPOINT = "/v1/validators/prediction"
DEFAULT_TIMEOUT_SECONDS = 15.0
AUTH_DOMAIN = "sub41-gateway-v1"


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
    resp.raise_for_status()

    payload = resp.json()
    if not isinstance(payload, dict):
        raise RuntimeError("orchestrator assignment response must be a JSON object")
    try:
        return AgentAndEventResponse.model_validate(payload)
    except ValidationError as exc:
        raise RuntimeError(f"invalid orchestrator assignment payload: {exc}") from exc


def submit_validator_prediction(
    *,
    base_url: str,
    loaded_hotkey: LoadedKeypair,
    payload: dict[str, Any],
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    http_client: httpx.Client | None = None,
) -> SubmitPredictionResponse:
    """Submit prediction payload to ``POST /v1/validators/prediction``."""
    body_text = _canonical_json(payload)
    body_bytes = body_text.encode("utf-8")
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
    resp.raise_for_status()
    raw = resp.json()
    if not isinstance(raw, dict):
        raise RuntimeError("validator prediction submit response must be a JSON object")
    try:
        return SubmitPredictionResponse.model_validate(raw)
    except ValidationError as exc:
        raise RuntimeError(f"invalid validator prediction submit response: {exc}") from exc


def _canonical_json(value: Any) -> str:
    """Match the API's canonical JSON serialization for signing."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ",".join(_canonical_json(v) for v in value) + "]"
    if isinstance(value, dict):
        parts = []
        for key in sorted(value.keys(), key=lambda k: str(k)):
            parts.append(f"{json.dumps(str(key), ensure_ascii=False)}:{_canonical_json(value[key])}")
        return "{" + ",".join(parts) + "}"
    raise TypeError(f"unsupported value for canonical JSON: {type(value).__name__}")
