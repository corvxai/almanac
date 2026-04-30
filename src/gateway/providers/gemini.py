"""Gemini (Google) provider — mock implementation.

When grounding with Google Search is enabled, the Gemini API returns
groundingMetadata in the response candidate. This includes:
- webSearchQueries: array of queries the model executed
- groundingChunks: array of {web: {uri, title}} source references
- groundingSupports: links between text segments and specific chunks

Visibility tier: HIGH with Google Search grounding — we get explicit
search queries, source URIs with titles, and text-to-source mappings.
Without grounding: text only.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any

from src.core.schemas import ProviderTier
from src.gateway.providers.base import BaseProvider

_MODELS = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"]

_MOCK_GROUNDING_CHUNKS = [
    {"web": {"uri": "https://reuters.com/markets/us-fed-rate", "title": "Reuters: US Fed Rate"}},
    {"web": {"uri": "https://en.wikipedia.org/wiki/Federal_Reserve", "title": "Wikipedia: Federal Reserve"}},
    {"web": {"uri": "https://polymarket.com/event/fed-rate", "title": "Polymarket: Fed Rate"}},
    {"web": {"uri": "https://bbc.co.uk/news/business-economy", "title": "BBC Business"}},
    {"web": {"uri": "https://ft.com/global-economy", "title": "FT: Global Economy"}},
]


class GeminiProvider(BaseProvider):
    provider_id: str = "gemini"
    provider_tier: ProviderTier = ProviderTier.INFERENCE

    def call(self, call_type: str, params: dict[str, Any]) -> dict[str, Any]:
        if call_type != "generate_content":
            raise ValueError(f"Unknown call_type for gemini: {call_type}")
        return self._mock_generate_content(params)

    def summarise_params(self, call_type: str, params: dict[str, Any]) -> str:
        model = params.get("model", _MODELS[0])
        grounding = params.get("grounding", True)
        return f'gemini.{call_type}(model={model}, grounding={"on" if grounding else "off"})'

    def _mock_generate_content(self, params: dict[str, Any]) -> dict[str, Any]:
        model = params.get("model", _MODELS[0])
        grounding_enabled = params.get("grounding", True)

        prediction = round(random.uniform(0.15, 0.85), 4)
        reasoning = (
            f"Analysis of current data suggests a {prediction:.1%} probability. "
            f"This reflects recent economic trends and policy signals."
        )
        text = f"PREDICTION: {prediction}\nREASONING: {reasoning}"

        candidate: dict[str, Any] = {
            "content": {
                "parts": [{"text": text}],
                "role": "model",
            },
            "finishReason": "STOP",
        }

        if grounding_enabled:
            num_chunks = random.randint(2, 5)
            chunks = random.sample(_MOCK_GROUNDING_CHUNKS, num_chunks)
            queries = [
                f"search query about {params.get('contents', 'the event')}"[:80]
            ]
            supports = [{
                "segment": {"startIndex": 0, "endIndex": len(text), "text": text[:100]},
                "groundingChunkIndices": list(range(num_chunks)),
                "confidenceScores": [round(random.uniform(0.7, 0.99), 2) for _ in range(num_chunks)],
            }]

            candidate["groundingMetadata"] = {
                "webSearchQueries": queries,
                "groundingChunks": chunks,
                "groundingSupports": supports,
                "searchEntryPoint": {"renderedContent": "<search widget html>"},
            }

        return {
            "candidates": [candidate],
            "modelVersion": model,
            "usageMetadata": {
                "promptTokenCount": random.randint(200, 600),
                "candidatesTokenCount": random.randint(150, 400),
                "totalTokenCount": random.randint(400, 1000),
            },
        }
