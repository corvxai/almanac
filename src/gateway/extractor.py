"""Standardised evidence extraction pipeline.

Runs on raw provider responses to produce structured ExtractedEvidence items.
Each provider has its own extraction logic tuned to the metadata that
provider actually returns (citations, search results, grounding chunks, etc.).

This is a rule-based implementation for the prototype — will get more
sophisticated (NLP, LLM-assisted) in later phases.
"""

from __future__ import annotations

import re
from typing import Any

from src.core.schemas import EvidenceType, ExtractedEvidence, ExtractionMethod


def extract_evidence(provider_id: str, call_type: str, raw: dict[str, Any]) -> list[ExtractedEvidence]:
    """Dispatch to provider-specific extraction logic."""
    extractors = {
        "polymarket": _extract_polymarket,
        "web_search": _extract_web_search,
        "grok": _extract_grok,
        "perplexity": _extract_perplexity,
        "anthropic": _extract_anthropic,
        "openai": _extract_openai,
        "gemini": _extract_gemini,
        "openrouter": _extract_openrouter,
    }
    fn = extractors.get(provider_id, _extract_generic)
    return fn(call_type, raw)


# ---------------------------------------------------------------------------
# Data providers
# ---------------------------------------------------------------------------

def _extract_polymarket(call_type: str, raw: dict[str, Any]) -> list[ExtractedEvidence]:
    items: list[ExtractedEvidence] = []

    if "price" in raw:
        items.append(ExtractedEvidence(
            evidence_type=EvidenceType.PRICE,
            content=f"Current market price: {raw['price']}",
            extraction_method=ExtractionMethod.DIRECT_VALUE,
            numeric_value=float(raw["price"]),
        ))

    if "volume_24h" in raw:
        items.append(ExtractedEvidence(
            evidence_type=EvidenceType.STATISTIC,
            content=f"24h trading volume: {raw['volume_24h']}",
            extraction_method=ExtractionMethod.DIRECT_VALUE,
            numeric_value=float(raw["volume_24h"]),
        ))

    if "price_history" in raw and isinstance(raw["price_history"], list):
        prices = [p["price"] for p in raw["price_history"] if "price" in p]
        if len(prices) >= 2:
            direction = "up" if prices[-1] > prices[0] else "down"
            delta = prices[-1] - prices[0]
            items.append(ExtractedEvidence(
                evidence_type=EvidenceType.TREND,
                content=f"Price trend: {direction} by {abs(delta):.4f} over {len(prices)} data points",
                extraction_method=ExtractionMethod.STATISTICAL_SUMMARY,
                numeric_value=delta,
            ))

    if "probability" in raw:
        items.append(ExtractedEvidence(
            evidence_type=EvidenceType.PROBABILITY,
            content=f"Implied probability: {raw['probability']}",
            extraction_method=ExtractionMethod.DIRECT_VALUE,
            numeric_value=float(raw["probability"]),
        ))

    return items


def _extract_web_search(call_type: str, raw: dict[str, Any]) -> list[ExtractedEvidence]:
    items: list[ExtractedEvidence] = []

    for r in raw.get("results", []):
        snippet = r.get("snippet", r.get("title", ""))
        if snippet:
            items.append(ExtractedEvidence(
                evidence_type=EvidenceType.QUOTE_SUMMARY,
                content=snippet,
                extraction_method=ExtractionMethod.NLP_EXTRACTION,
            ))

    if "result_count" in raw:
        items.append(ExtractedEvidence(
            evidence_type=EvidenceType.STATISTIC,
            content=f"Total search results: {raw['result_count']}",
            extraction_method=ExtractionMethod.DIRECT_VALUE,
            numeric_value=float(raw["result_count"]),
        ))

    return items


# ---------------------------------------------------------------------------
# LLM providers — extraction varies by what metadata each API returns
# ---------------------------------------------------------------------------

def _parse_prediction_reasoning(
    text: str,
) -> tuple[float | None, float | None, str | None]:
    """Parse PREDICTION: / CONVICTION: / REASONING: from LLM completion text."""
    prediction = None
    confidence = None
    reasoning = None
    for line in text.strip().split("\n"):
        cleaned = line.strip().strip("*`_ ").strip()
        if cleaned.startswith("PREDICTION:"):
            try:
                prediction = float(cleaned.replace("PREDICTION:", "").strip())
            except ValueError:
                pass
        elif cleaned.startswith("CONVICTION:"):
            try:
                confidence = float(cleaned.replace("CONVICTION:", "").strip())
            except ValueError:
                pass
        elif cleaned.startswith("REASONING:"):
            reasoning = cleaned.replace("REASONING:", "").strip()

    # Tolerant fallback for markdown/decorated outputs.
    if prediction is None:
        match = re.search(r"(?im)\bPREDICTION\b\s*:\s*([0-9]*\.?[0-9]+)", text)
        if match:
            try:
                prediction = float(match.group(1))
            except ValueError:
                pass
    if confidence is None:
        match = re.search(r"(?im)\b(?:CONVICTION|CONFIDENCE)\b\s*:\s*([0-9]*\.?[0-9]+)", text)
        if match:
            try:
                confidence = float(match.group(1))
            except ValueError:
                pass
    if reasoning is None:
        match = re.search(r"(?is)\bREASONING\b\s*:\s*(.+)$", text)
        if match:
            reasoning = match.group(1).strip()
    return prediction, confidence, reasoning


def _extract_llm_completion_text(raw: dict[str, Any]) -> str:
    """Get the completion text from various LLM response formats."""
    # OpenAI/Grok/Perplexity chat completions format
    choices = raw.get("choices", [])
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message", {})
            if isinstance(message, dict):
                content = message.get("content", "")
                if isinstance(content, str):
                    return content

    # OpenRouter may return plain text in top-level output.
    output = raw.get("output")
    if isinstance(output, str):
        return output

    # OpenAI Responses API format
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                content_blocks = item.get("content", [])
                if not isinstance(content_blocks, list):
                    continue
                for block in content_blocks:
                    if isinstance(block, dict) and block.get("type") == "output_text":
                        text = block.get("text", "")
                        if isinstance(text, str):
                            return text

    # Claude Messages API format
    content = raw.get("content", [])
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if isinstance(text, str):
                    return text

    # Gemini format
    candidates = raw.get("candidates", [])
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_content = candidate.get("content", {})
            if not isinstance(candidate_content, dict):
                continue
            parts = candidate_content.get("parts", [])
            if isinstance(parts, list) and parts and isinstance(parts[0], dict):
                text = parts[0].get("text", "")
                if isinstance(text, str):
                    return text

    return ""


def _extract_grok(call_type: str, raw: dict[str, Any]) -> list[ExtractedEvidence]:
    """Grok: citations[] URLs + usage.num_sources_used when search is on."""
    items: list[ExtractedEvidence] = []
    text = _extract_llm_completion_text(raw)

    pred, confidence, reasoning = _parse_prediction_reasoning(text)
    if pred is not None:
        items.append(ExtractedEvidence(
            evidence_type=EvidenceType.PROBABILITY,
            content=f"LLM prediction: {pred}",
            extraction_method=ExtractionMethod.NLP_EXTRACTION,
            numeric_value=pred,
        ))
    if confidence is not None:
        items.append(ExtractedEvidence(
            evidence_type=EvidenceType.PROBABILITY,
            content=f"LLM conviction: {confidence}",
            extraction_method=ExtractionMethod.NLP_EXTRACTION,
            numeric_value=confidence,
        ))
    if reasoning:
        items.append(ExtractedEvidence(
            evidence_type=EvidenceType.QUOTE_SUMMARY,
            content=f"LLM reasoning: {reasoning}",
            extraction_method=ExtractionMethod.NLP_EXTRACTION,
        ))

    # Grok returns citations as a top-level array of URLs when search is enabled
    for url in raw.get("citations", []):
        items.append(ExtractedEvidence(
            evidence_type=EvidenceType.FACT,
            content=f"Source cited by Grok: {url}",
            extraction_method=ExtractionMethod.SCHEMA_MAPPING,
        ))

    # Reasoning trace if available
    rc = raw.get("choices", [{}])[0].get("message", {}).get("reasoning_content")
    if rc:
        items.append(ExtractedEvidence(
            evidence_type=EvidenceType.QUOTE_SUMMARY,
            content=f"Grok reasoning trace: {rc[:500]}",
            extraction_method=ExtractionMethod.DIRECT_VALUE,
        ))

    return items


def _extract_perplexity(call_type: str, raw: dict[str, Any]) -> list[ExtractedEvidence]:
    """Perplexity: citations[] URLs + search_results[] with title/url/snippet."""
    items: list[ExtractedEvidence] = []
    text = _extract_llm_completion_text(raw)

    pred, confidence, reasoning = _parse_prediction_reasoning(text)
    if pred is not None:
        items.append(ExtractedEvidence(
            evidence_type=EvidenceType.PROBABILITY,
            content=f"LLM prediction: {pred}",
            extraction_method=ExtractionMethod.NLP_EXTRACTION,
            numeric_value=pred,
        ))
    if confidence is not None:
        items.append(ExtractedEvidence(
            evidence_type=EvidenceType.PROBABILITY,
            content=f"LLM conviction: {confidence}",
            extraction_method=ExtractionMethod.NLP_EXTRACTION,
            numeric_value=confidence,
        ))
    if reasoning:
        items.append(ExtractedEvidence(
            evidence_type=EvidenceType.QUOTE_SUMMARY,
            content=f"LLM reasoning: {reasoning}",
            extraction_method=ExtractionMethod.NLP_EXTRACTION,
        ))

    # Perplexity returns full search results with snippets — best transparency
    for sr in raw.get("search_results", []):
        title = sr.get("title", "")
        url = sr.get("url", "")
        snippet = sr.get("snippet", "")
        items.append(ExtractedEvidence(
            evidence_type=EvidenceType.QUOTE_SUMMARY,
            content=f"[{title}] ({url}): {snippet}" if snippet else f"[{title}] ({url})",
            extraction_method=ExtractionMethod.SCHEMA_MAPPING,
        ))

    # Fall back to citation URLs if no search_results
    if not raw.get("search_results"):
        for url in raw.get("citations", []):
            items.append(ExtractedEvidence(
                evidence_type=EvidenceType.FACT,
                content=f"Source cited by Perplexity: {url}",
                extraction_method=ExtractionMethod.SCHEMA_MAPPING,
            ))

    return items


def _extract_anthropic(call_type: str, raw: dict[str, Any]) -> list[ExtractedEvidence]:
    """Anthropic: web_search_tool_result blocks with URLs + text citations."""
    items: list[ExtractedEvidence] = []
    text = _extract_llm_completion_text(raw)

    pred, confidence, reasoning = _parse_prediction_reasoning(text)
    if pred is not None:
        items.append(ExtractedEvidence(
            evidence_type=EvidenceType.PROBABILITY,
            content=f"LLM prediction: {pred}",
            extraction_method=ExtractionMethod.NLP_EXTRACTION,
            numeric_value=pred,
        ))
    if confidence is not None:
        items.append(ExtractedEvidence(
            evidence_type=EvidenceType.PROBABILITY,
            content=f"LLM conviction: {confidence}",
            extraction_method=ExtractionMethod.NLP_EXTRACTION,
            numeric_value=confidence,
        ))
    if reasoning:
        items.append(ExtractedEvidence(
            evidence_type=EvidenceType.QUOTE_SUMMARY,
            content=f"LLM reasoning: {reasoning}",
            extraction_method=ExtractionMethod.NLP_EXTRACTION,
        ))

    # Extract sources from web_search_tool_result content blocks
    for block in raw.get("content", []):
        if block.get("type") == "web_search_tool_result":
            for result in block.get("content", []):
                if isinstance(result, dict) and result.get("type") == "web_search_result":
                    url = result.get("url", "")
                    title = result.get("title", "")
                    items.append(ExtractedEvidence(
                        evidence_type=EvidenceType.FACT,
                        content=f"Claude web search: [{title}] ({url})",
                        extraction_method=ExtractionMethod.SCHEMA_MAPPING,
                    ))

        # Extract inline citations from text blocks
        if block.get("type") == "text":
            for cit in block.get("citations", []):
                if cit.get("type") == "web_search_result_location":
                    url = cit.get("url", "")
                    cited_text = cit.get("cited_text", "")
                    items.append(ExtractedEvidence(
                        evidence_type=EvidenceType.QUOTE_SUMMARY,
                        content=f"Citation: {cited_text} — {url}",
                        extraction_method=ExtractionMethod.SCHEMA_MAPPING,
                    ))

    return items


def _extract_openai(call_type: str, raw: dict[str, Any]) -> list[ExtractedEvidence]:
    """OpenAI: annotations[] with url_citations in Responses API, plain text in Chat."""
    items: list[ExtractedEvidence] = []
    text = _extract_llm_completion_text(raw)

    pred, confidence, reasoning = _parse_prediction_reasoning(text)
    if pred is not None:
        items.append(ExtractedEvidence(
            evidence_type=EvidenceType.PROBABILITY,
            content=f"LLM prediction: {pred}",
            extraction_method=ExtractionMethod.NLP_EXTRACTION,
            numeric_value=pred,
        ))
    if confidence is not None:
        items.append(ExtractedEvidence(
            evidence_type=EvidenceType.PROBABILITY,
            content=f"LLM conviction: {confidence}",
            extraction_method=ExtractionMethod.NLP_EXTRACTION,
            numeric_value=confidence,
        ))
    if reasoning:
        items.append(ExtractedEvidence(
            evidence_type=EvidenceType.QUOTE_SUMMARY,
            content=f"LLM reasoning: {reasoning}",
            extraction_method=ExtractionMethod.NLP_EXTRACTION,
        ))

    # Responses API: output[] → message → content[] → annotations[]
    for item in raw.get("output", []):
        if item.get("type") == "message":
            for block in item.get("content", []):
                for ann in block.get("annotations", []):
                    if ann.get("type") == "url_citation":
                        url = ann.get("url", "")
                        title = ann.get("title", "")
                        items.append(ExtractedEvidence(
                            evidence_type=EvidenceType.FACT,
                            content=f"OpenAI citation: [{title}] ({url})",
                            extraction_method=ExtractionMethod.SCHEMA_MAPPING,
                        ))

    return items


def _extract_gemini(call_type: str, raw: dict[str, Any]) -> list[ExtractedEvidence]:
    """Gemini: groundingMetadata with webSearchQueries, groundingChunks, groundingSupports."""
    items: list[ExtractedEvidence] = []
    text = _extract_llm_completion_text(raw)

    pred, confidence, reasoning = _parse_prediction_reasoning(text)
    if pred is not None:
        items.append(ExtractedEvidence(
            evidence_type=EvidenceType.PROBABILITY,
            content=f"LLM prediction: {pred}",
            extraction_method=ExtractionMethod.NLP_EXTRACTION,
            numeric_value=pred,
        ))
    if confidence is not None:
        items.append(ExtractedEvidence(
            evidence_type=EvidenceType.PROBABILITY,
            content=f"LLM conviction: {confidence}",
            extraction_method=ExtractionMethod.NLP_EXTRACTION,
            numeric_value=confidence,
        ))
    if reasoning:
        items.append(ExtractedEvidence(
            evidence_type=EvidenceType.QUOTE_SUMMARY,
            content=f"LLM reasoning: {reasoning}",
            extraction_method=ExtractionMethod.NLP_EXTRACTION,
        ))

    # Grounding metadata from candidates
    for candidate in raw.get("candidates", []):
        gm = candidate.get("groundingMetadata", {})

        # Search queries the model executed
        for query in gm.get("webSearchQueries", []):
            items.append(ExtractedEvidence(
                evidence_type=EvidenceType.FACT,
                content=f"Gemini search query: {query}",
                extraction_method=ExtractionMethod.SCHEMA_MAPPING,
            ))

        # Grounding chunks — the actual source references
        for chunk in gm.get("groundingChunks", []):
            web = chunk.get("web", {})
            uri = web.get("uri", "")
            title = web.get("title", "")
            if uri:
                items.append(ExtractedEvidence(
                    evidence_type=EvidenceType.FACT,
                    content=f"Gemini source: [{title}] ({uri})",
                    extraction_method=ExtractionMethod.SCHEMA_MAPPING,
                ))

    return items


def _extract_openrouter(call_type: str, raw: dict[str, Any]) -> list[ExtractedEvidence]:
    """OpenRouter: standard chat completion, no source metadata.

    Models often ignore strict ``PREDICTION:`` / ``REASONING:`` lines; when parsing finds
    nothing but there is still assistant text, we keep a single quote-style evidence item
    so the gateway trace is never empty for a successful completion.
    """
    items: list[ExtractedEvidence] = []
    text = _extract_llm_completion_text(raw)

    pred, confidence, reasoning = _parse_prediction_reasoning(text)
    if pred is not None:
        items.append(ExtractedEvidence(
            evidence_type=EvidenceType.PROBABILITY,
            content=f"LLM prediction: {pred}",
            extraction_method=ExtractionMethod.NLP_EXTRACTION,
            numeric_value=pred,
        ))
    if confidence is not None:
        items.append(ExtractedEvidence(
            evidence_type=EvidenceType.PROBABILITY,
            content=f"LLM conviction: {confidence}",
            extraction_method=ExtractionMethod.NLP_EXTRACTION,
            numeric_value=confidence,
        ))
    if reasoning:
        items.append(ExtractedEvidence(
            evidence_type=EvidenceType.QUOTE_SUMMARY,
            content=f"LLM reasoning: {reasoning}",
            extraction_method=ExtractionMethod.NLP_EXTRACTION,
        ))

    if not items and text.strip():
        clipped = text.strip()
        if len(clipped) > 4000:
            clipped = clipped[:3997] + "..."
        items.append(ExtractedEvidence(
            evidence_type=EvidenceType.QUOTE_SUMMARY,
            content=f"LLM completion: {clipped}",
            extraction_method=ExtractionMethod.NLP_EXTRACTION,
        ))

    return items


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

def _extract_generic(call_type: str, raw: dict[str, Any]) -> list[ExtractedEvidence]:
    """Fallback: pull any top-level numeric values as statistics."""
    items: list[ExtractedEvidence] = []
    for key, val in raw.items():
        if isinstance(val, (int, float)):
            items.append(ExtractedEvidence(
                evidence_type=EvidenceType.STATISTIC,
                content=f"{key}: {val}",
                extraction_method=ExtractionMethod.DIRECT_VALUE,
                numeric_value=float(val),
            ))
    return items
