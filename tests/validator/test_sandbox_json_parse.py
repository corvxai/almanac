"""Unit tests for `_last_json_object` — the stdout result extractor.

The runner muxes its AgentResult JSON into stdout alongside any debug prints
from the agent or curated deps. The extractor must return the trailing balanced
JSON object, and must not be fooled by braces inside string values.
"""

from __future__ import annotations

import json

from src.validator import sandbox_docker as sd


def test_plain_object_roundtrips() -> None:
    obj = '{"prediction": 0.5, "reasoning": "ok"}'
    assert sd._last_json_object(obj) == obj


def test_ignores_leading_stdout_noise() -> None:
    text = 'INFO connecting...\nhttpx debug line\n{"prediction": 0.5}'
    assert sd._last_json_object(text) == '{"prediction": 0.5}'


def test_braces_inside_string_value_do_not_break_parsing() -> None:
    """Regression: a reasoning field containing braces must not be miscounted."""
    text = 'log {leftover}\n{"prediction": 0.5, "reasoning": "use {curly} and } braces"}'
    extracted = sd._last_json_object(text)
    # The extracted slice must be valid JSON with the full reasoning intact.
    parsed = json.loads(extracted)
    assert parsed["reasoning"] == "use {curly} and } braces"
    assert parsed["prediction"] == 0.5


def test_escaped_quote_inside_string_is_handled() -> None:
    text = r'{"reasoning": "she said \"hi\" and { left a brace"}'
    parsed = json.loads(sd._last_json_object(text))
    assert parsed["reasoning"] == 'she said "hi" and { left a brace'


def test_returns_last_of_multiple_objects() -> None:
    text = '{"first": 1}\nnoise\n{"second": 2}'
    assert json.loads(sd._last_json_object(text)) == {"second": 2}


def test_nested_object_preserved() -> None:
    text = 'prefix {"a": {"b": {"c": 1}}, "s": "}}}"}'
    parsed = json.loads(sd._last_json_object(text))
    assert parsed["a"]["b"]["c"] == 1
    assert parsed["s"] == "}}}"
