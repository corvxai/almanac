#!/usr/bin/env python3
"""Sync `config/pricing_cards.json` with Claude models seen in code.

This script scans Python files for model literals like:
- "claude-sonnet-4-6"
- "anthropic/claude-haiku-4-5"

For each discovered Claude model, it ensures a `claude/<model>` card exists.
Missing cards inherit from family defaults:
- `claude/__default_opus__`
- `claude/__default_sonnet__`
- `claude/__default_haiku__`
- fallback: `claude/__default__`
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRICING_FILE = PROJECT_ROOT / "config" / "pricing_cards.json"
SCAN_DIRS = (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts")

_CLAUDE_MODEL_RE = re.compile(r"(?:anthropic/)?(claude-[a-z0-9][a-z0-9.\-]*)", re.IGNORECASE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync pricing cards with discovered Claude models.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write updates back to config/pricing_cards.json",
    )
    args = parser.parse_args()

    cards = _load_cards(PRICING_FILE)
    discovered = sorted(_discover_claude_models())

    missing_keys = []
    for model in discovered:
        key = f"claude/{model}"
        if key in cards:
            continue
        template = _template_for_model(cards, model)
        if template is None:
            print(f"SKIP {key}: no default template found")
            continue
        cards[key] = template
        missing_keys.append(key)

    if not missing_keys:
        print("No missing Claude pricing cards found.")
        return

    print("Added pricing cards:")
    for key in missing_keys:
        print(f"  - {key}")

    if args.write:
        _write_cards(PRICING_FILE, cards)
        print(f"\nUpdated {PRICING_FILE}")
    else:
        print("\nDry run only. Re-run with --write to persist changes.")


def _discover_claude_models() -> set[str]:
    models: set[str] = set()
    for base in SCAN_DIRS:
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for match in _CLAUDE_MODEL_RE.findall(text):
                models.add(match.lower())
    return models


def _template_for_model(cards: dict[str, Any], model: str) -> dict[str, Any] | None:
    m = model.lower()
    if "opus" in m:
        key = "claude/__default_opus__"
    elif "sonnet" in m:
        key = "claude/__default_sonnet__"
    elif "haiku" in m:
        key = "claude/__default_haiku__"
    else:
        key = "claude/__default__"

    tmpl = cards.get(key) or cards.get("claude/__default__")
    return dict(tmpl) if isinstance(tmpl, dict) else None


def _load_cards(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a top-level JSON object")
    return data


def _write_cards(path: Path, cards: dict[str, Any]) -> None:
    ordered = {k: cards[k] for k in sorted(cards)}
    path.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
