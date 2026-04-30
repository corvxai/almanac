"""Gateway test fixtures (see root ``tests/conftest.py`` for CLI flags)."""

from __future__ import annotations

import json
from typing import Any

import pytest


@pytest.fixture
def maybe_pretty_print_raw(request: pytest.FixtureRequest):
    """When ``pytest --pretty-print`` is set, print a raw provider ``dict`` (use ``-s``)."""

    def _emit(raw: dict[str, Any], *, banner: str = "") -> None:
        if not request.config.getoption("--pretty-print"):
            return
        if banner:
            print(f"\n{'=' * 72}\n{banner}\n{'=' * 72}\n", flush=True)
        print(json.dumps(raw, indent=2, ensure_ascii=False, default=str), flush=True)

    return _emit
