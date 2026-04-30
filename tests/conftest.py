"""Pytest configuration — import paths are set via `pythonpath` in pyproject.toml."""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv

# Pytest does not read `.env` on its own; match `scripts/run_gateway.py` so live tests see API keys.
_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env", override=False)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.live (real network / API calls).",
    )
    parser.addoption(
        "--live-only",
        action="store_true",
        default=False,
        help=(
            "Only collect tests marked @pytest.mark.live (implies they are not skipped). "
            "Use with --provider to run a single provider's network smoke without golden/mock tests."
        ),
    )
    parser.addoption(
        "--provider",
        action="store",
        default=None,
        metavar="ID",
        help=(
            "Only run tests marked @pytest.mark.provider for this gateway id "
            "(e.g. openrouter, claude, polymarket). Case-insensitive."
        ),
    )
    parser.addoption(
        "--pretty-print",
        action="store_true",
        default=False,
        help=(
            "After each matching provider test, pretty-print the raw response dict to stdout "
            "(use with -s). Combine with --provider and usually --live for real API shapes."
        ),
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    live = config.getoption("--live")
    live_only = config.getoption("--live-only")
    effective_live = bool(live or live_only)

    want = (config.getoption("--provider") or "").strip().lower()
    if want:
        kept: list[pytest.Item] = []
        for item in items:
            mark = item.get_closest_marker("provider")
            if mark is None or not mark.args:
                continue
            if str(mark.args[0]).strip().lower() != want:
                continue
            kept.append(item)
        items[:] = kept

    if live_only:
        items[:] = [it for it in items if "live" in it.keywords]

    if not effective_live:
        skip = pytest.mark.skip(reason="pass --live to run tests that call external APIs")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip)


@pytest.fixture
def live_enabled(request: pytest.FixtureRequest) -> bool:
    """True when the user passed ``pytest --live``."""
    return bool(request.config.getoption("--live"))
