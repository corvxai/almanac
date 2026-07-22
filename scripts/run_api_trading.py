#!/usr/bin/env python3
"""Thin CLI wrapper for the Almanac API trading client.

This wrapper adds the repo root to ``sys.path`` so the trading script can be
invoked from anywhere via:

    python scripts/run_api_trading.py

See ``miner/market/api_trading.py`` for the interactive menu and credential setup.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


if __name__ == "__main__":
    runpy.run_path(
        str(PROJECT_ROOT / "miner" / "market" / "api_trading.py"),
        run_name="__main__",
    )
