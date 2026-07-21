#!/usr/bin/env python3
"""Thin CLI wrapper for the Almanac miner.

This wrapper just adds the repo root to ``sys.path`` so the miner script can be invoked from anywhere via:

    python scripts/run_market_miner.py [--polymarket-id ...] [--non-interactive]

See ``miner/market/miner.py`` ``--help`` for the full argument list.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


if __name__ == "__main__":
    runpy.run_path(
        str(PROJECT_ROOT / "miner" / "almanac" / "miner.py"),
        run_name="__main__",
    )
