"""Gateway-as-a-service base URL.

Validators forward every provider call to this HTTP API. Update
``DEFAULT_GATEWAY_SERVICE_URL`` for the repo-wide local default, or set the
``GATEWAY_SERVICE_URL`` environment variable in deployment (e.g. project ``.env``).

On import, if ``<project_root>/.env`` exists, it is loaded via ``python-dotenv`` with
``override=False`` so a shell-set ``GATEWAY_SERVICE_URL`` still wins.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from src.core.constants import GATEWAY_SERVICE_URL as CORE_GATEWAY_SERVICE_URL

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _PROJECT_ROOT / ".env"

if _ENV_FILE.is_file():
    load_dotenv(_ENV_FILE, override=False)

_ENV_KEY = "GATEWAY_SERVICE_URL"

# Default when ``GATEWAY_SERVICE_URL`` is unset (local gateway).
DEFAULT_GATEWAY_SERVICE_URL = CORE_GATEWAY_SERVICE_URL


def gateway_service_url() -> str:
    """Return the gateway service base URL with no trailing slash."""
    raw = os.environ.get(_ENV_KEY, DEFAULT_GATEWAY_SERVICE_URL).strip()
    if not raw:
        raw = DEFAULT_GATEWAY_SERVICE_URL
    return raw.rstrip("/")
