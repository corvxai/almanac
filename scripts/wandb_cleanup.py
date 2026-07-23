"""Delete W&B validator runs older than the retention window."""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import wandb  # type: ignore[import-not-found]
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.constants import WANDB_ENTITY, WANDB_PROJECT  # noqa: E402


DAYS_TO_KEEP = 2


def _created_at(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def delete_old_runs(api) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(days=DAYS_TO_KEEP)
    for run in api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}"):
        if _created_at(run.created_at) >= cutoff:
            continue

        for file in run.files():
            try:
                print(f"Deleting file {file.name} in run {run.id}")
                file.delete()
            except Exception as exc:
                print(f"Could not delete file {file.name}: {exc}")

        for artifact in run.logged_artifacts():
            try:
                print(f"Deleting artifact {artifact.id} in run {run.id}")
                artifact.delete()
            except wandb.errors.CommError as exc:
                if "system managed artifact" in str(exc):
                    print(f"Skipping system-managed artifact {artifact.id}")
                else:
                    print(f"Could not delete artifact {artifact.id}: {exc}")
            except Exception as exc:
                print(f"Could not delete artifact {artifact.id}: {exc}")

        try:
            print(f"Deleting run {run.id} created at {run.created_at}")
            run.delete()
        except Exception as exc:
            print(f"Could not delete run {run.id}: {exc}")

    print("Old W&B runs cleanup completed.")


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        raise RuntimeError(f"WANDB_API_KEY is missing from {PROJECT_ROOT / '.env'}")

    wandb.login(key=api_key)
    api = wandb.Api()

    while True:
        delete_old_runs(api)
        time.sleep(60 * 60)


if __name__ == "__main__":
    main()
