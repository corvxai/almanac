#!/usr/bin/env python3
"""
preview_eligible_markets.py

Pulls political events from Polymarket's gamma API, runs them through the
eligibility filter from event_ingestion_spec.md, and prints what would be
admitted to the prediction pool right now.

Usage:
    pip install httpx pydantic
    python preview_eligible_markets.py

Tweak the constants in the CONFIG section to explore different thresholds.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, field_validator
from tabulate import tabulate


# ─────────────────────────────────────────────────────────────────────────
# CONFIG  (mirrors event_ingestion_spec.md §12)
# ─────────────────────────────────────────────────────────────────────────

POLYMARKET_GAMMA_BASE = "https://gamma-api.polymarket.com"
ENABLED_TAG_IDS = [2]  # Politics

# Horizon
MIN_HORIZON_HOURS = 24
MAX_HORIZON_DAYS = 30
HOT_BUCKET_DAYS = 7
WARM_BUCKET_DAYS = 14

# Eligibility floors
MIN_LIQUIDITY_USD = 1000
MIN_VOLUME_USD = 1000
MIN_RECENT_VOLUME_USD = 100        # 1-week alternative path
MIN_DESCRIPTION_CHARS = 200
MIN_AGE_HOURS = 24
MIN_YES_PRICE = 0.20
MAX_YES_PRICE = 0.80

# negRisk
NEGRISK_TOP_K = 3
NEGRISK_PLACEHOLDER_PATTERN = re.compile(r"^Candidate [A-Z]$")

# HTTP
PAGE_LIMIT = 500
MAX_PAGES = 20                     # safety cap on pagination
HTTP_TIMEOUT_SECONDS = 30

# Display
DISPLAY_TOP_N_PER_BUCKET = 10


# ─────────────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────────────

class RejectionReason(str, Enum):
    TAG_MISMATCH = "tag_mismatch"
    INACTIVE = "inactive"
    NOT_BINARY = "not_binary"
    MISSING_DATES = "missing_dates"
    HORIZON_TOO_SHORT = "horizon_too_short"
    HORIZON_TOO_LONG = "horizon_too_long"
    LIQUIDITY_TOO_LOW = "liquidity_too_low"
    VOLUME_TOO_LOW = "volume_too_low"
    DESCRIPTION_TOO_SHORT = "description_too_short"
    TOO_NEW = "too_new"
    PRICE_OUT_OF_RANGE = "price_out_of_range"
    NEGRISK_PLACEHOLDER = "negrisk_placeholder"
    NEGRISK_NOT_TOP_K = "negrisk_not_top_k"


class Bucket(str, Enum):
    HOT = "hot"
    WARM = "warm"
    COOL = "cool"


class Market(BaseModel):
    """Subset of Polymarket market fields we care about."""
    model_config = ConfigDict(extra="ignore")

    # Identity
    id: str
    conditionId: str = ""
    slug: str = ""
    question: str = ""
    description: str = ""

    # Outcomes (Polymarket returns these as JSON-encoded strings)
    outcomes: list[str] = []
    outcomePrices: list[float] | None = None

    # Dates
    endDate: datetime | None = None
    startDate: datetime | None = None
    umaEndDate: datetime | None = None

    # Liquidity / volume
    liquidityNum: float = 0.0
    volumeNum: float = 0.0
    volume1wk: float = 0.0

    # State flags (default to "rejected" values so missing fields fail safe)
    active: bool = False
    closed: bool = True
    archived: bool = True
    acceptingOrders: bool = False

    # negRisk group membership
    negRisk: bool = False
    negRiskMarketID: str | None = None
    groupItemTitle: str | None = None

    # Inherited from parent event (set during extraction, not in payload)
    event_id: str = ""
    event_tags: list[int] = []
    event_tag_slugs: list[str] = []
    event_slug: str = ""
    event_title: str = ""
    event_description: str = ""
    event_created_at: datetime | None = None

    @field_validator("outcomes", mode="before")
    @classmethod
    def _parse_outcomes(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return []
        return v

    @field_validator("outcomePrices", mode="before")
    @classmethod
    def _parse_prices(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return [float(x) for x in json.loads(v)]
            except (json.JSONDecodeError, ValueError, TypeError):
                return None
        return v


# ─────────────────────────────────────────────────────────────────────────
# FETCH
# ─────────────────────────────────────────────────────────────────────────

def fetch_events(tag_id: int, end_date_max: datetime) -> list[dict]:
    """Paginate /events/keyset and return all event dicts for one tag."""
    events: list[dict] = []
    cursor: str | None = None

    with httpx.Client(base_url=POLYMARKET_GAMMA_BASE, timeout=HTTP_TIMEOUT_SECONDS) as client:
        for _ in range(MAX_PAGES):
            params: dict[str, Any] = {
                "tag_id": tag_id,
                "limit": PAGE_LIMIT,
                "active": "true",
                "archived": "false",
                "closed": "false",
                "end_date_max": end_date_max.isoformat().replace("+00:00", "Z"),
                "order": "updatedAt",
                "ascending": "false",
            }
            if cursor:
                params["after_cursor"] = cursor

            resp = client.get("/events/keyset", params=params)
            resp.raise_for_status()
            data = resp.json()

            page_events = data.get("events") or []
            events.extend(page_events)

            cursor = data.get("next_cursor")
            if not cursor or not page_events:
                break

    return events


def extract_markets(events: list[dict]) -> list[Market]:
    """Flatten events → markets. Markets inherit event id and tag IDs."""
    markets: list[Market] = []
    for event in events:
        event_id = str(event.get("id", ""))
        event_slug = str(event.get("slug", ""))
        event_title = str(event.get("title", ""))
        event_description = str(event.get("description", ""))
        event_created_at_raw = event.get("creationDate") or event.get("createdAt")
        event_created_at: datetime | None = None
        if event_created_at_raw:
            try:
                event_created_at = datetime.fromisoformat(str(event_created_at_raw).replace("Z", "+00:00"))
            except ValueError:
                event_created_at = None
        event_tags: list[int] = []
        event_tag_slugs: list[str] = []
        for t in event.get("tags") or []:
            try:
                event_tags.append(int(t["id"]))
            except (KeyError, ValueError, TypeError):
                continue
            slug = t.get("slug")
            if isinstance(slug, str) and slug:
                event_tag_slugs.append(slug)

        for raw_market in event.get("markets") or []:
            try:
                m = Market.model_validate(raw_market)
            except Exception:
                continue  # skip malformed payloads
            m.event_id = event_id
            m.event_tags = event_tags
            m.event_tag_slugs = event_tag_slugs
            m.event_slug = event_slug
            m.event_title = event_title
            m.event_description = event_description
            m.event_created_at = event_created_at
            markets.append(m)
    return markets


# ─────────────────────────────────────────────────────────────────────────
# ELIGIBILITY
# ─────────────────────────────────────────────────────────────────────────

def evaluate_market(m: Market, now: datetime) -> RejectionReason | None:
    """Run per-market eligibility checks. Returns None if eligible."""
    # 1. Tag match (event-level, since markets don't always carry tags inline)
    if not any(t in ENABLED_TAG_IDS for t in m.event_tags):
        return RejectionReason.TAG_MISMATCH

    # 2. State flags
    if not (m.active and not m.closed and not m.archived and m.acceptingOrders):
        return RejectionReason.INACTIVE

    # 3. Outcome format
    if m.outcomes != ["Yes", "No"]:
        return RejectionReason.NOT_BINARY

    # 4. Dates present
    if m.endDate is None or m.startDate is None:
        return RejectionReason.MISSING_DATES

    # 5. Resolution horizon
    horizon = m.endDate - now
    if horizon < timedelta(hours=MIN_HORIZON_HOURS):
        return RejectionReason.HORIZON_TOO_SHORT
    if horizon > timedelta(days=MAX_HORIZON_DAYS):
        return RejectionReason.HORIZON_TOO_LONG

    # 6. Liquidity floor
    if m.liquidityNum < MIN_LIQUIDITY_USD:
        return RejectionReason.LIQUIDITY_TOO_LOW

    # 7. Volume floor (cumulative OR recent)
    if m.volumeNum < MIN_VOLUME_USD and m.volume1wk < MIN_RECENT_VOLUME_USD:
        return RejectionReason.VOLUME_TOO_LOW

    # 8. Description quality
    if len(m.description) < MIN_DESCRIPTION_CHARS:
        return RejectionReason.DESCRIPTION_TOO_SHORT

    # 9. Maturity
    if (now - m.startDate) < timedelta(hours=MIN_AGE_HOURS):
        return RejectionReason.TOO_NEW

    # 10. Price band (exclude near-resolved/bond-like markets)
    yes_price = m.outcomePrices[0] if m.outcomePrices else None
    if yes_price is None or not (MIN_YES_PRICE <= yes_price <= MAX_YES_PRICE):
        return RejectionReason.PRICE_OUT_OF_RANGE

    # 11. negRisk placeholder check
    if m.negRisk:
        if m.groupItemTitle and NEGRISK_PLACEHOLDER_PATTERN.match(m.groupItemTitle):
            return RejectionReason.NEGRISK_PLACEHOLDER
        if m.volumeNum <= 0:
            return RejectionReason.NEGRISK_PLACEHOLDER

    return None


def apply_negrisk_top_k(
    eligible: list[Market],
) -> tuple[list[Market], list[Market]]:
    """
    Within each negRisk group, keep only the top NEGRISK_TOP_K members
    by liquidity (volume as tiebreaker). Non-negRisk markets pass through.
    Returns (admitted, demoted).
    """
    by_group: dict[str, list[Market]] = defaultdict(list)
    standalone: list[Market] = []
    for m in eligible:
        if m.negRisk and m.negRiskMarketID:
            by_group[m.negRiskMarketID].append(m)
        else:
            standalone.append(m)

    admitted = list(standalone)
    demoted: list[Market] = []
    for members in by_group.values():
        members.sort(key=lambda x: (x.liquidityNum, x.volumeNum), reverse=True)
        admitted.extend(members[:NEGRISK_TOP_K])
        demoted.extend(members[NEGRISK_TOP_K:])
    return admitted, demoted


def assign_bucket(m: Market, now: datetime) -> Bucket:
    horizon = m.endDate - now  # type: ignore[operator]
    if horizon <= timedelta(days=HOT_BUCKET_DAYS):
        return Bucket.HOT
    if horizon <= timedelta(days=WARM_BUCKET_DAYS):
        return Bucket.WARM
    return Bucket.COOL


# ─────────────────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────────────────

def fmt_money(x: float) -> str:
    if x >= 1_000_000:
        return f"${x / 1_000_000:.1f}M"
    if x >= 1_000:
        return f"${x / 1_000:.1f}K"
    return f"${x:.0f}"


def fmt_horizon(td: timedelta) -> str:
    total_hours = max(0, int(td.total_seconds() // 3600))
    days, hours = divmod(total_hours, 24)
    return f"{days}d {hours}h" if days else f"{hours}h"


def fmt_date(dt: datetime | None) -> str:
    if dt is None:
        return "?"
    return dt.strftime("%Y-%m-%d")


def fmt_question(q: str, width: int = 80) -> str:
    if len(q) <= width:
        return q
    return q[: width - 1].rstrip() + "…"


def display(
    *,
    events_count: int,
    markets: list[Market],
    rejections: dict[str, int],
    admitted: list[Market],
    now: datetime,
) -> None:
    print()
    print("═" * 92)
    print(f"  POLYMARKET EVENT POOL  —  {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(
        f"  Tags: {ENABLED_TAG_IDS}   "
        f"Horizon: ≤ {MAX_HORIZON_DAYS}d   "
        f"Liquidity floor: {fmt_money(MIN_LIQUIDITY_USD)}   "
        f"Volume floor: {fmt_money(MIN_VOLUME_USD)}"
    )
    print("═" * 92)
    print()
    print(f"  Events fetched:      {events_count}")
    print(f"  Markets evaluated:   {len(markets)}")
    print(f"  Eligible (admitted): {len(admitted)}")
    print(f"  Rejected:            {len(markets) - len(admitted)}")
    print()

    if rejections:
        print("  Rejection breakdown:")
        for reason, count in sorted(rejections.items(), key=lambda x: x[1], reverse=True):
            print(f"    {reason:<28} {count:>5}")
        print()

    if not admitted:
        print("  (No markets admitted. Try relaxing thresholds.)")
        print()
        return

    buckets: dict[Bucket, list[Market]] = defaultdict(list)
    for m in admitted:
        buckets[assign_bucket(m, now)].append(m)

    bucket_labels = [
        (Bucket.HOT, f"HOT  (≤ {HOT_BUCKET_DAYS}d)"),
        (Bucket.WARM, f"WARM ({HOT_BUCKET_DAYS}–{WARM_BUCKET_DAYS}d)"),
        (Bucket.COOL, f"COOL ({WARM_BUCKET_DAYS}–{MAX_HORIZON_DAYS}d)"),
    ]

    for bucket, label in bucket_labels:
        members = sorted(buckets[bucket], key=lambda m: m.endDate or now)
        if not members:
            continue
        shown_members = members[:DISPLAY_TOP_N_PER_BUCKET]
        print("─" * 120)
        print(
            f"  {label}   —   showing {len(shown_members)} of {len(members)} "
            f"market{'s' if len(members) != 1 else ''}"
        )
        print("─" * 120)
        rows: list[list[str]] = []
        for m in shown_members:
            yes_price = m.outcomePrices[0] if m.outcomePrices else None
            yes_str = f"{yes_price:.2f}" if yes_price is not None else "?"
            horizon_str = fmt_horizon((m.endDate - now)) if m.endDate else "?"
            rows.append([
                fmt_question(m.question, 54),
                fmt_date(m.endDate),
                horizon_str,
                fmt_date(m.umaEndDate),
                yes_str,
                fmt_money(m.liquidityNum),
                fmt_money(m.volumeNum),
            ])
        print(
            tabulate(
                rows,
                headers=["question", "end", "horizon", "uma_res", "yes", "liq", "vol"],
                tablefmt="github",
            )
        )
        if len(members) > DISPLAY_TOP_N_PER_BUCKET:
            print(f"    ... and {len(members) - DISPLAY_TOP_N_PER_BUCKET} more")
        print()


def market_to_event_json(m: Market) -> dict[str, Any]:
    event_slug = m.event_slug or m.slug or f"event-{m.event_id or m.id}"
    source_url = f"https://polymarket.com/event/{event_slug}"
    title = m.question or m.event_title
    description = m.description or m.event_description
    return {
        "event_id": str(uuid4()),
        "title": title,
        "description": description,
        "category": "geopolitical",
        "subcategory": "politics",
        "resolution_criteria": description,
        "resolution_deadline": (m.endDate or m.umaEndDate or datetime.now(timezone.utc)).isoformat().replace("+00:00", "Z"),
        "created_at": (m.startDate or m.event_created_at or datetime.now(timezone.utc)).isoformat().replace("+00:00", "Z"),
        "source": "polymarket",
        "source_id": str(m.id),
        "source_url": source_url,
        "tags": m.event_tag_slugs,
    }


def prompt_and_export_event(admitted: list[Market]) -> None:
    if not admitted:
        print("No admitted markets available to export.")
        return

    ordered = sorted(admitted, key=lambda m: m.endDate or datetime.max.replace(tzinfo=timezone.utc))
    print()
    print("Choose a market to export into data/events:")
    for idx, m in enumerate(ordered, start=1):
        print(f"  [{idx:>2}] {fmt_question(m.question, 72)}  ({fmt_date(m.endDate)})")

    selection = input("\nEnter number (blank to skip): ").strip()
    if not selection:
        print("Skipped export.")
        return

    try:
        chosen = ordered[int(selection) - 1]
    except (ValueError, IndexError):
        print("Invalid selection. Skipped export.")
        return

    filename_slug = chosen.event_slug or chosen.slug or f"event-{chosen.event_id or chosen.id}"
    filename_slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", filename_slug).strip("-").lower()
    if not filename_slug:
        filename_slug = f"event-{chosen.event_id or chosen.id}"
    out_path = Path("data/events") / f"{filename_slug}.json"

    payload = market_to_event_json(chosen)

    if out_path.exists():
        overwrite = input(f"{out_path} exists. Overwrite? [y/N]: ").strip().lower()
        if overwrite not in {"y", "yes"}:
            print("Skipped export.")
            return

    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Exported: {out_path}")


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Preview eligible Polymarket markets.")
    parser.add_argument(
        "--export-event",
        action="store_true",
        help="After preview, choose one admitted market and export it to data/events/*.json",
    )
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    end_date_max = now + timedelta(days=MAX_HORIZON_DAYS)

    print(f"Fetching events from Polymarket  (end_date_max = {end_date_max.date()})...")
    all_events: list[dict] = []
    for tag_id in ENABLED_TAG_IDS:
        events = fetch_events(tag_id, end_date_max)
        all_events.extend(events)
    print(f"  → {len(all_events)} events")

    markets = extract_markets(all_events)
    print(f"  → {len(markets)} markets extracted")
    print("  → evaluating eligibility...")

    rejections: Counter[str] = Counter()
    eligible_pre: list[Market] = []
    for m in markets:
        reason = evaluate_market(m, now)
        if reason is None:
            eligible_pre.append(m)
        else:
            rejections[reason.value] += 1

    admitted, demoted = apply_negrisk_top_k(eligible_pre)
    rejections[RejectionReason.NEGRISK_NOT_TOP_K.value] += len(demoted)

    display(
        events_count=len(all_events),
        markets=markets,
        rejections=dict(rejections),
        admitted=admitted,
        now=now,
    )

    if args.export_event:
        prompt_and_export_event(admitted)


if __name__ == "__main__":
    try:
        main()
    except httpx.HTTPError as e:
        print(f"\nHTTP error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(130)