"""Almanac incentive mechanism orchestration.

Standalone functions for the Almanac incentive mechanism. (trade-history fetch,
pagination, retry policy, profile validation, set_weights wrapper).

Public entry point: ``score_almanac`` — returns a score/weight vector of
length ``len(metagraph.uids)`` for the current epoch.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from requests.auth import HTTPBasicAuth

from .constants import ROLLING_HISTORY_IN_DAYS, TOTAL_MINER_ALPHA_PER_DAY
from .metadata_manager import MetadataManager

# ``scoring`` pulls cvxpy (and a chain of heavy scientific deps) at import
# time. Defer the import to ``score_almanac`` so the rest of the package
# is usable on installs that don't need the Almanac scoring math.

logger = logging.getLogger("arcratio.almanac")


_DEFAULT_TRADING_HISTORY_ENDPOINT_PROD = "https://api.almanac.market/api/v1/trading/trading-history"
#_DEFAULT_TRADING_HISTORY_ENDPOINT_TEST = "https://test-api.almanac.market/api/v1/trading/trading-history"
_DEFAULT_TRADING_HISTORY_ENDPOINT_TEST = "http://localhost:3001/api/v1/trading/trading-history"
_DEFAULT_TRADING_HISTORY_BATCH_LIMIT = 1000
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 30
_TAO_PRICE_URL = "https://api.coingecko.com/api/v3/simple/price?ids=bittensor&vs_currencies=usd"
_TAO_PRICE_TIMEOUT_SECONDS = 10


def default_trading_history_endpoint(network: str) -> str:
    """Return the sn41-compatible Almanac trading-history endpoint for a network."""
    if (network or "").lower() == "test":
        return _DEFAULT_TRADING_HISTORY_ENDPOINT_TEST
    return _DEFAULT_TRADING_HISTORY_ENDPOINT_PROD


def _retry_with_backoff(func, *args, **kwargs):
    """Generic retry wrapper with exponential backoff (sn41-verbatim shape)."""
    max_retries = 3
    base_delay = 3  # seconds

    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            if attempt == max_retries - 1:
                logger.error("%s failed after %d attempts: %s", func.__name__, max_retries, e)
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "%s attempt %d failed: %s. Retrying in %ds...",
                func.__name__,
                attempt + 1,
                e,
                delay,
            )
            time.sleep(delay)


def fetch_tao_price() -> float:
    """Fetch the live TAO/USD price from coingecko (sn41-verbatim)."""

    def _fetch() -> float:
        response = requests.get(_TAO_PRICE_URL, timeout=_TAO_PRICE_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()["bittensor"]["usd"]

    return _retry_with_backoff(_fetch)


def compute_epoch_budget(metagraph, tao_price_usd: float) -> float:
    """Per-epoch (24h) USD budget for the miner pool.

    Uses ``metagraph.pool.moving_price`` to convert from TAO to alpha and
    then multiplies by sn41's ``TOTAL_MINER_ALPHA_PER_DAY``. sn41-verbatim.
    """
    alpha_price_usd = metagraph.pool.moving_price * tao_price_usd
    return alpha_price_usd * TOTAL_MINER_ALPHA_PER_DAY


def fetch_trading_history(
    *,
    network: str,
    dendrite,
    metagraph,
    endpoint: Optional[str] = None,
    days: int = ROLLING_HISTORY_IN_DAYS,
    batch_limit: int = _DEFAULT_TRADING_HISTORY_BATCH_LIMIT,
    use_synthetic_data: bool = False,
    synthetic_data_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Fetch the rolling-window trade history for scoring.

    Mirrors sn41 ``Validator.fetch_trading_history`` verbatim: hotkey-signed
    HTTP basic auth, cursor pagination, retry on each batch, infinite-loop
    guard when the API echoes back the same cursor with no new rows.

    When ``use_synthetic_data`` is true, loads the sn41-shipped mock dataset
    from ``synthetic_data_path`` (defaults to the vendored test fixture under
    ``src/validator/almanac/tests/advanced_mock_data.json``) and rewrites
    ``miner_id`` 170 -> 17 + ``miner_hotkey`` per the sn41 convention.
    """

    def _fetch() -> List[Dict[str, Any]]:
        if use_synthetic_data:
            path = synthetic_data_path or Path(__file__).resolve().parent / "tests" / "advanced_mock_data.json"
            with open(path, "r") as f:
                trading_history = json.load(f)

            for trade in trading_history:
                if trade.get("miner_id") == 170:
                    trade["miner_id"] = 17
                miner_id = trade.get("miner_id")
                if (
                    miner_id is not None
                    and miner_id < len(metagraph.hotkeys)
                    and trade.get("is_general_pool") is False
                ):
                    trade["miner_hotkey"] = metagraph.hotkeys[miner_id]
            return trading_history

        url = endpoint or default_trading_history_endpoint(network)
        logger.info("Fetching trading history from %s for %d days", url, days)

        keypair = dendrite.keypair
        hotkey = keypair.ss58_address
        signature = f"0x{keypair.sign(hotkey).hex()}"

        def _cursor_is_blank(cursor: Optional[str]) -> bool:
            if cursor is None:
                return True
            if isinstance(cursor, str) and not cursor.strip():
                return True
            return False

        def _fetch_batch(cursor: Optional[str] = None):
            params: Dict[str, object] = {
                "days": days,
                "limit": batch_limit,
            }
            if not _cursor_is_blank(cursor):
                params["cursor"] = cursor

            response = requests.get(
                url,
                params=params,
                auth=HTTPBasicAuth(hotkey, signature),
                timeout=_DEFAULT_REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            api_response = response.json()

            if not isinstance(api_response, dict) or "data" not in api_response:
                raise ValueError(
                    f"Unexpected API response format: {type(api_response)}. Expected dict with 'data' field."
                )

            if api_response.get("success") is False:
                raise ValueError(f"API reported success=false: {api_response!r}")

            trading_history_batch = api_response["data"]
            if not isinstance(trading_history_batch, list):
                raise ValueError(
                    f"Expected 'data' field to be a list, got {type(trading_history_batch)}"
                )

            meta = api_response.get("meta", {})
            pagination = meta.get("pagination", {}) if isinstance(meta, dict) else {}
            next_cursor = (
                pagination.get("next_cursor") if isinstance(pagination, dict) else None
            )
            if next_cursor is not None and not isinstance(next_cursor, str):
                next_cursor = str(next_cursor)

            return trading_history_batch, next_cursor

        all_trading_history: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        batch_num = 1

        while True:
            cursor_log = "(initial)" if cursor is None else f"(cursor len={len(cursor)})"
            logger.info("Fetching trading history batch %d %s", batch_num, cursor_log)

            trading_history_batch, next_cursor = _retry_with_backoff(_fetch_batch, cursor)

            if trading_history_batch:
                all_trading_history.extend(trading_history_batch)
                logger.info(
                    "Batch %d: fetched %d trades (total so far: %d)",
                    batch_num,
                    len(trading_history_batch),
                    len(all_trading_history),
                )
            else:
                logger.warning(
                    "Empty data[] for batch %d %s (still honoring meta.pagination.next_cursor)",
                    batch_num,
                    cursor_log,
                )

            if _cursor_is_blank(next_cursor):
                logger.info(
                    "Finished fetching all trading history. Total trades: %d",
                    len(all_trading_history),
                )
                break

            if not trading_history_batch and cursor is not None and next_cursor == cursor:
                logger.error(
                    "Pagination stuck: empty data and next_cursor unchanged; stopping to avoid an infinite loop."
                )
                break

            cursor = next_cursor
            batch_num += 1

        return all_trading_history

    return _retry_with_backoff(_fetch)


def validate_miner_profiles(
    miner_history: Dict[str, Any],
    metadata_manager: Optional[MetadataManager],
) -> List[int]:
    """Return the list of miner UIDs to penalize based on profile validation.

    Verbatim port of sn41's inline profile-validation block: a miner is
    penalised (score forced to 0) if any of the following hold:

    - their stored profile id is missing
    - their stored profile id contains a comma (multiple ids registered)
    - we have no on-chain metadata commitment for them via the metadata manager
    - their on-chain metadata polymarket id is not a substring of the stored
      profile id (chain only stores a partial id for privacy)
    """
    miners_to_penalize: List[int] = []
    if metadata_manager is None:
        return miners_to_penalize

    miner_profiles = miner_history.get("miner_profiles", {}) if isinstance(miner_history, dict) else {}
    for miner_uid, profile_id in miner_profiles.items():
        if profile_id is None:
            logger.error(
                "Miner %s has no profile id defined. This should never happen. Setting score to 0.",
                miner_uid,
            )
            miners_to_penalize.append(miner_uid)
            continue
        if "," in profile_id:
            logger.warning(
                "Miner %s has multiple profile ids defined (%s). Setting score to 0.",
                miner_uid,
                profile_id,
            )
            miners_to_penalize.append(miner_uid)
            continue

        miner_metadata = metadata_manager.get_miner_metadata(miner_uid)
        if miner_metadata is None:
            logger.warning("Miner %s has no metadata defined. Setting score to 0.", miner_uid)
            miners_to_penalize.append(miner_uid)
            continue

        if miner_metadata not in profile_id:
            logger.warning(
                "Miner %s polymarket ids do not match (chain id not in profile id). Setting score to 0.",
                miner_uid,
            )
            miners_to_penalize.append(miner_uid)
            continue

        logger.info(
            "UID %s: Almanac polymarket id matches Bittensor chain metadata polymarket id.",
            miner_uid,
        )

    return miners_to_penalize


def score_almanac(
    *,
    network: str,
    wallet,  # unused in scoring, kept for symmetry with future signing paths
    dendrite,
    metagraph,
    metadata_manager: Optional[MetadataManager],
    use_synthetic_data: bool = False,
    write_trading_history_to: Optional[Path] = None,
    print_stats: bool = True,
    db_score_logging: bool = False,
) -> Optional[np.ndarray]:
    """Compute the Almanac scores for the current epoch.

    Orchestrates the scoring end-to-end: fetch the trade window, fetch
    TAO/alpha price, run the ``score_miners`` + ``calculate_weights``,
    apply profile-validation penalties, and return the final scores.

    Returns ``None`` if the upstream data fetch fails. Callers should treat
    that as "skip the Almanac mechanism for this epoch". Any non-fatal error inside
    scoring will propagate normally.
    """
    try:
        trading_history = fetch_trading_history(
            network=network,
            dendrite=dendrite,
            metagraph=metagraph,
            use_synthetic_data=use_synthetic_data,
        )

        if write_trading_history_to is not None:
            with open(write_trading_history_to, "w") as f:
                json.dump(trading_history, f, indent=2)
            logger.info(
                "Trading history written to %s (%d trades)",
                write_trading_history_to,
                len(trading_history),
            )

        tao_price_usd = fetch_tao_price()
        logger.info("TAO price: %.2f USD", tao_price_usd)

        current_epoch_budget = compute_epoch_budget(metagraph, tao_price_usd)
        logger.info("Current epoch (24h) budget: %.2f USD", current_epoch_budget)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to fetch required Almanac data for scoring: %s", exc)
        logger.warning("Skipping Almanac scoring for this epoch.")
        return None

    # Imported here so that callers who never enable the Almanac mechanism
    # don't have to install cvxpy / scipy / tabulate.
    from .scoring import calculate_weights, print_pool_stats, score_miners

    all_uids = metagraph.uids.tolist()
    all_hotkeys = metagraph.hotkeys

    (
        miner_history,
        general_pool_history,
        miners_scores,
        general_pool_scores,
        miner_budget,
        general_pool_budget,
    ) = score_miners(all_uids, all_hotkeys, trading_history, current_epoch_budget)

    if print_stats:
        print("\n############################## OVERALL POOL STATS ##############################")
        print_pool_stats(miner_history, general_pool_history)
        print("##################################################################################\n")
        print("\n########################## CURRENT EPOCH POOL STATS ############################")
        print_pool_stats(
            miner_history,
            general_pool_history,
            include_current_epoch=True,
            miner_scores=miners_scores,
            general_pool_scores=general_pool_scores,
        )
        print("##################################################################################\n")

    miners_to_penalize: List[int] = []
    if not use_synthetic_data:
        miners_to_penalize = validate_miner_profiles(miner_history, metadata_manager)

    weights = calculate_weights(
        miners_scores,
        general_pool_scores,
        current_epoch_budget,
        miner_budget,
        general_pool_budget,
        miners_to_penalize,
        all_uids,
    )

    if db_score_logging:
        try:
            from .storage.postgres_validator_storage import log_scores_to_database

            log_scores_to_database(
                miner_history,
                general_pool_history,
                miners_scores,
                general_pool_scores,
                miner_budget,
                general_pool_budget,
                all_hotkeys,
                weights,
            )
            logger.info("Logged Almanac scores to database.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to log Almanac scores to database: %s", exc)

    return np.asarray(weights, dtype=float)
