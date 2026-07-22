"""Bittensor validator for the Almanac subnet.

This is the entrypoint of the validator container's main process.

* :class:`Validator` — owns the bittensor objects (wallet, subtensor,
  dendrite, metagraph), the Almanac Market ``MetadataManager`` thread (when the
  Almanac Market mechanism is enabled), and the per-epoch tick that scores each
  enabled mechanism, blends, and sets weights on chain.
* :func:`combine_weights` + :class:`BlendConfig` — pure functions that
  combine two raw score vectors (Almanac Market trader-PnL + Almanac Forecasting
  Brier) into a single normalised weight vector that satisfies chain
  constraints. Tests import these directly.
* :func:`start_local_proxy` — lifts the
  ``scripts/run_forecast.py:_start_local_proxy`` helper so both the
  production entrypoint (``scripts/run_validator.py``) and the dev harness
  share one implementation.

The dev orchestrator + signing-proxy daemon stays running on a daemon
thread (it's needed by anything that spawns agent containers). The validator
loop now also polls orchestrator assignment availability when the forecasting
mechanism is enabled.
"""

from __future__ import annotations

import datetime
import math
import logging
import os
import random
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np

from src.core.config import AppConfig
from src.core.schemas import EvidenceDigest
from src.validator.forecasting.proxy_socket import (
    make_proxy_socket_connectable,
    proxy_socket_bind_umask,
)
from src.gateway.client import build_remote_providers
from src.gateway.constants import gateway_service_url
from src.gateway.signing import load_hotkey
from src.storage.store import TraceStore
from src.validator.forecasting.assignment_pipeline import (
    build_prediction_submit_payload,
    build_sandbox_assignment_agent,
    log_prediction_submit_invariants,
    normalize_prediction_values,
    process_single_assignment,
    resolve_binary_outcome_ids,
    trace_primary_model_info,
)
from src.validator.forecasting.orchestrator import Orchestrator
from src.validator.forecasting.orchestrator_api import (
    OrchestratorAssignment,
    fetch_all_scored_predictions,
    fetch_agent_event_assignment,
    submit_validator_prediction,
)
from src.validator.forecasting import scoring as forecasting_scoring

logger = logging.getLogger("almanac.validator")


@dataclass
class BlendConfig:
    """Configuration for :func:`combine_weights`.

    Each mechanism has an independent ``enabled`` toggle *and* a weight
    share. A disabled mechanism contributes nothing regardless of its
    share (and the main validator skips its scoring path entirely so we
    never even pay for the compute). When both mechanisms are present
    after the enabled filter, the blend is a simple convex combination of
    their renormalised score vectors.
    """

    market_enabled: bool
    forecasting_enabled: bool
    market_share: float
    forecasting_share: float


def combine_weights(
    weights_market: Optional[np.ndarray],
    weights_forecasting: Optional[np.ndarray],
    cfg: BlendConfig,
) -> np.ndarray:
    """Combine raw score vectors into a normalised weight vector.

    * ``None`` means "the mechanism is disabled or its score path was
      skipped this epoch" — treated as absent and ignored entirely.
    * Each present + enabled vector is normalised to sum=1 (zero-sum
      input -> zero vector).
    * Effective shares are renormalised over the remaining mechanisms so
      the final blend sums to 1 regardless of how the user set the
      nominal shares.
    * Both mechanisms absent or disabled is a hard error — refuse to set
      a flat-zero weight vector silently.
    """
    contributions: list[tuple[np.ndarray, float]] = []

    if cfg.market_enabled and weights_market is not None:
        contributions.append((_normalise(weights_market), max(cfg.market_share, 0.0)))
    if cfg.forecasting_enabled and weights_forecasting is not None:
        contributions.append((_normalise(weights_forecasting), max(cfg.forecasting_share, 0.0)))

    if not contributions:
        raise ValueError(
            "combine_weights: both mechanisms disabled or absent; refusing to set zeros."
        )

    if len(contributions) == 1:
        vec, _ = contributions[0]
        return _normalise(vec)

    total_share = sum(s for _, s in contributions)
    if total_share <= 0:
        # All shares are zero but both mechanisms are present: fall back to
        # equal weighting rather than emitting zeros silently. The validator
        # operator probably misconfigured shares; emit a warning.
        logger.warning("combine_weights: all enabled shares are zero; falling back to equal weighting.")
        n = len(contributions)
        return _normalise(sum(vec for vec, _ in contributions) / float(n))

    blended = np.zeros_like(contributions[0][0])
    for vec, share in contributions:
        blended = blended + vec * (share / total_share)
    return _normalise(blended)


def _normalise(vec: np.ndarray) -> np.ndarray:
    """Clamp negatives to zero then divide by sum (zero-sum -> zero vector)."""
    vec = np.clip(np.asarray(vec, dtype=float), 0.0, None)
    total = float(vec.sum())
    if total <= 0:
        return np.zeros_like(vec)
    return vec / total


# ---------------------------------------------------------------------------
# Local signing proxy + agent orchestrator daemon
# ---------------------------------------------------------------------------

def _prepare_proxy_socket_path(config: AppConfig) -> Path:
    """Return a writable proxy socket path, falling back if a stale root-owned file exists."""
    socket_dir = config.validator.sandbox_socket_dir
    socket_path = socket_dir / "proxy.sock"
    inputs_dir = socket_dir / "inputs"
    socket_dir.mkdir(parents=True, exist_ok=True)
    # Keep the runtime dir traversable by the sandbox user so it can read
    # per-run payload files under `inputs/`.
    try:
        os.chmod(socket_dir, 0o711)
    except OSError as exc:
        logger.warning("could not chmod proxy socket dir %s: %s", socket_dir, exc)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(inputs_dir, 0o755)
    except OSError as exc:
        logger.warning("could not chmod proxy inputs dir %s: %s", inputs_dir, exc)

    if not socket_path.exists():
        return socket_path

    try:
        socket_path.unlink()
        return socket_path
    except PermissionError:
        fallback_dir = Path("/tmp") / "almanac" / str(os.getuid())
        fallback_inputs_dir = fallback_dir / "inputs"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        fallback_inputs_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(fallback_dir, 0o711)
            os.chmod(fallback_inputs_dir, 0o755)
        except OSError as chmod_exc:
            logger.warning(
                "could not chmod fallback proxy socket dir %s: %s", fallback_dir, chmod_exc
            )
        fallback_socket_path = fallback_dir / "proxy.sock"
        if fallback_socket_path.exists():
            fallback_socket_path.unlink(missing_ok=True)
        config.validator.sandbox_socket_dir = fallback_dir
        print(
            "  WARNING: Cannot remove existing proxy socket at "
            f"{socket_path} (permission denied). Falling back to {fallback_dir}."
        )
        return fallback_socket_path


def start_local_proxy(config: AppConfig):
    """Start the validator-local signing proxy in a daemon thread.

    Returns the ``LocalProxyState`` so callers (orchestrator, tests) can
    register / drain runs without going through HTTP. Both
    ``scripts/run_validator.py`` and ``scripts/run_forecast.py`` import
    this helper so the proxy bootstrap stays in one place.
    """
    import uvicorn  # local import: only required when the proxy actually starts

    from src.gateway.local_proxy import LocalProxyState, create_app

    socket_path = _prepare_proxy_socket_path(config)

    app = create_app(config)
    state: LocalProxyState = app.state.proxy
    state.load_hotkey()

    uvicorn_config = uvicorn.Config(
        app,
        uds=str(socket_path),
        log_level="warning",
        lifespan="off",
    )
    server = uvicorn.Server(uvicorn_config)

    thread = threading.Thread(
        target=server.run,
        name="almanac-local-proxy",
        daemon=True,
    )
    old_umask = proxy_socket_bind_umask()
    try:
        thread.start()

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if socket_path.exists():
                try:
                    make_proxy_socket_connectable(socket_path)
                except OSError as exc:
                    logger.warning(
                        "could not chmod proxy socket %s: %s", socket_path, exc
                    )
                break
            time.sleep(0.05)
    finally:
        os.umask(old_umask)

    return state


# ---------------------------------------------------------------------------
# Main validator class
# ---------------------------------------------------------------------------

class Validator:
    """Long-running Bittensor validator for the Almanac subnet.

    Wakes once per minute, runs the scoring + ``set_weights`` flow at a
    randomised minute of each hour, and supervises a background
    ``MetadataManager`` (Almanac Market) when the Almanac Market mechanism is enabled.
    """

    _AUTO = object()

    def __init__(
        self,
        config: AppConfig,
        store: TraceStore,
        *,
        bt_objects=None,
        metadata_manager=_AUTO,
        clock=None,
        sleep=None,
        local_proxy_state=None,
    ) -> None:
        self._config = config
        self._store = store
        self._clock = clock or (lambda: datetime.datetime.now(datetime.timezone.utc))
        self._sleep = sleep or time.sleep

        self._validate_blend_shares()

        self._bt = bt_objects if bt_objects is not None else _build_bittensor_objects(config)

        if metadata_manager is Validator._AUTO:
            if (
                config.loop.market_enabled
                and config.loop.loop_enabled
                and config.loop.metadata_manager_enabled
            ):
                self._metadata_manager = _build_metadata_manager(config)
                self._metadata_owned = True
            else:
                self._metadata_manager = None
                self._metadata_owned = False
        else:
            self._metadata_manager = metadata_manager
            self._metadata_owned = False

        self._scoring_minute = random.randint(10, 25)
        self._stopped = False
        self._orchestrator_hotkey = Validator._AUTO
        self._orchestrator_poll_config_warned = False
        self._orchestrator_poll_signing_warned = False
        self._assignment_orchestrator: Orchestrator | None = None
        self._assignment_local_proxy_state = local_proxy_state
        self._last_assignment_execution_at: datetime.datetime | None = None

        if self._metadata_manager is not None and self._metadata_owned:
            self._metadata_manager.start()

    @property
    def metagraph(self):
        return self._bt.metagraph

    def stop(self) -> None:
        self._stopped = True
        if self._metadata_manager is not None and self._metadata_owned:
            try:
                self._metadata_manager.stop()
            except Exception:
                logger.exception("error stopping metadata manager")

    def _validate_blend_shares(self) -> None:
        loop_cfg = self._config.loop
        if not loop_cfg.loop_enabled:
            return
        if not loop_cfg.market_enabled and not loop_cfg.forecasting_enabled:
            raise ValueError(
                "Validator loop enabled but both mechanisms disabled in config constants. "
                "Refusing to start a validator that would set zero weights forever."
            )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        if not self._config.loop.loop_enabled:
            logger.info("Validator loop disabled in config constants; main loop will not start.")
            return

        logger.info("=========== STARTING ALMANAC VALIDATOR LOOP ===========")
        logger.info(
            "Scoring%s will run once per hour at minute %d (UTC). "
            "Almanac Market enabled=%s, Almanac Forecasting enabled=%s",
            "/set_weights" if self._config.loop.set_weights_enabled else " (dry-run, no set_weights)",
            self._scoring_minute,
            self._config.loop.market_enabled,
            self._config.loop.forecasting_enabled,
        )

        while not self._stopped:
            try:
                current_time = self._clock()
                if self._config.loop.forecasting_enabled:
                    self._poll_orchestrator_assignment()
                if self._should_score(current_time):
                    self.run_scoring_step()
                elif current_time.minute % 10 == 0:
                    self._log_metadata_stats()
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt; stopping validator.")
                self.stop()
                return
            except Exception:
                logger.error("Unhandled error in validator loop:\n%s", traceback.format_exc())

            self._sleep(60)

    def _should_score(self, current_time: datetime.datetime) -> bool:
        if current_time.minute != self._scoring_minute:
            return False
        if self._metadata_manager is None or not self._config.loop.market_enabled:
            return True
        # If Almanac is enabled, ensure the metadata sync isn't stale (sn41 behavior).
        stats = self._metadata_manager.get_stats()
        last_full_sync_str = stats.get("last_full_sync")
        if not last_full_sync_str:
            return True
        last_full_sync = datetime.datetime.fromisoformat(last_full_sync_str)
        if last_full_sync.tzinfo is None:
            last_full_sync = last_full_sync.replace(tzinfo=datetime.timezone.utc)
        if (
            last_full_sync < current_time - datetime.timedelta(hours=2)
            and getattr(self._bt, "network", None) != "test"
        ):
            logger.warning("Metadata last full sync is >2h ago; skipping scoring for this epoch.")
            return False
        return True

    def _log_metadata_stats(self) -> None:
        if self._metadata_manager is None:
            return
        stats = self._metadata_manager.get_stats()
        logger.info("Metadata Manager Stats: %s", stats)

    def _assignment_execution_cooldown_active(self, now: datetime.datetime) -> bool:
        if self._last_assignment_execution_at is None:
            return False
        cooldown = datetime.timedelta(
            seconds=self._config.loop.assignment_execution_cooldown_seconds,
        )
        return now - self._last_assignment_execution_at < cooldown

    def _poll_orchestrator_assignment(self) -> None:
        now = self._clock()
        if self._assignment_execution_cooldown_active(now):
            elapsed = (now - self._last_assignment_execution_at).total_seconds()
            logger.info(
                "Skipping agent and event assignment execution: last successful was "
                "%.0fs ago (cooldown=%ds).",
                elapsed,
                self._config.loop.assignment_execution_cooldown_seconds,
            )
            return

        base_url = self._config.validator.orchestrator_api_url.strip()
        if not base_url:
            if not self._orchestrator_poll_config_warned:
                logger.info(
                    "Almanac Forecasting assignment polling disabled: set validator.orchestrator_api_url "
                    "in src/core/constants.py."
                )
                self._orchestrator_poll_config_warned = True
            return

        if self._orchestrator_hotkey is Validator._AUTO:
            self._orchestrator_hotkey = load_hotkey(self._config.bittensor)
        if self._orchestrator_hotkey is None:
            if not self._orchestrator_poll_signing_warned:
                logger.warning(
                    "Almanac Forecasting assignment polling disabled: validator hotkey signing is unavailable."
                )
                self._orchestrator_poll_signing_warned = True
            return

        try:
            response = fetch_agent_event_assignment(
                base_url=base_url,
                loaded_hotkey=self._orchestrator_hotkey,
            )
        except Exception:
            logger.exception("Failed to fetch /v1/validators/agent-and-event assignment.")
            return

        if response.assignment is None:
            return

        self._handle_orchestrator_assignment(response.assignment)

    def _handle_orchestrator_assignment(self, assignment: OrchestratorAssignment) -> None:
        if not (assignment.event.description or "").strip():
            logger.info(
                "Assignment id=%s has null/empty description; defaulted description to question.",
                assignment.agentPredictionId,
            )
        if not (assignment.event.sourceMarketId or "").strip():
            logger.info(
                "Assignment id=%s missing sourceMarketId; defaulted source_id to marketId=%s.",
                assignment.agentPredictionId,
                assignment.event.marketId,
            )

        try:
            orchestrator = self._get_assignment_orchestrator()
        except Exception:
            logger.exception(
                "Failed to initialize execution orchestrator for assignment id=%s.",
                assignment.agentPredictionId,
            )
            return

        try:
            result = process_single_assignment(
                assignment=assignment,
                config=self._config,
                orchestrator=orchestrator,
                loaded_hotkey=None,
                submit_prediction=False,
            )
        except Exception:
            logger.exception(
                "Failed to process assignment id=%s through shared pipeline.",
                assignment.agentPredictionId,
            )
            return

        log_prediction_submit_invariants(logger, assignment, result.digest)
        submit_payload = result.submit_payload
        if self._orchestrator_hotkey is Validator._AUTO:
            self._orchestrator_hotkey = load_hotkey(self._config.bittensor)
        if self._orchestrator_hotkey is None:
            logger.warning(
                "Cannot submit prediction for assignment id=%s: validator hotkey unavailable.",
                assignment.agentPredictionId,
            )
            return
        try:
            ack = submit_validator_prediction(
                base_url=self._config.validator.orchestrator_api_url,
                loaded_hotkey=self._orchestrator_hotkey,
                payload=submit_payload,
                assignment=assignment,
            )
        except Exception:
            logger.exception(
                "Prediction submit failed for assignment id=%s execution_id=%s.",
                assignment.agentPredictionId,
                result.digest.execution_context.execution_id,
            )
            return

        self._last_assignment_execution_at = self._clock()
        logger.info(
            "Assignment processed id=%s status=%s miner_uid=%s source=%s source_id=%s "
            "event_id=%s execution_id=%s submit_status=%s",
            assignment.agentPredictionId,
            assignment.status,
            assignment.miner.minerUid,
            assignment.event.source,
            result.event.source_id,
            result.event.event_id,
            result.digest.execution_context.execution_id,
            ack.status,
        )

    def _build_sandbox_assignment_agent(self, assignment: OrchestratorAssignment):
        return build_sandbox_assignment_agent(assignment)

    def _get_assignment_orchestrator(self) -> Orchestrator:
        if self._assignment_orchestrator is not None:
            return self._assignment_orchestrator

        providers = build_remote_providers(gateway_service_url())
        proxy_state = self._assignment_local_proxy_state
        if self._config.validator.sandbox_type.startswith("docker") and proxy_state is None:
            proxy_state = start_local_proxy(self._config)
            self._assignment_local_proxy_state = proxy_state

        self._assignment_orchestrator = Orchestrator(
            config=self._config,
            store=self._store,
            providers=providers,
            local_proxy_state=proxy_state,
        )
        return self._assignment_orchestrator

    def _build_prediction_submit_payload(
        self,
        assignment: OrchestratorAssignment,
        digest: EvidenceDigest,
    ) -> dict:
        return build_prediction_submit_payload(assignment, digest)

    def _trace_primary_model_info(self, digest: EvidenceDigest) -> tuple[str | None, str | None]:
        return trace_primary_model_info(digest)

    def _normalize_prediction_values(
        self,
        digest: EvidenceDigest,
    ) -> tuple[float, float, bool, list[str], dict[str, str]]:
        return normalize_prediction_values(digest)

    def _coerce_unit_interval(
        self,
        value: object,
        field_name: str,
        reasons: list[str],
        *,
        fallback: float,
    ) -> float:
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            reasons.append(f"{field_name}_non_numeric")
            return fallback
        if not math.isfinite(number):
            reasons.append(f"{field_name}_non_finite")
            return fallback
        if number < 0.0 or number > 1.0:
            reasons.append(f"{field_name}_out_of_range")
            return fallback
        return number

    def _log_prediction_submit_invariants(
        self,
        assignment: OrchestratorAssignment,
        digest: EvidenceDigest,
    ) -> None:
        log_prediction_submit_invariants(logger, assignment, digest)

    def _resolve_binary_outcome_ids(self, assignment: OrchestratorAssignment) -> tuple[str, str]:
        return resolve_binary_outcome_ids(assignment)

    # ------------------------------------------------------------------
    # Per-epoch tick
    # ------------------------------------------------------------------

    def run_scoring_step(self) -> Optional[np.ndarray]:
        """Score, blend, and set weights for one epoch.

        Exposed as a public method so tests can drive a single tick
        without sleeping. Returns the final weight vector that was sent
        to ``set_weights`` (or ``None`` if both mechanisms returned no
        data this epoch and ``combine_weights`` was skipped).
        """
        self._bt.metagraph.sync()

        loop_cfg = self._config.loop

        weights_market: Optional[np.ndarray] = None
        if loop_cfg.market_enabled:
            weights_market = self._score_market()

        weights_forecasting: Optional[np.ndarray] = None
        if loop_cfg.forecasting_enabled:
            weights_forecasting = self._score_agent_predictions()

        blend_cfg = BlendConfig(
            market_enabled=loop_cfg.market_enabled,
            forecasting_enabled=loop_cfg.forecasting_enabled,
            market_share=loop_cfg.market_weight_share,
            forecasting_share=loop_cfg.forecasting_weight_share,
        )

        if weights_market is None and weights_forecasting is None:
            logger.warning("Both mechanism score paths returned no data this epoch; skipping set_weights.")
            return None

        weights = combine_weights(weights_market, weights_forecasting, blend_cfg)
        self._log_blended_weights(weights, weights_market, weights_forecasting, blend_cfg)

        if loop_cfg.set_weights_enabled:
            self._set_weights(weights)
        else:
            logger.info(
                "set_weights disabled; dry-run epoch complete (subnet=%d, size=%d).",
                self._config.bittensor.netuid,
                len(weights),
            )
        return weights

    def _log_blended_weights(
        self,
        weights: np.ndarray,
        weights_market: Optional[np.ndarray],
        weights_forecasting: Optional[np.ndarray],
        blend_cfg: BlendConfig,
    ) -> None:
        """Log the post-blend weight vector (compute only; does not touch chain)."""
        from src.validator.market.constants import BURN_UID

        market_present = blend_cfg.market_enabled and weights_market is not None
        forecasting_present = blend_cfg.forecasting_enabled and weights_forecasting is not None
        market_sum = float(np.asarray(weights_market).sum()) if market_present else 0.0
        forecasting_sum = float(np.asarray(weights_forecasting).sum()) if forecasting_present else 0.0

        # Recompute effective shares the same way combine_weights does.
        shares: list[tuple[str, float]] = []
        if market_present:
            shares.append(("market", max(blend_cfg.market_share, 0.0)))
        if forecasting_present:
            shares.append(("forecasting", max(blend_cfg.forecasting_share, 0.0)))
        total_share = sum(s for _, s in shares) or 1.0
        effective = {name: share / total_share for name, share in shares}

        burn = float(weights[BURN_UID]) if BURN_UID < len(weights) else 0.0
        nonzero = int(np.count_nonzero(weights > 0))

        logger.info(
            "Blended weight summary: market=%s (share=%.4f, raw_sum=%.4f, effective=%.4f) | "
            "forecasting=%s (share=%.4f, raw_sum=%.4f, effective=%.4f) | "
            "BURN_UID=%d weight=%.4f | non_zero_uids=%d | final_sum=%.6f",
            "present" if market_present else "absent",
            blend_cfg.market_share,
            market_sum,
            effective.get("market", 0.0),
            "present" if forecasting_present else "absent",
            blend_cfg.forecasting_share,
            forecasting_sum,
            effective.get("forecasting", 0.0),
            BURN_UID,
            burn,
            nonzero,
            float(weights.sum()),
        )
        logger.info("Blended weights: %s", weights.tolist())
        logger.info("Blended weight sum: %.6f", float(weights.sum()))

    def _score_market(self) -> Optional[np.ndarray]:
        try:
            from src.validator.market import score_market
        except Exception:
            logger.exception("Failed to import Almanac Market mechanism; skipping for this epoch.")
            return None

        loop_cfg = self._config.loop
        try:
            return score_market(
                network=self._bt.network,
                wallet=self._bt.wallet,
                dendrite=self._bt.dendrite,
                metagraph=self._bt.metagraph,
                metadata_manager=self._metadata_manager,
                use_synthetic_data=loop_cfg.use_synthetic_trading_data,
                write_trading_history_to=(
                    self._config.storage.data_dir / "trading_history.json"
                    if loop_cfg.write_trading_history
                    else None
                ),
                db_score_logging=loop_cfg.db_score_logging,
                budget_share=loop_cfg.market_weight_share,
            )
        except Exception:
            logger.exception("Almanac Market scoring failed for this epoch.")
            return None

    def _score_agent_predictions(self) -> Optional[np.ndarray]:
        base_url = self._config.validator.orchestrator_api_url.strip()
        if not base_url:
            logger.warning("Almanac Forecasting scoring disabled: validator.orchestrator_api_url is empty.")
            return None

        if self._orchestrator_hotkey is Validator._AUTO:
            self._orchestrator_hotkey = load_hotkey(self._config.bittensor)
        if self._orchestrator_hotkey is None:
            logger.warning("Almanac Forecasting scoring disabled: validator hotkey signing is unavailable.")
            return None

        try:
            scored_predictions = fetch_all_scored_predictions(
                base_url=base_url,
                loaded_hotkey=self._orchestrator_hotkey,
                days=self._config.loop.rolling_window_days,
                include_trace_summary=False,
            )
            return forecasting_scoring.score_agent_predictions(
                metagraph=self._bt.metagraph,
                scored_predictions=scored_predictions,
                rolling_window_days=self._config.loop.rolling_window_days,
                now=self._clock(),
            )
        except Exception:
            logger.exception("Almanac Forecasting scoring failed for this epoch.")
            return None

    def _set_weights(self, weights: np.ndarray) -> None:
        netuid = self._config.bittensor.netuid
        logger.info("Submitting weights to subnet %d (size=%d)...", netuid, len(weights))
        result = self._bt.subtensor.set_weights(
            netuid=netuid,
            wallet=self._bt.wallet,
            uids=self._bt.metagraph.uids,
            weights=weights,
            wait_for_inclusion=True,
        )
        if result:
            logger.info("Successfully set weights on subnet %d.", netuid)
        else:
            logger.error("Failed to set weights on subnet %d.", netuid)


# ---------------------------------------------------------------------------
# Bittensor object construction
# ---------------------------------------------------------------------------


@dataclass
class _BtObjects:
    """Tiny holder for the bittensor objects the validator needs."""

    wallet: object
    subtensor: object
    dendrite: object
    metagraph: object
    network: str


def _build_bittensor_objects(config: AppConfig) -> _BtObjects:
    """Construct wallet/subtensor/dendrite/metagraph from ``config.bittensor``.

    Imported lazily so the bittensor SDK (and its substrate deps) only
    load when the validator actually wires the chain. Tests inject a
    pre-built ``bt_objects=`` and never hit this path.
    """
    import bittensor as bt

    bt_cfg = config.bittensor
    wallet = bt.Wallet(
        name=bt_cfg.wallet_name,
        hotkey=bt_cfg.wallet_hotkey,
        path=str(bt_cfg.wallet_path),
    )
    subtensor = bt.Subtensor(network=bt_cfg.subtensor_network)
    dendrite = bt.Dendrite(wallet=wallet)
    metagraph = subtensor.metagraph(bt_cfg.netuid)

    hotkey_ss58 = wallet.hotkey.ss58_address
    if hotkey_ss58 not in metagraph.hotkeys:
        raise RuntimeError(
            f"Validator hotkey {hotkey_ss58} is not registered on netuid {bt_cfg.netuid}. "
            "Run 'btcli register' and try again."
        )

    return _BtObjects(
        wallet=wallet,
        subtensor=subtensor,
        dendrite=dendrite,
        metagraph=metagraph,
        network=bt_cfg.subtensor_network,
    )


def _build_metadata_manager(config: AppConfig):
    """Construct the Almanac Market ``MetadataManager`` against ``config.bittensor``."""
    from src.validator.market.metadata_manager import MetadataManager

    state_dir = config.storage.data_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file = str(state_dir / f"validator_state_{config.bittensor.netuid}.json")
    return MetadataManager(
        netuid=config.bittensor.netuid,
        network=config.bittensor.subtensor_network,
        state_file=state_file,
        update_interval_seconds=config.loop.metadata_update_interval_seconds,
        batch_size=config.loop.metadata_batch_size,
        batch_delay_seconds=config.loop.metadata_batch_delay_seconds,
        per_uid_delay_seconds=config.loop.metadata_per_uid_delay_seconds,
        use_bulk_commitments=config.loop.metadata_use_bulk_commitments,
    )
