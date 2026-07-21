"""Bittensor-aware request signing for validator → central gateway calls.

This module is loaded by the validator-local proxy and (logging-only) by the
central gateway. It is **not** loaded inside the agent sandbox: the runner
image deliberately omits `bittensor` so a compromised agent cannot reach
keypair material or chain code.

Wire format (`X-Auth-Version: almanac-v1`):

    canonical = f"almanac:v1:{netuid}:{unix_ts}:{nonce}:{sha256_hex(body)}"
    signature = sr25519_sign(hotkey_keypair, canonical.encode("utf-8"))

The verifier reconstructs `canonical` from the headers it receives plus the
raw body bytes and checks `sr25519_verify`.

In v1 the central gateway logs the verification result but does not enforce
it — see `src/gateway/server.py`.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import secrets
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from src.core.config import BittensorConfig

logger = logging.getLogger(__name__)


AUTH_VERSION = "almanac-v1"
SIGNATURE_SCHEME = "sr25519"


# Headers the proxy attaches and the gateway parses. Keep names stable —
# any rename is a wire-protocol break.
HEADER_HOTKEY = "X-Validator-Hotkey"
HEADER_COLDKEY = "X-Validator-Coldkey"
HEADER_NETUID = "X-Netuid"
HEADER_TIMESTAMP = "X-Timestamp"
HEADER_NONCE = "X-Nonce"
HEADER_SIGNATURE = "X-Signature"
HEADER_SCHEME = "X-Signature-Scheme"
HEADER_AUTH_VERSION = "X-Auth-Version"
HEADER_AUTHORIZATION = "Authorization"


@dataclass(frozen=True)
class LoadedKeypair:
    """Thin wrapper holding the underlying sr25519 keypair plus its ss58s.

    The `keypair` field is the substrate `Keypair` object (from
    `bittensor`/`substrateinterface`). We don't type it strictly to keep this
    module importable on environments without bittensor installed.
    """

    keypair: Any
    hotkey_ss58: str
    coldkey_ss58: Optional[str]


# ---------------------------------------------------------------------------
# Hotkey loading
# ---------------------------------------------------------------------------


def load_hotkey(cfg: BittensorConfig) -> Optional[LoadedKeypair]:
    """Load the validator's Bittensor hotkey from disk.

    Returns `None` if `cfg.signing_required` is False — in that mode the proxy
    forwards requests unsigned (dev-only). If `signing_required` is True and
    loading fails, raises so the proxy refuses to start.
    """
    if not cfg.signing_required:
        logger.warning(
            "BittensorConfig.signing_required=False — running without a hotkey. "
            "Outbound requests to the central gateway will be UNSIGNED."
        )
        return None

    try:
        import bittensor as wallet_mod  # type: ignore
    except ImportError:
        # Fall back to the lightweight `bittensor_wallet` package, which exposes
        # the same `Wallet` constructor and is sufficient for signing (mirrors
        # the `_load_keypair_class` fallback below).
        try:
            import bittensor_wallet as wallet_mod  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "neither bittensor nor bittensor_wallet is installed but "
                "signing_required=True. Install one (`pip install -r "
                "requirements.txt`) or run with --unsafe-no-signing for local dev."
            ) from exc

    wallet = wallet_mod.Wallet(
        name=cfg.wallet_name,
        hotkey=cfg.wallet_hotkey,
        path=str(cfg.wallet_path),
    )

    try:
        hotkey_kp = wallet.hotkey
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load hotkey '{cfg.wallet_hotkey}' from wallet "
            f"'{cfg.wallet_name}' at '{cfg.wallet_path}'. Check that the "
            f"wallet directory is mounted into the validator container and "
            f"the hotkey file is decryptable (no passphrase, or passphrase "
            f"available via env)."
        ) from exc

    coldkey_ss58: Optional[str] = None
    try:
        coldkey_ss58 = wallet.coldkeypub.ss58_address
    except Exception:
        coldkey_ss58 = None

    loaded = LoadedKeypair(
        keypair=hotkey_kp,
        hotkey_ss58=hotkey_kp.ss58_address,
        coldkey_ss58=coldkey_ss58,
    )
    logger.info(
        "Loaded validator hotkey ss58=%s netuid=%d (coldkey=%s)",
        loaded.hotkey_ss58,
        cfg.netuid,
        coldkey_ss58 or "<unknown>",
    )
    return loaded


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------


def _canonical_message(netuid: int, ts: int, nonce: str, body: bytes) -> bytes:
    body_hash = hashlib.sha256(body).hexdigest()
    return f"almanac:v1:{netuid}:{ts}:{nonce}:{body_hash}".encode("utf-8")


def sign_request(
    loaded: Optional[LoadedKeypair],
    body: bytes,
    netuid: int,
) -> dict[str, str]:
    """Mint the auth headers for a single outbound POST.

    If `loaded` is None (signing disabled) returns an empty dict — caller
    forwards the request unsigned.
    """
    if loaded is None:
        return {}

    ts = int(time.time())
    nonce = secrets.token_hex(8)
    canonical = _canonical_message(netuid, ts, nonce, body)
    signature_bytes = loaded.keypair.sign(canonical)
    signature_hex = signature_bytes.hex() if isinstance(signature_bytes, (bytes, bytearray)) else str(signature_bytes)
    basic_token = base64.b64encode(f"{loaded.hotkey_ss58}:{signature_hex}".encode("utf-8")).decode("ascii")

    headers = {
        HEADER_AUTHORIZATION: f"Basic {basic_token}",
        HEADER_HOTKEY: loaded.hotkey_ss58,
        HEADER_NETUID: str(netuid),
        HEADER_TIMESTAMP: str(ts),
        HEADER_NONCE: nonce,
        HEADER_SIGNATURE: signature_hex,
        HEADER_SCHEME: SIGNATURE_SCHEME,
        HEADER_AUTH_VERSION: AUTH_VERSION,
    }
    if loaded.coldkey_ss58:
        headers[HEADER_COLDKEY] = loaded.coldkey_ss58
    return headers


# ---------------------------------------------------------------------------
# Verification (logging-only at the gateway in v1; full enforcement is a
# follow-up phase with metagraph membership checks).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerificationResult:
    verified: bool
    hotkey: Optional[str]
    netuid: Optional[int]
    reason: Optional[str] = None


def verify_request_headers(
    headers: Mapping[str, str],
    body: bytes,
    *,
    max_skew_seconds: int = 300,
) -> VerificationResult:
    """Best-effort verification of the auth headers against the request body.

    Returns a `VerificationResult` with `verified=False` when any header is
    missing, the timestamp is out of skew, or the signature does not check
    out. Never raises — this is called from the gateway's logging path.
    """
    auth_version = _get_header(headers, HEADER_AUTH_VERSION)
    if not auth_version:
        return VerificationResult(verified=False, hotkey=None, netuid=None, reason="missing_auth_version")
    if auth_version != AUTH_VERSION:
        return VerificationResult(
            verified=False,
            hotkey=_get_header(headers, HEADER_HOTKEY),
            netuid=_safe_int(_get_header(headers, HEADER_NETUID)),
            reason=f"unsupported_auth_version:{auth_version}",
        )

    hotkey = _get_header(headers, HEADER_HOTKEY)
    netuid_raw = _get_header(headers, HEADER_NETUID)
    ts_raw = _get_header(headers, HEADER_TIMESTAMP)
    nonce = _get_header(headers, HEADER_NONCE)
    signature_hex = _get_header(headers, HEADER_SIGNATURE)
    scheme = _get_header(headers, HEADER_SCHEME) or SIGNATURE_SCHEME

    if not all([hotkey, netuid_raw, ts_raw, nonce, signature_hex]):
        return VerificationResult(
            verified=False,
            hotkey=hotkey,
            netuid=_safe_int(netuid_raw),
            reason="missing_signature_header",
        )

    if scheme != SIGNATURE_SCHEME:
        return VerificationResult(
            verified=False,
            hotkey=hotkey,
            netuid=_safe_int(netuid_raw),
            reason=f"unsupported_scheme:{scheme}",
        )

    netuid = _safe_int(netuid_raw)
    ts = _safe_int(ts_raw)
    if netuid is None or ts is None:
        return VerificationResult(verified=False, hotkey=hotkey, netuid=netuid, reason="bad_int_header")

    skew = abs(int(time.time()) - ts)
    if skew > max_skew_seconds:
        return VerificationResult(
            verified=False,
            hotkey=hotkey,
            netuid=netuid,
            reason=f"timestamp_skew:{skew}s",
        )

    try:
        Keypair = _load_keypair_class()
    except ImportError:
        return VerificationResult(
            verified=False,
            hotkey=hotkey,
            netuid=netuid,
            reason="bittensor_not_installed",
        )

    canonical = _canonical_message(netuid, ts, nonce or "", body)
    try:
        kp = Keypair(ss58_address=hotkey, crypto_type=_sr25519_crypto_type())
        ok = kp.verify(canonical, bytes.fromhex(signature_hex or ""))
    except Exception as exc:
        return VerificationResult(
            verified=False,
            hotkey=hotkey,
            netuid=netuid,
            reason=f"verify_error:{type(exc).__name__}",
        )

    if not ok:
        return VerificationResult(
            verified=False, hotkey=hotkey, netuid=netuid, reason="bad_signature"
        )

    return VerificationResult(verified=True, hotkey=hotkey, netuid=netuid)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_header(headers: Mapping[str, str], name: str) -> Optional[str]:
    # Mapping case-insensitivity differs across types (httpx Headers, FastAPI
    # request.headers, plain dicts in tests). Normalise.
    if name in headers:
        return headers[name]
    lower = name.lower()
    for k, v in headers.items():
        if k.lower() == lower:
            return v
    return None


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_keypair_class():
    """Return the substrate `Keypair` class.

    Tries `bittensor` first (re-exports it), then `substrateinterface` directly.
    Raises ImportError if neither is installed.
    """
    try:
        from bittensor_wallet.keypair import Keypair  # type: ignore

        return Keypair
    except ImportError:
        pass
    try:
        from substrateinterface import Keypair  # type: ignore

        return Keypair
    except ImportError:
        pass
    try:
        from bittensor import Keypair  # type: ignore

        return Keypair
    except ImportError as exc:
        raise ImportError(
            "Could not import substrate Keypair. Install `bittensor` or "
            "`substrate-interface`."
        ) from exc


def _sr25519_crypto_type() -> int:
    """Crypto-type constant for sr25519. 1 in scalecodec/substrate-interface."""
    try:
        from scalecodec.utils.ss58 import ss58_decode  # noqa: F401  # sanity import
    except ImportError:
        pass
    return 1
