# Constants for trade history
ROLLING_HISTORY_IN_DAYS = 30

# Constants for the scoring function
VOLUME_FEE = 0.01

# Price adjustment buffer for order placement
PRICE_BUFFER_ADJUSTMENT = 0.01

# Public Polymarket builder code (bytes32) used for order attribution.
POLY_BUILDER_CODE = "0x196258757463baebc045d1adc1c9c0a55cad7ac5d09ab7b7e1eb31803d9bfbe0"

# --- the Pareto knob -------------------------------------------------------
# 1.0 = pay volume only, 0.0 = pay PnL only. Everything between is on the
# frontier. This is the single most important number in the scoring module.
PARETO_ALPHA = 0.65

# --- skill / presence estimation ------------------------------------------
# One decay, applied to both PnL and volume. Half-life ~9.5 days at 0.93.
# Volume memory is the presence axis in the active score (and dust ranking);
# budget and fee floors still key off this epoch's fees only.
EDGE_DECAY = 0.93

# --- concentration ---------------------------------------------------------
# Max share of the epoch pool any single trader can take.
CONCENTRATION_CAP = 0.06
# A hard cap flattens the top of the distribution into equal payouts whenever
# it binds for everyone, which is what happens in thin epochs. So the effective
# cap never tightens below CAP_RELAX_FACTOR x the equal share of the traders who
# actually scored. With many scorers CONCENTRATION_CAP binds; with few, the cap
# only clips genuine outliers and the ranking survives.
CAP_RELAX_FACTOR = 2.5

# --- fee-return floor ------------------------------------------------------
# Active traders with real trailing edge get back at least this much of the
# fees they paid this epoch, even on a losing day. Must stay <= 1 / (1 + boost)
# so the boosted floor never exceeds 1.0x fees — otherwise PnL-neutral wash
# volume becomes a guaranteed money pump (validated at import time below).
FEE_FLOOR_MULTIPLIER = 0.57
# Floor eligibility: decayed PnL / decayed volume must clear this. `> 0` is not
# enough — a single old win plus PnL-neutral churn keeps decayed PnL positive
# forever. The trader's own volume grows the denominator, so holding the gate
# while churning requires maintaining real, proportional wins.
FEE_FLOOR_MIN_ROI = 0.005
# Floors may never consume more than this share of the active pool.
FEE_FLOOR_MAX_POOL_SHARE = 0.40

# --- dust ------------------------------------------------------------------
# Total reserve for dormant-but-historically-positive miners.
DUST_RESERVE_SHARE = 0.02
# Worst-ranked dormant miner gets this fraction of the best-ranked one's dust.
DUST_MIN_RATIO = 0.25
# Sanity check only: emitted dust weight relative to the largest weight in the
# vector. Below ~1/65535 the u16 quantisation in set_weights rounds it to zero
# and the dusting does nothing.
U16_QUANT_FLOOR = 1.0 / 65535.0

# --- gates -----------------------------------------------------------------
MIN_EPOCH_VOLUME = 1.0
MIN_EPOCHS_FOR_ELIGIBILITY = 3  # Must trade for X epochs
MIN_TRADES_FOR_ELIGIBILITY = 5  # Must have X predictions/trades
INACTIVITY_EPOCHS = 10

# --- pool split / boost ----------------------------------------------------
# Dynamic split: each pool's budget is the fees that pool generated.
# General pool is a separate track and is being retired: its history is still
# built (fees, reporting) but it earns zero tokens.
ENABLE_GENERAL_POOL_SCORING = False

# Testing only: score trades even when miner_id/hotkey do not match this
# network's metagraph, and skip profile/metadata penalties. Use when replaying
# mainnet trade history against a testnet (or otherwise mismatched) validator.
# MUST be False in production.
SKIP_METAGRAPH_MINER_ALIGNMENT = False

# Weighting parameters
# If ENABLE_STATIC_WEIGHTING is True, we will use the static weighting parameters below.
ENABLE_STATIC_WEIGHTING = False
GENERAL_POOL_WEIGHT_PERCENTAGE = 0.5
MINER_WEIGHT_PERCENTAGE = 1 - GENERAL_POOL_WEIGHT_PERCENTAGE

# This is used to give more weights (and in turn, more incentives) to the miners by taking the final miner pool weights and boosting them by this percentage.
# Set to 0 to disable.
MINER_POOL_WEIGHT_BOOST_PERCENTAGE = 0.75

TOTAL_MINER_ALPHA_PER_DAY = 2952  # 7200 alpha per day for entire subnet * 0.41 (41% for miners)

# Subnet owner burn UID
BURN_UID = 210
# Subnet owner excess miner weight UID
EXCESS_MINER_WEIGHT_UID = None
EXCESS_MINER_MIN_WEIGHT = 0  # 0.00001 should be low enough if used
EXCESS_MINER_TAKE_PERCENTAGE = 0  # percentage of the excess miner weight that is set to EXCESS_MINER_WEIGHT_UID. rest goes to BURN_UID.

# --- Fail-fast validation of weight-distribution / scoring constants -------
# These feed directly into on-chain weight setting. A value outside [0, 1]
# silently produces negative or >1 burn/excess weights (corrupting payouts)
# rather than erroring, so validate them at import time.
if not 0.0 <= EXCESS_MINER_TAKE_PERCENTAGE <= 1.0:
    raise ValueError(
        f"EXCESS_MINER_TAKE_PERCENTAGE must be in [0, 1], got {EXCESS_MINER_TAKE_PERCENTAGE}"
    )
if not 0.0 <= GENERAL_POOL_WEIGHT_PERCENTAGE <= 1.0:
    raise ValueError(
        f"GENERAL_POOL_WEIGHT_PERCENTAGE must be in [0, 1], got {GENERAL_POOL_WEIGHT_PERCENTAGE}"
    )
# Fail fast: if the boosted floor exceeds 1.0x fees, wash trading turns +EV.
if FEE_FLOOR_MULTIPLIER * (1 + MINER_POOL_WEIGHT_BOOST_PERCENTAGE) > 1.0:
    raise ValueError(
        f"FEE_FLOOR_MULTIPLIER ({FEE_FLOOR_MULTIPLIER}) x boost "
        f"(1 + {MINER_POOL_WEIGHT_BOOST_PERCENTAGE}) exceeds 1.0x fees; "
        "this makes fee-churning profitable. Lower one of them."
    )
