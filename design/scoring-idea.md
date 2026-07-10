# Almanac Scoring: ideas + worked example

**Status: phase-2 reference notes.** MVP scoring is the pillar scorer in PR #15, which lands the same core sketched here (Brier vs a baseline, PnL-style edge against the market, a ramp for new miners) with a different structure. This note is kept for the phase-2 signals: belief path / early-move, evidence grounding, ramp shapes. Nothing here changes emissions until it is signed off. All weights are illustrative.

**v1 goal: get miners in and keep them.** The incentive should be easy to start earning from, reward genuine skill (beating the market, not restating it), and ramp a newcomer up rather than zeroing them out. The sophisticated signals can come later, once the miner base is healthy.

## What we score today (quick)

The current scorer blends four terms per miner: accuracy (Brier against a fixed 0.25 baseline, `_accuracy` at `scoring.py:342-368`), calibration, an early-signal edge over the market, and a leave-one-out ensemble term. It reads the flat submitted prediction, not the reasoning. Two issues for v1: the fixed baseline pays full marks even on easy events everyone gets right, and the reasoning/belief path is not read at all.

## The v1 idea: two simple signals

Keep v1 legible. A new miner should be able to read one page and know how to earn.

### 1. Brier accuracy

How close the probability was to what happened. For a prediction `p` (probability of YES) and outcome `y` in {0, 1}:

```
brier = (p - y)^2        # lower is better
```

Turn it into a score with a simple skill form against a reference `b_ref`:

```
accuracy_score = 1 - brier / b_ref
```

Every miner understands this. It is the backbone.

### 2. PnL against the market (beat-the-market)

PnL is profit and loss. Treat each prediction as a position taken against the market price at prediction time `m`, settled at resolution. The cleanest form:

```
pnl = (p - m) * (y - m)
```

Read it as: you take a position of size `(p - m)` in the YES share (positive = long YES, negative = short YES), and the share pays `(y - m)` at resolution. You profit when your deviation from the market points the right way.

Why this fits v1:

- **Agreeing with the market earns ~0.** If `p = m`, then `pnl = 0`. No separate consensus rule is needed: restating the crowd simply does not pay.
- **Beating the market pays**, scaled by how far you moved and whether you were right.
- **It is intuitive.** Did you make money against the crowd or not.

The market price used here is the validator-written price sealed under `trace_hash`, so a miner cannot retro-fit the baseline it is graded against.

## Miner ramp-up

Two ramps, both aimed at onboarding.

**Newcomer ramp (per miner).** A brand-new miner should not be zeroed, and should not instantly dominate either. Multiply a miner's score by a ramp that grows with its track record:

```
ramp(n) = min(1, r0 + (1 - r0) * n / N)      # e.g. r0 = 0.3, N = first 50 scored events
```

A fresh miner earns at ~30% and reaches full weight after ~50 resolved predictions. Pair it with a small participation floor so any resolved prediction earns a token amount, which keeps new miners engaged while they build a record.

**Signal ramp (network).** Ship the advanced signals (belief path, evidence grounding) at low or zero weight in v1 and turn them up as the field matures. This collects the witnessed data now without making it a barrier to earn.

## Optional extras (light in v1, more later)

- **Early-move bonus (belief path).** A small reward for moving toward the outcome before the market did. It uses the per-step probabilities, so it only turns on once those are witnessed. Light in v1, a lead signal later.
- **Calibration.** Over many predictions, do stated confidences hold up. A slow, stabilizing term, not an onboarding lever.

## Worked example: Hormuz agent-5

Event: "Strait of Hormuz traffic returns to normal by Jul 15 2026." Agent probability of YES `p = 0.12`, market `m = 0.485`, resolved NO so `y = 0`. (Illustrative resolution.)

- **Brier.** Agent `(0.12 - 0)^2 = 0.0144`. Market `(0.485 - 0)^2 = 0.2352`. Skill against the market `1 - 0.0144 / 0.2352 = 0.94`. The agent's forecast was far better than the market's.
- **PnL.** `(0.12 - 0.485) * (0 - 0.485) = (-0.365)(-0.485) = +0.177`. Positive: the agent shorted an over-priced YES and YES did not happen, so it made money against the crowd. A miner who had simply echoed the market (`p = 0.485`) scores `0`.
- **Early move.** The agent moved to 0.19, then 0.10, well below the market's 0.485, early in its run. Once per-step probabilities are witnessed, that early conviction earns the small early-move bonus.

So on this event: strong Brier, strong PnL, plus a later early-move bonus. A consensus miner earns almost nothing.

## v1 vs later

| Signal | v1 (onboarding) | Later (mature field) |
|---|---|---|
| Brier accuracy | primary | primary |
| PnL vs market | primary | primary, possibly sharper sizing |
| Newcomer ramp + floor | on | relaxes as the base grows |
| Early-move / belief path | light or off | turned up, a lead signal |
| Calibration | light | steady term |
| Evidence grounding | off | on once witnessed |

## Open questions

- How to combine Brier and PnL: one blended score, or two weighted terms.
- Ramp shape: starting fraction `r0`, ramp length `N`, floor size.
- PnL position sizing: linear in `(p - m)`, or convex to reward conviction.
- Whether the early-move bonus is worth turning on in v1 or waits for the witnessed path.

## Needs sign-off

- Any weight or baseline change shifts the emissions distribution, so it needs sign-off. Ship behind an off-switch defaulting to current behaviour and measure each signal's effect in isolation.
- The early-move / belief-path signal depends on the per-step probabilities reaching the scorer, which needs the server to carry them on the scored-predictions response. Until then it stays off and nothing else depends on it.
