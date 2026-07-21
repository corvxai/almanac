# Pricing Cards and Cost Estimation

The gateway records provider usage metadata for each call and computes `cost_units`.
When a provider response includes explicit cost (for example OpenRouter `usage.cost`),
that authoritative value is used directly.

When a provider response includes token usage but no explicit cost (for example
Claude Messages API), the gateway computes a deterministic estimate from
token counts and a pricing table.

## Pricing file

Default location:

- `config/pricing_cards.json`

Current Claude baseline rates in that file are sourced from Anthropic's pricing
documentation (standard tier, not batch rates). Keep this file updated if
provider pricing changes.

Schema:

- Top-level object keyed by `"<provider>/<model>"`.
- Optional provider fallback key `"<provider>/__default__"`.
- Value fields:
  - `input_per_mtoken`
  - `output_per_mtoken`
  - `cache_read_input_per_mtoken` (optional)
  - `cache_creation_input_per_mtoken` (optional)

Example:

```json
{
  "claude/__default__": {
    "input_per_mtoken": 3.0,
    "output_per_mtoken": 15.0
  },
  "claude/claude-sonnet-4-6": {
    "input_per_mtoken": 3.0,
    "output_per_mtoken": 15.0,
    "cache_read_input_per_mtoken": 0.3,
    "cache_creation_input_per_mtoken": 3.75
  }
}
```

## Runtime overrides

- `FORECASTING_MODEL_PRICING_FILE`: absolute path to an alternate pricing cards file.
- `FORECASTING_MODEL_PRICING_JSON`: JSON object merged on top of file cards.

Use the JSON override for temporary hotfixes and keep long-term pricing in the
versioned file.

## Sync script

Use the sync helper to discover Claude models referenced in code and add any
missing model cards using family defaults:

```bash
python scripts/sync_pricing_cards.py
python scripts/sync_pricing_cards.py --write
```

The script scans `src/` and `scripts/` for model literals and creates missing
`claude/<model>` keys by copying one of:

- `claude/__default_opus__`
- `claude/__default_sonnet__`
- `claude/__default_haiku__`
- fallback: `claude/__default__`

## Estimation formula

For a single provider call:

- `uncached_input_tokens = max(input_tokens - cache_read_input_tokens, 0)`
- `estimated_cost = (uncached_input_tokens * input_rate + output_tokens * output_rate + cache_read_input_tokens * cache_read_rate + cache_creation_input_tokens * cache_creation_rate) / 1_000_000`

The final value is rounded and stored in:

- `provider_calls[*].usage_meta.cost`
- `provider_calls[*].cost_units`

Aggregate totals are stored in:

- `trace_integrity.total_provider_cost`
- `trace_integrity.usage_totals`

## Accuracy and maintenance

- Update price cards whenever providers change list pricing or your contract rates.
- Recommended cadence: weekly check, plus immediate updates on provider pricing announcements.
- Reconcile periodically against provider billing exports/APIs for finance-grade reporting.
