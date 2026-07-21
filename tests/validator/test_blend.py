"""Tests for the per-mechanism blend.

Imports the pure functions directly from ``src.validator.validator`` —
they're module-level helpers, no need to instantiate the ``Validator`` class.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.validator.validator import BlendConfig, combine_weights


def _cfg(
    *,
    market_enabled: bool = True,
    forecasting_enabled: bool = True,
    market_share: float = 1.0,
    forecasting_share: float = 1.0,
) -> BlendConfig:
    return BlendConfig(
        market_enabled=market_enabled,
        forecasting_enabled=forecasting_enabled,
        market_share=market_share,
        forecasting_share=forecasting_share,
    )


def test_single_mechanism_returns_normalised_vector() -> None:
    w = np.array([2.0, 4.0, 4.0])
    out = combine_weights(w, None, _cfg(forecasting_enabled=False))

    assert out.shape == (3,)
    assert pytest.approx(out.sum(), abs=1e-9) == 1.0
    np.testing.assert_allclose(out, np.array([0.2, 0.4, 0.4]))


def test_disabled_mechanism_is_ignored_even_if_vector_passed() -> None:
    w_market = np.array([1.0, 0.0, 0.0])
    w_forecasting = np.array([0.0, 1.0, 0.0])

    # Almanac flagged disabled — its vector must not contribute.
    out = combine_weights(w_market, w_forecasting, _cfg(market_enabled=False))

    np.testing.assert_allclose(out, np.array([0.0, 1.0, 0.0]))


def test_both_enabled_linear_blend_with_explicit_shares() -> None:
    w_market = np.array([1.0, 0.0, 0.0])
    w_forecasting = np.array([0.0, 1.0, 0.0])

    out = combine_weights(
        w_market, w_forecasting, _cfg(market_share=0.75, forecasting_share=0.25)
    )

    np.testing.assert_allclose(out, np.array([0.75, 0.25, 0.0]))


def test_shares_are_renormalised_so_final_sums_to_one() -> None:
    w_market = np.array([1.0, 0.0])
    w_forecasting = np.array([0.0, 1.0])

    # Nominal shares sum to 4; effective shares should be 1/4 and 3/4.
    out = combine_weights(
        w_market, w_forecasting, _cfg(market_share=1.0, forecasting_share=3.0)
    )

    np.testing.assert_allclose(out, np.array([0.25, 0.75]))
    assert pytest.approx(out.sum(), abs=1e-9) == 1.0


def test_zero_share_on_one_mechanism_collapses_to_other() -> None:
    w_market = np.array([1.0, 0.0])
    w_forecasting = np.array([0.0, 1.0])

    out = combine_weights(
        w_market, w_forecasting, _cfg(market_share=0.0, forecasting_share=1.0)
    )

    np.testing.assert_allclose(out, np.array([0.0, 1.0]))


def test_zero_sum_input_vector_does_not_explode() -> None:
    w_market = np.zeros(3)
    w_forecasting = np.array([1.0, 1.0, 1.0])

    out = combine_weights(w_market, w_forecasting, _cfg())

    # Almanac Market vector normalises to zero; blend reduces to forecasting share.
    np.testing.assert_allclose(out, np.array([1 / 3, 1 / 3, 1 / 3]))


def test_negative_components_are_clamped_before_normalisation() -> None:
    w = np.array([-1.0, 2.0, 3.0])

    out = combine_weights(w, None, _cfg(forecasting_enabled=False))

    np.testing.assert_allclose(out, np.array([0.0, 0.4, 0.6]))


def test_none_on_both_inputs_is_a_hard_error() -> None:
    with pytest.raises(ValueError, match="both mechanisms"):
        combine_weights(None, None, _cfg())


def test_both_disabled_is_a_hard_error_even_with_vectors() -> None:
    w_market = np.array([1.0, 0.0])
    w_forecasting = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="both mechanisms"):
        combine_weights(
            w_market,
            w_forecasting,
            _cfg(market_enabled=False, forecasting_enabled=False),
        )


def test_all_zero_shares_falls_back_to_equal_weighting_with_warning(caplog) -> None:
    w_market = np.array([1.0, 0.0])
    w_forecasting = np.array([0.0, 1.0])

    with caplog.at_level("WARNING", logger="almanac.validator"):
        out = combine_weights(
            w_market,
            w_forecasting,
            _cfg(market_share=0.0, forecasting_share=0.0),
        )

    np.testing.assert_allclose(out, np.array([0.5, 0.5]))
    assert any("equal weighting" in m for m in caplog.messages)


def test_blend_identity_matches_expected_convex_combination() -> None:
    w_market = np.array([0.5, 0.3, 0.2])
    w_forecasting = np.array([0.2, 0.7, 0.1])
    cfg = _cfg(market_share=2.0, forecasting_share=1.0)

    out = combine_weights(w_market, w_forecasting, cfg)
    expected = (2.0 / 3.0) * w_market + (1.0 / 3.0) * w_forecasting

    np.testing.assert_allclose(out, expected)
    assert pytest.approx(out.sum(), abs=1e-9) == 1.0


def test_blend_applies_shares_once_no_double_scaling() -> None:
    w_market = np.array([1.0, 0.0])
    w_forecasting = np.array([0.0, 1.0])

    out = combine_weights(
        w_market, w_forecasting, _cfg(market_share=0.8, forecasting_share=0.2)
    )

    # Guardrail: if share were accidentally applied twice to Almanac,
    # this would drift away from [0.8, 0.2].
    np.testing.assert_allclose(out, np.array([0.8, 0.2]))


def test_blend_preserves_expected_burn_uid_contribution() -> None:
    # Index 2 stands in for BURN_UID in this deterministic blend test.
    w_market = np.array([0.7, 0.0, 0.3])
    w_forecasting = np.array([0.0, 1.0, 0.0])

    out = combine_weights(
        w_market, w_forecasting, _cfg(market_share=0.6, forecasting_share=0.4)
    )

    np.testing.assert_allclose(out, np.array([0.42, 0.4, 0.18]))
    assert out[2] == pytest.approx(0.18)
