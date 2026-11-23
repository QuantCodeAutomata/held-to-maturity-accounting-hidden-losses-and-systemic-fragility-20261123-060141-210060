"""Unit tests for src/empirical.py on small synthetic fixtures (Experiment 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.empirical import (
    asset_weighted_correlation,
    asset_weighted_series,
    balance_sheet_ratios,
    daily_returns,
    mean_off_diagonal_correlation,
    pairwise_correlation,
    run_ols,
    top_n_average_ratios,
)


def _financials() -> pd.DataFrame:
    return pd.DataFrame({
        "CERT": [1, 2, 3],
        "REPDTE": ["20231231"] * 3,
        "ASSET": [1000.0, 2000.0, 500.0],
        "LIAB": [900.0, 1800.0, 400.0],
        "DEP": [700.0, 1400.0, 300.0],
        "SC": [200.0, 300.0, 100.0],
        "SCAGE": [50.0, 150.0, 20.0],
    })


def test_balance_sheet_ratios_values_and_bounds() -> None:
    ratios = balance_sheet_ratios(_financials())
    # liabilities/assets and htm/securities must lie in (0, 1) for these fixtures.
    assert (ratios["liabilities_to_assets"] > 0).all()
    assert (ratios["liabilities_to_assets"] < 1).all()
    assert (ratios["htm_to_securities"] > 0).all()
    assert (ratios["htm_to_securities"] < 1).all()
    row = ratios.iloc[1]
    assert row["securities_to_assets"] == pytest.approx(300.0 / 2000.0)
    assert row["htm_to_securities"] == pytest.approx(150.0 / 300.0)


def test_balance_sheet_ratios_missing_column_raises() -> None:
    bad = _financials().drop(columns=["SCAGE"])
    with pytest.raises(ValueError):
        balance_sheet_ratios(bad)


def test_top_n_average_ratios_averages_across_banks() -> None:
    ratios = balance_sheet_ratios(_financials())
    averaged = top_n_average_ratios({2023: ratios})
    assert averaged.loc[0, "n_banks"] == 3
    assert averaged.loc[0, "liabilities_to_assets"] == pytest.approx(
        ratios["liabilities_to_assets"].mean()
    )


def test_daily_returns_shape_and_no_lookahead() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    bars = pd.DataFrame({
        "date": list(dates) * 2,
        "ticker": ["A"] * 5 + ["B"] * 5,
        "close": [100, 101, 102, 101, 103, 50, 50.5, 51, 50, 50.5],
    })
    returns = daily_returns(bars)
    # One row is lost to the first-difference (pct_change) transform.
    assert returns.shape[0] == 4
    # Return at t only uses prices at t and t-1, never a future price.
    expected_first = 101 / 100 - 1
    assert returns.iloc[0]["A"] == pytest.approx(expected_first)


def test_pairwise_correlation_symmetric_unit_diagonal() -> None:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    returns = pd.DataFrame(rng.normal(size=(60, 3)), index=dates, columns=["A", "B", "C"])
    corr = pairwise_correlation(returns, "2024-01-01", "2024-03-01")
    assert np.allclose(np.diag(corr.to_numpy()), 1.0)
    assert np.allclose(corr.to_numpy(), corr.to_numpy().T)


def test_pairwise_correlation_too_few_observations_raises() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    returns = pd.DataFrame(np.zeros((5, 2)), index=dates, columns=["A", "B"])
    with pytest.raises(ValueError):
        pairwise_correlation(returns, "2024-01-01", "2024-01-01")


def test_mean_off_diagonal_correlation() -> None:
    corr = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=["A", "B"], columns=["A", "B"])
    assert mean_off_diagonal_correlation(corr) == pytest.approx(0.5)


def test_asset_weighted_correlation_matches_closed_form() -> None:
    corr = pd.DataFrame(
        [[1.0, 0.4, 0.2], [0.4, 1.0, 0.6], [0.2, 0.6, 1.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    exposures = pd.Series({"A": 100.0, "B": 300.0, "C": 600.0})
    aw = asset_weighted_correlation(corr, exposures)
    w = exposures / exposures.sum()
    expected = (
        w["A"] * w["B"] * 0.4 + w["A"] * w["C"] * 0.2 + w["B"] * w["C"] * 0.6
    )
    assert aw == pytest.approx(expected)
    # Equal weights with all off-diagonal correlations in [-1,1] keep AW in [-1,1].
    equal_weights = pd.Series({"A": 1.0, "B": 1.0, "C": 1.0})
    aw_equal = asset_weighted_correlation(corr, equal_weights)
    assert -1.0 <= aw_equal <= 1.0


def test_asset_weighted_correlation_missing_exposure_raises() -> None:
    corr = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=["A", "B"], columns=["A", "B"])
    exposures = pd.Series({"A": 100.0})
    with pytest.raises(ValueError):
        asset_weighted_correlation(corr, exposures)


def test_asset_weighted_series_uses_only_trailing_window() -> None:
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(rng.normal(size=(120, 2)), index=dates, columns=["A", "B"])
    as_of = dates[100]
    exposures = pd.DataFrame({"A": [100.0], "B": [200.0]}, index=[as_of])
    htm = pd.DataFrame({"A": [0.2], "B": [0.3]}, index=[as_of])
    sec_ratio = pd.DataFrame({"A": [0.4], "B": [0.5]}, index=[as_of])
    series = asset_weighted_series(returns, exposures, htm, sec_ratio, [as_of], window_days=60)
    assert len(series) == 1
    assert series.loc[0, "AW_HTM"] == pytest.approx(0.2 * (1 / 3) + 0.3 * (2 / 3))


def test_run_ols_recovers_known_linear_relationship() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=50)
    y = 0.5 + 2.0 * x  # noiseless: slope and intercept must be recovered exactly
    series = pd.DataFrame({"y": y, "x": x})
    result = run_ols(series, "y", "x")
    assert result["slope"] == pytest.approx(2.0, abs=1e-8)
    assert result["intercept"] == pytest.approx(0.5, abs=1e-8)
    assert result["r_squared"] > 0.999


def test_run_ols_too_few_observations_raises() -> None:
    series = pd.DataFrame({"y": [1.0, 2.0], "x": [1.0, 2.0]})
    with pytest.raises(ValueError):
        run_ols(series, "y", "x")
