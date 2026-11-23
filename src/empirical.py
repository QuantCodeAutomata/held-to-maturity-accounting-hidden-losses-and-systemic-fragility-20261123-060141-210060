"""Experiment 1: HTM holdings and asset-return comovement (empirical study).

Implements the balance-sheet ratios, pairwise and asset-weighted stock-return
correlations, and the two OLS regressions described on pp. 3, 11 and 37-39 of
the paper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def balance_sheet_ratios(financials: pd.DataFrame) -> pd.DataFrame:
    """Balance-sheet ratios used in the top-bank tables (paper, p. 37).

    `financials` must have columns CERT, REPDTE, ASSET, LIAB, DEP, SC, SCAGE
    (FDIC Call Report fields: total assets, total liabilities, total
    deposits, total securities, securities carried at amortized cost — the
    Call Report proxy for held-to-maturity securities). Returns one row per
    (CERT, REPDTE) with the four ratios the paper reports.
    """
    required = {"CERT", "REPDTE", "ASSET", "LIAB", "DEP", "SC", "SCAGE"}
    missing = required - set(financials.columns)
    if missing:
        raise ValueError(f"balance_sheet_ratios: missing columns {missing}")
    out = financials.copy()
    out["liabilities_to_assets"] = out["LIAB"] / out["ASSET"]
    out["deposits_to_liabilities"] = out["DEP"] / out["LIAB"]
    out["securities_to_assets"] = out["SC"] / out["ASSET"]
    out["htm_to_securities"] = out["SCAGE"] / out["SC"]
    return out


def top_n_average_ratios(ratios_by_year: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Average the four balance-sheet ratios across the banks in each year's top-N table."""
    rows = []
    for year, frame in ratios_by_year.items():
        rows.append({
            "year": year,
            "liabilities_to_assets": frame["liabilities_to_assets"].mean(),
            "deposits_to_liabilities": frame["deposits_to_liabilities"].mean(),
            "securities_to_assets": frame["securities_to_assets"].mean(),
            "htm_to_securities": frame["htm_to_securities"].mean(),
            "n_banks": len(frame),
        })
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def daily_returns(bars: pd.DataFrame) -> pd.DataFrame:
    """Wide table of daily simple returns, one column per ticker, indexed by date.

    `bars` is the tidy long table returned by `data.data_loader.load_ohlcv`
    (columns: date, ticker, close, ...).
    """
    wide_close = bars.pivot(index="date", columns="ticker", values="close").sort_index()
    return wide_close.pct_change().dropna(how="all")


def pairwise_correlation(returns: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Pairwise correlation matrix of daily returns within [start, end] (paper Table 3, p. 11)."""
    window = returns.loc[start:end]
    if window.shape[0] < 2:
        raise ValueError(f"pairwise_correlation: fewer than 2 observations in [{start}, {end}]")
    return window.corr()


def mean_off_diagonal_correlation(corr: pd.DataFrame) -> float:
    """Average of the off-diagonal entries of a correlation matrix."""
    values = corr.to_numpy()
    mask = ~np.eye(values.shape[0], dtype=bool)
    return float(values[mask].mean())


def asset_weighted_correlation(corr: pd.DataFrame, exposures: pd.Series) -> float:
    """Asset-weighted correlation AW = sum_{i<j} w_i w_j rho_ij (paper Eq., p. 37).

    `exposures` holds each bank's asset size E_i, indexed like `corr`'s rows.
    Weights are w_i = E_i / sum_j E_j.
    """
    tickers = corr.index
    exposures = exposures.reindex(tickers)
    if exposures.isna().any():
        raise ValueError("asset_weighted_correlation: exposures missing for some tickers")
    weights = exposures / exposures.sum()
    total = 0.0
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            total += weights.iloc[i] * weights.iloc[j] * corr.iloc[i, j]
    return float(total)


def asset_weighted_series(
    returns: pd.DataFrame,
    exposures_by_date: pd.DataFrame,
    htm_ratio_by_date: pd.DataFrame,
    securities_ratio_by_date: pd.DataFrame,
    quarter_ends: list[pd.Timestamp],
    window_days: int,
) -> pd.DataFrame:
    """Quarterly asset-weighted correlation, Securities Ratio and HTM Ratio series.

    For each date in `quarter_ends`, uses the trailing `window_days` daily
    returns to build the pairwise correlation matrix, and the balance-sheet
    ratios/exposures as of that quarter (nearest available Call Report row)
    to build AW, AW_SR and AW_HTM, matching the construction described on
    pp. 37-39.
    """
    rows = []
    for as_of in quarter_ends:
        window = returns.loc[:as_of].tail(window_days)
        if window.shape[0] < window_days // 2:
            continue
        tickers = [t for t in window.columns if window[t].notna().sum() >= window_days // 2]
        window = window[tickers].dropna()
        if window.shape[0] < 20 or len(tickers) < 2:
            continue
        corr = window.corr()
        exposures = exposures_by_date.loc[as_of, tickers] if as_of in exposures_by_date.index else None
        if exposures is None or exposures.isna().any():
            continue
        weights = exposures / exposures.sum()
        htm = htm_ratio_by_date.loc[as_of, tickers]
        sec_ratio = securities_ratio_by_date.loc[as_of, tickers]
        aw_corr = asset_weighted_correlation(corr, exposures)
        aw_htm = float((weights * htm).sum())
        aw_sr = float((weights * sec_ratio).sum())
        rows.append({"date": as_of, "AW_corr": aw_corr, "AW_HTM": aw_htm, "AW_SR": aw_sr})
    return pd.DataFrame(rows)


def run_ols(series: pd.DataFrame, y_col: str, x_col: str) -> dict:
    """Simple OLS of `y_col` on `x_col` (paper, p. 37-39: AW ~ AW_SR and AW ~ AW_HTM).

    Returns the intercept, slope, standard errors, t-statistics, p-values and
    R-squared, as produced by `statsmodels.api.OLS`.
    """
    clean = series[[y_col, x_col]].dropna()
    if clean.shape[0] < 3:
        raise ValueError(f"run_ols: fewer than 3 observations for {y_col} ~ {x_col}")
    x = sm.add_constant(clean[x_col])
    model = sm.OLS(clean[y_col], x).fit()
    return {
        "n_obs": int(model.nobs),
        "intercept": float(model.params["const"]),
        "slope": float(model.params[x_col]),
        "intercept_se": float(model.bse["const"]),
        "slope_se": float(model.bse[x_col]),
        "intercept_t": float(model.tvalues["const"]),
        "slope_t": float(model.tvalues[x_col]),
        "slope_pvalue": float(model.pvalues[x_col]),
        "r_squared": float(model.rsquared),
    }
