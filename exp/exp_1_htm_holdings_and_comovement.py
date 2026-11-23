"""Experiment 1: HTM holdings and asset-return comovement (empirical study).

Reproduces, using FDIC Call Report balance-sheet data and daily bank stock
returns (paper, pp. 3, 11, 37-39):

1. the growth in the number of FDIC-insured institutions and in asset
   concentration among the largest banks (p. 37);
2. balance-sheet ratios (Liabilities/Assets, Deposits/Liabilities,
   Securities/Assets, HTM/Securities) averaged over the top 4, 20 and 50
   banks by assets, 2004-2023 (p. 37);
3. the pre- vs. post-COVID rise in pairwise daily stock-return correlation
   among 10 large banks (p. 11, Table 3);
4. the asset-weighted correlation measure AW and its two OLS regressions on
   the asset-weighted Securities Ratio and HTM Ratio (p. 37-39).

Usage
-----
    python -m exp.exp_1_htm_holdings_and_comovement
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data.data_loader import (
    load_active_bank_count,
    load_bank_financials,
    load_ohlcv,
    load_top_banks_by_asset,
    load_total_industry_assets,
)
from src import config
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

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
TICKER_BY_CERT = {cert: ticker for ticker, cert in config.BANKS_20.items()}


def build_concentration_table() -> pd.DataFrame:
    """Institution count and asset concentration, 2001-2023 (p. 37).

    All asset figures are Call Report ASSET values in thousands of US
    dollars, as republished by the FDIC BankFind Suite `/financials`
    endpoint.
    """
    rows = []
    for year in range(config.EXP1_COUNT_START_YEAR, config.EXP1_COUNT_END_YEAR + 1):
        rep_date = f"{year}1231"
        total_assets, n_institutions = load_total_industry_assets(rep_date)
        row = {"year": year, "n_institutions": n_institutions, "total_assets_thousands_usd": total_assets}
        for top_n in config.EXP1_TOP_N_GROUPS:
            top = load_top_banks_by_asset(rep_date, top_n, fields=[])
            top_assets = pd.to_numeric(top["ASSET"], errors="coerce").sum()
            row[f"top{top_n}_assets_thousands_usd"] = float(top_assets)
            row[f"top{top_n}_share"] = float(top_assets / total_assets)
        rows.append(row)
        print(f"concentration {year}: n_institutions={n_institutions}, "
              f"top4_share={row['top4_share']:.3f}, top50_share={row['top50_share']:.3f}")
    return pd.DataFrame(rows)


def build_balance_sheet_ratio_tables() -> dict[int, pd.DataFrame]:
    """Average balance-sheet ratios for the top 4/20/50 banks, 2004-2023 (p. 37)."""
    tables = {}
    for top_n in config.EXP1_TOP_N_GROUPS:
        ratios_by_year = {}
        for year in range(config.EXP1_RATIO_START_YEAR, config.EXP1_RATIO_END_YEAR + 1):
            rep_date = f"{year}1231"
            top = load_top_banks_by_asset(rep_date, top_n, fields=["LIAB", "DEP", "SC", "SCAGE"])
            ratios_by_year[year] = balance_sheet_ratios(top)
        tables[top_n] = top_n_average_ratios(ratios_by_year)
        print(f"balance-sheet ratios for top {top_n}: "
              f"{tables[top_n][['year', 'securities_to_assets', 'htm_to_securities']].iloc[-1].to_dict()}")
    return tables


def load_bank_returns() -> pd.DataFrame:
    """Daily simple returns of the 20 largest publicly traded U.S. banks (p. 37)."""
    bars, meta = load_ohlcv(
        list(config.BANKS_20),
        config.EXP1_STOCK_START,
        config.EXP1_SAMPLE_END,
        identity_date=config.EXP1_SAMPLE_END,
    )
    non_massive = meta.get("non_massive_source", {})
    if non_massive:
        print(f"Tickers loaded from a non-Massive source: {non_massive}")
    if meta.get("failed"):
        print(f"Tickers that failed to load: {meta['failed']}")
    return daily_returns(bars)


def build_pairwise_correlation_table(returns: pd.DataFrame) -> dict:
    """Pre- vs. post-COVID pairwise correlation among 10 large banks (p. 11, Table 3)."""
    pairwise_banks = [t for t in config.EXP1_PAIRWISE_BANKS if t in returns.columns]
    pre = pairwise_correlation(returns[pairwise_banks], config.EXP1_PRE_COVID_START, config.EXP1_PRE_COVID_END)
    post = pairwise_correlation(returns[pairwise_banks], config.EXP1_POST_COVID_START, config.EXP1_POST_COVID_END)
    pre.to_csv(RESULTS_DIR / "exp1_pairwise_correlation_pre_covid.csv")
    post.to_csv(RESULTS_DIR / "exp1_pairwise_correlation_post_covid.csv")
    return {
        "pre_covid_mean_correlation": mean_off_diagonal_correlation(pre),
        "post_covid_mean_correlation": mean_off_diagonal_correlation(post),
        "n_banks": len(pairwise_banks),
    }


def build_quarterly_financials() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Quarterly ASSET, Securities Ratio and HTM Ratio for the 20 banks (p. 37-39).

    Returns three wide tables (quarter-end date x ticker): exposures
    (ASSET), securities_ratio (SC/ASSET) and htm_ratio (SCAGE/ASSET).

    The regression on pp. 37-39 (Table 9) reports a correlation of 0.86
    between "Securities Ratio" and "HTM Ratio": both are defined relative to
    total assets so that HTM, a subset of securities, co-moves mechanically
    with total securities. This differs from Table 2's "HTM Securities
    Ratio" (SCAGE/SC, a within-securities composition measure), which is
    used only for the Table 2 descriptive statistics above.
    """
    certs = list(config.BANKS_20.values())
    quarter_ends = pd.date_range(
        f"{config.EXP1_SAMPLE_START[:4]}-03-31", config.EXP1_SAMPLE_END, freq="QE"
    )
    rows = []
    for q in quarter_ends:
        rep_date = q.strftime("%Y%m%d")
        frame = load_bank_financials(certs, [rep_date], ["ASSET", "SC", "SCAGE"])
        frame["ticker"] = frame["CERT"].map(TICKER_BY_CERT)
        frame["date"] = q
        rows.append(frame)
    panel = pd.concat(rows, ignore_index=True)
    panel["ASSET"] = pd.to_numeric(panel["ASSET"], errors="coerce")
    panel["SC"] = pd.to_numeric(panel["SC"], errors="coerce")
    panel["SCAGE"] = pd.to_numeric(panel["SCAGE"], errors="coerce")
    panel["securities_ratio"] = panel["SC"] / panel["ASSET"]
    panel["htm_ratio"] = panel["SCAGE"] / panel["ASSET"]
    exposures = panel.pivot(index="date", columns="ticker", values="ASSET")
    securities_ratio = panel.pivot(index="date", columns="ticker", values="securities_ratio")
    htm_ratio = panel.pivot(index="date", columns="ticker", values="htm_ratio")
    return exposures, securities_ratio, htm_ratio


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("1/4 Building bank-count / asset-concentration table (2001-2023, p. 37)...")
    concentration = build_concentration_table()
    concentration.to_csv(RESULTS_DIR / "exp1_bank_concentration.csv", index=False)

    print("2/4 Building balance-sheet ratio tables (top 4/20/50, 2004-2023, p. 37)...")
    ratio_tables = build_balance_sheet_ratio_tables()
    for top_n, table in ratio_tables.items():
        table.to_csv(RESULTS_DIR / f"exp1_balance_sheet_ratios_top{top_n}.csv", index=False)

    print("3/4 Loading daily bank stock returns and pairwise correlation (p. 11)...")
    returns = load_bank_returns()
    pairwise_summary = build_pairwise_correlation_table(returns)
    print(f"Pairwise mean correlation: pre-COVID={pairwise_summary['pre_covid_mean_correlation']:.3f}, "
          f"post-COVID={pairwise_summary['post_covid_mean_correlation']:.3f}")

    print("4/4 Building asset-weighted correlation series and OLS regressions (p. 37-39)...")
    exposures, securities_ratio, htm_ratio = build_quarterly_financials()
    quarter_ends = list(exposures.index)
    aw_series = asset_weighted_series(
        returns, exposures, htm_ratio, securities_ratio, quarter_ends, config.EXP1_AW_CORR_WINDOW_DAYS
    )
    aw_series.to_csv(RESULTS_DIR / "exp1_asset_weighted_series.csv", index=False)

    ols_sr = run_ols(aw_series, "AW_corr", "AW_SR")
    ols_htm = run_ols(aw_series, "AW_corr", "AW_HTM")

    summary = {
        "pairwise_correlation": pairwise_summary,
        "ols_asset_weighted_correlation_on_securities_ratio": ols_sr,
        "ols_asset_weighted_correlation_on_htm_ratio": ols_htm,
        "n_aw_quarters": len(aw_series),
    }
    with open(RESULTS_DIR / "exp1_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
