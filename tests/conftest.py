"""Shared pytest fixtures and marker registration."""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "market_data: tests that load real data through data/data_loader.py"
    )


@pytest.fixture(scope="session")
def bank_financials_slice(request: pytest.FixtureRequest):
    """A few banks' Call Report rows for two report dates, cached across runs."""
    if not os.environ.get("DATALOADER_API_KEY") and not os.environ.get("MASSIVE_API_KEY"):
        pytest.skip("DATALOADER_API_KEY/MASSIVE_API_KEY not set")
    cache_dir = request.config.cache.mkdir("htm_financials_slice")
    cache_file = cache_dir / "financials.json"
    import pandas as pd

    if cache_file.exists():
        return pd.read_json(cache_file)

    from data.data_loader import load_bank_financials

    frame = load_bank_financials(
        certs=[628, 3510, 7213],
        rep_dates=["20221231", "20231231"],
        fields=["ASSET", "LIAB", "DEP", "SC", "SCAGE"],
    )
    frame.to_json(cache_file)
    return frame


@pytest.fixture(scope="session")
def bank_bars_slice(request: pytest.FixtureRequest):
    """A short daily-bar slice for two bank tickers, cached across runs."""
    if not os.environ.get("DATALOADER_API_KEY") or not os.environ.get("MASSIVE_API_KEY"):
        pytest.skip("DATALOADER_API_KEY/MASSIVE_API_KEY not set")
    cache_dir = request.config.cache.mkdir("htm_bars_slice")
    cache_file = cache_dir / "bars.json"
    import pandas as pd

    if cache_file.exists():
        bars = pd.read_json(cache_file)
        bars["date"] = pd.to_datetime(bars["date"])
        return bars

    from data.data_loader import load_ohlcv

    bars, _ = load_ohlcv(["JPM", "BAC"], "2022-01-01", "2022-03-01", use_cache=False)
    bars.to_json(cache_file)
    return bars
