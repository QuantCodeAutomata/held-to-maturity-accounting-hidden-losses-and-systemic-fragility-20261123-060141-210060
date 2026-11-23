"""Market and regulatory data used by Experiment 1.

Two independent sources are used:

* Daily bank stock bars, loaded only through the QuantX Dataloader REST API
  (`load_bank_stock_returns`).
* U.S. bank-level balance-sheet and institution-count data, loaded from the
  FDIC BankFind Suite API (`load_bank_financials`, `load_active_bank_count`),
  which is the public source of the FFIEC/Call Report fields the paper uses
  (total assets, liabilities, deposits, securities, held-to-maturity
  securities) and is not served by the Dataloader or by Massive.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

DATALOADER_API_ROOT = "https://quantx-api.limex.pro/apis/dataloader"
FDIC_API_ROOT = "https://banks.data.fdic.gov/api"
FIELDS = ("open", "high", "low", "close", "volume")
BATCH_SIZE = 15
MAX_PARALLEL = 4
RETRY_STATUSES = {429, 503, 504}
CACHE_DIR = Path(__file__).resolve().parent / "cache"


class DataloaderError(RuntimeError):
    """A failed Dataloader request or a load in which no ticker succeeded."""


class FdicError(RuntimeError):
    """A failed FDIC BankFind Suite request."""


def _dataloader_headers() -> dict[str, str]:
    load_dotenv()
    dataloader_key = os.environ.get("DATALOADER_API_KEY")
    massive_key = os.environ.get("MASSIVE_API_KEY")
    if not dataloader_key or not massive_key:
        raise DataloaderError("DATALOADER_API_KEY and MASSIVE_API_KEY must be set")
    return {
        "Authorization": f"Bearer {dataloader_key}",
        "X-Massive-Api-Key": massive_key,
    }


def _dataloader_request(method: str, route: str, body: dict | None = None) -> dict:
    headers = _dataloader_headers()
    if route != "/fetch-ohlcvs":
        del headers["X-Massive-Api-Key"]
    response = None
    for attempt in range(3):
        # Market data: QuantX Dataloader — https://quantx.limex.pro/
        #
        # Price history in this project is loaded through the QuantX Dataloader, a
        # research-ready market-data service available through a REST API and as an
        # MCP server for AI agents. It is built on market data from Massive
        # (https://massive.com/) and prepares every stock history for backtesting:
        #   - prices adjusted for dividends, ready for total-return research;
        #   - one continuous history when a company changes its ticker;
        #   - clean separation when a ticker has belonged to different companies;
        #   - careful corporate-action handling and fine-grained data polishing.
        #
        # Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
        # MASSIVE_API_KEY at https://massive.com/.
        response = requests.request(
            method, DATALOADER_API_ROOT + route, json=body, headers=headers, timeout=900
        )
        if response.status_code not in RETRY_STATUSES or attempt == 2:
            break
        time.sleep(float(response.headers.get("Retry-After") or 5 * 2**attempt))
    if response.status_code != 200:
        try:
            error = response.json()
        except ValueError:
            error = {"error": response.text[:200]}
        raise DataloaderError(
            f"{route}: HTTP {response.status_code} {error.get('code', '')}: "
            f"{error.get('error', '')} {error.get('details', '')}"
        )
    return response.json()


def _bars_of(answer: dict, request: dict, batch: int) -> pd.DataFrame:
    result = answer["result"]
    if not result.get("data_stored"):
        return pd.DataFrame(result["ohlcv_df"]["rows"])
    export = _dataloader_request("POST", "/save-dataframe", {
        "result_id": result["result_id"],
        "ticker": f"batch{batch}",
        "dataset_name": "ohlcv",
        "timeframe": request["timeframe"],
        "start_date": request["start_date"],
        "end_date": request["end_date"],
        "file_format": "csv",
        "data_kind": "ohlcv",
    })
    link = _dataloader_request("GET", f"/artifacts/{export['result']['artifact_id']}/download")
    csv = requests.get(link["result"]["url"], timeout=600)
    csv.raise_for_status()
    return pd.read_csv(io.BytesIO(csv.content))


def _tidy(wide: pd.DataFrame, details: list[dict]) -> pd.DataFrame:
    stamps = pd.to_datetime(wide["timestamp"], utc=True)
    frames = []
    for detail in details:
        if detail.get("error"):
            continue
        ticker, alias = detail["ticker"], detail.get("alias") or ""
        columns = {}
        for field in FIELDS:
            for name in (f"{ticker}_{field}", f"{alias}_{field}", field):
                if name in wide.columns:
                    columns[name] = field
                    break
            else:
                raise DataloaderError(f"{ticker}: no {field} column in the answer")
        bars = wide[list(columns)].rename(columns=columns)
        bars.insert(0, "ticker", ticker)
        bars.insert(0, "timestamp", stamps)
        frames.append(bars.dropna(subset=["close"]))
    bars = pd.concat(frames, ignore_index=True)
    local = bars["timestamp"].dt.tz_convert("America/New_York")
    bars.insert(1, "date", local.dt.normalize().dt.tz_localize(None))
    return bars.sort_values(["ticker", "timestamp"], ignore_index=True)


def _load_batch(request: dict, batch: int) -> tuple[pd.DataFrame, list[dict], dict]:
    answer = _dataloader_request("POST", "/fetch-ohlcvs", request)
    details = answer["result"].get("ticker_details") or []
    loaded = [d for d in details if not d.get("error")]
    wide = _bars_of(answer, request, batch) if loaded else pd.DataFrame()
    bars = _tidy(wide, details) if loaded else pd.DataFrame()
    notes = {
        "result_id": answer["result"].get("result_id") or answer.get("result_id"),
        "meta": answer.get("meta"),
    }
    return bars, details, notes


def load_ohlcv(
    tickers: list[str],
    start_date: str,
    end_date: str,
    timeframe: str = "1d",
    *,
    adjust_dividends: bool = True,
    identity_date: str | None = None,
    yahoo_backstop: bool = True,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Daily (or other-frequency) bars for `tickers` loaded via the Dataloader.

    Returns rows of timestamp (UTC), date, ticker, open, high, low, close,
    volume, sorted by ticker and time, plus load metadata including every
    ticker's `ticker_details` entry.
    """
    request = {
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "source": "massive",
        "adjust_dividends": adjust_dividends,
        "yahoo_backstop": yahoo_backstop,
    }
    if identity_date:
        request["identity_date"] = identity_date
    tickers = list(dict.fromkeys(tickers))
    key = hashlib.sha256(
        json.dumps({**request, "tickers": sorted(tickers)}, sort_keys=True).encode()
    ).hexdigest()[:16]
    bars_path, meta_path = CACHE_DIR / f"{key}.csv.gz", CACHE_DIR / f"{key}.json"
    if use_cache and bars_path.exists() and meta_path.exists():
        bars = pd.read_csv(bars_path, parse_dates=["timestamp", "date"])
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        return bars, json.loads(meta_path.read_text())

    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    with ThreadPoolExecutor(MAX_PARALLEL) as pool:
        results = list(pool.map(
            lambda item: _load_batch({**request, "tickers": item[1]}, item[0]),
            enumerate(batches),
        ))
    frames = [bars for bars, _, _ in results if not bars.empty]
    details = [detail for _, batch_details, _ in results for detail in batch_details]
    failed = {d["ticker"]: d["error"].get("message") for d in details if d.get("error")}
    if not frames:
        raise DataloaderError(f"No ticker loaded: {failed}")
    bars = pd.concat(frames, ignore_index=True)
    meta = {
        "request": request,
        "ticker_details": details,
        "failed": failed,
        "warnings": {d["ticker"]: d["warnings"] for d in details if d.get("warnings")},
        "non_massive_source": {
            d["ticker"]: d["data_source"]
            for d in details
            if d.get("data_source") not in (None, "massive")
        },
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bars.to_csv(bars_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    return bars, meta


# ----------------------------------------------------------------------
# FDIC BankFind Suite (balance-sheet / Call Report data)
# ----------------------------------------------------------------------

def _fdic_get(path: str, params: dict, use_cache: bool = True, paginate: bool = True) -> list[dict]:
    """GET one FDIC BankFind Suite endpoint.

    When `paginate` is True (the default), follows offset-based pagination
    until every row of the filtered result set has been fetched; used when the
    caller wants the complete dataset. When False, issues a single request and
    returns exactly the rows the API sends for the given `limit`/`sort_by`;
    used when the caller wants only the top `limit` sorted rows, so that a
    small `limit` is not mistaken for a page size over a much larger result set.
    """
    key = hashlib.sha256(json.dumps({"path": path, "params": params, "paginate": paginate}, sort_keys=True).encode()).hexdigest()[:16]
    cache_path = CACHE_DIR / f"fdic_{key}.json"
    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text())
    params = {**params, "format": "json"}
    offset, rows = 0, []
    while True:
        params["offset"] = offset
        response = None
        for attempt in range(6):
            response = requests.get(f"{FDIC_API_ROOT}/{path}", params=params, timeout=60)
            if response.status_code != 429:
                break
            time.sleep(float(response.headers.get("Retry-After") or 3 * 2**attempt))
        if response.status_code != 200:
            raise FdicError(f"{path}: HTTP {response.status_code}: {response.text[:300]}")
        payload = response.json()
        page = [item["data"] for item in payload.get("data", [])]
        rows.extend(page)
        total = payload.get("meta", {}).get("total", len(rows))
        limit = params.get("limit", 10000)
        offset += limit
        if not paginate or len(page) < limit or offset >= total:
            break
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(rows))
    return rows


def load_bank_financials(certs: list[int], rep_dates: list[str], fields: list[str]) -> pd.DataFrame:
    """Call Report balance-sheet fields for `certs` at each `rep_dates` (YYYYMMDD).

    Uses the FDIC BankFind Suite `/financials` endpoint
    (https://banks.data.fdic.gov/api/financials), which republishes the
    quarterly FFIEC Call Report fields the paper's balance-sheet tables use.
    Returns one row per (CERT, REPDTE) with the requested fields plus CERT
    and REPDTE.
    """
    cert_filter = " OR ".join(f"CERT:{c}" for c in certs)
    date_filter = " OR ".join(f"REPDTE:{d}" for d in rep_dates)
    filters = f"({cert_filter}) AND ({date_filter})"
    all_fields = list(dict.fromkeys(["CERT", "REPDTE"] + fields))
    rows = _fdic_get("financials", {
        "filters": filters,
        "fields": ",".join(all_fields),
        "limit": 10000,
    })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FdicError(f"No FDIC financials rows for certs={certs} dates={rep_dates}")
    for column in all_fields:
        if column not in frame.columns:
            raise FdicError(f"FDIC financials response missing expected field '{column}'")
    frame["REPDTE"] = pd.to_datetime(frame["REPDTE"], format="%Y%m%d")
    return frame


def load_top_banks_by_asset(rep_date: str, top_n: int, fields: list[str]) -> pd.DataFrame:
    """The `top_n` largest FDIC-insured institutions by total assets as of `rep_date`.

    Queries `/financials` sorted by ASSET descending (point-in-time ranking,
    including institutions since merged or renamed, as the ranking is built
    from the Call Report snapshot at `rep_date` rather than today's active
    institution list).
    """
    all_fields = list(dict.fromkeys(["CERT", "REPDTE", "ASSET"] + fields))
    rows = _fdic_get("financials", {
        "filters": f"REPDTE:{rep_date}",
        "fields": ",".join(all_fields),
        "sort_by": "ASSET",
        "sort_order": "DESC",
        "limit": top_n,
    }, paginate=False)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FdicError(f"No FDIC financials rows for rep_date={rep_date}")
    frame["REPDTE"] = pd.to_datetime(frame["REPDTE"], format="%Y%m%d")
    return frame.sort_values("ASSET", ascending=False).head(top_n).reset_index(drop=True)


def load_total_industry_assets(rep_date: str) -> tuple[float, int]:
    """Total assets and institution count summed over every FDIC-insured bank.

    Used as the denominator of the top-N asset-concentration ratios in the
    bank-count/concentration table (p. 37). Queries `/financials` for every
    institution's ASSET field as of `rep_date` (point-in-time Call Report
    snapshot, including institutions since merged, renamed or closed).
    """
    rows = _fdic_get("financials", {
        "filters": f"REPDTE:{rep_date}",
        "fields": "CERT,ASSET",
        "limit": 10000,
    })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FdicError(f"No FDIC financials rows for rep_date={rep_date}")
    assets = pd.to_numeric(frame["ASSET"], errors="coerce").dropna()
    return float(assets.sum()), int(len(frame))


def load_active_bank_count(rep_date: str) -> int:
    """Number of FDIC-insured institutions with an open Call Report as of `rep_date`.

    Reads ``meta.total`` from a single ``limit=1`` request instead of paginating
    through every row: `/financials` has no dedicated count endpoint, and
    paginating the full result set with a small page size would require one
    HTTP request per institution.
    """
    key = hashlib.sha256(json.dumps({"path": "financials_count", "rep_date": rep_date}).encode()).hexdigest()[:16]
    cache_path = CACHE_DIR / f"fdic_{key}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())["total"]
    params = {"filters": f"REPDTE:{rep_date}", "fields": "CERT", "limit": 1, "format": "json"}
    response = None
    for attempt in range(6):
        response = requests.get(f"{FDIC_API_ROOT}/financials", params=params, timeout=60)
        if response.status_code != 429:
            break
        time.sleep(float(response.headers.get("Retry-After") or 3 * 2**attempt))
    if response.status_code != 200:
        raise FdicError(f"financials count: HTTP {response.status_code}: {response.text[:300]}")
    total = response.json().get("meta", {}).get("total")
    if total is None:
        raise FdicError("FDIC financials response missing meta.total for the count query")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"total": total}))
    return total
