# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Disclaimer:
# Each user is responsible for checking the content of datasets and the
# applicable licenses and determining if suitable for the intended use.

"""
Download S&P 500 price-volume data using yfinance.

Changes from the original script, and why
-----------------------------------------
* **Batched, retried downloads.** One 392-ticker request makes Yahoo throttle
  and silently return empty columns, and ``dropna(axis=1, how="all")`` then
  deletes them without saying which. Tickers are fetched in batches with
  exponential backoff and a coverage report.
* **The default end date is today, not a date in the past.** The old default of
  2025-12-31 meant every run after that quietly trained on stale data.
* **A cache.** Repeated runs read the local CSVs and only fetch the missing
  tail instead of re-downloading fifteen years of history.
* **Survivorship is stated, not hidden.** The ticker list is today's index
  membership applied backwards. That is a real upward bias for momentum and
  quality, and it is printed on every run. Use the hedge-fund CRSP adapter
  (``phf_bridge.py``) when you need a survivorship-free universe.
* **Fundamentals are not fetched here.** The fork advertised 23 fields while
  only downloading 5. Fundamental and ratio panels come from the WRDS/CRSP
  lake in Cloudflare R2 via ``phf_bridge.py``; this script is the fallback
  price path only.

Usage:
    python -m signal_discovery_workflow.download_data
    python -m signal_discovery_workflow.download_data --start 2012-01-01 --end 2026-11-22
"""

import argparse
import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "data" / "sp500"
DATA_FIELDS = ["Open", "Close", "High", "Low", "Volume"]
BATCH_SIZE = 50
MAX_RETRIES = 3

# Tickers in the universe list that have been renamed. Yahoo only serves the
# current symbol, so request that and rename the column back; otherwise the
# history for these names is silently missing (which is itself a bias: it drops
# three long-lived companies from every cross-section).
TICKER_ALIASES = {
    "BK": "BNY",
    "FI": "FISV",
    "MMC": "MRSH",
}


SP500_TICKERS = [
    'A', 'AAPL', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG',
    'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALL', 'AMAT', 'AMD', 'AME', 'AMGN', 'AMT', 'AMZN', 'AON', 'AOS', 'APA',
    'APD', 'APH', 'ARE', 'ATO', 'AVB', 'AVY', 'AXON', 'AXP', 'AZO',
    'BA', 'BAC', 'BALL', 'BAX', 'BBWI', 'BBY', 'BDX', 'BEN', 'BG', 'BIIB', 'BIO', 'BK', 'BKNG', 'BKR', 'BLK',
    'BMY', 'BRO', 'BSX', 'BWA', 'BXP',
    'C', 'CAG', 'CAH', 'CAT', 'CB', 'CBRE', 'CCI', 'CCL', 'CDNS', 'CHD', 'CHRW', 'CI', 'CINF', 'CL', 'CLX', 'CMA',
    'CMCSA', 'CME', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COO', 'COP', 'COR', 'COST', 'CPB', 'CPRT', 'CPT', 'CRL',
    'CRM', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTRA', 'CTSH', 'CVS', 'CVX',
    'D', 'DD', 'DE', 'DECK', 'DGX', 'DHI', 'DHR', 'DIS', 'DLR', 'DLTR', 'DOC', 'DOV', 'DPZ', 'DRI', 'DTE', 'DUK',
    'DVA', 'DVN',
    'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EG', 'EIX', 'EL', 'ELV', 'EMN', 'EMR', 'EOG', 'EQIX', 'EQR', 'EQT', 'ES',
    'ESS', 'ETN', 'ETR', 'EVRG', 'EW', 'EXC', 'EXPD', 'EXR',
    'F', 'FAST', 'FCX', 'FDS', 'FDX', 'FE', 'FFIV', 'FI', 'FICO', 'FIS', 'FITB', 'FMC', 'FRT',
    'GD', 'GE', 'GEN', 'GILD', 'GIS', 'GL', 'GLW', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW',
    'HAL', 'HAS', 'HBAN', 'HD', 'HIG', 'HOLX', 'HON', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUBB', 'HUM',
    'IBM', 'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY', 'INTC', 'INTU', 'IP', 'IPG', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ',
    'J', 'JBHT', 'JBL', 'JCI', 'JKHY', 'JNJ', 'JPM',
    'K', 'KEY', 'KIM', 'KLAC', 'KMB', 'KMX', 'KO', 'KR',
    'L', 'LEN', 'LH', 'LHX', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNT', 'LOW', 'LRCX', 'LUV', 'LVS',
    'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'MGM', 'MHK', 'MKC', 'MKTX', 'MLM',
    'MMC', 'MMM', 'MNST', 'MO', 'MOH', 'MOS', 'MPWR', 'MRK', 'MS', 'MSFT', 'MSI', 'MTB', 'MTCH', 'MTD', 'MU',
    'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR',
    'O', 'ODFL', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OXY',
    'PAYX', 'PCAR', 'PCG', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PLD', 'PNC', 'PNR',
    'PNW', 'POOL', 'PPG', 'PPL', 'PRU', 'PSA', 'PTC', 'PWR', 'QCOM',
    'RCL', 'REG', 'REGN', 'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'RVTY',
    'SBAC', 'SBUX', 'SCHW', 'SHW', 'SJM', 'SLB', 'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'SRE', 'STE', 'STLD', 'STT',
    'STX', 'STZ', 'SWK', 'SWKS', 'SYK', 'SYY',
    'T', 'TAP', 'TDY', 'TECH', 'TER', 'TFC', 'TFX', 'TGT', 'TJX', 'TMO', 'TPR', 'TRMB', 'TROW', 'TRV', 'TSCO',
    'TSN', 'TT', 'TTWO', 'TXN', 'TXT', 'TYL',
    'UDR', 'UHS', 'UNH', 'UNP', 'UPS', 'URI', 'USB',
    'VLO', 'VMC', 'VRSN', 'VRTX', 'VTR', 'VTRS', 'VZ',
    'WAB', 'WAT', 'WDC', 'WEC', 'WELL', 'WFC', 'WM', 'WMB', 'WMT', 'WRB', 'WST', 'WTW', 'WY', 'WYNN',
    'XEL', 'XOM', 'YUM', 'ZBH', 'ZBRA',
]


def get_sp500_tickers() -> list[str]:
    """Return the configured S&P 500 ticker universe."""
    logger.info(f"Using {len(SP500_TICKERS)} S&P 500 tickers")
    return SP500_TICKERS


def _today() -> str:
    return date.today().isoformat()


def _normalize_columns(raw: pd.DataFrame, batch: list[str]) -> pd.DataFrame:
    """Return a (field, ticker) column frame regardless of yfinance's layout."""
    if isinstance(raw.columns, pd.MultiIndex):
        if raw.columns.names and raw.columns.names[0] in DATA_FIELDS:
            return raw
        if raw.columns.names and raw.columns.names[1] in DATA_FIELDS:
            return raw.swaplevel(0, 1, axis=1).sort_index(axis=1)
        return raw

    # Single ticker: yfinance returns flat field columns.
    if len(batch) == 1:
        frame = raw.copy()
        frame.columns = pd.MultiIndex.from_product(
            [frame.columns, [batch[0]]], names=[None, None]
        )
        return frame
    return raw


def _download_batch(batch: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Download one batch with retries, returning field -> wide frame."""
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = yf.download(
                batch,
                start=start,
                end=end,
                group_by="column",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
            if raw is None or raw.empty:
                raise RuntimeError("empty response")
            raw = _normalize_columns(raw, batch)
            out: dict[str, pd.DataFrame] = {}
            level0 = raw.columns.get_level_values(0)
            for field in DATA_FIELDS:
                if field not in level0:
                    continue
                df = raw[field]
                if isinstance(df, pd.Series):
                    df = df.to_frame(name=batch[0])
                # Yahoo returns rows for non-trading days as all-NaN.
                df = df.dropna(how="all")
                if not df.empty:
                    out[field] = df
            if out:
                return out
            raise RuntimeError("no fields in response")
        except Exception as e:
            last_error = e
            wait = 2 ** attempt
            logger.warning(f"Batch failed (attempt {attempt}/{MAX_RETRIES}): {e}; retrying in {wait}s")
            time.sleep(wait)
    logger.error(f"Batch permanently failed: {last_error}")
    return {}


def _existing_panel(output_dir: Path, field: str) -> pd.DataFrame | None:
    path = output_dir / f"{field}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df if not df.empty else None
    except Exception:
        return None


def download_sp500_data(
    start: str = "2012-01-01",
    end: str | None = None,
    output_dir: str | Path | None = None,
    batch_size: int = BATCH_SIZE,
    tickers: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Download S&P 500 price-volume data and save one CSV per field.

    Existing CSVs are reused: only the window after the last cached date is
    fetched and appended, so re-running is cheap.
    """
    output_path = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)
    end = end or (date.today() + timedelta(days=1)).isoformat()
    tickers = tickers or get_sp500_tickers()

    logger.warning(
        "Universe is TODAY'S S&P 500 membership applied backwards. This is "
        "survivorship-biased (winners that survived to today), which flatters "
        "momentum, quality and low-volatility signals. For a survivorship-free "
        "test use the hedge-fund CRSP panel via phf_bridge.py."
    )

    cached = {field: _existing_panel(output_path, field) for field in DATA_FIELDS}
    cached_close = cached.get("Close")
    fetch_start = start
    if cached_close is not None and not cached_close.empty:
        last_cached = cached_close.index.max()
        candidate = (last_cached + pd.Timedelta(days=1)).date().isoformat()
        if candidate > start:
            fetch_start = candidate
            logger.info(f"Cache ends {last_cached.date()}; fetching from {fetch_start}")

    collected: dict[str, list[pd.DataFrame]] = {field: [] for field in DATA_FIELDS}
    if pd.Timestamp(fetch_start) < pd.Timestamp(end):
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            logger.info(f"Downloading {i + 1}-{i + len(batch)} of {len(tickers)}...")
            fields = _download_batch(batch, fetch_start, end)
            for field, df in fields.items():
                collected[field].append(df)

    # Recover renamed tickers under their old symbols.
    missing_before_alias = [
        t for t in tickers
        if not collected.get("Close") or t not in pd.concat(collected["Close"], axis=1).columns
    ] if collected.get("Close") else []
    alias_targets = {old: new for old, new in TICKER_ALIASES.items() if old in missing_before_alias}
    if alias_targets:
        logger.info(f"Retrying renamed tickers through their current symbols: {alias_targets}")
        fields = _download_batch(list(alias_targets.values()), fetch_start, end)
        for field, df in fields.items():
            rename = {new: old for old, new in alias_targets.items() if new in df.columns}
            if rename:
                collected.setdefault(field, []).append(df.rename(columns=rename)[list(rename.values())])

    results: dict[str, pd.DataFrame] = {}
    for field in DATA_FIELDS:
        parts = []
        if cached.get(field) is not None:
            parts.append(cached[field])
        if collected[field]:
            parts.append(pd.concat(collected[field], axis=1))
        if not parts:
            logger.warning(f"No data for field '{field}'")
            continue

        df = pd.concat(parts, axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df = df.dropna(axis=1, how="all")

        df.index.name = "Date"
        df.to_csv(output_path / f"{field}.csv")
        results[field] = df
        coverage = df.notna().any(axis=0).sum()
        logger.info(
            f"Saved {field}.csv shape={df.shape}, {coverage}/{len(tickers)} tickers with data, "
            f"{df.index.min().date()}..{df.index.max().date()}"
        )

    if "Close" in results:
        close = results["Close"]
        missing = sorted(set(tickers) - set(close.columns))
        meta = {
            "downloaded_at": datetime.now().isoformat(timespec="seconds"),
            "requested_start": start,
            "requested_end": end,
            "fetched_from": fetch_start,
            "tickers_requested": len(tickers),
            "tickers_with_close": int(close.shape[1]),
            "missing_tickers": missing,
            "rows": int(close.shape[0]),
            "first_date": str(close.index.min().date()),
            "last_date": str(close.index.max().date()),
            "adjusted": True,
            "survivorship_bias": (
                "Ticker list is current index membership applied historically. "
                "Not a point-in-time universe."
            ),
        }
        with open(output_path / "_download_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        if missing:
            logger.warning(f"{len(missing)} tickers had no data: {missing[:20]}...")

    logger.info(f"Data saved to {output_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download S&P 500 price-volume and point-in-time fundamental data."
    )
    parser.add_argument("--start", default="2012-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD). Default: tomorrow")
    parser.add_argument("--output", default=None, help="Output directory. Default: data/sp500/")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Tickers per Yahoo request")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only use the first N tickers (a quick smoke test)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    tickers = SP500_TICKERS[: args.limit] if args.limit else None
    download_sp500_data(
        start=args.start,
        end=args.end,
        output_dir=args.output,
        batch_size=args.batch_size,
        tickers=tickers,
    )


if __name__ == "__main__":
    main()
