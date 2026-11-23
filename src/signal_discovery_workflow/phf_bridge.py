# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0

"""
Bridge to the personal hedge fund's release-pinned CRSP panel.

The Yahoo path in ``download_data.py`` has two structural limits that no amount
of prompt engineering fixes:

* it uses today's S&P 500 membership applied backwards (survivorship bias), and
* it has no delisting returns, so the worst outcomes silently disappear.

The hedge fund already has the correct data: ``packages/phf-research`` publishes
a monthly CRSP panel (26k securities, 1962-2024, delisting-adjusted
``ret_total``) with release pins. This module turns that panel into the wide
field panels the evaluator expects, **in memory, read in place from R2**.

    python -m signal_discovery_workflow.phf_bridge \
      --panel r2://financial-data-wrds-private/<key> --check

It is read-only: no R2 writes, no release-pointer changes, no orders.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import pandas as pd

from . import r2_store

logger = logging.getLogger(__name__)

# Coarse SIC division -> sector label. SIC is not GICS, but it is what CRSP
# carries; labels are prefixed "SIC:" so nobody mistakes them for GICS.
_SIC_DIVISIONS = [
    (100, 999, "SIC:Agriculture"),
    (1000, 1499, "SIC:Mining and Energy"),
    (1500, 1799, "SIC:Construction"),
    (2000, 3999, "SIC:Manufacturing"),
    (4000, 4899, "SIC:Transportation"),
    (4900, 4999, "SIC:Utilities"),
    (5000, 5199, "SIC:Wholesale"),
    (5200, 5999, "SIC:Retail"),
    (6000, 6799, "SIC:Finance"),
    (7000, 8999, "SIC:Services"),
    (9100, 9999, "SIC:Public Administration"),
]

PANEL_COLUMNS = [
    "date", "security_id", "ret_total", "dollar_volume",
    "siccd", "investable", "market_equity", "market_cap", "prc", "shrout",
]


def sic_to_sector(sic: Any) -> str:
    try:
        code = int(float(sic))
    except (TypeError, ValueError):
        return "Unknown"
    for low, high, label in _SIC_DIVISIONS:
        if low <= code <= high:
            return label
    return "Unknown"


def load_phf_panel(
    panel_uri: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Load the CRSP panel in memory and return ``(field_panels, sector_map)``.

    Required columns: ``date``, ``security_id``, ``ret_total``. Optional:
    ``dollar_volume`` and ``siccd``.

    Panels are ``Close`` (a total-return index per security) and ``Volume``
    (dollar volume). ``Close`` is an index level, not a price: that is what the
    return series supports, and TS_ operators are valid on it.

    Sector is mapped from SIC. Identifier is the stable permno-based
    ``security_id``, never the ticker: tickers are reused across companies over
    60 years and would merge unrelated firms.
    """
    if not panel_uri:
        raise FileNotFoundError(
            "No panel configured. Pass --panel r2://bucket/key, set PHF_PANEL_PATH, "
            "or set QSD_DATA_SOURCE. The panel is read in place from R2."
        )

    available = r2_store.parquet_schema(panel_uri)
    missing = {"date", "security_id", "ret_total"} - available
    if missing:
        raise ValueError(f"Panel is missing required columns: {sorted(missing)}")

    frame = r2_store.read_parquet(
        panel_uri, [c for c in PANEL_COLUMNS if c in available]
    )

    frame = frame[frame["security_id"].notna()]
    frame["security_id"] = frame["security_id"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"])
    if start:
        frame = frame[frame["date"] >= pd.Timestamp(start)]
    if end:
        frame = frame[frame["date"] <= pd.Timestamp(end)]

    frame = frame.sort_values("date").drop_duplicates(
        ["security_id", "date"], keep="last"
    )
    if "investable" in frame.columns:
        eligible = frame["investable"].fillna(1).astype(int) == 1
        dropped = int((~eligible).sum())
        if dropped:
            logger.info(f"Dropping {dropped} non-investable security-months")
        frame = frame[eligible]

    returns = frame.pivot(index="date", columns="security_id", values="ret_total")
    # A missing CRSP return means "no observation", not zero. Carry the index
    # forward so the level stays continuous; Returns itself stays NaN and is
    # reported by the coverage block.
    panels: dict[str, pd.DataFrame] = {
        "Close": (1.0 + returns.fillna(0.0)).cumprod(),
        "Returns": returns,
    }
    if "dollar_volume" in frame.columns:
        panels["Volume"] = frame.pivot(
            index="date", columns="security_id", values="dollar_volume"
        )

    # Market cap is the denominator for every valuation ratio. It is kept in
    # CRSP units ($ thousands), so Compustat $ millions must be scaled when the
    # two are combined.
    if "market_equity" in frame.columns:
        panels["MarketCap"] = frame.pivot(
            index="date", columns="security_id", values="market_equity"
        )
    elif "market_cap" in frame.columns:
        # The published panel stores an already-computed market cap ($ thousands).
        # Without this branch every valuation ratio silently disappears and the
        # generator is left with profitability only.
        panels["MarketCap"] = frame.pivot(
            index="date", columns="security_id", values="market_cap"
        )
    elif {"prc", "shrout"} <= set(frame.columns):
        cap = frame.assign(market_cap=frame["prc"].abs() * frame["shrout"])
        panels["MarketCap"] = cap.pivot(
            index="date", columns="security_id", values="market_cap"
        )

    if "prc" in frame.columns:
        panels["Price"] = frame.pivot(
            index="date", columns="security_id", values="prc"
        ).abs()

    sector_map: dict[str, str] = {}
    if "siccd" in frame.columns:
        latest = (
            frame.dropna(subset=["siccd"])
            .sort_values("date")
            .drop_duplicates("security_id", keep="last")
        )
        sector_map = {
            str(row["security_id"]): sic_to_sector(row["siccd"])
            for _, row in latest.iterrows()
        }
        known = sum(1 for v in sector_map.values() if v != "Unknown")
        logger.info(f"SIC sector map: {known}/{len(sector_map)} securities classified")

    for name, panel in panels.items():
        panel.index.name = "Date"
        logger.info(f"Loaded {name}: {panel.shape[0]} dates x {panel.shape[1]} securities")

    return panels, sector_map


def build_tradable_universe_mask(
    panels: dict[str, pd.DataFrame],
    min_price: float = 0.0,
    min_dollar_volume: float = 0.0,
    min_dollar_volume_pct: float = 0.0,
    top_n_market_cap: int = 0,
    min_market_cap: float = 0.0,
) -> pd.DataFrame | None:
    """Formation-date liquidity mask (True = tradable, False = excluded).

    Applied to signal values *before* IC and quintile screening so microcaps
    cannot drive rank IC or blow up spreads, while leaving ``Close`` and
    ``Returns`` untouched so delisting returns stay intact.

    Prefer ``min_dollar_volume_pct`` (e.g. ``0.20`` keeps the top 80% by
    cross-sectional dollar volume each date) over a fixed nominal
    ``min_dollar_volume`` across multi-decade panels: a $1M floor in 1975
    eliminates almost the entire market and collapses breadth to ~600 names.
    """
    close = panels.get("Close")
    if close is None or close.empty:
        return None
    if (
        min_price <= 0
        and min_dollar_volume <= 0
        and min_dollar_volume_pct <= 0
        and top_n_market_cap <= 0
        and min_market_cap <= 0
    ):
        return None

    mask = pd.DataFrame(True, index=close.index, columns=close.columns)
    if min_price > 0 and "Price" in panels:
        price = panels["Price"].reindex(index=close.index, columns=close.columns)
        mask = mask & price.ge(float(min_price))
    if "Volume" in panels:
        vol = panels["Volume"].reindex(index=close.index, columns=close.columns)
        if min_dollar_volume > 0:
            mask = mask & vol.ge(float(min_dollar_volume))
        if min_dollar_volume_pct > 0:
            pct_rank = vol.where(mask).rank(axis=1, pct=True)
            mask = mask & pct_rank.ge(float(min_dollar_volume_pct))
    if (top_n_market_cap > 0 or min_market_cap > 0) and "MarketCap" in panels:
        cap = panels["MarketCap"].reindex(index=close.index, columns=close.columns)
        if min_market_cap > 0:
            mask = mask & cap.ge(float(min_market_cap))
        if top_n_market_cap > 0:
            cap_rank = cap.where(mask).rank(axis=1, ascending=False, method="first")
            mask = mask & cap_rank.le(int(top_n_market_cap))

    median_names = float(mask.sum(axis=1).median()) if len(mask) else 0.0
    logger.info(
        f"Tradable universe mask active (min_price={min_price}, min_dollar_vol={min_dollar_volume}, "
        f"min_vol_pct={min_dollar_volume_pct}, top_n_cap={top_n_market_cap}, min_cap={min_market_cap}): "
        f"median {median_names:.0f} tradable names/date"
    )
    return mask


def load_crsp_weekly_panel(
    release_uri: str,
    start: str = "2000-01-01",
    end: str | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Read ``gold/crsp-merged`` daily CRSP Parquet parts in place and resample to W-FRI.

    ``gold/crsp-merged`` has 110.2M daily rows (1925-2025) with columns
    ``permno, date, price, ret, vol, cap``. Pushing ``date >= start`` down to
    PyArrow keeps memory bounded on an 8 GB Modal container while giving 260+
    weekly periods per 5-year validation window.
    """
    available = r2_store.dataset_schema(release_uri)
    cols = [c for c in ("permno", "date", "price", "ret", "vol", "cap") if c in available]
    missing = {"permno", "date", "ret"} - set(cols)
    if missing:
        raise ValueError(f"Daily CRSP release missing required columns: {sorted(missing)}")

    filters: list[tuple] = []
    if start:
        # ``gold/crsp-merged`` stores date as ISO text; a timestamp literal
        # cannot bind against a large_string column in the Arrow scanner.
        filters.append(("date", ">=", str(start)[:10]))
    if end:
        filters.append(("date", "<=", str(end)[:10]))

    frame = r2_store.read_parquet_parts(release_uri, cols, filters=filters or None)
    frame = frame.dropna(subset=["permno", "date"]).copy()
    frame["security_id"] = "crsp:permno:" + frame["permno"].astype("int64").astype(str)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["week"] = frame["date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
    frame["ret"] = pd.to_numeric(frame["ret"], errors="coerce")
    if "price" in frame.columns:
        frame["abs_price"] = pd.to_numeric(frame["price"], errors="coerce").abs()
    if {"abs_price", "vol"} <= set(frame.columns):
        frame["dollar_vol"] = frame["abs_price"] * pd.to_numeric(frame["vol"], errors="coerce")

    frame = frame.sort_values(["security_id", "date"])
    # Compound daily returns within each week: prod(1 + r) - 1, and derive
    # weekly microstructure panels from the daily observations already in memory.
    frame["one_plus_ret"] = 1.0 + frame["ret"].fillna(0.0)
    if "dollar_vol" in frame.columns:
        pos_dvol = frame["dollar_vol"].where(frame["dollar_vol"] > 0)
        # Amihud (2002) illiquidity: |r_d| / dollar_vol_d (scaled by 1e6 for readability).
        frame["amihud_daily"] = (frame["ret"].abs() / pos_dvol) * 1e6

    agg: dict[str, Any] = {
        "one_plus_ret": "prod",
        "ret": "std",
    }
    if "dollar_vol" in frame.columns:
        agg["dollar_vol"] = "sum"
    if "amihud_daily" in frame.columns:
        agg["amihud_daily"] = "mean"
    if "abs_price" in frame.columns:
        agg["abs_price"] = "last"
    if "cap" in frame.columns:
        agg["cap"] = "last"

    weekly = frame.groupby(["week", "security_id"], sort=False).agg(agg).reset_index()
    weekly["ret_week"] = weekly["one_plus_ret"] - 1.0

    returns = weekly.pivot(index="week", columns="security_id", values="ret_week")
    close = (1.0 + returns.fillna(0.0)).cumprod()
    panels: dict[str, pd.DataFrame] = {
        "Close": close,
        "Returns": returns,
        "IdioVol": weekly.pivot(index="week", columns="security_id", values="ret"),
    }
    # 52-week high proximity (George & Hwang 2004): Close / rolling_52w_max(Close).
    rolling_high = close.rolling(52, min_periods=13).max().replace(0.0, float("nan"))
    panels["High52wRatio"] = close / rolling_high

    if "dollar_vol" in weekly.columns:
        vol_panel = weekly.pivot(index="week", columns="security_id", values="dollar_vol")
        panels["Volume"] = vol_panel
        # Abnormal volume shock: log(dollar_vol / rolling_13w_mean(dollar_vol)).
        import numpy as np

        vol_ma = vol_panel.rolling(13, min_periods=4).mean().replace(0.0, np.nan)
        ratio = (vol_panel / vol_ma).where(lambda x: x > 0)
        panels["VolumeShock"] = np.log(ratio)
    if "amihud_daily" in weekly.columns:
        panels["AmihudIlliq"] = weekly.pivot(
            index="week", columns="security_id", values="amihud_daily"
        )
    if "abs_price" in weekly.columns:
        panels["Price"] = weekly.pivot(index="week", columns="security_id", values="abs_price")
    if "cap" in weekly.columns:
        panels["MarketCap"] = weekly.pivot(index="week", columns="security_id", values="cap")

    for name, panel in panels.items():
        panel.index.name = "Date"
        logger.info(f"Loaded weekly {name}: {panel.shape[0]} weeks x {panel.shape[1]} securities")
    return panels, {}


def coverage_report(
    panels: dict[str, pd.DataFrame], sector_map: dict[str, str]
) -> dict[str, Any]:
    close = panels.get("Close")
    if close is None:
        return {"error": "no Close panel"}
    returns = panels.get("Returns")
    return {
        "dates": int(close.shape[0]),
        "securities": int(close.shape[1]),
        "first_date": str(close.index.min().date()) if len(close) else None,
        "last_date": str(close.index.max().date()) if len(close) else None,
        "frequency": "monthly",
        "return_coverage": float(returns.notna().mean().mean()) if returns is not None else None,
        "median_names_per_date": float(close.notna().sum(axis=1).median()),
        "sectors": len({v for v in sector_map.values() if v != "Unknown"}),
        "note": "Close is a total-return index, so TS_Return/TS_Mean are valid; raw price levels are not.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read the PHF CRSP panel from R2 and report coverage. Writes nothing."
    )
    parser.add_argument(
        "--panel", default=None,
        help="r2://bucket/key (or PHF_PANEL_PATH / QSD_DATA_SOURCE). Read in place.",
    )
    parser.add_argument("--start", default=None, help="Start date, e.g. 1990-01-01")
    parser.add_argument("--end", default=None, help="End date, e.g. 2024-12-31")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    import os

    panels, sectors = load_phf_panel(
        args.panel or os.environ.get("PHF_PANEL_PATH") or os.environ.get("QSD_DATA_SOURCE"),
        start=args.start,
        end=args.end,
    )
    print(json.dumps(coverage_report(panels, sectors), indent=2))


if __name__ == "__main__":
    main()
