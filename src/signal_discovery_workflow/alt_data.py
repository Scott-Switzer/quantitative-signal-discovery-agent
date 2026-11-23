# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0

"""
Zero-local-staging SEC EDGAR Form 4 & Form 13F point-in-time alpha pipelines.

Both datasets are ingested directly from free public SEC EDGAR quarterly bulk
TSV archives in cloud memory (Modal) and persisted as compact Parquet tables in
Cloudflare R2:

1. **SEC Form 4 Insider Transactions (``InsiderNetBuyRatio``,
   ``InsiderClusterBuys``, ``DiscretionaryBuyRatio``):**
   * Strictly indexed by ``FILING_DATE`` (never ``TRANS_DATE``) to prevent
     publication look-ahead (QuantInsti Form 4 Study, 2026).
   * Excludes ``4/A`` amendments and filters out Rule 10b5-1 planned trades via
     the post-2023 ``AFF10B5ONE`` indicator plus footnote regex (SEC Form 4
     Signal Extraction, 2026).
2. **SEC Form 13F Institutional Holdings (``InstOwnershipDelta``,
   ``InstCrowding``):**
   * Strictly indexed by ``FILING_DATE`` (up to 45 days after quarter end,
     never ``PERIODOFREPORT``).
   * Maps CUSIPs to ``permno`` via the approved fail-closed ``crsp__ccm_lookup``
     table.
"""

from __future__ import annotations

import calendar
import gzip
import io
import logging
import os
import re
import urllib.request
import zipfile
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from . import r2_store
from .fundamentals import _load_cik_bridge, _normalize_cik, _scatter

logger = logging.getLogger(__name__)

DEFAULT_FORM4_URI = (
    "r2://financial-data-wrds-lakehouse/research/datasets/sec_form4/panel.parquet"
)
DEFAULT_F13_URI = (
    "r2://financial-data-wrds-lakehouse/research/datasets/sec_13f/panel.parquet"
)

_PLAN_10B5_1_RE = re.compile(r"10b\s*5[\-\s]*1", re.IGNORECASE)


def _sec_fetch_bytes(url: str, user_agent: str | None = None, timeout: int = 90) -> bytes:
    ua = user_agent or os.environ.get("SEC_USER_AGENT") or "QuantitativeResearch/1.0 (research@example.com)"
    req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        if resp.headers.get("Content-Encoding", "").lower() == "gzip":
            raw = gzip.decompress(raw)
        return raw


def _quarter_list(start_year: int, end_year: int) -> list[tuple[int, int]]:
    return [(y, q) for y in range(start_year, end_year + 1) for q in (1, 2, 3, 4)]


def _month_end(value: date) -> date:
    return date(value.year, value.month, calendar.monthrange(value.year, value.month)[1])


def _add_months(value: date, months: int) -> date:
    index = value.month - 1 + months
    return date(value.year + index // 12, index % 12 + 1, 1)


_FORM13F_BASES = (
    "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/",
    "https://www.sec.gov/files/datastandardsinnovation/data/form-13f-data-sets/",
)

_FORM4_BASES = (
    "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/",
    "https://www.sec.gov/files/datastandardsinnovation/data/insider-transactions-data-sets/",
)


def _form13f_archive_names(start_year: int, end_year: int) -> list[str]:
    """Enumerate SEC 13F archive filenames across both naming regimes.

    Calendar-quarter archives run through 2023. From 2024 the SEC publishes
    rolling two/three-month filing windows (Jan-Feb 2024, then Mar-May, Jun-Aug,
    Sep-Nov, Dec-Feb, ...) so a naive ``YYYYqN`` name returns 404.
    """
    names: list[str] = []
    for year in range(max(2013, start_year), min(2023, end_year) + 1):
        for quarter in (1, 2, 3, 4):
            if year == 2013 and quarter == 1:
                continue
            names.append(f"{year}q{quarter}")

    if end_year >= 2024:
        windows = [(date(2024, 1, 1), date(2024, 2, 29))]
        cursor = date(2024, 3, 1)
        today = date.today()
        while cursor <= today:
            window_end = _month_end(_add_months(cursor, 2))
            windows.append((cursor, window_end))
            cursor = window_end + timedelta(days=1)
        for start, end in windows:
            names.append(
                f"01{start.strftime('%b').lower()}{start.year}-"
                f"{end.day:02d}{end.strftime('%b').lower()}{end.year}"
            )
    return names


def _read_tsv(zf: zipfile.ZipFile, name: str, wanted: set[str]) -> pd.DataFrame:
    """Read only the requested upper-case columns from a bulk SEC TSV."""
    header = pd.read_csv(zf.open(name), sep="\t", nrows=0, dtype=str)
    cols = [c for c in header.columns if c.upper() in wanted]
    return pd.read_csv(
        zf.open(name), sep="\t", dtype=str, low_memory=False, on_bad_lines="skip", usecols=cols
    )


# ---------------------------------------------------------------------------
# SEC Form 4 Insider Buying (2006 - present)
# ---------------------------------------------------------------------------

def parse_form4_quarter_zip(raw_zip: bytes) -> pd.DataFrame:
    """Parse one SEC Form 3/4/5 quarterly TSV archive into issuer-filing-day rows."""
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
        names = {n.upper(): n for n in zf.namelist()}
        sub_name = next((v for k, v in names.items() if k.endswith("SUBMISSION.TSV")), None)
        tr_name = next((v for k, v in names.items() if k.endswith("NONDERIV_TRANS.TSV")), None)
        own_name = next((v for k, v in names.items() if k.endswith("REPORTINGOWNER.TSV")), None)
        if not (sub_name and tr_name):
            return pd.DataFrame()

        sub = _read_tsv(
            zf,
            sub_name,
            {"ACCESSION_NUMBER", "FILING_DATE", "ISSUERCIK", "DOCUMENT_TYPE", "AFF10B5ONE", "REMARKS"},
        )
        tr = _read_tsv(
            zf,
            tr_name,
            {"ACCESSION_NUMBER", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE"},
        )
        own = (
            _read_tsv(zf, own_name, {"ACCESSION_NUMBER", "RPTOWNERCIK"})
            if own_name
            else pd.DataFrame()
        )

    sub.columns = [c.upper() for c in sub.columns]
    tr.columns = [c.upper() for c in tr.columns]
    if not own.empty:
        own.columns = [c.upper() for c in own.columns]

    # Keep original Form 4 only (drop 4/A amendments to prevent double counting).
    if "DOCUMENT_TYPE" in sub.columns:
        sub = sub[sub["DOCUMENT_TYPE"].astype(str).str.upper().eq("4")]
    if sub.empty:
        return pd.DataFrame()

    # Detect Rule 10b5-1 plan checkbox when present (mandatory post-Apr 2023).
    if "AFF10B5ONE" in sub.columns:
        sub["is_10b5_1"] = sub["AFF10B5ONE"].astype(str).str.strip().str.lower().isin(
            {"1", "true", "t", "y", "yes"}
        )
    else:
        sub["is_10b5_1"] = False

    if "REMARKS" in sub.columns:
        sub["is_10b5_1"] = sub["is_10b5_1"] | sub["REMARKS"].fillna("").str.contains(_PLAN_10B5_1_RE)

    keep_sub = [c for c in ("ACCESSION_NUMBER", "FILING_DATE", "ISSUERCIK", "is_10b5_1") if c in sub.columns]
    sub = sub[keep_sub].dropna(subset=["ACCESSION_NUMBER", "FILING_DATE", "ISSUERCIK"])

    if "TRANS_CODE" not in tr.columns:
        return pd.DataFrame()
    tr = tr[tr["TRANS_CODE"].astype(str).str.upper().isin({"P", "S"})].copy()
    if tr.empty:
        return pd.DataFrame()

    tr["shares"] = pd.to_numeric(tr.get("TRANS_SHARES"), errors="coerce").abs()
    tr["price"] = pd.to_numeric(tr.get("TRANS_PRICEPERSHARE"), errors="coerce").abs()
    tr["dollars"] = tr["shares"] * tr["price"]
    tr = tr[tr["dollars"].gt(0) & tr["dollars"].lt(1e10)]

    merged = tr.merge(sub, on="ACCESSION_NUMBER", how="inner")
    if merged.empty:
        return pd.DataFrame()

    if not own.empty and "ACCESSION_NUMBER" in own.columns and "RPTOWNERCIK" in own.columns:
        own_first = own.drop_duplicates("ACCESSION_NUMBER")[["ACCESSION_NUMBER", "RPTOWNERCIK"]]
        merged = merged.merge(own_first, on="ACCESSION_NUMBER", how="left")
    else:
        merged["RPTOWNERCIK"] = merged["ACCESSION_NUMBER"]

    merged["cik"] = _normalize_cik(merged["ISSUERCIK"])
    merged["filed_date"] = pd.to_datetime(merged["FILING_DATE"], errors="coerce")
    merged = merged.dropna(subset=["cik", "filed_date"])

    is_buy = merged["TRANS_CODE"].str.upper().eq("P")
    is_sell = merged["TRANS_CODE"].str.upper().eq("S")
    disc = ~merged["is_10b5_1"].astype(bool)

    merged["buy_dollars"] = np.where(is_buy, merged["dollars"], 0.0)
    merged["sell_dollars"] = np.where(is_sell, merged["dollars"], 0.0)
    merged["disc_buy_dollars"] = np.where(is_buy & disc, merged["dollars"], 0.0)
    merged["disc_sell_dollars"] = np.where(is_sell & disc, merged["dollars"], 0.0)
    merged["buyer_id"] = np.where(is_buy & disc, merged["RPTOWNERCIK"].astype(str), None)

    agg = (
        merged.groupby(["cik", "filed_date"], sort=False)
        .agg(
            buy_dollars=("buy_dollars", "sum"),
            sell_dollars=("sell_dollars", "sum"),
            disc_buy_dollars=("disc_buy_dollars", "sum"),
            disc_sell_dollars=("disc_sell_dollars", "sum"),
            n_buyers=("buyer_id", lambda s: int(s.dropna().nunique())),
        )
        .reset_index()
    )
    return agg


def ingest_sec_form4_to_r2(
    *,
    start_year: int = 2010,
    end_year: int | None = None,
    output_uri: str = DEFAULT_FORM4_URI,
    user_agent: str | None = None,
) -> dict[str, Any]:
    """Download quarterly SEC Form 4 bulk zips in memory and publish Parquet to R2."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    target_year = end_year or date.today().year
    frames: list[pd.DataFrame] = []
    quarters_ok = 0
    failed: list[str] = []
    for year, q in _quarter_list(start_year, target_year):
        filename = f"{year}q{q}_form345.zip"
        raw = None
        last_error: Exception | None = None
        for base in _FORM4_BASES:
            try:
                raw = _sec_fetch_bytes(base + filename, user_agent=user_agent)
                break
            except Exception as exc:
                last_error = exc
        if raw is None:
            failed.append(f"{year}q{q}")
            logger.warning(f"Skipping Form 4 {year}Q{q}: {last_error}")
            continue
        try:
            df = parse_form4_quarter_zip(raw)
            if not df.empty:
                frames.append(df)
                quarters_ok += 1
                logger.info(f"Form 4 {year}Q{q}: {len(df):,} issuer-day rows")
        except Exception as e:
            failed.append(f"{year}q{q}")
            logger.warning(f"Skipping Form 4 {year}Q{q}: {e}")

    if not frames:
        raise RuntimeError("No SEC Form 4 quarters were ingested")

    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined.groupby(["cik", "filed_date"], sort=False)
        .agg(
            buy_dollars=("buy_dollars", "sum"),
            sell_dollars=("sell_dollars", "sum"),
            disc_buy_dollars=("disc_buy_dollars", "sum"),
            disc_sell_dollars=("disc_sell_dollars", "sum"),
            n_buyers=("n_buyers", "sum"),
        )
        .reset_index()
    )
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(combined, preserve_index=False), buf, compression="zstd")
    r2_store.write_bytes(output_uri, buf.getvalue())
    logger.info(f"Published {len(combined):,} Form 4 issuer-day rows to {output_uri}")
    return {
        "quarters_loaded": quarters_ok,
        "archives_failed": failed,
        "rows_written": len(combined),
        "output_uri": output_uri,
    }


def build_form4_panels(
    events: pd.DataFrame,
    bridge: pd.DataFrame,
    dates: pd.Index,
    securities: pd.Index,
    *,
    valid_days: int = 90,
) -> dict[str, pd.DataFrame]:
    """Scatter point-in-time Form 4 insider signals onto ``(dates, securities)``.

    Every signal is keyed on ``filed_date`` (never transaction date) and decays
    after ``valid_days`` (default 90 days = 1 quarter).
    """
    if events.empty:
        return {}

    dates = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()
    columns = pd.Index([str(s) for s in securities])
    column_of = {name.split(":")[-1]: idx for idx, name in enumerate(columns)}

    df = events.copy()
    df["cik"] = _normalize_cik(df["cik"])
    df["filed_date"] = pd.to_datetime(df["filed_date"], errors="coerce")
    df = df.dropna(subset=["cik", "filed_date"])

    merged = df.merge(bridge, on="cik", how="inner")
    merged = merged[
        (merged["filed_date"] >= merged["effective_from"])
        & (merged["filed_date"] <= merged["effective_to"])
    ]
    merged = merged[merged["permno"].isin(column_of)]
    if merged.empty:
        return {}

    merged = merged.sort_values("filed_date")
    merged["_disc_buy"] = pd.to_numeric(
        merged.get("disc_buy_dollars", merged.get("buy_dollars", 0.0)), errors="coerce"
    ).fillna(0.0)
    merged["_disc_sell"] = pd.to_numeric(
        merged.get("disc_sell_dollars", merged.get("sell_dollars", 0.0)), errors="coerce"
    ).fillna(0.0)
    merged["_all_buy"] = pd.to_numeric(merged.get("buy_dollars", 0.0), errors="coerce").fillna(0.0)
    merged["_all_sell"] = pd.to_numeric(merged.get("sell_dollars", 0.0), errors="coerce").fillna(0.0)
    merged["_n_buyers"] = (
        pd.to_numeric(merged.get("n_buyers", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    )

    merged["valid_from"] = merged["filed_date"]
    merged["valid_to"] = merged["filed_date"] + pd.Timedelta(days=valid_days)

    left = np.searchsorted(dates.values, merged["valid_from"].to_numpy(), side="left")
    right = np.searchsorted(dates.values, merged["valid_to"].to_numpy(), side="right") - 1
    lengths = right - left + 1
    keep = lengths > 0
    if not keep.any():
        return {}

    left, lengths = left[keep], lengths[keep]
    kept = merged.loc[keep]
    col_index = kept["permno"].map(column_of).to_numpy(dtype=np.int64)
    total_cells = int(lengths.sum())
    offsets = np.arange(total_cells) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    rows = np.repeat(left, lengths) + offsets
    cols = np.repeat(col_index, lengths)
    shape = (len(dates), len(columns))
    flat_idx = rows * shape[1] + cols

    # Aggregate rolling ``valid_days`` dollar flows and buyer counts rather than
    # overwriting prior filings (so a small sale on Day 3 does not erase a large
    # cluster buy from Day 1).
    def _accum(col_name: str) -> np.ndarray:
        buf = np.zeros(shape[0] * shape[1], dtype=np.float64)
        vals = kept[col_name].to_numpy(dtype=np.float64)
        np.add.at(buf, flat_idx, np.repeat(vals, lengths))
        return buf.reshape(shape)

    disc_buy = _accum("_disc_buy")
    disc_sell = _accum("_disc_sell")
    all_buy = _accum("_all_buy")
    all_sell = _accum("_all_sell")
    n_buyers = _accum("_n_buyers")

    total_disc = disc_buy + disc_sell
    total_all = all_buy + all_sell
    with np.errstate(divide="ignore", invalid="ignore"):
        net_buy_arr = np.where(total_disc > 0.0, (disc_buy - disc_sell) / total_disc, np.nan).astype(np.float32)
        disc_ratio_arr = np.where(total_all > 0.0, disc_buy / total_all, np.nan).astype(np.float32)
        cluster_arr = np.where((total_all > 0.0) | (n_buyers > 0.0), np.log1p(n_buyers), np.nan).astype(np.float32)

    panels: dict[str, pd.DataFrame] = {}
    for arr, panel_name in (
        (net_buy_arr, "InsiderNetBuyRatio"),
        (cluster_arr, "InsiderClusterBuys"),
        (disc_ratio_arr, "DiscretionaryBuyRatio"),
    ):
        frame = pd.DataFrame(arr, index=dates, columns=columns)
        if frame.notna().any().any():
            panels[panel_name] = frame

    if panels:
        logger.info("Form 4 insider panels loaded: " + ", ".join(sorted(panels)))
    return panels


def load_form4_panels(
    panel_uri: str,
    bridge_uri: str,
    dates: pd.Index,
    securities: pd.Index,
) -> dict[str, pd.DataFrame]:
    """Read pre-ingested Form 4 Parquet from R2 into wide panels."""
    schema = r2_store.dataset_schema(panel_uri)
    cols = [
        c
        for c in ("cik", "filed_date", "buy_dollars", "sell_dollars", "disc_buy_dollars", "disc_sell_dollars", "n_buyers")
        if c in schema
    ]
    events = r2_store.read_parquet_parts(panel_uri, cols)
    bridge = _load_cik_bridge(bridge_uri)
    return build_form4_panels(events, bridge, dates, securities)


# ---------------------------------------------------------------------------
# SEC Form 13F Institutional Holdings (2013 - present)
# ---------------------------------------------------------------------------

def parse_13f_quarter_zip(raw_zip: bytes) -> pd.DataFrame:
    """Parse one SEC Form 13F quarterly bulk TSV archive into CUSIP-quarter rows.

    Uses strictly ``FILING_DATE`` as the availability date (never period end).
    """
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
        names = {n.upper(): n for n in zf.namelist()}
        sub_name = next((v for k, v in names.items() if k.endswith("SUBMISSION.TSV")), None)
        info_name = next((v for k, v in names.items() if k.endswith("INFOTABLE.TSV")), None)
        if not (sub_name and info_name):
            return pd.DataFrame()

        sub = _read_tsv(
            zf, sub_name, {"ACCESSION_NUMBER", "FILING_DATE", "CIK", "SUBMISSIONTYPE"}
        )
        info = _read_tsv(
            zf, info_name, {"ACCESSION_NUMBER", "CUSIP", "VALUE"}
        )

    sub.columns = [c.upper() for c in sub.columns]
    info.columns = [c.upper() for c in info.columns]
    if "SUBMISSIONTYPE" in sub.columns:
        sub = sub[sub["SUBMISSIONTYPE"].astype(str).str.upper().eq("13F-HR")]
    if sub.empty or info.empty:
        return pd.DataFrame()

    sub = sub[["ACCESSION_NUMBER", "FILING_DATE", "CIK"]].dropna()
    info = info[[c for c in ("ACCESSION_NUMBER", "CUSIP", "VALUE", "SSHPRNAMT") if c in info.columns]].dropna(
        subset=["ACCESSION_NUMBER", "CUSIP", "VALUE"]
    )
    info["cusip8"] = info["CUSIP"].astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True).str[:8]
    info["value"] = pd.to_numeric(info["VALUE"], errors="coerce")
    info = info[info["cusip8"].str.len().eq(8) & info["value"].gt(0)]

    merged = info.merge(sub, on="ACCESSION_NUMBER", how="inner")
    if merged.empty:
        return pd.DataFrame()

    merged["filed_date"] = pd.to_datetime(merged["FILING_DATE"], errors="coerce")
    merged = merged.dropna(subset=["filed_date"])
    # Form 13F values were reported in thousands through 2022; the SEC's
    # January 2023 schema requires dollars. Normalize before QoQ deltas.
    merged["value"] = pd.to_numeric(merged["value"], errors="coerce") * np.where(
        merged["filed_date"] < pd.Timestamp("2023-01-03"), 1000.0, 1.0
    )

    # Sum sub-manager lines within each single filing (ACCESSION_NUMBER), then
    # keep only the latest filing per (cusip8, CIK) in this archive so a manager
    # filing twice inside a rolling 3-month window is never double-counted.
    per_filing = (
        merged.groupby(["cusip8", "CIK", "ACCESSION_NUMBER"], sort=False)
        .agg(mgr_val=("value", "sum"), filed_date=("filed_date", "max"))
        .reset_index()
    )
    mgr = (
        per_filing.sort_values("filed_date")
        .drop_duplicates(["cusip8", "CIK"], keep="last")
        .reset_index(drop=True)
    )
    tot = mgr.groupby("cusip8", sort=False)["mgr_val"].transform("sum")
    mgr["share_sq"] = (mgr["mgr_val"] / tot.replace(0.0, np.nan)) ** 2

    agg = (
        mgr.groupby("cusip8", sort=False)
        .agg(
            total_value=("mgr_val", "sum"),
            n_managers=("CIK", "nunique"),
            hhi=("share_sq", "sum"),
            filed_date=("filed_date", "max"),
        )
        .reset_index()
    )
    return agg


def ingest_sec_13f_to_r2(
    *,
    start_year: int = 2013,
    end_year: int | None = None,
    output_uri: str = DEFAULT_F13_URI,
    user_agent: str | None = None,
) -> dict[str, Any]:
    """Download SEC Form 13F bulk zips in memory and publish Parquet to R2."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    target_year = end_year or date.today().year
    frames: list[pd.DataFrame] = []
    quarters_ok = 0
    failed: list[str] = []
    for name in _form13f_archive_names(start_year, target_year):
        filename = f"{name}_form13f.zip"
        raw = None
        last_error: Exception | None = None
        for base in _FORM13F_BASES:
            try:
                raw = _sec_fetch_bytes(base + filename, user_agent=user_agent)
                break
            except Exception as exc:
                last_error = exc
        if raw is None:
            failed.append(name)
            logger.warning(f"Skipping Form 13F {name}: {last_error}")
            continue
        try:
            df = parse_13f_quarter_zip(raw)
            if not df.empty:
                frames.append(df)
                quarters_ok += 1
                logger.info(f"Form 13F {name}: {len(df):,} cusip rows")
        except Exception as e:
            failed.append(name)
            logger.warning(f"Skipping Form 13F {name}: {e}")

    if not frames:
        raise RuntimeError("No SEC Form 13F quarters were ingested")

    combined = pd.concat(frames, ignore_index=True)
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(combined, preserve_index=False), buf, compression="zstd")
    r2_store.write_bytes(output_uri, buf.getvalue())
    logger.info(f"Published {len(combined):,} Form 13F cusip-quarter rows to {output_uri}")
    return {
        "quarters_loaded": quarters_ok,
        "archives_failed": failed,
        "rows_written": len(combined),
        "output_uri": output_uri,
    }


def _load_cusip_bridge(uri: str) -> pd.DataFrame:
    """Load dated CUSIP8 -> permno bridge from ``crsp__ccm_lookup`` Parquet (fail-closed)."""
    schema = r2_store.parquet_schema(uri)
    cols = [c for c in ("lpermno", "permno", "cusip", "linkdt", "linkenddt") if c in schema]
    raw_df = r2_store.read_parquet(uri, cols)
    perm_col = "lpermno" if "lpermno" in raw_df.columns else "permno"
    frame = raw_df.dropna(subset=[perm_col, "cusip"]).copy()
    frame["permno"] = pd.to_numeric(frame[perm_col], errors="coerce")
    frame = frame.dropna(subset=["permno"]).copy()
    frame["permno"] = frame["permno"].astype("int64").astype(str)
    frame["cusip8"] = (
        frame["cusip"].astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True).str[:8]
    )
    frame = frame[frame["cusip8"].str.len().eq(8)]
    frame["effective_from"] = pd.to_datetime(
        frame["linkdt"] if "linkdt" in frame.columns else pd.NaT, errors="coerce"
    ).fillna(pd.Timestamp("1946-01-01"))
    frame["effective_to"] = pd.to_datetime(
        frame["linkenddt"] if "linkenddt" in frame.columns else pd.NaT, errors="coerce"
    ).fillna(pd.Timestamp("2099-12-31"))
    # Fail-closed unique-permno rule: drop any (cusip8, effective_from, effective_to)
    # that maps to more than one distinct permno.
    perm_counts = frame.groupby(["cusip8", "effective_from", "effective_to"])["permno"].transform("nunique")
    ambiguous = int((perm_counts > 1).sum())
    if ambiguous:
        logger.info(f"CCM CUSIP bridge: dropped {ambiguous} ambiguous multi-permno rows (fail-closed)")
    frame = frame[perm_counts == 1].drop_duplicates(["cusip8", "permno", "effective_from", "effective_to"])
    return frame[["cusip8", "permno", "effective_from", "effective_to"]]


def build_13f_panels(
    holdings: pd.DataFrame,
    cusip_bridge: pd.DataFrame,
    dates: pd.Index,
    securities: pd.Index,
    *,
    valid_days: int = 120,
) -> dict[str, pd.DataFrame]:
    """Scatter point-in-time Form 13F institutional flow & crowding panels."""
    if holdings.empty or cusip_bridge.empty:
        return {}

    dates = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()
    columns = pd.Index([str(s) for s in securities])
    column_of = {name.split(":")[-1]: idx for idx, name in enumerate(columns)}

    df = holdings.copy()
    df["cusip8"] = df["cusip8"].astype(str).str.upper().str[:8]
    df["filed_date"] = pd.to_datetime(df["filed_date"], errors="coerce")
    df = df.dropna(subset=["cusip8", "filed_date"]).sort_values(["cusip8", "filed_date"])

    # QoQ log change in aggregate institutional dollar holdings.
    prev_val = df.groupby("cusip8", sort=False)["total_value"].shift(1)
    df["ownership_delta"] = np.log(
        (pd.to_numeric(df["total_value"], errors="coerce") + 1.0)
        / (pd.to_numeric(prev_val, errors="coerce") + 1.0)
    )
    df["crowding"] = pd.to_numeric(df["hhi"], errors="coerce")

    merged = df.merge(cusip_bridge, on="cusip8", how="inner")
    merged = merged[
        (merged["filed_date"] >= merged["effective_from"])
        & (merged["filed_date"] <= merged["effective_to"])
    ]
    merged = merged[merged["permno"].isin(column_of)]
    if merged.empty:
        return {}

    merged = merged.sort_values("filed_date")
    merged["valid_from"] = merged["filed_date"]
    merged["valid_to"] = merged["filed_date"] + pd.Timedelta(days=valid_days)

    left = np.searchsorted(dates.values, merged["valid_from"].to_numpy(), side="left")
    right = np.searchsorted(dates.values, merged["valid_to"].to_numpy(), side="right") - 1
    lengths = right - left + 1
    keep = lengths > 0
    if not keep.any():
        return {}

    left, lengths = left[keep], lengths[keep]
    kept = merged.loc[keep]
    col_index = kept["permno"].map(column_of).to_numpy(dtype=np.int64)
    total_cells = int(lengths.sum())
    offsets = np.arange(total_cells) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    rows = np.repeat(left, lengths) + offsets
    cols = np.repeat(col_index, lengths)
    shape = (len(dates), len(columns))

    panels: dict[str, pd.DataFrame] = {}
    for src_col, panel_name in (
        ("ownership_delta", "InstOwnershipDelta"),
        ("crowding", "InstCrowding"),
    ):
        vals = pd.to_numeric(kept[src_col], errors="coerce").astype("float32")
        arr = _scatter(vals, lengths, rows, cols, shape)
        frame = pd.DataFrame(arr, index=dates, columns=columns)
        if frame.notna().any().any():
            panels[panel_name] = frame

    if panels:
        logger.info("Form 13F institutional panels loaded: " + ", ".join(sorted(panels)))
    return panels


def load_13f_panels(
    panel_uri: str,
    ccm_uri: str,
    dates: pd.Index,
    securities: pd.Index,
) -> dict[str, pd.DataFrame]:
    """Read pre-ingested Form 13F Parquet from R2 into wide panels."""
    schema = r2_store.dataset_schema(panel_uri)
    cols = [c for c in ("cusip8", "filed_date", "total_value", "n_managers", "hhi") if c in schema]
    holdings = r2_store.read_parquet_parts(panel_uri, cols)
    bridge = _load_cusip_bridge(ccm_uri)
    return build_13f_panels(holdings, bridge, dates, securities)
