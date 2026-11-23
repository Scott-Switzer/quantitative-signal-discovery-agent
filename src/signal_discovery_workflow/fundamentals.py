# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0

"""Point-in-time Compustat fundamentals on the CRSP monthly grid.

The price-only panel can only express price/volume signals, which is why the
first CRSP run kept proposing momentum and volatility. This module adds the
accounting side: value, profitability, investment, accruals and leverage.

Two R2 datasets are joined in memory, in place:

* ``compsamp_snapshot__wrds_csa_pit`` — annual Compustat with ``pitdate1`` /
  ``pitdate2``, the interval during which a record was the latest available.
  Using the PIT interval instead of ``datadate`` is what keeps a restated or
  late-filed number out of the months before it was knowable.
* ``crsp__ccmxpf_linktable`` — the gvkey <-> permno link with its own validity
  window.

Each record is then scattered onto the CRSP month grid, so a panel has the same
shape and index as ``Close`` and can be evaluated by the same code.

Units: CRSP market equity is in $ thousands and Compustat items are in
$ millions, so accounting values used as a ratio numerator are multiplied by
1000. Ratios of two accounting items are unit-free and left alone.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import r2_store

logger = logging.getLogger(__name__)

FUNDAMENTAL_ITEMS = [
    "gvkey",
    "datadate",
    "fdate",
    "pitdate1",
    "pitdate2",
    "at",
    "ceq",
    "dltt",
    "dlc",
    "ni",
    "sale",
    "cogs",
    "xsga",
    "xrd",
    "oancf",
    "capx",
    "oi",
    "csho",
    "curcd",
    "datafmt",
    "consol",
    "indfmt",
    "popsrc",
]

LINK_COLUMNS = [
    "gvkey",
    "lpermno",
    "linkdt",
    "linkenddt",
    "linkprim",
    "linktype",
    "usedflag",
]

_FILTERS = {
    "datafmt": "STD",
    "consol": "C",
    "indfmt": "INDL",
    "popsrc": "D",
    "curcd": "USD",
}

_MAX_SCATTER_CELLS = 40_000_000

# XBRL tags -> canonical item. Several tags can map to one item; the first
# non-null value wins. The release we read carries the first block; the rest
# are accepted if a future release adds them, so the loader does not need a
# rewrite to pick up cash-flow or cost-of-sales tags.
SEC_METRICS: dict[str, str] = {
    "Revenues": "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
    "SalesRevenueNet": "revenue",
    "NetIncomeLoss": "net_income",
    "StockholdersEquity": "equity",
    "Assets": "assets",
    "Liabilities": "liabilities",
    "OperatingIncomeLoss": "operating_income",
    "CashAndCashEquivalentsAtCarryingValue": "cash",
    "LiabilitiesCurrent": "current_liabilities",
    "LongTermDebt": "lt_debt",
    "CommonStockSharesOutstanding": "shares",
    "WeightedAverageNumberOfDilutedSharesOutstanding": "diluted_shares",
    # Optional tags. The published release does not include these yet, so the
    # accruals / gross-profit / capex panels are built only when they appear.
    "GrossProfit": "gross_profit",
    "NetCashProvidedByUsedInOperatingActivities": "cfo",
    "CostOfGoodsAndServicesSold": "cogs",
    "CostOfRevenue": "cogs",
    "PaymentsToAcquirePropertyPlantAndEquipment": "capex",
    "ResearchAndDevelopmentExpense": "rnd",
}

_SEC_COLUMNS = ["cik", "metric", "end", "filed", "value", "fp", "form"]

# companyfacts mixes quarterly and annual durations under the same
# ``(cik, period_end, metric)`` key. The release has no ``start`` column, so
# duration cannot be inferred from the row itself; ``fp`` and ``form`` are the
# only reliable annual markers. Without this filter a 10-Q's quarterly net
# income can win the "earliest filed" tie at a fiscal year end and be divided
# by a full-year market cap, which is why the first run reported an
# EarningsYield median of 0.0073 instead of roughly 0.04.
_ANNUAL_FORMS = {
    "10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A",
}


def _annual_facts(facts: pd.DataFrame) -> pd.DataFrame:
    """Keep annual-report rows only, so flow items are twelve-month values."""
    before = len(facts)
    if "fp" in facts.columns:
        facts = facts[facts["fp"].astype(str).str.upper().eq("FY")]
    if "form" in facts.columns:
        facts = facts[facts["form"].astype(str).str.upper().isin(_ANNUAL_FORMS)]
    logger.info(
        f"SEC facts: kept {len(facts):,} annual rows of {before:,} "
        f"({100.0 * len(facts) / max(before, 1):.1f}%)"
    )
    return facts


def _normalize_cik(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(10)
    )


DEFAULT_CCM_IDENTITY_URI = (
    "r2://financial-data-wrds-private/raw/wrds/crsp__ccm_lookup/"
    "releases/20260922T085328Z/part-000000.parquet"
)


def _load_cik_bridge(uri: str) -> pd.DataFrame:
    """Load dated permno <-> CIK assertions, fail-closed on ambiguous multi-CIK links.

    Supports both:
    * The approved ``crsp__ccm_lookup`` Parquet table (``lpermno, cik, linkdt,
      linkenddt``) — personal-hedge-fund's fail-closed identity contract.
    * Legacy ``.jsonl.gz`` assertion files (quarantined in PHF; kept only for
      unit-test compatibility).
    """
    import io

    if str(uri).lower().endswith(".parquet"):
        schema = r2_store.parquet_schema(uri)
        cols = [c for c in ("lpermno", "permno", "cik", "linkdt", "linkenddt") if c in schema]
        raw_df = r2_store.read_parquet(uri, cols)
        perm_col = "lpermno" if "lpermno" in raw_df.columns else "permno"
        frame = raw_df.dropna(subset=[perm_col, "cik"]).copy()
        frame["permno"] = pd.to_numeric(frame[perm_col], errors="coerce").dropna().astype("int64").astype(str)
        frame = frame.dropna(subset=["permno"])
        frame["cik"] = _normalize_cik(frame["cik"])
        frame = frame[frame["cik"].str.len().eq(10) & frame["cik"].ne("0000000000")]
        frame["effective_from"] = pd.to_datetime(
            frame["linkdt"] if "linkdt" in frame.columns else pd.NaT, errors="coerce"
        ).fillna(pd.Timestamp("1946-01-01"))
        frame["effective_to"] = pd.to_datetime(
            frame["linkenddt"] if "linkenddt" in frame.columns else pd.NaT, errors="coerce"
        ).fillna(pd.Timestamp("2099-12-31"))
        # Fail-closed unique-CIK rule: drop any (permno, effective_from, effective_to)
        # that maps to more than one distinct CIK.
        cik_counts = frame.groupby(["permno", "effective_from", "effective_to"])["cik"].transform("nunique")
        ambiguous = int((cik_counts > 1).sum())
        if ambiguous:
            logger.info(f"CCM identity bridge: dropped {ambiguous} ambiguous multi-CIK rows (fail-closed)")
        frame = frame[cik_counts == 1].drop_duplicates(["permno", "cik", "effective_from", "effective_to"])
        logger.info(f"CCM identity bridge: {len(frame)} fail-closed permno-CIK intervals")
        return frame[["cik", "permno", "effective_from", "effective_to"]]

    raw = r2_store.read_bytes(uri)
    # The object is named .jsonl.gz but some publications store it plain.
    compression = "gzip" if raw[:2] == b"\x1f\x8b" else None
    frame = pd.read_json(io.BytesIO(raw), lines=True, compression=compression)
    frame = frame[frame.get("identifier_type", "CIK") == "CIK"]
    frame = frame[frame["permno"].notna()].copy()
    frame["cik"] = _normalize_cik(frame["identifier_value"])
    frame["permno"] = frame["permno"].astype("int64").astype(str)
    for column in ("effective_from", "effective_to"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["effective_from"] = frame["effective_from"].fillna(pd.Timestamp("1990-01-01"))
    frame["effective_to"] = frame["effective_to"].fillna(pd.Timestamp("2262-01-01"))
    logger.info(f"CIK bridge: {len(frame)} dated CIK-permno assertions")
    return frame[["cik", "permno", "effective_from", "effective_to"]]


def _lagged(array: np.ndarray, rows: int = 12) -> np.ndarray:
    """Shift a monthly panel back one fiscal year, padding with NaN."""
    out = np.full_like(array, np.nan)
    if array.shape[0] > rows:
        out[rows:] = array[:-rows]
    return out


def build_sec_fundamental_panels(
    facts: pd.DataFrame,
    bridge: pd.DataFrame,
    dates: pd.Index,
    securities: pd.Index,
    market_cap: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Turn long-form annual XBRL facts into monthly ratio panels.

    Separated from the R2 reader so the point-in-time and scaling rules are
    testable without a bucket. ``facts`` must already be annual-report rows
    (see :func:`_annual_facts`); this function keeps the earliest filing per
    period, which is the number that was knowable at the time.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()
    columns = pd.Index([str(s) for s in securities])
    column_of = {name.split(":")[-1]: position for position, name in enumerate(columns)}

    facts = facts.dropna(subset=["cik", "value", "end", "filed"]).copy()
    facts["metric"] = facts["metric"].astype(str).str.split(":").str[-1]
    facts["item"] = facts["metric"].map(SEC_METRICS)
    facts = facts.dropna(subset=["item"])
    facts["cik"] = _normalize_cik(facts["cik"])
    facts["end"] = pd.to_datetime(facts["end"], errors="coerce")
    facts["filed"] = pd.to_datetime(facts["filed"], errors="coerce")
    facts = facts.dropna(subset=["end", "filed"])

    # Earliest filing per period: the number as first reported, not restated.
    facts = facts.sort_values("filed").drop_duplicates(["cik", "end", "item"], keep="first")

    wide = (
        facts.pivot_table(index=["cik", "end"], columns="item", values="value", aggfunc="first")
        .reset_index()
    )
    available = (
        facts.groupby(["cik", "end"])["filed"]
        .max()
        .rename("available")
        .reset_index()
    )
    wide = wide.merge(available, on=["cik", "end"], how="left")
    wide = wide.dropna(subset=["available"])

    merged = wide.merge(bridge, on="cik", how="inner")
    merged = merged[
        (merged["available"] >= merged["effective_from"])
        & (merged["available"] <= merged["effective_to"])
    ]
    merged = merged[merged["permno"].isin(column_of)]
    if merged.empty:
        raise ValueError("No SEC filer maps to a security in the CRSP panel")

    merged = merged.sort_values("available")
    merged["valid_from"] = merged["available"]
    # An annual report is the latest number for roughly a year. 460 days allows
    # for a late filer without carrying a dead company's last report forever.
    merged["valid_to"] = merged["available"] + pd.Timedelta(days=460)

    left = np.searchsorted(dates.values, merged["valid_from"].to_numpy(), side="left")
    right = np.searchsorted(dates.values, merged["valid_to"].to_numpy(), side="right") - 1
    lengths = right - left + 1
    keep = lengths > 0
    left, lengths = left[keep], lengths[keep]
    kept = merged.loc[keep]
    col_index = kept["permno"].map(column_of).to_numpy(dtype=np.int64)

    total = int(lengths.sum())
    if total > _MAX_SCATTER_CELLS:
        raise ValueError(f"SEC scatter would allocate {total:,} cells")
    offsets = np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    rows = np.repeat(left, lengths) + offsets
    cols = np.repeat(col_index, lengths)
    shape = (len(dates), len(columns))

    def panel(name: str) -> np.ndarray:
        if name not in kept.columns:
            return np.full(shape, np.nan, dtype=np.float32)
        values = pd.to_numeric(kept[name], errors="coerce").astype("float32")
        return _scatter(values, lengths, rows, cols, shape)

    panels: dict[str, pd.DataFrame] = {}

    def add(name: str, array: np.ndarray) -> None:
        frame = pd.DataFrame(array, index=dates, columns=columns)
        if frame.notna().any().any():
            panels[name] = frame

    # Published XBRL values and the panel's market_cap are both in dollars
    # (verified: median market cap ~7.8e8 for a 5,000-name month). Do not scale
    # either side, or every valuation ratio comes out 1000x wrong.
    assets = panel("assets")
    lag_assets = _lagged(assets)
    equity = panel("equity")
    lag_equity = _lagged(equity)
    net_income = panel("net_income")
    operating_income = panel("operating_income")
    revenue = panel("revenue")

    if market_cap is not None:
        cap = market_cap.reindex(index=dates, columns=columns).to_numpy(dtype=np.float32)
        add("BookToMarket", _ratio(equity, cap))
        add("EarningsYield", _ratio(net_income, cap))

    # Profitability is scaled by LAGGED assets. Scaling by current assets (the
    # usual Gpa construction) mechanically divides by asset growth, so the
    # "profitability" premium is contaminated by the investment premium; in a
    # 447-anomaly replication the lagged version is the one with an independent
    # effect (Hou, Xue & Zhang 2020, RFS).
    add("OperatingProfitability", _ratio(operating_income, lag_assets))
    add("Roa", _ratio(net_income, lag_assets))
    add("Roe", _ratio(net_income, lag_equity))
    add("SalesToAssets", _ratio(revenue, lag_assets))

    # Book leverage. The release has total liabilities but not a clean
    # short-term-debt tag, so total liabilities is the honest numerator; adding
    # a partial debt number would understate leverage for most filers.
    add("Leverage", _ratio(panel("liabilities"), assets))
    add("BookEquityToAssets", _ratio(equity, assets))

    # Optional quality panels: only built when the release carries the tag.
    gross_profit = panel("gross_profit")
    cogs = panel("cogs")
    if np.isfinite(gross_profit).any() or np.isfinite(cogs).any():
        derived_gross = np.where(np.isnan(gross_profit), revenue - cogs, gross_profit)
        add("GrossProfitability", _ratio(derived_gross, lag_assets))
    add("Accruals", _ratio(net_income - panel("cfo"), assets))
    add("CapexToAssets", _ratio(panel("capex"), lag_assets))
    add("RndToSales", _ratio(panel("rnd"), revenue))

    add("AssetGrowth", _ratio(assets, lag_assets) - 1.0)

    shares = panel("shares")
    if np.isfinite(shares).any():
        # Net share issuance: high issuance predicts low returns (Pontiff &
        # Woodgate 2008). Growth in shares is unit-free and survives splits only
        # imperfectly, so extreme values are clipped by the evaluator's ranks.
        issuance = _ratio(shares, _lagged(shares)) - 1.0
        add("ShareIssuance", issuance)

    if not panels:
        raise ValueError("SEC fundamentals loaded but produced no usable panel")
    logger.info("SEC fundamentals panels: " + ", ".join(sorted(panels)))
    return panels


def load_sec_fundamentals_panels(
    companyfacts_uri: str,
    bridge_uri: str,
    dates: pd.Index,
    securities: pd.Index,
    market_cap: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Monthly fundamental panels from SEC XBRL companyfacts (2009 onward).

    A fact becomes usable at its ``filed`` date, not at its fiscal period end:
    that is the point-in-time rule. Only annual reports are used, so a flow
    item is always a twelve-month value. Where several tags describe the same
    item, the first available tag wins; where a period is restated, the
    earliest filing is kept so the panel shows what was knowable then.
    """
    # companyfacts tags are namespaced (``us-gaap:Revenues``); filter on both
    # forms so a namespaced or bare table works.
    metric_filter = sorted({f"us-gaap:{m}" for m in SEC_METRICS} | set(SEC_METRICS))
    available_columns = r2_store.dataset_schema(companyfacts_uri)
    columns = [c for c in _SEC_COLUMNS if c in available_columns]
    facts = r2_store.read_parquet_parts(
        companyfacts_uri,
        columns,
        filters=[("metric", "in", metric_filter)],
    )
    facts = _annual_facts(facts)
    bridge = _load_cik_bridge(bridge_uri)
    return build_sec_fundamental_panels(
        facts, bridge, dates, securities, market_cap=market_cap
    )


def _load_records(uri: str) -> pd.DataFrame:
    schema = r2_store.dataset_schema(uri)
    columns = [c for c in FUNDAMENTAL_ITEMS if c in schema]
    missing = {"gvkey", "datadate"} - set(columns)
    if missing:
        raise ValueError(f"Compustat snapshot is missing {sorted(missing)}")

    frame = r2_store.read_parquet_parts(uri, columns)
    for column, value in _FILTERS.items():
        if column in frame.columns:
            frame = frame[frame[column].astype(str).str.upper() == value]

    frame = frame.dropna(subset=["gvkey", "datadate"]).copy()
    frame["gvkey"] = frame["gvkey"].astype(str)
    for column in ("datadate", "fdate", "pitdate1", "pitdate2"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")

    # One record per firm-fiscal-year: keep the newest PIT vintage.
    frame = frame.sort_values(["gvkey", "datadate"])
    frame = frame.drop_duplicates(["gvkey", "datadate"], keep="last")
    logger.info(f"Compustat: {len(frame)} firm-year records")
    return frame


def _load_links(uri: str) -> pd.DataFrame:
    schema = r2_store.dataset_schema(uri)
    columns = [c for c in LINK_COLUMNS if c in schema]
    missing = {"gvkey", "lpermno"} - set(columns)
    if missing:
        raise ValueError(f"CCM link table is missing {sorted(missing)}")

    links = r2_store.read_parquet_parts(uri, columns)
    links = links[links["lpermno"].notna()].copy()
    links["gvkey"] = links["gvkey"].astype(str)

    if "linktype" in links.columns:
        links = links[links["linktype"].isin(["LU", "LC"])]
    if "linkprim" in links.columns:
        links = links[links["linkprim"].isin(["P", "C"])]
    if "usedflag" in links.columns:
        links = links[links["usedflag"].fillna(1).astype(int) == 1]

    links["linkdt"] = pd.to_datetime(links["linkdt"], errors="coerce").fillna(
        pd.Timestamp("1900-01-01")
    )
    links["linkenddt"] = pd.to_datetime(links["linkenddt"], errors="coerce").fillna(
        pd.Timestamp("2262-01-01")
    )
    logger.info(f"CCM: {len(links)} gvkey-permno links")
    return links


def _scatter(
    values: pd.Series,
    lengths: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Write each record's value to every month it is valid for.

    ``rows``/``cols`` are already expanded to one entry per (month, security)
    cell, so the per-record values must be repeated by ``lengths`` to match.
    Records are sorted by validity start, so a later vintage overwrites an
    earlier one in a shared cell.
    """
    array = np.full(shape, np.nan, dtype=np.float32)
    flat = array.reshape(-1)
    flat[rows * shape[1] + cols] = np.repeat(
        values.to_numpy(dtype=np.float32), lengths
    )
    return array


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = numerator / denominator
    out[~np.isfinite(out)] = np.nan
    return out


def load_fundamentals_panels(
    fundamentals_uri: str,
    ccm_uri: str,
    dates: pd.Index,
    securities: pd.Index,
    market_cap: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Build monthly fundamental panels aligned to the CRSP grid."""
    if not ccm_uri:
        raise ValueError("ccm_uri is required to map gvkey to permno")

    dates = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()
    columns = pd.Index([str(s) for s in securities])
    # The CRSP bridge names securities ``crsp:permno:10000``; Compustat/CCM
    # carry the bare permno. Key on the bare id and keep the original labels.
    column_of = {name.split(":")[-1]: position for position, name in enumerate(columns)}

    records = _load_records(fundamentals_uri)
    links = _load_links(ccm_uri)
    merged = records.merge(links, on="gvkey", how="inner")
    if merged.empty:
        raise ValueError("Compustat and CCM share no gvkey values")

    merged["permno"] = merged["lpermno"].astype("int64").astype(str)
    merged = merged[merged["permno"].isin(column_of)]
    if merged.empty:
        raise ValueError("No Compustat issuer maps to a security in the CRSP panel")

    # Reporting-validity window, intersected with the link's validity window.
    if "pitdate1" in merged.columns:
        start = pd.to_datetime(merged["pitdate1"], errors="coerce")
    elif "fdate" in merged.columns:
        start = pd.to_datetime(merged["fdate"], errors="coerce")
    else:
        start = pd.Series(pd.NaT, index=merged.index, dtype="datetime64[ns]")
    start = start.fillna(merged["datadate"] + pd.Timedelta(days=180))

    if "pitdate2" in merged.columns:
        end = pd.to_datetime(merged["pitdate2"], errors="coerce")
    else:
        end = pd.Series(pd.NaT, index=merged.index, dtype="datetime64[ns]")
    end = end.fillna(start + pd.Timedelta(days=365))

    merged["valid_from"] = pd.concat([start, merged["linkdt"]], axis=1).max(axis=1)
    merged["valid_to"] = pd.concat([end, merged["linkenddt"]], axis=1).min(axis=1)
    merged = merged.sort_values("valid_from")

    left = np.searchsorted(dates.values, merged["valid_from"].to_numpy(), side="left")
    right = np.searchsorted(dates.values, merged["valid_to"].to_numpy(), side="right") - 1
    lengths = right - left + 1

    keep = (lengths > 0) & merged["permno"].notna().to_numpy()
    left = left[keep]
    lengths = lengths[keep]
    keep_records = merged.loc[keep]
    col_index = keep_records["permno"].map(column_of).to_numpy(dtype=np.int64)

    total = int(lengths.sum())
    if total > _MAX_SCATTER_CELLS:
        raise ValueError(
            f"fundamental scatter would allocate {total:,} cells; narrow the window"
        )
    offsets = np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    rows = np.repeat(left, lengths) + offsets
    cols = np.repeat(col_index, lengths)
    logger.info(
        f"Fundamentals: {len(keep_records)} records expanded to {total:,} "
        f"(month, security) cells"
    )

    shape = (len(dates), len(columns))

    def panel(values: pd.Series) -> np.ndarray:
        return _scatter(values, lengths, rows, cols, shape)

    def fresh(column: str) -> np.ndarray:
        if column not in keep_records.columns:
            return np.full(shape, np.nan, dtype=np.float32)
        return panel(keep_records[column])

    at = fresh("at")
    dltt = fresh("dltt")
    dlc = fresh("dlc")
    lag_at = _lagged(at)
    lag_ceq = _lagged(fresh("ceq")) if "ceq" in keep_records.columns else np.full(shape, np.nan, dtype=np.float32)

    # Book equity: reported common equity, else assets minus debt.
    if "ceq" in keep_records.columns:
        book_equity = keep_records["ceq"].fillna(
            keep_records["at"]
            - keep_records["dltt"].fillna(0.0)
            - keep_records["dlc"].fillna(0.0)
        )
    else:
        book_equity = keep_records["at"] - keep_records["dltt"].fillna(
            0.0
        ) - keep_records["dlc"].fillna(0.0)
    be = panel(book_equity)

    panels: dict[str, pd.DataFrame] = {}

    def add(name: str, array: np.ndarray) -> None:
        frame = pd.DataFrame(array, index=dates, columns=columns)
        if frame.notna().any().any():
            panels[name] = frame

    if market_cap is not None:
        cap = market_cap.reindex(index=dates, columns=columns).to_numpy(dtype=np.float32)
        # Compustat $ millions -> CRSP $ thousands.
        add("BookToMarket", _ratio(be * 1000.0, cap))
        ni = fresh("ni")
        add("EarningsYield", _ratio(ni * 1000.0, cap))
        del ni
    del be

    sale = fresh("sale")
    cogs = fresh("cogs")
    add("GrossProfitability", _ratio(sale - cogs, lag_at))
    add("SalesToAssets", _ratio(sale, lag_at))
    add("AssetTurnover", _ratio(sale, lag_at))
    del sale, cogs

    oancf = fresh("oancf")
    ni = fresh("ni")
    add("Accruals", _ratio(ni - oancf, at))
    add("OperatingProfitability", _ratio(fresh("oi"), lag_at)) if "oi" in keep_records.columns else None
    add("Roa", _ratio(ni, lag_at))
    add("Roe", _ratio(ni, lag_ceq))
    del oancf, ni

    # A missing short-term debt item must not erase a filer that reports
    # long-term debt (and vice versa); the old sum propagated NaN.
    debt = np.where(
        np.isnan(dltt), dlc,
        np.where(np.isnan(dlc), dltt, dltt + dlc),
    )
    add("Leverage", _ratio(debt, at))
    del dltt, dlc

    if "csho" in keep_records.columns:
        add("ShareIssuance", _ratio(fresh("csho"), _lagged(fresh("csho"))) - 1.0)

    growth = np.full(shape, np.nan, dtype=np.float32)
    if shape[0] > 12:
        with np.errstate(divide="ignore", invalid="ignore"):
            growth[12:] = at[12:] / at[:-12] - 1.0
        growth[~np.isfinite(growth)] = np.nan
    add("AssetGrowth", growth)

    if "xsga" in keep_records.columns:
        add("SgaToSales", _ratio(fresh("xsga"), fresh("sale")))
    if "xrd" in keep_records.columns:
        add("RndToSales", _ratio(fresh("xrd"), fresh("sale")))
    if "capx" in keep_records.columns:
        add("CapexToAssets", _ratio(fresh("capx"), at))

    del at
    if not panels:
        raise ValueError("Fundamentals loaded but produced no usable panel")
    logger.info("Fundamentals panels: " + ", ".join(sorted(panels)))
    return panels


# ---------------------------------------------------------------------------
# Lakehouse v2 pre-joined monthly PIT fundamentals
# (gold.security_monthly_fundamentals_v2 in financial-data-wrds-lakehouse)
# ---------------------------------------------------------------------------

LAKEHOUSE_V2_COLUMNS = [
    "security_id",
    "permno",
    "date",
    "market_cap",
    "revenue",
    "operating_income",
    "net_income",
    "eps",
    "assets",
    "liabilities",
    "equity",
    "cash_and_equivalents",
    "capex",
    "operating_cash_flow",
    "debt_total",
]

DEFAULT_LAKEHOUSE_V2_URI = (
    "r2://financial-data-wrds-lakehouse/__r2_data_catalog/"
    "01a0cf99-54a7-7e92-b0ea-6453c8e0b34b/"
    "01a0e5a6-7ed3-7540-801b-24ec2613cece/data/"
)


def build_lakehouse_v2_panels(
    frame: pd.DataFrame,
    dates: pd.Index,
    securities: pd.Index,
    market_cap: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Turn ``gold.security_monthly_fundamentals_v2`` rows into monthly ratio panels.

    The v2 Iceberg table is already point-in-time joined on ``(security_id, date)``
    with cash-flow items (``operating_cash_flow``, ``capex``, ``cash_and_equivalents``,
    ``debt_total``) that the annual-only ``companyfacts`` release lacks.
    """
    if frame.empty:
        raise ValueError("Lakehouse v2 fundamentals table is empty")

    dates = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()
    columns = pd.Index([str(s) for s in securities])
    col_by_id = {str(s): idx for idx, s in enumerate(columns)}
    col_by_permno = {str(s).split(":")[-1]: idx for idx, s in enumerate(columns)}
    date_to_row = {d: idx for idx, d in enumerate(dates)}

    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if "security_id" in df.columns:
        col_idx = df["security_id"].astype(str).map(col_by_id)
    else:
        col_idx = pd.Series(np.nan, index=df.index)
    if "permno" in df.columns and col_idx.isna().any():
        permno_str = (
            df["permno"]
            .dropna()
            .astype("int64")
            .astype(str)
            .reindex(df.index)
        )
        col_idx = col_idx.fillna(permno_str.map(col_by_permno))

    # Detect grid frequency so sub-monthly grids (e.g. weekly W-FRI) receive
    # point-in-time forward-filled monthly fundamentals and a true 1-year lag
    # (52 weeks / 252 days) rather than exact month-end date matches (~1/7 hit rate).
    if len(dates) >= 2:
        med_days = float(np.median(np.diff(dates.values).astype("timedelta64[D]").astype(float)))
    else:
        med_days = 30.0
    is_submonthly = med_days < 20.0
    lag_rows = 252 if med_days <= 2.0 else (52 if med_days <= 10.0 else 12)
    shape = (len(dates), len(columns))

    if is_submonthly:
        df = df.loc[col_idx.notna()].copy()
        if df.empty:
            raise ValueError("No Lakehouse v2 rows aligned to the market panel securities")
        df["_col"] = col_idx.loc[df.index].to_numpy(dtype=np.int64)
        df = df.sort_values("date")
        valid_from = df["date"].to_numpy()
        valid_to = (df["date"] + pd.Timedelta(days=45)).to_numpy()
        left = np.searchsorted(dates.values, valid_from, side="left")
        right = np.searchsorted(dates.values, valid_to, side="right") - 1
        lengths = right - left + 1
        keep_mask = lengths > 0
        if not keep_mask.any():
            raise ValueError("No Lakehouse v2 rows aligned to the market panel dates")
        left = left[keep_mask]
        lengths = lengths[keep_mask]
        kept = df.loc[keep_mask]
        col_arr = kept["_col"].to_numpy(dtype=np.int64)
        total_cells = int(lengths.sum())
        if total_cells > _MAX_SCATTER_CELLS:
            raise ValueError(f"Lakehouse v2 scatter would allocate {total_cells:,} cells")
        offsets = np.arange(total_cells) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        rows = np.repeat(left, lengths) + offsets
        cols = np.repeat(col_arr, lengths)

        def pivot_col(name: str) -> np.ndarray:
            if name not in kept.columns:
                return np.full(shape, np.nan, dtype=np.float32)
            vals = pd.to_numeric(kept[name], errors="coerce").astype("float32")
            return _scatter(vals, lengths, rows, cols, shape)
    else:
        row_idx = df["date"].map(date_to_row)
        keep = row_idx.notna() & col_idx.notna()
        df = df.loc[keep].copy()
        if df.empty:
            raise ValueError("No Lakehouse v2 rows aligned to the market panel dates/securities")
        rows = row_idx.loc[keep].to_numpy(dtype=np.int64)
        cols = col_idx.loc[keep].to_numpy(dtype=np.int64)

        def pivot_col(name: str) -> np.ndarray:
            arr = np.full(shape, np.nan, dtype=np.float32)
            if name not in df.columns:
                return arr
            vals = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=np.float32)
            flat = arr.reshape(-1)
            flat[rows * shape[1] + cols] = vals
            return arr

    assets = pivot_col("assets")
    lag_assets = _lagged(assets, rows=lag_rows)
    equity = pivot_col("equity")
    lag_equity = _lagged(equity, rows=lag_rows)
    revenue = pivot_col("revenue")
    lag_revenue = _lagged(revenue, rows=lag_rows)
    net_income = pivot_col("net_income")
    operating_income = pivot_col("operating_income")
    liabilities = pivot_col("liabilities")
    cash = pivot_col("cash_and_equivalents")
    capex = np.abs(pivot_col("capex"))
    cfo = pivot_col("operating_cash_flow")
    debt = pivot_col("debt_total")

    if market_cap is not None:
        cap = market_cap.reindex(index=dates, columns=columns).to_numpy(dtype=np.float32)
    else:
        cap = pivot_col("market_cap")

    debt_zero = np.where(np.isnan(debt), 0.0, debt)
    cash_zero = np.where(np.isnan(cash), 0.0, cash)
    ev = cap + debt_zero - cash_zero
    ev = np.where(ev > 0, ev, np.nan)

    fcf = cfo - np.where(np.isnan(capex), 0.0, capex)
    fcf = np.where(np.isnan(cfo), np.nan, fcf)

    panels: dict[str, pd.DataFrame] = {}

    def add(name: str, array: np.ndarray) -> None:
        frame_out = pd.DataFrame(array, index=dates, columns=columns)
        if frame_out.notna().any().any():
            panels[name] = frame_out

    pos_cap = np.where(cap > 0, cap, np.nan)
    add("BookToMarket", _ratio(equity, pos_cap))
    add("EarningsYield", _ratio(net_income, pos_cap))
    add("FcfYield", _ratio(fcf, pos_cap))
    add("EbitToEv", _ratio(operating_income, ev))

    pos_lag_assets = np.where(lag_assets > 0, lag_assets, np.nan)
    pos_lag_equity = np.where(lag_equity > 0, lag_equity, np.nan)
    pos_assets = np.where(assets > 0, assets, np.nan)
    pos_rev = np.where(revenue > 0, revenue, np.nan)
    pos_lag_rev = np.where(lag_revenue > 0, lag_revenue, np.nan)

    add("OperatingProfitability", _ratio(operating_income, pos_lag_assets))
    add("Roa", _ratio(net_income, pos_lag_assets))
    add("Roe", _ratio(net_income, pos_lag_equity))
    add("SalesToAssets", _ratio(revenue, pos_lag_assets))
    add("CashFlowToAssets", _ratio(cfo, pos_lag_assets))
    add("Accruals", _ratio(net_income - cfo, pos_assets))
    add("OperatingMargin", _ratio(operating_income, pos_rev))
    add("NetMargin", _ratio(net_income, pos_rev))

    add("Leverage", _ratio(liabilities, pos_assets))
    add("DebtToAssets", _ratio(debt, pos_assets))
    add("BookEquityToAssets", _ratio(equity, pos_assets))
    add("CashToAssets", _ratio(cash, pos_assets))
    add("CapexToAssets", _ratio(capex, pos_lag_assets))
    add("AssetGrowth", _ratio(assets, pos_lag_assets) - 1.0)
    add("RevenueGrowth", _ratio(revenue, pos_lag_rev) - 1.0)

    if not panels:
        raise ValueError("Lakehouse v2 fundamentals produced no usable panels")
    logger.info(f"Lakehouse v2 fundamental panels ({len(panels)}): " + ", ".join(sorted(panels)))
    return panels


def load_lakehouse_v2_fundamentals_panels(
    uri: str,
    dates: pd.Index,
    securities: pd.Index,
    market_cap: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Read ``gold.security_monthly_fundamentals_v2`` from R2 in place."""
    available = r2_store.dataset_schema(uri)
    cols = [c for c in LAKEHOUSE_V2_COLUMNS if c in available]
    if "date" not in cols or not ({"security_id", "permno"} & set(cols)):
        raise ValueError(f"Lakehouse v2 dataset at {uri} missing key columns: {sorted(available)}")
    frame = r2_store.read_parquet_parts(uri, cols)
    return build_lakehouse_v2_panels(frame, dates, securities, market_cap=market_cap)


# ---------------------------------------------------------------------------
# Quarterly SEC acceleration panels (fundamentals.sec_quarterly, 6.77M rows)
# ---------------------------------------------------------------------------

DEFAULT_SEC_QUARTERLY_URI = (
    "r2://financial-data-wrds-lakehouse/__r2_data_catalog/"
    "01a0cf99-54a7-7e92-b0ea-6453c8e0b34b/"
    "01a0d45e-84e1-7333-a610-9b70beb81488/data/"
)

_QUARTERLY_METRICS = {"revenue", "net_income", "operating_income", "equity"}


def build_sec_quarterly_acceleration_panels(
    quarterly: pd.DataFrame,
    bridge: pd.DataFrame,
    dates: pd.Index,
    securities: pd.Index,
) -> dict[str, pd.DataFrame]:
    """Build point-in-time QoQ/YoY earnings & revenue acceleration panels."""
    if quarterly.empty:
        return {}

    dates = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()
    columns = pd.Index([str(s) for s in securities])
    column_of = {name.split(":")[-1]: idx for idx, name in enumerate(columns)}

    df = quarterly.dropna(
        subset=["cik", "source_period_end", "information_available_date", "value"]
    ).copy()
    df["canonical_metric"] = df["canonical_metric"].astype(str).str.lower()
    df = df[df["canonical_metric"].isin(_QUARTERLY_METRICS)]
    if df.empty:
        return {}

    df["cik"] = _normalize_cik(df["cik"])
    df["source_period_end"] = pd.to_datetime(df["source_period_end"], errors="coerce")
    df["available"] = pd.to_datetime(df["information_available_date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["source_period_end", "available", "value"])

    df = df.sort_values("available").drop_duplicates(
        ["cik", "source_period_end", "canonical_metric"], keep="first"
    )
    wide = df.pivot_table(
        index=["cik", "source_period_end"],
        columns="canonical_metric",
        values="value",
        aggfunc="first",
    ).reset_index()
    avail = (
        df.groupby(["cik", "source_period_end"])["available"]
        .max()
        .rename("available")
        .reset_index()
    )
    wide = wide.merge(avail, on=["cik", "source_period_end"], how="left")
    wide = wide.sort_values(["cik", "source_period_end"])

    g = wide.groupby("cik", sort=False)
    if "revenue" in wide.columns:
        rev = wide["revenue"]
        rev_yoy = (rev - g["revenue"].shift(4)) / g["revenue"].shift(4).abs().replace(0.0, np.nan)
        wide["revenue_accel"] = rev_yoy - rev_yoy.groupby(wide["cik"]).shift(1)
    if {"net_income", "revenue"} <= set(wide.columns):
        ni_yoy = (
            (wide["net_income"] - g["net_income"].shift(4))
            / g["revenue"].shift(4).abs().replace(0.0, np.nan)
        )
        wide["earnings_accel"] = ni_yoy - ni_yoy.groupby(wide["cik"]).shift(1)
    if {"operating_income", "revenue"} <= set(wide.columns):
        margin = wide["operating_income"] / wide["revenue"].abs().replace(0.0, np.nan)
        wide["margin_delta"] = margin - margin.groupby(wide["cik"]).shift(4)
    if {"net_income", "equity"} <= set(wide.columns):
        q_roe = wide["net_income"] / wide["equity"].where(wide["equity"] > 0)
        wide["q_roe_delta"] = q_roe - q_roe.groupby(wide["cik"]).shift(4)

    merged = wide.merge(bridge, on="cik", how="inner")
    merged = merged[
        (merged["available"] >= merged["effective_from"])
        & (merged["available"] <= merged["effective_to"])
    ]
    merged = merged[merged["permno"].isin(column_of)]
    if merged.empty:
        return {}

    merged = merged.sort_values("available")
    merged["valid_from"] = merged["available"]
    merged["valid_to"] = merged["available"] + pd.Timedelta(days=135)

    left = np.searchsorted(dates.values, merged["valid_from"].to_numpy(), side="left")
    right = np.searchsorted(dates.values, merged["valid_to"].to_numpy(), side="right") - 1
    lengths = right - left + 1
    keep = lengths > 0
    if not keep.any():
        return {}

    left, lengths = left[keep], lengths[keep]
    kept = merged.loc[keep]
    col_index = kept["permno"].map(column_of).to_numpy(dtype=np.int64)
    total = int(lengths.sum())
    if total > _MAX_SCATTER_CELLS:
        raise ValueError(f"SEC quarterly scatter would allocate {total:,} cells")
    offsets = np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    rows = np.repeat(left, lengths) + offsets
    cols = np.repeat(col_index, lengths)
    shape = (len(dates), len(columns))

    mapping = {
        "revenue_accel": "RevenueAccel",
        "earnings_accel": "EarningsAccel",
        "margin_delta": "MarginDelta",
        "q_roe_delta": "QuarterlyRoeDelta",
    }
    panels: dict[str, pd.DataFrame] = {}
    for src_col, panel_name in mapping.items():
        if src_col not in kept.columns:
            continue
        vals = (
            pd.to_numeric(kept[src_col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .astype("float32")
        )
        arr = _scatter(vals, lengths, rows, cols, shape)
        frame = pd.DataFrame(arr, index=dates, columns=columns)
        if frame.notna().any().any():
            panels[panel_name] = frame

    if panels:
        logger.info("SEC quarterly acceleration panels: " + ", ".join(sorted(panels)))
    return panels


def load_sec_quarterly_acceleration_panels(
    quarterly_uri: str,
    bridge_uri: str,
    dates: pd.Index,
    securities: pd.Index,
) -> dict[str, pd.DataFrame]:
    """Read ``fundamentals.sec_quarterly`` from R2 with metric pushdown."""
    schema = r2_store.dataset_schema(quarterly_uri)
    cols = [
        c
        for c in (
            "cik",
            "source_period_end",
            "information_available_date",
            "canonical_metric",
            "value",
        )
        if c in schema
    ]
    frame = r2_store.read_parquet_parts(
        quarterly_uri,
        cols,
        filters=[("canonical_metric", "in", sorted(_QUARTERLY_METRICS))],
    )
    bridge = _load_cik_bridge(bridge_uri)
    return build_sec_quarterly_acceleration_panels(frame, bridge, dates, securities)
