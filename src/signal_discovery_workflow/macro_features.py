# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0

"""
Macro regime panels (FRED) and factor-return benchmarks (Free Resource Lake).

Both datasets are read in place from Cloudflare R2 via ``r2_store``:

* **FRED macro regimes** (``gold/fred/CURRENT.json`` in
  ``financial-system-datasets``): ``VIXCLS`` (1990+), ``T10Y3M`` (1982+), and
  ``NFCI`` (1971+). Series are resampled to the market-panel frequency, lagged
  by one period so there is zero look-ahead, and broadcast across securities so
  standard ``TS_*`` / ``Mul`` / ``If_Then_Else`` operators can build
  regime-conditional signals. (``BAMLH0A0HYM2`` is intentionally excluded from
  historical panels because the free FRED graph CSV only begins in Nov 2023.)
* **Factor-return benchmarks** (``free-resource-lake/parquet/factor_returns/``
  in ``financial-system-datasets``): AQR and JKP factor return tables used to
  regress a candidate strategy's net long/short spread against known risk
  factors and gate on residual alpha t-stat.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from . import r2_store
from .signal_evaluator import newey_west_t

logger = logging.getLogger(__name__)

# Multi-decade FRED series verified in gold/fred/CURRENT.json.
FRED_SERIES_TO_PANEL = {
    "VIXCLS": "MacroVIX",
    "T10Y3M": "MacroYieldCurve",
    "NFCI": "MacroNFCI",
}

DEFAULT_FRED_CURRENT_URI = "r2://financial-system-datasets/gold/fred/CURRENT.json"
DEFAULT_FACTOR_RETURNS_URI = (
    "r2://financial-system-datasets/free-resource-lake/parquet/factor_returns/"
)

# Preferred canonical factor names when a long-format factor table carries many
# factors (AQR/JKP). Matching is case-insensitive on normalized factor keys.
_PREFERRED_FACTORS = (
    "mkt",
    "mktrf",
    "mkt_rf",
    "market",
    "smb",
    "size",
    "hml",
    "value",
    "rmw",
    "profitability",
    "qmj",
    "cma",
    "investment",
    "umd",
    "mom",
    "momentum",
    "bab",
)


def resolve_fred_release_uri(current_uri: str = DEFAULT_FRED_CURRENT_URI) -> str:
    """Resolve ``gold/fred/CURRENT.json`` to its release directory URI."""
    if not current_uri.endswith(".json"):
        return current_uri
    payload = json.loads(r2_store.read_text(current_uri))
    manifest_key = payload.get("manifest")
    if not manifest_key:
        raise ValueError(f"FRED pointer {current_uri} is missing 'manifest'")
    bucket_root = current_uri.split("://", 1)[0] + "://" + current_uri.split("://", 1)[1].split("/", 1)[0]
    release_dir = manifest_key.rsplit("/", 1)[0]
    return r2_store.join(bucket_root, release_dir)


def build_macro_panels(
    fred_frame: pd.DataFrame,
    dates: pd.Index,
    securities: pd.Index,
    lag_periods: int = 1,
) -> dict[str, pd.DataFrame]:
    """Align FRED macro series onto ``(dates, securities)`` with a strict lag.

    Separated from the R2 reader so look-ahead protection and alignment are
    testable without a bucket.
    """
    if fred_frame.empty:
        return {}

    dates = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()
    columns = pd.Index([str(s) for s in securities])

    df = fred_frame.copy()
    if "missing" in df.columns:
        df = df[~df["missing"].fillna(False).astype(bool)]
    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["series_id", "observation_date", "value"])

    panels: dict[str, pd.DataFrame] = {}
    for series_id, panel_name in FRED_SERIES_TO_PANEL.items():
        sub = df[df["series_id"].astype(str).str.upper() == series_id]
        if sub.empty:
            continue
        series = (
            sub.sort_values("observation_date")
            .drop_duplicates("observation_date", keep="last")
            .set_index("observation_date")["value"]
            .astype("float32")
        )
        # Reindex using last known observation on or before each panel date,
        # then shift forward by lag_periods so date t only sees t - lag.
        aligned = series.reindex(dates, method="ffill")
        if lag_periods > 0:
            aligned = aligned.shift(lag_periods)
        if not aligned.notna().any():
            continue
        arr = np.broadcast_to(
            aligned.to_numpy(dtype=np.float32)[:, None],
            (len(dates), len(columns)),
        ).copy()
        panels[panel_name] = pd.DataFrame(arr, index=dates, columns=columns)

    if panels:
        logger.info("Macro regime panels loaded: " + ", ".join(sorted(panels)))
    return panels


def load_macro_panels(
    fred_uri: str,
    dates: pd.Index,
    securities: pd.Index,
    lag_periods: int = 1,
) -> dict[str, pd.DataFrame]:
    """Read FRED macro regimes from R2 and broadcast onto ``(dates, securities)``."""
    release_uri = resolve_fred_release_uri(fred_uri)
    schema = r2_store.dataset_schema(release_uri)
    cols = [c for c in ("series_id", "observation_date", "value", "missing") if c in schema]
    if not {"series_id", "observation_date", "value"} <= set(cols):
        raise ValueError(f"FRED release at {release_uri} missing required columns: {sorted(schema)}")
    frame = r2_store.read_parquet_parts(release_uri, cols)
    return build_macro_panels(frame, dates, securities, lag_periods=lag_periods)


# ---------------------------------------------------------------------------
# Factor-return benchmark loader & residual alpha regression
# ---------------------------------------------------------------------------

# One coherent US equity factor model. JKP publishes these as long/short
# characteristic portfolios over the full 1926-2025 span. Mixing in the AQR
# ``qmj``/``bab`` rows would silently average two different constructions of
# the same factor name; the loader picks one dataset before pivoting.
US_EQUITY_FACTOR_MODEL = ("be_me", "ret_12_1", "cop_at", "at_gr1", "beta_60m", "qmj")


def _select_factor_dataset(
    frame: pd.DataFrame,
    factor_names: tuple[str, ...],
) -> pd.DataFrame:
    """Keep the US monthly dataset that covers the most requested factors.

    The release is a union of AQR and JKP exports. Pivoting both together would
    average duplicate factor names from different constructions and cover
    different date ranges, so one dataset has to win before the pivot.
    """
    cols = {c.lower(): c for c in frame.columns}
    if not {"factor"} <= set(cols):
        return frame
    region_col = cols.get("region") or cols.get("location") or cols.get("country")
    if region_col is not None:
        region = frame[region_col].astype(str).str.upper()
        us = frame[region.isin({"USA", "US", "UNITED STATES"})]
        if not us.empty:
            frame = us
    if not {"source", "dataset"} <= set(cols):
        return frame

    wanted = {name.lower() for name in factor_names}
    coverage = (
        frame.assign(
            _factor=frame[cols["factor"]].astype(str).str.lower(),
            _source=frame[cols["source"]].astype(str),
            _dataset=frame[cols["dataset"]].astype(str),
        )
        .assign(_hit=lambda d: d["_factor"].isin(wanted))
        .groupby(["_source", "_dataset"])["_hit"]
        .sum()
    )
    if coverage.empty or int(coverage.max()) == 0:
        logger.info("Factor release has no US dataset covering the requested model; using all rows")
        return frame
    source, dataset = coverage.idxmax()
    chosen = frame[
        (frame[cols["source"]].astype(str) == source)
        & (frame[cols["dataset"]].astype(str) == dataset)
    ]
    logger.info(
        f"Factor dataset selected: {source}/{dataset} "
        f"({int(coverage.max())}/{len(factor_names)} model factors)"
    )
    return chosen


def normalize_factor_returns_frame(
    frame: pd.DataFrame,
    max_factors: int = 6,
    factor_names: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Convert a long or wide factor-return table into a clean ``date x factor`` DataFrame."""
    if frame.empty:
        return pd.DataFrame()

    cols_lower = {c.lower(): c for c in frame.columns}
    date_col = next(
        (cols_lower[k] for k in ("date", "month", "observation_date", "eom", "period_end") if k in cols_lower),
        None,
    )
    if date_col is None:
        return pd.DataFrame()

    df = frame.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # Filter to US / monthly rows when those metadata columns exist (JKP/AQR).
    for region_key in ("location", "region", "country", "market"):
        if region_key in cols_lower:
            rc = cols_lower[region_key]
            us_mask = df[rc].astype(str).str.upper().isin({"USA", "US", "UNITED STATES", "GLOBAL_EX_US_FALSE"})
            if us_mask.any():
                df = df[us_mask]
            break
    for freq_key in ("freq", "frequency"):
        if freq_key in cols_lower:
            fc = cols_lower[freq_key]
            m_mask = df[fc].astype(str).str.lower().isin({"m", "monthly", "month"})
            if m_mask.any():
                df = df[m_mask]
            break

    # Long format: (date, factor_name, ret)
    name_col = next(
        (cols_lower[k] for k in ("factor", "factor_name", "name", "series_id", "characteristic") if k in cols_lower),
        None,
    )
    val_col = next(
        (cols_lower[k] for k in ("ret", "return", "value", "ret_vw", "ret_ew") if k in cols_lower),
        None,
    )
    if name_col and val_col:
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        df = df.dropna(subset=[name_col, val_col])
        if factor_names:
            wanted = {name.lower() for name in factor_names}
            df = df[df[name_col].astype(str).str.lower().isin(wanted)]
            if df.empty:
                return pd.DataFrame()
        wide = df.pivot_table(index=date_col, columns=name_col, values=val_col, aggfunc="mean")
    else:
        numeric_cols = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            return pd.DataFrame()
        wide = df.groupby(date_col)[numeric_cols].mean()

    wide = wide.sort_index()
    # Select up to max_factors canonical factors if the table is very wide.
    if wide.shape[1] > max_factors:
        chosen: list[str] = []
        norm_map = {str(c).lower().replace("-", "_").replace(" ", "_"): c for c in wide.columns}
        for pref in _PREFERRED_FACTORS:
            for norm_key, orig in norm_map.items():
                if orig in chosen:
                    continue
                if norm_key == pref or norm_key.endswith(f"_{pref}") or norm_key.startswith(f"{pref}_"):
                    chosen.append(orig)
                    break
            if len(chosen) >= max_factors:
                break
        if not chosen:
            # Fall back to the columns with highest non-null coverage.
            chosen = list(wide.notna().sum().sort_values(ascending=False).head(max_factors).index)
        wide = wide[chosen]

    # Align monthly dates to month-end so CRSP monthly dates join cleanly.
    wide.index = pd.DatetimeIndex(wide.index).to_period("M").to_timestamp("M")
    wide = wide.groupby(level=0).mean()
    return wide


def load_factor_returns(
    uri: str = DEFAULT_FACTOR_RETURNS_URI,
    max_factors: int = 6,
    factor_names: tuple[str, ...] = US_EQUITY_FACTOR_MODEL,
) -> pd.DataFrame:
    """Load monthly US factor return benchmarks from R2."""
    schema = r2_store.dataset_schema(uri)
    # Read only the identifying/value columns: the release carries per-dataset
    # metadata columns that differ between exports (``sheet``, ``weighting``).
    wanted_columns = [
        c
        for c in ("source", "dataset", "factor", "region", "date", "frequency", "value", "unit")
        if c in schema
    ]
    frame = r2_store.read_parquet_parts(uri, wanted_columns)
    frame = _select_factor_dataset(frame, factor_names)
    factors = normalize_factor_returns_frame(
        frame, max_factors=max_factors, factor_names=factor_names
    )
    if not factors.empty:
        logger.info(
            f"Loaded factor benchmarks ({factors.shape[0]} months x {factors.shape[1]} factors): "
            + ", ".join(str(c) for c in factors.columns)
        )
    return factors


def compute_factor_alpha(
    strategy_returns: pd.Series,
    factor_returns: pd.DataFrame | None,
    periods_per_year: int = 12,
    min_periods: int = 24,
) -> dict[str, Any]:
    """OLS factor-model regression of strategy net returns on benchmark factors.

    Returns annualized residual alpha, HAC t-statistic/p-value on the
    intercept, R-squared, and factor betas. Aligns monthly dates by calendar
    month so CRSP month-end dates match AQR/JKP month-end dates cleanly.
    """
    if factor_returns is None or factor_returns.empty or strategy_returns is None or strategy_returns.empty:
        return {"error": "no factor benchmarks available"}

    y = strategy_returns.dropna().copy()
    effective_periods_per_year = periods_per_year
    if periods_per_year == 12:
        y.index = pd.DatetimeIndex(y.index).to_period("M").to_timestamp("M")
        y = y.groupby(level=0).mean()
    elif periods_per_year > 12:
        # Compound higher-frequency (weekly/daily) net returns into monthly
        # returns: prod(1 + r) - 1 so they align with monthly JKP factor rows.
        y.index = pd.DatetimeIndex(y.index).to_period("M").to_timestamp("M")
        y = (1.0 + y).groupby(level=0).prod() - 1.0
        effective_periods_per_year = 12

    joined = pd.concat([y.rename("_y"), factor_returns], axis=1, join="inner").dropna()
    if len(joined) < min_periods:
        return {
            "error": f"insufficient overlapping periods with factor benchmarks ({len(joined)} < {min_periods})",
            "num_periods": int(len(joined)),
        }

    yv = joined["_y"].to_numpy(dtype=float)
    factor_cols = [c for c in joined.columns if c != "_y"]
    # Drop constant/degenerate factor columns inside this window.
    valid_cols = [c for c in factor_cols if float(joined[c].std(ddof=1)) > 1e-12]
    if not valid_cols:
        return {"error": "factor benchmarks have zero variance in window"}

    X_raw = joined[valid_cols].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(yv)), X_raw])
    coef, _, _, _ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ coef
    alpha_per_period = float(coef[0])
    betas = {str(col): float(b) for col, b in zip(valid_cols, coef[1:])}

    # Newey-West HAC covariance for OLS coefficients (lag = 3 for monthly).
    n, k = X.shape
    lags = min(3, max(0, n // 6))
    XtX_inv = np.linalg.pinv(X.T @ X)
    S = np.zeros((k, k), dtype=float)
    Xu = X * resid[:, None]
    S += Xu.T @ Xu
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        gamma = Xu[lag:].T @ Xu[:-lag]
        S += w * (gamma + gamma.T)
    cov = XtX_inv @ S @ XtX_inv
    se_alpha = float(np.sqrt(max(cov[0, 0], 0.0)))
    if se_alpha > 1e-12:
        alpha_t = float(alpha_per_period / se_alpha)
        alpha_p = float(2.0 * (1.0 - stats.norm.cdf(abs(alpha_t))))
    else:
        alpha_t, alpha_p, _ = newey_west_t(resid + alpha_per_period, lags)

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    return {
        "alpha_per_period": alpha_per_period,
        "alpha_annualized": alpha_per_period * effective_periods_per_year,
        "alpha_t_stat": alpha_t,
        "alpha_p_value": alpha_p,
        "r_squared": r2,
        "betas": betas,
        "factors_used": [str(c) for c in valid_cols],
        "num_periods": int(n),
    }
