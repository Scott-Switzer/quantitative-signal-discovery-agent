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

"""
Signal Evaluator: rank IC, HAC significance, purged splits, and cost-aware screening.

Why this file looks the way it does
-----------------------------------
The rank IC is the cross-sectional Spearman correlation between a signal and
forward returns, averaged over dates. Four things make the naive version of
that statistic lie:

1. **Overlapping labels.** A 5-day forward return sampled every day overlaps
   its neighbours by 4 days. The IC series is then autocorrelated and a naive
   ``mean / (std / sqrt(n))`` t-stat understates the standard error by roughly
   ``sqrt(horizon)``. Every t-stat here uses a Newey-West (Bartlett) long-run
   variance with a lag tied to the horizon. The naive number is still reported
   next to it so the inflation is visible.
2. **Selection on the test set.** Optimizing a closed loop against one holdout
   burns it. The split is train / validation / test, acceptance and model
   selection use validation, and the test window is reported once, untouched.
3. **Leakage across the split.** Forward returns computed at the end of train
   look into the validation window. A purge gap of ``horizon`` rows is removed
   at every boundary.
4. **No costs.** An IC with no turnover or cost is not an alpha. The screen
   builds the daily-rebalanced long/short portfolio implied by the signal and
   reports gross and net Sharpe, turnover, drawdown, and the IC decay curve.
"""

import inspect
import json
import logging
import re
import traceback
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Local CSV fallback only. The production path is an R2 panel URI passed to the
# workflow (data_source), which loads in memory and never touches this directory.
DATA_DIR = Path(__file__).parent / "data" / "sp500"
CALCULATOR_PATH = Path(__file__).parent / "template" / "calculator.json"
SECTORS_PATH = DATA_DIR / "sectors.csv"

TRADING_DAYS = 252
DEFAULT_MIN_NAMES = 10

# Fundamentals and ratios are optional panels. Keep these lists in sync with
# signal_generator.py; a missing file is a warning, not a failure, so the
# price-only path still works.
FUNDAMENTAL_FIELDS = [
    "Revenue",
    "NetIncome",
    "OperatingIncome",
    "Assets",
    "Liabilities",
    "Equity",
    "Cash",
    "LongTermDebt",
]
RATIO_FIELDS = [
    "BookToMarket",
    "EarningsYield",
    "FcfYield",
    "EbitToEv",
    "GrossProfitability",
    "AssetTurnover",
    "Accruals",
    "CashFlowToAssets",
    "AssetGrowth",
    "Leverage",
    "DebtToAssets",
    "SgaToSales",
    "RndToSales",
    "CapexToAssets",
    "ROE",
    "Roa",
    "Roe",
    "ROA",
    "OperatingProfitability",
    "SalesToAssets",
    "BookEquityToAssets",
    "ShareIssuance",
    "OperatingMargin",
    "NetMargin",
    "RevenueGrowth",
    "NetIncomeGrowth",
    "DebtToEquity",
    "LongTermDebtToEquity",
    "EquityToAssets",
    "CashToAssets",
    "EarningsAccel",
    "RevenueAccel",
    "MarginDelta",
    "QuarterlyRoeDelta",
    "FilingTextChange",
    "FilingJaccardDiff",
    "FilingLengthDelta",
    "AmihudIlliq",
    "IdioVol",
    "High52wRatio",
    "VolumeShock",
    "InsiderNetBuyRatio",
    "InsiderClusterBuys",
    "DiscretionaryBuyRatio",
    "InstOwnershipDelta",
    "InstCrowding",
]
MACRO_FIELDS = ["MacroVIX", "MacroYieldCurve", "MacroNFCI"]
OHLCV_FIELDS = ["Open", "Close", "High", "Low", "Volume", "MarketCap", "Returns"]


def _load_all_operators() -> dict[str, callable]:
    """Load and compile all operator functions from calculator.json into a callable dict."""
    if not CALCULATOR_PATH.exists():
        return {}
    with open(CALCULATOR_PATH, "r") as f:
        operators = json.load(f)
    op_namespace = {"pd": pd, "np": np}
    for op in operators:
        try:
            exec(op["code"], op_namespace)
        except Exception as e:
            logger.warning(f"Failed to compile operator {op.get('name')}: {e}")
    return {
        name: obj
        for name, obj in op_namespace.items()
        if callable(obj) and name not in ("pd", "np")
    }


_OPERATOR_FUNCTIONS = _load_all_operators()


def get_operator_arities() -> dict[str, tuple[int, int]]:
    """Return ``{operator_name: (min_required_args, max_args)}`` for every loaded operator.

    ``max_args`` is ``-1`` if the operator accepts ``*args``.
    Used by the code generator to validate formulas before execution.
    """
    arities: dict[str, tuple[int, int]] = {}
    for name, fn in _OPERATOR_FUNCTIONS.items():
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            continue
        required = sum(
            1 for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
        has_varargs = any(
            p.kind is inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
        )
        max_args = -1 if has_varargs else len([
            p for p in sig.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ])
        arities[name] = (required, max_args)
    return arities


def load_stock_data(data_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Load every available panel (OHLCV, fundamentals, ratios) from CSV files.

    Missing panels are logged at debug level, not warning: an OHLCV-only setup
    is valid, and logging 18 warnings on every run is noise.
    """
    directory = Path(data_dir) if data_dir else DATA_DIR
    data: dict[str, pd.DataFrame] = {}
    data_files = OHLCV_FIELDS + FUNDAMENTAL_FIELDS + RATIO_FIELDS

    for field in data_files:
        file_path = directory / f"{field}.csv"
        if not file_path.exists():
            continue
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            if df.empty:
                continue
            data[field] = df
            logger.debug(f"Loaded {field}.csv with shape {df.shape}")
        except Exception as e:
            logger.warning(f"Failed to load {field}.csv: {e}")

    if not data:
        logger.warning(f"No data panels found in {directory}. Run download_data first.")
    return data


def load_sector_map(sectors_path: str | Path | None = None) -> dict[str, str] | None:
    """Load the ticker -> GICS sector map.

    Returns ``None`` (with a warning) if the file is missing or unreadable, so
    callers can fall back to the raw (non-neutralized) IC.
    """
    path = Path(sectors_path) if sectors_path else SECTORS_PATH
    if not path.exists():
        logger.warning(f"Sector map not found: {path} (sector neutralization disabled)")
        return None
    try:
        df = pd.read_csv(path, dtype=str)
        mapping = {
            str(row["ticker"]).strip(): str(row["sector"]).strip()
            for _, row in df.iterrows()
        }
        known = sum(1 for v in mapping.values() if v and v != "Unknown")
        logger.info(f"Loaded sector map: {known}/{len(mapping)} tickers with a known sector")
        return mapping
    except Exception as e:
        logger.warning(f"Failed to load sector map: {e}")
        return None


# ---------------------------------------------------------------------------
# Forward returns and panels
# ---------------------------------------------------------------------------

def compute_forward_returns(close: pd.DataFrame, periods: int = 5) -> pd.DataFrame:
    """Forward total return over the next ``periods`` rows: ``P[t+n]/P[t] - 1``."""
    return close.shift(-periods) / close - 1


def _sector_neutralize_panel(
    signal_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    sector_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Demean both panels by sector within each date's cross-section.

    Columns with an unknown sector are dropped. If fewer than two sectors are
    represented, both panels come back empty so no neutralized IC is computed
    (there is nothing to neutralize against).
    """
    columns = list(signal_values.columns)
    if not columns:
        return signal_values, forward_returns

    codes = np.array([sector_map.get(c) for c in columns], dtype=object)
    known = np.array([c is not None and c != "Unknown" for c in codes], dtype=bool)
    if known.sum() == 0:
        return signal_values.iloc[:, :0], forward_returns.iloc[:, :0]

    known_positions = np.flatnonzero(known)
    _, inverse = np.unique(codes[known], return_inverse=True)
    sizes = np.bincount(inverse, minlength=inverse.max() + 1)
    if (sizes >= 2).sum() < 2:
        return signal_values.iloc[:, :0], forward_returns.iloc[:, :0]

    # Keep only sectors with at least two members, and drop everything else.
    keep = np.zeros(len(columns), dtype=bool)
    keep[known_positions] = sizes[inverse] >= 2
    positions = np.flatnonzero(keep)
    if len(positions) == 0:
        return signal_values.iloc[:, :0], forward_returns.iloc[:, :0]

    kept_columns = [columns[i] for i in positions]
    sig = signal_values.iloc[:, positions].to_numpy(dtype=float, copy=True)
    ret = forward_returns.iloc[:, positions].to_numpy(dtype=float, copy=True)

    _, inverse = np.unique(codes[positions], return_inverse=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for sector in range(inverse.max() + 1):
            members = inverse == sector
            block = sig[:, members]
            sig[:, members] = block - np.nanmean(block, axis=1, keepdims=True)
            block = ret[:, members]
            ret[:, members] = block - np.nanmean(block, axis=1, keepdims=True)

    return (
        pd.DataFrame(sig, index=signal_values.index, columns=kept_columns),
        pd.DataFrame(ret, index=forward_returns.index, columns=kept_columns),
    )


def _sector_demean_signal(
    signal_values: pd.DataFrame,
    sector_map: dict[str, str],
) -> pd.DataFrame:
    """Demean a signal within each date's sector, matching the IC's definition.

    The IC subtracts the sector mean from both the signal and the forward
    return. A portfolio built from the raw signal therefore trades a different
    object: a pure sector tilt can dominate the quintiles while the
    within-sector ordering is what the IC measures. Sectors with fewer than two
    members are dropped, as they are for the IC.
    """
    columns = list(signal_values.columns)
    codes = np.array([sector_map.get(c) for c in columns], dtype=object)
    known = np.array([c is not None and c != "Unknown" for c in codes], dtype=bool)
    if known.sum() == 0:
        return signal_values.iloc[:, :0]

    positions = np.flatnonzero(known)
    _, inverse = np.unique(codes[positions], return_inverse=True)
    sizes = np.bincount(inverse, minlength=inverse.max() + 1)
    keep = sizes[inverse] >= 2
    positions = positions[keep]
    inverse = inverse[keep]
    if len(positions) == 0:
        return signal_values.iloc[:, :0]

    values = signal_values.iloc[:, positions].to_numpy(dtype=float, copy=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for sector in range(inverse.max() + 1):
            members = inverse == sector
            block = values[:, members]
            values[:, members] = block - np.nanmean(block, axis=1, keepdims=True)
    return pd.DataFrame(
        values,
        index=signal_values.index,
        columns=[columns[i] for i in positions],
    )


def _cross_sectional_rank_ic(
    signal_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    sector_map: dict[str, str] | None = None,
    min_names: int = DEFAULT_MIN_NAMES,
    chunk_size: int = 128,
) -> pd.Series:
    """Per-date Spearman IC for aligned panels, vectorized and chunked.

    A row-wise Spearman is just a Pearson correlation of the row ranks, so the
    panel is ranked and correlated with matrix products instead of calling
    ``scipy.stats.spearmanr`` once per date.

    Dates are processed in chunks. Ranking the whole panel at once allocates
    several float64 copies of ``dates x securities``; on the 25k-name CRSP panel
    that is gigabytes and the process gets killed. A 128-date chunk is about
    26 MB, and the result is identical.
    """
    common_dates = signal_values.index.intersection(forward_returns.index)
    common_stocks = signal_values.columns.intersection(forward_returns.columns)
    if len(common_dates) == 0 or len(common_stocks) == 0:
        return pd.Series(dtype=float)

    sig_all = signal_values.loc[common_dates, common_stocks]
    ret_all = forward_returns.loc[common_dates, common_stocks]

    collected: list[pd.Series] = []
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for start in range(0, len(common_dates), chunk_size):
            stop = start + chunk_size
            sig = sig_all.iloc[start:stop].replace([np.inf, -np.inf], np.nan)
            ret = ret_all.iloc[start:stop].replace([np.inf, -np.inf], np.nan)
            if sector_map is not None:
                sig, ret = _sector_neutralize_panel(sig, ret, sector_map)
                if sig.shape[1] == 0:
                    continue

            valid = sig.notna() & ret.notna()
            counts = valid.sum(axis=1)

            sig_rank = sig.where(valid).rank(axis=1)
            ret_rank = ret.where(valid).rank(axis=1)

            sig_rank = sig_rank.sub(sig_rank.mean(axis=1), axis=0)
            ret_rank = ret_rank.sub(ret_rank.mean(axis=1), axis=0)

            numerator = (sig_rank * ret_rank).sum(axis=1)
            denominator = np.sqrt(
                (sig_rank.pow(2).sum(axis=1)) * (ret_rank.pow(2).sum(axis=1))
            )
            ic = numerator / denominator
            ic = ic.where((counts >= min_names) & (denominator > 1e-12))
            collected.append(ic.dropna())

    return pd.concat(collected) if collected else pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Significance
# ---------------------------------------------------------------------------

def newey_west_t(values: Iterable[float], lags: int) -> tuple[float | None, float | None, float]:
    """Newey-West (Bartlett) t-statistic and two-sided p-value for a mean.

    ``lags=0`` reproduces the naive i.i.d. t-test. For overlapping ``h``-period
    forward returns use ``lags >= h - 1``; twice the overlap is the safer
    convention and is what the caller uses by default.

    Returns ``(t_stat, p_value, long_run_variance)``.
    """
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return None, None, 0.0

    mean = float(x.mean())
    errors = x - mean
    lrv = float((errors * errors).mean())
    max_lag = min(int(lags), n - 1)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        autocov = float((errors[lag:] * errors[:-lag]).mean())
        lrv += 2.0 * weight * autocov

    if lrv <= 0:
        # A perfectly constant IC series: only a zero mean is a meaningful
        # statement. Anything else has no estimable standard error.
        if abs(mean) < 1e-12:
            return 0.0, 1.0, 0.0
        return None, None, lrv

    se = np.sqrt(lrv / n)
    t_stat = mean / se
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat))))
    return float(t_stat), p_value, lrv


def _auto_lags(periods: int, n_obs: int) -> int:
    """HAC lag for an ``h``-period overlapping label, capped by sample size."""
    if periods <= 1:
        return 0
    return int(min(2 * (periods - 1), max(0, n_obs // 4)))


def compute_rank_ic(
    signal_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    sector_map: dict[str, str] | None = None,
    periods: int = 1,
    min_names: int = DEFAULT_MIN_NAMES,
    hac_lags: int | None = None,
    periods_per_year: int = TRADING_DAYS,
) -> dict[str, Any]:
    """Rank IC with HAC significance.

    Args:
        signal_values: signal panel (rows=dates, cols=tickers).
        forward_returns: forward return panel, same shape.
        sector_map: optional ticker -> sector map for sector-neutral IC.
        periods: forward-return horizon in rows. Drives the Newey-West lag and
            the annualization of IC_IR. Pass the real horizon for overlapping
            labels; the default of 1 means "labels do not overlap".
        min_names: minimum cross-section size for a date to count.
        hac_lags: override the automatic lag (default ``2 * (periods - 1)``).

    Returns:
        Dict with ``mean_ic``, ``ic_std``, ``ic_ir``, ``ic_ir_annualized``,
        ``t_stat`` (HAC), ``p_value`` (HAC), ``t_stat_naive``,
        ``p_value_naive``, ``num_periods``, ``effective_periods``,
        ``hac_lags``, ``positive_ic_ratio`` and ``ic_percentiles``.
    """
    common_dates = signal_values.index.intersection(forward_returns.index)
    common_stocks = signal_values.columns.intersection(forward_returns.columns)

    empty = {
        "mean_ic": None,
        "ic_std": None,
        "ic_ir": None,
        "ic_ir_annualized": None,
        "t_stat": None,
        "p_value": None,
        "t_stat_naive": None,
        "p_value_naive": None,
        "num_periods": 0,
        "effective_periods": 0,
        "hac_lags": 0,
        "positive_ic_ratio": None,
        "ic_percentiles": None,
    }

    if len(common_dates) == 0 or len(common_stocks) == 0:
        return {**empty, "error": "No common dates or stocks between signal and returns"}

    sig = signal_values.loc[common_dates, common_stocks]
    ret = forward_returns.loc[common_dates, common_stocks]

    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        ic_series = _cross_sectional_rank_ic(
            sig, ret, sector_map=sector_map, min_names=min_names
        )

    if len(ic_series) == 0:
        return {**empty, "error": "Could not compute IC for any period"}

    return _summarize_ic_series(ic_series, periods, periods_per_year, hac_lags)


def _ic_series_for(
    signal_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    sector_map: dict[str, str] | None,
    min_names: int = DEFAULT_MIN_NAMES,
) -> pd.Series:
    """Per-date rank IC (optionally sector-neutral) for two aligned panels."""
    common_dates = signal_values.index.intersection(forward_returns.index)
    common_stocks = signal_values.columns.intersection(forward_returns.columns)
    if len(common_dates) == 0 or len(common_stocks) == 0:
        return pd.Series(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return _cross_sectional_rank_ic(
            signal_values.loc[common_dates, common_stocks],
            forward_returns.loc[common_dates, common_stocks],
            sector_map=sector_map,
            min_names=min_names,
        )


def _summarize_ic_series(
    ic_series: pd.Series,
    periods: int = 1,
    periods_per_year: int = TRADING_DAYS,
    hac_lags: int | None = None,
) -> dict[str, Any]:
    """Turn a series of per-date rank ICs into the statistics block."""
    values = ic_series.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return dict(_EMPTY_IC)

    mean_ic = float(np.mean(values))
    ic_std = float(np.std(values))
    n_obs = len(values)

    lags = _auto_lags(periods, n_obs) if hac_lags is None else max(0, int(hac_lags))
    t_stat, p_value, _ = newey_west_t(values, lags)
    naive_se = ic_std / np.sqrt(n_obs) if ic_std > 0 else None
    t_naive = mean_ic / naive_se if naive_se else None
    p_naive = (
        float(2.0 * (1.0 - stats.norm.cdf(abs(t_naive)))) if t_naive is not None else None
    )

    ic_ir = mean_ic / ic_std if ic_std > 0 else None
    # IC observations per year: daily rows are 252/yr, monthly CRSP rows are
    # 12/yr. Annualizing a monthly IC with 252 inflates IC_IR by ~4.6x.
    observations_per_year = periods_per_year / max(periods, 1)

    return {
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "ic_ir_annualized": ic_ir * np.sqrt(observations_per_year) if ic_ir is not None else None,
        "t_stat": t_stat,
        "p_value": p_value,
        "t_stat_naive": float(t_naive) if t_naive is not None else None,
        "p_value_naive": p_naive,
        "num_periods": n_obs,
        # Non-overlapping observations behind the HAC estimate. This is the
        # honest "how much evidence is there really" count.
        "effective_periods": int(round(n_obs / max(periods, 1))),
        "hac_lags": lags,
        "positive_ic_ratio": float(np.mean(values > 0)),
        "ic_percentiles": {
            "5th": float(np.percentile(values, 5)),
            "25th": float(np.percentile(values, 25)),
            "50th": float(np.percentile(values, 50)),
            "75th": float(np.percentile(values, 75)),
            "95th": float(np.percentile(values, 95)),
        },
    }


_EMPTY_IC: dict[str, Any] = {
    "mean_ic": None,
    "ic_std": None,
    "ic_ir": None,
    "ic_ir_annualized": None,
    "t_stat": None,
    "p_value": None,
    "t_stat_naive": None,
    "p_value_naive": None,
    "num_periods": 0,
    "effective_periods": 0,
    "hac_lags": 0,
    "positive_ic_ratio": None,
    "ic_percentiles": None,
}


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def _slice_dates(df: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    out = df
    if start is not None:
        out = out.loc[out.index >= pd.Timestamp(start)]
    if end is not None:
        out = out.loc[out.index <= pd.Timestamp(end)]
    return out


def _purge_tail(df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """Drop the last ``n_rows`` rows so their forward label cannot reach across a boundary."""
    if n_rows <= 0 or len(df) == 0:
        return df
    return df.iloc[:-n_rows] if len(df) > n_rows else df.iloc[:0]


def _window(
    series: pd.Series,
    start: str | None = None,
    end: str | None = None,
    purge: int = 0,
) -> pd.Series:
    """Slice a per-date IC series to a window and purge its tail."""
    out = series
    if start is not None:
        out = out.loc[out.index >= pd.Timestamp(start)]
    if end is not None:
        out = out.loc[out.index <= pd.Timestamp(end)]
    if purge and len(out):
        out = out.iloc[:-purge] if len(out) > purge else out.iloc[:0]
    return out


def _diagnostic_block(ic: dict[str, Any]) -> dict[str, Any]:
    """The compact subset of an IC dict worth nesting in a result payload."""
    keys = (
        "mean_ic",
        "ic_std",
        "ic_ir",
        "ic_ir_annualized",
        "t_stat",
        "p_value",
        "t_stat_naive",
        "p_value_naive",
        "num_periods",
        "effective_periods",
        "hac_lags",
        "positive_ic_ratio",
    )
    return {k: ic.get(k) for k in keys if ic.get(k) is not None}


def compute_ic_split(
    signal_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    train_end_date: str,
    test_start_date: str,
    sector_map: dict[str, str] | None = None,
    validation_end_date: str | None = None,
    purge_periods: int = 0,
    periods: int = 1,
    min_names: int = DEFAULT_MIN_NAMES,
    periods_per_year: int = TRADING_DAYS,
    include_raw: bool = True,
) -> dict[str, Any]:
    """Rank IC on train / validation / test, with an optional purge gap.

    Without ``validation_end_date`` this is the original two-way split: the
    headline is the test IC. With it, the out-of-sample window is split into a
    validation window (used for selection and acceptance) and a later test
    window that the loop never sees; the headline becomes the validation IC.

    ``purge_periods`` removes that many rows from the end of every earlier
    segment, so a forward label at the end of train cannot reach into
    validation, and one at the end of validation cannot reach into test.
    """
    validation_start = (
        str((pd.Timestamp(train_end_date) + pd.Timedelta(days=1)).date())
        if validation_end_date
        else None
    )

    # Rank IC for a date does not depend on which window it ends up in, and
    # sector neutralization is per date. Compute the IC series once and slice
    # it, instead of re-ranking the whole panel for every window and again for
    # the un-neutralized variants. On the 25k-name CRSP panel that is the
    # difference between ~3s and ~25s per signal.
    ic_all = _ic_series_for(signal_values, forward_returns, sector_map, min_names)
    # The un-neutralized series is diagnostic only. Computing it for every
    # candidate doubles the ranking work; callers that score in bulk pass
    # include_raw=False and compute it once for the chosen signal.
    raw_all = (
        _ic_series_for(signal_values, forward_returns, None, min_names)
        if sector_map is not None and include_raw
        else None
    )

    train_ic = _summarize_ic_series(
        _window(ic_all, end=train_end_date, purge=purge_periods), periods, periods_per_year
    )
    val_ic = (
        _summarize_ic_series(
            _window(ic_all, start=validation_start, end=validation_end_date, purge=purge_periods),
            periods,
            periods_per_year,
        )
        if validation_end_date
        else None
    )
    test_ic = _summarize_ic_series(
        _window(ic_all, start=test_start_date), periods, periods_per_year
    )
    full_ic = _summarize_ic_series(ic_all, periods, periods_per_year)

    if validation_end_date:
        # Selection is validation-only, full stop. If the validation window is
        # empty the candidate has no selection evidence; falling back to the
        # test window would make the holdout the selection set. This run did
        # exactly that: a signal with zero train and zero validation periods
        # was scored on 42 holdout months and accepted with selected_on=test.
        headline_source = val_ic
        headline_name = "validation"
    else:
        headline_source = test_ic
        headline_name = "test"

    result: dict[str, Any] = dict(headline_source)
    result["selected_on"] = headline_name
    if validation_end_date:
        result["insufficient_history"] = bool(
            (train_ic.get("num_periods") or 0) == 0
            or ((val_ic or {}).get("num_periods") or 0) == 0
        )
    result["sector_neutralized"] = sector_map is not None
    result["purge_periods"] = purge_periods
    result["ic_train"] = train_ic
    result["ic_test"] = test_ic
    result["ic_full"] = full_ic
    if val_ic is not None:
        result["ic_validation"] = val_ic
    result["diagnostics"] = {
        "train": _diagnostic_block(train_ic),
        "validation": _diagnostic_block(val_ic) if val_ic is not None else None,
        "test": _diagnostic_block(test_ic),
        "full": _diagnostic_block(full_ic),
    }
    if raw_all is not None:
        result["ic_raw_train"] = _summarize_ic_series(
            _window(raw_all, end=train_end_date, purge=purge_periods), periods, periods_per_year
        )
        result["ic_raw_test"] = _summarize_ic_series(
            _window(raw_all, start=test_start_date), periods, periods_per_year
        )
        result["ic_raw_full"] = _summarize_ic_series(raw_all, periods, periods_per_year)
        if validation_end_date:
            result["ic_raw_validation"] = _summarize_ic_series(
                _window(raw_all, start=validation_start, end=validation_end_date, purge=purge_periods),
                periods,
                periods_per_year,
            )
    return result


def compute_ic_decay(
    signal_values: pd.DataFrame,
    close: pd.DataFrame,
    horizons: Iterable[int] = (1, 5, 10, 21, 63),
    sector_map: dict[str, str] | None = None,
    min_names: int = DEFAULT_MIN_NAMES,
) -> dict[str, Any]:
    """Mean IC and HAC t-stat at several forward horizons.

    The decay curve says whether a signal is a one-day blip or a persistent
    premium, and therefore which rebalance horizon it can support.
    """
    out: dict[str, Any] = {}
    for horizon in horizons:
        try:
            forward = compute_forward_returns(close, int(horizon))
            ic = compute_rank_ic(
                signal_values, forward, sector_map, periods=int(horizon), min_names=min_names
            )
            out[str(int(horizon))] = {
                "mean_ic": ic.get("mean_ic"),
                "t_stat": ic.get("t_stat"),
                "p_value": ic.get("p_value"),
            }
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"IC decay failed at horizon {horizon}: {e}")
    return out


# ---------------------------------------------------------------------------
# Cost-aware long/short screen
# ---------------------------------------------------------------------------

def compute_long_short_screen(
    signal_values: pd.DataFrame,
    close: pd.DataFrame,
    quantile: float = 0.2,
    cost_bps: float = 10.0,
    min_names: int = 20,
    annualization: int = TRADING_DAYS,
    direction: float = 1.0,
    sector_map: dict[str, str] | None = None,
    include_series: bool = False,
    buffer_quantile: float | None = None,
    short_borrow_bps: float = 0.0,
    impact_coeff: float = 0.0,
    volume_panel: pd.DataFrame | None = None,
    aum_usd: float = 1_000_000.0,
    vol_scaled: bool = False,
    max_weight: float | None = None,
    holding_periods: int = 1,
) -> dict[str, Any]:
    """Quantile long/short returns implied by a signal, net of realistic costs.

    Supports Garleanu-Pedersen hysteresis buffers, annualized short-borrow
    drag, square-root market impact against daily ADV, inverse-vol risk-scaled
    legs with dollar-neutral water-fill single-name caps, and Jegadeesh-Titman
    overlapping holding tranches when ``holding_periods > 1``.
    """
    if close is None or close.empty:
        return {"error": "no close panel"}

    daily_returns = close.pct_change()
    dates = signal_values.index.intersection(daily_returns.index)
    tickers = signal_values.columns.intersection(close.columns)
    if len(dates) < 30 or len(tickers) < min_names:
        return {"error": "not enough dates or names for a screen"}

    sig_all = signal_values.loc[dates, tickers]
    if sector_map is not None:
        sig_all = _sector_demean_signal(sig_all, sector_map)
        tickers = sig_all.columns.intersection(close.columns)
        sig_all = sig_all.loc[:, tickers]
    sig_all = sig_all * float(direction)
    next_all = daily_returns.loc[dates, tickers].shift(-1)

    vol_win = 12 if annualization <= 12 else (26 if annualization <= 52 else 63)
    rolling_vol = (
        daily_returns.loc[:, tickers]
        .rolling(vol_win, min_periods=max(6, vol_win // 3))
        .std()
        .reindex(index=dates, columns=tickers)
    )
    adv_all = (
        volume_panel.reindex(index=dates, columns=tickers).replace([np.inf, -np.inf], np.nan)
        if volume_panel is not None
        else None
    )

    trading_days_per_period = max(1.0, float(TRADING_DAYS) / float(max(annualization, 1)))
    tranches = max(1, int(holding_periods))
    use_buffer = buffer_quantile is not None and float(buffer_quantile) > float(quantile)
    gross_parts: list[pd.Series] = []
    turnover_parts: list[pd.Series] = []
    impact_parts: list[pd.Series] = []
    borrow_parts: list[pd.Series] = []
    bucket_parts: dict[int, list[pd.Series]] = {bucket: [] for bucket in range(5)}
    previous_last: pd.Series | None = None
    prev_long_mask: np.ndarray | None = None
    prev_short_mask: np.ndarray | None = None
    prev_target_tail: pd.DataFrame | None = None
    prev_short_tail: pd.DataFrame | None = None

    def _cap_and_norm(w_raw: pd.DataFrame) -> pd.DataFrame:
        arr = w_raw.to_numpy(dtype=float, copy=True)
        arr = np.where(np.isfinite(arr) & (arr > 0.0), arr, 0.0)
        row_sums = arr.sum(axis=1, keepdims=True)
        arr = np.divide(arr, row_sums, out=np.zeros_like(arr), where=row_sums > 0.0)
        if max_weight is not None and float(max_weight) > 0:
            cap = float(max_weight)
            active = arr > 0.0
            active_counts = active.sum(axis=1)
            # Rows where even putting `cap` on every active name cannot reach 1.0
            under = (active_counts > 0) & (active_counts * cap <= 1.0 + 1e-12)
            if under.any():
                arr[under] = np.where(active[under], cap, 0.0)
            # Iterative water-fill for rows that can reach full 1.0 gross exposure
            # without any single name exceeding `cap`.
            fillable = (active_counts > 0) & ~under
            if fillable.any():
                sub = arr[fillable]
                sub_active = active[fillable]
                for _ in range(12):
                    over = sub > cap
                    if not over.any():
                        break
                    sub = np.minimum(sub, cap)
                    deficit = 1.0 - sub.sum(axis=1, keepdims=True)
                    if float(deficit.max()) <= 1e-10:
                        break
                    free = sub_active & (sub < cap - 1e-12)
                    free_sum = np.where(free, sub, 0.0).sum(axis=1, keepdims=True)
                    add = np.divide(
                        np.where(free, sub, 0.0) * deficit,
                        free_sum,
                        out=np.zeros_like(sub),
                        where=free_sum > 1e-12,
                    )
                    sub = sub + add
                arr[fillable] = np.minimum(sub, cap)
        return pd.DataFrame(arr, index=w_raw.index, columns=w_raw.columns)

    for start in range(0, len(dates), 128):
        block = sig_all.iloc[start:start + 128].replace([np.inf, -np.inf], np.nan)
        nxt = next_all.loc[block.index]
        vol_b = rolling_vol.loc[block.index]

        counts = block.notna().sum(axis=1)
        leg = (counts * float(quantile)).floordiv(1).clip(lower=1)
        ranks = block.rank(axis=1, method="first")
        if not use_buffer:
            longs = ranks.gt(counts - leg, axis=0)
            shorts = ranks.le(leg, axis=0)
        else:
            buf_leg = (counts * float(buffer_quantile)).floordiv(1).clip(lower=1)
            enter_l = ranks.gt(counts - leg, axis=0).to_numpy(dtype=bool)
            keep_l = ranks.gt(counts - buf_leg, axis=0).to_numpy(dtype=bool)
            enter_s = ranks.le(leg, axis=0).to_numpy(dtype=bool)
            keep_s = ranks.le(buf_leg, axis=0).to_numpy(dtype=bool)
            n_rows, n_cols = enter_l.shape
            l_arr = np.zeros((n_rows, n_cols), dtype=bool)
            s_arr = np.zeros((n_rows, n_cols), dtype=bool)
            if prev_long_mask is None:
                prev_long_mask = np.zeros(n_cols, dtype=bool)
                prev_short_mask = np.zeros(n_cols, dtype=bool)
            for r in range(n_rows):
                cur_l = enter_l[r] | (prev_long_mask & keep_l[r])
                cur_s = enter_s[r] | (prev_short_mask & keep_s[r])
                both = cur_l & cur_s
                if both.any():
                    cur_l = cur_l & ~both
                    cur_s = cur_s & ~both
                l_arr[r] = cur_l
                s_arr[r] = cur_s
                prev_long_mask = cur_l
                prev_short_mask = cur_s
            longs = pd.DataFrame(l_arr, index=block.index, columns=block.columns)
            shorts = pd.DataFrame(s_arr, index=block.index, columns=block.columns)

        if vol_scaled:
            inv_vol = (1.0 / vol_b.clip(lower=1e-4)).fillna(1.0)
            long_w = _cap_and_norm(longs.astype(float) * inv_vol)
            short_w = _cap_and_norm(shorts.astype(float) * inv_vol)
        else:
            long_w = _cap_and_norm(longs.astype(float))
            short_w = _cap_and_norm(shorts.astype(float))

        # Preserve strict dollar neutrality even when `max_weight` caps one leg
        # below 1.0 (scales the larger leg down, never above `max_weight`).
        long_gross = long_w.sum(axis=1)
        short_gross = short_w.sum(axis=1)
        matched_gross = np.minimum(long_gross, short_gross)
        long_w = long_w.mul(matched_gross.div(long_gross.replace(0.0, np.nan)).fillna(0.0), axis=0)
        short_w = short_w.mul(matched_gross.div(short_gross.replace(0.0, np.nan)).fillna(0.0), axis=0)

        target_weights = (long_w - short_w).where(counts >= min_names, 0.0)
        target_short_w = short_w.where(counts >= min_names, 0.0)
        if tranches > 1:
            if prev_target_tail is not None:
                w_cat = pd.concat([prev_target_tail, target_weights])
                s_cat = pd.concat([prev_short_tail, target_short_w])
                weights = w_cat.rolling(tranches, min_periods=1).mean().iloc[len(prev_target_tail):]
                eff_short_w = s_cat.rolling(tranches, min_periods=1).mean().iloc[len(prev_short_tail):]
            else:
                weights = target_weights.rolling(tranches, min_periods=1).mean()
                eff_short_w = target_short_w.rolling(tranches, min_periods=1).mean()
            prev_target_tail = target_weights.iloc[-(tranches - 1):]
            prev_short_tail = target_short_w.iloc[-(tranches - 1):]
        else:
            weights = target_weights
            eff_short_w = target_short_w

        gross_parts.append((weights * nxt).sum(axis=1).where(counts >= min_names))

        buckets = (
            ranks.sub(1.0).mul(5.0).div(counts, axis=0).floordiv(1).clip(0, 4)
        )
        eligible = counts.ge(min_names).to_numpy()[:, None]
        values = nxt.to_numpy(dtype=float, na_value=np.nan)
        frame = pd.DataFrame(values, index=block.index, columns=block.columns)
        for bucket in range(5):
            mask = buckets.eq(bucket) & frame.notna() & eligible
            bucket_parts[bucket].append(frame.where(mask).mean(axis=1))

        dw = weights.diff().abs()
        if previous_last is not None:
            dw.iloc[0] = (weights.iloc[0] - previous_last).abs()
        else:
            dw.iloc[0] = 0.0
        block_turnover = dw.sum(axis=1) / 2.0
        turnover_parts.append(block_turnover)

        if short_borrow_bps > 0:
            short_gross_eff = eff_short_w.sum(axis=1)
            borrow_parts.append(short_gross_eff * (float(short_borrow_bps) / 10_000.0 / float(annualization)))
        else:
            borrow_parts.append(pd.Series(0.0, index=block.index))

        if impact_coeff > 0 and adv_all is not None and aum_usd > 0:
            # Convert period dollar volume (monthly ~21d / weekly ~5d) to daily ADV
            # and period volatility to daily volatility for consistent impact units.
            adv_b = (adv_all.loc[block.index] / trading_days_per_period).clip(lower=1_000.0)
            daily_vol = vol_b / np.sqrt(trading_days_per_period)
            med_vol = float(daily_vol.median(axis=1).median()) if not daily_vol.empty else 0.02
            sig_b = daily_vol.fillna(med_vol if np.isfinite(med_vol) else 0.02).clip(lower=0.002, upper=0.20)
            participation = (dw * float(aum_usd) / adv_b).clip(lower=0.0, upper=1.0)
            slippage_rate = (float(impact_coeff) * sig_b * np.sqrt(participation)).clip(upper=0.05)
            impact_parts.append((dw * slippage_rate).sum(axis=1).fillna(0.0))
        else:
            impact_parts.append(pd.Series(0.0, index=block.index))

        previous_last = weights.iloc[-1]

    gross = pd.concat(gross_parts)
    turnover = pd.concat(turnover_parts)
    impact_cost = pd.concat(impact_parts)
    borrow_cost = pd.concat(borrow_parts)
    net = gross - turnover * (cost_bps / 10_000.0) - impact_cost - borrow_cost

    gross = gross.dropna()
    net = net.dropna()
    if len(net) < 30:
        return {"error": "not enough observations after screening"}

    def _stats(series: pd.Series) -> dict[str, float | None]:
        mean = float(series.mean())
        std = float(series.std(ddof=1))
        sharpe = mean / std * np.sqrt(annualization) if std > 0 else None
        curve = (1.0 + series).cumprod()
        drawdown = float((curve / curve.cummax() - 1.0).min()) if len(curve) else None
        return {
            "annual_return": mean * annualization,
            "annual_vol": std * np.sqrt(annualization),
            "sharpe": sharpe,
            "max_drawdown": drawdown,
            "hit_rate": float((series > 0).mean()),
        }

    gross_stats = _stats(gross)
    net_stats = _stats(net)
    spread_t_stat = None
    if len(net) > 2 and float(net.std(ddof=1)) > 0:
        spread_t_stat = float(net.mean() / net.std(ddof=1) * np.sqrt(len(net)))
    if annualization >= TRADING_DAYS:
        rebalance = "daily"
    elif annualization == 12:
        rebalance = "monthly"
    elif annualization == 52:
        rebalance = "weekly"
    else:
        rebalance = f"every {annualization}/year"
    quantile_returns: list[float | None] = []
    for bucket in range(5):
        parts = [part.dropna() for part in bucket_parts[bucket]]
        parts = [part for part in parts if not part.empty]
        quantile_returns.append(
            float(pd.concat(parts).mean()) if parts else None
        )
    monotonicity = None
    ordered = [value for value in quantile_returns if value is not None]
    if len(ordered) == 5:
        monotonicity = float(np.corrcoef(np.arange(5), ordered)[0, 1])

    return {
        "horizon": f"{rebalance} rebalance",
        "sector_neutral": sector_map is not None,
        "vol_scaled": bool(vol_scaled),
        "max_weight": max_weight,
        "quantile_returns": quantile_returns,
        "monotonicity": monotonicity,
        "periods_per_year": annualization,
        "direction": float(direction),
        "direction_basis": "+1 long high signal; -1 long low signal",
        "quantile": quantile,
        "cost_bps_one_way": cost_bps,
        "short_borrow_bps": short_borrow_bps,
        "impact_coeff": impact_coeff,
        "num_days": len(net),
        "avg_turnover": float(turnover.reindex(net.index).mean()),
        "annual_turnover": float(turnover.reindex(net.index).mean() * annualization),
        "avg_impact_cost": float(impact_cost.reindex(net.index).mean()),
        "avg_borrow_cost": float(borrow_cost.reindex(net.index).mean()),
        "gross": gross_stats,
        "net": net_stats,
        "spread_t_stat": spread_t_stat,
        "notes": (
            "Dollar-neutral, gross 1 per leg, close fills. Costs include linear "
            "spread bps, optional square-root market impact, and short-borrow drag."
        ),
        **({"_net_series": net} if include_series else {}),
    }


# ---------------------------------------------------------------------------
# Code extraction and execution
# ---------------------------------------------------------------------------

def extract_code_from_response(code_response: str) -> str:
    """Extract Python code from markdown code blocks, falling back to raw text."""
    code_blocks = re.findall(r"```python\n(.*?)```", code_response, re.DOTALL)
    if code_blocks:
        return "\n".join(code_blocks)

    code_blocks = re.findall(r"```\n(.*?)```", code_response, re.DOTALL)
    if code_blocks:
        return "\n".join(code_blocks)

    return code_response


# Operator function names, derived from the actual operators loaded from
# calculator.json so this set always stays in sync with available operators.
OPERATOR_NAMES = set(_OPERATOR_FUNCTIONS.keys())

# Order matters: when a signal function uses a generic parameter name (e.g.
# `x`, `data`, `prices`), parameters are filled positionally from this list.
STANDARD_FIELDS = ["Close", "Volume", "High", "Low", "Open"] + FUNDAMENTAL_FIELDS + RATIO_FIELDS + MACRO_FIELDS


def _is_dataframe_param(p: inspect.Parameter) -> bool:
    """Whether this parameter expects a DataFrame to be passed in."""
    ann = p.annotation
    if ann is pd.DataFrame:
        return True
    if isinstance(ann, str) and "DataFrame" in ann:
        return True
    if ann is inspect.Parameter.empty:
        if p.default is inspect.Parameter.empty:
            return True
        if p.default is None or isinstance(p.default, pd.DataFrame):
            return True
        return False
    return False


def _resolve_signal_args(
    sig: inspect.Signature,
    stock_data: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame] | None:
    """Map signal function parameters to stock data DataFrames.

    Non-DataFrame parameters keep their defaults. DataFrame parameters resolve
    by exact name, then a normalized prefix/suffix match, then position.

    Returns ``None`` when a parameter names a *known* field that is not loaded
    (e.g. ``ROE`` while only OHLCV panels exist). The old behaviour was to
    consume the next available field positionally, so ``Rank(ROE)`` was silently
    evaluated as ``Rank(Close)`` and reported a meaningless near-zero IC. A
    signal that references missing data must be rejected, not rewritten.
    """
    available_fields = [f for f in STANDARD_FIELDS if f in stock_data]
    known_fields = set(STANDARD_FIELDS)
    df_params = [name for name, p in sig.parameters.items() if _is_dataframe_param(p)]

    kwargs: dict[str, pd.DataFrame] = {}
    used_fields: set[str] = set()
    unmatched: list[str] = []

    for param in df_params:
        p_norm = re.sub(r"[^a-z0-9]", "", param.lower())
        match: str | None = None
        for field in available_fields:
            f_norm = re.sub(r"[^a-z0-9]", "", field.lower())
            if p_norm == f_norm:
                match = field
                break

        if match is None:
            for field in available_fields:
                f_norm = re.sub(r"[^a-z0-9]", "", field.lower())
                if p_norm.endswith(f_norm) or p_norm.startswith(f_norm):
                    match = field
                    break

        if match is None:
            # The name is a real field that simply was not loaded: refuse.
            for field in known_fields:
                f_norm = re.sub(r"[^a-z0-9]", "", field.lower())
                if p_norm == f_norm:
                    logger.warning(
                        f"Signal parameter '{param}' refers to the '{field}' panel, "
                        "which is not loaded; rejecting instead of substituting another field"
                    )
                    return None

        if match and match not in used_fields:
            kwargs[param] = stock_data[match]
            used_fields.add(match)
        else:
            unmatched.append(param)

    fallback_pool = [f for f in available_fields if f not in used_fields]
    for param in unmatched:
        if not fallback_pool:
            break
        kwargs[param] = stock_data[fallback_pool.pop(0)]

    return kwargs


_SMART_UNICODE = str.maketrans(
    {
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-",
        "\u00a0": " ",
    }
)


def _detect_helpers(candidates: list[tuple[str, Any]]) -> set[str]:
    """Return the names of candidate functions that are called by another candidate."""
    helpers: set[str] = set()
    names = {name for name, _ in candidates}
    for _, fn in candidates:
        try:
            src = inspect.getsource(fn)
        except (OSError, TypeError):
            continue
        for name in names:
            if name != fn.__name__ and re.search(rf"\b{re.escape(name)}\s*\(", src):
                helpers.add(name)
    return helpers


def _candidate_signals(namespace: dict[str, Any]) -> list[tuple[str, Any]]:
    candidates = [
        (name, obj)
        for name, obj in namespace.items()
        if not name.startswith("_")
        and callable(obj)
        and hasattr(obj, "__code__")
        and name not in OPERATOR_NAMES
    ]
    helpers = _detect_helpers(candidates)
    signals = [(n, f) for n, f in candidates if n not in helpers]
    return signals or candidates


def _call_signal(
    name: str,
    func: Any,
    stock_data: dict[str, pd.DataFrame],
) -> pd.DataFrame | None:
    """Call one signal function, resolving its arguments. Returns None on failure."""
    try:
        sig = inspect.signature(func)
        kwargs = _resolve_signal_args(sig, stock_data)
        if kwargs is None:
            return None
        df_param_count = sum(1 for p in sig.parameters.values() if _is_dataframe_param(p))

        if len(kwargs) == df_param_count:
            result = func(**kwargs)
        elif len(sig.parameters) == 0:
            result = func()
        else:
            logger.warning(f"Cannot determine args for {name} (params={list(sig.parameters)}), skipping")
            return None

        if isinstance(result, pd.Series):
            result = result.to_frame()
        if not isinstance(result, pd.DataFrame):
            return None
        # A formula that divides by a zero return produces +/-inf. inf is not a
        # usable cross-sectional value: it survives `notna()` and then poisons a
        # sector mean (inf - inf = NaN) and a rank. Treat it as missing.
        return result.reindex(
            index=stock_data["Close"].index, columns=stock_data["Close"].columns
        ).replace([np.inf, -np.inf], np.nan)
    except Exception as e:
        logger.warning(f"Error executing {name}: {e}")
        return None


def execute_signal_functions(
    code: str,
    stock_data: dict[str, pd.DataFrame],
) -> list[tuple[str, pd.DataFrame]]:
    """Execute generated code and return *every* working signal it defines.

    Returning all of them lets the optimizer score each candidate separately
    and account for the number of trials, instead of silently discarding every
    signal except one.
    """
    namespace: dict[str, Any] = {}
    try:
        exec(code.translate(_SMART_UNICODE), namespace)
    except SyntaxError as e:
        lineno = getattr(e, "lineno", None) or 0
        lines = code.splitlines()
        start, end = max(0, lineno - 3), min(len(lines), lineno + 2)
        snippet = "\n".join(
            f"  {i + 1:4d} {'>>' if i + 1 == lineno else '  '} {line}"
            for i, line in enumerate(lines[start:end], start=start)
        )
        logger.error(f"SyntaxError in generated signal code: {e}\n{snippet}")
        return []
    except Exception as e:
        logger.error(f"Error executing signal code: {e}")
        logger.debug(traceback.format_exc())
        return []

    out: list[tuple[str, pd.DataFrame]] = []
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for name, func in _candidate_signals(namespace):
            df = _call_signal(name, func, stock_data)
            if df is not None:
                out.append((name, df))
    return out


def execute_signal_code(
    code: str,
    stock_data: dict[str, pd.DataFrame],
    selection_periods: int = 5,
    train_end_date: str | None = None,
    sector_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, str] | None:
    """Execute code and return the single best signal.

    Selection uses the train window when ``train_end_date`` is given, so the
    choice among candidates cannot peek at validation or test. Without it, the
    full history is used (legacy behaviour, kept for callers that only have one
    candidate or that are evaluating research code directly).
    """
    candidates = execute_signal_functions(code, stock_data)
    if not candidates:
        logger.warning("No signal functions found in the code")
        return None

    logger.info(f"Found {len(candidates)} signal function(s): {[c[0] for c in candidates]}")
    if len(candidates) == 1:
        name, df = candidates[0]
        return df, name

    close = stock_data["Close"]
    forward = compute_forward_returns(close, selection_periods)

    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        scored: list[tuple[float, str, pd.DataFrame]] = []
        for name, df in candidates:
            sig = df
            ret = forward
            if train_end_date:
                sig = _slice_dates(sig, end=train_end_date)
                ret = _slice_dates(ret, end=train_end_date)
            ic = compute_rank_ic(
                sig, ret, sector_map, periods=selection_periods
            ).get("mean_ic")
            score = abs(ic) if ic is not None else -1.0
            logger.info(f"  {name}: |IC| = {score:.4f}" if ic is not None else f"  {name}: no IC")
            scored.append((score, name, df))

    best_score, best_name, best_df = max(scored, key=lambda item: item[0])
    if best_score < 0:
        best_name, best_df = candidates[0][0], candidates[0][1]
    logger.info(f"Selected {best_name} with |IC| = {max(best_score, 0):.4f}")
    return best_df, best_name
