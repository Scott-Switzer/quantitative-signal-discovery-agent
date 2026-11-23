# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0

"""
Barra-style cross-sectional return residualization (Item 3).

Institutional equity market-neutral pods decompose stock returns into
systematic risk-factor returns (size, momentum, realized volatility, and
industry) plus idiosyncratic (residual) returns. Scoring LLM candidates on raw
returns wastes most of the search rediscovering beta/size/momentum tilts that
the factor-alpha gate immediately rejects.

This module computes cross-sectional OLS residuals of forward returns per date
using strictly backward-looking style exposures (all knowable at date ``t``):

* ``size``: cross-sectional z-score of ``log(MarketCap)``
* ``momentum``: cross-sectional z-score of trailing 12-1 (or 52-4 weekly) return
* ``volatility``: cross-sectional z-score of trailing realized return volatility
* ``sector``: one-hot SIC division indicators

Zero look-ahead: every exposure at date ``t`` uses data up to ``t`` only, and
the regression at date ``t`` explains the forward return from ``t`` to ``t+h``.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _cs_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score per row, clipped to [-3, 3] for outlier stability."""
    mean = frame.mean(axis=1)
    std = frame.std(axis=1).replace(0.0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0).clip(-3.0, 3.0)


def build_style_exposures(
    close: pd.DataFrame,
    market_cap: pd.DataFrame | None = None,
    periods_per_year: int = 12,
) -> dict[str, pd.DataFrame]:
    """Compute backward-looking size, momentum, and volatility style exposures."""
    exposures: dict[str, pd.DataFrame] = {}
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan)

    # Size: log(MarketCap)
    if market_cap is not None and not market_cap.empty:
        cap = market_cap.reindex_like(close)
        # Parquet decimal columns arrive as ``decimal.Decimal`` objects, which
        # NumPy's log cannot consume; coerce to float before taking logs.
        if any(pd.api.types.is_object_dtype(dtype) for dtype in cap.dtypes):
            cap = cap.apply(pd.to_numeric, errors="coerce")
        log_cap = np.log(cap.where(cap > 0))
        exposures["size"] = _cs_zscore(log_cap)

    # 12-1 momentum (12m/1m skip on monthly, 52w/4w skip on weekly, 252d/21d on daily)
    if periods_per_year == 12:
        mom_win, skip, vol_win = 12, 1, 12
    elif periods_per_year == 52:
        mom_win, skip, vol_win = 52, 4, 26
    else:
        mom_win, skip, vol_win = 252, 21, 63

    if len(close) > mom_win:
        mom = close.shift(skip) / close.shift(mom_win) - 1.0
        exposures["momentum"] = _cs_zscore(mom.replace([np.inf, -np.inf], np.nan))

    vol = returns.rolling(vol_win, min_periods=max(6, vol_win // 3)).std()
    exposures["volatility"] = _cs_zscore(vol.replace([np.inf, -np.inf], np.nan))
    return exposures


def residualize_forward_returns(
    forward_returns: pd.DataFrame,
    close: pd.DataFrame,
    market_cap: pd.DataFrame | None = None,
    sector_map: dict[str, str] | None = None,
    periods_per_year: int = 12,
    min_names: int = 30,
) -> pd.DataFrame:
    """Return idiosyncratic forward returns orthogonal to size, momentum, vol, and sector.

    On dates where fewer than ``min_names`` valid stocks exist (e.g. early
    warm-up rows before momentum/vol windows fill), falls back cleanly to
    cross-sectionally demeaned forward returns so no history is silently lost.
    """
    exposures = build_style_exposures(
        close, market_cap=market_cap, periods_per_year=periods_per_year
    )
    dates = forward_returns.index
    columns = list(forward_returns.columns)
    n_dates, n_cols = len(dates), len(columns)

    # Pre-encode sector dummies once across columns.
    sector_dummies: np.ndarray | None = None
    if sector_map:
        codes = np.array([sector_map.get(c, "Unknown") for c in columns], dtype=object)
        unique_sectors = sorted({c for c in codes if c and c != "Unknown"})
        if len(unique_sectors) >= 2:
            # Drop one dummy to avoid collinearity with intercept.
            sector_dummies = np.column_stack(
                [(codes == sec).astype(float) for sec in unique_sectors[1:]]
            )

    style_arrays = [
        df.reindex(index=dates, columns=columns).to_numpy(dtype=float)
        for df in exposures.values()
    ]
    y_all = forward_returns.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    out = np.full((n_dates, n_cols), np.nan, dtype=float)

    residualized_dates = 0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for i in range(n_dates):
            y = y_all[i]
            valid = np.isfinite(y)
            if valid.sum() < min_names:
                continue

            cols_x = [np.ones(n_cols, dtype=float)]
            active_mask = valid.copy()
            for arr in style_arrays:
                row = arr[i]
                if np.isfinite(row[valid]).sum() >= min_names:
                    # Fill missing style values with 0 (cross-sectional mean of z-score)
                    # so a missing vol/mom observation does not drop the stock.
                    filled = np.where(np.isfinite(row), row, 0.0)
                    cols_x.append(filled)

            if sector_dummies is not None:
                cols_x.append(sector_dummies)

            X_full = np.column_stack(cols_x)
            idx = np.flatnonzero(active_mask)
            X = X_full[idx]
            yv = y[idx]
            if X.shape[0] <= X.shape[1] + 5:
                out[i, idx] = yv - np.mean(yv)
                continue

            coef, _, _, _ = np.linalg.lstsq(X, yv, rcond=None)
            out[i, idx] = yv - X @ coef
            residualized_dates += 1

    logger.info(
        f"Residualized forward returns across {residualized_dates}/{n_dates} dates "
        f"(styles={sorted(exposures)}, sectors={'yes' if sector_dummies is not None else 'no'})"
    )
    return pd.DataFrame(out, index=dates, columns=columns)
