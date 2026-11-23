# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0

"""
Anti-overfitting statistical gates (Bailey & López de Prado 2014).

When an automated discovery loop scales to hundreds of trials across parallel
themes and alternative datasets, naive Sharpe ratios and single-window t-stats
inevitably pass noise. This module provides two orthogonal defenses:

1. **Deflated Sharpe Ratio (``deflated_sharpe_ratio``):** Computes the
   probability that a strategy's observed per-period Sharpe ratio exceeds the
   expected maximum Sharpe ratio from ``n_trials`` independent zero-edge trials,
   adjusting for sample length ``T``, skewness, and kurtosis (Bailey & López de
   Prado, 2014, *Journal of Portfolio Management*).
2. **Sub-Period Stability (``subperiod_stability``):** Splits the validation
   return series into contiguous chronological blocks and measures the fraction
   of blocks with positive mean return, rejecting strategies driven by a single
   lucky regime (e.g., 2020 COVID rebound).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum per-period Sharpe ratio across ``n_trials`` noise trials."""
    n = max(1, int(n_trials))
    if n <= 1 or sr_variance <= 0.0:
        return 0.0
    p1 = max(1e-12, min(1.0 - 1e-12, 1.0 - 1.0 / n))
    p2 = max(1e-12, min(1.0 - 1e-12, 1.0 - 1.0 / (n * math.e)))
    z1 = float(stats.norm.ppf(p1))
    z2 = float(stats.norm.ppf(p2))
    return float(math.sqrt(sr_variance) * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))


def deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    n_trials: int,
    *,
    periods_per_year: int = 12,
    sr_variance: float | None = None,
) -> dict[str, Any]:
    """Compute Bailey & López de Prado (2014) Deflated Sharpe Ratio (DSR).

    Parameters
    ----------
    returns:
        Per-period net strategy return series.
    n_trials:
        Total number of trials tested across the ledger history (``N``).
    periods_per_year:
        Annualization factor (12 for monthly, 52 for weekly, 252 for daily).
    sr_variance:
        Variance of per-period Sharpe ratios across tested trials. When omitted,
        falls back to the theoretical null sampling variance ``1 / (T - 1)``.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    t = len(arr)
    if t < 6:
        return {"error": f"too few observations for DSR ({t} < 6)", "dsr": None}

    std = float(np.std(arr, ddof=1))
    if std <= 1e-12:
        return {"error": "zero return volatility", "dsr": None}

    sr_obs = float(np.mean(arr) / std)
    skew = float(stats.skew(arr, bias=False)) if t >= 3 else 0.0
    # Raw kurtosis (Gaussian = 3.0) = excess kurtosis + 3.0.
    kurt = float(stats.kurtosis(arr, fisher=False, bias=False)) if t >= 4 else 3.0
    if not np.isfinite(skew):
        skew = 0.0
    if not np.isfinite(kurt):
        kurt = 3.0

    var_sr = float(sr_variance) if (sr_variance is not None and sr_variance > 0) else (1.0 / (t - 1.0))
    sr_zero = expected_max_sharpe(max(1, int(n_trials)), var_sr)

    denom_sq = 1.0 - skew * sr_obs + ((kurt - 1.0) / 4.0) * (sr_obs ** 2)
    denom = math.sqrt(max(denom_sq, 1e-8))
    z_stat = (sr_obs - sr_zero) * math.sqrt(t - 1.0) / denom
    dsr_prob = float(stats.norm.cdf(z_stat))

    return {
        "dsr": dsr_prob,
        "z_stat": float(z_stat),
        "sr_per_period": sr_obs,
        "sr_annualized": sr_obs * math.sqrt(periods_per_year),
        "sr_benchmark_per_period": sr_zero,
        "sr_benchmark_annualized": sr_zero * math.sqrt(periods_per_year),
        "n_trials": max(1, int(n_trials)),
        "num_periods": int(t),
        "skewness": skew,
        "kurtosis": kurt,
    }


def subperiod_stability(
    returns: pd.Series | np.ndarray,
    *,
    n_blocks: int = 4,
    periods_per_year: int = 12,
) -> dict[str, Any]:
    """Measure return consistency across contiguous chronological sub-periods."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    t = len(arr)
    blocks = max(2, min(int(n_blocks), t // 4)) if t >= 8 else 0
    if blocks < 2:
        return {
            "positive_block_ratio": None,
            "num_blocks": 0,
            "block_means": [],
            "min_block_sharpe": None,
        }

    splits = np.array_split(arr, blocks)
    block_means: list[float] = []
    block_sharpes: list[float] = []
    for chunk in splits:
        if len(chunk) < 2:
            continue
        m = float(np.mean(chunk))
        s = float(np.std(chunk, ddof=1))
        block_means.append(m)
        block_sharpes.append(float(m / s * math.sqrt(periods_per_year)) if s > 1e-12 else 0.0)

    if not block_means:
        return {
            "positive_block_ratio": None,
            "num_blocks": 0,
            "block_means": [],
            "min_block_sharpe": None,
        }

    pos_ratio = float(sum(1 for m in block_means if m > 0) / len(block_means))
    return {
        "positive_block_ratio": pos_ratio,
        "num_blocks": len(block_means),
        "block_means": block_means,
        "block_sharpes": block_sharpes,
        "min_block_sharpe": float(min(block_sharpes)),
    }
