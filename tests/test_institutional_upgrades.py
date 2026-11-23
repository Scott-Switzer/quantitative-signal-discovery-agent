# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for institutional upgrades (Items 1-6):
1. Two-stage weak-alpha pool with pairwise correlation gate + train-only Ridge combiner
2. Cross-sectional percentile liquidity universe mask
3. Barra-style residual idiosyncratic return target
4. Lazy Prices YoY 10-K/10-Q text diff panels
5. Quarterly SEC acceleration panels (RevenueAccel, EarningsAccel, MarginDelta, QuarterlyRoeDelta)
6. Garleanu-Pedersen buffered long/short portfolio screen
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_discovery_workflow.alpha_combiner import AlphaPool
from signal_discovery_workflow.fundamentals import build_sec_quarterly_acceleration_panels
from signal_discovery_workflow.lazy_prices import build_lazy_prices_panels, compute_filing_diffs
from signal_discovery_workflow.phf_bridge import build_tradable_universe_mask
from signal_discovery_workflow.residualize import residualize_forward_returns
from signal_discovery_workflow.signal_evaluator import compute_long_short_screen


def test_alpha_pool_rejects_correlated_duplicate_and_combines_orthogonal_signals():
    rng = np.random.default_rng(123)
    dates = pd.date_range("2015-01-31", periods=60, freq="ME")
    cols = [f"S{i}" for i in range(40)]

    sig_a = pd.DataFrame(rng.normal(size=(60, 40)), index=dates, columns=cols)
    sig_b = pd.DataFrame(rng.normal(size=(60, 40)), index=dates, columns=cols)
    # Near-duplicate of sig_a
    sig_dup = sig_a + pd.DataFrame(rng.normal(0.0, 0.1, size=(60, 40)), index=dates, columns=cols)
    # Forward returns driven by both orthogonal signals
    fwd = 0.02 * sig_a + 0.02 * sig_b + pd.DataFrame(rng.normal(0.0, 0.05, size=(60, 40)), index=dates, columns=cols)

    pool = AlphaPool(max_pairwise_corr=0.50, max_size=10)
    ok_a, _, _ = pool.try_admit(
        name="sig_a", formula="A", values=sig_a, direction=1.0, train_ic=0.05, val_ic=0.04, val_p=0.01
    )
    ok_dup, corr_dup, reason = pool.try_admit(
        name="sig_dup", formula="A2", values=sig_dup, direction=1.0, train_ic=0.05, val_ic=0.04, val_p=0.01
    )
    ok_b, corr_b, _ = pool.try_admit(
        name="sig_b", formula="B", values=sig_b, direction=1.0, train_ic=0.05, val_ic=0.04, val_p=0.01
    )

    assert ok_a is True
    assert ok_dup is False and corr_dup > 0.80 and "corr=" in reason
    assert ok_b is True and corr_b < 0.50
    assert len(pool.members) == 2

    combined, weights = pool.combine_ridge(fwd, train_end_date="2018-12-31", ridge_alpha=5.0)
    assert set(weights) == {"sig_a", "sig_b"}
    assert pytest.approx(sum(weights.values()), rel=1e-5) == 1.0
    assert combined.shape == fwd.shape


def test_percentile_liquidity_mask_preserves_breadth_without_nominal_choking():
    dates = pd.date_range("1975-01-31", periods=3, freq="ME")
    cols = [f"S{i}" for i in range(10)]
    close = pd.DataFrame(1.0, index=dates, columns=cols)
    # Small nominal 1975 dollar volumes (all < $1M), but clear cross-sectional dispersion
    vol = pd.DataFrame(
        [np.linspace(1e4, 5e5, 10)] * 3, index=dates, columns=cols
    )
    cap = pd.DataFrame(
        [np.linspace(1e6, 1e8, 10)] * 3, index=dates, columns=cols
    )
    price = pd.DataFrame([[10.0] * 10] * 3, index=dates, columns=cols)

    mask = build_tradable_universe_mask(
        {"Close": close, "Volume": vol, "MarketCap": cap, "Price": price},
        min_price=2.0,
        min_dollar_volume=0.0,
        min_dollar_volume_pct=0.20,
        top_n_market_cap=8,
    )
    assert mask is not None
    # Keeps 8 of 10 liquid names instead of 0 under a fixed $1M floor
    assert int(mask.iloc[0].sum()) == 8


def test_residualize_forward_returns_orthogonalizes_size_and_sector():
    rng = np.random.default_rng(99)
    dates = pd.date_range("2018-01-31", periods=24, freq="ME")
    cols = [f"S{i}" for i in range(60)]
    sector_map = {c: f"SIC:{i % 3}" for i, c in enumerate(cols)}

    rets = pd.DataFrame(rng.normal(0.01, 0.04, size=(24, 60)), index=dates, columns=cols)
    close = (1.0 + rets).cumprod()
    cap = pd.DataFrame(np.exp(rng.normal(20.0, 1.5, size=(24, 60))), index=dates, columns=cols)

    # Inject a huge size tilt into forward returns
    log_cap = np.log(cap)
    fwd = 0.05 * log_cap.sub(log_cap.mean(axis=1), axis=0) + pd.DataFrame(
        rng.normal(0.0, 0.01, size=(24, 60)), index=dates, columns=cols
    )
    resid = residualize_forward_returns(
        fwd, close, market_cap=cap, sector_map=sector_map, periods_per_year=12, min_names=20
    )
    # Correlation between residualized return and log(MarketCap) on last date ~ 0
    corr = np.corrcoef(resid.iloc[-1].to_numpy(), log_cap.iloc[-1].to_numpy())[0, 1]
    assert abs(corr) < 0.05


def test_lazy_prices_computes_yoy_cosine_jaccard_and_scatters_pit():
    filings = pd.DataFrame(
        [
            {
                "cik": "320193",
                "filed_date": "2020-11-01",
                "form": "10-K",
                "text": "<html>revenue growth strong cloud services margin expansion</html>",
            },
            {
                "cik": "320193",
                "filed_date": "2021-11-01",
                "form": "10-K",
                "text": "<html>litigation antitrust regulatory investigation risk uncertainty margin pressure</html>",
            },
        ]
    )
    diffs = compute_filing_diffs(filings)
    assert len(diffs) == 1
    assert diffs.iloc[0]["text_change"] > 0.70  # substantial language shift
    assert diffs.iloc[0]["jaccard_diff"] > 0.70

    bridge = pd.DataFrame(
        [
            {
                "cik": "0000320193",
                "permno": "14593",
                "effective_from": pd.Timestamp("1990-01-01"),
                "effective_to": pd.Timestamp("2030-01-01"),
            }
        ]
    )
    dates = pd.date_range("2021-09-30", periods=4, freq="ME")
    panels = build_lazy_prices_panels(diffs, bridge, dates, pd.Index(["crsp:permno:14593"]))
    assert set(panels) == {"FilingTextChange", "FilingJaccardDiff", "FilingLengthDelta"}
    # Before 2021-11-01 filing date: must be NaN (strict PIT)
    assert np.isnan(panels["FilingTextChange"].loc["2021-10-31", "crsp:permno:14593"])
    assert panels["FilingTextChange"].loc["2021-11-30", "crsp:permno:14593"] > 0.70


def test_sec_quarterly_acceleration_panels_strict_pit():
    # 6 quarters for one CIK so YoY (4Q) and QoQ acceleration (5Q+) are well-defined
    q_ends = pd.date_range("2020-03-31", periods=6, freq="QE")
    revs = [100.0, 100.0, 100.0, 100.0, 110.0, 130.0]  # YoY jumps from +10% to +30% -> accel +20%
    nis = [10.0, 10.0, 10.0, 10.0, 12.0, 18.0]
    rows = []
    for dt, rev, ni in zip(q_ends, revs, nis):
        avail = dt + pd.Timedelta(days=35)
        rows.append({"cik": "1001", "source_period_end": dt, "information_available_date": avail, "canonical_metric": "revenue", "value": rev})
        rows.append({"cik": "1001", "source_period_end": dt, "information_available_date": avail, "canonical_metric": "net_income", "value": ni})
        rows.append({"cik": "1001", "source_period_end": dt, "information_available_date": avail, "canonical_metric": "operating_income", "value": ni * 1.2})
        rows.append({"cik": "1001", "source_period_end": dt, "information_available_date": avail, "canonical_metric": "equity", "value": 200.0})

    bridge = pd.DataFrame(
        [{"cik": "0000001001", "permno": "55555", "effective_from": pd.Timestamp("2000-01-01"), "effective_to": pd.Timestamp("2030-01-01")}]
    )
    dates = pd.date_range("2021-01-31", periods=10, freq="ME")
    panels = build_sec_quarterly_acceleration_panels(
        pd.DataFrame(rows), bridge, dates, pd.Index(["crsp:permno:55555"])
    )
    assert {"RevenueAccel", "EarningsAccel", "MarginDelta", "QuarterlyRoeDelta"} <= set(panels)
    # Q6 ends 2021-06-30, available 2021-08-04 -> first visible on 2021-08-31 month-end
    assert pytest.approx(panels["RevenueAccel"].loc["2021-08-31", "crsp:permno:55555"], rel=1e-4) == 0.20


def test_buffered_long_short_screen_reduces_turnover():
    rng = np.random.default_rng(11)
    dates = pd.date_range("2018-01-31", periods=48, freq="ME")
    cols = [f"S{i}" for i in range(40)]
    rets = pd.DataFrame(rng.normal(0.005, 0.04, size=(48, 40)), index=dates, columns=cols)
    close = (1.0 + rets).cumprod()
    # Slowly drifting noisy signal where borderline names flip every month
    base = rng.normal(size=(1, 40))
    signal = pd.DataFrame(base + rng.normal(0.0, 0.5, size=(48, 40)), index=dates, columns=cols)

    unbuffered = compute_long_short_screen(signal, close, quantile=0.15, annualization=12)
    buffered = compute_long_short_screen(
        signal, close, quantile=0.15, buffer_quantile=0.30, annualization=12
    )
    assert buffered["annual_turnover"] < unbuffered["annual_turnover"]
