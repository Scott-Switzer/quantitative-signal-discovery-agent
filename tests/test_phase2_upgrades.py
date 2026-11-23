# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for Phase 2 institutional upgrades."""

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_discovery_workflow.alt_data import build_13f_panels, build_form4_panels
from signal_discovery_workflow.fundamentals import _load_cik_bridge
from signal_discovery_workflow.overfitting import deflated_sharpe_ratio, subperiod_stability
from signal_discovery_workflow.runner import load_yaml_config
from signal_discovery_workflow.signal_evaluator import compute_long_short_screen


def test_fail_closed_ccm_parquet_bridge(tmp_path):
    ccm = pd.DataFrame(
        [
            # Clean unique mapping
            {"lpermno": 10001, "cik": "0000123456", "linkdt": "2010-01-01", "linkenddt": "2025-12-31"},
            # Ambiguous multi-CIK mapping on identical interval -> must be dropped fail-closed
            {"lpermno": 10002, "cik": "0000222222", "linkdt": "2010-01-01", "linkenddt": "2025-12-31"},
            {"lpermno": 10002, "cik": "0000333333", "linkdt": "2010-01-01", "linkenddt": "2025-12-31"},
        ]
    )
    pq_path = tmp_path / "ccm_lookup.parquet"
    ccm.to_parquet(pq_path, index=False)

    bridge = _load_cik_bridge(str(pq_path))
    assert list(bridge["permno"]) == ["10001"]
    assert list(bridge["cik"]) == ["0000123456"]


def test_deflated_sharpe_ratio_and_subperiod_stability():
    rng = np.random.default_rng(42)
    # Modest positive edge (~0.8 annualized monthly Sharpe) over 60 months.
    rets = pd.Series(0.008 + rng.normal(0.0, 0.03, size=60))

    dsr_1 = deflated_sharpe_ratio(rets, n_trials=1, periods_per_year=12)
    dsr_500 = deflated_sharpe_ratio(rets, n_trials=500, periods_per_year=12)
    assert dsr_1["dsr"] is not None and dsr_500["dsr"] is not None
    # Higher trial count must strictly deflate the probability of real skill.
    assert dsr_1["dsr"] > dsr_500["dsr"]

    # Single-lucky-block fluke: 1 huge positive block, 3 slightly negative blocks.
    fluke = np.concatenate([np.full(15, 0.05), np.full(45, -0.002)])
    stab = subperiod_stability(fluke, n_blocks=4, periods_per_year=12)
    assert stab["num_blocks"] == 4
    assert stab["positive_block_ratio"] == 0.25


def test_screen_impact_borrow_and_vol_scaled_caps():
    rng = np.random.default_rng(99)
    dates = pd.date_range("2016-01-31", periods=60, freq="ME")
    cols = [f"S{i:03d}" for i in range(40)]
    rets = pd.DataFrame(rng.normal(0.005, 0.05, size=(60, 40)), index=dates, columns=cols)
    close = (1.0 + rets).cumprod()
    vol = pd.DataFrame(rng.uniform(5e5, 5e6, size=(60, 40)), index=dates, columns=cols)
    sig = pd.DataFrame(rng.normal(0.0, 1.0, size=(60, 40)), index=dates, columns=cols)

    base = compute_long_short_screen(sig, close, quantile=0.2, cost_bps=10.0, annualization=12)
    realistic = compute_long_short_screen(
        sig,
        close,
        quantile=0.2,
        cost_bps=10.0,
        annualization=12,
        short_borrow_bps=50.0,
        impact_coeff=0.15,
        volume_panel=vol,
        aum_usd=5_000_000.0,
        vol_scaled=True,
        max_weight=0.15,
    )
    assert "error" not in base and "error" not in realistic
    assert realistic["avg_borrow_cost"] > 0.0
    assert realistic["avg_impact_cost"] > 0.0
    assert realistic["vol_scaled"] is True


def test_form4_and_13f_pit_panel_builders():
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    secs = pd.Index(["crsp:permno:10001", "crsp:permno:10002"])
    cik_bridge = pd.DataFrame(
        [
            {
                "cik": "0000123456",
                "permno": "10001",
                "effective_from": pd.Timestamp("2015-01-01"),
                "effective_to": pd.Timestamp("2025-12-31"),
            }
        ]
    )
    events = pd.DataFrame(
        [
            {
                "cik": "0000123456",
                "filed_date": "2020-04-10",
                "buy_dollars": 500_000.0,
                "sell_dollars": 100_000.0,
                "disc_buy_dollars": 500_000.0,
                "disc_sell_dollars": 0.0,
                "n_buyers": 3,
            }
        ]
    )
    f4 = build_form4_panels(events, cik_bridge, dates, secs)
    assert {"InsiderNetBuyRatio", "InsiderClusterBuys", "DiscretionaryBuyRatio"} <= set(f4)
    # Zero look-ahead: must be NaN before April 2020 filing date, populated after.
    assert pd.isna(f4["InsiderNetBuyRatio"].loc["2020-03-31", "crsp:permno:10001"])
    assert f4["InsiderNetBuyRatio"].loc["2020-04-30", "crsp:permno:10001"] > 0.9

    cusip_bridge = pd.DataFrame(
        [
            {
                "cusip8": "03783310",
                "permno": "10001",
                "effective_from": pd.Timestamp("2015-01-01"),
                "effective_to": pd.Timestamp("2025-12-31"),
            }
        ]
    )
    holdings = pd.DataFrame(
        [
            {"cusip8": "03783310", "filed_date": "2020-02-14", "total_value": 1_000_000.0, "n_managers": 50, "hhi": 0.08},
            {"cusip8": "03783310", "filed_date": "2020-05-15", "total_value": 1_500_000.0, "n_managers": 65, "hhi": 0.06},
        ]
    )
    f13 = build_13f_panels(holdings, cusip_bridge, dates, secs)
    assert {"InstOwnershipDelta", "InstCrowding"} <= set(f13)
    assert pd.isna(f13["InstCrowding"].loc["2020-01-31", "crsp:permno:10001"])
    assert f13["InstOwnershipDelta"].loc["2020-05-31", "crsp:permno:10001"] > 0.0


def test_lean_runner_loads_all_configs():
    for path in (
        "configs/config-crsp-fundamentals.yml",
        "configs/config-crsp-weekly.yml",
        "configs/config-crsp-quarterly-horizon.yml",
    ):
        llms, wf = load_yaml_config(path)
        assert "signal_generator" in llms
        assert wf["min_dsr"] == 0.95
        assert wf["cik_bridge_uri"].endswith(".parquet")
