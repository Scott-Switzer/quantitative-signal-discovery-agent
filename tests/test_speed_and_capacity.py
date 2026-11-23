# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for speed and research-capacity upgrades."""

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_discovery_workflow.alpha_combiner import AlphaPool, AlphaPoolLibrary
from signal_discovery_workflow.feature_store import load_feature_store, save_feature_store
from signal_discovery_workflow.macro_features import compute_factor_alpha
from signal_discovery_workflow.signal_generator import (
    MICROSTRUCTURE_FIELDS,
    VALID_DATA_FIELDS,
    build_signal_prompt,
)


def _synthetic_panels(n_dates: int = 48, n_stocks: int = 30):
    rng = np.random.default_rng(7)
    dates = pd.date_range("2016-01-31", periods=n_dates, freq="ME")
    cols = [f"S{i:03d}" for i in range(n_stocks)]
    rets = pd.DataFrame(rng.normal(0.005, 0.04, size=(n_dates, n_stocks)), index=dates, columns=cols)
    close = (1.0 + rets).cumprod()
    return dates, cols, close, rets


def test_feature_store_roundtrip(tmp_path):
    dates, cols, close, rets = _synthetic_panels(24, 15)
    mask = pd.DataFrame(True, index=dates, columns=cols)
    sectors = {c: ("SIC:Tech" if i % 2 == 0 else "SIC:Finance") for i, c in enumerate(cols)}

    store_dir = str(tmp_path / "feature_store_v1")
    manifest = save_feature_store(
        store_dir,
        {"Close": close, "Returns": rets},
        sector_map=sectors,
        universe_mask=mask,
        residual_returns=rets,
        metadata={"sleeve": "monthly"},
        max_workers=2,
    )
    assert manifest["num_panels"] == 2
    assert len(manifest["manifest_sha256"]) == 64

    loaded = load_feature_store(store_dir, max_workers=2)
    assert set(loaded["panels"]) == {"Close", "Returns"}
    assert loaded["sector_map"] == sectors
    assert loaded["universe_mask"].shape == mask.shape
    assert loaded["residual_returns"].shape == rets.shape
    assert loaded["manifest_sha256"] == manifest["manifest_sha256"]


def test_weekly_factor_alpha_compounds_to_monthly():
    rng = np.random.default_rng(11)
    # 156 weekly strategy returns (~36 months) with +0.4%/week pure alpha.
    weeks = pd.date_range("2017-01-06", periods=156, freq="W-FRI")
    weekly_strat = pd.Series(0.004 + rng.normal(0.0, 0.005, size=len(weeks)), index=weeks)

    months = pd.date_range("2016-12-31", periods=42, freq="ME")
    factors = pd.DataFrame(
        {
            "be_me": rng.normal(0.0, 0.02, size=len(months)),
            "ret_12_1": rng.normal(0.0, 0.03, size=len(months)),
        },
        index=months,
    )

    res = compute_factor_alpha(weekly_strat, factors, periods_per_year=52, min_periods=24)
    assert "error" not in res
    assert res["num_periods"] >= 30
    assert res["alpha_annualized"] > 0.10
    assert res["alpha_t_stat"] > 2.0


def test_heterogeneous_stacker_and_persistent_library(tmp_path):
    rng = np.random.default_rng(23)
    dates, cols, _, fwd = _synthetic_panels(48, 40)
    # Two orthogonal signals predictive on train and validation.
    s1 = fwd * 0.4 + pd.DataFrame(rng.normal(0, 0.05, size=fwd.shape), index=dates, columns=cols)
    s2 = (fwd ** 2 - fwd.mean().mean()) * 0.5 + pd.DataFrame(
        rng.normal(0, 0.05, size=fwd.shape), index=dates, columns=cols
    )

    pool = AlphaPool(max_pairwise_corr=0.60, max_size=10)
    ok1, _, _ = pool.try_admit(
        name="sig_linear",
        formula="CS_Rank(FcfYield)",
        values=s1,
        direction=1.0,
        train_ic=0.03,
        val_ic=0.025,
        val_p=0.01,
        val_sharpe=0.65,
    )
    ok2, _, _ = pool.try_admit(
        name="sig_nonlinear",
        formula="CS_Rank(EarningsAccel)",
        values=s2,
        direction=1.0,
        train_ic=0.02,
        val_ic=0.018,
        val_p=0.02,
        val_sharpe=0.40,
    )
    assert ok1 and ok2

    blended, details = pool.combine_heterogeneous(
        fwd, train_end_date="2018-06-30", ridge_alpha=10.0, gbdt_weight=0.45
    )
    assert blended.shape == fwd.shape
    assert details["backend"] in {"lightgbm", "hist_gbdt", "poly_interaction_ridge"}

    lib = AlphaPoolLibrary(tmp_path / "alpha_pool")
    shard = lib.save_members(pool.members, run_id="test-run", horizon=1)
    assert shard is not None
    parents = lib.top_parents(5, horizon=1)
    assert len(parents) == 2
    assert parents[0]["name"] == "sig_linear"


def test_evolutionary_parent_archive_and_microstructure_fields():
    assert set(MICROSTRUCTURE_FIELDS) <= VALID_DATA_FIELDS
    prompt = build_signal_prompt(
        request="weekly microstructure signals",
        num_signals=4,
        operators_list="- CS_Rank(x)",
        template_block="[]",
        available_fields={"Close", "Volume", "AmihudIlliq", "IdioVol", "High52wRatio", "VolumeShock"},
        parent_signals=[
            {"name": "parent_fcf", "formula": "CS_Rank(FcfYield)", "val_ic": 0.018, "val_sharpe": 0.42}
        ],
    )
    assert "EVOLUTIONARY PARENT ARCHIVE" in prompt
    assert "parent_fcf" in prompt
    assert "AmihudIlliq" in prompt
    assert "High52wRatio" in prompt
