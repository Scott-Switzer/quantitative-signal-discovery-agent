# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for Items 1-4: Lakehouse v2 fundamentals, tradable universe
filter, factor-neutral alpha gate, lagged FRED macro panels, sharded trial
ledgers, and personal-hedge-fund receipt export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from signal_discovery_workflow import r2_store
from signal_discovery_workflow.fundamentals import build_lakehouse_v2_panels
from signal_discovery_workflow.macro_features import (
    build_macro_panels,
    compute_factor_alpha,
    load_factor_returns,
    normalize_factor_returns_frame,
)
from signal_discovery_workflow.phf_bridge import build_tradable_universe_mask
from signal_discovery_workflow.phf_export import export_phf_run
from signal_discovery_workflow.signal_evaluator import compute_long_short_screen
from signal_discovery_workflow.trial_registry import TrialRegistry


def test_lakehouse_v2_panels_cashflow_and_ev_ratios():
    dates = pd.date_range("2020-01-31", periods=15, freq="ME")
    securities = pd.Index(["crsp:permno:10001", "crsp:permno:10002"])

    rows = []
    for dt in dates:
        rows.append(
            {
                "security_id": "crsp:permno:10001",
                "permno": 10001,
                "date": dt,
                "market_cap": 1000.0,
                "revenue": 500.0,
                "operating_income": 100.0,
                "net_income": 80.0,
                "assets": 1000.0,
                "liabilities": 400.0,
                "equity": 600.0,
                "cash_and_equivalents": 200.0,
                "capex": -30.0,  # negative outflow convention handled via abs()
                "operating_cash_flow": 130.0,
                "debt_total": 300.0,
            }
        )
    frame = pd.DataFrame(rows)
    panels = build_lakehouse_v2_panels(frame, dates, securities)

    # FCF = 130 - |-30| = 100; FcfYield = 100 / 1000 = 0.10
    assert pytest.approx(panels["FcfYield"].iloc[-1]["crsp:permno:10001"], rel=1e-5) == 0.10
    # EV = 1000 + 300 - 200 = 1100; EbitToEv = 100 / 1100
    assert pytest.approx(panels["EbitToEv"].iloc[-1]["crsp:permno:10001"], rel=1e-5) == 100.0 / 1100.0
    # Accruals = (80 - 130) / 1000 = -0.05
    assert pytest.approx(panels["Accruals"].iloc[-1]["crsp:permno:10001"], rel=1e-5) == -0.05
    # Lagged-asset CashFlowToAssets only defined after 12 months
    assert np.isnan(panels["CashFlowToAssets"].iloc[0]["crsp:permno:10001"])
    assert pytest.approx(panels["CashFlowToAssets"].iloc[-1]["crsp:permno:10001"], rel=1e-5) == 0.13
    assert pytest.approx(panels["DebtToAssets"].iloc[-1]["crsp:permno:10001"], rel=1e-5) == 0.30


def test_tradable_universe_mask_excludes_microcaps_and_penny_stocks():
    dates = pd.date_range("2022-01-31", periods=3, freq="ME")
    cols = ["BIG", "MID", "PENNY", "ILLIQ"]
    close = pd.DataFrame(1.0, index=dates, columns=cols)
    price = pd.DataFrame(
        [[100.0, 20.0, 2.5, 15.0]] * 3, index=dates, columns=cols
    )
    volume = pd.DataFrame(
        [[5e7, 5e6, 5e6, 1e4]] * 3, index=dates, columns=cols
    )
    cap = pd.DataFrame(
        [[1e10, 2e9, 5e8, 1e9]] * 3, index=dates, columns=cols
    )
    mask = build_tradable_universe_mask(
        {"Close": close, "Price": price, "Volume": volume, "MarketCap": cap},
        min_price=5.0,
        min_dollar_volume=1e6,
        top_n_market_cap=2,
    )
    assert mask is not None
    assert bool(mask.iloc[0]["BIG"]) is True
    assert bool(mask.iloc[0]["MID"]) is True
    assert bool(mask.iloc[0]["PENNY"]) is False
    assert bool(mask.iloc[0]["ILLIQ"]) is False


def test_macro_panels_strictly_lagged_no_lookahead():
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    securities = pd.Index(["A", "B"])
    fred = pd.DataFrame(
        [
            {"series_id": "VIXCLS", "observation_date": "2020-01-31", "value": 18.0, "missing": False},
            {"series_id": "VIXCLS", "observation_date": "2020-02-29", "value": 40.0, "missing": False},
            {"series_id": "VIXCLS", "observation_date": "2020-03-31", "value": 55.0, "missing": False},
            {"series_id": "T10Y3M", "observation_date": "2020-01-31", "value": 0.25, "missing": False},
            {"series_id": "T10Y3M", "observation_date": "2020-02-29", "value": -0.10, "missing": False},
            {"series_id": "NFCI", "observation_date": "2020-01-31", "value": -0.50, "missing": False},
        ]
    )
    panels = build_macro_panels(fred, dates, securities, lag_periods=1)
    assert set(panels) == {"MacroVIX", "MacroYieldCurve", "MacroNFCI"}
    # Date 0 must be NaN (cannot peek at same-month VIX=18.0)
    assert np.isnan(panels["MacroVIX"].iloc[0]["A"])
    # Date 1 sees Date 0's VIX=18.0, not Date 1's 40.0
    assert pytest.approx(panels["MacroVIX"].iloc[1]["A"]) == 18.0
    assert pytest.approx(panels["MacroVIX"].iloc[1]["B"]) == 18.0
    assert pytest.approx(panels["MacroVIX"].iloc[2]["A"]) == 40.0


def test_factor_alpha_distinguishes_factor_clone_from_true_alpha():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2015-01-31", periods=60, freq="ME")
    mkt = pd.Series(rng.normal(0.008, 0.04, size=60), index=dates)
    hml = pd.Series(rng.normal(0.002, 0.03, size=60), index=dates)
    long_df = pd.concat(
        [
            pd.DataFrame({"date": dates, "location": "USA", "factor": "mkt", "ret": mkt.values}),
            pd.DataFrame({"date": dates, "location": "USA", "factor": "hml", "ret": hml.values}),
        ],
        ignore_index=True,
    )
    factors = normalize_factor_returns_frame(long_df)
    assert set(factors.columns) == {"mkt", "hml"}

    # Pure factor clone: alpha ~ 0, t-stat near 0
    clone = 1.2 * mkt + 0.8 * hml + pd.Series(rng.normal(0.0, 0.002, size=60), index=dates)
    res_clone = compute_factor_alpha(clone, factors, periods_per_year=12)
    assert abs(res_clone["alpha_t_stat"]) < 2.0
    assert res_clone["r_squared"] > 0.95

    # Genuine residual alpha: +1.5%/month net of factors
    alpha_strat = 0.015 + 0.5 * mkt + pd.Series(rng.normal(0.0, 0.005, size=60), index=dates)
    res_alpha = compute_factor_alpha(alpha_strat, factors, periods_per_year=12)
    assert res_alpha["alpha_t_stat"] > 5.0
    assert res_alpha["alpha_annualized"] > 0.12


def test_screen_include_series_flag():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-31", periods=48, freq="ME")
    tickers = [f"S{i}" for i in range(30)]
    rets = pd.DataFrame(rng.normal(0.005, 0.05, size=(48, 30)), index=dates, columns=tickers)
    close = (1.0 + rets).cumprod()
    signal = pd.DataFrame(rng.normal(size=(48, 30)), index=dates, columns=tickers)

    without = compute_long_short_screen(signal, close, annualization=12, include_series=False)
    with_ser = compute_long_short_screen(signal, close, annualization=12, include_series=True)
    assert "_net_series" not in without
    assert isinstance(with_ser.get("_net_series"), pd.Series)
    assert len(with_ser["_net_series"]) == with_ser["num_days"]


def test_read_parquet_parts_intersects_heterogeneous_schemas(tmp_path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"a": [1, 2], "b": ["x", "y"]}), tmp_path / "p1.parquet")
    pq.write_table(pa.table({"a": [3], "c": [9.0]}), tmp_path / "p2.parquet")

    frame = r2_store.read_parquet_parts(str(tmp_path), ["a", "b"])
    assert list(frame["a"]) == [1, 2, 3]
    assert frame["b"].isna().sum() == 1


def test_factor_loader_uses_one_coherent_us_dataset(tmp_path: Path):
    dates = pd.date_range("2015-01-31", periods=60, freq="ME")
    model = ("be_me", "ret_12_1", "cop_at", "at_gr1", "beta_60m", "qmj")
    aqr = pd.DataFrame(
        {
            "source": "aqr",
            "dataset": "qmj",
            "factor": "qmj",
            "region": "USA",
            "date": dates,
            "frequency": "monthly",
            "value": 999.0,
            "sheet": "QMJ Factors",
        }
    )
    rows = []
    for dt in dates:
        for name in model:
            rows.append(
                {
                    "source": "jkp",
                    "dataset": "jkp_usa_all_factors_monthly_vw_cap",
                    "factor": name,
                    "region": "usa",
                    "date": dt,
                    "frequency": "monthly",
                    "value": 0.01,
                    "unit": "decimal_return",
                }
            )
    jkp = pd.DataFrame(rows)
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pandas(aqr), tmp_path / "a_aqr.parquet")
    pq.write_table(pa.Table.from_pandas(jkp), tmp_path / "b_jkp.parquet")

    factors = load_factor_returns(str(tmp_path), factor_names=model)
    assert set(factors.columns) == set(model)
    # AQR qmj=999 must not be averaged into the JKP qmj=0.01.
    assert factors["qmj"].dropna().eq(0.01).all()
    assert len(factors) == 60


def test_sharded_trial_registry_merges_legacy_and_concurrent_shards(tmp_path: Path):
    ledger = tmp_path / "trials.jsonl"
    # Seed legacy single-file ledger
    legacy = TrialRegistry(ledger, sharded=False)
    legacy.record({"formula": "CS_Rank(BookToMarket)", "mean_ic": 0.03, "p_value": 0.01})

    # Two independent concurrent sharded writers
    r1 = TrialRegistry(ledger, sharded=True)
    r2 = TrialRegistry(ledger, sharded=True)
    r1.record({"formula": "CS_Rank(FcfYield)", "mean_ic": 0.04, "p_value": 0.005})
    r2.record({"formula": "CS_Rank(EbitToEv)", "mean_ic": 0.035, "p_value": 0.008})

    # Legacy file untouched; two separate shard files exist
    shards = list((tmp_path / "trials").glob("run_*.jsonl"))
    assert len(shards) == 2

    # Fresh reader merges legacy + both shards
    reader = TrialRegistry(ledger, sharded=True)
    formulas = set(reader.tried_formulas())
    assert formulas == {"CS_Rank(BookToMarket)", "CS_Rank(FcfYield)", "CS_Rank(EbitToEv)"}
    assert reader.summary()["num_trials"] == 3


def test_phf_export_writes_receipt_and_results(tmp_path: Path):
    out = export_phf_run(
        str(tmp_path),
        signal_name="signal_fcf_quality",
        formula="CS_Rank(FcfYield)",
        request_text="cash flow value",
        ic_results={
            "selected_on": "validation",
            "ic_train": {"mean_ic": 0.03},
            "ic_validation": {"mean_ic": 0.04, "t_stat": 3.5},
            "ic_test": {"mean_ic": 0.035},
        },
        screen_validation={"sector_neutral": True, "net": {"sharpe": 0.85}, "monotonicity": 0.9, "spread_t_stat": 2.6},
        screen_full={"sector_neutral": True, "net": {"sharpe": 0.72}},
        factor_alpha={"alpha_t_stat": 2.4, "alpha_annualized": 0.05},
        data_pins={"market_data": {"sha256": "abc123"}},
        config_summary={"min_price": 5.0},
        signal_artifact_uri="r2://bucket/research/signals/sig.json",
    )
    receipt = json.loads(Path(out["receipt_uri"]).read_text())
    results = json.loads(Path(out["results_uri"]).read_text())
    assert receipt["schema"] == "phf-research-receipt/1.0.0"
    assert receipt["checks"]["sector_neutral_screen"]["ok"] is True
    assert receipt["checks"]["factor_alpha_checked"]["ok"] is True
    assert results["headline"]["validation_net_sharpe"] == 0.85
