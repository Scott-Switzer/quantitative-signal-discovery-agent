# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for Phase 2 bugfixes and scalable Lazy Prices pipeline."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pandas as pd

from signal_discovery_workflow.alt_data import (
    _load_cusip_bridge,
    build_form4_panels,
    parse_13f_quarter_zip,
)
from signal_discovery_workflow.fundamentals import build_lakehouse_v2_panels
from signal_discovery_workflow.lazy_prices import (
    list_lazy_prices_cik_shards,
    merge_lazy_prices_shards,
    process_lazy_prices_cik_shard,
)
from signal_discovery_workflow.r2_store import read_bytes_prefix
from signal_discovery_workflow.signal_evaluator import compute_long_short_screen


def test_waterfill_weight_cap_preserves_dollar_neutrality_and_cap():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2018-01-31", periods=48, freq="ME")
    cols = [f"S{i:03d}" for i in range(60)]
    # Create stark volatility dispersion so inv-vol weighting triggers the 3% cap heavily.
    vols = np.linspace(0.005, 0.15, 60)
    rets = pd.DataFrame(rng.normal(0.004, vols, size=(48, 60)), index=dates, columns=cols)
    close = (1.0 + rets).cumprod()
    sig = pd.DataFrame(rng.normal(0.0, 1.0, size=(48, 60)), index=dates, columns=cols)

    # With 60 names and quantile=0.20, each leg has 12 names -> max_weight=0.10 allows
    # full 1.0 exposure via water-filling (12 * 0.10 = 1.20 > 1.0).
    res = compute_long_short_screen(
        sig,
        close,
        quantile=0.20,
        cost_bps=10.0,
        annualization=12,
        vol_scaled=True,
        max_weight=0.10,
        short_borrow_bps=50.0,
    )
    assert "error" not in res
    assert res["vol_scaled"] is True
    assert res["max_weight"] == 0.10
    # Borrow cost reflects matched short gross exposure (~1.0 * 50bps / 12).
    assert np.isclose(res["avg_borrow_cost"], 50.0 / 10_000.0 / 12.0, rtol=0.05)


def test_holding_periods_tranches_reduce_turnover():
    rng = np.random.default_rng(11)
    dates = pd.date_range("2016-01-31", periods=60, freq="ME")
    cols = [f"S{i:03d}" for i in range(50)]
    rets = pd.DataFrame(rng.normal(0.005, 0.04, size=(60, 50)), index=dates, columns=cols)
    close = (1.0 + rets).cumprod()
    sig = pd.DataFrame(rng.normal(0.0, 1.0, size=(60, 50)), index=dates, columns=cols)

    h1 = compute_long_short_screen(sig, close, quantile=0.20, annualization=12, holding_periods=1)
    h3 = compute_long_short_screen(sig, close, quantile=0.20, annualization=12, holding_periods=3)
    assert "error" not in h1 and "error" not in h3
    assert h3["annual_turnover"] < h1["annual_turnover"] * 0.65


def test_lakehouse_v2_weekly_grid_alignment_and_52w_lag():
    # 70 Fridays (~1.35 years) so 52-week lagged assets are populated for weeks 52..69.
    dates = pd.date_range("2020-01-03", periods=70, freq="W-FRI")
    secs = pd.Index(["crsp:permno:10001", "crsp:permno:10002"])
    months = pd.date_range("2020-01-31", "2021-05-31", freq="ME")

    rows = []
    for m_idx, dt in enumerate(months):
        # Year-over-year asset growth of +20% (100 -> 120 after 12 months).
        assets = 100.0 if m_idx < 12 else 120.0
        rows.append(
            {
                "security_id": "crsp:permno:10001",
                "date": dt,
                "market_cap": 500.0,
                "revenue": 80.0,
                "operating_income": 20.0,
                "net_income": 15.0,
                "assets": assets,
                "liabilities": 40.0,
                "equity": 60.0,
                "cash_and_equivalents": 10.0,
                "capex": 5.0,
                "operating_cash_flow": 25.0,
                "debt_total": 20.0,
            }
        )
    frame = pd.DataFrame(rows)
    panels = build_lakehouse_v2_panels(frame, dates, secs)
    assert "BookToMarket" in panels and "AssetGrowth" in panels

    btm = panels["BookToMarket"]["crsp:permno:10001"]
    # Before first month-end (2020-01-31), must be NaN (zero look-ahead).
    assert btm.loc[dates < pd.Timestamp("2020-01-31")].isna().all()
    # After 2020-01-31, EVERY Friday should be populated (not just ~1/7 months).
    after_first = btm.loc[dates >= pd.Timestamp("2020-01-31")]
    assert after_first.notna().mean() > 0.95

    # AssetGrowth should use a 52-week lag (not 12-week): at week 65 (April 2021),
    # assets=120 vs lagged 52w assets=100 -> +0.20.
    ag = panels["AssetGrowth"]["crsp:permno:10001"]
    assert np.isclose(ag.iloc[-1], 0.20, atol=1e-4)


def test_form4_rolling_90d_aggregation_does_not_overwrite():
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    secs = pd.Index(["crsp:permno:10001"])
    bridge = pd.DataFrame(
        [
            {
                "cik": "0000123456",
                "permno": "10001",
                "effective_from": pd.Timestamp("2015-01-01"),
                "effective_to": pd.Timestamp("2025-12-31"),
            }
        ]
    )
    # Day 1: huge $1M cluster buy (4 buyers); Day 3: tiny $10k sale (0 buyers).
    events = pd.DataFrame(
        [
            {
                "cik": "0000123456",
                "filed_date": "2020-02-05",
                "buy_dollars": 1_000_000.0,
                "sell_dollars": 0.0,
                "disc_buy_dollars": 1_000_000.0,
                "disc_sell_dollars": 0.0,
                "n_buyers": 4,
            },
            {
                "cik": "0000123456",
                "filed_date": "2020-02-07",
                "buy_dollars": 0.0,
                "sell_dollars": 10_000.0,
                "disc_buy_dollars": 0.0,
                "disc_sell_dollars": 10_000.0,
                "n_buyers": 0,
            },
        ]
    )
    panels = build_form4_panels(events, bridge, dates, secs)
    # Net buy ratio on 2020-02-29 must reflect ($1M - $10k) / ($1M + $10k) ~= +0.98,
    # NOT -1.0 from the tiny sale overwriting the big buy.
    assert panels["InsiderNetBuyRatio"].loc["2020-02-29", "crsp:permno:10001"] > 0.95
    assert np.isclose(
        panels["InsiderClusterBuys"].loc["2020-02-29", "crsp:permno:10001"],
        np.log1p(4.0),
        atol=1e-4,
    )


def test_13f_cusip_bridge_fail_closed_and_manager_dedupe(tmp_path):
    ccm = pd.DataFrame(
        [
            {"lpermno": "10001", "cusip": "037833100", "linkdt": "2010-01-01", "linkenddt": "2025-12-31"},
            # Ambiguous CUSIP8 mapping to two permnos -> must be dropped fail-closed
            {"lpermno": "20001", "cusip": "594918104", "linkdt": "2010-01-01", "linkenddt": "2025-12-31"},
            {"lpermno": "20002", "cusip": "594918104", "linkdt": "2010-01-01", "linkenddt": "2025-12-31"},
            # Non-numeric permno -> must be dropped cleanly
            {"lpermno": "bad", "cusip": "02079K107", "linkdt": "2010-01-01", "linkenddt": "2025-12-31"},
        ]
    )
    pq_path = tmp_path / "ccm.parquet"
    ccm.to_parquet(pq_path, index=False)
    bridge = _load_cusip_bridge(str(pq_path))
    assert list(bridge["cusip8"]) == ["03783310"]
    assert list(bridge["permno"]) == ["10001"]

    # Build a synthetic 13F zip where the same manager files twice in one rolling window.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "SUBMISSION.tsv",
            "ACCESSION_NUMBER\tFILING_DATE\tCIK\tSUBMISSIONTYPE\n"
            "0001-24-000001\t2024-03-05\t0001000001\t13F-HR\n"
            "0001-24-000002\t2024-05-15\t0001000001\t13F-HR\n",
        )
        zf.writestr(
            "INFOTABLE.tsv",
            "ACCESSION_NUMBER\tCUSIP\tVALUE\n"
            "0001-24-000001\t037833100\t1000000\n"
            "0001-24-000002\t037833100\t1200000\n",
        )
    parsed = parse_13f_quarter_zip(buf.getvalue())
    assert len(parsed) == 1
    # Must keep latest filing (1,200,000), NOT double-count 2,200,000.
    assert parsed.iloc[0]["total_value"] == 1_200_000.0


def test_lazy_prices_sharded_streaming_pipeline_and_prefix_read(tmp_path):
    # Create a mini corpus with 10-K, 8-K (must be skipped), and pre-2010 (must be skipped).
    corpus = tmp_path / "corpus"
    filings = corpus / "raw" / "sec" / "filings"

    def _write_filing(cik: str, acc: str, form: str, filed: str, html: str) -> None:
        d = filings / f"cik={cik}" / acc
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(json.dumps({"form": form, "filing_date": filed}))
        (d / "primary.html").write_text(html)

    _write_filing("0000123456", "acc_old", "10-K", "2008-03-01", "<p>ancient filing</p>")
    _write_filing("0000123456", "acc_8k", "8-K", "2020-06-01", "<p>current report noise</p>")
    _write_filing(
        "0000123456",
        "acc_1",
        "10-K",
        "2020-03-01",
        "<html><body>revenue growth strong margin expansion stable operations</body></html>",
    )
    _write_filing(
        "0000123456",
        "acc_2",
        "10-K",
        "2021-03-01",
        "<html><body>litigation restructuring impairment risk debt covenant breach</body></html>",
    )
    # Non-CRSP CIK that should be filtered out by CCM bridge
    _write_filing("0000999999", "acc_x1", "10-K", "2020-03-01", "<p>trust filing alpha</p>")
    _write_filing("0000999999", "acc_x2", "10-K", "2021-03-01", "<p>trust filing beta</p>")

    ccm = pd.DataFrame(
        [{"lpermno": 10001, "cik": "0000123456", "linkdt": "2010-01-01", "linkenddt": "2025-12-31"}]
    )
    ccm_path = tmp_path / "ccm.parquet"
    ccm.to_parquet(ccm_path, index=False)

    # Verify bounded prefix read works
    sample_html = filings / "cik=0000123456" / "acc_1" / "primary.html"
    assert len(read_bytes_prefix(sample_html, max_bytes=12)) == 12

    shards = list_lazy_prices_cik_shards(str(corpus), num_shards=4, ccm_uri=str(ccm_path))
    assert len(shards) == 1  # only 1 CRSP-mapped CIK survived filter

    shards_dir = tmp_path / "lp_shards"
    res = process_lazy_prices_cik_shard(
        shards[0],
        shard_id=0,
        shards_prefix=str(shards_dir),
        start_year=2010,
        max_bytes=50_000,
        max_workers=2,
    )
    assert res["diffs_written"] == 1
    assert res["filings_read"] == 2  # skipped 2008 10-K and 2020 8-K

    out_pq = tmp_path / "panel.parquet"
    merged = merge_lazy_prices_shards(str(shards_dir), str(out_pq))
    assert merged["diffs_written"] == 1
    df = pd.read_parquet(out_pq)
    assert list(df["cik"]) == ["0000123456"]
    assert df.iloc[0]["text_change"] > 0.8
