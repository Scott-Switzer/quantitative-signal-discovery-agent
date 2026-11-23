# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regressions for the defects that made the first fundamentals run wrong.

Each test reproduces a concrete way a panel or a selection can look plausible
while being numerically wrong: a quarterly flow used as an annual value,
profitability scaled by the wrong assets, a pin that blocks a daily release, or
a ratio whose denominator crosses zero.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from signal_discovery_workflow import pins, r2_store
from signal_discovery_workflow.fundamentals import (
    _annual_facts,
    build_sec_fundamental_panels,
)
from signal_discovery_workflow.signal_code_generator import _check_unstable_denominator
from signal_discovery_workflow.signal_evaluator import compute_long_short_screen


def _bridge() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cik": ["0000000001"],
            "permno": ["10000"],
            "effective_from": [pd.Timestamp("1990-01-01")],
            "effective_to": [pd.Timestamp("2262-01-01")],
        }
    )


def _facts(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_annual_filter_prevents_a_quarterly_flow_becoming_the_valuation_input():
    """A 10-Q filed before the 10-K must not win the same period end.

    The old loader dropped duplicate (cik, end, metric) rows by earliest filed
    date without looking at ``fp``/``form``. A Q3 net income could therefore be
    divided by a full-year market cap, which is why the first run's
    EarningsYield median was 0.0073 instead of a plausible ~0.04.
    """
    facts = _facts(
        [
            # Quarterly figure, filed first (10-Q).
            {"cik": "1", "metric": "us-gaap:NetIncomeLoss", "end": "2020-12-31",
             "filed": "2020-11-15", "fp": "Q3", "form": "10-Q", "value": 25.0},
            # Annual figure for the same end, filed later (10-K).
            {"cik": "1", "metric": "us-gaap:NetIncomeLoss", "end": "2020-12-31",
             "filed": "2021-02-15", "fp": "FY", "form": "10-K", "value": 100.0},
            {"cik": "1", "metric": "us-gaap:Assets", "end": "2020-12-31",
             "filed": "2021-02-15", "fp": "FY", "form": "10-K", "value": 1000.0},
            {"cik": "1", "metric": "us-gaap:StockholdersEquity", "end": "2020-12-31",
             "filed": "2021-02-15", "fp": "FY", "form": "10-K", "value": 500.0},
        ]
    )
    annual = _annual_facts(facts)
    assert len(annual) == 3
    assert 25.0 not in annual["value"].tolist()

    dates = pd.date_range("2020-12-31", periods=8, freq="ME")
    securities = pd.Index(["crsp:permno:10000"])
    market_cap = pd.DataFrame(2000.0, index=dates, columns=securities)

    panels = build_sec_fundamental_panels(
        annual, _bridge(), dates, securities, market_cap=market_cap
    )
    earnings_yield = panels["EarningsYield"]
    # 100 / 2000, not 25 / 2000.
    assert float(earnings_yield.loc["2021-03-31", "crsp:permno:10000"]) == pytest.approx(0.05)
    assert float(panels["BookToMarket"].loc["2021-03-31", "crsp:permno:10000"]) == pytest.approx(0.25)


def test_profitability_scales_by_lagged_assets():
    """Scaling by current assets embeds asset growth (Hou, Xue & Zhang 2020)."""
    facts = _facts(
        [
            {"cik": "1", "metric": "us-gaap:Assets", "end": "2019-12-31",
             "filed": "2020-02-15", "fp": "FY", "form": "10-K", "value": 1000.0},
            {"cik": "1", "metric": "us-gaap:Assets", "end": "2020-12-31",
             "filed": "2021-02-15", "fp": "FY", "form": "10-K", "value": 1100.0},
            {"cik": "1", "metric": "us-gaap:OperatingIncomeLoss", "end": "2020-12-31",
             "filed": "2021-02-15", "fp": "FY", "form": "10-K", "value": 100.0},
        ]
    )
    annual = _annual_facts(facts)
    dates = pd.date_range("2019-01-31", periods=30, freq="ME")
    securities = pd.Index(["crsp:permno:10000"])
    panels = build_sec_fundamental_panels(annual, _bridge(), dates, securities)

    # 2021-02 has the 2020 report; lagged assets are the 2019 value, 1000.
    value = float(panels["OperatingProfitability"].loc["2021-02-28", "crsp:permno:10000"])
    assert value == pytest.approx(0.10)
    # Current assets would give 100/1100 = 0.0909.
    assert value != pytest.approx(100.0 / 1100.0)
    # Asset growth remains available and is measured against the same lag.
    assert float(panels["AssetGrowth"].loc["2021-02-28", "crsp:permno:10000"]) == pytest.approx(0.10)


def test_pin_record_policy_logs_instead_of_blocking_a_daily_release(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b'{"release":"v1"}')
    original = pins.pin(str(manifest))["sha256"]

    manifest.write_bytes(b'{"release":"v2"}')
    recorded = pins.verify(str(manifest), original, policy="record")
    assert recorded["verified"] is False
    assert recorded["mismatch"] is True
    assert recorded["sha256"] != original

    with pytest.raises(pins.PinMismatch):
        pins.verify(str(manifest), original, policy="enforce")


def test_resolve_release_picks_the_newest_child_by_mtime(tmp_path):
    family = tmp_path / "crsp-panel"
    for name, marker in (("release-old", 1_000.0), ("release-new", 2_000.0)):
        release = family / name
        release.mkdir(parents=True)
        (release / "panel.parquet").write_bytes(b"parquet")
        (release / "manifest.json").write_bytes(json.dumps({"release": name}).encode())
        os.utime(release / "panel.parquet", (marker, marker))

    resolved = pins.resolve_release(str(family))
    assert resolved["release_uri"].endswith("release-new")
    assert resolved["artifact_uri"].endswith("release-new/panel.parquet")
    assert resolved["manifest_uri"].endswith("release-new/manifest.json")
    assert resolved["candidates"] == 2


def test_ratio_of_a_demeaned_panel_is_rejected():
    assert _check_unstable_denominator(
        "Div(BookToMarket, CS_Demean(EarningsYield))"
    )
    assert _check_unstable_denominator("Div(CS_Demean(BookToMarket), EarningsYield)") is None
    assert _check_unstable_denominator("Div(BookToMarket, EarningsYield)") is None


def test_monthly_screen_is_labelled_monthly_and_can_flip_the_book():
    rng = np.random.default_rng(11)
    dates = pd.date_range("2018-01-31", periods=120, freq="ME")
    tickers = [f"S{i}" for i in range(30)]
    returns = pd.DataFrame(
        rng.normal(0, 0.04, (len(dates), len(tickers))), index=dates, columns=tickers
    )
    close = (1 + returns).cumprod()
    signal = returns.shift(-1)  # deliberately predictive for the test

    long_high = compute_long_short_screen(
        signal, close, cost_bps=0.0, min_names=20, annualization=12, direction=1.0
    )
    long_low = compute_long_short_screen(
        signal, close, cost_bps=0.0, min_names=20, annualization=12, direction=-1.0
    )
    assert long_high["horizon"] == "monthly rebalance"
    assert long_high["gross"]["sharpe"] > 1.0
    assert long_low["gross"]["sharpe"] == pytest.approx(-long_high["gross"]["sharpe"])


def test_list_entries_keeps_the_remote_scheme(monkeypatch):
    """A bare `bucket/key` is a local path to `stat`, so listing must keep r2://.

    Without this, `resolve_release` joined `manifest.json` onto a scheme-less
    path, `stat` looked for it on the local disk, every release looked empty and
    the run died with "neither a release nor a directory of usable releases".
    """
    import pyarrow.fs as pafs

    class FakeFS:
        def get_file_info(self, selector):  # noqa: ANN001 - mimics pyarrow
            return [pafs.FileInfo("bucket/prefix/release-1", type=pafs.FileType.Directory)]

    monkeypatch.setattr(r2_store, "_filesystem", FakeFS())

    entries = r2_store.list_entries("r2://bucket/prefix/")

    assert [entry["path"] for entry in entries] == ["r2://bucket/prefix/release-1"]
    assert entries[0]["is_dir"] is True


def test_screen_trades_the_same_sector_neutral_object_as_the_ic():
    """A within-sector effect must not be washed out by a sector bet.

    The signal is a sector offset plus a within-sector rank; returns are a
    sector offset plus the same rank. The raw quintile book buys one whole
    sector and loses the sector spread, while the neutral book trades the
    ordering the IC measures and earns it.
    """
    dates = pd.date_range("2000-01-31", periods=240, freq="ME")
    groups = {"A": -0.01, "B": +0.01}
    tickers = [f"{sector}{i}" for sector in groups for i in range(10)]
    sector_map = {ticker: ticker[0] for ticker in tickers}
    rank = {ticker: int(ticker[1:]) for ticker in tickers}
    # The signal ranks sector A highest, but sector A is the one that loses.
    signal = pd.DataFrame(
        {
            ticker: (1000.0 if ticker[0] == "A" else 0.0) + rank[ticker]
            for ticker in tickers
        },
        index=dates,
        dtype=float,
    )

    monthly = pd.DataFrame(
        {
            ticker: groups[ticker[0]] + 0.002 * (rank[ticker] - 4.5)
            for ticker in tickers
        },
        index=dates,
    )
    monthly.iloc[0] = 0.0
    close = (1.0 + monthly).cumprod()

    raw = compute_long_short_screen(
        signal, close, cost_bps=0.0, annualization=12, min_names=20
    )
    neutral = compute_long_short_screen(
        signal,
        close,
        cost_bps=0.0,
        annualization=12,
        min_names=20,
        sector_map=sector_map,
    )

    assert raw["net"]["sharpe"] < 0
    assert neutral["sector_neutral"] is True
    assert neutral["net"]["sharpe"] > 1.0
    assert neutral["monotonicity"] > 0.9
    assert neutral["spread_t_stat"] > 2.0
