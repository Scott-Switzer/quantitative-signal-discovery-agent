# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the evaluation, split, FDR and ledger changes.

These are the properties the old implementation got wrong; each test names the
failure it prevents.
"""

import json

import numpy as np
import pandas as pd
import pytest

from signal_discovery_workflow import r2_store
from signal_discovery_workflow.phf_bridge import coverage_report, load_phf_panel
from signal_discovery_workflow.signal_evaluator import (
    compute_ic_split,
    compute_long_short_screen,
    compute_rank_ic,
    newey_west_t,
)
from signal_discovery_workflow.signal_generator import (
    format_operators_for_prompt,
    load_calculator_operators,
)
from signal_discovery_workflow.trial_registry import (
    TrialRegistry,
    benjamini_hochberg,
    normalize_formula,
)


def _panel(n_dates=200, n_stocks=25, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=n_dates, freq="B")
    tickers = [f"S{i:02d}" for i in range(n_stocks)]
    returns = pd.DataFrame(rng.standard_normal((n_dates, n_stocks)) * 0.01, index=dates, columns=tickers)
    return dates, tickers, returns


# ---------------------------------------------------------------------------
# HAC significance
# ---------------------------------------------------------------------------

class TestNeweyWest:
    def test_hac_shrinks_t_stat_on_autocorrelated_series(self):
        """The old code divided by std/sqrt(n) and ignored overlap.

        For a positively autocorrelated IC series the HAC standard error is
        larger, so |t| must fall. This is the difference between a factor that
        looks significant on 1,000 overlapping days and one that is not.
        """
        rng = np.random.default_rng(3)
        n = 500
        series = np.zeros(n)
        for i in range(1, n):
            series[i] = 0.85 * series[i - 1] + rng.normal(0, 0.1)
        series += 0.08

        t_naive, _, _ = newey_west_t(series, lags=0)
        t_hac, _, _ = newey_west_t(series, lags=8)

        assert t_naive is not None and t_hac is not None
        assert abs(t_hac) < abs(t_naive)

    def test_zero_mean_gives_p_one(self):
        t, p, _ = newey_west_t([0.0, 0.0, 0.0, 0.0], lags=0)
        assert t == pytest.approx(0.0)
        assert p == pytest.approx(1.0)

    def test_short_series_is_none(self):
        assert newey_west_t([1.0], lags=3)[0] is None


class TestRankICHac:
    def test_rank_ic_reports_hac_and_naive(self):
        _, _, returns = _panel()
        result = compute_rank_ic(returns, returns, periods=5)
        assert result["hac_lags"] > 0
        assert result["num_periods"] > 0
        # With a 5-day horizon the honest effective sample is about n/5.
        assert result["effective_periods"] <= result["num_periods"] / 4

    def test_default_periods_is_not_overlap_adjusted(self):
        _, _, returns = _panel()
        result = compute_rank_ic(returns, returns)
        assert result["hac_lags"] == 0

    def test_infinite_values_do_not_drop_whole_cross_sections(self):
        """Divide-by-zero returns become inf; inf must read as missing, not poison a sector mean."""
        dates, tickers, returns = _panel(n_dates=80, n_stocks=20)
        signal = returns.copy()
        signal.iloc[10, :5] = np.inf
        signal.iloc[20, :5] = -np.inf
        sector_map = {t: ("A" if i < 10 else "B") for i, t in enumerate(tickers)}
        clean = compute_rank_ic(signal, returns, sector_map=sector_map)
        assert clean["num_periods"] > 0


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

class TestPurgedSplit:
    def test_validation_split_headline_is_validation(self):
        dates, tickers, returns = _panel(n_dates=400)
        result = compute_ic_split(
            returns, returns, "2019-06-30", "2019-07-01",
            validation_end_date="2020-01-31", purge_periods=5,
        )
        assert result["selected_on"] == "validation"
        assert result["mean_ic"] == pytest.approx(result["ic_validation"]["mean_ic"])
        assert result["ic_test"]["mean_ic"] is not None

    def test_purge_removes_rows_at_the_boundary(self):
        dates, tickers, returns = _panel(n_dates=400)
        without = compute_ic_split(returns, returns, "2019-06-30", "2019-07-01", purge_periods=0)
        with_purge = compute_ic_split(returns, returns, "2019-06-30", "2019-07-01", purge_periods=10)
        assert with_purge["ic_train"]["num_periods"] == without["ic_train"]["num_periods"] - 10


# ---------------------------------------------------------------------------
# Multiple testing
# ---------------------------------------------------------------------------

class TestBenjaminiHochberg:
    def test_monotone_and_controls_family(self):
        pvalues = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]
        qvalues, rejected = benjamini_hochberg(pvalues, q=0.05)
        assert qvalues == sorted(qvalues)
        assert rejected[0] is True
        assert rejected[-1] is False

    def test_empty(self):
        assert benjamini_hochberg([], q=0.05) == ([], [])


# ---------------------------------------------------------------------------
# Trial ledger
# ---------------------------------------------------------------------------

class TestTrialRegistry:
    def test_records_and_deduplicates(self, tmp_path):
        path = tmp_path / "trials.jsonl"
        registry = TrialRegistry(path)
        registry.record({"formula": "Rank(TS_Return(Close, 20))", "mean_ic": 0.01, "p_value": 0.4})
        registry.record({"formula": "Rank ( TS_Return(Close, 20) )", "mean_ic": 0.02, "p_value": 0.3})

        assert len(registry.load()) == 2
        # Same formula modulo whitespace -> one entry in the tried list.
        assert len(registry.tried_keys()) == 1
        assert normalize_formula("A ( B )") == normalize_formula("A(B)")

    def test_fdr_across_history(self, tmp_path):
        registry = TrialRegistry(tmp_path / "trials.jsonl")
        for ic, p in [(0.05, 0.001), (0.01, 0.4), (0.02, 0.02), (0.015, 0.3)]:
            registry.record({"formula": f"f{ic}", "mean_ic": ic, "p_value": p})
        mt = registry.multiple_testing(q=0.05)
        assert mt["num_trials"] == 4
        assert mt["num_significant"] >= 1

    def test_summary_and_leaderboard_use_one_read(self, tmp_path, monkeypatch):
        """The ledger is a network object: repeated queries must not re-read it."""
        path = tmp_path / "trials.jsonl"
        registry = TrialRegistry(path)
        registry.record({"formula": "f", "name": "F", "mean_ic": 0.05, "p_value": 0.001, "t_stat": 3.0})

        reads = []
        original = r2_store.read_text
        monkeypatch.setattr(r2_store, "read_text", lambda uri: reads.append(uri) or original(uri))

        registry.summary()
        registry.leaderboard()
        assert reads == []


# ---------------------------------------------------------------------------
# Operator exposure and screen
# ---------------------------------------------------------------------------

class TestOperatorExposure:
    def test_all_operators_reach_the_prompt_by_default(self):
        """The old default of 30 hid 36 of 66 operators from the generator."""
        operators = load_calculator_operators()
        rendered = format_operators_for_prompt(operators)
        shown = [line for line in rendered.splitlines() if line.startswith("- ")]
        assert len(shown) == len(operators)
        assert "CS_Rank" in rendered
        assert "Neutralize" in rendered


class TestR2Store:
    def test_local_round_trip_and_join(self, tmp_path):
        uri = r2_store.join(str(tmp_path), "nested/ledger.jsonl")
        assert uri.endswith("nested/ledger.jsonl")
        r2_store.write_text(uri, "a\nb\n")
        assert r2_store.exists(uri)
        assert r2_store.read_text(uri) == "a\nb\n"

    def test_remote_detection(self):
        assert r2_store.is_remote("r2://bucket/key")
        assert r2_store.is_remote("s3://bucket/key")
        assert not r2_store.is_remote("/tmp/key")

    def test_missing_remote_but_well_formed_uri_raises_without_credentials(self, monkeypatch):
        monkeypatch.delenv("R2_S3_ENDPOINT", raising=False)
        monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
        r2_store._filesystem = None
        with pytest.raises(RuntimeError, match="missing R2_S3_ENDPOINT"):
            r2_store.read_bytes("r2://bucket/key")
        r2_store._filesystem = None


class TestBridgeInMemory:
    def _panel(self, path):
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-01-31", "2020-02-29"]),
                "security_id": ["a", "a", "b", "b"],
                "ret_total": [0.10, 0.10, -0.05, 0.02],
                "dollar_volume": [1e6, 2e6, 3e6, 4e6],
                "siccd": [2834, 2834, 6020, 6020],
                "investable": [1, 1, 1, 1],
            }
        )
        frame.to_parquet(path, index=False)
        return path

    def test_builds_panels_without_writing_csvs(self, tmp_path):
        path = self._panel(tmp_path / "panel.parquet")
        panels, sectors = load_phf_panel(path)

        assert set(panels) == {"Close", "Returns", "Volume"}
        assert panels["Close"].loc[pd.Timestamp("2020-02-29"), "a"] == pytest.approx(1.21)
        assert sectors["a"] == "SIC:Manufacturing"
        assert sectors["b"] == "SIC:Finance"
        # The only file in the directory is the source panel.
        assert [p.name for p in tmp_path.iterdir()] == ["panel.parquet"]

    def test_missing_required_column_raises(self, tmp_path):
        path = tmp_path / "bad.parquet"
        pd.DataFrame({"date": ["2020-01-31"], "security_id": ["a"]}).to_parquet(path, index=False)
        with pytest.raises(ValueError, match="ret_total"):
            load_phf_panel(path)

    def test_coverage_reports_frequency(self, tmp_path):
        panels, sectors = load_phf_panel(self._panel(tmp_path / "p.parquet"))
        report = coverage_report(panels, sectors)
        assert report["frequency"] == "monthly"
        assert report["securities"] == 2


class TestLongShortScreen:
    def test_perfect_signal_is_profitable_before_costs(self):
        _, _, returns = _panel(n_dates=300, n_stocks=30)
        close = (1 + returns).cumprod() * 100
        # Signal is the next-day return: leakage by construction, used only to
        # prove the screen finds a real edge.
        signal = returns.shift(-1)
        result = compute_long_short_screen(signal, close, cost_bps=0.0, min_names=20)
        assert "error" not in result
        assert result["gross"]["sharpe"] > 1.0
        assert result["net"]["sharpe"] <= result["gross"]["sharpe"]

    def test_costs_reduce_net_sharpe(self):
        _, _, returns = _panel(n_dates=300, n_stocks=30, seed=4)
        close = (1 + returns).cumprod() * 100
        signal = returns.shift(-1)
        free = compute_long_short_screen(signal, close, cost_bps=0.0, min_names=20)
        costly = compute_long_short_screen(signal, close, cost_bps=50.0, min_names=20)
        assert costly["net"]["sharpe"] < free["net"]["sharpe"]
