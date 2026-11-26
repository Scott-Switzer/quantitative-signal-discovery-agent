# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for QSD governance gates, PIT truncation check, PBO, and selftest."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from signal_discovery_workflow import governance, phf_export


class TestGovernanceAndSelftest(unittest.TestCase):
    def test_full_selftest_exam_passes(self) -> None:
        report = governance.run_selftest(seed=42)
        self.assertTrue(report["ok"], msg=json.dumps(report, indent=2))
        self.assertEqual(report["passed"], report["total"])
        self.assertGreaterEqual(report["total"], 8)

    def test_code_truncation_catches_lookahead_and_passes_causal(self) -> None:
        panel = governance._make_synthetic_panel(n_days=120, n_stocks=30, seed=7)

        leaky_code = (
            "def leaky_sig(Close):\n"
            "    return Close.shift(-1) / Close - 1.0\n"
        )
        leaky = governance.check_code_truncation_invariance(leaky_code, panel)
        self.assertFalse(leaky["ok"])

        causal_code = (
            "def causal_sig(Close):\n"
            "    return (Close / Close.shift(10) - 1.0).rank(axis=1, pct=True)\n"
        )
        causal = governance.check_code_truncation_invariance(causal_code, panel)
        self.assertTrue(causal["ok"])
        self.assertEqual(causal["max_abs_diff"], 0.0)

    def test_breakeven_cost_bps(self) -> None:
        bps = governance.compute_breakeven_cost_bps({
            "gross": {"annual_return": 0.12},
            "annual_turnover": 40.0,
            "periods_per_year": 252,
            "avg_impact_cost": 0.0,
            "avg_borrow_cost": 0.0,
        })
        self.assertIsNotNone(bps)
        self.assertAlmostEqual(bps, 30.0, places=4)

    def test_novelty_flags_disguised_momentum_and_passes_orthogonal(self) -> None:
        panel = governance._make_synthetic_panel(n_days=320, n_stocks=50, seed=11)
        canonical = governance.build_canonical_factor_panels(panel, periods_per_year=252)
        disguised = np.tanh((panel["Close"].shift(21) / panel["Close"].shift(252) - 1.0) * 5.0)
        res_bad = governance.check_factor_novelty(disguised, known_signal_panels=canonical)
        self.assertFalse(res_bad["ok"])
        self.assertEqual(res_bad["closest_factor"], "momentum_12_1")

        rng = np.random.default_rng(99)
        ortho = pd.DataFrame(
            rng.normal(0.0, 1.0, size=panel["Close"].shape),
            index=panel["Close"].index,
            columns=panel["Close"].columns,
        )
        res_ok = governance.check_factor_novelty(ortho, known_signal_panels=canonical)
        self.assertTrue(res_ok["ok"])

    def test_export_qsd_ui_latest_writes_valid_envelope(self) -> None:
        entry = governance.build_survivor_ledger_entry(
            signal_id="qsd_001",
            signal_name="test_factor",
            formula="cs_rank(close / delay(close, 20) - 1)",
            reason={
                "counterparty": "benchmark-constrained funds",
                "edge_type": "structural",
                "decay": "crowding in liquid large caps",
            },
            ic_results={
                "ic_validation": {"mean_ic": 0.032, "t_stat": 3.8},
                "ic_test": {"mean_ic": 0.024, "t_stat": 2.9},
            },
            screen_validation={
                "net": {"sharpe": 1.15},
                "gross": {"annual_return": 0.10},
                "annual_turnover": 25.0,
                "periods_per_year": 252,
                "deflated_sharpe": {"dsr": 0.98, "n_trials": 40},
                "subperiod_stability": {"positive_block_ratio": 1.0},
            },
            pit_check={"ok": True, "detail": "bit-identical"},
            novelty_check={"ok": True, "closest_factor": "momentum_12_1", "closest_corr": 0.14, "detail": "novel"},
            post_cutoff_status="PASS",
            pin_hash="pin-abc",
        )
        self.assertEqual(entry["gate_status"], "PROMOTE")
        self.assertEqual(entry["checks_passed"], entry["checks_total"])

        with tempfile.TemporaryDirectory() as tmp:
            uri = phf_export.export_qsd_ui_latest(
                tmp,
                survivors=[entry],
                n_trials_total=40,
                selftest_summary={"ok": True, "passed": 8, "total": 8},
                data_pins={"crsp_panel": "pin-abc"},
            )
            written = json.loads(Path(uri).read_text())
            self.assertEqual(written["schema_version"], "qsd-survivor-ledger/1.0.0")
            self.assertEqual(written["promoted_count"], 1)
            self.assertFalse(written["live_orders_allowed"])


if __name__ == "__main__":
    unittest.main()
