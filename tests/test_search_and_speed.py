# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for search-loop quality, deterministic codegen, 2025 vault cap, and PHF receipts."""

from __future__ import annotations

import asyncio
import json

from signal_discovery_workflow.phf_export import build_phf_run_payload
from signal_discovery_workflow.runner import load_yaml_config
from signal_discovery_workflow.signal_code_generator import generate_signal_code
from signal_discovery_workflow.signal_discovery_optimization_workflow import (
    _compose_feedback,
    _no_candidate_advice,
)
from signal_discovery_workflow.signal_generator import build_signal_prompt, load_calculator_operators


def test_recent_tried_formulas_and_frequency_aware_prompt():
    tried = [f"TS_Mean(Close, {i})" for i in range(1, 85)]
    prompt = build_signal_prompt(
        "value and quality",
        4,
        "- CS_Rank(x)",
        "```json\n[]\n```",
        tried_formulas=tried,
        horizon=1,
        periods_per_year=12,
        residualize_returns=True,
    )
    # Must show the MOST RECENT 60 formulas (25..84), NOT the oldest 60 (1..60).
    assert "TS_Mean(Close, 84)" in prompt
    assert "TS_Mean(Close, 1)\n" not in prompt
    # Must reflect monthly frequency & Barra-residualized target.
    assert "monthly panel" in prompt
    assert "Barra-residualized 1-month forward return" in prompt


def test_deterministic_codegen_and_single_winner_feedback():
    ops = load_calculator_operators()
    sig_json = json.dumps(
        [
            {
                "name": "FCF Quality",
                "formula": "CS_Rank(FcfYield)",
                "meaning": "Cross-sectional FCF yield rank",
                "data_fields_used": ["FcfYield"],
            }
        ]
    )
    # llm=None proves no LLM call is required when specs pass AST validation.
    code = asyncio.run(generate_signal_code(None, sig_json, ops))
    assert "def signal_fcf_quality(FcfYield: pd.DataFrame) -> pd.DataFrame:" in code
    assert "return CS_Rank(FcfYield)" in code

    best_result = {
        "name": "signal_winner",
        "iteration": 2,
        "meta": {"display_name": "Winner Only", "formula": "CS_Rank(FcfYield)"},
        "validation_screen": {"net": {"sharpe": 0.62}, "monotonicity": 0.75},
        "signal_json": json.dumps(
            [
                {"name": "Winner Only", "formula": "CS_Rank(FcfYield)"},
                {"name": "Bad Sibling", "formula": "CS_Rank(Close)"},
            ]
        ),
    }
    fb = _compose_feedback("- try macro gating", best_result, 0.025)
    assert "Winner Only: CS_Rank(FcfYield)" in fb
    assert "val_sharpe=+0.62" in fb
    assert "Bad Sibling" not in fb


def test_duplicate_only_batch_reports_ledger_collision_not_degeneracy():
    duplicate_advice = _no_candidate_advice(2)
    assert "already in the trial ledger" in duplicate_advice
    assert "All 2 generated formulas" in duplicate_advice
    assert "degenerate" not in duplicate_advice
    assert "degenerate" in _no_candidate_advice(0)


def test_weekly_vault_cap_and_phf_phase2_receipt_fields():
    _, wf = load_yaml_config("configs/config-crsp-weekly.yml")
    assert wf["weekly_end_date"] == "2024-12-31"

    receipt, results = build_phf_run_payload(
        run_id="test-run",
        signal_name="signal_test",
        formula="CS_Rank(FcfYield)",
        request_text="test",
        ic_results={"selected_on": "validation", "ic_validation": {"mean_ic": 0.03}},
        screen_validation={
            "sector_neutral": True,
            "vol_scaled": True,
            "short_borrow_bps": 50.0,
            "impact_coeff": 0.15,
            "max_weight": 0.03,
            "net": {"sharpe": 0.8},
            "deflated_sharpe": {"dsr": 0.97, "n_trials": 50},
            "subperiod_stability": {"positive_block_ratio": 0.75},
        },
        screen_full={"sector_neutral": True, "vol_scaled": True, "net": {"sharpe": 0.6}},
        factor_alpha={"alpha_t_stat": 2.4},
        data_pins={},
        config_summary={"min_dsr": 0.95},
    )
    assert receipt["checks"]["deflated_sharpe_checked"]["ok"] is True
    assert receipt["checks"]["subperiod_stability_checked"]["ok"] is True
    assert receipt["checks"]["execution_realism"]["ok"] is True
    assert receipt["results_digest"]["signal_test"]["validation_dsr"] == 0.97
    assert results["headline"]["validation_subperiod_stability"] == 0.75


def test_alpha_pool_library_separates_monthly_and_weekly_sleeves(tmp_path):
    from signal_discovery_workflow.alpha_combiner import AlphaPoolLibrary

    lib_m = AlphaPoolLibrary(tmp_path / "pool")
    lib_m.save_members(
        [{"name": "m_sig", "formula": "CS_Rank(FcfYield)", "direction": 1.0, "val_ic": 0.02}],
        run_id="r_m",
        horizon=1,
        periods_per_year=12,
    )
    lib_w = AlphaPoolLibrary(tmp_path / "pool")
    lib_w.save_members(
        [{"name": "w_sig", "formula": "CS_Rank(AmihudIlliq)", "direction": 1.0, "val_ic": 0.018}],
        run_id="r_w",
        horizon=1,
        periods_per_year=52,
    )

    monthly_only = lib_m.load_all(horizon=1, periods_per_year=12)
    weekly_only = lib_w.load_all(horizon=1, periods_per_year=52)
    assert [m["name"] for m in monthly_only] == ["m_sig"]
    assert [m["name"] for m in weekly_only] == ["w_sig"]


def test_trial_registry_no_double_append_and_parallel_shards(tmp_path):
    from signal_discovery_workflow.trial_registry import TrialRegistry

    r1 = TrialRegistry(tmp_path / "trials.jsonl", sharded=True)
    cached = r1.load()
    r1.record({"formula": "CS_Rank(FcfYield)", "mean_ic": 0.02, "p_value": 0.01})
    assert len(cached) == 1  # record() mutates _cache in place; no duplicate

    r2 = TrialRegistry(tmp_path / "trials.jsonl", sharded=True)
    r2.record({"formula": "CS_Rank(EbitToEv)", "mean_ic": 0.025, "p_value": 0.005})

    r3 = TrialRegistry(tmp_path / "trials.jsonl", sharded=True)
    assert len(r3.load()) == 2

