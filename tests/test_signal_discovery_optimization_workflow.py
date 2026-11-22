# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for signal discovery workflow result formatting helpers."""

import json
from types import SimpleNamespace

from signal_discovery_workflow.signal_discovery_optimization_workflow import (
    _compose_feedback,
    _format_workflow_result,
)


def _fenced_signal_json() -> str:
    return """```json
[
  {
    "name": "Ten Day Rank Momentum",
    "formula": "Rank(TS_Return(Close, 10))",
    "meaning": "Ranks 10-day close-price momentum",
    "category": "momentum",
    "data_fields_used": ["Close"],
    "lookback_periods": [10]
  }
]
```"""


def test_format_workflow_result_summarizes_fenced_signal_json():
    result = _format_workflow_result(
        status="best_effort",
        request="momentum signals",
        iteration=1,
        total_iterations=3,
        signal_json=_fenced_signal_json(),
        ic_results={"selected_signal": "signal_ten_day_rank_momentum", "mean_ic": 0.0123},
        saved_path="/tmp/signal.json",
        config=SimpleNamespace(ic_threshold=0.02, p_value_threshold=0.05),
    )

    payload = json.loads(result)
    assert payload["signals"] == [
        {
            "name": "Ten Day Rank Momentum",
            "formula": "Rank(TS_Return(Close, 10))",
            "category": "momentum",
            "data_fields_used": ["Close"],
            "lookback_periods": [10],
        }
    ]


def test_format_workflow_result_includes_evaluation_split():
    """The payload carries the train/test/full ICs and the split dates."""
    ic_results = {
        "selected_signal": "signal_roe",
        "mean_ic": 0.008,
        "p_value": 0.01,
        "num_periods": 1200,
        "sector_neutralized": True,
        "sector_coverage": {"mapped": 367, "total": 392},
        "ic_train": {"mean_ic": 0.012, "p_value": 0.001, "num_periods": 1300},
        "ic_test": {"mean_ic": 0.008, "p_value": 0.01, "num_periods": 1200},
        "ic_full": {"mean_ic": 0.010, "p_value": 0.002, "num_periods": 2500},
        "ic_raw_test": {"mean_ic": 0.015, "p_value": 0.001, "num_periods": 1200},
    }
    result = _format_workflow_result(
        status="best_effort",
        request="quality signals",
        iteration=3,
        total_iterations=3,
        signal_json=_fenced_signal_json(),
        ic_results=ic_results,
        saved_path="/tmp/signal.json",
        config=SimpleNamespace(
            ic_threshold=0.02,
            p_value_threshold=0.05,
            train_end_date="2020-12-31",
            test_start_date="2021-01-01",
        ),
    )
    payload = json.loads(result)
    evaluation = payload["evaluation"]
    assert evaluation["train_end_date"] == "2020-12-31"
    assert evaluation["test_start_date"] == "2021-01-01"
    assert evaluation["sector_neutralized"] is True
    assert evaluation["sector_coverage"] == {"mapped": 367, "total": 392}
    assert evaluation["ic_train"]["mean_ic"] == 0.012
    assert evaluation["ic_test"]["mean_ic"] == 0.008
    assert evaluation["ic_full"]["mean_ic"] == 0.010
    assert evaluation["ic_raw_test"]["mean_ic"] == 0.015
    # Headline metrics are the test IC.
    assert payload["metrics"]["mean_ic"] == 0.008


def test_compose_feedback_summarizes_fenced_best_signal_json():
    feedback = _compose_feedback(
        advice="- Try a shorter lookback",
        best_result={"signal_json": _fenced_signal_json(), "iteration": 1},
        best_ic=0.0123,
    )

    assert "Ten Day Rank Momentum: Rank(TS_Return(Close, 10))" in feedback
    assert "unable to summarize" not in feedback
