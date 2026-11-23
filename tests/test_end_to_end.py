# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0

"""End-to-end regressions for the pieces a run depends on.

These are the failures that are silent in production: a generated function that
never executes, a window that comes back empty, a scatter that drops months, or
a pin that does not actually verify. Each test asserts the observable result,
not an internal call count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_discovery_workflow import pins
from signal_discovery_workflow.fundamentals import _scatter
from signal_discovery_workflow.signal_evaluator import (
    compute_ic_split,
    compute_long_short_screen,
    execute_signal_functions,
)


def _panels(n_dates: int = 180, n_stocks: int = 40, seed: int = 7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2008-01-31", periods=n_dates, freq="ME")
    tickers = [f"crsp:permno:{10000 + i}" for i in range(n_stocks)]
    returns = pd.DataFrame(
        rng.normal(0.0, 0.05, size=(n_dates, n_stocks)), index=dates, columns=tickers
    )
    close = (1.0 + returns).cumprod()
    return dates, tickers, close, returns


def test_generated_signal_executes_and_uses_the_named_panel():
    dates, tickers, close, _ = _panels()
    bm = pd.DataFrame(
        np.tile(np.linspace(-1.0, 1.0, len(tickers)), (len(dates), 1)),
        index=dates,
        columns=tickers,
    )
    code = (
        "import pandas as pd\n"
        "import numpy as np\n"
        "def signal_value(book_to_market: pd.DataFrame):\n"
        "    return -book_to_market\n"
    )
    candidates = execute_signal_functions(code, {"BookToMarket": bm, "Close": close})
    assert [name for name, _ in candidates] == ["signal_value"]
    out = candidates[0][1]
    assert out.shape == bm.shape
    assert float(out.iloc[0, 0]) == pytest.approx(float(-bm.iloc[0, 0]))


def test_validation_window_is_populated_and_selection_uses_it():
    dates, tickers, close, returns = _panels()
    signal = close.pct_change(3)
    forward = close.shift(-1) / close - 1

    result = compute_ic_split(
        signal,
        forward,
        train_end_date="2010-12-31",
        test_start_date="2014-01-01",
        validation_end_date="2013-12-31",
        periods_per_year=12,
    )
    assert result["ic_validation"]["num_periods"] > 0
    assert result["ic_test"]["num_periods"] > 0
    assert result["selected_on"] == "validation"
    # The windows must not overlap: validation ends before test begins.
    assert result["ic_validation"]["num_periods"] + result["ic_test"]["num_periods"] <= len(dates)


def test_screen_returns_finite_metrics_on_a_real_signal():
    _, _, close, _ = _panels()
    signal = -close.pct_change(3)

    screen = compute_long_short_screen(
        signal, close, quantile=0.2, cost_bps=10.0, min_names=10, annualization=12
    )
    assert screen["num_days"] > 0
    assert np.isfinite(screen["gross"]["sharpe"])
    assert np.isfinite(screen["net"]["sharpe"])
    assert screen["annual_turnover"] >= 0.0


def test_scatter_repeats_each_record_across_its_valid_months():
    """A record valid for three months must appear in all three, not once."""
    values = pd.Series([5.0, 9.0])
    lengths = np.array([2, 1])          # record 0: Jan+Feb, record 1: Mar
    rows = np.array([0, 1, 2])
    cols = np.array([0, 0, 0])

    array = _scatter(values, lengths, rows, cols, (3, 1))
    assert array[:, 0].tolist() == [5.0, 5.0, 9.0]


def test_signal_with_no_validation_periods_is_flagged_not_scored_on_holdout():
    """A signal that only exists after 2021 must not be selected on 2021+."""
    _, tickers, close, _ = _panels()
    signal = close.pct_change(3)
    signal.loc[signal.index < "2021-01-01"] = np.nan
    forward = close.shift(-1) / close - 1

    result = compute_ic_split(
        signal,
        forward,
        train_end_date="2010-12-31",
        test_start_date="2014-01-01",
        validation_end_date="2013-12-31",
        periods_per_year=12,
    )
    assert result["selected_on"] == "validation"
    assert result["insufficient_history"] is True
    assert result["ic_validation"]["num_periods"] == 0
    assert result["mean_ic"] is None


def test_fundamentals_mandate_rejects_price_only_formulas():
    from signal_discovery_workflow.signal_generator import uses_fundamental_fields

    assert uses_fundamental_fields("Rank(Sqrt(TS_Std(Close, 90)))") is False
    assert uses_fundamental_fields("TS_Mean(Close, 20)") is False
    assert uses_fundamental_fields("Rank(BookToMarket)") is True
    assert uses_fundamental_fields("Div(ROE, TotalAssets)") is True
    assert uses_fundamental_fields(None) is False


def test_pin_verifies_exact_bytes_and_rejects_a_changed_release(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b'{"release":"v1"}')

    record = pins.pin(str(manifest), role="fundamentals")
    assert record["bytes"] == len(b'{"release":"v1"}')

    verified = pins.verify(str(manifest), record["sha256"], role="fundamentals")
    assert verified["verified"] is True

    with pytest.raises(pins.PinMismatch):
        pins.verify(str(manifest), "0" * 64, role="fundamentals")
