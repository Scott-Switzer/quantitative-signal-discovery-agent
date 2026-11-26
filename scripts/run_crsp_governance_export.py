#!/usr/bin/env python3
"""Evaluate a pre-specified signal family on the pinned CRSP panel and export
``ui_latest.json`` through :func:`phf_export.export_qsd_ui_latest`.

This is the production-data path, not a fixture: it reads the pinned
``crsp-panel-1.2.2`` parquet (downloaded from R2), evaluates every candidate
with QSD's own ``signal_evaluator`` and ``governance`` gates, selects on the
validation window only, and records the untouched 2022-2024 test window as a
post-cutoff audit. It is a manual governance evaluation, not an LLM discovery
loop; the provenance block says so.

    PYTHONPATH=src .venv/bin/python scripts/run_crsp_governance_export.py \
      --panel output/crsp-panel-1.2.2/panel.parquet \
      --out output/qsd-ui-latest
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from signal_discovery_workflow import governance, overfitting, phf_bridge, phf_export, signal_evaluator

PANEL_URI = (
    "r2://financial-data-wrds-lakehouse/research/datasets/crsp_monthly/"
    "crsp-panel-1.2.2/918ff8c6a5c4e25e-9d684a69b938e435f85c068b16040cf3/panel.parquet"
)
PANEL_SHA256 = "395a3f5eef94e5e481c1cc5ffb00b9317b5010836fb887c34c08fc1ffb27e7d4"
PANEL_PIN = "9d684a69b938e435f85c068b16040cf3"
START = "2010-01-01"
END = "2024-12-31"
TRAIN_END = "2017-12-31"
VALIDATION_END = "2021-12-31"
TEST_START = "2022-01-01"
FORWARD_PERIODS = 21
PERIODS_PER_YEAR = 252
MIN_NAMES = 100
COST_BPS = 10.0


def _rank(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, pct=True)


def _returns(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return panel["Close"].pct_change()


def _volume_surge(panel: dict[str, pd.DataFrame], lookback: int) -> pd.DataFrame:
    volume = panel["Volume"]
    return _rank(volume / volume.rolling(lookback, min_periods=max(3, lookback // 3)).mean())


def _dollar_turnover_trend(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    turnover = panel["Volume"] / panel["MarketCap"]
    return _rank(turnover.rolling(20, min_periods=8).mean())


def _amihud_12m(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    returns = _returns(panel)
    return _rank((returns.abs() / panel["Volume"]).rolling(12, min_periods=8).mean())


def _skew_36m(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    returns = _returns(panel)
    return returns.rolling(36, min_periods=24).skew()


def _vw_flow_12m(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    returns = _returns(panel)
    volume = panel["Volume"]
    return _rank(
        (returns * volume).rolling(12, min_periods=8).sum()
        / volume.rolling(12, min_periods=8).sum()
    )


def candidates() -> list[dict[str, Any]]:
    """The pre-specified family. Direction is fixed before evaluation."""
    return [
        {
            "signal_id": "qsd_volume_surge_10d",
            "signal_name": "volume_surge_10d",
            "formula": "cs_rank(volume / mean(volume, 10))",
            "family": "liquidity_flow",
            "direction": 1.0,
            "reason": {
                "counterparty": "short-horizon liquidity demanders absorbing attention shocks",
                "edge_type": "behavioral",
                "decay": "fades as liquidity normalises or the flow becomes crowded",
            },
            "fn": lambda panel: _volume_surge(panel, 10),
        },
        {
            "signal_id": "qsd_volume_surge_20d",
            "signal_name": "volume_surge_20d",
            "formula": "cs_rank(volume / mean(volume, 20))",
            "family": "liquidity_flow",
            "direction": 1.0,
            "reason": {
                "counterparty": "short-horizon liquidity demanders absorbing attention shocks",
                "edge_type": "behavioral",
                "decay": "fades as liquidity normalises or the flow becomes crowded",
            },
            "fn": lambda panel: _volume_surge(panel, 20),
        },
        {
            "signal_id": "qsd_volume_surge_60d",
            "signal_name": "volume_surge_60d",
            "formula": "cs_rank(volume / mean(volume, 60))",
            "family": "liquidity_flow",
            "direction": 1.0,
            "reason": {
                "counterparty": "short-horizon liquidity demanders absorbing attention shocks",
                "edge_type": "behavioral",
                "decay": "fades as liquidity normalises or the flow becomes crowded",
            },
            "fn": lambda panel: _volume_surge(panel, 60),
        },
        {
            "signal_id": "qsd_dollar_turnover_trend",
            "signal_name": "dollar_turnover_trend",
            "formula": "cs_rank(mean(dollar_volume / market_cap, 20))",
            "family": "liquidity_flow",
            "direction": 1.0,
            "reason": {
                "counterparty": "slow institutions under-reacting to rising participation",
                "edge_type": "structural",
                "decay": "fades when participation mean-reverts or trading costs absorb the edge",
            },
            "fn": _dollar_turnover_trend,
        },
        {
            "signal_id": "qsd_amihud_12m",
            "signal_name": "amihud_illiquidity_12m",
            "formula": "cs_rank(mean(abs(return) / dollar_volume, 12))",
            "family": "liquidity_flow",
            "direction": 1.0,
            "reason": {
                "counterparty": "liquidity-constrained sellers needing immediacy",
                "edge_type": "risk_premium",
                "decay": "fades if the illiquidity premium is arbitraged away",
            },
            "fn": _amihud_12m,
        },
        {
            "signal_id": "qsd_skew_36m",
            "signal_name": "return_skew_36m",
            "formula": "cs_rank(skew(return) over 36 months)",
            "family": "behavioral",
            "direction": -1.0,
            "reason": {
                "counterparty": "lottery-seeking investors overpaying for positive skew",
                "edge_type": "behavioral",
                "decay": "fades when lottery demand weakens or the sample regime changes",
            },
            "fn": _skew_36m,
        },
        {
            "signal_id": "qsd_vw_flow_12m",
            "signal_name": "volume_weighted_flow_12m",
            "formula": "cs_rank(sum(return * dollar_volume, 12) / sum(dollar_volume, 12))",
            "family": "liquidity_flow",
            "direction": 1.0,
            "reason": {
                "counterparty": "price-insensitive flow that moves volume before price",
                "edge_type": "informational",
                "decay": "fades when volume no longer leads price or the flow reverses",
            },
            "fn": _vw_flow_12m,
        },
    ]


def cutoff_status(validation: dict[str, Any], test: dict[str, Any]) -> str:
    """Classify the untouched 2022-2024 window relative to validation."""
    val_ic = validation.get("mean_ic")
    test_ic = test.get("mean_ic")
    test_t = test.get("t_stat")
    if val_ic is None or test_ic is None or test_t is None:
        return "UNVERIFIED"
    same_sign = (val_ic >= 0) == (test_ic >= 0)
    if same_sign and abs(test_t) >= 2.0:
        return "PASS"
    if same_sign:
        return "WEAK"
    return "FAIL"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, help="Local copy of the pinned panel.parquet")
    parser.add_argument("--out", required=True, help="Directory that receives ui_latest.json")
    parser.add_argument("--no-selftest", action="store_false", dest="selftest")
    args = parser.parse_args()

    panel_path = Path(args.panel)
    digest = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    if digest != PANEL_SHA256:
        raise SystemExit(f"panel sha256 mismatch: {digest} != {PANEL_SHA256}")

    panel, _sectors = phf_bridge.load_phf_panel(str(panel_path), start=START, end=END)
    family = candidates()
    # Evaluate once, preserving each candidate's validation screen alongside
    # its ledger entry so the PBO rebuild cannot silently drop the cost gate.
    close = panel["Close"]
    forward = signal_evaluator.compute_forward_returns(close, periods=FORWARD_PERIODS)
    canonical = governance.build_canonical_factor_panels(panel, periods_per_year=PERIODS_PER_YEAR)
    evaluated: list[dict[str, Any]] = []
    for candidate in family:
        signal = candidate["fn"](panel)
        ic = signal_evaluator.compute_ic_split(
            signal,
            forward,
            train_end_date=TRAIN_END,
            validation_end_date=VALIDATION_END,
            test_start_date=TEST_START,
            purge_periods=FORWARD_PERIODS,
            periods=FORWARD_PERIODS,
            min_names=MIN_NAMES,
            periods_per_year=PERIODS_PER_YEAR,
        )
        screen = signal_evaluator.compute_long_short_screen(
            signal.loc[:VALIDATION_END],
            close.loc[:VALIDATION_END],
            quantile=0.2,
            cost_bps=COST_BPS,
            min_names=MIN_NAMES,
            annualization=PERIODS_PER_YEAR,
            direction=candidate["direction"],
            include_series=True,
        )
        if "error" in screen:
            raise RuntimeError(f"{candidate['signal_name']}: {screen['error']}")
        net = screen.pop("_net_series")
        screen["deflated_sharpe"] = overfitting.deflated_sharpe_ratio(
            net, n_trials=len(family), periods_per_year=PERIODS_PER_YEAR
        )
        screen["subperiod_stability"] = overfitting.subperiod_stability(
            net, n_blocks=4, periods_per_year=PERIODS_PER_YEAR
        )
        evaluated.append(
            {
                "candidate": candidate,
                "ic": ic,
                "screen": screen,
                "net": net,
                "novelty": governance.check_factor_novelty(signal, known_signal_panels=canonical),
                "pit": governance.check_truncation_invariance(candidate["fn"], panel),
            }
        )

    net_matrix = pd.concat([row["net"] for row in evaluated], axis=1).dropna()
    pbo = governance.probability_of_backtest_overfitting(net_matrix, n_blocks=8)

    entries = []
    for row in evaluated:
        candidate = row["candidate"]
        validation = row["ic"]["ic_validation"]
        test = row["ic"]["ic_test"]
        entries.append(
            governance.build_survivor_ledger_entry(
                signal_id=candidate["signal_id"],
                signal_name=candidate["signal_name"],
                formula=candidate["formula"],
                family=candidate["family"],
                hold_periods=FORWARD_PERIODS,
                reason=candidate["reason"],
                ic_results={"ic_validation": validation, "ic_test": test, "ic_full": row["ic"]["ic_full"]},
                screen_validation=row["screen"],
                pit_check=row["pit"],
                novelty_check=row["novelty"],
                pbo_check=pbo,
                post_cutoff_status=cutoff_status(validation, test),
                pin_hash=PANEL_PIN,
            )
        )

    selftest = governance.run_selftest(seed=42) if args.selftest else None
    selftest_summary = (
        {key: selftest[key] for key in ("ok", "passed", "total", "ran_at")}
        if selftest is not None
        else None
    )
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    provenance = {
        "run_id": f"qsd-crsp-2010-2024-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "dataset": "CRSP monthly panel 1.2.2, 2010-01-29..2024-12-31, 7,713 securities",
        "source": PANEL_URI,
        "panel_sha256": PANEL_SHA256,
        "panel_pin_hash": PANEL_PIN,
        "command": (
            "PYTHONPATH=src .venv/bin/python scripts/run_crsp_governance_export.py "
            "--panel output/crsp-panel-1.2.2/panel.parquet --out output/qsd-ui-latest"
        ),
        "executed_at": now,
        "data_class": "PRODUCTION_RESEARCH",
        "production": True,
        "run_kind": "manual_governance_evaluation",
        "selection": {
            "train_end": TRAIN_END,
            "validation_end": VALIDATION_END,
            "test_start": TEST_START,
            "forward_periods": FORWARD_PERIODS,
            "cost_bps": COST_BPS,
            "min_names": MIN_NAMES,
        },
        "note": (
            "Pre-specified 7-signal price/volume family selected on validation only; "
            "2022-2024 is reported as a post-cutoff audit. Not an LLM discovery loop."
        ),
    }
    path = phf_export.export_qsd_ui_latest(
        args.out,
        survivors=entries,
        n_trials_total=len(family),
        selftest_summary=selftest_summary,
        data_pins={
            "crsp_panel": PANEL_SHA256,
            "crsp_panel_version": "crsp-panel-1.2.2",
            "crsp_panel_pin_hash": PANEL_PIN,
        },
        provenance=provenance,
    )
    summary = {
        "path": path,
        "n_trials_total": len(family),
        "entries": [
            {
                "signal": entry["signal_name"],
                "gate_status": entry["gate_status"],
                "post_cutoff_status": entry["post_cutoff_status"],
                "checks_passed": entry["checks_passed"],
                "checks_total": entry["checks_total"],
                "validation_ic": entry["metrics"]["validation_ic"],
                "validation_hac_t": entry["metrics"]["validation_hac_t"],
                "validation_net_sharpe": entry["metrics"]["validation_net_sharpe"],
                "breakeven_cost_bps": entry["metrics"]["breakeven_cost_bps"],
                "dsr": entry["metrics"]["dsr"],
            }
            for entry in entries
        ],
        "pbo": pbo.get("pbo"),
        "selftest": selftest_summary,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
