#!/usr/bin/env python3
"""Evaluate an orthogonal 7-signal corporate-action / structural / seasonal family
on the pinned CRSP monthly panel (crsp-panel-1.2.2) through the frozen QSD
governance gates.

Uses proper monthly periodicity (periods_per_year=12, 1-month forward return)
and builds signals orthogonal to raw price/volume/size:
  1. net_share_issuance_12m    — split-adjusted 12m share contraction (buyback yield)
  2. issuance_accel_6v12       — 6m vs prior 6m share-count deceleration
  3. div_yield_12m             — trailing 12m implied dividend yield from ret_total vs retx
  4. ind_rel_mom_12_1          — 2-digit SIC demeaned 12-1 momentum
  5. high_52w_ind_rel          — 2-digit SIC demeaned 12m-high proximity (anchoring)
  6. max_ret_lottery_12m       — negative trailing 12m max monthly return (SIC demeaned)
  7. same_month_seasonality_5y — Heston-Sadka 5y same-calendar-month historical return
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from signal_discovery_workflow import governance, overfitting, phf_export, signal_evaluator

PANEL_PATH = Path("output/crsp-panel-1.2.2/panel.parquet")
PANEL_URI = (
    "r2://financial-data-wrds-lakehouse/research/datasets/crsp_monthly/"
    "crsp-panel-1.2.2/918ff8c6a5c4e25e-9d684a69b938e435f85c068b16040cf3/panel.parquet"
)
PANEL_SHA256 = "395a3f5eef94e5e481c1cc5ffb00b9317b5010836fb887c34c08fc1ffb27e7d4"
PANEL_PIN = "9d684a69b938e435f85c068b16040cf3"
OUT_DIR = Path("output/qsd-ui-latest-orthogonal")

START = "2005-01-01"
END = "2024-12-31"
TRAIN_END = "2016-12-31"
VALIDATION_END = "2021-12-31"
TEST_START = "2022-01-01"
FORWARD_PERIODS = 1
PERIODS_PER_YEAR = 12
MIN_NAMES = 100
COST_BPS = 10.0


def load_rich_crsp_panels(path: Path) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    cols = [
        "date", "permno", "siccd", "ret_total", "retx",
        "prc_abs", "shrout", "cfacshr", "market_cap", "dollar_volume", "investable",
    ]
    df = pd.read_parquet(path, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= START) & (df["date"] <= END)].copy()
    df["permno"] = df["permno"].astype(str)

    # Filter to investable universe to avoid penny/microcap artifacts.
    if "investable" in df.columns:
        df = df[df["investable"].fillna(False).astype(bool)].copy()

    # Keep latest known 2-digit SIC per permno for industry neutralization.
    sic_series = (
        df.dropna(subset=["siccd"])
        .sort_values("date")
        .groupby("permno")["siccd"]
        .last()
    )
    sector_map = {k: str(int(v) // 100) for k, v in sic_series.items() if np.isfinite(v) and v > 0}

    # Build total-return price index from ret_total so forward returns include delisting & dividends.
    ret_total = df.pivot_table(index="date", columns="permno", values="ret_total", aggfunc="last").sort_index()
    retx = df.pivot_table(index="date", columns="permno", values="retx", aggfunc="last").sort_index()
    # Cumulative total-return index per security (starts at 1.0 on first valid observation).
    close = (1.0 + ret_total.clip(lower=-0.99)).cumprod()
    # Mask dates where ret_total was NaN so Close is NaN when stock wasn't trading.
    close = close.where(ret_total.notna())

    # Split-adjusted shares outstanding (shrout * cfacshr).
    df["adj_shares"] = df["shrout"] * df["cfacshr"].replace(0, np.nan)
    adj_shares = df.pivot_table(index="date", columns="permno", values="adj_shares", aggfunc="last").sort_index()
    mcap = df.pivot_table(index="date", columns="permno", values="market_cap", aggfunc="last").sort_index()
    dvol = df.pivot_table(index="date", columns="permno", values="dollar_volume", aggfunc="last").sort_index()

    # Implied monthly cash dividend return: (1 + ret_total) / (1 + retx) - 1
    div_ret = ((1.0 + ret_total) / (1.0 + retx.replace(-1.0, np.nan)) - 1.0).clip(lower=0.0, upper=0.5)

    panels = {
        "Close": close,
        "RetTotal": ret_total,
        "DivRet": div_ret,
        "AdjShares": adj_shares,
        "MarketCap": mcap,
        "Volume": dvol,
    }
    return panels, sector_map


def _sic_demean(sig: pd.DataFrame, sector_map: dict[str, str]) -> pd.DataFrame:
    return signal_evaluator._sector_demean_signal(sig, sector_map)


def _rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True)


def build_candidates(sector_map: dict[str, str]) -> list[dict[str, Any]]:
    def net_share_issuance_12m(p: dict[str, pd.DataFrame]) -> pd.DataFrame:
        sh = p["AdjShares"].replace(0, np.nan)
        growth = np.log(sh / sh.shift(12))
        return _rank(-growth)

    def issuance_accel_6v12(p: dict[str, pd.DataFrame]) -> pd.DataFrame:
        sh = p["AdjShares"].replace(0, np.nan)
        g_recent = np.log(sh / sh.shift(6))
        g_prior = np.log(sh.shift(6) / sh.shift(12))
        return _rank(-(g_recent - g_prior))

    def div_yield_12m(p: dict[str, pd.DataFrame]) -> pd.DataFrame:
        dy = p["DivRet"].rolling(12, min_periods=6).sum()
        return _rank(dy)

    def ind_rel_mom_12_1(p: dict[str, pd.DataFrame]) -> pd.DataFrame:
        mom = p["Close"].shift(1) / p["Close"].shift(12) - 1.0
        return _rank(_sic_demean(mom, sector_map))

    def high_52w_ind_rel(p: dict[str, pd.DataFrame]) -> pd.DataFrame:
        ratio = p["Close"] / p["Close"].rolling(12, min_periods=6).max()
        return _rank(_sic_demean(ratio, sector_map))

    def max_ret_lottery_12m(p: dict[str, pd.DataFrame]) -> pd.DataFrame:
        mx = p["RetTotal"].rolling(12, min_periods=6).max()
        return _rank(-_sic_demean(mx, sector_map))

    def same_month_seasonality_5y(p: dict[str, pd.DataFrame]) -> pd.DataFrame:
        r = p["RetTotal"]
        # Strictly causal: average of t-12, t-24, t-36, t-48, t-60 returns.
        lags = [r.shift(12 * k) for k in range(1, 6)]
        stacked = sum(lag.fillna(0.0) for lag in lags)
        counts = sum(lag.notna().astype(float) for lag in lags).replace(0.0, np.nan)
        return _rank(stacked / counts)

    return [
        {
            "signal_id": "qsd_net_share_issuance_12m",
            "signal_name": "net_share_issuance_12m",
            "formula": "cs_rank(-log(adj_shares / delay(adj_shares, 12)))",
            "family": "corporate_actions",
            "direction": 1.0,
            "reason": {
                "counterparty": "passive and index funds absorbing dilutive secondary equity issuance",
                "edge_type": "structural",
                "decay": "fades if corporate issuance timing loses information or buybacks crowd",
            },
            "fn": net_share_issuance_12m,
        },
        {
            "signal_id": "qsd_issuance_accel_6v12",
            "signal_name": "issuance_accel_6v12",
            "formula": "cs_rank(-(log(adj_shares / delay(adj_shares, 6)) - log(delay(adj_shares, 6) / delay(adj_shares, 12))))",
            "family": "corporate_actions",
            "direction": 1.0,
            "reason": {
                "counterparty": "slow investors missing inflection from share dilution to buybacks",
                "edge_type": "structural",
                "decay": "fades as buyback announcements are priced immediately",
            },
            "fn": issuance_accel_6v12,
        },
        {
            "signal_id": "qsd_div_yield_12m",
            "signal_name": "div_yield_12m",
            "formula": "cs_rank(ts_sum((1 + ret_total)/(1 + retx) - 1, 12))",
            "family": "cash_return",
            "direction": 1.0,
            "reason": {
                "counterparty": "growth-chasing investors under-weighting steady cash distributions",
                "edge_type": "risk_premium",
                "decay": "fades in low-rate growth regimes or value traps",
            },
            "fn": div_yield_12m,
        },
        {
            "signal_id": "qsd_ind_rel_mom_12_1",
            "signal_name": "ind_rel_mom_12_1",
            "formula": "cs_rank(sic_demean(delay(close, 1) / delay(close, 12) - 1))",
            "family": "industry_relative",
            "direction": 1.0,
            "reason": {
                "counterparty": "investors anchoring on sector moves while underreacting to firm-specific leaders",
                "edge_type": "behavioral",
                "decay": "fades during sharp momentum crashes and leadership reversals",
            },
            "fn": ind_rel_mom_12_1,
        },
        {
            "signal_id": "qsd_high_52w_ind_rel",
            "signal_name": "high_52w_ind_rel",
            "formula": "cs_rank(sic_demean(close / ts_max(close, 12)))",
            "family": "behavioral",
            "direction": 1.0,
            "reason": {
                "counterparty": "anchored investors selling winners near 52-week highs too early",
                "edge_type": "behavioral",
                "decay": "fades if anchoring bias weakens in systematic markets",
            },
            "fn": high_52w_ind_rel,
        },
        {
            "signal_id": "qsd_max_ret_lottery_12m",
            "signal_name": "max_ret_lottery_12m",
            "formula": "cs_rank(-sic_demean(ts_max(ret_total, 12)))",
            "family": "behavioral",
            "direction": 1.0,
            "reason": {
                "counterparty": "lottery-seeking retail buyers overpaying for extreme upside spikes",
                "edge_type": "behavioral",
                "decay": "fades when retail speculative demand subsides",
            },
            "fn": max_ret_lottery_12m,
        },
        {
            "signal_id": "qsd_same_month_seasonality_5y",
            "signal_name": "same_month_seasonality_5y",
            "formula": "cs_rank(mean(delay(ret_total, 12..60 step 12)))",
            "family": "seasonality",
            "direction": 1.0,
            "reason": {
                "counterparty": "recurring calendar-driven institutional and earnings-cycle flows",
                "edge_type": "structural",
                "decay": "fades if calendar rebalancing schedules shift or are arbitraged",
            },
            "fn": same_month_seasonality_5y,
        },
    ]


def cutoff_status(validation: dict[str, Any], test: dict[str, Any]) -> str:
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
    digest = hashlib.sha256(PANEL_PATH.read_bytes()).hexdigest()
    if digest != PANEL_SHA256:
        raise SystemExit(f"panel sha256 mismatch: {digest} != {PANEL_SHA256}")

    panels, sector_map = load_rich_crsp_panels(PANEL_PATH)
    close = panels["Close"]
    forward = panels["RetTotal"].shift(-FORWARD_PERIODS)
    canonical = governance.build_canonical_factor_panels(panels, periods_per_year=PERIODS_PER_YEAR)
    family = build_candidates(sector_map)

    evaluated: list[dict[str, Any]] = []
    for cand in family:
        sig = cand["fn"](panels)
        ic = signal_evaluator.compute_ic_split(
            sig,
            forward,
            train_end_date=TRAIN_END,
            validation_end_date=VALIDATION_END,
            test_start_date=TEST_START,
            purge_periods=FORWARD_PERIODS,
            periods=FORWARD_PERIODS,
            min_names=MIN_NAMES,
            periods_per_year=PERIODS_PER_YEAR,
            sector_map=sector_map,
            include_raw=False,
        )
        screen = signal_evaluator.compute_long_short_screen(
            sig.loc[TRAIN_END:VALIDATION_END],
            close.loc[TRAIN_END:VALIDATION_END],
            quantile=0.2,
            cost_bps=COST_BPS,
            min_names=MIN_NAMES,
            annualization=PERIODS_PER_YEAR,
            direction=cand["direction"],
            sector_map=sector_map,
            include_series=True,
        )
        if "error" in screen:
            raise RuntimeError(f"{cand['signal_name']}: {screen['error']}")
        net = screen.pop("_net_series")
        screen["deflated_sharpe"] = overfitting.deflated_sharpe_ratio(
            net, n_trials=len(family), periods_per_year=PERIODS_PER_YEAR
        )
        screen["subperiod_stability"] = overfitting.subperiod_stability(
            net, n_blocks=4, periods_per_year=PERIODS_PER_YEAR
        )
        evaluated.append({
            "candidate": cand,
            "ic": ic,
            "screen": screen,
            "net": net,
            "novelty": governance.check_factor_novelty(sig, known_signal_panels=canonical),
            "pit": governance.check_truncation_invariance(cand["fn"], panels),
        })

    net_matrix = pd.concat([row["net"] for row in evaluated], axis=1).dropna()
    pbo = governance.probability_of_backtest_overfitting(net_matrix, n_blocks=8)

    entries = []
    for row in evaluated:
        cand = row["candidate"]
        val_ic = row["ic"]["ic_validation"]
        test_ic = row["ic"]["ic_test"]
        entries.append(
            governance.build_survivor_ledger_entry(
                signal_id=cand["signal_id"],
                signal_name=cand["signal_name"],
                formula=cand["formula"],
                family=cand["family"],
                hold_periods=FORWARD_PERIODS,
                reason=cand["reason"],
                ic_results={"ic_validation": val_ic, "ic_test": test_ic, "ic_full": row["ic"]["ic_full"]},
                screen_validation=row["screen"],
                pit_check=row["pit"],
                novelty_check=row["novelty"],
                pbo_check=pbo,
                post_cutoff_status=cutoff_status(val_ic, test_ic),
                pin_hash=PANEL_PIN,
            )
        )

    selftest = governance.run_selftest(seed=42)
    selftest_summary = {k: selftest[k] for k in ("ok", "passed", "total", "ran_at")}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    provenance = {
        "run_id": f"qsd-crsp-orthogonal-2005-2024-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "dataset": (
            f"CRSP monthly panel 1.2.2, {close.index.min().date()}..{close.index.max().date()}, "
            f"{close.shape[1]:,} investable securities, {len(close)} months"
        ),
        "source": PANEL_URI,
        "panel_sha256": PANEL_SHA256,
        "panel_pin_hash": PANEL_PIN,
        "command": "PYTHONPATH=src .venv/bin/python scripts/run_crsp_orthogonal_governance.py",
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
            "Pre-specified 7-signal orthogonal corporate-action / cash-return / "
            "industry-relative / behavioral / seasonal family selected on validation only; "
            "2022-2024 is reported as a post-cutoff audit. Not an LLM discovery loop."
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = phf_export.export_qsd_ui_latest(
        str(OUT_DIR),
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
        "out_path": out_path,
        "panel_months": len(close),
        "panel_securities": close.shape[1],
        "pbo": pbo.get("pbo"),
        "selftest": selftest_summary,
        "entries": [
            {
                "signal": e["signal_name"],
                "family": e["family"],
                "gate": e["gate_status"],
                "checks": f"{e['checks_passed']}/{e['checks_total']}",
                "val_ic": e["metrics"]["validation_ic"],
                "val_t": e["metrics"]["validation_hac_t"],
                "test_ic": e["metrics"]["test_ic"],
                "net_sharpe": e["metrics"]["validation_net_sharpe"],
                "breakeven_bps": e["metrics"]["breakeven_cost_bps"],
                "dsr": e["metrics"]["dsr"],
                "closest_factor": e["metrics"]["closest_factor"],
                "closest_corr": e["metrics"]["closest_factor_corr"],
                "post_cutoff": e["post_cutoff_status"],
                "failed": [c["id"] for c in e["checks"] if not c["ok"]],
            }
            for e in entries
        ],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
