# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0

"""
Governance checks, point-in-time truncation verification, novelty gating,
CSCV Probability of Backtest Overfitting (PBO), survivor ledger builder,
and deterministic ``selftest`` exam for QSD.

Designed to validate and freeze the existing QSD evaluator without rewriting
the discovery loop:
  1. Check 2 (Point-in-Time Truncation Shift Test): recomputes any signal on
     data truncated at interior dates ``t`` and requires exact agreement for
     all rows strictly before ``t``. Catches centered windows, negative shifts,
     and full-sample scaling that statistical gates (DSR/PBO) cannot detect.
  2. Check 5 (Breakeven Cost): computes one-way bps that drives net return to 0.
  3. Check 7 (Factor Novelty): checks cross-sectional rank correlation and
     return correlation against canonical factors (12-1 momentum, 1m reversal,
     low-vol, size) plus any supplied benchmark library.
  4. Check 10 (CSCV PBO): Bailey et al. (2015) Combinatorial Symmetric
     Cross-Validation over contiguous blocks for multi-variant families.
  5. ``run_selftest()``: self-grading calibration exam (noise deflation,
     planted signal recovery, leaky oracle kill, centered rolling kill,
     causal invariance, disguised momentum novelty flag, noise-family PBO,
     and deterministic freeze hash).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import overfitting, signal_evaluator


# ---------------------------------------------------------------------------
# Check 2: Point-in-Time Truncation Shift Test
# ---------------------------------------------------------------------------

def _slice_stock_data(stock_data: dict[str, pd.DataFrame], end_pos: int) -> dict[str, pd.DataFrame]:
    """Slice every DataFrame panel to its first ``end_pos`` rows."""
    return {k: df.iloc[:end_pos].copy() for k, df in stock_data.items()}


def check_truncation_invariance(
    signal_fn: Callable[[dict[str, pd.DataFrame]], pd.DataFrame],
    stock_data: dict[str, pd.DataFrame],
    *,
    cut_fractions: tuple[float, ...] = (0.45, 0.70, 0.85),
    atol: float = 1e-10,
) -> dict[str, Any]:
    """Verify a signal is strictly causal via day-``t`` truncation tests.

    Recomputes ``signal_fn`` on prefix slices of ``stock_data`` and compares
    every row ``< cut`` against the full-sample output. Any discrepancy means
    the formula or loader peeks into future rows.
    """
    close = stock_data.get("Close")
    if close is None or close.empty or len(close) < 20:
        return {
            "ok": False,
            "reason": "Close panel missing or too short (< 20 rows) for truncation test",
            "max_abs_diff": None,
            "nan_mismatches": None,
            "cuts_tested": [],
        }

    full_sig = signal_fn(stock_data)
    if not isinstance(full_sig, pd.DataFrame) or full_sig.empty:
        return {
            "ok": False,
            "reason": "signal_fn returned empty or non-DataFrame output",
            "max_abs_diff": None,
            "nan_mismatches": None,
            "cuts_tested": [],
        }

    n_rows = len(close)
    cut_positions = sorted({
        max(10, min(n_rows - 2, int(round(n_rows * frac))))
        for frac in cut_fractions
    })

    max_diff = 0.0
    total_nan_mismatches = 0
    failed_cuts: list[dict[str, Any]] = []

    for cut in cut_positions:
        truncated_data = _slice_stock_data(stock_data, cut)
        trunc_sig = signal_fn(truncated_data)

        full_prefix = full_sig.iloc[:cut].reindex(index=trunc_sig.index, columns=trunc_sig.columns)
        a = full_prefix.to_numpy(dtype=float, copy=False)
        b = trunc_sig.to_numpy(dtype=float, copy=False)

        a_fin = np.isfinite(a)
        b_fin = np.isfinite(b)
        nan_mismatch = int(np.sum(a_fin != b_fin))
        both_fin = a_fin & b_fin
        cut_max_diff = float(np.max(np.abs(a[both_fin] - b[both_fin]))) if np.any(both_fin) else 0.0

        max_diff = max(max_diff, cut_max_diff)
        total_nan_mismatches += nan_mismatch
        if nan_mismatch > 0 or cut_max_diff > atol:
            failed_cuts.append({
                "cut_pos": cut,
                "cut_date": str(close.index[cut - 1].date()) if hasattr(close.index[cut - 1], "date") else str(close.index[cut - 1]),
                "max_abs_diff": cut_max_diff,
                "nan_mismatches": nan_mismatch,
            })

    ok = len(failed_cuts) == 0
    return {
        "ok": ok,
        "max_abs_diff": max_diff,
        "nan_mismatches": total_nan_mismatches,
        "cuts_tested": cut_positions,
        "failed_cuts": failed_cuts,
        "detail": (
            f"bit-identical across {len(cut_positions)} truncation cuts (max_diff={max_diff:.2e})"
            if ok
            else f"look-ahead leak detected at {len(failed_cuts)}/{len(cut_positions)} cuts "
            f"(max_diff={max_diff:.3e}, nan_mismatches={total_nan_mismatches})"
        ),
    }


def check_code_truncation_invariance(
    code: str,
    stock_data: dict[str, pd.DataFrame],
    *,
    signal_name: str | None = None,
    cut_fractions: tuple[float, ...] = (0.45, 0.70, 0.85),
    atol: float = 1e-10,
) -> dict[str, Any]:
    """Run ``check_truncation_invariance`` on generated Python signal code."""
    def _runner(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        funcs = signal_evaluator.execute_signal_functions(code, data)
        if not funcs:
            return pd.DataFrame()
        if signal_name is not None:
            for name, df in funcs:
                if name == signal_name:
                    return df
        return funcs[0][1]

    return check_truncation_invariance(_runner, stock_data, cut_fractions=cut_fractions, atol=atol)


# ---------------------------------------------------------------------------
# Check 5: Breakeven Cost (bps per side)
# ---------------------------------------------------------------------------

def compute_breakeven_cost_bps(screen: dict[str, Any]) -> float | None:
    """Compute one-way transaction cost in bps that drives net return to zero.

    Uses ``gross.annual_return``, ``annual_turnover``, and any non-spread drags
    (``avg_impact_cost``, ``avg_borrow_cost``) from ``compute_long_short_screen``.
    """
    if not isinstance(screen, dict) or screen.get("error"):
        return None
    gross = screen.get("gross") or {}
    ann_ret = gross.get("annual_return")
    ann_to = screen.get("annual_turnover")
    if ann_ret is None or ann_to is None or not np.isfinite(ann_ret) or not np.isfinite(ann_to):
        return None
    if ann_to <= 1e-9:
        return None

    ppy = float(screen.get("periods_per_year") or 252)
    other_drag = (
        float(screen.get("avg_impact_cost") or 0.0)
        + float(screen.get("avg_borrow_cost") or 0.0)
    ) * ppy
    net_before_spread = float(ann_ret) - other_drag
    return float((net_before_spread / float(ann_to)) * 10_000.0)


# ---------------------------------------------------------------------------
# Check 7: Factor Novelty Gate
# ---------------------------------------------------------------------------

def build_canonical_factor_panels(
    stock_data: dict[str, pd.DataFrame],
    *,
    periods_per_year: int = 252,
) -> dict[str, pd.DataFrame]:
    """Construct canonical baseline factor panels for novelty screening.

    Includes 12-1 momentum, short-term reversal, and low-volatility directly
    from ``Close`` (plus ``Size`` when ``MarketCap`` is present).
    """
    close = stock_data.get("Close")
    if close is None or close.empty:
        return {}

    factors: dict[str, pd.DataFrame] = {}
    if periods_per_year >= 200:
        mom_long, mom_skip, rev_win, vol_win = 252, 21, 21, 63
    elif periods_per_year >= 40:
        mom_long, mom_skip, rev_win, vol_win = 52, 4, 4, 13
    else:
        mom_long, mom_skip, rev_win, vol_win = 12, 1, 1, 6

    if len(close) > mom_long + 2:
        factors["momentum_12_1"] = close.shift(mom_skip) / close.shift(mom_long) - 1.0
    if len(close) > rev_win + 2:
        factors["short_reversal"] = -(close / close.shift(rev_win) - 1.0)
    if len(close) > vol_win + 2:
        factors["low_volatility"] = -close.pct_change().rolling(vol_win, min_periods=max(5, vol_win // 2)).std()
    if "MarketCap" in stock_data and not stock_data["MarketCap"].empty:
        factors["size"] = -np.log(stock_data["MarketCap"].clip(lower=1.0))

    return factors


def _mean_cross_sectional_rank_corr(a: pd.DataFrame, b: pd.DataFrame, min_names: int = 10) -> float | None:
    """Mean per-date Spearman rank correlation between two signal panels."""
    dates = a.index.intersection(b.index)
    cols = a.columns.intersection(b.columns)
    if len(dates) == 0 or len(cols) < min_names:
        return None
    sa = a.loc[dates, cols].replace([np.inf, -np.inf], np.nan)
    sb = b.loc[dates, cols].replace([np.inf, -np.inf], np.nan)
    valid = sa.notna() & sb.notna()
    counts = valid.sum(axis=1)
    ra = sa.where(valid).rank(axis=1)
    rb = sb.where(valid).rank(axis=1)
    ra = ra.sub(ra.mean(axis=1), axis=0)
    rb = rb.sub(rb.mean(axis=1), axis=0)
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra.pow(2).sum(axis=1)) * (rb.pow(2).sum(axis=1)))
    corr = (num / den).where((counts >= min_names) & (den > 1e-12)).dropna()
    return float(corr.mean()) if len(corr) > 0 else None


def check_factor_novelty(
    signal_values: pd.DataFrame,
    *,
    known_signal_panels: dict[str, pd.DataFrame] | None = None,
    signal_returns: pd.Series | None = None,
    known_factor_returns: dict[str, pd.Series] | None = None,
    rank_corr_threshold: float = 0.70,
    return_corr_threshold: float = 0.70,
) -> dict[str, Any]:
    """Check whether a candidate signal is merely a disguised known factor."""
    rank_corrs: dict[str, float] = {}
    for name, panel in (known_signal_panels or {}).items():
        corr = _mean_cross_sectional_rank_corr(signal_values, panel)
        if corr is not None and np.isfinite(corr):
            rank_corrs[name] = corr

    ret_corrs: dict[str, float] = {}
    if signal_returns is not None and len(signal_returns) >= 12:
        for name, series in (known_factor_returns or {}).items():
            aligned = pd.concat([signal_returns, series], axis=1).dropna()
            if len(aligned) >= 12:
                c = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
                if np.isfinite(c):
                    ret_corrs[name] = c

    max_rank_name, max_rank_val = (
        max(rank_corrs.items(), key=lambda kv: abs(kv[1])) if rank_corrs else (None, 0.0)
    )
    max_ret_name, max_ret_val = (
        max(ret_corrs.items(), key=lambda kv: abs(kv[1])) if ret_corrs else (None, 0.0)
    )

    rank_hit = abs(max_rank_val) >= rank_corr_threshold
    ret_hit = abs(max_ret_val) >= return_corr_threshold
    novel = not (rank_hit or ret_hit)

    closest = max_rank_name if abs(max_rank_val) >= abs(max_ret_val) else max_ret_name
    closest_corr = max_rank_val if abs(max_rank_val) >= abs(max_ret_val) else max_ret_val

    return {
        "ok": novel,
        "novel": novel,
        "closest_factor": closest,
        "closest_corr": float(closest_corr) if closest is not None else None,
        "max_abs_rank_corr": abs(float(max_rank_val)) if max_rank_name is not None else None,
        "max_abs_return_corr": abs(float(max_ret_val)) if max_ret_name is not None else None,
        "rank_correlations": rank_corrs,
        "return_correlations": ret_corrs,
        "detail": (
            f"novel (max |corr|={abs(closest_corr):.2f} vs {closest})"
            if novel and closest is not None
            else (
                "novel (no benchmark factors supplied)"
                if novel
                else f"crowded: |corr|={abs(closest_corr):.2f} against {closest}"
            )
        ),
    }


# ---------------------------------------------------------------------------
# Check 10: Combinatorial Symmetric Cross-Validation (CSCV) PBO
# ---------------------------------------------------------------------------

def probability_of_backtest_overfitting(
    variant_returns: pd.DataFrame | np.ndarray,
    *,
    n_blocks: int = 8,
    pbo_threshold: float = 0.50,
) -> dict[str, Any]:
    """Compute CSCV Probability of Backtest Overfitting (Bailey et al. 2015).

    ``variant_returns`` has shape ``(T, M)`` where each column is a strategy
    variant's per-period return series. Splits ``T`` into ``S`` even contiguous
    blocks (``n_blocks``), evaluates all ``C(S, S/2)`` IS/OOS partitions, and
    returns the fraction of partitions where the IS-best Sharpe variant ranks
    below the OOS median.
    """
    mat = np.asarray(variant_returns, dtype=float)
    if mat.ndim != 2 or mat.shape[1] < 2:
        return {"ok": None, "pbo": None, "error": "need at least 2 strategy variants"}

    # Drop rows with any non-finite value across variants.
    finite_rows = np.all(np.isfinite(mat), axis=1)
    mat = mat[finite_rows]
    t, m = mat.shape

    s = int(n_blocks)
    if s % 2 == 1:
        s -= 1
    s = max(4, min(s, 16))
    while s >= 4 and (t // s) < 4:
        s -= 2
    if s < 4 or (t // s) < 4:
        return {"ok": None, "pbo": None, "error": f"too few observations ({t}) for CSCV"}

    usable_t = (t // s) * s
    blocks = mat[:usable_t].reshape(s, usable_t // s, m)
    half = s // 2
    all_idx = set(range(s))

    def _sharpes(subset_indices: tuple[int, ...]) -> np.ndarray:
        sub = blocks[list(subset_indices)].reshape(-1, m)
        mean = sub.mean(axis=0)
        std = sub.std(axis=0, ddof=1)
        return np.divide(mean, std, out=np.zeros(m, dtype=float), where=std > 1e-12)

    below_median = 0
    total_splits = 0
    logits: list[float] = []

    for is_idx in itertools.combinations(range(s), half):
        oos_idx = tuple(sorted(all_idx - set(is_idx)))
        is_sr = _sharpes(is_idx)
        oos_sr = _sharpes(oos_idx)

        winner = int(np.argmax(is_sr))
        # Relative rank in (0, 1): 1.0 = best OOS, 0.0 = worst OOS.
        rank = float(np.sum(oos_sr < oos_sr[winner]) + 0.5 * np.sum(oos_sr == oos_sr[winner])) / float(m)
        if rank < 0.5:
            below_median += 1
        clipped = min(1.0 - 1e-6, max(1e-6, rank))
        logits.append(math.log(clipped / (1.0 - clipped)))
        total_splits += 1

    pbo = float(below_median / total_splits) if total_splits > 0 else None
    return {
        "ok": (pbo < pbo_threshold) if pbo is not None else None,
        "pbo": pbo,
        "n_blocks": s,
        "n_splits": total_splits,
        "n_variants": m,
        "median_logit": float(np.median(logits)) if logits else None,
        "threshold": pbo_threshold,
        "detail": (
            f"PBO={pbo:.3f} across {total_splits} CSCV splits ({m} variants, threshold < {pbo_threshold:.2f})"
            if pbo is not None
            else "PBO unavailable"
        ),
    }


# ---------------------------------------------------------------------------
# Survivor Ledger & Compact UI Contract (for terminal-v2 Quant Lab)
# ---------------------------------------------------------------------------

def compute_freeze_hash(payload: dict[str, Any]) -> str:
    """Deterministic SHA-256 hash of a survivor's identity, formula, and checks."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_survivor_ledger_entry(
    *,
    signal_id: str,
    signal_name: str,
    formula: str | None,
    family: str = "unspecified",
    hold_periods: int = 5,
    reason: dict[str, str] | None = None,
    ic_results: dict[str, Any] | None = None,
    screen_validation: dict[str, Any] | None = None,
    pit_check: dict[str, Any] | None = None,
    novelty_check: dict[str, Any] | None = None,
    pbo_check: dict[str, Any] | None = None,
    post_cutoff_status: str = "UNVERIFIED",
    pin_hash: str | None = None,
    dsr_threshold: float = 0.95,
    ic_threshold: float = 0.02,
    t_threshold: float = 3.0,
) -> dict[str, Any]:
    """Assemble a 1-page survivor ledger record matching ``alphadesk``'s gate model."""
    ic_val = (ic_results or {}).get("ic_validation") or (ic_results or {})
    ic_test = (ic_results or {}).get("ic_test") or {}
    val_net = (screen_validation or {}).get("net") or {}
    val_dsr = (screen_validation or {}).get("deflated_sharpe") or {}
    val_stab = (screen_validation or {}).get("subperiod_stability") or {}

    mean_ic = ic_val.get("mean_ic")
    hac_t = ic_val.get("t_stat")
    net_sharpe = val_net.get("sharpe")
    dsr_val = val_dsr.get("dsr")
    pos_blocks = val_stab.get("positive_block_ratio")
    breakeven_bps = compute_breakeven_cost_bps(screen_validation or {})

    clean_reason = {
        "counterparty": str((reason or {}).get("counterparty") or "").strip(),
        "edge_type": str((reason or {}).get("edge_type") or "").strip(),
        "decay": str((reason or {}).get("decay") or "").strip(),
    }
    reason_ok = bool(clean_reason["counterparty"] and clean_reason["edge_type"] and clean_reason["decay"])
    pit_ok = bool((pit_check or {}).get("ok"))
    ic_ok = (
        mean_ic is not None
        and hac_t is not None
        and abs(float(mean_ic)) >= ic_threshold
        and abs(float(hac_t)) >= t_threshold
    )
    ic_leak_flag = mean_ic is not None and abs(float(mean_ic)) > 0.10
    cost_ok = net_sharpe is not None and float(net_sharpe) > 0.0
    novelty_ok = bool((novelty_check or {}).get("ok", True))
    subperiod_ok = pos_blocks is not None and float(pos_blocks) >= 0.75
    dsr_ok = dsr_val is not None and float(dsr_val) >= dsr_threshold
    pbo_ok = True if pbo_check is None or pbo_check.get("ok") is None else bool(pbo_check.get("ok"))

    checks = [
        {
            "id": "reason",
            "name": "0. Economic reason",
            "ok": reason_ok,
            "detail": (
                f"{clean_reason['edge_type']} vs {clean_reason['counterparty']}"
                if reason_ok
                else "missing counterparty, edge_type, or decay mechanism"
            ),
        },
        {
            "id": "point_in_time",
            "name": "2. Point-in-time shift test",
            "ok": pit_ok,
            "detail": (pit_check or {}).get("detail") or "truncation check not run",
        },
        {
            "id": "rank_ic",
            "name": "3. Rank IC & HAC t-stat",
            "ok": ic_ok and not ic_leak_flag,
            "detail": (
                f"IC={mean_ic:.4f}, HAC t={hac_t:.2f}"
                + (" [FLAG: |IC| > 0.10 leak audit]" if ic_leak_flag else "")
                if mean_ic is not None and hac_t is not None
                else "IC unavailable"
            ),
        },
        {
            "id": "costs",
            "name": "5. Net of costs",
            "ok": cost_ok,
            "detail": (
                f"net Sharpe={net_sharpe:.2f}"
                + (f", breakeven={breakeven_bps:.1f} bps" if breakeven_bps is not None else "")
                if net_sharpe is not None
                else "cost screen unavailable"
            ),
        },
        {
            "id": "novelty",
            "name": "7. Factor novelty",
            "ok": novelty_ok,
            "detail": (novelty_check or {}).get("detail") or "novelty check not run",
        },
        {
            "id": "subperiods",
            "name": "8. Subperiod stability",
            "ok": subperiod_ok,
            "detail": (
                f"{pos_blocks * 100:.0f}% positive blocks"
                if pos_blocks is not None
                else "subperiod stability unavailable"
            ),
        },
        {
            "id": "deflated_sharpe",
            "name": "9. Deflated Sharpe (DSR)",
            "ok": dsr_ok,
            "detail": (
                f"DSR={dsr_val:.3f} (N={val_dsr.get('n_trials')})"
                if dsr_val is not None
                else "DSR unavailable"
            ),
        },
        {
            "id": "pbo",
            "name": "10. Backtest overfitting (PBO)",
            "ok": pbo_ok,
            "detail": (pbo_check or {}).get("detail") or "single candidate (CSCV N/A)",
        },
    ]

    passed_count = sum(1 for c in checks if c["ok"])
    all_passed = passed_count == len(checks)
    if all_passed and post_cutoff_status in ("PASS", "WEAK"):
        gate_status = "PROMOTE"
    elif pit_ok and ic_ok and cost_ok and novelty_ok and not ic_leak_flag:
        gate_status = "WATCH"
    else:
        gate_status = "REJECT"

    freeze_material = {
        "signal_id": signal_id,
        "signal_name": signal_name,
        "formula": formula,
        "pin_hash": pin_hash,
        "checks": [(c["id"], c["ok"]) for c in checks],
    }

    return {
        "signal_id": signal_id,
        "signal_name": signal_name,
        "formula": formula,
        "family": family,
        "hold_periods": hold_periods,
        "reason": clean_reason,
        "gate_status": gate_status,
        "post_cutoff_status": post_cutoff_status,
        "checks_passed": passed_count,
        "checks_total": len(checks),
        "checks": checks,
        "metrics": {
            "validation_ic": mean_ic,
            "validation_hac_t": hac_t,
            "test_ic": ic_test.get("mean_ic"),
            "validation_net_sharpe": net_sharpe,
            "breakeven_cost_bps": breakeven_bps,
            "dsr": dsr_val,
            "n_trials": val_dsr.get("n_trials"),
            "pbo": (pbo_check or {}).get("pbo"),
            "subperiod_positive_ratio": pos_blocks,
            "closest_factor": (novelty_check or {}).get("closest_factor"),
            "closest_factor_corr": (novelty_check or {}).get("closest_corr"),
        },
        "freeze_sha256": compute_freeze_hash(freeze_material),
    }


def build_ui_latest_payload(
    survivors: list[dict[str, Any]],
    *,
    n_trials_total: int,
    selftest_summary: dict[str, Any] | None = None,
    data_pins: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build compact ``ui_latest.json`` for ``terminal-v2`` Quant Lab.

    ``provenance`` records the executed run that produced the ledger (run id,
    dataset/source, command, date, and whether the data is production-grade).
    ``generated_at`` is overridable so a fixture can be regenerated byte for
    byte; production callers should leave it unset.
    """
    promoted = [s for s in survivors if s.get("gate_status") == "PROMOTE"]
    watching = [s for s in survivors if s.get("gate_status") == "WATCH"]
    return {
        "schema_version": "qsd-survivor-ledger/1.0.0",
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decision_use": "RESEARCH_ONLY",
        "live_orders_allowed": False,
        "n_trials_total": int(n_trials_total),
        "promoted_count": len(promoted),
        "watch_count": len(watching),
        "survivors_count": len(survivors),
        "selftest": selftest_summary,
        "pins": data_pins or {},
        "provenance": provenance or {},
        "survivors": survivors,
    }


# ---------------------------------------------------------------------------
# Part 7: The Self-Grading Calibration Exam (``selftest``)
# ---------------------------------------------------------------------------

def _make_synthetic_panel(
    *,
    n_days: int = 380,
    n_stocks: int = 80,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    tickers = [f"S{i:03d}" for i in range(n_stocks)]
    rets = rng.normal(0.0003, 0.018, size=(n_days, n_stocks))
    close = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=tickers)
    volume = pd.DataFrame(rng.lognormal(14.0, 0.5, size=(n_days, n_stocks)), index=dates, columns=tickers)
    mcap = pd.DataFrame(rng.lognormal(22.0, 1.0, size=(n_days, n_stocks)), index=dates, columns=tickers)
    return {"Close": close, "Volume": volume, "MarketCap": mcap}


def run_selftest(*, seed: int = 42) -> dict[str, Any]:
    """Run the 8-check calibration exam on QSD's evaluator and governance gates."""
    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []

    # 1. 1,000 random noise strategies on 1 year: raw max Sharpe is high, 0 survive DSR >= 0.95
    noise_rets = rng.normal(0.0, 0.01, size=(252, 1000))
    means = noise_rets.mean(axis=0)
    stds = noise_rets.std(axis=0, ddof=1)
    ann_sharpes = (means / stds) * math.sqrt(252)
    best_idx = int(np.argmax(ann_sharpes))
    best_raw_sharpe = float(ann_sharpes[best_idx])
    per_period_srs = means / stds
    sr_var = float(np.var(per_period_srs, ddof=1))
    survivors_dsr = 0
    for j in range(1000):
        res = overfitting.deflated_sharpe_ratio(
            noise_rets[:, j], n_trials=1000, periods_per_year=252, sr_variance=sr_var
        )
        if res.get("dsr") is not None and res["dsr"] >= 0.95:
            survivors_dsr += 1
    results.append({
        "id": "noise_1000_deflation",
        "ok": best_raw_sharpe >= 2.0 and survivors_dsr == 0,
        "detail": f"best_raw_sharpe={best_raw_sharpe:.2f}, dsr_survivors={survivors_dsr}/1000",
    })

    # 2. Planted signal with daily IC ~ 0.04 on 380 days x 80 stocks: recovered cleanly
    panel = _make_synthetic_panel(n_days=380, n_stocks=80, seed=seed)
    fwd1 = signal_evaluator.compute_forward_returns(panel["Close"], periods=1)
    planted = 0.045 * (fwd1.sub(fwd1.mean(axis=1), axis=0).div(fwd1.std(axis=1), axis=0)) + pd.DataFrame(
        rng.normal(0.0, 1.0, size=fwd1.shape), index=fwd1.index, columns=fwd1.columns
    )
    ic_planted = signal_evaluator.compute_rank_ic(planted, fwd1, periods=1, periods_per_year=252)
    results.append({
        "id": "planted_signal_detected",
        "ok": (
            ic_planted.get("mean_ic") is not None
            and ic_planted["mean_ic"] >= 0.02
            and ic_planted.get("t_stat") is not None
            and ic_planted["t_stat"] >= 3.0
        ),
        "detail": f"mean_ic={ic_planted.get('mean_ic'):.4f}, t_stat={ic_planted.get('t_stat'):.2f}",
    })

    # 3. Leaky oracle (reads tomorrow's return via shift(-1)): killed at Check 2
    leaky_check = check_truncation_invariance(
        lambda d: d["Close"].shift(-1) / d["Close"] - 1.0,
        panel,
    )
    results.append({
        "id": "leaky_oracle_killed_at_pit",
        "ok": not leaky_check["ok"],
        "detail": leaky_check["detail"],
    })

    # 4. Centered rolling mean (reaches ahead): caught at Check 2
    centered_check = check_truncation_invariance(
        lambda d: d["Close"].rolling(20, center=True).mean(),
        panel,
    )
    results.append({
        "id": "centered_rolling_caught",
        "ok": not centered_check["ok"],
        "detail": centered_check["detail"],
    })

    # 5. Strictly causal signal (20d lagged return rank): bit-identical before cutoff
    causal_check = check_truncation_invariance(
        lambda d: (d["Close"] / d["Close"].shift(20) - 1.0).rank(axis=1, pct=True),
        panel,
    )
    results.append({
        "id": "causal_operators_bit_identical",
        "ok": causal_check["ok"] and causal_check["max_abs_diff"] == 0.0,
        "detail": causal_check["detail"],
    })

    # 6. Disguised 12-1 momentum (rescaled + shifted): flagged as non-novel at Check 7
    canonical = build_canonical_factor_panels(panel, periods_per_year=252)
    disguised_mom = (panel["Close"].shift(21) / panel["Close"].shift(252) - 1.0) * 3.7 + 0.15
    novelty_res = check_factor_novelty(disguised_mom, known_signal_panels=canonical)
    results.append({
        "id": "disguised_momentum_flagged",
        "ok": (not novelty_res["ok"]) and novelty_res["closest_factor"] == "momentum_12_1",
        "detail": novelty_res["detail"],
    })

    # 7. 20 noise variants tuned on 1 dataset: high PBO
    noise_variants = rng.normal(0.0, 0.01, size=(256, 20))
    pbo_res = probability_of_backtest_overfitting(noise_variants, n_blocks=8)
    results.append({
        "id": "noise_family_high_pbo",
        "ok": pbo_res.get("pbo") is not None and pbo_res["pbo"] >= 0.35,
        "detail": pbo_res.get("detail") or str(pbo_res),
    })

    # 8. Deterministic verdict & freeze hash across 2 runs
    entry_a = build_survivor_ledger_entry(
        signal_id="selftest_s1",
        signal_name="causal_mom",
        formula="cs_rank(close / delay(close, 20) - 1)",
        reason={"counterparty": "slow institutions", "edge_type": "behavioral", "decay": "crowding"},
        ic_results={"ic_validation": ic_planted, "ic_test": ic_planted},
        pit_check=causal_check,
        novelty_check={"ok": True, "detail": "novel"},
        pin_hash="pin-selftest",
    )
    entry_b = build_survivor_ledger_entry(
        signal_id="selftest_s1",
        signal_name="causal_mom",
        formula="cs_rank(close / delay(close, 20) - 1)",
        reason={"counterparty": "slow institutions", "edge_type": "behavioral", "decay": "crowding"},
        ic_results={"ic_validation": ic_planted, "ic_test": ic_planted},
        pit_check=causal_check,
        novelty_check={"ok": True, "detail": "novel"},
        pin_hash="pin-selftest",
    )
    results.append({
        "id": "deterministic_verdict",
        "ok": entry_a["freeze_sha256"] == entry_b["freeze_sha256"] and entry_a["gate_status"] == entry_b["gate_status"],
        "detail": f"freeze_sha256={entry_a['freeze_sha256'][:12]}…",
    })

    passed = sum(1 for r in results if r["ok"])
    return {
        "ok": passed == len(results),
        "passed": passed,
        "total": len(results),
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checks": results,
    }


if __name__ == "__main__":  # pragma: no cover
    import sys

    report = run_selftest()
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["ok"] else 1)
