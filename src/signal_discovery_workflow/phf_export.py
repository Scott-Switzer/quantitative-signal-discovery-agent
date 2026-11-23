# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0

"""
Export accepted QSD signals in the ``personal-hedge-fund`` research contract.

``phf-private-research-viewer`` and the fund's portfolio engine read
``research/runs/<run_id>/receipt.json`` (schema ``phf-research-receipt/1.0.0``)
and ``results.json`` from ``financial-data-wrds-lakehouse``. Writing accepted
discovery signals in that exact envelope keeps QSD outputs compatible with the
rest of the fund without touching ``research/latest.json`` automatically.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from . import r2_store

logger = logging.getLogger(__name__)


def build_phf_run_payload(
    *,
    run_id: str,
    signal_name: str,
    formula: str | None,
    request_text: str,
    ic_results: dict[str, Any],
    screen_validation: dict[str, Any],
    screen_full: dict[str, Any],
    factor_alpha: dict[str, Any] | None,
    data_pins: dict[str, Any],
    config_summary: dict[str, Any],
    signal_artifact_uri: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build ``(receipt, results)`` dicts matching ``phf-research-receipt/1.0.0``."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    val_net = (screen_validation.get("net") or {}) if isinstance(screen_validation, dict) else {}
    full_net = (screen_full.get("net") or {}) if isinstance(screen_full, dict) else {}
    val_ic = (ic_results.get("ic_validation") or {}) if isinstance(ic_results, dict) else {}
    test_ic = (ic_results.get("ic_test") or {}) if isinstance(ic_results, dict) else {}
    train_ic = (ic_results.get("ic_train") or {}) if isinstance(ic_results, dict) else {}

    val_dsr = (screen_validation.get("deflated_sharpe") or {}) if isinstance(screen_validation, dict) else {}
    val_stab = (screen_validation.get("subperiod_stability") or {}) if isinstance(screen_validation, dict) else {}

    receipt = {
        "schema": "phf-research-receipt/1.0.0",
        "run_id": run_id,
        "started_at": now,
        "finished_at": now,
        "status": "succeeded",
        "code": {
            "package": "signal-discovery-workflow",
            "signal_name": signal_name,
            "formula": formula,
        },
        "config": config_summary,
        "pins": data_pins,
        "checks": {
            "validation_only_selection": {
                "ok": ic_results.get("selected_on") == "validation",
                "detail": f"selected_on={ic_results.get('selected_on')}",
            },
            "sector_neutral_screen": {
                "ok": bool(screen_validation.get("sector_neutral"))
                and bool(screen_full.get("sector_neutral")),
                "detail": "both validation and full screens demeaned within sector",
            },
            "factor_alpha_checked": {
                "ok": bool(factor_alpha and factor_alpha.get("alpha_t_stat") is not None),
                "detail": (
                    f"alpha_t={factor_alpha.get('alpha_t_stat'):.2f}"
                    if factor_alpha and factor_alpha.get("alpha_t_stat") is not None
                    else "factor benchmarks unavailable"
                ),
            },
            "deflated_sharpe_checked": {
                "ok": bool(val_dsr.get("dsr") is not None),
                "detail": (
                    f"dsr={val_dsr.get('dsr'):.3f} (n_trials={val_dsr.get('n_trials')})"
                    if val_dsr.get("dsr") is not None
                    else "dsr unavailable"
                ),
            },
            "subperiod_stability_checked": {
                "ok": bool(val_stab.get("positive_block_ratio") is not None),
                "detail": (
                    f"positive_blocks={val_stab.get('positive_block_ratio'):.2f}"
                    if val_stab.get("positive_block_ratio") is not None
                    else "stability unavailable"
                ),
            },
            "execution_realism": {
                "ok": bool(screen_validation.get("vol_scaled"))
                and bool(screen_full.get("vol_scaled")),
                "detail": (
                    f"borrow_bps={screen_validation.get('short_borrow_bps')}, "
                    f"impact_coeff={screen_validation.get('impact_coeff')}, "
                    f"max_weight={screen_validation.get('max_weight')}"
                ),
            },
        },
        "outputs": [
            {"key": signal_artifact_uri, "role": "qsd_signal_json"}
        ] if signal_artifact_uri else [],
        "results_digest": {
            signal_name: {
                "train_ic": train_ic.get("mean_ic"),
                "validation_ic": val_ic.get("mean_ic"),
                "test_ic": test_ic.get("mean_ic"),
                "validation_net_sharpe": val_net.get("sharpe"),
                "full_net_sharpe": full_net.get("sharpe"),
                "validation_alpha_t_stat": (factor_alpha or {}).get("alpha_t_stat"),
                "validation_dsr": val_dsr.get("dsr"),
                "validation_subperiod_stability": val_stab.get("positive_block_ratio"),
            }
        },
    }

    results = {
        "schema": "phf-qsd-signal-results/1.0.0",
        "run_id": run_id,
        "created_at": now,
        "request": request_text,
        "signal_name": signal_name,
        "formula": formula,
        "headline": {
            "validation_ic": val_ic.get("mean_ic"),
            "validation_hac_t": val_ic.get("t_stat"),
            "test_ic": test_ic.get("mean_ic"),
            "validation_net_sharpe": val_net.get("sharpe"),
            "validation_monotonicity": screen_validation.get("monotonicity"),
            "validation_spread_t_stat": screen_validation.get("spread_t_stat"),
            "validation_dsr": val_dsr.get("dsr"),
            "validation_subperiod_stability": val_stab.get("positive_block_ratio"),
            "factor_alpha": factor_alpha,
        },
        "config": config_summary,
        "pins": data_pins,
        "screen_validation": screen_validation,
        "screen_full": screen_full,
        "ic": {
            "train": train_ic,
            "validation": val_ic,
            "test": test_ic,
        },
    }
    return receipt, results


def export_phf_run(
    runs_prefix_uri: str,
    *,
    signal_name: str,
    formula: str | None,
    request_text: str,
    ic_results: dict[str, Any],
    screen_validation: dict[str, Any],
    screen_full: dict[str, Any],
    factor_alpha: dict[str, Any] | None,
    data_pins: dict[str, Any],
    config_summary: dict[str, Any],
    signal_artifact_uri: str | None = None,
) -> dict[str, str]:
    """Write ``receipt.json`` and ``results.json`` under ``<runs_prefix_uri>/<run_id>/``."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{ts}-qsd-{uuid.uuid4().hex[:6]}"
    receipt, results = build_phf_run_payload(
        run_id=run_id,
        signal_name=signal_name,
        formula=formula,
        request_text=request_text,
        ic_results=ic_results,
        screen_validation=screen_validation,
        screen_full=screen_full,
        factor_alpha=factor_alpha,
        data_pins=data_pins,
        config_summary=config_summary,
        signal_artifact_uri=signal_artifact_uri,
    )
    run_dir = r2_store.join(runs_prefix_uri, run_id)
    receipt_uri = r2_store.join(run_dir, "receipt.json")
    results_uri = r2_store.join(run_dir, "results.json")
    r2_store.write_text(receipt_uri, json.dumps(receipt, indent=2, default=str))
    r2_store.write_text(results_uri, json.dumps(results, indent=2, default=str))
    logger.info(f"Exported PHF research run to {run_dir}")
    return {"run_id": run_id, "receipt_uri": receipt_uri, "results_uri": results_uri}
