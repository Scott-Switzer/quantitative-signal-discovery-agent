# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Signal Discovery Optimization Workflow.

    signal_generator -> signal_code_generator -> evaluate every candidate
                ^                                           |
                |__________ optimization advisor ___________|

Changes from the original loop, and why
---------------------------------------
* **Every candidate is scored.** The old code executed all signals in a module
  and threw away all but one, chosen by full-sample |IC|. That is both a
  data-snooping choice and a loss of most of the search. Each function is now
  evaluated on its own and recorded.
* **Selection and acceptance use validation, not test.** The loop previously
  optimized against the same window it reported, so after the first iteration
  the "out-of-sample" IC was in-sample. The split is now
  train / validation / test with a purge gap, the loop only sees train and
  validation, and the test IC is reported once for the chosen signal.
* **Every trial is recorded.** A persistent JSONL ledger enables de-duplication
  (the generator is told what was already tried) and Benjamini-Hochberg FDR over
  the full search history instead of pretending each test was the only one.
* **A signal is screened for cost.** The chosen signal also gets an IC decay
  curve and a cost-aware long/short screen, because an IC that dies at 10 bps of
  turnover is not alpha.
"""

import json
import logging
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import Field

from . import r2_store
from ._compat import (
    Builder,
    FunctionBaseConfig,
    FunctionInfo,
    HumanMessage,
    LLMFrameworkEnum,
    SystemMessage,
    register_function,
)
from .alpha_combiner import AlphaPool, AlphaPoolLibrary
from .feature_store import load_feature_store
from .llm_utils import NO_THINK_INSTRUCTION, extract_json_array, extract_response_text
from .macro_features import compute_factor_alpha, load_factor_returns, load_macro_panels
from .overfitting import deflated_sharpe_ratio, subperiod_stability
from .phf_bridge import build_tradable_universe_mask, load_crsp_weekly_panel, load_phf_panel
from .phf_export import export_phf_run
from .residualize import residualize_forward_returns
from .signal_code_generator import generate_signal_code
from .signal_evaluator import (
    compute_forward_returns,
    compute_ic_decay,
    compute_ic_split,
    compute_long_short_screen,
    execute_signal_functions,
    extract_code_from_response,
    load_sector_map,
    load_stock_data,
)
from .signal_generator import (
    generate_signal_json,
    load_calculator_operators,
    uses_fundamental_fields,
)
from .trial_registry import TrialRegistry, benjamini_hochberg, normalize_formula

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"

_STATUS_HEADLINE = {
    "accepted": "Signal accepted",
    "best_effort": "Best-effort result (threshold not met on validation)",
    "failed": "No valid signals generated",
}


def _python_name(display_name: str | None, index: int) -> str:
    base = re.sub(r"[^A-Za-z0-9_]+", "_", (display_name or f"signal_{index + 1}").lower()).strip("_")
    if not base:
        base = f"signal_{index + 1}"
    if not base.startswith("signal"):
        base = f"signal_{base}"
    return base


def _signal_metadata(signal_json: str) -> dict[str, dict[str, Any]]:
    """Map generated Python function name -> its JSON description."""
    try:
        items = extract_json_array(signal_json)
    except Exception:
        return {}
    meta: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        name = _python_name(item.get("name"), index)
        meta[name] = {
            "display_name": item.get("name"),
            "formula": item.get("formula"),
            "category": item.get("category"),
            "meaning": item.get("meaning"),
        }
    return meta


def _no_candidate_advice(duplicate_skipped: int) -> str:
    """Explain why a batch produced no evaluated candidates.

    A batch that is entirely ledger duplicates needs different guidance than
    one whose formulas were degenerate: the generator must change mechanism or
    field, not just widen a lookback window.
    """
    if duplicate_skipped:
        return (
            f"All {duplicate_skipped} generated formulas are already in the trial ledger "
            "at this horizon. Do not resubmit them. Choose a different data field, "
            "operator, and economic mechanism, and use a lookback that does not appear "
            "in the recent-formula list."
        )
    return (
        "No candidate produced a usable IC. Avoid degenerate formulas "
        "(constant outputs, all-NaN windows) and use longer lookbacks."
    )


def _compose_feedback(
    advice: str,
    best_result: dict | None,
    best_ic: float | None,
) -> str:
    """Bundle the advisor's advice with ONLY the single winning candidate so
    the next iteration anchors on what worked rather than rejected batch siblings."""
    if not best_result or best_ic is None:
        return advice

    meta = best_result.get("meta") or {}
    win_name = meta.get("display_name") or best_result.get("name") or "?"
    win_formula = meta.get("formula")
    if win_formula:
        sc = best_result.get("validation_screen") or {}
        sh = (sc.get("net") or {}).get("sharpe")
        mono = sc.get("monotonicity")
        extra_parts = []
        if sh is not None:
            extra_parts.append(f"val_sharpe={sh:+.2f}")
        if mono is not None:
            extra_parts.append(f"mono={mono:+.2f}")
        suffix = f" ({', '.join(extra_parts)})" if extra_parts else ""
        best_summary = f"- {win_name}: {win_formula}{suffix}"
    else:
        try:
            best_signals = extract_json_array(best_result["signal_json"])
            items = best_signals if best_signals else [json.loads(best_result["signal_json"])]
            first = next((f for f in items if isinstance(f, dict)), None)
            best_summary = f"- {first.get('name', '?')}: {first.get('formula', '?')}" if first else "(unable to summarize)"
        except (json.JSONDecodeError, KeyError, TypeError):
            best_summary = "(unable to summarize)"

    return (
        f"BEST SIGNAL SO FAR (iteration {best_result.get('iteration', '?')}, "
        f"|IC| = {abs(best_ic):.4f} on validation):\n{best_summary}\n\n"
        f"ADVICE FROM LAST ITERATION:\n{advice}\n\n"
        "Try to BEAT both |IC| and validation net Sharpe above with a monotone, "
        "low-turnover tail spread. Do not repeat an already-tried formula."
    )


def _parse_request(raw: str) -> tuple[str, str | None]:
    """Accept a plain request string or ``{"request": ..., "seed_feedback": ...}``."""
    text = (raw or "").strip()
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "request" in obj:
                return str(obj["request"]), obj.get("seed_feedback")
        except json.JSONDecodeError:
            pass
    return raw, None


def _format_workflow_result(
    status: str,
    request: str,
    iteration: int,
    total_iterations: int,
    signal_json: str,
    ic_results: dict,
    saved_path: str | None,
    config,
    last_feedback: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Format the workflow's final result as structured, human-readable JSON."""
    signals_summary = []
    try:
        data = extract_json_array(signal_json) if signal_json else []
        if not data and signal_json:
            parsed = json.loads(signal_json)
            data = parsed if isinstance(parsed, list) else [parsed]
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            signals_summary.append(
                {
                    "name": item.get("name"),
                    "formula": item.get("formula"),
                    "category": item.get("category"),
                    "data_fields_used": item.get("data_fields_used"),
                    "lookback_periods": item.get("lookback_periods"),
                }
            )
    except json.JSONDecodeError:
        pass

    metrics = {
        k: ic_results.get(k)
        for k in ("mean_ic", "ic_std", "ic_ir", "ic_ir_annualized", "t_stat", "p_value",
                  "t_stat_naive", "p_value_naive", "num_periods", "effective_periods",
                  "hac_lags", "positive_ic_ratio")
        if ic_results.get(k) is not None
    }

    evaluation = {
        "train_end_date": getattr(config, "train_end_date", None),
        "validation_end_date": getattr(config, "validation_end_date", None),
        "test_start_date": getattr(config, "test_start_date", None),
        "selected_on": ic_results.get("selected_on"),
        "purge_periods": ic_results.get("purge_periods"),
        "sector_neutralized": ic_results.get("sector_neutralized"),
        "sector_coverage": ic_results.get("sector_coverage"),
        "forward_periods": getattr(config, "forward_periods", None),
    }
    for key in ("ic_train", "ic_validation", "ic_test", "ic_full",
                "ic_raw_train", "ic_raw_validation", "ic_raw_test", "ic_raw_full"):
        if ic_results.get(key):
            evaluation[key] = ic_results[key]
    if ic_results.get("diagnostics"):
        evaluation["diagnostics"] = ic_results["diagnostics"]

    payload: dict[str, Any] = {
        "status": status,
        "headline": _STATUS_HEADLINE.get(status, status),
        "request": request,
        "iteration": iteration,
        "total_iterations": total_iterations,
        "selected_signal": ic_results.get("selected_signal"),
        "thresholds": {
            "ic_threshold": getattr(config, "ic_threshold", None),
            "p_value_threshold": getattr(config, "p_value_threshold", None),
        },
        "metrics": metrics,
        "evaluation": evaluation,
        "signals": signals_summary,
        "saved_path": saved_path,
    }

    # The holdout is reported separately and is never used for selection.
    if ic_results.get("ic_test"):
        payload["holdout"] = ic_results["ic_test"]

    if ic_results.get("error"):
        payload["error"] = ic_results["error"]

    if last_feedback:
        payload["last_feedback"] = last_feedback

    if extra:
        payload.update(extra)

    return json.dumps(payload, indent=2, default=str)


def save_signal_results(
    signal_json: str,
    signal_code: str,
    ic_results: dict,
    iteration: int,
    extra: dict[str, Any] | None = None,
    output_uri: str | None = None,
) -> str:
    """Persist one result, to R2 when ``output_uri`` is given.

    ``output_uri`` is a prefix (``r2://bucket/prefix``). Without it the JSON is
    written next to the package, which is only for tests and an explicit local
    run; the workflow passes an R2 prefix whenever one is configured.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"signal_{timestamp}_iter{iteration}.json"

    metrics = {k: v for k, v in ic_results.items() if k != "selected_signal"}
    payload: dict[str, Any] = {
        "timestamp": timestamp,
        "iteration": iteration,
        "selected_signal": ic_results.get("selected_signal"),
        "signal_description": signal_json,
        "signal_code": signal_code,
        "evaluation_metrics": metrics,
    }
    if extra:
        payload.update(extra)
    body = json.dumps(payload, indent=2, default=str)

    if output_uri:
        uri = r2_store.join(output_uri, filename)
        r2_store.write_text(uri, body)
        logger.info(f"Saved signal results to {uri}")
        return uri

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_DIR / filename
    filepath.write_text(body)
    logger.info(f"Saved signal results to {filepath}")
    return str(filepath)


class SignalOptimizerConfig(FunctionBaseConfig, name="signal_optimizer"):
    """Iteratively generate, evaluate, and refine quantitative signals."""

    signal_generator_llm: str | None = Field(default=None, description="LLM for signal generation")
    code_generator_llm: str | None = Field(default=None, description="LLM for code generation")
    optimization_advisor_llm: str | None = Field(default=None, description="LLM for optimization feedback")
    llm_name: str | None = Field(default=None, description="Fallback LLM if specific ones not set")
    ic_threshold: float = Field(default=0.02, description="Minimum |IC| on validation to accept")
    p_value_threshold: float = Field(default=0.05, description="Maximum HAC p-value")
    max_iterations: int = Field(default=4, description="Max optimization attempts")
    num_signals: int = Field(default=6, description="Signals per iteration")
    forward_periods: int = Field(default=5, description="Forward return periods")
    train_end_date: str = Field(default="2020-12-31", description="Last date of the train period")
    validation_end_date: str | None = Field(
        default="2024-12-31",
        description="Last date of the validation window. The test holdout starts after it.",
    )
    test_start_date: str = Field(default="2021-01-01", description="First date of the out-of-sample period")
    sector_neutralize: bool = Field(default=True, description="Sector-neutral rank IC")
    min_names: int = Field(default=10, description="Minimum names in a cross-section")
    fundamentals_only: bool = Field(
        default=False,
        description=(
            "Discard candidates that never touch a fundamental or ratio panel; "
            "price-only formulas are momentum or volatility in disguise"
        ),
    )
    periods_per_year: int = Field(
        default=252,
        description="Rows per year: 252 for daily panels, 12 for the monthly CRSP panel",
    )
    fundamentals_uri: str | None = Field(
        default=None,
        description="R2 URI of a point-in-time Compustat release directory",
    )
    ccm_uri: str | None = Field(
        default=None,
        description="R2 URI of the CRSP/Compustat CCM link release",
    )
    sec_companyfacts_uri: str | None = Field(
        default=None,
        description="R2 URI of the SEC XBRL companyfacts release (2009 onward)",
    )
    cik_bridge_uri: str | None = Field(
        default=None,
        description="R2 URI of the dated CIK-to-permno bridge assertions",
    )
    data_manifest_uri: str | None = Field(
        default=None,
        description="Manifest object pinning the market-data release",
    )
    data_manifest_sha256: str | None = Field(
        default=None,
        description="Expected SHA-256 of the market-data manifest bytes",
    )
    sec_companyfacts_manifest_uri: str | None = Field(
        default=None,
        description="MANIFEST.json pinning the SEC companyfacts release",
    )
    sec_manifest_sha256: str | None = Field(
        default=None,
        description="Expected SHA-256 of the SEC companyfacts manifest bytes",
    )
    purge_periods: int | None = Field(default=None, description="Rows purged at split boundaries; default = forward_periods")
    cost_bps: float = Field(default=10.0, description="One-way cost in basis points for the screen")
    screen_quantile: float = Field(default=0.2, description="Long/short leg size for the screen")
    ic_decay_horizons: str = Field(default="1,5,10,21,63", description="Comma-separated IC decay horizons")
    registry_path: str | None = Field(
        default=None,
        description="Trial ledger URI (r2://bucket/key). Defaults to <results_uri>/trials.jsonl",
    )
    results_uri: str | None = Field(
        default=None,
        description="R2 prefix for saved results and the ledger, e.g. r2://bucket/research/signals",
    )
    data_source: str | None = Field(
        default=None,
        description="Panel URI (r2://...). When set, panels are read in memory; no CSV staging",
    )
    require_remote: bool = Field(
        default=False,
        description="Fail instead of using local CSV panels when data_source is unset",
    )
    dedupe_trials: bool = Field(default=True, description="Skip formulas already in the ledger at this horizon")
    save_results: bool = Field(default=True, description="Save results")
    pin_policy: str = Field(
        default="enforce",
        description=(
            "enforce: refuse to run when a manifest hash changed. record: log the "
            "mismatch and store the hash actually read. Use record for daily "
            "research on a rolling dataset, enforce to reproduce a result."
        ),
    )
    data_release_prefix: str | None = Field(
        default=None,
        description=(
            "Release family prefix. The newest child release is resolved by "
            "modification time, so daily republication needs no config edit."
        ),
    )
    sec_release_prefix: str | None = Field(
        default=None,
        description="Release family prefix for the fundamentals dataset",
    )
    stop_on_acceptance: bool = Field(
        default=False,
        description=(
            "Stop at the first acceptance. Leave false for a full search: "
            "stopping early makes the Benjamini-Hochberg q-values invalid "
            "because the number of trials was itself a stopping decision."
        ),
    )
    require_fdr_significance: bool = Field(
        default=False,
        description="Accept only candidates whose BH q-value over the full ledger is below fdr_q",
    )
    fdr_q: float = Field(default=0.10, description="Benjamini-Hochberg false-discovery rate")
    min_net_sharpe: float = Field(
        default=0.0,
        description=(
            "Minimum net Sharpe on the validation-window screen. 0 disables "
            "the economic gate. The holdout screen is reported for context."
        ),
    )
    min_monotonicity: float = Field(
        default=0.0,
        description=(
            "Minimum direction-adjusted quintile monotonicity on the validation "
            "screen. 0 disables the check. A high IC with flat or inverted "
            "quintiles is not a tradable spread."
        ),
    )
    min_spread_tstat: float = Field(
        default=0.0,
        description=(
            "Minimum t-stat of the net long/short spread on the validation "
            "screen. 0 disables the check."
        ),
    )
    lakehouse_v2_uri: str | None = Field(
        default=None,
        description="R2 Parquet data prefix for gold.security_monthly_fundamentals_v2",
    )
    min_price: float = Field(default=0.0, description="Formation-date price floor for tradable universe")
    min_dollar_volume: float = Field(default=0.0, description="Formation-date monthly dollar volume floor")
    top_n_market_cap: int = Field(default=0, description="Keep top-N securities by lagged market cap per date")
    min_market_cap: float = Field(default=0.0, description="Formation-date market cap floor")
    fred_macro_uri: str | None = Field(
        default=None,
        description="R2 pointer or release URI for FRED macro regime panels",
    )
    factor_returns_uri: str | None = Field(
        default=None,
        description="R2 prefix for AQR/JKP factor returns used for residual alpha regression",
    )
    min_alpha_tstat: float = Field(
        default=0.0,
        description="Minimum HAC t-stat on factor-adjusted residual alpha in validation (0 disables)",
    )
    sharded_ledger: bool = Field(
        default=False,
        description="Write immutable per-run shards under <ledger_dir>/trials/ instead of read-modify-write",
    )
    phf_runs_uri: str | None = Field(
        default=None,
        description="R2 prefix (e.g. r2://.../research/runs) to export accepted signals in PHF receipt format",
    )
    screen_buffer_quantile: float | None = Field(
        default=None,
        description="Garleanu-Pedersen exit hysteresis quantile (e.g. 0.30 with screen_quantile=0.15)",
    )
    min_dollar_volume_pct: float = Field(
        default=0.0,
        description="Cross-sectional dollar volume percentile floor (e.g. 0.20 keeps top 80% liquid names)",
    )
    residualize_returns: bool = Field(
        default=False,
        description="Cross-sectionally residualize forward returns against size, momentum, volatility, and sector",
    )
    enable_alpha_pool: bool = Field(
        default=False,
        description="Admit orthogonal weak-alpha candidates and combine via train-only L2 Ridge stacking",
    )
    pool_min_ic: float = Field(default=0.010, description="Minimum validation |IC| for alpha pool admission")
    pool_max_corr: float = Field(default=0.50, description="Max pairwise cross-sectional rank correlation in alpha pool")
    pool_max_size: int = Field(default=25, description="Maximum orthogonal signals in alpha pool")
    pool_ridge_alpha: float = Field(default=10.0, description="L2 Ridge shrinkage parameter for meta-alpha stacking")
    sec_quarterly_uri: str | None = Field(
        default=None,
        description="R2 Parquet prefix for fundamentals.sec_quarterly acceleration panels",
    )
    lazy_prices_uri: str | None = Field(
        default=None,
        description="R2 Parquet URI for precomputed SEC filing text diffs (Lazy Prices)",
    )
    weekly_crsp_uri: str | None = Field(
        default=None,
        description="R2 prefix for gold/crsp-merged daily CRSP resampled weekly (W-FRI)",
    )
    weekly_start_date: str = Field(
        default="2005-01-01",
        description="Start date pushdown when loading weekly CRSP from daily crsp-merged",
    )
    weekly_end_date: str | None = Field(
        default=None,
        description="Optional end date pushdown for weekly CRSP (e.g. 2024-12-31 to keep 2025 vault untouched)",
    )
    feature_store_uri: str | None = Field(
        default=None,
        description="Optional precomputed R2 feature-store prefix; skips raw Parquet rebuilds",
    )
    alpha_pool_uri: str | None = Field(
        default=None,
        description="Optional R2 prefix for sharded cross-run AlphaPool persistence and evolutionary parents",
    )
    pool_gbdt_weight: float = Field(
        default=0.45,
        description="Blend weight for shallow GBDT/non-linear stacker in heterogeneous meta-alpha (0 = Ridge only)",
    )
    eval_workers: int = Field(
        default=4,
        description="Thread pool workers for parallel candidate IC evaluation within each iteration",
    )
    form4_uri: str | None = Field(
        default=None,
        description="R2 Parquet URI for pre-ingested SEC Form 4 insider buying events",
    )
    f13_uri: str | None = Field(
        default=None,
        description="R2 Parquet URI for pre-ingested SEC Form 13F institutional holdings",
    )
    short_borrow_bps: float = Field(
        default=0.0,
        description="Annualized short-borrow fee in bps charged on the short leg (e.g. 50.0)",
    )
    impact_coeff: float = Field(
        default=0.0,
        description="Square-root market impact coefficient against dollar volume (e.g. 0.15)",
    )
    aum_usd: float = Field(
        default=1_000_000.0,
        description="Reference strategy AUM in USD for square-root market impact estimation",
    )
    vol_scaled: bool = Field(
        default=False,
        description="Use inverse-volatility risk-scaled weights inside long/short legs",
    )
    max_weight: float | None = Field(
        default=None,
        description="Single-name maximum leg weight cap (e.g. 0.03)",
    )
    min_dsr: float = Field(
        default=0.0,
        description="Minimum Bailey & Lopez de Prado Deflated Sharpe Ratio probability (0 disables; 0.95 recommended)",
    )
    min_subperiod_stability: float = Field(
        default=0.0,
        description="Minimum fraction of contiguous validation sub-periods with positive return (0 disables; 0.50 recommended)",
    )


def _auto_purge(config: SignalOptimizerConfig) -> int:
    if config.purge_periods is not None:
        return max(0, int(config.purge_periods))
    return max(0, int(config.forward_periods))


@register_function(config_type=SignalOptimizerConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def signal_optimizer_function(config: SignalOptimizerConfig, builder: Builder):
    """Compose the signal / code / evaluator agents into an optimization loop."""

    signal_llm_name = config.signal_generator_llm or config.llm_name
    code_llm_name = config.code_generator_llm or config.llm_name
    advisor_llm_name = config.optimization_advisor_llm or config.llm_name
    if not signal_llm_name:
        raise ValueError("Must specify llm_name or signal_generator_llm")

    signal_llm = await builder.get_llm(signal_llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    code_llm = await builder.get_llm(code_llm_name or signal_llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    advisor_llm = await builder.get_llm(advisor_llm_name or signal_llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    logger.info(f"LLMs: signal={signal_llm_name}, code={code_llm_name}, advisor={advisor_llm_name}")

    operators = load_calculator_operators()

    from . import pins as pin_store

    pin_policy = config.pin_policy or os.environ.get("QSD_PIN_POLICY", "enforce")
    resolved_releases: dict[str, Any] = {}

    # Fast path: load precomputed R2 feature-store snapshot (~3-5s instead of ~2.5m).
    # Falls back cleanly to raw R2 loaders if the snapshot has not been built yet.
    data_pins: dict[str, Any] = {}
    loaded_from_store = False
    precomputed_mask: pd.DataFrame | None = None
    precomputed_resid: pd.DataFrame | None = None
    feature_store_uri = config.feature_store_uri or os.environ.get("QSD_FEATURE_STORE_URI")
    if feature_store_uri:
        try:
            bundle = load_feature_store(feature_store_uri)
            stock_data = dict(bundle["panels"])
            panel_sectors = bundle.get("sector_map") or {}
            sector_map = panel_sectors if (config.sector_neutralize and panel_sectors) else None
            precomputed_mask = bundle.get("universe_mask")
            if config.forward_periods == 1 and config.residualize_returns:
                precomputed_resid = bundle.get("residual_returns")
            data_pins["feature_store"] = {
                "uri": bundle["manifest_uri"],
                "sha256": bundle["manifest_sha256"],
                "policy": pin_policy,
            }
            loaded_from_store = True
        except Exception as e:
            logger.warning(
                f"Feature store unavailable at {feature_store_uri} ({e}); falling back to raw R2 loaders"
            )

    if not loaded_from_store:
        weekly_crsp_uri = config.weekly_crsp_uri or os.environ.get("QSD_WEEKLY_CRSP_URI")
        data_source = config.data_source or os.environ.get("QSD_DATA_SOURCE")
        if config.data_release_prefix and not weekly_crsp_uri:
            resolved_releases["market_data"] = pin_store.resolve_release(
                config.data_release_prefix, "panel.parquet", "manifest.json"
            )
            data_source = resolved_releases["market_data"]["artifact_uri"]
        if config.require_remote and not data_source and not weekly_crsp_uri:
            raise ValueError(
                "require_remote is set but no data_source or weekly_crsp_uri was provided. "
                "Set data_source/QSD_DATA_SOURCE to an R2 panel URI; refusing to fall back to local CSVs."
            )
        if weekly_crsp_uri:
            panels, panel_sectors = load_crsp_weekly_panel(
                weekly_crsp_uri, start=config.weekly_start_date, end=config.weekly_end_date
            )
            stock_data = dict(panels)
            sector_map = panel_sectors if (config.sector_neutralize and panel_sectors) else None
        elif data_source:
            panels, panel_sectors = load_phf_panel(data_source)
            stock_data = dict(panels)
            sector_map = panel_sectors if config.sector_neutralize else None
        else:
            stock_data = load_stock_data()
            sector_map = load_sector_map() if config.sector_neutralize else None

    close_panel = stock_data.get("Close")
    if close_panel is None or close_panel.empty:
        raise ValueError(
            "No Close panel loaded. Set data_source/QSD_DATA_SOURCE to an R2 panel URI, "
            "or run python -m signal_discovery_workflow.download_data for the local fallback."
        )

    fundamentals_uri = config.fundamentals_uri or os.environ.get("QSD_FUNDAMENTALS_URI")
    sec_companyfacts_uri = config.sec_companyfacts_uri or os.environ.get(
        "QSD_SEC_COMPANYFACTS_URI"
    )
    if config.sec_release_prefix:
        resolved_releases["fundamentals"] = pin_store.resolve_release(
            config.sec_release_prefix, None, "MANIFEST.json"
        )
        sec_companyfacts_uri = resolved_releases["fundamentals"]["release_uri"]

    if not loaded_from_store:
        data_manifest_uri = (resolved_releases.get("market_data") or {}).get(
            "manifest_uri"
        ) or config.data_manifest_uri
        if data_manifest_uri:
            data_pins["market_data"] = pin_store.verify(
                data_manifest_uri,
                config.data_manifest_sha256 or "",
                role="market_data",
                policy=pin_policy,
            )
        if sec_companyfacts_uri:
            sec_manifest = (
                (resolved_releases.get("fundamentals") or {}).get("manifest_uri")
                or config.sec_companyfacts_manifest_uri
                or pin_store.manifest_for(sec_companyfacts_uri, "MANIFEST.json")
            )
            data_pins["fundamentals"] = pin_store.verify(
                sec_manifest,
                config.sec_manifest_sha256 or "",
                role="fundamentals",
                policy=pin_policy,
            )
        lakehouse_v2_uri = config.lakehouse_v2_uri or os.environ.get("QSD_LAKEHOUSE_V2_URI")
        if lakehouse_v2_uri:
            from .fundamentals import load_lakehouse_v2_fundamentals_panels

            fundamental_panels = load_lakehouse_v2_fundamentals_panels(
                uri=lakehouse_v2_uri,
                dates=close_panel.index,
                securities=close_panel.columns,
                market_cap=stock_data.get("MarketCap"),
            )
            stock_data.update(fundamental_panels)
            logger.info(
                f"Loaded {len(fundamental_panels)} Lakehouse v2 fundamental panels: "
                + ", ".join(sorted(fundamental_panels))
            )
        elif sec_companyfacts_uri:
            from .fundamentals import load_sec_fundamentals_panels

            fundamental_panels = load_sec_fundamentals_panels(
                companyfacts_uri=sec_companyfacts_uri,
                bridge_uri=config.cik_bridge_uri or os.environ.get("QSD_CIK_BRIDGE_URI", ""),
                dates=close_panel.index,
                securities=close_panel.columns,
                market_cap=stock_data.get("MarketCap"),
            )
            stock_data.update(fundamental_panels)
            logger.info(
                f"Loaded {len(fundamental_panels)} SEC fundamental panels: "
                + ", ".join(sorted(fundamental_panels))
            )
        elif fundamentals_uri:
            from .fundamentals import load_fundamentals_panels

            fundamental_panels = load_fundamentals_panels(
                fundamentals_uri=fundamentals_uri,
                ccm_uri=config.ccm_uri or os.environ.get("QSD_CCM_URI", ""),
                dates=close_panel.index,
                securities=close_panel.columns,
                market_cap=stock_data.get("MarketCap"),
            )
            stock_data.update(fundamental_panels)
            logger.info(
                f"Loaded {len(fundamental_panels)} fundamental panels: "
                + ", ".join(sorted(fundamental_panels))
            )

        cik_bridge_uri = config.cik_bridge_uri or os.environ.get("QSD_CIK_BRIDGE_URI", "")
        sec_quarterly_uri = config.sec_quarterly_uri or os.environ.get("QSD_SEC_QUARTERLY_URI")
        if sec_quarterly_uri and cik_bridge_uri:
            try:
                from .fundamentals import load_sec_quarterly_acceleration_panels

                q_panels = load_sec_quarterly_acceleration_panels(
                    quarterly_uri=sec_quarterly_uri,
                    bridge_uri=cik_bridge_uri,
                    dates=close_panel.index,
                    securities=close_panel.columns,
                )
                stock_data.update(q_panels)
            except Exception as e:
                logger.warning(f"Could not load SEC quarterly acceleration panels: {e}")

        lazy_prices_uri = config.lazy_prices_uri or os.environ.get("QSD_LAZY_PRICES_URI")
        if lazy_prices_uri and cik_bridge_uri:
            try:
                from .lazy_prices import load_lazy_prices_panels

                lp_panels = load_lazy_prices_panels(
                    panel_uri=lazy_prices_uri,
                    bridge_uri=cik_bridge_uri,
                    dates=close_panel.index,
                    securities=close_panel.columns,
                )
                stock_data.update(lp_panels)
            except Exception as e:
                logger.warning(f"Could not load Lazy Prices filing diff panels: {e}")

        form4_uri = config.form4_uri or os.environ.get("QSD_FORM4_URI")
        if form4_uri and cik_bridge_uri:
            try:
                from .alt_data import load_form4_panels

                f4_panels = load_form4_panels(
                    panel_uri=form4_uri,
                    bridge_uri=cik_bridge_uri,
                    dates=close_panel.index,
                    securities=close_panel.columns,
                )
                stock_data.update(f4_panels)
            except Exception as e:
                logger.warning(f"Could not load SEC Form 4 insider panels: {e}")

        f13_uri = config.f13_uri or os.environ.get("QSD_F13_URI")
        if f13_uri and cik_bridge_uri and cik_bridge_uri.lower().endswith(".parquet"):
            try:
                from .alt_data import load_13f_panels

                f13_panels = load_13f_panels(
                    panel_uri=f13_uri,
                    ccm_uri=cik_bridge_uri,
                    dates=close_panel.index,
                    securities=close_panel.columns,
                )
                stock_data.update(f13_panels)
            except Exception as e:
                logger.warning(f"Could not load SEC Form 13F institutional panels: {e}")

        fred_macro_uri = config.fred_macro_uri or os.environ.get("QSD_FRED_MACRO_URI")
        if fred_macro_uri:
            if fred_macro_uri.endswith(".json"):
                try:
                    data_pins["fred_macro"] = pin_store.verify(
                        fred_macro_uri, "", role="fred_macro", policy="record"
                    )
                except Exception as e:
                    logger.warning(f"Could not record FRED pointer pin: {e}")
            macro_panels = load_macro_panels(
                fred_uri=fred_macro_uri,
                dates=close_panel.index,
                securities=close_panel.columns,
                lag_periods=1,
            )
            stock_data.update(macro_panels)

    if data_pins:
        logger.info(
            "Data pins: "
            + ", ".join(f"{role}={rec['sha256'][:12]}" for role, rec in data_pins.items())
        )

    factor_returns_uri = config.factor_returns_uri or os.environ.get("QSD_FACTOR_RETURNS_URI")
    factor_returns = None
    if factor_returns_uri:
        try:
            factor_returns = load_factor_returns(factor_returns_uri)
        except Exception as e:
            logger.warning(f"Could not load factor benchmarks from {factor_returns_uri}: {e}")

    # A configured alpha gate that cannot run must not silently disappear: a
    # run that accepts signals without it would report an ungated result as
    # factor-neutral.
    if config.min_alpha_tstat > 0 and (factor_returns is None or factor_returns.empty):
        raise RuntimeError(
            "min_alpha_tstat is set but US factor benchmarks could not be loaded; "
            "refusing to accept signals without the factor-neutral alpha gate."
        )

    if precomputed_mask is not None:
        universe_mask = precomputed_mask
    else:
        universe_mask = build_tradable_universe_mask(
            stock_data,
            min_price=config.min_price,
            min_dollar_volume=config.min_dollar_volume,
            min_dollar_volume_pct=config.min_dollar_volume_pct,
            top_n_market_cap=config.top_n_market_cap,
            min_market_cap=config.min_market_cap,
        )
    # Keep internal Price out of LLM-advertised panels; Close is the total-return index.
    stock_data.pop("Price", None)

    results_uri = config.results_uri or os.environ.get("QSD_RESULTS_URI")
    ledger_uri = config.registry_path or os.environ.get("QSD_LEDGER_URI")
    if not ledger_uri and results_uri:
        ledger_uri = r2_store.join(results_uri, "trials.jsonl")
    if not ledger_uri:
        logger.warning(
            "No results_uri/ledger_uri configured: the trial ledger is written next to "
            "the package. Set QSD_RESULTS_URI=r2://bucket/prefix to keep it in R2."
        )
    alpha_pool_uri = config.alpha_pool_uri or os.environ.get("QSD_ALPHA_POOL_URI")
    if not alpha_pool_uri and config.enable_alpha_pool and results_uri:
        alpha_pool_uri = r2_store.join(results_uri, "alpha_pool")
    pool_library = AlphaPoolLibrary(alpha_pool_uri) if alpha_pool_uri else None

    sector_coverage = None
    if sector_map is not None:
        universe = list(stock_data["Close"].columns)
        known = sum(1 for t in universe if sector_map.get(t) not in (None, "Unknown"))
        sector_coverage = {"mapped": known, "total": len(universe)}
        logger.info(f"Sector neutralization active: {known}/{len(universe)} tickers mapped")

    purge = _auto_purge(config)
    close = stock_data["Close"]
    if precomputed_resid is not None:
        forward_returns = precomputed_resid
    else:
        raw_forward_returns = compute_forward_returns(close, periods=config.forward_periods)
        if config.residualize_returns:
            forward_returns = residualize_forward_returns(
                raw_forward_returns,
                close,
                market_cap=stock_data.get("MarketCap"),
                sector_map=sector_map,
                periods_per_year=config.periods_per_year,
                min_names=max(config.min_names, 30),
            )
        else:
            forward_returns = raw_forward_returns
    registry = TrialRegistry(ledger_uri, sharded=config.sharded_ledger)
    # Load all existing ledger shards ONCE in memory and append locally per trial.
    # Calling registry.load() per candidate causes 100+ redundant R2 scans per run.
    cached_trials: list[dict[str, Any]] = registry.load()
    tried = [str(t.get("formula") or "") for t in cached_trials if t.get("formula")]
    logger.info(f"Trial ledger: {len(cached_trials)} prior trials ({len(tried)} formulas) at {registry.path}")

    decay_horizons = [
        int(h.strip()) for h in str(config.ic_decay_horizons).split(",") if h.strip().isdigit()
    ]

    # ---- per-step helpers -------------------------------------------------

    def _apply_universe_mask(values: pd.DataFrame) -> pd.DataFrame:
        if universe_mask is None:
            return values
        return values.where(universe_mask.reindex_like(values).fillna(False))

    def evaluate_candidate(signal_values, name: str, meta: dict[str, Any]) -> dict[str, Any]:
        """Score one candidate on train/validation/test with HAC significance."""
        masked = _apply_universe_mask(signal_values)
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            ic = compute_ic_split(
                masked,
                forward_returns,
                train_end_date=config.train_end_date,
                test_start_date=config.test_start_date,
                sector_map=sector_map,
                validation_end_date=config.validation_end_date,
                purge_periods=purge,
                periods=config.forward_periods,
                min_names=config.min_names,
                periods_per_year=config.periods_per_year,
                include_raw=False,
            )
        ic["selected_signal"] = name
        ic["candidate"] = {**meta, "name": name}
        if sector_coverage is not None:
            ic["sector_coverage"] = sector_coverage
        return ic

    def passes_statistics(ic: dict) -> bool:
        if ic.get("insufficient_history"):
            return False
        if ic.get("selected_on") == "validation" and (
            (ic.get("ic_validation") or {}).get("num_periods") or 0
        ) == 0:
            return False
        mean_ic = ic.get("mean_ic")
        p_value = ic.get("p_value")
        if mean_ic is None or abs(mean_ic) < config.ic_threshold:
            return False
        if p_value is not None and p_value > config.p_value_threshold:
            return False
        train_ic = (ic.get("ic_train") or {}).get("mean_ic")
        if train_ic is not None and train_ic * mean_ic < 0:
            return False
        return True

    def fdr_annotation(p_value: float | None) -> dict[str, Any]:
        """BH q-value for a new trial against the whole ledger history."""
        if p_value is None:
            return {"q_value": None, "significant_fdr": False}
        existing = [
            float(t["p_value"])
            for t in cached_trials
            if t.get("p_value") is not None
        ]
        pvalues = existing + [float(p_value)]
        qvalues, rejected = benjamini_hochberg(pvalues, q=config.fdr_q)
        return {"q_value": qvalues[-1], "significant_fdr": bool(rejected[-1])}

    def screen_on_validation(values, direction: float) -> dict[str, Any]:
        """Cost/impact-aware screen + factor-alpha + DSR + stability on validation only."""
        if not config.validation_end_date:
            return {"error": "no validation window configured"}
        masked = _apply_universe_mask(values)
        end = pd.Timestamp(config.validation_end_date)
        start = pd.Timestamp(config.train_end_date) if config.train_end_date else None
        in_window = masked.index <= end
        if start is not None:
            in_window = in_window & (masked.index > start)
            # Slice close to validation + 2-year rolling-vol warmup instead of
            # recomputing pct_change().rolling() across 60+ years per candidate.
            warmup_start = start - pd.Timedelta(days=760)
            close_slice = close.loc[(close.index >= warmup_start) & (close.index <= end)]
        else:
            close_slice = close.loc[close.index <= end]
        need_series = bool(
            factor_returns is not None or config.min_dsr > 0 or config.min_subperiod_stability > 0
        )
        screen = compute_long_short_screen(
            masked.loc[in_window],
            close_slice,
            quantile=config.screen_quantile,
            buffer_quantile=config.screen_buffer_quantile,
            cost_bps=config.cost_bps,
            min_names=max(config.min_names, 20),
            annualization=config.periods_per_year,
            direction=direction,
            sector_map=sector_map,
            include_series=need_series,
            short_borrow_bps=config.short_borrow_bps,
            impact_coeff=config.impact_coeff,
            volume_panel=stock_data.get("Volume"),
            aum_usd=config.aum_usd,
            vol_scaled=config.vol_scaled,
            max_weight=config.max_weight,
            holding_periods=config.forward_periods,
        )
        net_series = screen.pop("_net_series", None)
        if net_series is not None:
            if factor_returns is not None:
                screen["factor_alpha"] = compute_factor_alpha(
                    net_series,
                    factor_returns,
                    periods_per_year=config.periods_per_year,
                )
            n_trials = max(1, len(cached_trials) + 1)
            screen["deflated_sharpe"] = deflated_sharpe_ratio(
                net_series,
                n_trials=n_trials,
                periods_per_year=config.periods_per_year,
            )
            screen["subperiod_stability"] = subperiod_stability(
                net_series,
                periods_per_year=config.periods_per_year,
            )
        return screen

    def candidate_key(candidate: dict) -> tuple[int, int, int, float, float]:
        """Rank candidates using train/validation tail-spread Sharpe and IC.

        Accepted beats rejected; sign-consistent beats flipped; positive
        validation net Sharpe beats negative tail spread; then Sharpe and |IC|.
        """
        ic = candidate.get("ic") or {}
        validation = ic.get("mean_ic")
        train = (ic.get("ic_train") or {}).get("mean_ic")
        consistent = (
            train is not None and validation is not None and train * validation > 0
        )
        val_sh = ((candidate.get("validation_screen") or {}).get("net") or {}).get("sharpe")
        return (
            1 if candidate.get("accepted") else 0,
            1 if consistent else 0,
            1 if (val_sh is not None and val_sh > 0) else 0,
            float(val_sh) if val_sh is not None else -99.0,
            abs(validation or 0.0),
        )

    def _trial_key(meta: dict[str, Any]) -> str:
        return f"{normalize_formula(meta.get('formula'))}|h={config.forward_periods}"

    async def generate_feedback(candidates: list[dict], iteration: int) -> str:
        """Ask the advisor for compact, actionable feedback using train/validation only.

        The holdout IC is deliberately not shown: the advisor is part of the
        search, and a search that can see the test set is not out of sample.
        """
        rows = []
        for cand in candidates:
            ic = cand["ic"]
            sc = cand.get("validation_screen") or {}
            sh = (sc.get("net") or {}).get("sharpe")
            mono = sc.get("monotonicity")
            to = sc.get("annual_turnover")
            at = (sc.get("factor_alpha") or {}).get("alpha_t_stat")
            dsr_v = (sc.get("deflated_sharpe") or {}).get("dsr")
            stab_v = (sc.get("subperiod_stability") or {}).get("positive_block_ratio")
            diag = ""
            if sh is not None or at is not None:
                diag = (
                    f" | val Sharpe={sh:.2f}" if sh is not None else ""
                ) + (
                    f" | mono={mono:.2f}" if mono is not None else ""
                ) + (
                    f" | turnover={to:.1f}x" if to is not None else ""
                ) + (
                    f" | alpha_t={at:.2f}" if at is not None else ""
                ) + (
                    f" | DSR={dsr_v:.2f}" if dsr_v is not None else ""
                ) + (
                    f" | stab={stab_v:.2f}" if stab_v is not None else ""
                )
            rows.append(
                "- {name} [{cat}]: formula={formula}\n"
                "    train IC={tr} | validation IC={val} | p={p}{diag} | verdict={verdict}".format(
                    name=cand["name"],
                    cat=(cand["meta"].get("category") or "unknown"),
                    formula=(cand["meta"].get("formula") or "?"),
                    tr=f"{ic['ic_train']['mean_ic']:.4f}" if ic.get("ic_train", {}).get("mean_ic") is not None else "n/a",
                    val=f"{ic['mean_ic']:.4f}" if ic.get("mean_ic") is not None else "n/a",
                    p=f"{ic['p_value']:.4f}" if ic.get("p_value") is not None else "n/a",
                    diag=diag,
                    verdict="accepted" if cand.get("accepted") else "rejected",
                )
            )
        best = max(
            (c for c in candidates if c["ic"].get("mean_ic") is not None),
            key=lambda c: abs(c["ic"]["mean_ic"]),
            default=None,
        )
        best_line = (
            f"Best so far: {best['name']} ({best['meta'].get('formula')}) with "
            f"{best['ic']['mean_ic']:.4f} validation IC."
            if best else "No candidate produced a usable IC."
        )
        target_desc = (
            "idiosyncratic residual forward return (orthogonal to size, momentum, vol, sector)"
            if config.residualize_returns
            else f"{config.forward_periods}-period forward return"
        )

        prompt = f"""Iteration {iteration} of a closed-loop factor search.

Objective: maximize |rank IC| on the VALIDATION window, sector-neutral, against
{target_desc}. The test holdout is hidden from
you. Acceptance needs |IC| >= {config.ic_threshold}, HAC p <= {config.p_value_threshold}, and orthogonal factor alpha.

{chr(10).join(rows)}

{best_line}

Output exactly 3 bullet points, max 15 words each, suggesting concrete
changes (different operator, different lookback, different data field, a
different economic mechanism). No prose, no preamble. Format:
- <change 1>
- <change 2>
- <change 3>"""

        response = await advisor_llm.ainvoke(
            [
                SystemMessage(content=NO_THINK_INSTRUCTION),
                SystemMessage(content="You are a senior quant providing concise signal optimization advice."),
                HumanMessage(content=prompt),
            ]
        )
        feedback = extract_response_text(response).strip()
        if len(feedback) > 600:
            feedback = feedback[:600].rsplit("\n", 1)[0]
        return feedback

    # ---- main optimization loop -------------------------------------------

    async def run_optimization(request: str) -> str:
        """
        Run the closed-loop signal discovery optimization.

        Args:
            request: Either a plain signal request string (e.g.,
                ``"momentum signals"``) or a JSON object with the shape
                ``{"request": "momentum signals", "seed_feedback": "..."}``
                to resume from a prior run's ``last_feedback``.
        """
        request_text, seed_feedback = _parse_request(request)

        best_result: dict | None = None
        best_ic: float | None = None
        feedback: str | None = seed_feedback
        last_error: str | None = None
        trials_this_run: list[dict] = []
        all_candidates: list[dict] = []
        accepted_candidates: list[dict] = []
        seen_this_run: set[str] = set()
        duplicate_skipped = 0
        ledger_keys = registry.tried_keys()
        alpha_pool = (
            AlphaPool(max_pairwise_corr=config.pool_max_corr, max_size=config.pool_max_size)
            if config.enable_alpha_pool
            else None
        )
        archived_parents = (
            pool_library.top_parents(6, horizon=config.forward_periods)
            if pool_library is not None
            else []
        )

        if seed_feedback:
            logger.info(f"Resuming with seed feedback ({len(seed_feedback)} chars)")

        for iteration in range(1, config.max_iterations + 1):
            logger.info(f"=== Iteration {iteration}/{config.max_iterations} ===")

            logger.info("Generating signals...")
            generation_request = request_text
            if config.fundamentals_only:
                generation_request += (
                    "\n\nHARD CONSTRAINT: every signal must be built from the FUNDAMENTALS "
                    "and RATIOS panels above. A signal whose formula uses only price, "
                    "volume or returns is invalid and is discarded without evaluation."
                    "\nFundamentals are reported quarterly and forward-filled monthly. "
                    "Keep every lookback window at 12 months or fewer: a 60-month "
                    "rolling window on a panel that begins in 2016 has no values to "
                    "evaluate before 2021, so the candidate cannot be tested."
                )
            live_parents = [
                {
                    "name": m["name"],
                    "formula": m["formula"],
                    "val_ic": m["val_ic"],
                    "val_sharpe": m.get("val_sharpe"),
                }
                for m in (alpha_pool.members if alpha_pool is not None else [])
            ]
            parent_archive = (live_parents + archived_parents)[:6] or None
            signal_json = await generate_signal_json(
                signal_llm,
                generation_request,
                config.num_signals,
                operators,
                feedback,
                tried_formulas=(tried + [t.get("formula", "") for t in trials_this_run]) if config.dedupe_trials else None,
                horizon=config.forward_periods,
                available_fields=set(stock_data),
                parent_signals=parent_archive,
                periods_per_year=config.periods_per_year,
                residualize_returns=config.residualize_returns,
            )
            metadata = _signal_metadata(signal_json)

            logger.info("Generating code...")
            codegen_errors: list[str] = []
            signal_code = await generate_signal_code(
                code_llm, signal_json, operators, errors_out=codegen_errors
            )

            logger.info("Evaluating every candidate...")
            clean_code = extract_code_from_response(signal_code)
            candidates_raw = execute_signal_functions(clean_code, stock_data)
            if not candidates_raw:
                last_error = "; ".join(codegen_errors) or "code execution produced no signal functions"
                logger.warning(last_error)

            eligible_raw: list[tuple[str, pd.DataFrame, dict[str, Any], str]] = []
            for name, values in candidates_raw:
                meta = metadata.get(name, {})
                if config.fundamentals_only and not uses_fundamental_fields(meta.get("formula")):
                    logger.info(
                        f"  {name}: price-only formula {meta.get('formula')!r}, discarded "
                        "by the fundamentals mandate"
                    )
                    continue
                key = _trial_key(meta)
                if config.dedupe_trials and key in ledger_keys:
                    logger.info(f"  {name}: already in the ledger at this horizon, skipping")
                    duplicate_skipped += 1
                    continue
                if key in seen_this_run:
                    logger.info(f"  {name}: duplicate within this run, skipping")
                    duplicate_skipped += 1
                    continue
                seen_this_run.add(key)
                eligible_raw.append((name, values, meta, key))

            eval_workers = max(1, min(int(config.eval_workers), len(eligible_raw) or 1))
            if eval_workers > 1 and len(eligible_raw) > 1:
                with ThreadPoolExecutor(max_workers=eval_workers) as eval_pool:
                    ic_list = list(
                        eval_pool.map(lambda item: evaluate_candidate(item[1], item[0], item[2]), eligible_raw)
                    )
            else:
                ic_list = [evaluate_candidate(v, n, m) for n, v, m, _ in eligible_raw]

            candidates: list[dict] = []
            for (name, values, meta, key), ic in zip(eligible_raw, ic_list):
                if ic.get("error"):
                    last_error = str(ic["error"])
                train_ic = (ic.get("ic_train") or {}).get("mean_ic")
                validation_ic = ic.get("mean_ic")
                holdout_ic = (ic.get("ic_test") or {}).get("mean_ic")

                # Direction and stability are decided on train/validation only.
                # The holdout is recorded for context, never used to flip or
                # reject a signal.
                direction = (
                    1.0
                    if (train_ic if train_ic is not None else validation_ic or 0.0) >= 0
                    else -1.0
                )
                sign_consistent = (
                    train_ic is not None
                    and validation_ic is not None
                    and train_ic * validation_ic > 0
                )

                stats_ok = passes_statistics(ic)
                fdr: dict[str, Any] = {}
                validation_screen: dict[str, Any] = {}
                validation_sharpe = None
                validation_monotonicity = None
                validation_spread_t = None
                validation_alpha_t = None
                validation_alpha_ann = None
                if stats_ok and config.require_fdr_significance:
                    fdr = fdr_annotation(ic.get("p_value"))
                validation_dsr = None
                validation_stability = None
                pool_eligible = bool(
                    alpha_pool is not None
                    and sign_consistent
                    and validation_ic is not None
                    and abs(validation_ic) >= config.pool_min_ic
                    and not ic.get("insufficient_history")
                )
                if (
                    stats_ok
                    and (
                        config.min_net_sharpe > 0
                        or config.min_alpha_tstat > 0
                        or config.min_dsr > 0
                        or config.min_subperiod_stability > 0
                    )
                ) or pool_eligible:
                    validation_screen = screen_on_validation(values, direction)
                    validation_sharpe = (validation_screen.get("net") or {}).get("sharpe")
                    validation_monotonicity = validation_screen.get("monotonicity")
                    validation_spread_t = validation_screen.get("spread_t_stat")
                    fa = validation_screen.get("factor_alpha") or {}
                    validation_alpha_t = fa.get("alpha_t_stat")
                    validation_alpha_ann = fa.get("alpha_annualized")
                    validation_dsr = (validation_screen.get("deflated_sharpe") or {}).get("dsr")
                    validation_stability = (validation_screen.get("subperiod_stability") or {}).get("positive_block_ratio")

                alpha_gate_active = (
                    config.min_alpha_tstat > 0
                    and factor_returns is not None
                    and not factor_returns.empty
                )
                accepted = bool(
                    stats_ok
                    and sign_consistent
                    and (
                        not config.require_fdr_significance
                        or fdr.get("significant_fdr", False)
                    )
                    and (
                        config.min_net_sharpe <= 0
                        or (
                            validation_sharpe is not None
                            and validation_sharpe >= config.min_net_sharpe
                        )
                    )
                    and (
                        config.min_monotonicity <= 0
                        or (
                            validation_monotonicity is not None
                            and validation_monotonicity >= config.min_monotonicity
                        )
                    )
                    and (
                        config.min_spread_tstat <= 0
                        or (
                            validation_spread_t is not None
                            and validation_spread_t >= config.min_spread_tstat
                        )
                    )
                    and (
                        not alpha_gate_active
                        or (
                            validation_alpha_t is not None
                            and validation_alpha_t >= config.min_alpha_tstat
                        )
                    )
                    and (
                        config.min_dsr <= 0
                        or (validation_dsr is not None and validation_dsr >= config.min_dsr)
                    )
                    and (
                        config.min_subperiod_stability <= 0
                        or (validation_stability is not None and validation_stability >= config.min_subperiod_stability)
                    )
                )

                trial = {
                    "sign_consistent": sign_consistent,
                    "run_id": request_text[:80],
                    "iteration": iteration,
                    "name": name,
                    "display_name": meta.get("display_name"),
                    "formula": meta.get("formula"),
                    "category": meta.get("category"),
                    "trial_key": key,
                    "horizon": config.forward_periods,
                    "mean_ic": validation_ic,
                    "ic_std": ic.get("ic_std"),
                    "t_stat": ic.get("t_stat"),
                    "p_value": ic.get("p_value"),
                    "q_value": fdr.get("q_value"),
                    "significant_fdr": fdr.get("significant_fdr"),
                    "t_stat_naive": ic.get("t_stat_naive"),
                    "p_value_naive": ic.get("p_value_naive"),
                    "train_ic": train_ic,
                    "holdout_ic": holdout_ic,
                    "num_periods": ic.get("num_periods"),
                    "validation_net_sharpe": validation_sharpe,
                    "validation_monotonicity": validation_monotonicity,
                    "validation_spread_tstat": validation_spread_t,
                    "validation_alpha_tstat": validation_alpha_t,
                    "validation_alpha_annualized": validation_alpha_ann,
                    "direction": direction,
                    "accepted": accepted,
                    "selected_on": ic.get("selected_on"),
                    "data_pin_policy": pin_policy,
                }
                registry.record(trial)
                cached_trials.append(trial)
                trials_this_run.append(trial)
                ledger_keys.add(key)

                candidate = {
                    "name": name,
                    "meta": meta,
                    "ic": ic,
                    "accepted": accepted,
                    "values": values,
                    "validation_screen": validation_screen,
                    "signal_json": signal_json,
                    "signal_code": signal_code,
                    "iteration": iteration,
                }
                candidates.append(candidate)
                all_candidates.append(candidate)

                if (
                    alpha_pool is not None
                    and sign_consistent
                    and validation_ic is not None
                    and abs(validation_ic) >= config.pool_min_ic
                    and not ic.get("insufficient_history")
                    and (validation_monotonicity is None or validation_monotonicity >= 0.20)
                ):
                    admitted, max_c, reason = alpha_pool.try_admit(
                        name=name,
                        formula=meta.get("formula"),
                        values=_apply_universe_mask(values),
                        direction=direction,
                        train_ic=train_ic,
                        val_ic=validation_ic,
                        val_p=ic.get("p_value"),
                        val_sharpe=validation_sharpe,
                        val_monotonicity=validation_monotonicity,
                    )
                    logger.info(f"  AlphaPool {name}: {reason} (max_corr={max_c:.2f}, size={len(alpha_pool.members)})")

                logger.info(
                    "  {name}: validation IC={ic}, train IC={tr}, test IC={te}, "
                    "HAC p={p}, q={q}, val net Sharpe={sharpe}, {verdict}".format(
                        name=name,
                        ic=f"{validation_ic:.4f}" if validation_ic is not None else "n/a",
                        tr=f"{train_ic:.4f}" if train_ic is not None else "n/a",
                        te=f"{holdout_ic:.4f}" if holdout_ic is not None else "n/a",
                        p=f"{ic['p_value']:.4f}" if ic.get("p_value") is not None else "n/a",
                        q=f"{fdr['q_value']:.4f}" if fdr.get("q_value") is not None else "n/a",
                        sharpe=(
                            f"{validation_sharpe:.2f}"
                            if validation_sharpe is not None
                            else "n/a"
                        ),
                        verdict="ACCEPT" if accepted else "reject",
                    )
                )

                if validation_ic is not None:
                    if best_result is None or candidate_key(candidate) > candidate_key(best_result):
                        best_ic = validation_ic
                        best_result = candidate

                if accepted:
                    accepted_candidates.append(candidate)
                    logger.info(f"Signal accepted on validation: {name}")
                    if config.stop_on_acceptance:
                        return _finish(
                            candidate, "accepted", request_text, feedback,
                            all_candidates, trials_this_run,
                        )

            # Structural failures skip the advisor round-trip: there are no IC
            # numbers to discuss, and the parse errors are the useful feedback.
            if not candidates and codegen_errors:
                last_error = "; ".join(codegen_errors)
                advice = (
                    "Your previous signals were ALL REJECTED before evaluation. "
                    "Fix exactly the failures listed below and do not resubmit the "
                    "same formula. Note that a ratio whose denominator is a "
                    "cross-sectional operator (e.g. Div(x, CS_Demean(y))) is rejected "
                    "as unstable: divide by the raw field first, then demean.\n"
                    + "\n".join(f"- {e}" for e in codegen_errors)
                )
                logger.info(f"Skipping advisor: {len(codegen_errors)} structural errors")
            elif not candidates:
                advice = _no_candidate_advice(duplicate_skipped)
            else:
                logger.info("Generating optimization feedback...")
                advice = await generate_feedback(candidates, iteration)

            feedback = _compose_feedback(advice, best_result, best_ic)

        if alpha_pool is not None and alpha_pool.members and pool_library is not None:
            try:
                pool_library.save_members(
                    alpha_pool.members,
                    run_id=request_text[:80],
                    horizon=config.forward_periods,
                )
            except Exception as e:
                logger.warning(f"Could not persist AlphaPool shard: {e}")

        if alpha_pool is not None and len(alpha_pool.members) >= 2:
            try:
                meta_values, meta_details = alpha_pool.combine_heterogeneous(
                    forward_returns,
                    train_end_date=config.train_end_date,
                    ridge_alpha=config.pool_ridge_alpha,
                    gbdt_weight=config.pool_gbdt_weight,
                )
                meta_weights = meta_details.get("ridge_weights") or {}
                stack_backend = meta_details.get("backend", "ridge")
                meta_formula = " + ".join(
                    f"{wt:.3f}*{n}" for n, wt in meta_weights.items() if wt > 0
                )
                meta_info = {
                    "display_name": f"Meta-Alpha Heterogeneous Ensemble ({stack_backend})",
                    "formula": meta_formula,
                    "category": "meta_ensemble",
                    "meaning": f"Train-only Ridge+{stack_backend} combination of {len(alpha_pool.members)} orthogonal signals",
                    "weights": meta_weights,
                    "stacker": meta_details,
                }
                meta_ic = evaluate_candidate(meta_values, "meta_alpha_ensemble", meta_info)
                meta_train = (meta_ic.get("ic_train") or {}).get("mean_ic")
                meta_val = meta_ic.get("mean_ic")
                meta_dir = 1.0 if (meta_train if meta_train is not None else meta_val or 0.0) >= 0 else -1.0
                meta_consistent = (
                    meta_train is not None and meta_val is not None and meta_train * meta_val > 0
                )
                meta_stats = passes_statistics(meta_ic)
                meta_fdr = fdr_annotation(meta_ic.get("p_value")) if (meta_stats and config.require_fdr_significance) else {}
                meta_screen = screen_on_validation(meta_values, meta_dir)
                m_sh = (meta_screen.get("net") or {}).get("sharpe")
                m_mono = meta_screen.get("monotonicity")
                m_st = meta_screen.get("spread_t_stat")
                m_fa = meta_screen.get("factor_alpha") or {}
                m_at = m_fa.get("alpha_t_stat")
                m_dsr = (meta_screen.get("deflated_sharpe") or {}).get("dsr")
                m_stab = (meta_screen.get("subperiod_stability") or {}).get("positive_block_ratio")
                alpha_gate_active = (
                    config.min_alpha_tstat > 0
                    and factor_returns is not None
                    and not factor_returns.empty
                )
                meta_accepted = bool(
                    meta_stats
                    and meta_consistent
                    and (not config.require_fdr_significance or meta_fdr.get("significant_fdr", False))
                    and (config.min_net_sharpe <= 0 or (m_sh is not None and m_sh >= config.min_net_sharpe))
                    and (config.min_monotonicity <= 0 or (m_mono is not None and m_mono >= config.min_monotonicity))
                    and (config.min_spread_tstat <= 0 or (m_st is not None and m_st >= config.min_spread_tstat))
                    and (not alpha_gate_active or (m_at is not None and m_at >= config.min_alpha_tstat))
                    and (config.min_dsr <= 0 or (m_dsr is not None and m_dsr >= config.min_dsr))
                    and (config.min_subperiod_stability <= 0 or (m_stab is not None and m_stab >= config.min_subperiod_stability))
                )
                meta_trial = {
                    "sign_consistent": meta_consistent,
                    "run_id": request_text[:80],
                    "iteration": config.max_iterations,
                    "name": "meta_alpha_ensemble",
                    "display_name": meta_info["display_name"],
                    "formula": meta_formula,
                    "category": "meta_ensemble",
                    "trial_key": f"meta_{stack_backend}|{len(alpha_pool.members)}|h={config.forward_periods}",
                    "horizon": config.forward_periods,
                    "mean_ic": meta_val,
                    "p_value": meta_ic.get("p_value"),
                    "q_value": meta_fdr.get("q_value"),
                    "train_ic": meta_train,
                    "holdout_ic": (meta_ic.get("ic_test") or {}).get("mean_ic"),
                    "validation_net_sharpe": m_sh,
                    "validation_monotonicity": m_mono,
                    "validation_spread_tstat": m_st,
                    "validation_alpha_tstat": m_at,
                    "validation_alpha_annualized": m_fa.get("alpha_annualized"),
                    "direction": meta_dir,
                    "accepted": meta_accepted,
                    "pool_weights": meta_weights,
                    "stacker_backend": stack_backend,
                }
                registry.record(meta_trial)
                trials_this_run.append(meta_trial)
                meta_cand = {
                    "name": "meta_alpha_ensemble",
                    "meta": meta_info,
                    "ic": meta_ic,
                    "accepted": meta_accepted,
                    "values": meta_values,
                    "validation_screen": meta_screen,
                    "signal_json": json.dumps([meta_info]),
                    "signal_code": f"# Train-only Ridge+{stack_backend} meta-alpha ({len(alpha_pool.members)} members)\n# Weights: {json.dumps(meta_weights)}",
                    "iteration": config.max_iterations,
                }
                all_candidates.append(meta_cand)
                if meta_accepted:
                    accepted_candidates.append(meta_cand)
                logger.info(
                    f"Meta-Alpha ({stack_backend}, {len(alpha_pool.members)} signals): val IC={meta_val}, "
                    f"val Sharpe={m_sh}, alpha_t={m_at}, accepted={meta_accepted}"
                )
            except Exception as e:
                logger.warning(f"Meta-alpha heterogeneous combination failed: {e}")

        if accepted_candidates or all_candidates:
            # With stop_on_acceptance off, pick the strongest accepted signal;
            # otherwise the strongest eligible candidate, where "strongest" is
            # a train/validation ordering that never looks at the holdout.
            pool = accepted_candidates or all_candidates
            eligible = [
                c for c in pool if not (c.get("ic") or {}).get("insufficient_history")
            ]
            chosen = max(eligible or pool, key=candidate_key)
            status = "accepted" if chosen.get("accepted") else "best_effort"
            return _finish(
                chosen, status, request_text, feedback, all_candidates, trials_this_run
            )

        return _format_workflow_result(
            status="failed",
            request=request_text,
            iteration=0,
            total_iterations=config.max_iterations,
            signal_json="",
            ic_results={"error": last_error} if last_error else {},
            saved_path=None,
            config=config,
            last_feedback=feedback,
        )

    def _finish(
        chosen: dict,
        status: str,
        request_text: str,
        feedback: str | None,
        all_candidates: list[dict],
        trials_this_run: list[dict],
    ) -> str:
        """Persist and format the chosen candidate."""
        train_ic = ((chosen.get("ic") or {}).get("ic_train") or {}).get("mean_ic")
        # Match the loop's direction rule exactly: train IC when it exists,
        # otherwise validation IC. Defaulting to +1 here silently flipped the
        # sign of the reported full-history screen relative to the screen the
        # candidate was accepted on.
        anchor_ic = train_ic
        if anchor_ic is None:
            anchor_ic = (chosen.get("ic") or {}).get("mean_ic")
        direction = 1.0 if (anchor_ic if anchor_ic is not None else 0.0) >= 0 else -1.0
        extra = _final_extras(
            chosen.get("values"),
            chosen["ic"],
            all_candidates,
            trials_this_run,
            registry,
            direction=direction,
        )
        saved_path = (
            save_signal_results(
                chosen["signal_json"],
                chosen["signal_code"],
                chosen["ic"],
                chosen["iteration"],
                extra=extra,
                output_uri=results_uri,
            )
            if config.save_results
            else None
        )
        phf_runs_uri = config.phf_runs_uri or os.environ.get("QSD_PHF_RUNS_URI")
        if status == "accepted" and phf_runs_uri:
            try:
                extra["phf_export"] = export_phf_run(
                    phf_runs_uri,
                    signal_name=chosen["name"],
                    formula=(chosen.get("meta") or {}).get("formula"),
                    request_text=request_text,
                    ic_results=chosen["ic"],
                    screen_validation=extra.get("screen_validation") or {},
                    screen_full=extra.get("screen") or {},
                    factor_alpha=(extra.get("screen_validation") or {}).get("factor_alpha"),
                    data_pins=data_pins,
                    config_summary={
                        "min_price": config.min_price,
                        "min_dollar_volume": config.min_dollar_volume,
                        "min_dollar_volume_pct": config.min_dollar_volume_pct,
                        "top_n_market_cap": config.top_n_market_cap,
                        "cost_bps": config.cost_bps,
                        "short_borrow_bps": config.short_borrow_bps,
                        "impact_coeff": config.impact_coeff,
                        "aum_usd": config.aum_usd,
                        "vol_scaled": config.vol_scaled,
                        "max_weight": config.max_weight,
                        "holding_periods": config.forward_periods,
                        "screen_quantile": config.screen_quantile,
                        "screen_buffer_quantile": config.screen_buffer_quantile,
                        "min_net_sharpe": config.min_net_sharpe,
                        "min_monotonicity": config.min_monotonicity,
                        "min_spread_tstat": config.min_spread_tstat,
                        "min_alpha_tstat": config.min_alpha_tstat,
                        "min_dsr": config.min_dsr,
                        "min_subperiod_stability": config.min_subperiod_stability,
                    },
                    signal_artifact_uri=saved_path,
                )
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"PHF export failed: {e}")
                extra["phf_export_error"] = str(e)
        return _format_workflow_result(
            status=status,
            request=request_text,
            iteration=chosen["iteration"],
            total_iterations=config.max_iterations,
            signal_json=chosen["signal_json"],
            ic_results=chosen["ic"],
            saved_path=saved_path,
            config=config,
            last_feedback=feedback,
            extra=extra,
        )

    def _final_extras(
        values,
        ic: dict,
        candidates: list[dict],
        trials_this_run: list[dict],
        registry: TrialRegistry,
        direction: float = 1.0,
    ) -> dict[str, Any]:
        """Screen + decay + ledger context attached to the chosen signal.

        The screen and decay use all available history (they are descriptive,
        not selection inputs); the holdout IC stays the untouched test window.
        """
        summary = registry.summary()
        extras: dict[str, Any] = {
            "data_pins": data_pins,
            "data_resolution": resolved_releases,
            "pin_policy": pin_policy,
            "selection_basis": "train/validation only; holdout never used to select",
            "universe_filter": {
                "min_price": config.min_price,
                "min_dollar_volume": config.min_dollar_volume,
                # The percentile floor is what actually defines a multi-decade
                # universe; omitting it made the run irreproducible from its
                # own payload (0.0 nominal reads as "no volume filter").
                "min_dollar_volume_pct": config.min_dollar_volume_pct,
                "top_n_market_cap": config.top_n_market_cap,
                "min_market_cap": config.min_market_cap,
            },
            "data_coverage": {
                name: {
                    "first": str(frame.index.min().date()) if len(frame.index) else None,
                    "last": str(frame.index.max().date()) if len(frame.index) else None,
                    "cells": int(frame.notna().sum().sum()),
                }
                for name, frame in stock_data.items()
                if hasattr(frame, "index") and len(frame.index)
            },
            "trials": {
                "this_run": len(trials_this_run),
                "ledger_total": summary["num_trials"],
                "fdr": summary["multiple_testing"],
                "leaderboard": registry.leaderboard(top=5),
            },
        }
        if values is None:
            extras["screen"] = {"error": "signal values unavailable to recompute the screen"}
            return extras
        masked = _apply_universe_mask(values)
        try:
            raw = compute_ic_split(
                masked,
                forward_returns,
                train_end_date=config.train_end_date,
                test_start_date=config.test_start_date,
                sector_map=sector_map,
                validation_end_date=config.validation_end_date,
                purge_periods=purge,
                periods=config.forward_periods,
                min_names=config.min_names,
                periods_per_year=config.periods_per_year,
                include_raw=True,
            )
            for key in (
                "ic_raw_train", "ic_raw_validation", "ic_raw_test", "ic_raw_full"
            ):
                if raw.get(key):
                    ic[key] = raw[key]
        except Exception as e:  # pragma: no cover - defensive
            extras["raw_ic_error"] = str(e)

        try:
            extras["screen_validation"] = screen_on_validation(values, direction)
        except Exception as e:  # pragma: no cover - defensive
            extras["screen_validation"] = {"error": str(e)}
        try:
            extras["screen"] = compute_long_short_screen(
                masked,
                close,
                quantile=config.screen_quantile,
                buffer_quantile=config.screen_buffer_quantile,
                cost_bps=config.cost_bps,
                min_names=max(config.min_names, 20),
                annualization=config.periods_per_year,
                direction=direction,
                sector_map=sector_map,
                short_borrow_bps=config.short_borrow_bps,
                impact_coeff=config.impact_coeff,
                volume_panel=stock_data.get("Volume"),
                aum_usd=config.aum_usd,
                vol_scaled=config.vol_scaled,
                max_weight=config.max_weight,
                holding_periods=config.forward_periods,
            )
        except Exception as e:  # pragma: no cover - defensive
            extras["screen"] = {"error": str(e)}
        try:
            extras["ic_decay"] = compute_ic_decay(
                masked,
                close,
                horizons=decay_horizons or (1, 5, 10, 21, 63),
                sector_map=sector_map,
                min_names=config.min_names,
            )
        except Exception as e:  # pragma: no cover - defensive
            extras["ic_decay"] = {"error": str(e)}
        return extras

    yield FunctionInfo.from_fn(
        run_optimization,
        description="Run signal discovery optimization loop: generate -> code -> evaluate -> feedback.",
    )
