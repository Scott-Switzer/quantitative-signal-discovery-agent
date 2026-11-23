# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0

"""Run the discovery loop on Modal: cloud compute, R2 read and written in place.

Phase 2 architecture (decoupled from deprecated nvidia-nat CLI):
  - Uses ``python -m signal_discovery_workflow.runner`` directly against NIM.
  - 8-CPU / 16 GB containers with LightGBM for heterogeneous Ridge+GBDT stacking.
  - ``ingest_sec_form4`` & ``ingest_sec_13f`` ingest free SEC EDGAR bulk zips
    straight into R2 with zero local disk staging.
  - ``build_feature_store`` precomputes aligned monthly & weekly Parquet bundles
    (including Lazy Prices, Form 4, and Form 13F when present).
  - ``fanout_discovery`` runs 6 orthogonal themes in parallel via ``.map()``.
  - ``run_cross_run_meta_combiner`` stacks the top orthogonal signals accumulated
    across all themes/runs in ``AlphaPoolLibrary`` with DSR + JKP alpha gates.

Secrets:
  qsd-r2    R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY / R2_S3_ENDPOINT
  nim-api   NVIDIA_API_KEY
"""

from __future__ import annotations

import subprocess

import modal

APP_NAME = "qsd-signal-discovery"
REMOTE_ROOT = "/root/qsd"
RESULTS_URI = "r2://financial-data-wrds-lakehouse/research/signals"

SOURCE_IGNORE = [
    ".git/**",
    ".venv/**",
    "**/__pycache__/**",
    "**/*.pyc",
    ".pytest_cache/**",
    ".ruff_cache/**",
    "notebooks/**",
    "output/**",
    "brev/**",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    # Install into the interpreter Modal runs functions with, not a venv the
    # function process cannot see. Without this, every direct project import
    # (ingest/build/meta jobs) fails with ModuleNotFoundError.
    .env({"UV_PROJECT_ENVIRONMENT": "/usr/local"})
    .apt_install("git", "build-essential", "libgomp1")
    .pip_install("uv", "pyyaml")
    .add_local_file("pyproject.toml", f"{REMOTE_ROOT}/pyproject.toml", copy=True)
    .add_local_file("uv.lock", f"{REMOTE_ROOT}/uv.lock", copy=True)
    .add_local_file("README.md", f"{REMOTE_ROOT}/README.md", copy=True)
    .run_commands(f"cd {REMOTE_ROOT} && uv sync --frozen --no-install-project")
    .add_local_dir(".", remote_path=REMOTE_ROOT, ignore=SOURCE_IGNORE, copy=True)
    .run_commands(f"cd {REMOTE_ROOT} && uv sync --frozen")
    # Install after the final sync, which otherwise prunes LightGBM and
    # silently downgrades the stacker to its fallback backend.
    .run_commands(f"cd {REMOTE_ROOT} && uv pip install --system lightgbm")
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("qsd-r2"), modal.Secret.from_name("nim-api")],
    timeout=60 * 60,
    cpu=8.0,
    memory=16384,
)
def run_loop(
    request: str,
    config_file: str = "configs/config-crsp-fundamentals.yml",
    max_iterations: int | None = None,
    num_signals: int | None = None,
) -> int:
    """Run the loop via the lean plain-Python runner (no deprecated nat CLI)."""
    import os
    import sys

    env = dict(os.environ)
    env["QSD_RESULTS_URI"] = RESULTS_URI
    env["PYTHONUNBUFFERED"] = "1"

    # Invoke the /usr/local Python directly (never `uv run`, which runs an
    # implicit `uv sync` against uv.lock and prunes LightGBM).
    cmd = [
        sys.executable,
        "-m",
        "signal_discovery_workflow.runner",
        "--config_file",
        config_file,
        "--input",
        request,
    ]
    if max_iterations is not None:
        cmd.extend(["--max-iterations", str(int(max_iterations))])
    if num_signals is not None:
        cmd.extend(["--num-signals", str(int(num_signals))])

    process = subprocess.Popen(
        cmd,
        cwd=REMOTE_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    return process.wait()


FANOUT_THEMES = [
    ("configs/config-crsp-fundamentals.yml", "cash-flow value, FcfYield, EbitToEv, and cash-backed quality signals"),
    ("configs/config-crsp-fundamentals.yml", "quarterly EarningsAccel, RevenueAccel, MarginDelta, and QuarterlyRoeDelta momentum"),
    ("configs/config-crsp-fundamentals.yml", "discretionary Form 4 InsiderNetBuyRatio, InsiderClusterBuys, and DiscretionaryBuyRatio crossed with quality"),
    ("configs/config-crsp-fundamentals.yml", "SEC 13F InstOwnershipDelta and low InstCrowding crossed with value and macro regimes"),
    ("configs/config-crsp-fundamentals.yml", "Lazy Prices FilingTextChange, FilingJaccardDiff, and FilingLengthDelta crossed with quality"),
    ("configs/config-crsp-quarterly-horizon.yml", "3-month horizon deep value, BookToMarket, FcfYield, and low Leverage signals"),
    ("configs/config-crsp-weekly.yml", "weekly AmihudIlliq, low IdioVol, High52wRatio, and VolumeShock crossed with FcfYield and EarningsAccel"),
]


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("qsd-r2")],
    timeout=2 * 60 * 60,
    cpu=4.0,
    memory=8192,
)
def ingest_sec_form4(start_year: int = 2010, end_year: int | None = None) -> dict:
    """Ingest free SEC EDGAR Form 4 quarterly bulk zips straight into R2."""
    import sys

    sys.path.insert(0, f"{REMOTE_ROOT}/src")
    from signal_discovery_workflow.alt_data import ingest_sec_form4_to_r2

    result = ingest_sec_form4_to_r2(start_year=start_year, end_year=end_year)
    print(f"Form 4 ingest complete: {result}")
    return result


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("qsd-r2")],
    timeout=4 * 60 * 60,
    cpu=4.0,
    memory=16384,
)
def ingest_sec_13f(start_year: int = 2013, end_year: int | None = None) -> dict:
    """Ingest free SEC EDGAR Form 13F quarterly bulk zips straight into R2."""
    import sys

    sys.path.insert(0, f"{REMOTE_ROOT}/src")
    from signal_discovery_workflow.alt_data import ingest_sec_13f_to_r2

    result = ingest_sec_13f_to_r2(start_year=start_year, end_year=end_year)
    print(f"Form 13F ingest complete: {result}")
    return result


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("qsd-r2")],
    timeout=60 * 60,
    cpu=8.0,
    memory=16384,
)
def build_feature_store(sleeve: str = "monthly") -> dict:
    """Precompute and save an R2 feature-store snapshot for fast (~3-5s) discovery runs."""
    import pathlib
    import sys

    import yaml

    sys.path.insert(0, f"{REMOTE_ROOT}/src")
    from signal_discovery_workflow import pins as pin_store
    from signal_discovery_workflow.alt_data import load_13f_panels, load_form4_panels
    from signal_discovery_workflow.feature_store import (
        DEFAULT_MONTHLY_FEATURE_STORE_URI,
        DEFAULT_WEEKLY_FEATURE_STORE_URI,
        save_feature_store,
    )
    from signal_discovery_workflow.fundamentals import (
        load_lakehouse_v2_fundamentals_panels,
        load_sec_quarterly_acceleration_panels,
    )
    from signal_discovery_workflow.lazy_prices import load_lazy_prices_panels
    from signal_discovery_workflow.macro_features import load_macro_panels
    from signal_discovery_workflow.phf_bridge import (
        build_tradable_universe_mask,
        load_crsp_weekly_panel,
        load_phf_panel,
    )
    from signal_discovery_workflow.residualize import residualize_forward_returns
    from signal_discovery_workflow.signal_evaluator import compute_forward_returns

    cfg_path = (
        "configs/config-crsp-weekly.yml"
        if sleeve == "weekly"
        else "configs/config-crsp-fundamentals.yml"
    )
    wf = yaml.safe_load((pathlib.Path(REMOTE_ROOT) / cfg_path).read_text())["workflow"]
    store_uri = wf.get("feature_store_uri") or (
        DEFAULT_WEEKLY_FEATURE_STORE_URI if sleeve == "weekly" else DEFAULT_MONTHLY_FEATURE_STORE_URI
    )

    source_pins: dict[str, dict] = {}
    if sleeve == "weekly":
        panels, sector_map = load_crsp_weekly_panel(
            wf["weekly_crsp_uri"],
            start=wf.get("weekly_start_date", "2005-01-01"),
            end=wf.get("weekly_end_date"),
        )
        source_pins["weekly_crsp"] = {
            "uri": wf["weekly_crsp_uri"],
            "start": wf.get("weekly_start_date", "2005-01-01"),
            "end": wf.get("weekly_end_date"),
        }
    else:
        rel = pin_store.resolve_release(wf["data_release_prefix"], "panel.parquet", "manifest.json")
        panels, sector_map = load_phf_panel(rel["artifact_uri"])
        try:
            source_pins["market_data"] = pin_store.pin(rel["manifest_uri"], role="market_data")
        except Exception:
            pass

    close = panels["Close"]
    bridge_uri = wf.get("cik_bridge_uri", "")
    if wf.get("lakehouse_v2_uri"):
        panels.update(
            load_lakehouse_v2_fundamentals_panels(
                uri=wf["lakehouse_v2_uri"],
                dates=close.index,
                securities=close.columns,
                market_cap=panels.get("MarketCap"),
            )
        )
    if wf.get("sec_quarterly_uri") and bridge_uri:
        panels.update(
            load_sec_quarterly_acceleration_panels(
                quarterly_uri=wf["sec_quarterly_uri"],
                bridge_uri=bridge_uri,
                dates=close.index,
                securities=close.columns,
            )
        )
    for key, loader_fn, label in (
        ("lazy_prices_uri", load_lazy_prices_panels, "Lazy Prices"),
        ("form4_uri", load_form4_panels, "Form 4"),
        ("f13_uri", load_13f_panels, "Form 13F"),
    ):
        uri = wf.get(key)
        if uri and bridge_uri:
            try:
                panels.update(loader_fn(uri, bridge_uri, close.index, close.columns))
            except Exception as e:
                print(f"Skipping {label} (not built yet): {e}")

    if wf.get("fred_macro_uri"):
        panels.update(
            load_macro_panels(
                fred_uri=wf["fred_macro_uri"],
                dates=close.index,
                securities=close.columns,
                lag_periods=1,
            )
        )

    mask = build_tradable_universe_mask(
        panels,
        min_price=wf.get("min_price", 0.0),
        min_dollar_volume=wf.get("min_dollar_volume", 0.0),
        min_dollar_volume_pct=wf.get("min_dollar_volume_pct", 0.0),
        top_n_market_cap=wf.get("top_n_market_cap", 0),
        min_market_cap=wf.get("min_market_cap", 0.0),
    )
    panels.pop("Price", None)

    fwd = compute_forward_returns(close, periods=int(wf.get("forward_periods", 1)))
    resid = residualize_forward_returns(
        fwd,
        close,
        market_cap=panels.get("MarketCap"),
        sector_map=sector_map or None,
        periods_per_year=int(wf.get("periods_per_year", 12)),
        min_names=max(int(wf.get("min_names", 100)), 30),
    )
    from signal_discovery_workflow import r2_store

    if wf.get("fred_macro_uri", "").endswith(".json"):
        try:
            source_pins["fred_macro"] = pin_store.pin(wf["fred_macro_uri"], role="fred_macro")
        except Exception:
            pass
    for pin_key in ("lazy_prices_uri", "form4_uri", "f13_uri", "cik_bridge_uri"):
        uri = wf.get(pin_key)
        if uri:
            try:
                info = r2_store.stat(uri)
                if info:
                    source_pins[pin_key] = {"uri": uri, "size": info["size"], "mtime": info["mtime"]}
            except Exception:
                pass

    manifest = save_feature_store(
        store_uri,
        panels,
        sector_map=sector_map,
        universe_mask=mask,
        residual_returns=resid,
        metadata={"sleeve": sleeve, "config": cfg_path, "source_pins": source_pins},
    )
    print(f"Feature store ({sleeve}) ready: {manifest['manifest_uri']} ({manifest['num_panels']} panels)")
    return manifest


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("qsd-r2")],
    timeout=60 * 60,
    cpu=8.0,
    memory=16384,
)
def run_cross_run_meta_combiner(
    config_file: str = "configs/config-crsp-fundamentals.yml",
    max_members: int = 20,
) -> dict:
    """Offline cross-run Meta-Combiner: stack orthogonal signals from AlphaPoolLibrary."""
    import pathlib
    import sys

    import pandas as pd
    import yaml

    sys.path.insert(0, f"{REMOTE_ROOT}/src")
    from signal_discovery_workflow import r2_store
    from signal_discovery_workflow.alpha_combiner import AlphaPool, AlphaPoolLibrary
    from signal_discovery_workflow.feature_store import load_feature_store
    from signal_discovery_workflow.macro_features import compute_factor_alpha, load_factor_returns
    from signal_discovery_workflow.overfitting import deflated_sharpe_ratio, subperiod_stability
    from signal_discovery_workflow.phf_export import export_phf_run
    from signal_discovery_workflow.signal_code_generator import (
        _infer_fields_from_formula,
        assemble_module,
        collect_operator_code,
    )
    from signal_discovery_workflow.signal_evaluator import (
        compute_forward_returns,
        compute_ic_split,
        compute_long_short_screen,
        execute_signal_functions,
    )
    from signal_discovery_workflow.signal_generator import get_operator_code_map, load_calculator_operators
    from signal_discovery_workflow.trial_registry import TrialRegistry

    wf = yaml.safe_load((pathlib.Path(REMOTE_ROOT) / config_file).read_text())["workflow"]
    pool_uri = wf.get("alpha_pool_uri") or r2_store.join(RESULTS_URI, "alpha_pool")
    lib = AlphaPoolLibrary(pool_uri)
    horizon = int(wf.get("forward_periods", 1))
    ppy = int(wf.get("periods_per_year", 12))
    archived = lib.load_all(horizon=horizon, periods_per_year=ppy)
    if len(archived) < 2:
        return {"status": "insufficient_pool_members", "archived_count": len(archived)}

    bundle = load_feature_store(wf["feature_store_uri"])
    panels = bundle["panels"]
    sector_map = bundle.get("sector_map") or None
    mask = bundle.get("universe_mask")
    close = panels["Close"]
    if horizon == 1 and wf.get("residualize_returns", True) and bundle.get("residual_returns") is not None:
        fwd = bundle["residual_returns"]
    else:
        raw_fwd = compute_forward_returns(close, periods=horizon)
        if wf.get("residualize_returns", True):
            from signal_discovery_workflow.residualize import residualize_forward_returns

            fwd = residualize_forward_returns(
                raw_fwd,
                close,
                market_cap=panels.get("MarketCap"),
                sector_map=sector_map if wf.get("sector_neutralize", True) else None,
                periods_per_year=ppy,
                min_names=max(int(wf.get("min_names", 100)), 30),
            )
        else:
            fwd = raw_fwd

    ops = load_calculator_operators()
    code_map = get_operator_code_map(ops)
    pool = AlphaPool(
        max_pairwise_corr=float(wf.get("pool_max_corr", 0.50)),
        max_size=int(max_members),
    )

    for idx, rec in enumerate(archived):
        formula = rec.get("formula")
        if not formula:
            continue
        fields = _infer_fields_from_formula(formula) or ["Close"]
        fn_name = f"signal_archived_{idx}"
        spec = [{"name": fn_name, "formula": formula, "fields": fields}]
        op_code = collect_operator_code(spec, code_map)
        fn_code = f"def {fn_name}({', '.join(fields)}):\n    return {formula}\n"
        res = execute_signal_functions(assemble_module(op_code, fn_code), panels)
        if not res:
            continue
        _, vals = res[0]
        if mask is not None:
            vals = vals.where(mask.reindex_like(vals).fillna(False))
        pool.try_admit(
            name=rec.get("name") or fn_name,
            formula=formula,
            values=vals,
            direction=float(rec.get("direction", 1.0)),
            train_ic=rec.get("train_ic"),
            val_ic=rec.get("val_ic"),
            val_p=rec.get("val_p"),
            val_sharpe=rec.get("val_sharpe"),
        )

    if len(pool.members) < 2:
        return {"status": "insufficient_orthogonal_members", "admitted": len(pool.members)}

    meta_vals, details = pool.combine_heterogeneous(
        fwd,
        train_end_date=wf["train_end_date"],
        ridge_alpha=float(wf.get("pool_ridge_alpha", 10.0)),
        gbdt_weight=float(wf.get("pool_gbdt_weight", 0.45)),
    )
    masked_meta = meta_vals.where(mask.reindex_like(meta_vals).fillna(False)) if mask is not None else meta_vals
    ic = compute_ic_split(
        masked_meta,
        fwd,
        train_end_date=wf["train_end_date"],
        test_start_date=wf["test_start_date"],
        sector_map=sector_map if wf.get("sector_neutralize", True) else None,
        validation_end_date=wf["validation_end_date"],
        purge_periods=int(wf.get("forward_periods", 1)),
        periods=int(wf.get("forward_periods", 1)),
        min_names=int(wf.get("min_names", 100)),
        periods_per_year=int(wf.get("periods_per_year", 12)),
    )
    meta_train = (ic.get("ic_train") or {}).get("mean_ic")
    meta_val = ic.get("mean_ic")
    meta_dir = 1.0 if (meta_train if meta_train is not None else meta_val or 0.0) >= 0 else -1.0
    sign_consistent = bool(
        meta_train is not None and meta_val is not None and meta_train * meta_val > 0
    )
    end = pd.Timestamp(wf["validation_end_date"])
    start = pd.Timestamp(wf["train_end_date"])
    in_val = (masked_meta.index <= end) & (masked_meta.index > start)
    screen = compute_long_short_screen(
        masked_meta.loc[in_val],
        close.loc[close.index <= end],
        quantile=float(wf.get("screen_quantile", 0.15)),
        buffer_quantile=wf.get("screen_buffer_quantile"),
        cost_bps=float(wf.get("cost_bps", 10.0)),
        min_names=max(int(wf.get("min_names", 100)), 20),
        annualization=int(wf.get("periods_per_year", 12)),
        direction=meta_dir,
        sector_map=sector_map if wf.get("sector_neutralize", True) else None,
        include_series=True,
        short_borrow_bps=float(wf.get("short_borrow_bps", 0.0)),
        impact_coeff=float(wf.get("impact_coeff", 0.0)),
        volume_panel=panels.get("Volume"),
        aum_usd=float(wf.get("aum_usd", 1_000_000.0)),
        vol_scaled=bool(wf.get("vol_scaled", False)),
        max_weight=wf.get("max_weight"),
        holding_periods=int(wf.get("forward_periods", 1)),
    )
    net_series = screen.pop("_net_series", None)
    factors = load_factor_returns(wf["factor_returns_uri"]) if wf.get("factor_returns_uri") else None
    fa = compute_factor_alpha(net_series, factors, periods_per_year=int(wf.get("periods_per_year", 12))) if (net_series is not None and factors is not None) else {}
    from signal_discovery_workflow.trial_registry import benjamini_hochberg

    registry = TrialRegistry(r2_store.join(RESULTS_URI, "trials.jsonl"), sharded=True)
    existing_trials = registry.load()
    n_trials = max(1, len(existing_trials) + 1)
    dsr = deflated_sharpe_ratio(net_series, n_trials=n_trials, periods_per_year=int(wf.get("periods_per_year", 12))) if net_series is not None else {}
    stab = subperiod_stability(net_series, periods_per_year=int(wf.get("periods_per_year", 12))) if net_series is not None else {}

    p_val = ic.get("p_value")
    fdr_ok = True
    if wf.get("require_fdr_significance", True):
        if p_val is None:
            fdr_ok = False
        else:
            existing_p = [float(t["p_value"]) for t in existing_trials if t.get("p_value") is not None]
            _, rejected = benjamini_hochberg(existing_p + [float(p_val)], q=float(wf.get("fdr_q", 0.10)))
            fdr_ok = bool(rejected[-1])

    accepted = bool(
        not ic.get("insufficient_history")
        and sign_consistent
        and meta_val is not None
        and abs(meta_val) >= float(wf.get("ic_threshold", 0.02))
        and (p_val is None or p_val <= float(wf.get("p_value_threshold", 0.05)))
        and fdr_ok
        and ((screen.get("net") or {}).get("sharpe") or -99) >= float(wf.get("min_net_sharpe", 0.50))
        and (screen.get("monotonicity") or -99) >= float(wf.get("min_monotonicity", 0.50))
        and (screen.get("spread_t_stat") or -99) >= float(wf.get("min_spread_tstat", 2.0))
        and (fa.get("alpha_t_stat") or -99) >= float(wf.get("min_alpha_tstat", 2.0))
        and (dsr.get("dsr") or 0) >= float(wf.get("min_dsr", 0.95))
        and (stab.get("positive_block_ratio") or 0) >= float(wf.get("min_subperiod_stability", 0.50))
    )
    summary = {
        "status": "accepted" if accepted else "rejected",
        "members": [m["name"] for m in pool.members],
        "stacker": details,
        "val_ic": ic.get("mean_ic"),
        "val_net_sharpe": (screen.get("net") or {}).get("sharpe"),
        "factor_alpha": fa,
        "deflated_sharpe": dsr,
        "subperiod_stability": stab,
    }
    if accepted and wf.get("phf_runs_uri"):
        summary["phf_export"] = export_phf_run(
            wf["phf_runs_uri"],
            signal_name="cross_run_meta_alpha",
            formula=" + ".join(f"{w:.3f}*{k}" for k, w in (details.get("ridge_weights") or {}).items()),
            request_text="cross-run offline meta-combiner",
            ic_results=ic,
            screen_validation=screen,
            screen_full=screen,
            factor_alpha=fa,
            data_pins={"feature_store": {"uri": bundle["manifest_uri"], "sha256": bundle["manifest_sha256"]}},
            config_summary=wf,
        )
    print(f"Cross-run Meta-Combiner result: {summary}")
    return summary


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("qsd-r2"), modal.Secret.from_name("nim-api")],
    timeout=60 * 60,
    cpu=2.0,
    memory=4096,
)
def fanout_discovery(
    max_iterations: int | None = None,
    num_signals: int | None = None,
) -> list[int]:
    """Launch all 6 orthogonal research themes concurrently across Modal containers."""
    import sys

    sys.path.insert(0, f"{REMOTE_ROOT}/src")
    from signal_discovery_workflow import r2_store
    from signal_discovery_workflow.lazy_prices import DEFAULT_LAZY_PRICES_OUTPUT_URI

    has_lazy_prices = False
    try:
        # Require a real panel (>= 1,000 filing diffs), not a tiny 7-row stub.
        has_lazy_prices = r2_store.parquet_rows(DEFAULT_LAZY_PRICES_OUTPUT_URI) >= 1000
    except Exception:
        has_lazy_prices = False

    themes = list(FANOUT_THEMES)
    if not has_lazy_prices:
        themes = [
            (
                cfg,
                "cash-backed quality, low Accruals, low Leverage, and MacroVIX/MacroNFCI regime-conditional value"
                if "Lazy Prices" in req
                else req,
            )
            for cfg, req in themes
        ]

    requests = [req for _, req in themes]
    configs = [cfg for cfg, _ in themes]
    iters = [max_iterations] * len(themes)
    sigs = [num_signals] * len(themes)
    statuses = list(run_loop.map(requests, configs, iters, sigs))
    print(f"Fan-out complete ({len(statuses)} themes): exit statuses={statuses}")
    return statuses


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("qsd-r2"), modal.Secret.from_name("nim-api")],
    schedule=modal.Cron("0 7 * * 1"),
    timeout=3 * 60 * 60,
    cpu=2.0,
    memory=4096,
)
def scheduled_discovery() -> dict:
    """Autonomous weekly Monday 07:00 UTC: refresh monthly & weekly stores, fan out 7 themes, then stack."""
    build_feature_store.remote(sleeve="monthly")
    build_feature_store.remote(sleeve="weekly")
    statuses = fanout_discovery.local()
    meta_monthly = run_cross_run_meta_combiner.remote(
        config_file="configs/config-crsp-fundamentals.yml"
    )
    meta_quarterly = run_cross_run_meta_combiner.remote(
        config_file="configs/config-crsp-quarterly-horizon.yml"
    )
    meta_weekly = run_cross_run_meta_combiner.remote(
        config_file="configs/config-crsp-weekly.yml"
    )
    return {
        "fanout_statuses": statuses,
        "meta_combiner_monthly": meta_monthly,
        "meta_combiner_quarterly": meta_quarterly,
        "meta_combiner_weekly": meta_weekly,
    }


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("qsd-r2")],
    timeout=60 * 60,
    cpu=4.0,
    memory=4096,
)
def build_lazy_prices_shard(
    cik_entries: list[dict],
    shard_id: int,
    start_year: int = 2010,
    max_bytes: int = 250_000,
    max_workers: int = 16,
) -> dict:
    """Process one shard of CRSP-mapped CIKs in bounded streaming memory."""
    import sys

    sys.path.insert(0, f"{REMOTE_ROOT}/src")
    from signal_discovery_workflow.lazy_prices import process_lazy_prices_cik_shard

    return process_lazy_prices_cik_shard(
        cik_entries,
        shard_id=shard_id,
        start_year=start_year,
        max_bytes=max_bytes,
        max_workers=max_workers,
    )


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("qsd-r2")],
    timeout=2 * 60 * 60,
    cpu=4.0,
    memory=8192,
)
def build_lazy_prices_panel(
    num_shards: int = 32,
    start_year: int = 2010,
    max_bytes: int = 250_000,
) -> dict:
    """Parallel sharded Modal job: compute Lazy Prices 10-K/10-Q diffs from R2 corpus v2."""
    import sys

    sys.path.insert(0, f"{REMOTE_ROOT}/src")
    from signal_discovery_workflow.lazy_prices import (
        list_lazy_prices_cik_shards,
        merge_lazy_prices_shards,
    )

    shards = list_lazy_prices_cik_shards(num_shards=num_shards)
    if not shards:
        return {"status": "empty_corpus", "diffs_written": 0}

    shard_ids = list(range(len(shards)))
    years = [start_year] * len(shards)
    limits = [max_bytes] * len(shards)
    shard_results = list(build_lazy_prices_shard.map(shards, shard_ids, years, limits))
    merged = merge_lazy_prices_shards()
    result = {
        "shards": len(shards),
        "ciks_processed": sum(r.get("ciks_processed", 0) for r in shard_results),
        "filings_read": sum(r.get("filings_read", 0) for r in shard_results),
        **merged,
    }
    print(f"Lazy Prices parallel batch complete: {result}")
    return result


@app.local_entrypoint()
def main(
    request: str = "cash-flow value, FcfYield, EbitToEv, EarningsAccel, InsiderNetBuyRatio, InstOwnershipDelta, and macro-regime-conditional quality signals",
    config_file: str = "configs/config-crsp-fundamentals.yml",
    max_iterations: int | None = None,
    num_signals: int | None = None,
    fanout: bool = False,
) -> None:
    if fanout:
        statuses = fanout_discovery.remote(max_iterations=max_iterations, num_signals=num_signals)
        print(f"fanout_discovery exit statuses: {statuses}")
        if any(s != 0 for s in statuses):
            raise SystemExit(1)
        return
    returncode = run_loop.remote(
        request,
        config_file,
        max_iterations=max_iterations,
        num_signals=num_signals,
    )
    print(f"run_loop exit status: {returncode}")
    if returncode != 0:
        raise SystemExit(returncode)
