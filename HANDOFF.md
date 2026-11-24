# Quantitative Signal Discovery Agent — Engineering Handoff

**Repository:** `/Users/scottthomasswitzer/Documents/Secret/quantitative-signal-discovery-agent`  
**Sibling Portfolio Repo:** `/Users/scottthomasswitzer/Documents/Secret/personal-hedge-fund`  
**Modal App:** `qsd-signal-discovery` (`modal_run.py`)  
**R2 Results & Ledger Root:** `r2://financial-data-wrds-lakehouse/research/signals`  
**PHF Receipt Export Root:** `r2://financial-data-wrds-lakehouse/research/runs`

## 1. Current Status (Honest Snapshot)

- **Alpha discovery status:** No candidate signal has cleared all acceptance gates in live production yet. The pipeline is now instrumented with institutional-grade point-in-time (PIT) data, realistic execution friction, and strict anti-overfitting gates.
- **Verification status:**
  - **R2 data plane (live):** Form 4 (`674,246` issuer-day rows, 67 quarters) and Form 13F (`1,293,113` CUSIP-quarter rows, 54 archives) are populated in R2. Monthly feature store (`monthly_v1`, 35 panels) is built; `weekly_v1` must be rebuilt after the Lakehouse v2 weekly alignment/52-week lag fix below.
  - **Post-audit bugfix & scalable Lazy Prices pass (code complete, awaiting `uv run pytest`, `ruff`, Lazy Prices run, weekly store rebuild, & `modal deploy`):**
    - Fixed `run_loop` in `modal_run.py` to invoke `sys.executable -m signal_discovery_workflow.runner` directly so `uv run` no longer prunes LightGBM at runtime.
    - Fixed `_cap_and_norm` in `signal_evaluator.py` with iterative water-fill capping + matched long/short gross exposure (strict dollar neutrality).
    - Fixed market impact ADV & daily volatility scaling (`trading_days_per_period`) and added `holding_periods` overlapping tranches for the 3-month quarterly-horizon sleeve.
    - Wired all 6 Phase 2 execution realism params into full-history `extras["screen"]` and added validation screening for `AlphaPool`-eligible weak signals.
    - Fixed `build_lakehouse_v2_panels` in `fundamentals.py` for sub-monthly grids (PIT 45-day forward-fill scatter + 52-week/252-day lag).
    - Fixed `alt_data.py`: rolling 90-day Form 4 accumulation (`np.add.at`), Form 13F per-accession aggregation + latest-filing-per-manager dedupe, and fail-closed unique-permno CUSIP bridge.
    - Added `_compat.py` optional shim so `runner.py` and the workflow stack run cleanly even without `nvidia-nat` or `langchain_core`.
    - Redesigned `lazy_prices.py` + `modal_run.py::build_lazy_prices_panel` into a 32-shard parallel `.map()` streaming pipeline (CRSP-mapped CIK filter, pre-HTML 10-K/10-Q & `start_year=2010` filter, 250 KB prefix reads via `r2_store.read_bytes_prefix`).
    - Fixed `run_cross_run_meta_combiner` (train-anchored direction + full acceptance gates including BH-FDR and `spread_t_stat`) and `fanout_discovery` (swaps Lazy Prices theme if `panel.parquet` is absent).
  - **Git state:** All changes are uncommitted in the working tree.

## 2. Hard Constraints & Architecture Rules

- **Zero local data staging:** All dataset reads, SEC EDGAR bulk ingests, feature-store snapshots, sharded trial ledgers, and AlphaPool shards read/write directly against Cloudflare R2 (`r2_store.py`).
- **Cloud compute only:** No local Docker builds. All batch ingestion, feature-store builds, parallel discovery fan-outs, and meta-combiner runs execute on Modal (`modal_run.py`) using secrets `qsd-r2` and `nim-api`.
- **Fail-closed identity resolution:** Replaced the quarantined `dated_permno_cik_bridge_v2` (which contained 159,263 unresolved and 54,695 ambiguous multi-CIK rows) with the approved fail-closed `crsp__ccm_lookup` Parquet release, dropping ambiguous multi-CIK intervals.
- **No WRDS trial/sample contamination:** Excluded WRDS `samp`/`trial` tables found in `wrds_registry.jsonl` (`ibessamp_kpi`, `fssamp__wrds_unadj_13f_holdings_hist`, `optionmsamp_us`) to prevent sample bias; substituted full free SEC EDGAR Form 4 and Form 13F bulk ingestion.

## 3. Architecture & File Map

### Phase 1 — Core Institutional Signal Pipeline (Verified Live on Modal)
- `src/signal_discovery_workflow/alpha_combiner.py`: Two-stage orthogonal `AlphaPool` (`pool_min_ic`, `pool_max_corr: 0.50`, `pool_max_size: 25`) + train-only L2 Ridge stacking.
- `src/signal_discovery_workflow/phf_bridge.py`: Percentile liquidity filter (`min_dollar_volume_pct: 0.20`, `top_n_market_cap: 1800`, `min_price: 2.0`, restoring tradable breadth from 601 to 1,800 names/date) + `load_crsp_weekly_panel`.
- `src/signal_discovery_workflow/residualize.py`: Barra-style cross-sectional OLS residualization of forward returns against size, 12-1 momentum, volatility, and SIC sector.
- `src/signal_discovery_workflow/lazy_prices.py`: SEC 10-K/10-Q year-over-year text diff panels (`FilingTextChange`, `FilingJaccardDiff`, `FilingLengthDelta`) based on [Cohen, Malloy & Nguyen (2020) Lazy Prices](https://ideas.repec.org/a/bla/jfinan/v75y2020i3p1371-1415.html).
- `src/signal_discovery_workflow/fundamentals.py`: `load_sec_quarterly_acceleration_panels` (`RevenueAccel`, `EarningsAccel`, `MarginDelta`, `QuarterlyRoeDelta`) and `_load_cik_bridge`.
- `src/signal_discovery_workflow/signal_evaluator.py`: Gârleanu-Pedersen hysteresis buffer (`screen_quantile: 0.15`, `screen_buffer_quantile: 0.30` / `0.35`).
- `src/signal_discovery_workflow/r2_store.py`: Concurrent 12-worker `read_parquet_parts` (cut Lakehouse v2 load from ~2 min to ~15 s and SEC quarterly build from >20 min to ~2 min).
- `tests/test_institutional_upgrades.py`: Unit/regression suite covering Phase 1 modules.

### Speed & Research-Capacity Upgrades (Code Complete, Needs Live Run)
- `src/signal_discovery_workflow/feature_store.py`: Precomputed zstd float32 Parquet feature-store snapshots in R2 with SHA-256 `manifest.json` and concurrent `ThreadPoolExecutor` load/save (inspired by [Aggregated Parquet Cache patterns](https://marketmaker.cc/blog/parquet-cache-multitimeframe-backtest/) — cuts startup from ~2.5 min to ~3–5 s, with automatic raw-R2 fallback).
- `src/signal_discovery_workflow/phf_bridge.py`: Added 4 weekly microstructure panels (`AmihudIlliq`, `IdioVol`, `High52wRatio`, `VolumeShock`) derived from daily `gold/crsp-merged` rows at zero extra I/O.
- `src/signal_discovery_workflow/macro_features.py`: `compute_factor_alpha` compounds higher-frequency (`periods_per_year > 12`, e.g., weekly 52) net returns to monthly `(1 + r).prod() - 1` before JKP 6-factor regression, enabling `min_alpha_tstat: 2.0` on the weekly sleeve.
- `src/signal_discovery_workflow/alpha_combiner.py`: Added sharded R2 `AlphaPoolLibrary` (`<results_uri>/alpha_pool/shards/`) and `combine_heterogeneous` (train-only L2 Ridge + shallow LightGBM / `HistGradientBoostingRegressor` / pairwise-interaction Ridge fallback, grounded in [Heterogeneous Ensemble Learning (2026)](https://arxiv.org/html/2611.12389v1)).
- `src/signal_discovery_workflow/signal_generator.py`: Registered microstructure panels and added evolutionary `parent_signals` mutation/crossover prompt injection grounded in [QuantaAlpha (2026)](https://arxiv.org/abs/2602.07085) and [CogAlpha (2026)](https://www.alphaxiv.org/abs/2511.18850).
- `src/signal_discovery_workflow/signal_discovery_optimization_workflow.py`: Wired `feature_store_uri` fast path, `eval_workers: 8` parallel candidate evaluation, tail-spread Sharpe-aware `candidate_key`, `AlphaPoolLibrary`, and `combine_heterogeneous`.
- `configs/config-crsp-quarterly-horizon.yml`: 3-month forward-return horizon sleeve (`forward_periods: 3`) for higher fundamental IC and ~3× lower turnover.
- `tests/test_speed_and_capacity.py`: 4 regression tests for feature store, weekly factor compounding, heterogeneous stacking, and evolutionary prompts.

### Phase 2 — Alt-Data, Execution Realism, Anti-Overfitting & NAT Migration (Code Complete, Needs Live Run)
- `src/signal_discovery_workflow/fundamentals.py`: Upgraded `_load_cik_bridge` to read approved fail-closed `crsp__ccm_lookup` Parquet and drop ambiguous multi-CIK intervals.
- `src/signal_discovery_workflow/overfitting.py`: [Bailey & López de Prado (2014) Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) (`deflated_sharpe_ratio`, `min_dsr: 0.95`; reference overview at [Keel DSR guide](https://usekeel.io/learn/deflated-sharpe-ratio)) and contiguous `subperiod_stability` (`min_subperiod_stability: 0.50`).
- `src/signal_discovery_workflow/signal_evaluator.py`: Upgraded `compute_long_short_screen` with short borrow (`short_borrow_bps: 50.0`), square-root market impact (`impact_coeff: 0.15`, `aum_usd: 1_000_000.0`, `volume_panel`), and inverse-volatility risk-scaled legs with single-name weight cap (`vol_scaled: true`, `max_weight: 0.03`).
- `src/signal_discovery_workflow/alt_data.py`: Free SEC EDGAR bulk ingestion straight to R2 and PIT panel builders:
  - **Form 4 Insider Trading** (`ingest_sec_form4_to_r2`, `load_form4_panels` → `InsiderNetBuyRatio`, `InsiderClusterBuys`, `DiscretionaryBuyRatio`; filing-date PIT, Rule 10b5-1 filtered; informed by [QuantInsti Form 4 Event Study (2026)](https://blog.quantinsti.com/sec-form-4-insider-trading-python-event-study/) and [SEC Form 4 Signal Extraction (2026)](https://ijsdr.org/papers/IJSDR2606078.pdf)).
  - **Form 13F Institutional Holdings** (`ingest_sec_13f_to_r2`, `load_13f_panels` → `InstOwnershipDelta`, `InstCrowding`; filing-date PIT, CUSIP8 mapped via CCM Parquet).
  - All 5 fields registered in `signal_generator.py`, `signal_code_generator.py`, and `signal_evaluator.py`.
- `src/signal_discovery_workflow/runner.py`: Standalone plain-Python `asyncio` runner (`DirectNIMChatClient`, `DirectBuilder`, `run_workflow_direct`) replacing the deprecated `nvidia-nat` CLI ([NVIDIA NeMo Agent Toolkit deprecation notice, EOL end of 2026](https://github.com/NVIDIA/NeMo-Agent-Toolkit?ref=blog.langchain.com)).
- `modal_run.py`: 8-CPU / 16 GB containers with `lightgbm` installed; `run_loop` invokes `python -m signal_discovery_workflow.runner`; added `ingest_sec_form4`, `ingest_sec_13f`, `build_feature_store`, 6-theme parallel `fanout_discovery` via `.map()`, `run_cross_run_meta_combiner`, and Monday 07:00 UTC `scheduled_discovery` (`build_feature_store` → `fanout_discovery` → `run_cross_run_meta_combiner`).
- `tests/test_phase2_upgrades.py`: 5 regression tests covering CCM bridge fail-closed filtering, DSR/stability gates, impact/borrow/vol-cap screen, Form 4/13F panels, and `runner.py`.

## 4. R2 Data Plane & Pinned URIs

| Dataset / Store | R2 URI | Notes / Pin |
| --- | --- | --- |
| Monthly CRSP Panel | `r2://financial-data-wrds-lakehouse/research/datasets/crsp_monthly/crsp-panel-1.2.2/` | Manifest SHA-256 `9b8c3c545366cbc8f12ea78a2459fa1c25e17d10501225a79d132f7c7d7daa55` (3,459,627 rows, 1962–2024) |
| Weekly CRSP Source | `r2://financial-system-datasets/gold/crsp-merged/releases/59a8f2e042ce218391401db889fff6a5d50dd71c49751268eb882dfbeeaa4d94/` | 110.2M daily rows (1925–2025); `weekly_start_date: 2005-01-01` |
| Approved CCM Identity Bridge | `r2://financial-data-wrds-private/raw/wrds/crsp__ccm_lookup/releases/20260922T085328Z/part-000000.parquet` | 109,912 rows; fail-closed, replaces quarantined `dated_permno_cik_bridge_v2` |
| Lakehouse v2 PIT Fundamentals | `r2://financial-data-wrds-lakehouse/__r2_data_catalog/01a0cf99-54a7-7e92-b0ea-6453c8e0b34b/01a0e5a6-7ed3-7540-801b-24ec2613cece/data/` | 104 Parquet parts, 19 panels (`FcfYield`, `EbitToEv`, `Accruals`, etc.) |
| SEC Quarterly Acceleration | `r2://financial-data-wrds-lakehouse/__r2_data_catalog/01a0cf99-54a7-7e92-b0ea-6453c8e0b34b/01a0d45e-84e1-7333-a610-9b70beb81488/data/` | 6.77M rows, 4 acceleration panels |
| FRED Macro Regimes | `r2://financial-system-datasets/gold/fred/CURRENT.json` | `MacroVIX`, `MacroYieldCurve`, `MacroNFCI` (1-period lagged) |
| JKP/AQR Factor Benchmarks | `r2://financial-system-datasets/free-resource-lake/parquet/factor_returns/` | Pivots 6 US JKP factors: `be_me`, `ret_12_1`, `cop_at`, `at_gr1`, `beta_60m`, `qmj` |
| Lazy Prices Panel | `r2://financial-data-wrds-lakehouse/research/datasets/lazy_prices/panel.parquet` | Build via `modal run modal_run.py::build_lazy_prices_panel` |
| SEC Form 4 Panel | `r2://financial-data-wrds-lakehouse/research/datasets/sec_form4/panel.parquet` | Build via `modal run modal_run.py::ingest_sec_form4` |
| SEC Form 13F Panel | `r2://financial-data-wrds-lakehouse/research/datasets/sec_13f/panel.parquet` | Build via `modal run modal_run.py::ingest_sec_13f` |
| Monthly Feature Store | `r2://financial-data-wrds-lakehouse/research/feature_store/monthly_v1` | Build via `modal run modal_run.py::build_feature_store --sleeve monthly` |
| Weekly Feature Store | `r2://financial-data-wrds-lakehouse/research/feature_store/weekly_v1` | Build via `modal run modal_run.py::build_feature_store --sleeve weekly` |

## 5. Acceptance Gates & Execution Contract

All 3 configs share the purged split: **Train** `<= 2015-12-31`, **Validation** `2016-01-01` to `2020-12-31`, **Holdout Test** `>= 2021-01-01` (with `purge_periods = forward_periods`).

| Parameter | Monthly (`fundamentals`) | Weekly (`weekly`) | Quarterly (`quarterly-horizon`) |
| --- | --- | --- | --- |
| `forward_periods` / `periods_per_year` | `1` / `12` | `1` / `52` | `3` / `12` |
| `ic_threshold` / `pool_min_ic` | `0.02` / `0.010` | `0.015` / `0.008` | `0.025` / `0.015` |
| `fdr_q` (`require_fdr_significance`) | `0.10` (`true`) | `0.10` (`true`) | `0.10` (`true`) |
| `min_net_sharpe` / `min_monotonicity` | `0.50` / `0.50` | `0.50` / `0.50` | `0.50` / `0.50` |
| `min_spread_tstat` / `min_alpha_tstat` | `2.0` / `2.0` | `2.0` / `2.0` | `2.0` / `2.0` |
| `min_dsr` / `min_subperiod_stability` | `0.95` / `0.50` | `0.95` / `0.50` | `0.95` / `0.50` |
| Execution (`cost_bps` / `short_borrow_bps` / `impact_coeff` / `aum_usd`) | `10` / `50` / `0.15` / `$1M` | `10` / `50` / `0.15` / `$1M` | `10` / `50` / `0.15` / `$1M` |
| Sizing (`vol_scaled` / `max_weight` / `screen_quantile` / `buffer`) | `true` / `0.03` / `0.15` / `0.30` | `true` / `0.03` / `0.15` / `0.30` | `true` / `0.03` / `0.15` / `0.35` |
| Universe (`min_price` / `min_dollar_volume_pct` / `top_n_market_cap`) | `$2` / `0.20` / `1800` | `$5` / `0.20` / `1500` | `$2` / `0.20` / `1800` |
| AlphaPool (`pool_max_corr` / `pool_max_size` / `pool_gbdt_weight`) | `0.50` / `25` / `0.45` | `0.50` / `25` / `0.45` | `0.50` / `25` / `0.45` |

## 6. Step-by-Step Verification & Launch Runbook

Run from `/Users/scottthomasswitzer/Documents/Secret/quantitative-signal-discovery-agent`:

### Step 1 — Local Tests & Lint
```bash
uv run pytest
uv run ruff check src tests modal_run.py
```

### Step 2 — One-Time R2 Alt-Data Ingestion (Run in Parallel Terminals or Sequentially)
```bash
modal run modal_run.py::build_lazy_prices_panel
modal run modal_run.py::ingest_sec_form4
modal run modal_run.py::ingest_sec_13f
```

### Step 3 — Build R2 Feature-Store Snapshots
Run after Step 2 so Lazy Prices, Form 4, and Form 13F are baked into the precomputed snapshots:
```bash
modal run modal_run.py::build_feature_store --sleeve monthly
modal run modal_run.py::build_feature_store --sleeve weekly
```

### Step 4 — Smoke Test `runner.py` & Parallel 6-Theme Fan-Out
```bash
# Quick 1x2 smoke check on the new plain-Python runner + feature store
modal run modal_run.py --max-iterations 1 --num-signals 2

# Full 6-theme parallel fan-out (8x8 per theme)
modal run modal_run.py --fanout

# Offline cross-run heterogeneous Ridge + LightGBM Meta-Combiner
modal run modal_run.py::run_cross_run_meta_combiner
```

### Step 5 — Deploy Weekly Cron & Commit
```bash
modal deploy modal_run.py
git status
git add -A && git commit -m "Phase 2: R2 feature store, Form 4/13F alt-data, CCM bridge, DSR/stability, impact screen, NAT-free runner"
```

## 7. Known Gaps & Next Priorities

1. **Complete `nvidia-nat` removal:** Once Step 4 confirms `runner.py` runs cleanly on Modal, strip `nvidia-nat` dependencies from `pyproject.toml` and regenerate `uv.lock` before its end-of-2026 EOL.
2. **Untouched 2025 out-of-sample vault:** `crsp_daily_primary` in `gold/crsp-merged` runs through `2025-12-31`. Keep 2025 completely untouched during discovery and reserve it as a final one-shot live-equivalent promotion gate before handing receipts to `personal-hedge-fund`.
3. **SHA-pinning non-CRSP releases:** Currently only `crsp-panel-1.2.2` is SHA-pinned; Lakehouse v2, FRED, SEC quarterly, and JKP factor releases are path-referenced. Pinning their manifests in `feature_store.py` metadata or config will tighten reproducibility.
4. **Post-discovery shadow-book decay tracker:** Add a lightweight scheduled check in `personal-hedge-fund` or `modal_run.py` that tracks rolling out-of-sample IC and net Sharpe of promoted `AlphaPoolLibrary` signals to detect alpha decay early (aligned with [AlphaLab (2026)](https://arxiv.org/html/2604.08590v1) and [Event-MoE (2026)](https://aimspress.com/article/doi/10.3934/era.2026392?viewType=HTML) regime-monitoring practices).
5. **Pre-existing notebook lint errors:** `notebooks/` contains pre-existing `ruff` failures unrelated to `src/`, `tests/`, or `modal_run.py` (ignored in Modal builds via `SOURCE_IGNORE`).

## Chapman Value Line research inputs

QSD can load `valueline__screener`, `valueline__reports`, and
`valueline__history` with `signal_discovery_workflow.valueline.load_table(dataset_id)`.
The catalog pins release `20261124T074240Z` in the private Chapman R2 bucket.
The reader verifies the manifest and each Parquet part before returning an Arrow table.

The screener has ranks but lacks source observation timestamps. The report and
history tables have parsed report dates, including dates outside the plausible
snapshot window (1986 to 2028 in the November 2026 extraction). Fiscal year and
ingest timestamps do not establish point-in-time availability. Before a historical
test, validate report dates against source PDFs, match identifiers as of each
validated observation, and use only subsequent returns. Keep projection targets
separate from realized fundamentals and returns.
