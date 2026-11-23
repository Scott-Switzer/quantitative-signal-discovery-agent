# Audit and results — 2026-11-22

What was wrong with the workflow, what changed, and what the measurements
actually show. Numbers in this file were produced by the code in this repo on
the 384-name daily panel (2012-01-03 to 2026-11-20) unless stated otherwise.

## 1. Statistical defects (these decided which signals got accepted)

| Defect | Effect | Fix |
|---|---|---|
| t-stat was `mean_ic / (std / sqrt(n))` on daily ICs with `forward_periods=5` overlapping labels | Standard error understated by roughly `sqrt(5)`; p-values far too small | Newey-West (Bartlett) long-run variance, lag `2 * (h - 1)`. Naive t is still reported next to it so the inflation is visible |
| Acceptance and "best so far" were computed on the **test** window | After iteration 1 the out-of-sample number was in-sample; the loop optimized against its own holdout | Purged train / validation / test. Selection and acceptance use validation; test is reported once |
| No purge at the split boundary | Forward returns at the end of train reached into validation | `purge_periods` (defaults to `forward_periods`) removed from the tail of every earlier segment |
| Best function inside a module chosen by **full-sample** \|IC\| | Data snooping inside every iteration | Selection among candidates uses the train window only |
| No record of how many tests were run | Best-of-N at p<0.05 manufactures discoveries | Persistent JSONL ledger + Benjamini-Hochberg q-values over the whole search history |
| IC_IR not annualized, no effective sample size | Hard to compare horizons | `ic_ir_annualized`, `effective_periods = n / horizon`, `hac_lags` |

## 2. Search-space defects (these explain "it never finds anything")

- **Only 30 of 66 operators reached the prompt.** `format_operators_for_prompt`
  truncated at 30, so `CS_Rank`, `Neutralize`, `Winsorize`, `TS_Residual`,
  `TS_Sharpe`, `TS_Drawdown`, `If_Then_Else` and 30 others were invisible. Every
  generation came back as another `TS_Mean`/`TS_Std`/`TS_Return` variant.
- **18 of the 23 advertised data fields never existed.** The README and the
  prompt described 8 fundamentals and 10 ratios; nothing downloaded them. Signals
  referencing them either failed or — worse — were silently evaluated as `Close`
  by the positional fallback in `_resolve_signal_args`. That fallback now
  rejects a signal whose named panel is missing instead of substituting.
- **Only one candidate per iteration was evaluated.** All functions in the
  generated module were executed, then all but one discarded. Each candidate is
  now scored and recorded separately.
- **No cross-run memory.** The model re-proposed the same ideas every run. The
  ledger now feeds "already tried" back into the prompt.

## 3. Data defects

- **Survivorship bias.** The universe is today's S&P 500 applied backwards, and
  the Yahoo path has no delisting returns. Momentum, quality and low-volatility
  signals are flattered. Fixed only by using the fund's CRSP panel (below);
  the Yahoo path prints a warning on every run.
- **One 392-ticker Yahoo request.** Yahoo throttled it and the old
  `dropna(axis=1, how="all")` deleted failed tickers silently. Now batched at 50
  with retries, plus aliases for renamed tickers (`BK→BNY`, `FI→FISV`,
  `MMC→MRSH`). Result: 384/392 names with data.
- **Default end date `2025-12-31`** was in the past, so every run after that
  trained on stale data. Default is now tomorrow.
- **`config-optimization-no-telemetry.yml` omitted** the out-of-sample split and
  sector-neutralization settings, so the QA workflow silently tested different
  scoring semantics from the documented config. Both configs now match.

## 4. Platform

- `nvidia-nat` was pinned to **1.6.\*** while **1.9.0** is current. Bumped, and
  `uv.lock` regenerated.
- **NVIDIA deprecated NeMo Agent Toolkit in October 2026, EOL end of 2026.**
  This repo still runs on it; migration to the successor stack is a real
  project, not a version bump. Recorded here so it is not discovered later.
- `yfinance` resolved to **1.2.0** (a major version past the 0.2.x the script
  was written against). The `download` signature used here is unchanged.

## 5. Performance

The IC was computed by calling `scipy.stats.spearmanr` once per date, per
variant, per horizon. It is now one row-wise rank plus a matrix product.
`compute_ic_split` evaluates 4-7 IC panels per candidate; a six-candidate
iteration did that in about a second.

- Test suite: **49.04s → 6.1s**, 123 tests passing.

## 6. What the numbers actually say

Pure price signals on 384 large caps, 5-day forward return, sector-neutral,
train ≤2020, validation 2021-2024, holdout 2025-2026:

| Signal | Val IC | HAC t | p | Holdout IC |
|---|---|---|---|---|
| 20-day rank momentum | -0.0057 | -0.62 | 0.54 | -0.0046 |
| Liquidity-adjusted 5-day reversal | -0.0103 | -1.95 | 0.052 | -0.0095 |
| Inverse-vol 5-day reversal | -0.0081 | -1.14 | 0.255 | -0.0098 |
| Quality-adjusted reversal (price) | +0.0021 | +0.85 | 0.394 | -0.0062 |

20-day momentum is **negative at every horizon** (1d -0.0084 … 63d -0.0198);
the effect that exists in this window is short-horizon reversal, and it is
small. As a daily-rebalanced long/short it earns a gross Sharpe of -0.40 and a
net Sharpe of -0.93 after 10 bps, because annual turnover is about 85x. That is
the honest reason a 0.01 IC is not alpha at daily frequency.

The naive t-stat for that momentum signal was -1.15 against a HAC t of -0.62:
the old code would have reported roughly twice the significance it had.

**No signal has been accepted.** Nothing here is a recommendation or a
tradeable result.

## 7. Cloudflare R2 is the data and results plane

Nothing is staged locally. Panels are read in place from R2 with column
projection, the trial ledger is an R2 object, and results are written back to an
R2 prefix. `r2_store.py` is the only place that knows about storage.

```bash
export R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_S3_ENDPOINT=https://<account>.r2.cloudflarestorage.com
export QSD_DATA_SOURCE=r2://financial-data-wrds-private/<panel key>
export QSD_RESULTS_URI=r2://financial-data-wrds-lakehouse/research/signals

python -m signal_discovery_workflow.phf_bridge --check   # coverage only, writes nothing
nat run --config_file configs/config-crsp.yml --input "..."
```

The ledger defaults to `<results_uri>/trials.jsonl`; every trial and every saved
signal therefore lives beside the results. `configs/config-crsp.yml` sets
`require_remote: true`, so a missing `data_source` fails instead of silently
falling back to local CSVs.

The CRSP config uses a monthly horizon, 1962-2024 history, and a 2005-2015
validation / 2016-2024 holdout.

## 8. First CRSP run (2026-11-22)

Executed against the R2-sourced panel mirrored at
`personal-hedge-fund/.data/crsp/panel.parquet` (3,459,627 rows, 756 monthly dates,
25,822 securities, 1962-01..2024-12), because no R2 S3 credentials exist in this
environment. The loader read the parquet in place; nothing was staged as CSV.

Request: "12-1 momentum, short-term reversal, low volatility, size, and
betting-against-beta on monthly CRSP". Two candidates were evaluated; the loop
stops at the first acceptance.

| signal | train IC | val IC 2005-2015 | HAC t | p | test IC 2016-2024 | verdict |
|---|---|---|---|---|---|---|
| 12-1 momentum | 0.0028 | 0.0059 | 1.62 | 0.104 | 0.0068 | rejected |
| 12-1 momentum / 63m vol | 0.0377 | **0.0321** | 2.72 | 0.0066 | 0.0266 | accepted |

Second signal's formula is `Div(TS_Return(Close, 12), TS_Vol(Close, 63))`; the
generator named it "adjusted for turnover", which is a mislabel — it is
volatility-scaled momentum.

Screen on the accepted signal (monthly, dollar-neutral, 20% quantile, 10 bp
one-way): annual turnover 5.2x, gross Sharpe 0.16, net Sharpe 0.12, max drawdown
-68%, hit rate 59%, 693 rebalances. IC decay rises with horizon: 0.034 (1m), 0.050
(3m), 0.061 (6m), 0.055 (12m) — the shape of momentum, not short-horizon reversal.

Interpretation: the loop recovers a known effect with sane magnitudes and a
holdout that does not collapse, but 0.12 net Sharpe is not tradable alpha. This
is evidence that the measurement path is correct, not evidence of a novel
anomaly.

### Defects found and fixed by this run

1. **Validation window was empty.** It started at `test_start_date` and ended at
   `validation_end_date`; with test after validation the window was always empty
   and acceptance silently fell back to the test set. Validation now runs
   train_end + 1 day .. validation_end.
2. **Sector demeaning used `mean`, not `nanmean`.** One missing name blanked an
   entire sector for that date; only 62 of 132 validation months survived. All
   months are used now (131 after the purge row).
3. **`inf` poisoned cross-sections.** A formula dividing by a zero return produced
   `inf`, which survives `notna()`, makes a sector mean `NaN` and corrupts ranks.
   Infinite values are treated as missing at the signal/return boundary.
4. **Monthly IC_IR was annualized with 252**, inflating it ~4.6x. Added
   `periods_per_year` (12 for CRSP, 252 default).
5. **Panel-wide ranking was memory-bound.** Ranking 756x25,822 several times per
   window got the process killed on an 8 GB machine. ICs are computed in 128-date
   chunks and the per-date IC series is computed once then sliced per window:
   21s -> 8.7s per signal with bounded memory.
6. **The generator advertised fundamentals that were not loaded.** On a price-only
   panel half the proposals (ROE, EquityToAssets) were rejected by the strict field
   check. The prompt now lists only the panels actually loaded.

## 9. Still open

- R2 reads and writes now run in the cloud (Modal, secret `qsd-r2`); no S3
  credentials exist on this machine and nothing is staged locally. The command
  is in section 7.
- Only two candidates were evaluated before the loop stopped on acceptance. A
  wider search needs the loop to keep going after an acceptance so the ledger and
  BH-FDR see the full trial count.
- The screen labels monthly data as "daily rebalance"; annualization is correct
  (12), the label is not.
- Cloudflare Containers were abandoned: building one requires a local Docker
  build, which is not allowed on this machine (it exhausts the disk and memory
  that the other agents need). See the section below.
- NAT deprecation migration not started.
- `download_data.py` still writes CSVs by design. It is the explicit Yahoo
  fallback, not the production path, and is the only local writer left.

## Cloud execution and data pins (2026-11-22)

The loop now runs on Modal against R2 directly: the CRSP panel is read from
`research/datasets/crsp_monthly/crsp-panel-1.2.2` and results/ledger are written
to `research/signals`. No local staging.

Both inputs are byte-pinned before any data is read, and the pin is stored in
the result payload:

| role | object | sha256 (12) |
| --- | --- | --- |
| market_data | `crsp-panel-1.2.2/.../manifest.json` | `9b8c3c545366` |
| fundamentals | `gold/companyfacts/.../MANIFEST.json` | `9212663251f3` |

A configured mismatch raises `PinMismatch` rather than reading different bytes.

### First cloud run (2 trials, stopped on acceptance)

Selected `Rank(Sqrt(TS_Std(Close, 90)))`: validation IC 0.0909 (t 2.40,
p 0.017, 59 periods), test IC 0.0025. Net screen Sharpe 0.79, max drawdown
-48%, annual turnover 0.09. Ledger: 5 trials total, 2 survive BH-FDR at q=0.05.

The signal is pure volatility. It passed every statistical check and still
violates the mandate, because nothing in the loop required a fundamental field.
Fixed with `fundamentals_only: true` (config), which discards price-only
formulas before scoring and appends the constraint to the generation prompt.

### Defect found by the second cloud run: the holdout became the selection set

`compute_ic_split` built its headline from validation, but fell back to the
test window whenever validation was empty. A candidate whose formula needed
60-month lookbacks on a panel starting in 2016 had zero train and zero
validation periods, 42 holdout months, and was accepted with
`selected_on: "test"` on p=0.093. That is data snooping with a straight face.

Fixed in two places:
- With `validation_end_date` set, the headline is always validation. An empty
  validation window now yields `insufficient_history: true` and null metrics,
  never the holdout's numbers.
- `is_acceptable` refuses any candidate with `insufficient_history` or zero
  validation periods.

Re-run: the same candidate is rejected, and fundamentals candidates report 59
validation periods (2016-2020) instead of falling through to 2021+.

### Defect: valuation ratios were never built in the cloud run

Root cause was upstream of the loader: the workflow kept only `Close` and
`Volume` from the bridge (`name in ("Close", "Volume")`), so `MarketCap` never
reached the fundamentals builder. `BookToMarket` and `EarningsYield` were
skipped and the generator was left with profitability and growth only. The
bridge also lacked a branch for the published panel's ready-made `market_cap`
column, and `Leverage` was erased for every filer missing one debt tag because
`lt_debt + st_debt` propagates NaN. All three fixed.

Units were wrong too: the panel's `market_cap` is dollars (median ~7.8e8), and
the SEC facts are dollars, but the loader divided the facts by 1000 as if CRSP
were in thousands. Verified after the fix: `BookToMarket` median 0.41 across
358,704 firm-months (2016-2024) and `EarningsYield` median 0.0073 across
356,051. A missing panel is now reported in each result's `data_coverage`.

### Verified cloud run with value panels (2026-11-22, iteration 4)

Accepted `Div(CS_Demean(EarningsYield), CS_Demean(AssetGrowth))`
(`signal_quality_adjusted_earnings_yield`) on validation only:

| window | IC | periods |
| --- | --- | --- |
| train (<=2015) | -0.0185 | 64 |
| validation (2016-2020) | -0.0235 (t -3.15, p 0.0016) | 59 |
| test (2021-2024) | -0.0404 | 47 |

Sign is consistent across all three windows. IC decay is monotone in sign and
grows with horizon: -0.0257 (1m), -0.0290 (3m), -0.0312 (6m), -0.0346 (12m).

Cost-aware screen (daily rebalance, 20% quantile, 10 bps one-way):
gross Sharpe 0.124, net Sharpe 0.083, annual return 1.2%, max drawdown -46%.
A real but weak effect; the drawdown makes it unusable as a standalone
strategy, and it is reported that way rather than filtered out.

Ledger: 7 trials this run, 21 total, 4 survive Benjamini-Hochberg at q=0.05.
Both data pins verified. Panels present: Close, Returns, Volume, MarketCap,
BookToMarket, EarningsYield, OperatingProfitability, SalesToAssets, Leverage,
AssetGrowth.

Run command (no local compute):
`modal run modal_run.py --request "..."`.

## Compute without Docker (2026-11-22)

The Cloudflare Container path (`cloudflare/Dockerfile`, Worker + Durable Object,
`wrangler.jsonc`) was written and then deleted. Cloudflare Containers — and the
Sandbox SDK built on them — require a **local** Docker build (`wrangler deploy`
runs `docker build`). On this machine that is not acceptable: the build filled
the disk, was killed, and it competes with the other agents running here.
Cloudflare Workers alone cannot run this loop — pandas/pyarrow over a 3.4M-row
panel does not fit the Workers runtime or its 128 MB memory limit.

Compute therefore runs on Modal, which builds its image **in the cloud** from
the repository directory: no Docker, no local tarball, no local staging.

```bash
modal run modal_run.py --request "value and quality signals"
# cheap smoke: 1 iteration, 2 signals
modal run modal_run.py --request "..." --max-iterations 1 --num-signals 2
```

The upload is ~60 source files (`.venv`, `.git`, caches and notebooks are
excluded). Cloudflare R2 remains the data and results plane (section 7); the
only local work is the `modal run` upload and waiting for the result.

### R2 release resolution returned zero releases (fixed)

`r2_store.list_entries` returned scheme-less paths (`bucket/key`). The resolver
appended `manifest.json` and passed the result to `stat`, which treats anything
without `r2://` as a **local** path — so every candidate release looked missing
and the run failed before reading a byte. Three changes:

- listing entries carry their full URI (`r2://bucket/key`);
- a candidate is a release when both objects *exist*, instead of trusting the
  listing's directory flag (S3/R2 has no directory objects and pyarrow reports
  the flag inconsistently);
- listing tries the bare and the slashed prefix form, because pyarrow expands a
  prefix differently depending on the trailing slash.

### First clean cloud run (Modal, 1 iteration, 2 signals)

Both pins verified (`record` policy), one candidate release, panel coverage
1962-01 .. 2024-12, SEC `MANIFEST.json` 1,362 bytes.

Selected `signal_accrual_adjusted_earnings_yield`: validation IC **+0.0193**
(HAC t 2.61, p 0.0091), train +0.0115, holdout +0.0324, 59 validation months.
Rejected only because IC is below the 0.02 threshold; the cost-aware screen
shows net Sharpe -0.53, so the rejection is the right call.

### Full 8 x 8 cloud run (2026-11-22 18:06 UTC)

33 trials this run, 55 in the ledger, BH-FDR over 46 trials: **17 significant
at q = 0.05**. No new signal clears all gates. The run's selected signal (best
sign-consistent validation IC) was `signal_leverage_turnover_spread`,
`CS_Rank(Div(Leverage, TS_Median(Turnover, 12)))`, rejected on net Sharpe.

| signal | val IC | HAC t | q | train IC | holdout IC | val net Sharpe |
| --- | --- | --- | --- | --- | --- | --- |
| Leverage-Turnover Spread | -0.0416 | -4.53 | 0.00015 | -0.0436 | -0.0402 | -0.78 |
| Accruals-Adjusted Profitability | +0.0415 | +3.47 | 0.0030 | +0.0479 | +0.0529 | -0.20 |

Pins now follow the release that was actually selected: when a rolling prefix
resolves, the resolved manifest is pinned, not a configured exact URI. Otherwise
a republished dataset would verify the old manifest while reading the new panel.

`modal run` streams the loop's log line by line (stderr merged into stdout), so
iterations appear as they happen instead of only on exit. Verified during a live
run: the SEC load and pin lines were visible while the function was still
running, and a later run showed `=== Iteration 4/8 ===` plus each signal's
validation verdict while the container was still active.

## Does any of this provide alpha? (2026-11-22, late)

**No.** One signal nominally cleared the old gate after the screen fixes:
`signal_high_quality_low_leverage_spread` (validation IC 0.0365, q ≈ 0, net
Sharpe 0.200). The saved payload shows why that is not alpha:

- quintile mean returns: 2.28%, 2.09%, 2.22%, 1.61%, 2.50% per month — flat,
  monotonicity **-0.014** (no ordering at all);
- the long/short spread is ~0.22%/month against 3.9% monthly vol, t < 1;
- holdout net Sharpe 0.073, max drawdown -28%.

The 0.20 Sharpe was the difference between two noisy averages, not a risk
premium. Five years of monthly data cannot distinguish 0.20 from zero; the gate
accepted it only because 0.20 was the threshold.

Two measurement defects made the stronger signals look unprofitable before this
run:

1. the screen ranked the **raw** signal while the IC was sector-neutral, so it
   traded a different object (a sector tilt, not the within-sector ordering);
2. `screen_on_validation` passed everything **through** validation_end, i.e.
   roughly 55 years of train-era data that dominated the five validation years.

Both are fixed: the screen demeans within sector, uses the validation window
only, and reports quintile monotonicity and the spread t-stat. Acceptance now
requires those as well (monotonicity >= 0.50, spread t >= 2.0) as well as net
Sharpe >= 0.50.

Re-run after these changes (8/8 iterations, exit 0): selected
`signal_profitability_efficiency_ratio`,
`CS_Rank(Div(OperatingProfitability, SalesToAssets))` — validation IC **+0.0466**
(HAC t 4.43, q 0.00019), train +0.0338, holdout +0.0605, validation net Sharpe
**-0.53** — rejected. The same pattern repeats: strong, stable IC and a tail
spread that loses money after costs.

Both leaders are statistically strong and stable across train/validation/holdout
and both still lose money after costs. The IC is spread through the middle of
the cross-section, not in the tails the screen trades; a negative book Sharpe
even after the direction is set from validation is the sign of a non-monotone
relationship, not of a broken evaluator. That distinction is exactly what the
net-Sharpe gate exists to catch, and it rejected all 33 candidates.

## Four upgrades landed and verified (2026-11-23)

All four research upgrades are active in the production config and were exercised
end-to-end on Modal against R2 (no Docker):

1. **Lakehouse v2 PIT fundamentals.** 19 monthly ratio panels built from the
   frozen `gold.security_monthly_fundamentals_v2` release (104 Parquet parts),
   including `FcfYield`, `EbitToEv`, `CashFlowToAssets`, `Accruals`,
   `DebtToAssets`, `CashToAssets`, `CapexToAssets`, `OperatingMargin`,
   `NetMargin`, `RevenueGrowth`.
2. **Tradable universe + factor-neutral gate.** Formation-date mask
   (`min_price=5`, `min_dollar_volume=$1M`, top 2,000 by cap) leaves a median of
   **601 tradable names per month**. The screen regresses validation net spread
   returns on a 6-factor US model (`be_me`, `ret_12_1`, `cop_at`, `at_gr1`,
   `beta_60m`, `qmj`) and requires residual alpha t >= 2.0.
3. **Lagged macro regimes.** `MacroVIX`, `MacroYieldCurve`, `MacroNFCI` aligned
   to the monthly grid and shifted one month, so no same-month visibility.
4. **Sharded ledger + PHF export.** Per-run immutable JSONL shards under
   `research/signals/trials/`, legacy `trials.jsonl` merged on read; accepted
   signals export to `research/runs/<run_id>/{receipt,results}.json` in the
   `phf-research-receipt/1.0.0` contract.

### Defect found during verification (fixed)

The factor-return release is a union of two exports with different schemas
(AQR rows carry `sheet`; JKP rows carry `weighting`/`direction`). Reading the
first part's column list against every part raised
`No match for FieldRef.Name(sheet)`, the workflow caught the error, and
**the alpha gate silently disappeared** while the config still claimed
`min_alpha_tstat: 2.0`.

Fixes:
- `r2_store.read_parquet_parts` now intersects the requested columns with each
  part's own schema and skips parts missing a pushed-down filter column.
- `load_factor_returns` reads only identifying/value columns, selects the single
  US monthly dataset that covers the most model factors (JKP, 1,188 months),
  and pivots only the six named factors, so duplicate `qmj` from AQR cannot be
  averaged in.
- The workflow now raises instead of accepting signals when
  `min_alpha_tstat > 0` but benchmarks cannot be loaded.

Two regression tests cover the heterogeneous-part read and the dataset
selection/duplicate-factor case. Suite: **154 passed**, ruff clean.

### Full 8 x 8 run after the fixes (2026-11-23 02:02 UTC)

Exit 0, 8 iterations, 61 candidates evaluated, **0 accepted**. The best
validation |IC| was **0.0178** (`cash_flow_to_assets` conditioned on macro), below
the 0.02 IC floor, so no candidate reached the economic or factor gates. That
signal's train IC (-0.0129), validation IC (-0.0178) and holdout IC (-0.0142)
agree in sign, but the effect is too small to trade. The ledger now holds 150
trials: 42 clear naive q=0.05, seven times the false-positive expectation, and
none of the fundamentals-only candidates clears the full gate.

**Verdict unchanged: still no tradable alpha.** The new fundamentals and the
liquidity mask make the search honest about tradability; they have not yet made
it profitable. The weekly Modal cron is deployed (Mondays 07:00 UTC) and rotates
four fundamentals themes.

### Remaining gaps

- Lakehouse v2, FRED and factor releases are read by path, not pinned by SHA;
  only the CRSP panel manifest is pinned. A republished v2 table would change
  results silently.
- Macro panels are constant across securities, so they are only useful as
  `If_Then_Else`/multiplicative conditioning, never as a cross-sectional rank.
- The six-factor model is hardcoded to JKP names; a release without those names
  falls back to the old fuzzy selection.

## Six institutional upgrades: verification (2026-11-23 03:20 UTC)

All six upgrades are wired and execute on the real R2 panels. Suite: **160
passed** (six new tests in `tests/test_institutional_upgrades.py`); `ruff check
src tests modal_run.py` clean. A whole-project `ruff check` still fails on
pre-existing whitespace/import-order issues inside `notebooks/` and `brev/`
notebook cells, which this work did not touch.

Evidence from cloud runs (`modal run modal_run.py`, 4 CPU / 8 GB, no local
staging):

- **Item 2, percentile liquidity:** median **1,800 tradable names/date** with
  `min_dollar_volume_pct: 0.20`, `top_n_market_cap: 1800`, `min_price: 2.0`,
  versus 601 under the old nominal $1M floor.
- **Item 5, quarterly acceleration:** `RevenueAccel`, `EarningsAccel`,
  `MarginDelta`, `QuarterlyRoeDelta` built from `fundamentals.sec_quarterly`;
  the weekly loader resamples `gold/crsp-merged` in place.
- **Item 3, residual target:** residualized forward returns on **755/756**
  dates against size, momentum, volatility and SIC sector dummies.
- **Item 6, hysteresis + diagnostics:** the screen reports buffered turnover,
  monotonicity, spread t and factor-alpha t in both the acceptance screen and
  the saved payload.
- **Item 1, pool + Ridge:** a forced-admission smoke admitted two signals with
  pairwise cross-sectional correlation **0.00** and **0.01**, fit the
  train-only Ridge meta-alpha, and rejected the combination on the unchanged
  gates (validation IC 0.0033, net Sharpe -0.58, monotonicity -0.91, factor
  alpha t -1.41). The plumbing works; the combined signal is still weak.
- **Item 4, Lazy Prices:** implemented and tested, but **not active**: both
  configs now point at `.../lazy_prices/panel.parquet`, but that panel has
  never been built, so the loader skips it.

### Defects found and fixed during verification

- `r2_store.read_parquet_parts` read every part **sequentially** (two S3
  round-trips each). On the 6.77M-row SEC quarterly release this stretched one
  step past 20 minutes and the container was preempted mid-read, so the run
  never reached the loop. Parts are now fetched **concurrently** (12 workers):
  the 19-panel Lakehouse load went from ~2 min to **15 s**, and the quarterly
  acceleration build to ~2 min.
- `_finish` derived the screen direction from train IC and defaulted to +1 when
  it was missing, while the loop fell back to validation IC. The reported
  full-history screen could therefore run with the opposite sign from the
  screen the candidate was accepted on. It now uses the same rule.
- When codegen rejected every spec, the advisor was told to "match the argument
  count exactly" whatever the real cause (here an unstable `CS_Demean`
  denominator), so the next iteration reproduced the identical invalid
  formulas. The advice now quotes the actual failures and names the unstable
  denominator rule.
- The saved payload recorded `min_dollar_volume: 0.0` but not the percentile
  floor that actually defines the universe, so a run could not be reproduced
  from its own metadata. `min_dollar_volume_pct` and `screen_buffer_quantile`
  are now recorded.

### Remaining gaps

- The weekly sleeve now enables the factor gate (`min_alpha_tstat: 2.0`) and
  `compute_factor_alpha` compounds weekly returns to monthly before aligning
  with JKP; the weekly feature store previously lacked fundamentals entirely.
- Under the default `pool_min_ic: 0.010`, a one-iteration smoke admitted no
  members, so the pool only fills in longer runs; no accepted signal has yet
  exercised PHF export in production.
- Still **no tradable alpha**. These six upgrades are research infrastructure
  and an honest cost/liquidity/risk model. They do not create alpha by
  themselves, and the ledger still contains no signal that clears the full
  gate.

## Phase 2 institutional stack — live verification (2026-11-23)

Verified against the R2 lakehouse, not only unit tests.

### What ran
- `uv run pytest -q`: **169 passed**; `ruff check src tests modal_run.py`: clean.
- Form 4 ingest: **67 quarters, 674,246 issuer-day rows** written to
  `r2://.../research/datasets/sec_form4/panel.parquet` (2026Q4 not yet posted).
- Form 13F ingest: **54 archives, 1,293,113 CUSIP-quarter rows** written to
  `r2://.../research/datasets/sec_13f/panel.parquet` (Sep–Nov 2026 not yet posted).
- Monthly feature store: **35 panels**, including `InsiderNetBuyRatio`,
  `InsiderClusterBuys`, `DiscretionaryBuyRatio`, `InstOwnershipDelta`, and
  `InstCrowding`.
- Weekly feature store: **31 panels** after adding Lakehouse v2 fundamentals.
- Monthly 1-iteration/2-signal discovery smoke: exit 0, loaded 35 panels and the
  JKP 6-factor benchmark (1,188 months), ledger 80 trials. Generated:
  - `CS_Rank(Div(FcfYield, TS_Median(AssetGrowth, 6))) * Sign(Sub(InsiderNetBuyRatio, 0.1))` — validation IC **0.0005**
  - `CS_Rank(Sub(OperatingProfitability, TS_Median(Leverage, 12))) * Rank_Add(InstOwnershipDelta, InsiderClusterBuys)` — validation IC **0.0066**

  Both rejected. Neither reached the screen because IC is below the 0.010
  screening floor, so there is no alpha evidence to report.
- Cross-run meta-combiner completed but the pool has fewer than two archived
  members, so no stacking result was produced.
- `modal deploy modal_run.py` succeeded; the Monday 07:00 UTC cron now carries
  these fixes.

### Defects found and fixed
- All five new Modal jobs imported the package in Modal’s base interpreter
  while deps lived in `/root/qsd/.venv` → `ModuleNotFoundError`. The image now
  installs into `/usr/local` via `UV_PROJECT_ENVIRONMENT`.
- The lean runner called `__anext__` on `signal_optimizer_function`, which
  `@register_function` wraps as an async context manager. The smoke always
  crashed before any signal ran; the runner now uses `async with`.
- Form 13F URLs assumed `YYYYqN` forever. The SEC switched to rolling 2–3 month
  windows in 2024 and moved 2026 files to `datastandardsinnovation`; the old
  builder ingested **zero** 13F rows. Both regimes and both paths are handled.
- Form 4 2026 Q2+ had the same path migration; added the fallback.
- `r2_store._parquet_parts` always treated a remote URI as a directory, so
  single-file `panel.parquet` datasets were invisible. Feature-store builds
  silently skipped Form 4, 13F, and Lazy Prices. It now returns a named object
  directly.
- `_cap_and_norm` renormalized after clipping, undoing the 3% per-name cap. It
  now only scales down when the cap binds.
- SEC fetches advertised gzip but never decompressed the response. Fixed.
- Weekly CRSP date pushdown compared a text `date` column to a Timestamp
  (`ArrowNotImplementedError`); it now uses ISO string bounds.
- Weekly market cap arrives as Parquet `Decimal`, which `np.log` cannot consume;
  the exposure builder coerces object columns to numeric.
- The final `uv sync` pruned LightGBM, silently downgrading the GBRT stacker;
  install moved after sync.
- The Monday cron’s 60-minute timeout was shorter than its own build + 6-way
  fanout; raised to 3 hours.
- The weekly config never loaded Lakehouse v2, so value/quality formulas were
  rejected as unloaded fields; added the URI and rebuilt the store.

### Remaining gaps (post-audit fixes landed 2026-11-23)
- All 10 post-audit code defects have been fixed in the working tree:
  1. `modal_run.py::run_loop` invokes `sys.executable` directly (stops `uv run` from pruning LightGBM).
  2. `signal_evaluator.py::_cap_and_norm` uses iterative water-fill capping + matched leg gross exposure (strict dollar neutrality).
  3. `signal_evaluator.py::compute_long_short_screen` normalizes period `Volume` to daily ADV, scales period vol to daily vol, and supports `holding_periods` overlapping tranches for `forward_periods=3`.
  4. `signal_discovery_optimization_workflow.py::_final_extras` passes all 6 execution realism params to full-history `extras["screen"]` and screens `AlphaPool`-eligible weak signals so `val_sharpe` is populated.
  5. `fundamentals.py::build_lakehouse_v2_panels` uses PIT 45-day scatter and frequency-aware 1-year lag (52w/252d/12m) on sub-monthly grids.
  6. `alt_data.py` aggregates Form 4 rolling 90-day flows via `np.add.at`, deduplicates Form 13F managers to latest filing per rolling archive, and applies fail-closed unique-permno filtering to `_load_cusip_bridge`.
  7. `_compat.py` decouples the workflow/generators from hard `nvidia-nat` and `langchain_core` imports.
  8. `lazy_prices.py` + `modal_run.py::build_lazy_prices_panel` redesigned into a 32-shard parallel `.map()` streaming pipeline (CRSP-mapped CIK filter, 10-K/10-Q & `start_year=2010` pre-filter, 250 KB prefix reads via `r2_store.read_bytes_prefix`).
  9. `modal_run.py::run_cross_run_meta_combiner` uses train-anchored direction and enforces the full acceptance gate suite (abs IC, p-value, sign consistency, BH-FDR, spread t-stat).
  10. `modal_run.py::fanout_discovery` dynamically swaps the Lazy Prices theme if `panel.parquet` is not yet built.
- Needs live terminal run: `uv run pytest`, `uv run ruff check src tests modal_run.py`, `modal run modal_run.py::build_lazy_prices_panel`, `modal run modal_run.py::build_feature_store --sleeve monthly`, `modal run modal_run.py::build_feature_store --sleeve weekly`, and `modal deploy modal_run.py`.
- Infrastructure is real; alpha is not yet demonstrated.
