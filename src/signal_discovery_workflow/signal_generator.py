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
Signal Generator Agent.

Calls an LLM to produce signal descriptions in the JSON format defined by
``template/signal_output_template.json``. Each signal is a dict with at least
``name``, ``formula``, ``meaning``, ``data_fields_used``, ``operators_used``.

This module also provides:
  - ``load_calculator_operators`` / ``format_operators_for_prompt`` helpers
  - ``signal_validator_function`` and ``list_operators_function`` NAT tools
"""

import json
import logging
import re
from pathlib import Path

from pydantic import Field

from ._compat import (
    Builder,
    FunctionBaseConfig,
    FunctionInfo,
    HumanMessage,
    LLMFrameworkEnum,
    SystemMessage,
    register_function,
)
from .llm_utils import NO_THINK_INSTRUCTION, extract_response_text

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "template"
CALCULATOR_JSON = TEMPLATE_DIR / "calculator.json"
OUTPUT_TEMPLATE_JSON = TEMPLATE_DIR / "signal_output_template.json"

VALID_DATA_FIELDS = {"Open", "Close", "High", "Low", "Volume", "MarketCap", "Returns"}

# Fundamental data fields (point-in-time, forward-filled from each filing's
# FILED date; see the fundamentals README). Raw reported values only, never
# derived ratios: the operators build ratios themselves, e.g.
# Div(NetIncome, Equity) is ROE. Keep this dict in sync with the
# FUNDAMENTAL_FIELDS list in signal_evaluator.py.
FUNDAMENTAL_FIELDS = {
    "Revenue": "latest reported quarterly revenue (point-in-time)",
    "NetIncome": "latest reported quarterly net income (point-in-time)",
    "OperatingIncome": "latest reported quarterly operating income (point-in-time)",
    "Assets": "latest reported total assets (point-in-time)",
    "Liabilities": "latest reported total liabilities (point-in-time)",
    "Equity": "latest reported stockholders equity (point-in-time); ROE is Div(NetIncome, Equity)",
    "Cash": "latest reported cash and cash equivalents (point-in-time)",
    "LongTermDebt": "latest reported long-term debt (point-in-time)",
}
VALID_DATA_FIELDS = VALID_DATA_FIELDS | set(FUNDAMENTAL_FIELDS)

# Pre-computed ratio and percentage panels (derived from the raw fundamental
# panels above, same point-in-time property; see the fundamentals README).
# Raw fundamentals are company-specific absolute numbers that do not compare
# across companies. These ratios are comparable cross-sectionally, so the LLM
# can use them directly instead of rebuilding them with Div. A ratio is
# defined only where its denominator is positive and non-missing; division
# by zero and negative denominators give NaN. Keep this dict in sync with
# the RATIO_FIELDS list in signal_evaluator.py.
RATIO_FIELDS = {
    "BookToMarket": "book equity / market cap (point-in-time)",
    "EarningsYield": "annual net income / market cap (point-in-time)",
    "FcfYield": "free cash flow (operating cash flow - capex) / market cap (point-in-time)",
    "EbitToEv": "operating income / enterprise value (market cap + debt - cash, point-in-time)",
    "GrossProfitability": "gross profit / LAGGED assets (point-in-time; only when COGS is in the release)",
    "AssetTurnover": "sales / lagged assets (point-in-time)",
    "SalesToAssets": "sales / lagged assets (point-in-time)",
    "Accruals": "(net income - operating cash flow) / assets; high accruals underperform (point-in-time)",
    "CashFlowToAssets": "operating cash flow / LAGGED assets, cash-backed quality (point-in-time)",
    "AssetGrowth": "year-over-year asset growth; high growth underperforms (point-in-time)",
    "Leverage": "total liabilities / assets (point-in-time)",
    "DebtToAssets": "total debt / assets (point-in-time)",
    "BookEquityToAssets": "book equity / assets, a solvency measure (point-in-time)",
    "ShareIssuance": "year-over-year growth in shares outstanding; high issuance underperforms (point-in-time)",
    "SgaToSales": "SG&A expense / sales (point-in-time)",
    "RndToSales": "R&D expense / sales (point-in-time)",
    "CapexToAssets": "capital expenditures / lagged assets (point-in-time)",
    "Roe": "annual net income / LAGGED book equity (point-in-time)",
    "Roa": "annual net income / LAGGED assets (point-in-time)",
    "ROE": "pre-computed return on equity, NetIncome / Equity, comparable across companies (point-in-time)",
    "ROA": "pre-computed return on assets, NetIncome / Assets, comparable across companies (point-in-time)",
    "OperatingProfitability": "annual operating income / LAGGED assets (point-in-time)",
    "OperatingMargin": "pre-computed operating margin, OperatingIncome / Revenue (point-in-time)",
    "NetMargin": "pre-computed net margin, NetIncome / Revenue (point-in-time)",
    "RevenueGrowth": "pre-computed year-over-year revenue growth (point-in-time)",
    "NetIncomeGrowth": "pre-computed year-over-year net income growth (point-in-time)",
    "DebtToEquity": "pre-computed leverage, Liabilities / Equity (point-in-time)",
    "LongTermDebtToEquity": "pre-computed leverage, LongTermDebt / Equity (point-in-time)",
    "EquityToAssets": "pre-computed equity share of assets, Equity / Assets (point-in-time)",
    "CashToAssets": "pre-computed cash share of assets, Cash / Assets (point-in-time)",
    "EarningsAccel": "quarter-over-quarter acceleration in YoY earnings scaled by lagged revenue (point-in-time)",
    "RevenueAccel": "quarter-over-quarter acceleration in YoY revenue growth (point-in-time)",
    "MarginDelta": "year-over-year change in quarterly operating margin (point-in-time)",
    "QuarterlyRoeDelta": "year-over-year change in quarterly ROE (point-in-time)",
    "FilingTextChange": "YoY 10-K/10-Q cosine text distance (Lazy Prices; higher change predicts lower returns)",
    "FilingJaccardDiff": "YoY 10-K/10-Q vocabulary Jaccard distance (Lazy Prices, point-in-time)",
    "FilingLengthDelta": "YoY log change in 10-K/10-Q token length (point-in-time)",
    "InsiderNetBuyRatio": "trailing 90d discretionary Form 4 insider net buying ratio in [-1, 1] (filing-date PIT)",
    "InsiderClusterBuys": "log(1 + distinct discretionary Form 4 insider buyers) in trailing 90d (filing-date PIT)",
    "DiscretionaryBuyRatio": "non-10b5-1 Form 4 insider buy dollars / total insider trade dollars (filing-date PIT)",
    "InstOwnershipDelta": "QoQ log change in aggregate SEC 13F institutional dollar holdings (filing-date PIT)",
    "InstCrowding": "Herfindahl concentration (HHI) of SEC 13F institutional holders (high = crowded)",
}

MICROSTRUCTURE_FIELDS = {
    "AmihudIlliq": "weekly mean of |daily return| / dollar volume (Amihud illiquidity premium)",
    "IdioVol": "weekly realized daily return volatility (high idiosyncratic vol underperforms)",
    "High52wRatio": "Close / rolling 52-week high (George-Hwang 52-week high anchor)",
    "VolumeShock": "log(weekly dollar volume / rolling 13-week mean dollar volume)",
}

MACRO_FIELDS = {
    "MacroVIX": "lagged Cboe VIX volatility index (1990+, broadcast across securities for regime conditioning)",
    "MacroYieldCurve": "lagged US 10Y-3M Treasury spread T10Y3M (1982+, broadcast across securities)",
    "MacroNFCI": "lagged Chicago Fed National Financial Conditions Index (1971+, positive = tighter conditions)",
}

# What the published record actually supports. These priors go into the prompt
# so the generator starts from measured effects instead of rediscovering
# momentum in accounting data. Directions are the ones reported in the cited
# papers and replications; the loop still has to find them out of sample.
ANOMALY_PRIORS = """
ESTABLISHED FUNDAMENTALS PRIORS (use these as starting points, not as proof):
- Value: high BookToMarket and high EarningsYield predict higher returns
  (Fama & French 1992; Graham & Dodd).
- Quality: high Roe, Roa and OperatingProfitability predict higher returns
  (Novy-Marx 2013; Hou, Xue & Zhang 2015). Scale profits by LAGGED assets, not
  current assets: current-asset scaling embeds asset growth and contaminates
  the quality premium (Hou, Xue & Zhang 2020).
- Investment: high AssetGrowth and high ShareIssuance predict LOWER returns
  (Cooper, Gulen & Schill 2008; Pontiff & Woodgate 2008).
- Leverage: high Leverage and low BookEquityToAssets predict lower returns
  (Bhandari 1988; Penman, Richardson & Tuna 2007).
- Profitability is stronger than accruals in modern samples; accruals need a
  cash-flow tag that this release may not carry.
- Realistic monthly rank IC for a single fundamentals signal is 0.02-0.05.
  An IC above 0.10 usually means a lookback/point-in-time defect, not alpha.
- Combining two signals only helps when they are not near-duplicates. Do not
  build a ratio of two cross-sectionally demeaned panels: the denominator
  crosses zero and the ratio explodes.
"""


def anomaly_priors_block(available_fields: set[str] | None = None) -> str:
    """Return the priors block, or an empty string on a price-only panel."""
    if available_fields is not None and not (
        available_fields & (set(FUNDAMENTAL_FIELDS) | set(RATIO_FIELDS))
    ):
        return ""
    return ANOMALY_PRIORS.strip()
VALID_DATA_FIELDS = VALID_DATA_FIELDS | set(RATIO_FIELDS) | set(MACRO_FIELDS) | set(MICROSTRUCTURE_FIELDS)


def format_data_fields_for_prompt(available_fields: set[str] | None = None) -> str:
    """Format the DATA section of the prompt.

    ``available_fields`` restricts the list to the panels actually loaded for
    this run. Advertising ROE on a price-only panel wastes generator slots: the
    model proposes profitability signals, the strict field check rejects them,
    and every iteration loses half its candidates.
    """
    def keep(name: str) -> bool:
        return available_fields is None or name in available_fields

    ohlcv = [f for f in ("Open", "Close", "High", "Low", "Volume") if keep(f)]
    lines = ["DATA: " + (", ".join(ohlcv) if ohlcv else "(none)")]
    market = [f for f in ("MarketCap", "Returns") if keep(f)]
    if market:
        lines.append(
            "MARKET PANELS: " + ", ".join(market)
            + " (MarketCap in dollars; Returns is the period total return)"
        )

    fundamentals = {k: v for k, v in FUNDAMENTAL_FIELDS.items() if keep(k)}
    ratios = {k: v for k, v in RATIO_FIELDS.items() if keep(k)}
    micro = {k: v for k, v in MICROSTRUCTURE_FIELDS.items() if keep(k)}
    macros = {k: v for k, v in MACRO_FIELDS.items() if keep(k)}

    if fundamentals:
        lines.append("FUNDAMENTALS (raw reported values, point-in-time, forward-filled from filed date):")
        for name, meaning in fundamentals.items():
            lines.append(f"  {name}: {meaning}")
    if ratios:
        lines.append("RATIOS (pre-computed comparable metrics, derived from the fundamentals above, point-in-time):")
        for name, meaning in ratios.items():
            lines.append(f"  {name}: {meaning}")
    if micro:
        lines.append("MICROSTRUCTURE PANELS (derived from daily CRSP price/volume, backward-looking):")
        for name, meaning in micro.items():
            lines.append(f"  {name}: {meaning}")
    if macros:
        lines.append("MACRO REGIME PANELS (lagged, broadcast across securities for regime conditioning):")
        for name, meaning in macros.items():
            lines.append(f"  {name}: {meaning}")
    if not fundamentals and not ratios:
        lines.append(
            "No fundamental or ratio panels are loaded in this run. Do not reference them; "
            "build the signal from the price, volume and identifier fields above."
        )
    return "\n".join(lines)


def uses_fundamental_fields(formula: str | None) -> bool:
    """True when a formula references at least one fundamental or ratio panel.

    Used to enforce a fundamentals mandate: a price-only formula is a
    momentum or volatility signal in disguise, whatever the prompt asked for.
    """
    from .signal_code_generator import _infer_fields_from_formula

    fields = set(_infer_fields_from_formula(formula or ""))
    return bool(fields & (set(FUNDAMENTAL_FIELDS) | set(RATIO_FIELDS)))


def load_calculator_operators() -> list[dict]:
    """Load operator definitions from ``template/calculator.json``."""
    if not CALCULATOR_JSON.exists():
        logger.warning(f"Calculator template not found at {CALCULATOR_JSON}")
        return []
    with open(CALCULATOR_JSON, "r") as f:
        return json.load(f)


def load_output_template() -> dict:
    """Load the signal output schema from ``template/signal_output_template.json``."""
    if not OUTPUT_TEMPLATE_JSON.exists():
        logger.warning(f"Output template not found at {OUTPUT_TEMPLATE_JSON}")
        return {}
    with open(OUTPUT_TEMPLATE_JSON, "r") as f:
        return json.load(f)


def get_operator_code_map(operators: list[dict]) -> dict[str, str]:
    """Map operator name -> Python implementation source string."""
    return {op["name"]: op["code"] for op in operators}


def format_operators_for_prompt(operators: list[dict], max_operators: int | None = None) -> str:
    """
    Format operator signatures + descriptions for inclusion in a prompt.

    Prioritizes common families (TS_*, Rank, basic arithmetic, Decay/EMA, CS_)
    but defaults to the whole catalogue. The old default of 30 hid 36 of the
    66 operators (CS_Rank, Neutralize, Winsorize, TS_Residual, TS_Sharpe, ...)
    from the generator, which is why every idea came back as the same handful
    of TS_Mean/TS_Std momentum variants.
    """
    if max_operators is None:
        max_operators = len(operators)
    priority_prefixes = ["TS_", "Rank", "Add", "Sub", "Mul", "Div", "Decay", "EMA", "CS_"]
    priority_ops, other_ops = [], []
    for op in operators:
        if any(op["name"].startswith(p) or op["name"] == p for p in priority_prefixes):
            priority_ops.append(op)
        else:
            other_ops.append(op)

    selected = priority_ops[:max_operators]
    if len(selected) < max_operators:
        selected.extend(other_ops[: max_operators - len(selected)])

    lines: list[str] = []
    for op in selected:
        signature = op.get("signature", op["name"])
        if signature.startswith("def "):
            signature = signature[4:]
        lines.append(f"- {signature}")
        lines.append(f"  Description: {op['meanings']}")
    return "\n".join(lines)


def build_signal_template(num_signals: int, template: dict | None = None) -> str:
    """
    Render the placeholder JSON the signal generator should fill in.

    The shape comes from ``signal_output_template.json::signal_template`` so
    the schema is the single source of truth: edit the template file to
    change every place the placeholder is shown.
    """
    template = template if template is not None else load_output_template()
    signal_template = template.get("signal_template")
    if not signal_template:
        raise RuntimeError(
            "signal_output_template.json is missing the 'signal_template' field"
        )
    item = json.dumps(signal_template, indent=2)
    items = ",\n".join([item] * num_signals)
    return "```json\n[\n" + items + "\n]\n```"


def build_signal_example(template: dict | None = None) -> str:
    """Return an illustrative few-shot example from the output template."""
    template = template if template is not None else load_output_template()
    example = template.get("output_format", {}).get("example") or []
    if not example:
        return ""
    return "```json\n" + json.dumps(example[:1], indent=2) + "\n```"


def build_signal_prompt(
    request: str,
    num_signals: int,
    operators_list: str,
    template_block: str,
    feedback: str | None = None,
    example_block: str = "",
    tried_formulas: list[str] | None = None,
    horizon: int = 5,
    available_fields: set[str] | None = None,
    parent_signals: list[dict] | None = None,
    periods_per_year: int = 252,
    residualize_returns: bool = False,
) -> str:
    """Assemble the user prompt for the signal generator LLM.

    Includes optional evolutionary parent signals (QuantaAlpha / CogAlpha
    trajectory mutation and crossover) so the generator can mutate strong
    archived parents or recombine low-correlation mechanisms.
    """
    feedback_section = f"\n\nPREVIOUS FEEDBACK:\n{feedback}\n" if feedback else ""
    example_section = (
        f"\nFor reference, here is an example of one valid signal:\n{example_block}\n"
        if example_block
        else ""
    )
    parents_section = ""
    if parent_signals:
        parent_lines = []
        for p in parent_signals[:6]:
            ic_v = p.get("val_ic")
            sh_v = p.get("val_sharpe")
            ic_s = f"{ic_v:+.4f}" if ic_v is not None else "n/a"
            sh_s = f"{sh_v:+.2f}" if sh_v is not None else "n/a"
            parent_lines.append(
                f"- {p.get('name', '?')}: {p.get('formula', '?')} (val_ic={ic_s}, val_sharpe={sh_s})"
            )
        parents_section = (
            "\nEVOLUTIONARY PARENT ARCHIVE (QuantaAlpha mutation & crossover):\n"
            "Use at least half of your slots to (a) MUTATE one parent (adjust lookback, "
            "add volatility scaling, winsorize extremes, or gate on a macro regime) or "
            "(b) CROSSOVER two parents from DIFFERENT economic families using Add(CS_Rank(A), CS_Rank(B)). "
            "Use the remaining slots for genuinely novel hypotheses:\n"
            + "\n".join(parent_lines)
            + "\n"
        )
    tried_section = ""
    if tried_formulas:
        # Show the MOST RECENT 60 unique formulas (not the oldest 60) so the
        # generator never loops on recent ideas once the ledger grows past 60.
        recent_unique = list(dict.fromkeys(f for f in tried_formulas if f))[-60:]
        listed = "\n".join(f"- {f}" for f in recent_unique)
        tried_section = (
            "\nALREADY TRIED — these were recently evaluated and did not clear the bar. "
            "Do NOT repeat them or trivial rescalings of them (different constant, "
            "same operator chain). Propose genuinely different economic mechanisms:\n"
            f"{listed}\n"
        )
    if periods_per_year <= 12:
        freq_label, unit_label, lookback_hint = "monthly", "month", "3 to 12 months (never 60+ months)"
    elif periods_per_year <= 52:
        freq_label, unit_label, lookback_hint = "weekly", "week", "4 to 26 weeks"
    else:
        freq_label, unit_label, lookback_hint = "daily", "trading-day", "5 to 63 trading days"

    target_desc = (
        f"Barra-residualized {horizon}-{unit_label} forward return (size, 12-1 momentum, "
        "volatility, and sector tilts are ALREADY stripped out — do not rediscover simple "
        "size, momentum, or low-vol factors)"
        if residualize_returns
        else f"{horizon}-{unit_label} forward return, computed after demeaning by sector"
    )
    priors = anomaly_priors_block(available_fields)
    priors_section = f"\n{priors}\n" if priors else ""
    return f"""You are a senior quantitative equity researcher. Generate {num_signals} stock selection signals.

REQUEST: {request}
{priors_section}
OBJECTIVE: This is a {freq_label} panel (1 bar = 1 {unit_label}; choose TS_* lookbacks
around {lookback_hint}). Each signal is ranked cross-sectionally every {unit_label} and
scored by its Spearman (rank) IC against the {target_desc}. The deciding window is
out-of-sample validation; the final holdout is hidden. Acceptance requires BOTH
statistically significant IC AND a monotone, low-turnover long/short tail spread
(top 15% vs bottom 15%) that clears 10 bps spread + market impact + 50 bps short borrow
+ JKP 6-factor alpha + Deflated Sharpe Ratio.
{feedback_section}
{format_data_fields_for_prompt(available_fields)}

OPERATORS:
{operators_list}

STRICT RULES:
- Every operator call MUST match its signature exactly — same number of arguments,
  same order. Compose multiple operators if you need a different shape.
- Do NOT invent operators or pass extra arguments to single-arg operators
  (e.g. `Rank(x)` takes ONE argument; write `Rank(Sub(a, b))`, never `Rank(a, b)`).
- Use only the exact operator names shown above (case-sensitive).
- No look-ahead. Never use a negative shift, a future value, or a statistic
  computed over a window that includes the scoring date. Every operator is
  backward-looking; keep it that way.
- Each of the {num_signals} signals must come from a DIFFERENT economic family
  (momentum, reversal, volatility, liquidity/volume, quality, value, growth,
  financing, accruals). Duplicates of the same idea with a different window are
  wasted slots.
- `Rank(x)` ranks each stock against its OWN history (time series); `CS_Rank(x)`
  ranks across stocks on each date (cross-section). A cross-sectional value or
  quality signal needs `CS_Rank`, not `Rank`.
- Never divide by a cross-sectionally demeaned or ranked panel, e.g.
  `Div(x, CS_Demean(y))`: the denominator crosses zero and the signal becomes
  an outlier explosion. Demean AFTER forming a ratio, not before.
- Prefer one economically interpretable ratio per signal over a stack of
  operators. A signal the model cannot explain cannot be defended.
- Prefer formulas that are not just a monotone rescaling of another signal:
  the cross-sectional rank is all that matters, so adding a constant or
  multiplying by a positive scalar changes nothing.
- If a ratio is already a panel field (ROE, ROA, OperatingMargin, NetMargin,
  RevenueGrowth, NetIncomeGrowth, DebtToEquity, LongTermDebtToEquity,
  EquityToAssets, CashToAssets), use it directly instead of rebuilding it.
{parents_section}{tried_section}{example_section}
Fill in this exact template at the END of your reply (inside a ```json block):

{template_block}

Generate {num_signals} signals now."""


async def generate_signal_json(
    llm,
    request: str,
    num_signals: int,
    operators: list[dict],
    feedback: str | None = None,
    tried_formulas: list[str] | None = None,
    horizon: int = 5,
    available_fields: set[str] | None = None,
    parent_signals: list[dict] | None = None,
    periods_per_year: int = 252,
    residualize_returns: bool = False,
) -> str:
    """
    Call the signal LLM and return its raw response text.

    The response is expected to contain a JSON array of signal objects (often
    inside a ```json fence at the end of a reasoning trace). Downstream code
    is responsible for extracting the JSON.
    """
    template = load_output_template()
    prompt = build_signal_prompt(
        request,
        num_signals,
        format_operators_for_prompt(operators),
        build_signal_template(num_signals, template),
        feedback,
        build_signal_example(template),
        tried_formulas=tried_formulas,
        horizon=horizon,
        available_fields=available_fields,
        parent_signals=parent_signals,
        periods_per_year=periods_per_year,
        residualize_returns=residualize_returns,
    )

    # Disable Nemotron's chain-of-thought trace so the entire token budget
    # goes toward producing the JSON answer (not reasoning prose). Without
    # this, small-context responses can be truncated mid-formula.
    response = await llm.ainvoke(
        [
            SystemMessage(content=NO_THINK_INSTRUCTION),
            HumanMessage(content=prompt),
        ]
    )
    content = extract_response_text(response)

    if not content.strip():
        extras = list((getattr(response, "additional_kwargs", {}) or {}).keys())
        logger.warning(
            "Signal generator returned empty .content. Increase max_tokens "
            f"for the signal_generator LLM. additional_kwargs={extras}"
        )

    logger.debug(f"Signal generator output: {len(content)} chars")
    return content


# =============================================================================
# NAT-registered functions
# =============================================================================


class SignalGeneratorConfig(FunctionBaseConfig, name="signal_generator"):
    """Generate quantitative signals using the calculator operators."""

    num_signals: int = Field(default=3, description="Number of signals to generate.")
    llm_name: str | None = Field(
        default=None,
        description="LLM to use for generation. If None, returns the prompt for the agent to process.",
    )


@register_function(config_type=SignalGeneratorConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def signal_generator_function(config: SignalGeneratorConfig, builder: Builder):
    """NAT function wrapper around ``generate_signal_json``."""
    operators = load_calculator_operators()

    llm = None
    if config.llm_name:
        llm = await builder.get_llm(llm_name=config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    async def generate_signals(request: str) -> str:
        """
        Generate quantitative signals for stock selection.

        Args:
            request: What kind of signals to generate. e.g. "momentum signals",
                     "volatility signals", "volume-price divergence signals".

        Returns:
            JSON array (as a string) of signal objects with name, formula,
            meaning, data_fields_used, operators_used.
        """
        if llm:
            return await generate_signal_json(llm, request, config.num_signals, operators)

        template = load_output_template()
        prompt = build_signal_prompt(
            request,
            config.num_signals,
            format_operators_for_prompt(operators),
            build_signal_template(config.num_signals, template),
            example_block=build_signal_example(template),
        )
        return f"Please generate signals based on this specification:\n\n{prompt}"

    yield FunctionInfo.from_fn(
        generate_signals,
        description=(
            "Generate quantitative stock selection signals in JSON format. "
            "Input: description of signals needed (e.g., 'momentum signals')."
        ),
    )


class SignalValidatorConfig(FunctionBaseConfig, name="signal_validator"):
    """Validate signal formulas against calculator.json operators."""

    strict_mode: bool = Field(default=True, description="Reject signals with unknown operators.")


@register_function(config_type=SignalValidatorConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def signal_validator_function(config: SignalValidatorConfig, builder: Builder):
    """Validate signal formulas to ensure they use known operators."""
    operators = load_calculator_operators()
    valid_operators = {op["name"] for op in operators}

    async def validate_signal(formula: str) -> str:
        """
        Validate a signal formula.

        Example: ``Div(TS_Return(Close, 20), TS_Std(Close, 20))``
        """
        matches = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", formula)
        unknown = [m for m in matches if m not in valid_operators and m not in VALID_DATA_FIELDS]
        valid_used = [m for m in matches if m in valid_operators]
        if unknown and config.strict_mode:
            return f"INVALID: Unknown operators: {', '.join(unknown)}"
        return f"VALID: Uses operators: {', '.join(set(valid_used))}"

    yield FunctionInfo.from_fn(
        validate_signal,
        description="Validate a signal formula against available operators.",
    )


class ListOperatorsConfig(FunctionBaseConfig, name="list_operators"):
    """List available calculator operators."""

    category: str | None = Field(
        default=None,
        description="Filter by category prefix (e.g., 'TS_', 'CS_').",
    )


@register_function(config_type=ListOperatorsConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def list_operators_function(config: ListOperatorsConfig, builder: Builder):
    """List available operators for signal construction."""
    operators = load_calculator_operators()

    async def list_operators(category: str | None = None) -> str:
        """List available operators, optionally filtered by name prefix."""
        cat = category if category is not None else config.category
        filtered = operators
        if cat:
            filtered = [op for op in operators if op["name"].startswith(cat)]
        if not filtered:
            return f"No operators found for category: {cat}"

        result = [f"Available Operators ({len(filtered)} total):\n"]
        for op in filtered[:25]:
            result.append(f"- {op['name']}: {op['meanings'][:80]}...")
        if len(filtered) > 25:
            result.append(f"\n... and {len(filtered) - 25} more")
        return "\n".join(result)

    yield FunctionInfo.from_fn(
        list_operators,
        description="List available calculator operators for signal construction.",
    )
