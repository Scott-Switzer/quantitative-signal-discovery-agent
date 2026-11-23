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
Signal Code Generator Agent.

Takes the JSON output from ``signal_generator`` and an LLM, and produces a
self-contained, executable Python module:

    import pandas as pd
    import numpy as np

    # operator definitions inlined from calculator.json
    def Div(...): ...
    def TS_Return(...): ...
    ...

    # signal functions written by the LLM, one per JSON entry
    def signal_xxx(Close, Volume, ...) -> pd.DataFrame:
        \"\"\"<meaning>\"\"\"
        return <formula>

The LLM only writes the signal function bodies. Operator code and imports
are added deterministically so the output is self-contained and portable.
"""

import ast
import json
import logging
import re
from typing import Iterable

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
from .llm_utils import (
    NO_THINK_INSTRUCTION,
    extract_json_array,
    extract_python_block,
    extract_response_text,
    normalize_operator_names,
    sanitize_unicode,
)
from .signal_evaluator import get_operator_arities
from .signal_generator import (
    VALID_DATA_FIELDS,
    get_operator_code_map,
    load_calculator_operators,
    load_output_template,
)

logger = logging.getLogger(__name__)

DATA_FIELD_ALIASES = {
    "adj_close": "Close",
    "adjclose": "Close",
    "adjusted_close": "Close",
    "adjustedclose": "Close",
    "close_price": "Close",
    "closing_price": "Close",
    "price": "Close",
    "open_price": "Open",
    "opening_price": "Open",
    "high_price": "High",
    "low_price": "Low",
    "trading_volume": "Volume",
    "volume_traded": "Volume",
    # Fundamental aliases (canonical names are Revenue, NetIncome,
    # OperatingIncome, Assets, Liabilities, Equity, Cash, LongTermDebt).
    "net_income": "NetIncome",
    "netincome": "NetIncome",
    "operating_income": "OperatingIncome",
    "operatingincome": "OperatingIncome",
    "ebit": "OperatingIncome",
    "total_assets": "Assets",
    "total_liabilities": "Liabilities",
    "stockholders_equity": "Equity",
    "shareholders_equity": "Equity",
    "shareholder_equity": "Equity",
    "stockholder_equity": "Equity",
    "total_equity": "Equity",
    "cash_and_equivalents": "Cash",
    "cash_equivalents": "Cash",
    "long_term_debt": "LongTermDebt",
    "longtermdebt": "LongTermDebt",
    "total_debt": "LongTermDebt",
    # Ratio aliases (canonical names are ROE, ROA, OperatingMargin,
    # NetMargin, RevenueGrowth, NetIncomeGrowth, DebtToEquity,
    # LongTermDebtToEquity, EquityToAssets, CashToAssets).
    "roe": "ROE",
    "return_on_equity": "ROE",
    "returnonequity": "ROE",
    "roa": "ROA",
    "return_on_assets": "ROA",
    "returnonassets": "ROA",
    "operating_margin": "OperatingMargin",
    "operatingmargin": "OperatingMargin",
    "net_margin": "NetMargin",
    "netmargin": "NetMargin",
    "revenue_growth": "RevenueGrowth",
    "revenuegrowth": "RevenueGrowth",
    "net_income_growth": "NetIncomeGrowth",
    "netincomegrowth": "NetIncomeGrowth",
    "book_to_market": "BookToMarket",
    "booktomarket": "BookToMarket",
    "earnings_yield": "EarningsYield",
    "earningsyield": "EarningsYield",
    "operating_profitability": "OperatingProfitability",
    "operatingprofitability": "OperatingProfitability",
    "asset_growth": "AssetGrowth",
    "assetgrowth": "AssetGrowth",
    "share_issuance": "ShareIssuance",
    "shareissuance": "ShareIssuance",
    "leverage": "Leverage",
    "book_equity_to_assets": "BookEquityToAssets",
    "bookequitytoassets": "BookEquityToAssets",
    "market_cap": "MarketCap",
    "marketcap": "MarketCap",
    "returns": "Returns",
    "debt_to_equity": "DebtToEquity",
    "debttoequity": "DebtToEquity",
    "debt_equity": "DebtToEquity",
    "long_term_debt_to_equity": "LongTermDebtToEquity",
    "longtermdebttoequity": "LongTermDebtToEquity",
    "equity_to_assets": "EquityToAssets",
    "equitytoassets": "EquityToAssets",
    "cash_to_assets": "CashToAssets",
    "cashtoassets": "CashToAssets",
    "fcf_yield": "FcfYield",
    "fcfy": "FcfYield",
    "free_cash_flow_yield": "FcfYield",
    "ebit_to_ev": "EbitToEv",
    "ebit_ev": "EbitToEv",
    "cash_flow_to_assets": "CashFlowToAssets",
    "cfo_to_assets": "CashFlowToAssets",
    "debt_to_assets": "DebtToAssets",
    "macro_vix": "MacroVIX",
    "vix": "MacroVIX",
    "macro_yield_curve": "MacroYieldCurve",
    "yield_curve": "MacroYieldCurve",
    "macro_nfci": "MacroNFCI",
    "nfci": "MacroNFCI",
    "earnings_accel": "EarningsAccel",
    "revenue_accel": "RevenueAccel",
    "margin_delta": "MarginDelta",
    "quarterly_roe_delta": "QuarterlyRoeDelta",
    "filing_text_change": "FilingTextChange",
    "filing_jaccard_diff": "FilingJaccardDiff",
    "filing_length_delta": "FilingLengthDelta",
    "amihud_illiq": "AmihudIlliq",
    "amihud": "AmihudIlliq",
    "idio_vol": "IdioVol",
    "idiosyncratic_volatility": "IdioVol",
    "high_52w_ratio": "High52wRatio",
    "high52w": "High52wRatio",
    "volume_shock": "VolumeShock",
    "insider_net_buy_ratio": "InsiderNetBuyRatio",
    "insider_cluster_buys": "InsiderClusterBuys",
    "discretionary_buy_ratio": "DiscretionaryBuyRatio",
    "inst_ownership_delta": "InstOwnershipDelta",
    "inst_crowding": "InstCrowding",
}


def _python_function_name(name: str | None, index: int) -> str:
    """Convert a signal's display name into a valid ``signal_*`` Python identifier."""
    base = re.sub(r"[^A-Za-z0-9_]+", "_", (name or f"signal_{index + 1}").lower()).strip("_")
    if not base:
        base = f"signal_{index + 1}"
    if not base.startswith("signal"):
        base = f"signal_{base}"
    if base[0].isdigit():
        base = f"signal_{base}"
    return base


def _infer_fields_from_formula(formula: str) -> list[str]:
    """Detect which loaded fields the formula references.

    This must be derived from the valid field set, not a hand-maintained list:
    a field missing here is not passed to the generated function, the formula
    then raises ``NameError``, and the whole iteration silently produces no
    candidates. That is exactly what happened to the first fundamentals run.
    """
    ordered = ["Open", "Close", "High", "Low", "Volume"] + sorted(
        VALID_DATA_FIELDS - {"Open", "Close", "High", "Low", "Volume"},
        key=len,
        reverse=True,
    )
    return [
        field
        for field in ordered
        if re.search(rf"\b{re.escape(field)}\b", formula)
    ]


def _normalize_data_field_names(text: str) -> str:
    """Replace common LLM-emitted data-field aliases with canonical OHLCV names."""
    canonical_by_lower = {f.lower(): f for f in VALID_DATA_FIELDS}

    def replace(match: re.Match) -> str:
        name = match.group(0)
        lower = name.lower()
        return canonical_by_lower.get(lower) or DATA_FIELD_ALIASES.get(lower) or name

    return re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", replace, text)


def _normalize_declared_fields(fields: Iterable) -> list[str]:
    """Normalize the JSON ``data_fields_used`` list and discard unknown fields."""
    normalized: list[str] = []
    for field in fields:
        if not isinstance(field, str):
            continue
        canonical = _normalize_data_field_names(field).strip()
        if canonical in VALID_DATA_FIELDS and canonical not in normalized:
            normalized.append(canonical)
    return normalized


def _check_formula_arity(formula: str, arities: dict[str, tuple[int, int]]) -> str | None:
    """
    Statically verify each operator call in a formula uses the right arg count.

    Returns ``None`` if the formula is well-formed (or unparseable, in which case
    we let runtime catch it). Returns a human-readable error string otherwise so
    the caller can skip the spec and surface useful feedback.
    """
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError:
        return None  # not Python — let downstream parser deal with it

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        if name not in arities:
            continue
        min_args, max_args = arities[name]
        n_args = len(node.args)
        if n_args < min_args or (max_args >= 0 and n_args > max_args):
            expected = (
                f"{min_args}" if min_args == max_args
                else f"{min_args}-{max_args}" if max_args >= 0
                else f"at least {min_args}"
            )
            return f"{name} expects {expected} arg(s), got {n_args}"
    return None


def _check_unstable_denominator(formula: str) -> str | None:
    """Reject divisions by a cross-sectionally demeaned/ranked panel.

    ``Div(x, CS_Demean(y))`` is the defect that produced the first accepted SEC
    signal: a demeaned denominator crosses zero, so the ratio has enormous
    outliers and the rank IC is driven by a handful of names. It is not a
    signal, it is a numerical accident.
    """
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError:
        return None
    unstable = {"CS_Demean", "Neutralize", "CS_Rank", "Rank"}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Div"
            and len(node.args) >= 2
        ):
            denominator = node.args[1]
            calls = {
                inner.func.id
                for inner in ast.walk(denominator)
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
            }
            bad = calls & unstable
            if bad:
                return (
                    f"denominator uses {sorted(bad)[0]}, which crosses zero; "
                    "divide first, then demean"
                )
    return None


def _check_formula_operators(formula: str, arities: dict[str, tuple[int, int]]) -> str | None:
    """
    Statically verify every operator call in a formula names a real operator.

    The arity check silently skips operators it does not recognize, so a
    hallucinated name like ``Rank_Exp`` would otherwise slip through to code
    generation (its code is never inlined) and only fail at runtime as a
    ``NameError``, wasting the signal slot. This check catches it at parse
    time so the signal is rejected with a clear message and the generator can
    self-correct on the next iteration.

    Returns ``None`` if every called name is a known operator (or the formula
    is unparseable, in which case runtime catches it). Returns a
    human-readable error string naming the unknown operator(s) otherwise.
    """
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError:
        return None  # not Python — let downstream parser deal with it

    unknown: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        if name not in arities and name not in unknown:
            unknown.append(name)
    if not unknown:
        return None
    return f"unknown operator(s): {', '.join(unknown)}"


def parse_signal_specs_with_errors(
    signal_json: str, valid_operators: Iterable[str]
) -> tuple[list[dict], list[str]]:
    """
    Parse the signal generator's JSON into normalized specs and a list of
    skip messages explaining why any signals were rejected.

    Each spec contains ``name``, ``formula``, ``fields``, ``doc``. Skip messages
    are short strings like ``"#0 (Momentum: Rank expects 1 arg(s), got 2)"``
    suitable for embedding directly into the next iteration's feedback prompt.
    """
    sanitized = sanitize_unicode(signal_json)
    try:
        data = json.loads(sanitized)
    except json.JSONDecodeError:
        data = extract_json_array(sanitized)
    if not isinstance(data, list):
        data = [data] if isinstance(data, dict) else []

    if not data:
        cleaned = signal_json.strip().replace("\n", " ")
        head = cleaned[:300]
        tail = cleaned[-300:] if len(cleaned) > 600 else ""
        logger.warning(
            f"Could not parse signal generator output as JSON ({len(signal_json)} chars). "
            f"Head: {head!r}" + (f" ... Tail: {tail!r}" if tail else "")
        )
        return [], ["generator output was not valid JSON"]

    template = load_output_template()
    required_fields = template.get("validation_rules", {}).get(
        "required_fields", ["name", "formula", "meaning"]
    )
    arities = get_operator_arities()

    specs: list[dict] = []
    skipped: list[str] = []
    for idx, signal in enumerate(data):
        if not isinstance(signal, dict):
            skipped.append(f"#{idx} (not a dict)")
            continue

        missing = [f for f in required_fields if not signal.get(f)]
        if missing:
            skipped.append(f"#{idx} ({signal.get('name', '?')}: missing {missing})")
            continue

        formula = normalize_operator_names(
            sanitize_unicode(signal["formula"]).strip(), valid_operators
        )
        formula = _normalize_data_field_names(formula)

        operator_error = _check_formula_operators(formula, arities)
        if operator_error:
            skipped.append(f"#{idx} ({signal.get('name', '?')}: {operator_error})")
            continue

        stability_error = _check_unstable_denominator(formula)
        if stability_error:
            skipped.append(f"#{idx} ({signal.get('name', '?')}: {stability_error})")
            continue

        arity_error = _check_formula_arity(formula, arities)
        if arity_error:
            skipped.append(f"#{idx} ({signal.get('name', '?')}: {arity_error})")
            continue

        formula_fields = _infer_fields_from_formula(formula)
        declared_fields = _normalize_declared_fields(signal.get("data_fields_used") or [])
        fields = []
        for field in [*formula_fields, *declared_fields]:
            if field not in fields:
                fields.append(field)
        if not fields:
            fields = ["Close"]

        specs.append(
            {
                "name": _python_function_name(signal.get("name"), idx),
                "formula": formula,
                "fields": fields,
                "doc": signal.get("meaning") or signal.get("name", "Signal calculation"),
            }
        )

    if skipped:
        logger.warning(f"Skipped {len(skipped)} malformed signal(s): {skipped}")
    return specs, skipped


def parse_signal_specs(signal_json: str, valid_operators: Iterable[str]) -> list[dict]:
    """
    Parse the signal generator's JSON into a list of normalized specs.

    Backwards-compatible wrapper around :func:`parse_signal_specs_with_errors`
    for callers that only care about the specs.
    """
    specs, _ = parse_signal_specs_with_errors(signal_json, valid_operators)
    return specs


def collect_operator_code(specs: list[dict], code_map: dict[str, str]) -> str:
    """
    Concatenate the Python source for every operator referenced by ``specs``.

    The returned string is meant to be inlined into the generated module so
    that the signal functions can call ``Div(...)``, ``TS_Mean(...)``, etc.
    """
    used: set[str] = set()
    for spec in specs:
        used.update(re.findall(r"\b([A-Za-z_]\w*)\s*\(", spec["formula"]))
    valid = sorted(op for op in used if op in code_map)
    return "\n".join(code_map[op] for op in valid)


def _build_code_prompt(specs: list[dict], operator_signatures: str) -> tuple[str, str]:
    """Return (system, user) prompt strings for the code generator LLM."""
    system = (
        "You translate signal specifications into Python functions. "
        "Output ONLY the function definitions in a single ```python block. "
        "Use the exact operator names from the formula verbatim (case-sensitive). "
        "Do NOT redefine operators. Do NOT add helper functions. Do NOT add imports."
    )
    spec_block = "\n".join(
        f"{i + 1}. name={s['name']}, fields={s['fields']}, formula={s['formula']}"
        for i, s in enumerate(specs)
    )
    user = f"""Operator signatures (already defined, just call):
{operator_signatures}

Generate one function per spec below. The function body must be `return <formula>`.

EXAMPLE INPUT:
1. name=signal_momentum, fields=['Close'], formula=TS_Return(Close, 20)

EXAMPLE OUTPUT:
```python
def signal_momentum(Close: pd.DataFrame) -> pd.DataFrame:
    \"\"\"20-day momentum\"\"\"
    return TS_Return(Close, 20)
```

SPECS:
{spec_block}"""
    return system, user


async def generate_signal_function_code(
    llm,
    specs: list[dict],
    operators: list[dict],
) -> str:
    """
    Use the LLM to write signal function bodies for the given specs.

    The LLM only emits ``def signal_xxx(...): return ...`` blocks; the caller
    is responsible for prepending imports and operator definitions.
    """
    sig_map = {op["name"]: op.get("signature", op["name"]) for op in operators}
    used_op_names = sorted({
        op for spec in specs for op in re.findall(r"\b([A-Za-z_]\w*)\s*\(", spec["formula"])
        if op in sig_map
    })
    operator_signatures = "\n".join(f"- {sig_map[op]}" for op in used_op_names)

    system, user = _build_code_prompt(specs, operator_signatures)

    response = await llm.ainvoke(
        [
            SystemMessage(content=NO_THINK_INSTRUCTION),
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]
    )

    raw = extract_response_text(response)
    code = extract_python_block(raw)
    valid_operator_names = {op["name"] for op in operators}
    return normalize_operator_names(sanitize_unicode(code), valid_operator_names)


def assemble_module(operator_code: str, signal_function_code: str) -> str:
    """Wrap operator + signal function code into a self-contained Python module."""
    return f"import pandas as pd\nimport numpy as np\n\n{operator_code}\n\n{signal_function_code}\n"


def render_signal_functions_deterministic(specs: list[dict]) -> str:
    """Deterministically compile AST-validated signal specs into Python functions.

    Eliminates the redundant code-generator LLM round-trip (~15-30s/iteration)
    and prevents LLM syntax/formatting hallucinations while producing identical
    typed ``def signal_*(...): return <formula>`` functions.
    """
    blocks: list[str] = []
    for spec in specs:
        params = ", ".join(f"{field}: pd.DataFrame" for field in spec["fields"])
        doc = json.dumps(str(spec.get("doc") or spec["name"]))
        blocks.append(
            f"def {spec['name']}({params}) -> pd.DataFrame:\n"
            f"    {doc}\n"
            f"    return {spec['formula']}\n"
        )
    return "\n".join(blocks)


async def generate_signal_code(
    llm,
    signal_json: str,
    operators: list[dict],
    errors_out: list[str] | None = None,
    use_llm: bool = False,
) -> str:
    """
    End-to-end: signal JSON -> self-contained executable Python module.

    1. Parse and AST-validate the signal JSON into normalized specs.
    2. Deterministically emit Python function definitions (or call ``llm`` if
       ``use_llm=True`` is explicitly requested).
    3. Inline the operator definitions and imports.
    """
    code_map = get_operator_code_map(operators)
    specs, parse_errors = parse_signal_specs_with_errors(signal_json, code_map.keys())
    if errors_out is not None:
        errors_out.extend(parse_errors)

    if not specs:
        logger.warning("No usable signals found in generator output")
        return assemble_module("", "# No valid signals generated\n")

    if use_llm and llm is not None:
        function_code = await generate_signal_function_code(llm, specs, operators)
    else:
        function_code = render_signal_functions_deterministic(specs)
    operator_code = collect_operator_code(specs, code_map)
    return assemble_module(operator_code, function_code)


class SignalCodeGeneratorConfig(FunctionBaseConfig, name="signal_code_generator"):
    """Generate executable Python code from signal JSON descriptions."""

    llm_name: str | None = Field(default=None, description="LLM to use for code generation.")


@register_function(config_type=SignalCodeGeneratorConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def signal_code_generator_function(config: SignalCodeGeneratorConfig, builder: Builder):
    """NAT function wrapper around ``generate_signal_code``."""
    operators = load_calculator_operators()

    if not config.llm_name:
        raise ValueError("signal_code_generator requires an llm_name to be configured.")
    llm = await builder.get_llm(llm_name=config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    async def generate(signal_json: str) -> str:
        """Generate executable Python code from signal JSON.

        Args:
            signal_json: The JSON array produced by ``signal_generator``.

        Returns:
            A self-contained Python module (imports + operator defs + signal
            function defs) ready to be ``exec``'d.
        """
        return await generate_signal_code(llm, signal_json, operators)

    yield FunctionInfo.from_fn(
        generate,
        description=(
            "Generate executable Python code from signal_generator JSON output. "
            "Returns a self-contained module with imports, operator defs, and signal functions."
        ),
    )
