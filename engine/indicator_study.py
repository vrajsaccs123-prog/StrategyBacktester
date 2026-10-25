"""Study records for the indicator catalog in indicators.json."""

from __future__ import annotations

from dataclasses import dataclass

from engine.catalog import catalog_categories, load_catalog, reference_reason
from engine.indicator_lessons import GROUP_INTROS, LESSONS
from engine.indicators import engine_for_catalog

INPUT_LABELS = {
    "O": "open",
    "H": "high",
    "L": "low",
    "C": "close",
    "V": "volume",
    "P": "a price series, usually the close",
    "Q": "a second price series",
    "M": "a benchmark series",
    "OI": "open interest",
    "V_buy": "aggressive buy volume",
    "V_sell": "aggressive sell volume",
    "advancers": "count of advancing issues",
    "decliners": "count of declining issues",
    "adv": "count of advancing issues",
    "dec": "count of declining issues",
    "adv_vol": "volume of advancing issues",
    "dec_vol": "volume of declining issues",
    "universe": "a universe of symbols",
    "put_vol": "put volume",
    "call_vol": "call volume",
    "VIX": "VIX",
    "VIX3M": "three-month VIX",
    "IV": "implied volatility",
    "S": "underlying price",
    "K": "strike",
    "T": "time to expiry",
    "r": "interest rate",
    "sigma": "volatility input",
    "chain": "option chain",
    "put_oi": "put open interest",
    "call_oi": "call open interest",
    "F": "futures price",
    "funding": "funding-rate series",
    "long": "long positions",
    "short": "short positions",
    "spread": "a spread series",
    "equity": "an equity curve",
    "trades": "a closed-trade list",
    "timestamp": "bar timestamp",
    "events": "an event calendar",
    "multiple": "several sentiment components",
    "short_int": "shares sold short",
    "price": "price",
    "eps": "earnings per share",
    "bvps": "book value per share",
    "ev": "enterprise value",
    "ebitda": "EBITDA",
    "net_income": "net income",
    "financials": "financial statements",
    "fundamentals": "fundamental data",
}


@dataclass(frozen=True)
class IndicatorLesson:
    id: str
    name: str
    category: str
    what: str
    how: str
    why: str
    formula: str
    inputs: tuple[str, ...]
    input_text: str
    params: dict
    param_text: str
    outputs: tuple[str, ...]
    calculated: bool
    code_id: str | None
    limit: str


def group_intro(category: str) -> str:
    return GROUP_INTROS.get(category, "")


def input_text(inputs: list[str]) -> str:
    if not inputs:
        return "none"
    parts = []
    for item in inputs:
        label = INPUT_LABELS.get(item)
        parts.append(f"{label} ({item})" if label else item)
    return ", ".join(parts)


def param_text(params: dict) -> str:
    if not params:
        return "none"
    parts = []
    for key, value in params.items():
        if isinstance(value, list):
            shown = ", ".join(str(piece) for piece in value)
            parts.append(f"{key} = {shown}")
        else:
            parts.append(f"{key} = {value}")
    return "; ".join(parts)


def lessons() -> list[IndicatorLesson]:
    records = []
    for item in load_catalog():
        what, how, why = LESSONS[item["id"]]
        code_id = engine_for_catalog(item["id"])
        records.append(
            IndicatorLesson(
                id=item["id"],
                name=item["name"],
                category=item["category"],
                what=what,
                how=how,
                why=why,
                formula=item["formula"],
                inputs=tuple(item["inputs"]),
                input_text=input_text(item["inputs"]),
                params=dict(item["params"]),
                param_text=param_text(item["params"]),
                outputs=tuple(item["outputs"]),
                calculated=code_id is not None,
                code_id=code_id,
                limit="" if code_id is not None else reference_reason(item),
            )
        )
    return records


def categories() -> list[str]:
    return catalog_categories()
