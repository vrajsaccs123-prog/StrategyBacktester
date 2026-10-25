"""Indicator catalog loaded from indicators.json."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "indicators.json"
DESCRIPTION_PATH = ROOT / "indicators.md"

_REFERENCE = {
    "renko": "Renko replaces the time axis, so it does not line up bar-for-bar with this backtest.",
    "gann_fan": "A fan is a set of angled lines, not one value per bar.",
    "elliott": "Wave counts are not unique, so this stays a reference entry.",
    "cvd": "Needs aggressive buy and sell volume. The CSV only has total volume.",
    "oi_change": "Needs open interest.",
    "correlation": "Needs a second series.",
    "beta": "Needs a benchmark series.",
    "rs_ratio": "Needs a benchmark series.",
    "rrg": "Needs a benchmark series.",
    "dual_momentum": "Needs a cross-section of symbols.",
    "pairs_spread": "Needs a second symbol.",
    "cointegration": "Needs a second symbol.",
    "half_life": "Needs a spread series.",
    "kelly": "Needs a closed trade list, which indicators do not receive.",
    "event_flag": "Needs an event calendar.",
    "value_factor": "Needs fundamental data.",
    "quality_factor": "Needs fundamental data.",
    "ad_breadth": "Needs advance and decline counts.",
    "mcclellan_osc": "Needs advance and decline counts.",
    "mcclellan_sum": "Needs advance and decline counts.",
    "trin": "Needs advance and decline counts and their volume.",
    "pct_above_ma": "Needs a universe of symbols.",
    "new_hl": "Needs a universe of symbols.",
    "put_call": "Needs put and call volume.",
    "vix_term": "Needs VIX and VIX3M.",
}

_CATEGORY = {
    "Options": "Needs option prices, implied volatility, or an option chain.",
    "Fundamental": "Needs fundamental data.",
    "Sentiment": "Needs a sentiment or short-interest series.",
    "Futures/Crypto": "Needs futures, funding, or positioning data.",
    "Relative/Breadth": "Needs a benchmark, a second symbol, or market-wide breadth.",
    "Chart Patterns": "The shape is discretionary, so it is listed here and not turned into a signal. N-bar breakout is available in the builder.",
    "Factor": "Needs fundamental ranks. 12-1 momentum and low volatility are available from price.",
}


@lru_cache(maxsize=1)
def load_document() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def load_catalog() -> list[dict]:
    return list(load_document()["indicators"])


def catalog_notes() -> str:
    return str(load_document().get("notes", ""))


def catalog_categories(items: list[dict] | None = None) -> list[str]:
    seen: list[str] = []
    for item in items if items is not None else load_catalog():
        category = item["category"]
        if category not in seen:
            seen.append(category)
    return seen


def reference_reason(item: dict) -> str:
    specific = _REFERENCE.get(item["id"])
    if specific:
        return specific
    return _CATEGORY.get(
        item["category"],
        "Needs inputs other than a single OHLCV series.",
    )
