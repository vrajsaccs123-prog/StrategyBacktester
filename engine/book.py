"""Run one coded strategy across a universe of stocks.

Every book strategy has the same shape. `signals` decides filter, buy, and
sell on each name. The simulator applies the stop, the share count, and the
costs. `indicator_frame` is what the chart draws for that name.
"""

from __future__ import annotations

from string import Template

import numpy as np
import pandas as pd

from engine.models import ParamSpec
from engine.portfolio import simulate
from engine.strategy import Strategy, resolve_params

SIZING_PARAMS = (
    "capital",
    "max_position_inr",
    "risk_per_atr",
    "stop_atr",
    "partial_pct",
    "entry_rank",
)

COST_PARAMS = (
    "slippage_bps",
    "brokerage_bps",
    "brokerage_cap_inr",
    "stt_buy_bps",
    "stt_sell_bps",
    "exchange_bps",
    "sebi_bps",
    "stamp_buy_bps",
    "gst_pct",
)

REQUIRED_PARAMS = SIZING_PARAMS + COST_PARAMS

SEARCH_STAGES = ("filter", "buy", "sell", "stop")

def size_unit(params: dict) -> str:
    """Rupee unit in a sizing sentence: portfolio value / 20, times the factor when it is not 1."""

    factor = float(params["risk_per_atr"])
    if factor == 1:
        return "portfolio value / 20"
    return f"(portfolio value / 20) × {factor:g}"


COMMON_LABELS = {
    "capital": "Portfolio capital, rupees",
    "max_position_inr": "Max position, rupees",
    "risk_per_atr": "Factor on portfolio value / 20",
    "stop_atr": "Stop distance, in entry ATRs",
    "partial_pct": "Shares sold at the stop, percent",
    "entry_rank": "Who gets the cash first",
    "slippage_bps": "Slippage, bps",
    "brokerage_bps": "Brokerage, bps",
    "brokerage_cap_inr": "Brokerage cap per order, rupees",
    "stt_buy_bps": "STT on buys, bps",
    "stt_sell_bps": "STT on sells, bps",
    "exchange_bps": "Exchange charge, bps",
    "sebi_bps": "SEBI charge, bps",
    "stamp_buy_bps": "Stamp duty on buys, bps",
    "gst_pct": "GST on brokerage and exchange charges, percent",
}


def run_book(
    strategy: Strategy,
    data: dict[str, pd.DataFrame],
    params: dict | None = None,
    start=None,
    end=None,
    progress=None,
):
    """Backtest every name as one account.

    Signals are built on bars up to `end`, so a later window is not visible.
    Entries and exits before `start` are dropped. Indicator history inside
    that slice is kept. Fills happen on the next traded open.
    """

    resolved = resolve_params(strategy.param_space(), params or {})
    signals: dict[str, pd.DataFrame] = {}
    names = list(data)
    for index, symbol in enumerate(names, start=1):
        frame = data[symbol]
        if end is not None:
            frame = frame.loc[: pd.Timestamp(end)]
        signals[symbol] = strategy.signals(frame, resolved)
        if start is not None:
            flagged = signals[symbol]
            flagged.loc[flagged.index < pd.Timestamp(start), ["entry", "exit"]] = False
        if progress is not None:
            progress(index, len(names))
    window: dict[str, pd.DataFrame] = {}
    for symbol, frame in data.items():
        sliced = frame
        if start is not None:
            sliced = sliced.loc[pd.Timestamp(start) :]
        if end is not None:
            sliced = sliced.loc[: pd.Timestamp(end)]
        if len(sliced):
            window[symbol] = sliced
            signals[symbol] = signals[symbol].reindex(sliced.index)
    return simulate(window, signals, resolved)


def param_label(strategy: Strategy, name: str) -> str:
    labels = getattr(strategy, "param_labels", {}) or {}
    return labels.get(name) or COMMON_LABELS.get(name) or name


def searchable_names(strategy: Strategy, stages: tuple[str, ...] = SEARCH_STAGES) -> list[str]:
    """Parameters a study searches by default: signal stages with a range."""

    space = strategy.param_space()
    chosen = []
    allowed = set(stages)
    for stage in strategy.stages:
        if stage.key not in allowed:
            continue
        for name in stage.params:
            spec = space.get(name)
            if spec is not None and spec.searchable and name not in chosen:
                chosen.append(name)
    return chosen


def indicator_names(strategy: Strategy) -> list[str]:
    """Indicator settings the train window may choose.

    Capital, costs, the stop, the partial stop, and entry rank stay at the file default.
    ATR length is included when it has a range, including when the size stage owns it.
    """

    space = strategy.param_space()
    fixed = set(SIZING_PARAMS) | set(COST_PARAMS)
    chosen = []
    for stage in strategy.stages:
        if stage.key not in {"filter", "buy", "sell"}:
            continue
        for name in stage.params:
            spec = space.get(name)
            if name in fixed or name in chosen or spec is None or not spec.searchable:
                continue
            chosen.append(name)
    atr_length = space.get("atr_length")
    if atr_length is not None and atr_length.searchable and "atr_length" not in chosen:
        chosen.append("atr_length")
    return chosen


def check_book_strategy(strategy: Strategy) -> list[str]:
    """Problems that would keep the desk from running or drawing the strategy."""

    problems: list[str] = []
    stages = getattr(strategy, "stages", ()) or ()
    keys = [getattr(stage, "key", "") for stage in stages]
    for required in ("filter", "buy", "sell", "size"):
        if required not in keys:
            problems.append(f"Add a '{required}' stage.")
    if not getattr(strategy, "cost_params", ()):
        problems.append("Set cost_params to the brokerage, tax, and slippage names.")
    try:
        space = strategy.param_space()
    except Exception as exc:
        problems.append(f"param_space failed: {exc}")
        return problems
    for name in REQUIRED_PARAMS:
        if name not in space:
            problems.append(f"param_space is missing '{name}'. The book needs it to size and charge fills.")
    if all(required in keys for required in ("filter", "buy", "sell", "size")) and not indicator_names(strategy):
        problems.append("Give an indicator parameter a min and a max. The train window chooses those settings.")
    if not callable(getattr(strategy, "signals", None)):
        problems.append("Add signals(bars, params) with columns entry, exit, and atr.")
        return problems
    try:
        problems.extend(_smoke(strategy, space))
    except Exception as exc:
        problems.append(f"A sample series failed: {exc}")
    return problems


def _smoke(strategy: Strategy, space: dict[str, ParamSpec]) -> list[str]:
    problems = []
    index = pd.bdate_range("2020-01-02", periods=90)
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, len(index))))
    open_ = np.r_[close[0], close[:-1]]
    bars = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.01,
            "low": np.minimum(open_, close) * 0.99,
            "close": close,
            "volume": 1_000.0,
        },
        index=index,
    )
    params = resolve_params(space, None)
    frame = strategy.signals(bars, params)
    if not isinstance(frame, pd.DataFrame):
        return ["signals must return a DataFrame indexed like the bars."]
    for column in ("entry", "exit", "atr"):
        if column not in frame.columns:
            problems.append(f"signals is missing the '{column}' column.")
    if len(frame) and not frame.index.equals(bars.index):
        problems.append("signals must use the same index as the bars.")
    plotted, kinds = strategy.indicator_frame(bars, params)
    if not isinstance(kinds, dict) or not kinds:
        problems.append("indicator_frame must return a kind for every series the chart should draw.")
        return problems
    if not isinstance(plotted, pd.DataFrame):
        problems.append("indicator_frame must return a DataFrame of indicator values.")
        return problems
    for name, kind in kinds.items():
        if kind not in {"overlay", "oscillator"}:
            problems.append(f"Indicator '{name}' must be 'overlay' or 'oscillator', not '{kind}'.")
        elif name not in plotted.columns:
            problems.append(f"indicator_frame names '{name}' but does not return that column.")
    return problems


_SKELETON = Template(
    '''"""$title. Filter, buy, sell, size, and costs on the whole universe."""

from __future__ import annotations

import pandas as pd

from engine.book import COST_PARAMS
from engine.indicators import atr, sma
from engine.models import ParamSpec
from engine.strategy import Stage, Strategy, resolve_params


class $class_name(Strategy):
    key = "$key"
    name = "$title"
    stages = (
        Stage(
            "filter",
            "Filter",
            "The close is above the slow average, so a falling name cannot be bought.",
            (("Trend", ("slow",)),),
        ),
        Stage(
            "buy",
            "Buy",
            "The fast average crosses above the slow average. The fill is the next open.",
            (("Cross", ("fast", "atr_length")),),
        ),
        Stage(
            "sell",
            "Sell",
            "The fast average crosses below the slow average. The fill is the next open.",
            (("Cross", ()),),
        ),
        Stage(
            "stop",
            "Stop loss",
            "The stop is frozen under the entry fill. On an open where a stop and a sell both fill, the stop sells first.",
            (("Protective stop", ("stop_atr", "partial_pct")),),
        ),
        Stage(
            "size",
            "Position sizing",
            "Shares are the portfolio value divided by 20, times the size factor, divided by ATR, then cut by the rupee cap and by cash.",
            (("Shares", ("risk_per_atr", "max_position_inr", "capital", "entry_rank")),),
        ),
    )
    cost_params = COST_PARAMS
    param_labels = {
        "fast": "Fast average",
        "slow": "Slow average",
        "atr_length": "ATR length",
    }

    def param_space(self) -> dict[str, ParamSpec]:
        return {
            "fast": ParamSpec(10, 5, 30, 1, "int"),
            "slow": ParamSpec(30, 20, 80, 5, "int"),
            "atr_length": ParamSpec(14, 5, 30, 1, "int"),
            "stop_atr": ParamSpec(2.0, 0.5, 5.0, 0.5, "float"),
            "partial_pct": ParamSpec(100, 0, 100, 10, "int"),
            "capital": ParamSpec(20_000_000, None, None, None, "int"),
            "max_position_inr": ParamSpec(1_000_000, None, None, None, "int"),
            "risk_per_atr": ParamSpec(1, None, None, None, "float"),
            "entry_rank": ParamSpec(2, 0, 2, 1, "int"),
            "slippage_bps": ParamSpec(5, None, None, None, "float"),
            "brokerage_bps": ParamSpec(0, None, None, None, "float"),
            "brokerage_cap_inr": ParamSpec(20, None, None, None, "float"),
            "stt_buy_bps": ParamSpec(10, None, None, None, "float"),
            "stt_sell_bps": ParamSpec(10, None, None, None, "float"),
            "exchange_bps": ParamSpec(0.297, None, None, None, "float"),
            "sebi_bps": ParamSpec(0.01, None, None, None, "float"),
            "stamp_buy_bps": ParamSpec(1.5, None, None, None, "float"),
            "gst_pct": ParamSpec(18, None, None, None, "float"),
        }

    def _lines(self, bars: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series, pd.Series]:
        fast_n = int(params["fast"])
        slow_n = int(params["slow"])
        fast = sma(bars["close"], fast_n)
        slow = sma(bars["close"], slow_n)
        width = atr(bars, int(params["atr_length"]))
        return fast, slow, width

    def signals(self, bars: pd.DataFrame, params: dict) -> pd.DataFrame:
        resolved = resolve_params(self.param_space(), params)
        fast, slow, width = self._lines(bars, resolved)
        tradable = (bars["close"] > slow) & width.gt(0)
        crossed = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        lost = (fast < slow) & (fast.shift(1) >= slow.shift(1))
        if int(resolved["fast"]) >= int(resolved["slow"]):
            crossed = crossed & False
        gap = (fast - slow) / width
        return pd.DataFrame(
            {
                "entry": (crossed & tradable).fillna(False),
                "exit": lost.fillna(False),
                "atr": width,
                "udr": gap,
                "spread_atr": gap,
            },
            index=bars.index,
        )

    def indicator_frame(self, bars: pd.DataFrame, params: dict):
        resolved = resolve_params(self.param_space(), params)
        fast, slow, width = self._lines(bars, resolved)
        frame = pd.DataFrame(
            {"Fast average": fast, "Slow average": slow, "ATR": width},
            index=bars.index,
        )
        kinds = {"Fast average": "overlay", "Slow average": "overlay", "ATR": "oscillator"}
        return frame, kinds
'''
)


def new_strategy_source(class_name: str, key: str, title: str) -> str:
    return _SKELETON.substitute(class_name=class_name, key=key, title=title)
