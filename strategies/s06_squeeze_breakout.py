"""Volatility squeeze breakout. Long only on the Nifty 500.

Buy when Bollinger bands leave the Keltner channel after at least the required
number of squeeze bars, and squeeze momentum is positive. Chaikin money flow
above zero takes full size. Otherwise the position is half size. Sell when
momentum turns negative, or the close is under the highest close of the
positive-momentum run minus the trail multiple. The fill is the next open.

The initial stop, and the risk distance, are 2 times the 14-day ATR. The size
factor is 0.2, so full size risks 1% of equity over that stop. The trail is a
separate exit. The first 300 bars are not bought. One position is capped at
25% of equity. The short mirror is not traded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.book import COST_PARAMS
from engine.indicators import atr, bollinger, ema, sma
from engine.models import ParamSpec
from engine.strategy import Stage, Strategy, resolve_params
from strategies._desk import (
    WARMUP,
    book_space,
    cap_sentence,
    chaikin,
    choice,
    pack,
    pick,
    rank_line,
    rising_max,
    rolling_linreg_end,
    run_length,
)

SQUEEZE_BARS = (3, 5, 8)
BB_LENGTH = 20

PARAM_LABELS = {
    "min_squeeze_bars": "Squeeze bars before a release (0 = 3, 1 = 5, 2 = 8)",
    "kc_mult": "Keltner multiple",
    "atr_trail": "Trail, in ATRs",
    "atr_length": "ATR length",
}

RANK_LABELS = {
    0: "Strongest squeeze momentum",
    1: "Highest Chaikin money flow",
    2: "Symbol name",
}

STAGES = (
    Stage(
        "filter",
        "Filter",
        "A release follows a squeeze of at least the required bars. The first 300 bars are not bought.",
        (("Compression", ("min_squeeze_bars", "kc_mult")),),
    ),
    Stage(
        "buy",
        "Buy",
        "The squeeze has just released and momentum is positive. The fill is the next open.",
        (("Release", ()),),
    ),
    Stage(
        "sell",
        "Sell",
        "Momentum turns negative, or the close gives back the ATR trail. The fill is the next open.",
        (("Giveback", ("atr_trail",)),),
    ),
    Stage(
        "stop",
        "Stop loss",
        "The stop is 2 ATRs under the fill and sells the whole position.",
        (("Protective stop", ("stop_atr", "partial_pct")),),
    ),
    Stage(
        "size",
        "Position sizing",
        "Full size risks 1% of equity over the 2 ATR stop. Negative money flow uses half.",
        (
            ("Shares", ("atr_length", "risk_per_atr", "max_position_pct", "max_position_inr", "capital")),
            ("Several buys the same morning", ("entry_rank",)),
        ),
    ),
)


def _min_bars(params: dict) -> int:
    return int(pick(params["min_squeeze_bars"], SQUEEZE_BARS))


def stage_line(key: str, params: dict) -> str:
    if key == "filter":
        return (
            f"Bollinger(20, 2) was inside Keltner(EMA 20, ATR 10, {float(params['kc_mult']):g}) "
            f"for at least {_min_bars(params)} bars. The first {WARMUP} bars of a name are not bought."
        )
    if key == "buy":
        return (
            "The squeeze releases and 20-bar regression momentum is positive. "
            "Chaikin money flow above zero takes full size, otherwise half. The fill is the next open."
        )
    if key == "sell":
        return (
            "Momentum turns negative, or the close is under the highest close of the positive-momentum run "
            f"minus {float(params['atr_trail']):g} ATRs. The fill is the next open."
        )
    if key == "stop":
        return (
            f"The stop is 2 times the {int(params['atr_length'])}-day ATR under the fill. "
            f"{int(params['partial_pct'])}% of the shares sell there."
        )
    if key == "size":
        return (
            f"Full size is floor((portfolio value / 20) × {float(params['risk_per_atr']):g} "
            f"/ (2 × {int(params['atr_length'])}-day ATR)), which is 1% of equity over the stop. "
            "Chaikin money flow at or below zero multiplies that by one half. "
            f"{cap_sentence(params)} "
            f"Same-morning buys are filled by {rank_line(params, RANK_LABELS)}."
        )
    raise KeyError(key)


def _lines(bars: pd.DataFrame, params: dict):
    close = bars["close"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    width = atr(bars, int(params["atr_length"]))
    bands = bollinger(close, BB_LENGTH, 2.0)
    center = ema(close, 20)
    channel = float(params["kc_mult"]) * atr(bars, 10)
    upper = center + channel
    lower = center - channel
    squeezed = ((bands["upper"] < upper) & (bands["lower"] > lower)).fillna(False)
    highest = high.rolling(BB_LENGTH, min_periods=BB_LENGTH).max()
    lowest = low.rolling(BB_LENGTH, min_periods=BB_LENGTH).min()
    basis = ((highest + lowest) / 2.0 + sma(close, BB_LENGTH)) / 2.0
    momentum = pd.Series(rolling_linreg_end((close - basis).to_numpy(dtype=float), BB_LENGTH), index=bars.index)
    flow = chaikin(bars, 20)
    return width, bands, upper, lower, squeezed, momentum, flow


def prepare_signals(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    index = bars.index
    if bars.empty:
        return pack(index, [], [], [], [], [])
    width, _bands, _upper, _lower, squeezed, momentum, flow = _lines(bars, params)
    flags = squeezed.to_numpy(dtype=bool)
    count = run_length(flags)
    previous = np.zeros(len(flags), dtype=bool)
    if len(flags) > 1:
        previous[1:] = flags[:-1] & (count[:-1] >= _min_bars(params))
    mom = momentum.to_numpy(dtype=float)
    released = previous & ~flags & np.isfinite(mom) & (mom > 0)
    positive = np.isfinite(mom) & (mom < 0)
    active = np.array(np.isfinite(mom) & (mom > 0), dtype=bool, copy=True)
    if len(active) > WARMUP:
        active[:WARMUP] = False
    else:
        active[:] = False
    peak = rising_max(bars["close"].to_numpy(dtype=float), active)
    gap = (float(params["atr_trail"]) * width).to_numpy(dtype=float)
    trailed = np.isfinite(peak) & np.isfinite(gap) & (bars["close"].to_numpy(dtype=float) < peak - gap)
    exit_ = positive | trailed
    entry = released & ~exit_
    distance = (2.0 * width).to_numpy(dtype=float)
    flow_values = flow.to_numpy(dtype=float)
    size = np.where(np.isfinite(flow_values) & (flow_values > 0), 1.0, 0.5)
    return pack(index, entry, exit_, distance, mom, flow_values, size=size, stop_pad=distance)


def chart_frame(bars: pd.DataFrame, params: dict):
    mult = float(params["kc_mult"])
    upper = f"Keltner upper {mult:g}"
    lower = f"Keltner lower {mult:g}"
    if bars.empty:
        columns = ["Bollinger upper", "Bollinger lower", upper, lower, "Squeeze momentum", "CMF"]
        frame = pd.DataFrame(columns=columns, index=bars.index)
    else:
        _width, bands, kc_upper, kc_lower, _squeezed, momentum, flow = _lines(bars, params)
        frame = pd.DataFrame(
            {
                "Bollinger upper": bands["upper"],
                "Bollinger lower": bands["lower"],
                upper: kc_upper,
                lower: kc_lower,
                "Squeeze momentum": momentum,
                "CMF": flow,
            },
            index=bars.index,
        )
    kinds = {
        "Bollinger upper": "overlay",
        "Bollinger lower": "overlay",
        upper: "overlay",
        lower: "overlay",
        "Squeeze momentum": "oscillator",
        "CMF": "oscillator",
    }
    return frame, kinds


class SqueezeBreakout(Strategy):
    key = "s06_squeeze_breakout"
    name = "Volatility Squeeze Breakout"
    portfolio = True
    stages = STAGES
    cost_params = COST_PARAMS
    param_labels = PARAM_LABELS
    param_choices = {"entry_rank": tuple((RANK_LABELS[index], index) for index in range(3))}

    def param_space(self) -> dict[str, ParamSpec]:
        return book_space(
            {
                "min_squeeze_bars": choice(1, len(SQUEEZE_BARS)),
                "kc_mult": ParamSpec(1.5, 1.5, 2.0, 0.5, "float"),
                "atr_trail": ParamSpec(3.0, 2.5, 3.0, 0.5, "float"),
            },
            risk_per_atr=0.2,
            stop_atr=1.0,
            max_position_pct=25.0,
        )

    def target_positions(self, bars: pd.DataFrame, params: dict, start: int = 0) -> pd.Series:
        raise RuntimeError("Squeeze breakout trades the book. Run it from the bank.")

    def signals(self, bars: pd.DataFrame, params: dict) -> pd.DataFrame:
        return prepare_signals(bars, resolve_params(self.param_space(), params))

    def indicator_frame(self, bars: pd.DataFrame, params: dict):
        return chart_frame(bars, resolve_params(self.param_space(), params))

    def stage_line(self, key: str, params: dict) -> str:
        return stage_line(key, params)
