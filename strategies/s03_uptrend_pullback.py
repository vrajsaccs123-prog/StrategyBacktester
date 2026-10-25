"""Uptrend pullback. A dip inside a rising name on the Nifty 500.

Buy when the close is above the 200-day average and either the short RSI is
under its threshold or internal bar strength is under 0.2. One of those is
enough. Sell when the close is back above the short average, or 7 bars after
the buy signal. The fill is the next open.

Each position is 20% of equity: the signal ATR column is the close and the
size factor is 4, so shares are floor((portfolio value / 20) × 4 / close).
At most 5 positions are open. The same morning, the lowest RSI is filled first.
The disaster stop is 3 ATRs under the fill. The first 300 bars are not bought.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.book import COST_PARAMS
from engine.indicators import atr, rsi, sma
from engine.models import ParamSpec
from engine.strategy import Stage, Strategy, resolve_params
from strategies._desk import WARMUP, book_space, cap_sentence, pack, rank_line

HOLD_BARS = 7
STOP_MULT = 3.0
IBS_MAX = 0.2

PARAM_LABELS = {
    "rsi_period": "RSI length",
    "rsi_thresh": "RSI below",
    "exit_ma": "Exit average",
    "trend_ma": "Trend average",
    "hold_bars": "Bars held before a time exit",
    "atr_length": "ATR length",
}

RANK_LABELS = {
    0: "Lowest RSI",
    1: "Lowest internal bar strength",
    2: "Symbol name",
}

STAGES = (
    Stage(
        "filter",
        "Filter",
        "The close is above the 200-day average. The first 300 bars are not bought.",
        (("Trend", ("trend_ma",)),),
    ),
    Stage(
        "buy",
        "Buy",
        "The short RSI is under its threshold or internal bar strength is under 0.2. The fill is the next open.",
        (("Pullback", ("rsi_period", "rsi_thresh")),),
    ),
    Stage(
        "sell",
        "Sell",
        "The close is above the short average, or 7 bars have passed since the buy signal. The fill is the next open.",
        (("Bounce", ("exit_ma", "hold_bars")),),
    ),
    Stage(
        "stop",
        "Stop loss",
        "A disaster stop 3 ATRs under the fill sells the whole position.",
        (("Protective stop", ("stop_atr", "partial_pct")),),
    ),
    Stage(
        "size",
        "Position sizing",
        "Each position is 20% of equity, with at most 5 open. The lowest RSI is filled first.",
        (
            ("Shares", ("atr_length", "risk_per_atr", "max_position_pct", "max_positions", "max_position_inr", "capital")),
            ("Several buys the same morning", ("entry_rank",)),
        ),
    ),
)


def stage_line(key: str, params: dict) -> str:
    if key == "filter":
        return (
            f"The close is above the {int(params['trend_ma'])}-day average. "
            f"The first {WARMUP} bars of a name are not bought."
        )
    if key == "buy":
        return (
            f"The {int(params['rsi_period'])}-day RSI is under {float(params['rsi_thresh']):g}, "
            f"or internal bar strength is under {IBS_MAX:g}. The fill is the next open."
        )
    if key == "sell":
        return (
            f"The close is above the {int(params['exit_ma'])}-day average, "
            f"or {int(params['hold_bars'])} bars have passed since the buy signal. The fill is the next open."
        )
    if key == "stop":
        return (
            f"The disaster stop is {STOP_MULT:g} times the {int(params['atr_length'])}-day ATR under the fill. "
            f"{int(params['partial_pct'])}% of the shares sell there."
        )
    if key == "size":
        return (
            f"Shares are floor((portfolio value / 20) × {float(params['risk_per_atr']):g} / close), "
            f"which is {float(params['max_position_pct']):g}% of equity. "
            f"{cap_sentence(params)} "
            f"Same-morning buys are filled by {rank_line(params, RANK_LABELS)}."
        )
    raise KeyError(key)


def _ibs(bars: pd.DataFrame) -> pd.Series:
    span = (bars["high"] - bars["low"]).replace(0, np.nan)
    return (bars["close"] - bars["low"]) / span


def _lines(bars: pd.DataFrame, params: dict):
    close = bars["close"]
    width = atr(bars, int(params["atr_length"]))
    trend = sma(close, int(params["trend_ma"]))
    exit_average = sma(close, int(params["exit_ma"]))
    strength = rsi(close, int(params["rsi_period"]))
    ibs = _ibs(bars)
    return width, trend, exit_average, strength, ibs


def prepare_signals(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    index = bars.index
    if bars.empty:
        return pack(index, [], [], [], [], [])
    width, trend, exit_average, strength, ibs = _lines(bars, params)
    close = bars["close"]
    pullback = (strength < float(params["rsi_thresh"])) | (ibs < IBS_MAX)
    setup = ((close > trend) & pullback).fillna(False)
    aged = setup.shift(int(params["hold_bars"])).fillna(False)
    above = (close > exit_average).fillna(False)
    exit_ = (above | aged).to_numpy(dtype=bool)
    entry = (
        setup.to_numpy(dtype=bool)
        & np.logical_not(above.to_numpy(dtype=bool))
        & np.logical_not(aged.to_numpy(dtype=bool))
        & width.notna().to_numpy(dtype=bool)
    )
    stop_pad = (STOP_MULT * width).to_numpy(dtype=float)
    return pack(
        index,
        entry,
        exit_,
        close.to_numpy(dtype=float),
        (-strength).to_numpy(dtype=float),
        (-ibs).to_numpy(dtype=float),
        stop_pad=stop_pad,
    )


def chart_frame(bars: pd.DataFrame, params: dict):
    trend_name = f"SMA {int(params['trend_ma'])}"
    exit_name = f"SMA {int(params['exit_ma'])}"
    if trend_name == exit_name:
        trend_name = f"Trend SMA {int(params['trend_ma'])}"
        exit_name = f"Exit SMA {int(params['exit_ma'])}"
    if bars.empty:
        frame = pd.DataFrame(columns=[trend_name, exit_name, "RSI", "IBS"], index=bars.index)
    else:
        _width, trend, exit_average, strength, ibs = _lines(bars, params)
        frame = pd.DataFrame(
            {trend_name: trend, exit_name: exit_average, "RSI": strength, "IBS": ibs},
            index=bars.index,
        )
    kinds = {trend_name: "overlay", exit_name: "overlay", "RSI": "oscillator", "IBS": "oscillator"}
    return frame, kinds


class UptrendPullback(Strategy):
    key = "s03_uptrend_pullback"
    name = "Uptrend Pullback Buyer"
    portfolio = True
    stages = STAGES
    cost_params = COST_PARAMS
    param_labels = PARAM_LABELS
    param_choices = {"entry_rank": tuple((RANK_LABELS[index], index) for index in range(3))}

    def param_space(self) -> dict[str, ParamSpec]:
        return book_space(
            {
                "rsi_period": ParamSpec(3, 2, 4, 1, "int"),
                "rsi_thresh": ParamSpec(20, 10, 30, 10, "float"),
                "exit_ma": ParamSpec(5, 5, 10, 5, "int"),
                "trend_ma": ParamSpec(200, None, None, None, "int"),
                "hold_bars": ParamSpec(HOLD_BARS, None, None, None, "int"),
            },
            risk_per_atr=4.0,
            stop_atr=1.0,
            max_position_pct=20.0,
            max_positions=5,
        )

    def target_positions(self, bars: pd.DataFrame, params: dict, start: int = 0) -> pd.Series:
        raise RuntimeError("Uptrend pullback trades the book. Run it from the bank.")

    def signals(self, bars: pd.DataFrame, params: dict) -> pd.DataFrame:
        return prepare_signals(bars, resolve_params(self.param_space(), params))

    def indicator_frame(self, bars: pd.DataFrame, params: dict):
        return chart_frame(bars, resolve_params(self.param_space(), params))

    def stage_line(self, key: str, params: dict) -> str:
        return stage_line(key, params)
