"""Bollinger range reversion. Mean reversion while the market is chopping.

Buy when the 14-day choppiness index is above its threshold and the close is
under the lower Bollinger band. RSI under 40 takes full size. Otherwise the
position is half size. RSI is not required. Sell when the close is back at the
middle band, or when the time stop has passed since the buy signal. The fill
is the next open.

The signal ATR is 2 times the 14-day ATR. The size factor is 0.2, so full size
risks 1% of equity over that stop. The stop is the same distance under the
fill. The first 300 bars are not bought. One position is capped at 20% of equity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.book import COST_PARAMS
from engine.indicators import atr, bollinger, compute_indicator, rsi
from engine.models import ParamSpec
from engine.strategy import Stage, Strategy, resolve_params
from strategies._desk import WARMUP, book_space, cap_sentence, choice, pack, pick, rank_line

CHOP_LEVELS = (50.0, 55.0, 61.0)
TIME_STOPS = (7, 10, 15)
RSI_CONFIRM = 40.0

PARAM_LABELS = {
    "chop_thresh": "Choppiness minimum (0 = 50, 1 = 55, 2 = 61)",
    "bb_std": "Bollinger standard deviations",
    "time_stop": "Time stop, bars (0 = 7, 1 = 10, 2 = 15)",
    "atr_length": "ATR length",
}

RANK_LABELS = {
    0: "Lowest RSI",
    1: "Furthest below the lower band",
    2: "Symbol name",
}

STAGES = (
    Stage(
        "filter",
        "Filter",
        "Choppiness is above the threshold. The first 300 bars are not bought.",
        (("Range", ("chop_thresh",)),),
    ),
    Stage(
        "buy",
        "Buy",
        "The close is under the lower Bollinger band. The fill is the next open.",
        (("Stretch", ("bb_std",)),),
    ),
    Stage(
        "sell",
        "Sell",
        "The close is back at the middle band, or the time stop has passed. The fill is the next open.",
        (("Mean", ("time_stop",)),),
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
        "Full size risks 1% of equity over the 2 ATR stop. Missing RSI confirmation uses half.",
        (
            ("Shares", ("atr_length", "risk_per_atr", "max_position_pct", "max_position_inr", "capital")),
            ("Several buys the same morning", ("entry_rank",)),
        ),
    ),
)


def _chop(params: dict) -> float:
    return float(pick(params["chop_thresh"], CHOP_LEVELS))


def _hold(params: dict) -> int:
    return int(pick(params["time_stop"], TIME_STOPS))


def stage_line(key: str, params: dict) -> str:
    if key == "filter":
        return (
            f"The 14-day choppiness index is above {_chop(params):g}. "
            f"The first {WARMUP} bars of a name are not bought."
        )
    if key == "buy":
        return (
            f"The close is under the lower Bollinger band of 20 days and {float(params['bb_std']):g} "
            f"standard deviations. RSI(14) under {RSI_CONFIRM:g} takes full size, otherwise half. "
            "The fill is the next open."
        )
    if key == "sell":
        return (
            f"The close is at or above the middle band, or {_hold(params)} bars have passed "
            "since the buy signal. The fill is the next open."
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
            f"RSI at or above {RSI_CONFIRM:g} multiplies that by one half. "
            f"{cap_sentence(params)} "
            f"Same-morning buys are filled by {rank_line(params, RANK_LABELS)}."
        )
    raise KeyError(key)


def _lines(bars: pd.DataFrame, params: dict):
    close = bars["close"]
    width = atr(bars, int(params["atr_length"]))
    bands = bollinger(close, 20, float(params["bb_std"]))
    chop = compute_indicator("choppiness", bars, {"period": 14})["value"]
    strength = rsi(close, 14)
    return width, bands, chop, strength


def prepare_signals(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    index = bars.index
    if bars.empty:
        return pack(index, [], [], [], [], [])
    width, bands, chop, strength = _lines(bars, params)
    close = bars["close"]
    setup = ((chop > _chop(params)) & (close < bands["lower"])).fillna(False)
    aged = setup.shift(_hold(params)).fillna(False)
    back = (close >= bands["mid"]).fillna(False)
    exit_ = (back | aged).to_numpy(dtype=bool)
    entry = setup.to_numpy(dtype=bool) & np.logical_not(back.to_numpy(dtype=bool)) & np.logical_not(aged.to_numpy(dtype=bool))
    distance = (2.0 * width).to_numpy(dtype=float)
    confirmed = strength.to_numpy(dtype=float)
    size = np.where(np.isfinite(confirmed) & (confirmed < RSI_CONFIRM), 1.0, 0.5)
    stretch = ((bands["lower"] - close) / width.replace(0, np.nan)).to_numpy(dtype=float)
    return pack(
        index,
        entry,
        exit_,
        distance,
        (-strength).to_numpy(dtype=float),
        stretch,
        size=size,
        stop_pad=distance,
    )


def chart_frame(bars: pd.DataFrame, params: dict):
    std = float(params["bb_std"])
    upper = f"Bollinger upper {std:g}"
    middle = "Bollinger middle"
    lower = f"Bollinger lower {std:g}"
    if bars.empty:
        frame = pd.DataFrame(columns=[upper, middle, lower, "Choppiness", "RSI"], index=bars.index)
    else:
        _width, bands, chop, strength = _lines(bars, params)
        frame = pd.DataFrame(
            {upper: bands["upper"], middle: bands["mid"], lower: bands["lower"], "Choppiness": chop, "RSI": strength},
            index=bars.index,
        )
    kinds = {upper: "overlay", middle: "overlay", lower: "overlay", "Choppiness": "oscillator", "RSI": "oscillator"}
    return frame, kinds


class BollingerRangeReversion(Strategy):
    key = "s05_bollinger_range_reversion"
    name = "Range-Market Bollinger Reversion"
    portfolio = True
    stages = STAGES
    cost_params = COST_PARAMS
    param_labels = PARAM_LABELS
    param_choices = {"entry_rank": tuple((RANK_LABELS[index], index) for index in range(3))}

    def param_space(self) -> dict[str, ParamSpec]:
        return book_space(
            {
                "chop_thresh": choice(1, len(CHOP_LEVELS)),
                "bb_std": ParamSpec(2.0, 1.8, 2.2, 0.2, "float"),
                "time_stop": choice(1, len(TIME_STOPS)),
            },
            risk_per_atr=0.2,
            stop_atr=1.0,
            max_position_pct=20.0,
        )

    def target_positions(self, bars: pd.DataFrame, params: dict, start: int = 0) -> pd.Series:
        raise RuntimeError("Bollinger reversion trades the book. Run it from the bank.")

    def signals(self, bars: pd.DataFrame, params: dict) -> pd.DataFrame:
        return prepare_signals(bars, resolve_params(self.param_space(), params))

    def indicator_frame(self, bars: pd.DataFrame, params: dict):
        return chart_frame(bars, resolve_params(self.param_space(), params))

    def stage_line(self, key: str, params: dict) -> str:
        return stage_line(key, params)
