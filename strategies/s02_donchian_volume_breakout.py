"""Donchian breakout with volume. Momentum on the Nifty 500.

Buy when the close is above the highest high of the prior entry window.
Relative volume below the threshold takes half size and does not skip the trade.
Sell when the close is under the lowest low of the prior exit window, or under
the highest close of the current breakout minus 2.5 ATRs. The peak resets when
the close is back under that exit channel. The fill is the next open.

The signal ATR is 2.5 times the 14-day ATR. The size factor is 0.2, so a full
position risks 1% of equity over the stop. The stop is that distance under the
fill. The first 300 bars are not bought. One position is capped at 25% of equity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.book import COST_PARAMS
from engine.indicators import atr
from engine.models import ParamSpec
from engine.strategy import Stage, Strategy, resolve_params
from strategies._desk import WARMUP, book_space, cap_sentence, choice, pack, pick, rank_line, relative_volume, rising_max

ENTRY_WINDOWS = (20, 40, 55)
TRAIL = 2.5

PARAM_LABELS = {
    "entry_n": "Breakout window (0 = 20, 1 = 40, 2 = 55)",
    "exit_n": "Exit channel",
    "rvol_min": "Full-size relative volume",
    "atr_length": "ATR length",
}

RANK_LABELS = {
    0: "Largest breakout per ATR",
    1: "Highest relative volume",
    2: "Symbol name",
}

STAGES = (
    Stage(
        "filter",
        "Filter",
        "The first 300 bars of a name are not bought. Volume never vetoes a breakout.",
        (("History", ()),),
    ),
    Stage(
        "buy",
        "Buy",
        "The close is above the prior high. Low relative volume buys half size. The fill is the next open.",
        (("Breakout", ("entry_n", "rvol_min")),),
    ),
    Stage(
        "sell",
        "Sell",
        "The close is under the prior low, or under the breakout peak minus 2.5 ATRs. The fill is the next open.",
        (("Channel", ("exit_n",)),),
    ),
    Stage(
        "stop",
        "Stop loss",
        "The stop is 2.5 ATRs under the fill and sells the whole position.",
        (("Protective stop", ("stop_atr", "partial_pct")),),
    ),
    Stage(
        "size",
        "Position sizing",
        "Full size risks 1% of equity over the 2.5 ATR stop. Low relative volume uses half.",
        (
            ("Shares", ("atr_length", "risk_per_atr", "max_position_pct", "max_position_inr", "capital")),
            ("Several buys the same morning", ("entry_rank",)),
        ),
    ),
)


def _entry_window(params: dict) -> int:
    return int(pick(params["entry_n"], ENTRY_WINDOWS))


def stage_line(key: str, params: dict) -> str:
    entry_n = _entry_window(params)
    exit_n = int(params["exit_n"])
    if key == "filter":
        return f"The first {WARMUP} bars of a name are not bought. A quiet bar is bought at half size."
    if key == "buy":
        return (
            f"The close is above the highest high of the prior {entry_n} bars. "
            f"Relative volume below {float(params['rvol_min']):g} buys half size. The fill is the next open."
        )
    if key == "sell":
        return (
            f"The close is under the lowest low of the prior {exit_n} bars, "
            f"or under the highest close of this breakout minus {TRAIL:g} ATRs. The fill is the next open."
        )
    if key == "stop":
        return (
            f"The stop is {TRAIL:g} times the {int(params['atr_length'])}-day ATR under the fill. "
            f"{int(params['partial_pct'])}% of the shares sell there."
        )
    if key == "size":
        return (
            f"Full size is floor((portfolio value / 20) × {float(params['risk_per_atr']):g} "
            f"/ ({TRAIL:g} × {int(params['atr_length'])}-day ATR)), which is 1% of equity over the stop. "
            f"Relative volume below {float(params['rvol_min']):g} multiplies that by one half. "
            f"{cap_sentence(params)} "
            f"Same-morning buys are filled by {rank_line(params, RANK_LABELS)}."
        )
    raise KeyError(key)


def _lines(bars: pd.DataFrame, params: dict):
    entry_n = _entry_window(params)
    exit_n = int(params["exit_n"])
    length = int(params["atr_length"])
    width = atr(bars, length)
    upper = bars["high"].rolling(entry_n, min_periods=entry_n).max().shift(1)
    lower = bars["low"].rolling(exit_n, min_periods=exit_n).min().shift(1)
    rvol = relative_volume(bars["volume"].astype(float), 20)
    return width, upper, lower, rvol


def prepare_signals(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    index = bars.index
    if bars.empty:
        return pack(index, [], [], [], [], [])
    width, upper, lower, rvol = _lines(bars, params)
    close = bars["close"]
    above = (close > upper).fillna(False).to_numpy(dtype=bool)
    below = (close < lower).fillna(False).to_numpy(dtype=bool)
    active = np.logical_not(below).copy()
    if len(active) > WARMUP:
        active[:WARMUP] = False
    else:
        active[:] = False
    peak = rising_max(close.to_numpy(dtype=float), active)
    gap = (TRAIL * width).to_numpy(dtype=float)
    trailed = np.isfinite(peak) & np.isfinite(gap) & (close.to_numpy(dtype=float) < peak - gap)
    exit_ = below | trailed
    entry = above & ~exit_
    rvol_values = rvol.to_numpy(dtype=float)
    size = np.where(np.isfinite(rvol_values) & (rvol_values >= float(params["rvol_min"])), 1.0, 0.5)
    distance = gap
    stretch = ((close - upper) / width.replace(0, np.nan)).to_numpy(dtype=float)
    return pack(index, entry, exit_, distance, stretch, rvol_values, size=size, stop_pad=distance)


def chart_frame(bars: pd.DataFrame, params: dict):
    entry_n = _entry_window(params)
    exit_n = int(params["exit_n"])
    high_name = f"{entry_n}-day high"
    low_name = f"{exit_n}-day low"
    if bars.empty:
        frame = pd.DataFrame(columns=[high_name, low_name, "Relative volume", "ATR"], index=bars.index)
    else:
        width, upper, lower, rvol = _lines(bars, params)
        frame = pd.DataFrame(
            {high_name: upper, low_name: lower, "Relative volume": rvol, "ATR": width},
            index=bars.index,
        )
    kinds = {high_name: "overlay", low_name: "overlay", "Relative volume": "oscillator", "ATR": "oscillator"}
    return frame, kinds


class DonchianVolumeBreakout(Strategy):
    key = "s02_donchian_volume_breakout"
    name = "Donchian Breakout with Volume"
    portfolio = True
    stages = STAGES
    cost_params = COST_PARAMS
    param_labels = PARAM_LABELS
    param_choices = {"entry_rank": tuple((RANK_LABELS[index], index) for index in range(3))}

    def param_space(self) -> dict[str, ParamSpec]:
        return book_space(
            {
                "entry_n": choice(0, len(ENTRY_WINDOWS)),
                "exit_n": ParamSpec(10, 10, 20, 10, "int"),
                "rvol_min": ParamSpec(1.0, 1.0, 1.2, 0.2, "float"),
            },
            risk_per_atr=0.2,
            stop_atr=1.0,
            max_position_pct=25.0,
        )

    def target_positions(self, bars: pd.DataFrame, params: dict, start: int = 0) -> pd.Series:
        raise RuntimeError("Donchian breakout trades the book. Run it from the bank.")

    def signals(self, bars: pd.DataFrame, params: dict) -> pd.DataFrame:
        return prepare_signals(bars, resolve_params(self.param_space(), params))

    def indicator_frame(self, bars: pd.DataFrame, params: dict):
        return chart_frame(bars, resolve_params(self.param_space(), params))

    def stage_line(self, key: str, params: dict) -> str:
        return stage_line(key, params)
