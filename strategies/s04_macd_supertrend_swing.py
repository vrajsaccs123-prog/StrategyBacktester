"""MACD and SuperTrend swing. Momentum on the Nifty 500.

Buy when the MACD histogram crosses above zero and SuperTrend(10) is up.
Money flow above its threshold takes full size. Below it, or missing, takes half.
Sell when SuperTrend turns down, or the histogram is negative for two closes.
The fill is the next open. The stop is the SuperTrend line on the signal bar.

The signal ATR is the close minus that line. The size factor is 0.15, so a
full position risks 0.75% of equity over that distance. The first 300 bars are
not bought. One position is capped at 25% of equity. Short signals are ignored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.book import COST_PARAMS
from engine.indicators import compute_indicator, macd
from engine.models import ParamSpec
from engine.strategy import Stage, Strategy, resolve_params
from strategies._desk import WARMUP, book_space, cap_sentence, choice, money_flow, pack, pick, rank_line

MACD_PAIRS = ((12, 26, 9), (8, 21, 5))

PARAM_LABELS = {
    "st_mult": "SuperTrend multiple",
    "macd_set": "MACD set (0 = 12/26/9, 1 = 8/21/5)",
    "mfi_thresh": "Full-size money flow",
    "atr_length": "ATR length",
}

RANK_LABELS = {
    0: "Highest MACD histogram",
    1: "Highest money flow",
    2: "Symbol name",
}

STAGES = (
    Stage(
        "filter",
        "Filter",
        "SuperTrend is up. The first 300 bars are not bought.",
        (("Direction", ("st_mult",)),),
    ),
    Stage(
        "buy",
        "Buy",
        "The MACD histogram crosses above zero while SuperTrend is up. The fill is the next open.",
        (("Cross", ("macd_set", "mfi_thresh")),),
    ),
    Stage(
        "sell",
        "Sell",
        "SuperTrend turns down, or the histogram is negative for two closes. The fill is the next open.",
        (("Reversal", ()),),
    ),
    Stage(
        "stop",
        "Stop loss",
        "The stop is the SuperTrend line from the signal bar.",
        (("Protective stop", ("stop_atr", "partial_pct")),),
    ),
    Stage(
        "size",
        "Position sizing",
        "Full size risks 0.75% of equity over the distance to the SuperTrend line. Quiet money flow uses half.",
        (
            ("Shares", ("atr_length", "risk_per_atr", "max_position_pct", "max_position_inr", "capital")),
            ("Several buys the same morning", ("entry_rank",)),
        ),
    ),
)


def _periods(params: dict) -> tuple[int, int, int]:
    fast, slow, signal = pick(params["macd_set"], MACD_PAIRS)
    return int(fast), int(slow), int(signal)


def stage_line(key: str, params: dict) -> str:
    fast, slow, signal = _periods(params)
    mult = float(params["st_mult"])
    if key == "filter":
        return (
            f"SuperTrend(10, {mult:g}) is up. The first {WARMUP} bars of a name are not bought."
        )
    if key == "buy":
        return (
            f"The MACD histogram ({fast}, {slow}, {signal}) crosses above zero while SuperTrend is up. "
            f"Money flow above {float(params['mfi_thresh']):g} takes full size, otherwise half. "
            "The fill is the next open."
        )
    if key == "sell":
        return (
            "SuperTrend turns down, or the MACD histogram is negative for two closes. "
            "The fill is the next open."
        )
    if key == "stop":
        return (
            "The stop is the SuperTrend line on the signal bar. "
            f"{int(params['partial_pct'])}% of the shares sell there."
        )
    if key == "size":
        return (
            f"Full size is floor((portfolio value / 20) × {float(params['risk_per_atr']):g} "
            "/ (close − SuperTrend)), which is 0.75% of equity over that distance. "
            f"Money flow at or below {float(params['mfi_thresh']):g} multiplies that by one half. "
            f"{cap_sentence(params)} "
            f"Same-morning buys are filled by {rank_line(params, RANK_LABELS)}."
        )
    raise KeyError(key)


def _lines(bars: pd.DataFrame, params: dict):
    fast, slow, signal = _periods(params)
    hist = macd(bars["close"], fast, slow, signal)["hist"]
    trend = compute_indicator(
        "supertrend",
        bars,
        {"period": 10, "multiplier": float(params["st_mult"])},
    )
    flow = money_flow(bars, 14)
    return hist, trend["supertrend"], trend["direction"], flow


def prepare_signals(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    index = bars.index
    if bars.empty:
        return pack(index, [], [], [], [], [])
    hist, line, direction, flow = _lines(bars, params)
    crossed = ((hist > 0) & (hist.shift(1) <= 0)).fillna(False)
    up = (direction > 0).fillna(False)
    two_down = ((hist < 0) & (hist.shift(1) < 0)).fillna(False)
    exit_ = ((direction < 0) | two_down).fillna(False).to_numpy(dtype=bool)
    distance = (bars["close"] - line).to_numpy(dtype=float)
    entry = (crossed & up).to_numpy(dtype=bool) & ~exit_ & np.isfinite(distance) & (distance > 0)
    flow_values = flow.to_numpy(dtype=float)
    size = np.where(np.isfinite(flow_values) & (flow_values > float(params["mfi_thresh"])), 1.0, 0.5)
    return pack(
        index,
        entry,
        exit_,
        distance,
        hist.to_numpy(dtype=float),
        flow_values,
        size=size,
        stop_price=line.to_numpy(dtype=float),
    )


def chart_frame(bars: pd.DataFrame, params: dict):
    fast, slow, signal = _periods(params)
    hist_name = f"MACD {fast}/{slow}/{signal}"
    if bars.empty:
        frame = pd.DataFrame(columns=["SuperTrend", hist_name, "MFI"], index=bars.index)
    else:
        hist, line, _direction, flow = _lines(bars, params)
        frame = pd.DataFrame({"SuperTrend": line, hist_name: hist, "MFI": flow}, index=bars.index)
    kinds = {"SuperTrend": "overlay", hist_name: "oscillator", "MFI": "oscillator"}
    return frame, kinds


class MacdSupertrendSwing(Strategy):
    key = "s04_macd_supertrend_swing"
    name = "MACD + SuperTrend Swing"
    portfolio = True
    stages = STAGES
    cost_params = COST_PARAMS
    param_labels = PARAM_LABELS
    param_choices = {"entry_rank": tuple((RANK_LABELS[index], index) for index in range(3))}

    def param_space(self) -> dict[str, ParamSpec]:
        return book_space(
            {
                "st_mult": ParamSpec(3, 2, 4, 1, "float"),
                "macd_set": choice(0, len(MACD_PAIRS)),
                "mfi_thresh": ParamSpec(40, 40, 50, 10, "float"),
            },
            risk_per_atr=0.15,
            stop_atr=1.0,
            max_position_pct=25.0,
        )

    def target_positions(self, bars: pd.DataFrame, params: dict, start: int = 0) -> pd.Series:
        raise RuntimeError("MACD SuperTrend trades the book. Run it from the bank.")

    def signals(self, bars: pd.DataFrame, params: dict) -> pd.DataFrame:
        return prepare_signals(bars, resolve_params(self.param_space(), params))

    def indicator_frame(self, bars: pd.DataFrame, params: dict):
        return chart_frame(bars, resolve_params(self.param_space(), params))

    def stage_line(self, key: str, params: dict) -> str:
        return stage_line(key, params)
