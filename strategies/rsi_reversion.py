"""RSI reversal. Buy when RSI leaves oversold, sell when it reaches overbought."""

from __future__ import annotations

import pandas as pd

from engine.indicators import rsi
from engine.models import ParamSpec
from engine.rules import positions_from_flags
from engine.strategy import Strategy


class RsiReversal(Strategy):
    key = "rsi_reversal"
    name = "RSI Reversal"

    def param_space(self) -> dict[str, ParamSpec]:
        return {
            "length": ParamSpec(14, 5, 30, 1, "int"),
            "entry": ParamSpec(30, 10, 45, 1, "int"),
            "exit": ParamSpec(70, 55, 90, 1, "int"),
        }

    def target_positions(self, bars: pd.DataFrame, params: dict, start: int = 0) -> pd.Series:
        length = int(params["length"])
        entry_level = float(params["entry"])
        exit_level = float(params["exit"])
        if entry_level >= exit_level:
            return pd.Series(0, index=bars.index[start:], dtype=int)
        values = rsi(bars["close"], length)
        entry = (values > entry_level) & (values.shift(1) <= entry_level)
        exit_ = (values > exit_level) & (values.shift(1) <= exit_level)
        return positions_from_flags(entry, exit_, start, side="long_only")

    def indicator_frame(self, bars: pd.DataFrame, params: dict):
        values = rsi(bars["close"], int(params["length"]))
        frame = pd.DataFrame({"rsi": values}, index=bars.index)
        return frame, {"rsi": "oscillator"}
