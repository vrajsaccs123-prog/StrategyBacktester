"""Bollinger reversal. Buy a close below the lower band, sell a close back through the middle."""

from __future__ import annotations

import pandas as pd

from engine.indicators import bollinger
from engine.models import ParamSpec
from engine.rules import positions_from_flags
from engine.strategy import Strategy


class BollingerReversion(Strategy):
    key = "bollinger"
    name = "Bollinger Reversion"

    def param_space(self) -> dict[str, ParamSpec]:
        return {
            "length": ParamSpec(20, 10, 40, 1, "int"),
            "std": ParamSpec(2.0, 1.0, 3.0, 0.25, "float"),
        }

    def _bands(self, bars: pd.DataFrame, params: dict) -> pd.DataFrame:
        return bollinger(bars["close"], int(params["length"]), float(params["std"]))

    def target_positions(self, bars: pd.DataFrame, params: dict, start: int = 0) -> pd.Series:
        bands = self._bands(bars, params)
        close = bars["close"]
        entry = (close < bands["lower"]) & (close.shift(1) >= bands["lower"].shift(1))
        exit_ = (close > bands["mid"]) & (close.shift(1) <= bands["mid"].shift(1))
        return positions_from_flags(entry, exit_, start, side="long_only")

    def indicator_frame(self, bars: pd.DataFrame, params: dict):
        bands = self._bands(bars, params)
        frame = bands.rename(columns={"mid": "mid", "upper": "upper", "lower": "lower"})
        return frame, {"mid": "overlay", "upper": "overlay", "lower": "overlay"}
