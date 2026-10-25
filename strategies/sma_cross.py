"""Moving-average crossover. Long when the fast average crosses above the slow average."""

from __future__ import annotations

import pandas as pd

from engine.indicators import sma
from engine.models import ParamSpec
from engine.rules import positions_from_flags
from engine.strategy import Strategy


class SmaCross(Strategy):
    key = "sma_cross"
    name = "SMA Cross"

    def param_space(self) -> dict[str, ParamSpec]:
        return {
            "fast": ParamSpec(10, 5, 40, 1, "int"),
            "slow": ParamSpec(30, 20, 120, 5, "int"),
        }

    def _averages(self, bars: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series, int, int]:
        fast_n = int(params["fast"])
        slow_n = int(params["slow"])
        return sma(bars["close"], fast_n), sma(bars["close"], slow_n), fast_n, slow_n

    def target_positions(self, bars: pd.DataFrame, params: dict, start: int = 0) -> pd.Series:
        fast, slow, fast_n, slow_n = self._averages(bars, params)
        if fast_n >= slow_n:
            return pd.Series(0, index=bars.index[start:], dtype=int)
        entry = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        exit_ = (fast < slow) & (fast.shift(1) >= slow.shift(1))
        return positions_from_flags(entry, exit_, start, side="long_only")

    def indicator_frame(self, bars: pd.DataFrame, params: dict):
        fast, slow, _, _ = self._averages(bars, params)
        frame = pd.DataFrame({"fast": fast, "slow": slow}, index=bars.index)
        return frame, {"fast": "overlay", "slow": "overlay"}
