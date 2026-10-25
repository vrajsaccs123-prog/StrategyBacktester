"""Medium frequency momentum with no stop sale.

Filter, buy, sell, and size match medium frequency momentum. The book still
records a stop under the entry fill. This strategy sells none of the shares
there, so the weekly EMA sell closes the whole position.
"""

from __future__ import annotations

import pandas as pd

from engine.book import COST_PARAMS
from engine.models import ParamSpec
from engine.strategy import Strategy, resolve_params
from strategies import medium_frequency_momentum as momentum

STAGES = tuple(stage for stage in momentum.STAGES if stage.key != "stop")


def stage_line(key: str, params: dict) -> str:
    """One sentence that states the current settings for a stage."""

    if key == "sell":
        return "Fast weekly EMA below the slow one. The next open sells the whole position."
    if key not in {"filter", "buy", "size"}:
        raise KeyError(key)
    return momentum.stage_line(key, params)


class MediumFrequencyMomentumNoStop(Strategy):
    key = "medium_frequency_momentum_no_stop"
    name = "Medium frequency momentum, no stop"
    portfolio = True
    stages = STAGES
    cost_params = COST_PARAMS
    param_labels = momentum.PARAM_LABELS
    param_help = momentum.PARAM_HELP
    param_choices = momentum.CHOICES

    def param_space(self) -> dict[str, ParamSpec]:
        space = momentum.MediumFrequencyMomentum().param_space()
        space["stop_atr"] = ParamSpec(2.0, None, None, None, "float")
        space["partial_pct"] = ParamSpec(0, None, None, None, "int")
        return space

    def target_positions(self, bars: pd.DataFrame, params: dict, start: int = 0) -> pd.Series:
        raise RuntimeError("Medium frequency momentum without a stop trades a portfolio. Run it from the bank.")

    def signals(self, bars: pd.DataFrame, params: dict) -> pd.DataFrame:
        return momentum.prepare_signals(bars, resolve_params(self.param_space(), params))

    def indicator_frame(self, bars: pd.DataFrame, params: dict):
        return momentum.chart_frame(bars, resolve_params(self.param_space(), params))

    def stage_line(self, key: str, params: dict) -> str:
        return stage_line(key, params)
