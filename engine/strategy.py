"""Strategy interface shared by rule specs and Python strategy files."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from engine.indicators import build_feature_frame, indicator_plot_frame
from engine.models import ParamSpec
from engine.rules import apply_params, eval_rule, extract_param_space, positions_from_flags


@dataclass(frozen=True)
class Stage:
    """One step of a book strategy. `groups` is (heading, parameter names)."""

    key: str
    title: str
    summary: str
    groups: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def params(self) -> tuple[str, ...]:
        return tuple(name for _heading, names in self.groups for name in names)


class Strategy:
    """A book strategy filters a universe, then buys, sells, stops, and sizes.

    A one-symbol strategy still emits a target position of 1, 0, or -1.
    A book strategy sets `stages` and is run across many names.
    """

    key: str = ""
    name: str = ""
    stages: tuple[Stage, ...] = ()
    cost_params: tuple[str, ...] = ()
    param_labels: dict[str, str] = {}
    param_help: dict[str, str] = {}
    param_choices: dict[str, tuple[tuple[str, int], ...]] = {}

    def param_space(self) -> dict[str, ParamSpec]:
        return {}

    def stage_line(self, key: str, params: dict) -> str:
        for stage in self.stages:
            if stage.key == key:
                return stage.summary
        raise KeyError(key)

    def target_positions(self, bars: pd.DataFrame, params: dict, start: int = 0) -> pd.Series:
        raise NotImplementedError

    def signals(self, bars: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Entry and exit flags for one name. Book strategies implement this."""

        raise NotImplementedError(f"{self.name or self.key} does not trade a book")

    def indicator_frame(self, bars: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, dict[str, str]]:
        return pd.DataFrame(index=bars.index), {}


class RuleStrategy(Strategy):
    key = "rule"

    def __init__(self, spec: dict):
        self.spec = spec
        self.name = str(spec.get("name") or "Rule strategy")
        self.key = f"rule:{self.name}"

    def param_space(self) -> dict[str, ParamSpec]:
        return extract_param_space(self.spec)

    def target_positions(self, bars: pd.DataFrame, params: dict, start: int = 0) -> pd.Series:
        spec = apply_params(self.spec, params)
        frame = build_feature_frame(bars, spec.get("indicators", []))
        entry = eval_rule(frame, spec.get("entry"))
        exit_ = eval_rule(frame, spec.get("exit"))
        side = spec.get("side", "long_only")
        entry_short = eval_rule(frame, spec.get("entry_short")) if side == "long_short" else None
        exit_short = eval_rule(frame, spec.get("exit_short")) if side == "long_short" else None
        return positions_from_flags(entry, exit_, start, entry_short, exit_short, side)

    def indicator_frame(self, bars: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, dict[str, str]]:
        spec = apply_params(self.spec, params)
        return indicator_plot_frame(bars, spec.get("indicators", []))


def resolve_params(space: dict[str, ParamSpec], overrides: dict | None = None) -> dict:
    overrides = overrides or {}
    resolved = {}
    for name, spec in space.items():
        resolved[name] = spec.coerce(overrides.get(name, spec.default))
    return resolved
