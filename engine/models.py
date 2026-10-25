"""Shared data types for backtests, parameters, and stored runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd


@dataclass(frozen=True)
class ParamSpec:
    """A strategy parameter. Searchable when both bounds are set and min < max."""

    default: float
    min: float | None = None
    max: float | None = None
    step: float | None = None
    kind: str = "float"

    @property
    def searchable(self) -> bool:
        return self.min is not None and self.max is not None and float(self.min) < float(self.max)

    def coerce(self, value: float) -> float | int:
        if self.kind == "int":
            return int(round(float(value)))
        return float(value)


@dataclass
class BacktestSettings:
    initial_capital: float = 100_000.0
    position_pct: float = 1.0
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    allow_short: bool = False

    def as_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if not 0 < self.position_pct <= 1:
            raise ValueError("Position size must be between 0 and 100 percent of equity")
        if self.commission_bps < 0 or self.slippage_bps < 0:
            raise ValueError("Commission and slippage cannot be negative")


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    settings: BacktestSettings
    open_side: str | None = None


@dataclass
class CompletedRun:
    result: BacktestResult
    metrics: dict
    indicators: pd.DataFrame
    indicator_kinds: dict[str, str]
    params: dict


@dataclass
class FoldScore:
    fold: int
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    score: float


@dataclass
class OptimizationResult:
    best_params: dict
    objective: str
    validation_median: float
    fold_scores: list[FoldScore]
    trials: pd.DataFrame
    pretest: CompletedRun
    test: CompletedRun
    gap: float
    test_start: pd.Timestamp
    scored_windows: list[tuple[pd.Timestamp, pd.Timestamp]] = field(default_factory=list)
    max_dd_cap: float | None = None
