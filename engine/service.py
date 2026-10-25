"""Run a strategy on a bar series and package the report inputs."""

from __future__ import annotations

import pandas as pd

from engine.backtest import run_backtest
from engine.metrics import compute_metrics
from engine.models import BacktestSettings, CompletedRun
from engine.strategy import Strategy, resolve_params


def complete_run(
    bars: pd.DataFrame,
    strategy: Strategy,
    params: dict,
    settings: BacktestSettings,
    start: int = 0,
) -> CompletedRun:
    """Backtest `bars.iloc[start:]`. Earlier rows are indicator history only."""

    target = strategy.target_positions(bars, params, start=start)
    segment = bars.iloc[start:]
    result = run_backtest(segment, target, settings)
    frame, kinds = strategy.indicator_frame(bars, params)
    return CompletedRun(
        result=result,
        metrics=compute_metrics(result),
        indicators=frame.reindex(segment.index),
        indicator_kinds=kinds,
        params=params,
    )


def execute(
    bars: pd.DataFrame,
    strategy: Strategy,
    params: dict | None,
    settings: BacktestSettings,
) -> CompletedRun:
    resolved = resolve_params(strategy.param_space(), params or {})
    return complete_run(bars, strategy, resolved, settings, start=0)
