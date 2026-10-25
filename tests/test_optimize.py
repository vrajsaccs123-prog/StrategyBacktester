"""Walk-forward search must not score the locked holdout."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.models import BacktestSettings, ParamSpec
from engine.optimize import optimize
from engine.rules import positions_from_flags
from engine.strategy import Strategy


def _bars(n=120, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n)))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    index = pd.bdate_range("2018-01-02", periods=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1_000.0},
        index=index,
    )


class HoldStrategy(Strategy):
    key = "hold"
    name = "Hold"

    def param_space(self):
        return {"hold": ParamSpec(4, 2, 8, 1, "int")}

    def target_positions(self, bars, params, start=0):
        hold = max(int(params["hold"]), 1)
        position = 0
        age = 0
        values = []
        for _ in range(start, len(bars)):
            if position == 0:
                position = 1
                age = 0
            else:
                age += 1
                if age >= hold:
                    position = 0
            values.append(position)
        return pd.Series(values, index=bars.index[start:], dtype=int)


def test_holdout_is_excluded_from_scoring_and_does_not_change_the_choice():
    bars = _bars()
    settings = BacktestSettings(initial_capital=50_000, commission_bps=1, slippage_bps=1)
    kwargs = dict(objective="sharpe", holdout_frac=0.25, n_folds=2, n_trials=5, seed=3)
    first = optimize(bars, HoldStrategy(), settings, **kwargs)

    altered = bars.copy()
    test_rows = altered.index >= first.test_start
    altered.loc[test_rows, ["open", "high", "low", "close"]] = (
        altered.loc[test_rows, ["open", "high", "low", "close"]] * 3 + 40
    )
    second = optimize(altered, HoldStrategy(), settings, **kwargs)

    assert first.scored_windows
    for _, end in first.scored_windows:
        assert end < first.test_start
    assert first.best_params == second.best_params
    assert first.pretest.result.equity.index[-1] < first.test_start
    assert first.test.result.equity.index[0] == first.test_start
    assert len(first.fold_scores) == 2


def test_rule_strategy_search_runs():
    n = 160
    wave = np.linspace(0, 8 * np.pi, n)
    close = 100 + 8 * np.sin(wave)
    open_ = np.r_[close[0], close[:-1]]
    bars = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 0.2,
            "low": np.minimum(open_, close) - 0.2,
            "close": close,
            "volume": 1_000.0,
        },
        index=pd.bdate_range("2018-01-02", periods=n),
    )
    spec = {
        "name": "Fast cross",
        "side": "long_only",
        "indicators": [
            {"id": "fast", "type": "sma", "params": {"length": {"value": 5, "min": 3, "max": 8, "step": 1, "kind": "int"}}},
            {"id": "slow", "type": "sma", "params": {"length": {"value": 15, "min": 10, "max": 20, "step": 5, "kind": "int"}}},
        ],
        "entry": {"all": [{"left": "fast", "op": "crosses_above", "right": "slow"}]},
        "exit": {"all": [{"left": "fast", "op": "crosses_below", "right": "slow"}]},
        "entry_short": {"all": []},
        "exit_short": {"all": []},
    }
    from engine.strategy import RuleStrategy

    result = optimize(
        bars,
        RuleStrategy(spec),
        BacktestSettings(),
        objective="sharpe",
        holdout_frac=0.2,
        n_folds=2,
        n_trials=3,
        seed=1,
    )
    assert set(result.best_params) >= {"ind.fast.length", "ind.slow.length"}
    assert result.test_start == bars.index[int(round(len(bars) * 0.8))]
    assert all(end < result.test_start for _, end in result.scored_windows)


def test_positions_helper_stays_long_until_exit():
    index = pd.bdate_range("2024-01-01", periods=4)
    entry = pd.Series([True, False, False, False], index=index)
    exit_ = pd.Series([False, False, True, False], index=index)
    assert list(positions_from_flags(entry, exit_)) == [1, 1, 0, 0]
