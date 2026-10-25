"""Medium frequency momentum with the stop sale removed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.book import COST_PARAMS, check_book_strategy
from engine.portfolio import simulate
from engine.strategy import resolve_params
from strategies.medium_frequency_momentum import MediumFrequencyMomentum
from strategies.medium_frequency_momentum_no_stop import MediumFrequencyMomentumNoStop


def _bars(periods: int = 400, seed: int = 3) -> pd.DataFrame:
    dates = pd.bdate_range("2018-01-01", periods=periods)
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, periods)))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.01,
            "low": np.minimum(open_, close) * 0.99,
            "close": close,
            "volume": 1_000.0,
        },
        index=dates,
    )


def _book(index, open_, high, low, close) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": np.asarray(open_, dtype=float),
            "high": np.asarray(high, dtype=float),
            "low": np.asarray(low, dtype=float),
            "close": np.asarray(close, dtype=float),
            "volume": 1.0,
        },
        index=pd.DatetimeIndex(index, name="date"),
    )


def _flags(bars: pd.DataFrame, entry: int, exit_: int | None = None, atr_value: float = 10) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "entry": False,
            "exit": False,
            "atr": atr_value,
            "udr": 1.0,
            "spread_atr": 0.1,
        },
        index=bars.index,
    )
    frame.iloc[entry, frame.columns.get_loc("entry")] = True
    if exit_ is not None:
        frame.iloc[exit_, frame.columns.get_loc("exit")] = True
    return frame


def _params(**overrides) -> dict:
    resolved = resolve_params(MediumFrequencyMomentumNoStop().param_space(), {})
    for name in COST_PARAMS:
        resolved[name] = 0
    resolved.update(overrides)
    return resolved


def test_book_accepts_the_strategy_without_a_stop_stage():
    strategy = MediumFrequencyMomentumNoStop()
    assert check_book_strategy(strategy) == []
    assert [stage.key for stage in strategy.stages] == ["filter", "buy", "sell", "size"]
    named = [name for stage in strategy.stages for name in stage.params]
    assert len(named) == len(set(named))
    assert set(named).isdisjoint(strategy.cost_params)
    assert set(named) | set(strategy.cost_params) | {"stop_atr", "partial_pct"} == set(strategy.param_space())
    space = strategy.param_space()
    assert space["partial_pct"].default == 0
    assert not space["partial_pct"].searchable
    assert space["stop_atr"].default == 2.0
    resolved = resolve_params(space, {})
    text = " ".join(strategy.stage_line(stage.key, resolved) for stage in strategy.stages)
    assert "upside/downside" in text
    assert "sells the whole position" in text
    assert "portfolio value / 20" in text
    with pytest.raises(KeyError):
        strategy.stage_line("stop", resolved)
    with pytest.raises(RuntimeError, match="bank"):
        strategy.target_positions(pd.DataFrame(), {})


def test_signals_and_indicators_match_medium_frequency_momentum():
    bars = _bars()
    params = resolve_params(MediumFrequencyMomentumNoStop().param_space(), {})
    replica = MediumFrequencyMomentumNoStop()
    original = MediumFrequencyMomentum()
    pd.testing.assert_frame_equal(replica.signals(bars, params), original.signals(bars, params))
    left, left_kinds = replica.indicator_frame(bars, params)
    right, right_kinds = original.indicator_frame(bars, params)
    pd.testing.assert_frame_equal(left, right)
    assert left_kinds == right_kinds


def test_a_gap_through_the_recorded_stop_sells_no_shares():
    dates = pd.bdate_range("2024-06-03", periods=6)
    bars = _book(
        dates,
        open_=[999, 500, 470, 470, 470, 450],
        high=[1000, 510, 475, 470, 470, 455],
        low=[990, 500, 460, 400, 400, 440],
        close=[999, 505, 472, 410, 410, 452],
    )
    held = simulate({"AAA": bars}, {"AAA": _flags(bars, entry=0)}, _params(max_position_inr=20_000_000))
    assert held.trades.empty
    assert held.open_side == "long"

    closed = simulate(
        {"AAA": bars},
        {"AAA": _flags(bars, entry=0, exit_=4)},
        _params(max_position_inr=20_000_000),
    )
    assert list(closed.trades["reason"]) == ["crossover"]
    assert closed.trades.iloc[0]["shares"] == 40_000
    assert closed.trades.iloc[0]["exit_time"] == dates[5]
