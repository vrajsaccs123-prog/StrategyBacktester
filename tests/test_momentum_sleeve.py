"""Leaders by rate of change, a market filter, a trailing-high sell, and an ATR stop."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.book import check_book_strategy, indicator_names
from engine.strategy import resolve_params
from strategies.momentum_sleeve import MomentumSleeve, book_signals, single_signals


def params(**overrides) -> dict:
    resolved = resolve_params(MomentumSleeve().param_space(), {})
    resolved.update(overrides)
    return resolved


def frame(index, close) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": 1_000.0,
        },
        index=pd.DatetimeIndex(index, name="date"),
    )


def test_book_accepts_the_strategy():
    strategy = MomentumSleeve()
    assert check_book_strategy(strategy) == []
    assert [stage.key for stage in strategy.stages] == ["filter", "buy", "sell", "stop", "size"]
    named = [name for stage in strategy.stages for name in stage.params]
    assert len(named) == len(set(named))
    assert set(named).isdisjoint(strategy.cost_params)
    assert set(named) | set(strategy.cost_params) == set(strategy.param_space())
    assert set(indicator_names(strategy)) == {"ema_weeks"}
    assert "trail_pct" not in indicator_names(strategy)
    assert "slots" not in indicator_names(strategy)


def test_the_file_holds_five_leaders_and_a_16_percent_trail():
    strategy = MomentumSleeve()
    space = strategy.param_space()
    assert space["ema_weeks"].default == 30
    assert space["roc_weeks"].default == 52
    assert space["slots"].default == 5
    assert space["hold_rank"].default == 10
    assert space["trail_pct"].default == 16
    assert space["index_exit_pct"].default == 8
    assert space["index_entry_pct"].default == 2
    assert space["stop_atr"].default == 4
    assert space["capital"].default == 2_000_000
    assert space["max_position_inr"].default == 200_000
    resolved = resolve_params(space, {})
    text = " ".join(strategy.stage_line(stage.key, resolved) for stage in strategy.stages)
    assert "30-week EMA" in text
    assert "within 2%" in text
    assert "5 strongest 52-week" in text
    assert "past 10" in text
    assert "16%" in text
    assert "8%" in text
    assert "4 entry ATRs" in text
    assert "200,000" in text
    assert "strongest rate of change" in text
    with pytest.raises(RuntimeError, match="bank"):
        strategy.target_positions(pd.DataFrame(), {})


def test_a_rise_is_bought_and_a_giveback_is_sold():
    dates = pd.bdate_range("2016-01-04", periods=420)
    close = np.linspace(50, 180, len(dates))
    close[-12:] = np.linspace(close[-13] * 0.97, close[-13] * 0.80, 12)
    bars = frame(dates, close)
    tuned = params(ema_weeks=8, roc_weeks=8, trail_days=15, trail_pct=8, atr_length=10)
    signals = single_signals(bars, tuned)
    assert signals["entry"].iloc[:400].any()
    assert signals["exit"].iloc[-12:].any()
    assert not (signals["entry"] & signals["exit"]).any()
    assert signals.index.equals(bars.index)
    assert (signals["atr"].iloc[30:] > 0).all()


def test_a_falling_name_is_not_bought():
    dates = pd.bdate_range("2016-01-04", periods=300)
    falling = frame(dates, np.linspace(200, 50, len(dates)))
    assert not single_signals(falling, params(ema_weeks=8, roc_weeks=8))["entry"].any()


def test_only_the_strongest_names_are_bought():
    dates = pd.bdate_range("2014-01-01", periods=500)
    tuned = params(
        ema_weeks=8,
        roc_weeks=12,
        slots=1,
        hold_rank=2,
        trail_pct=40,
        trail_days=20,
        index_days=40,
        index_exit_pct=40,
        index_entry_pct=40,
        atr_length=10,
    )
    signals = book_signals(
        {
            "FAST": frame(dates, np.linspace(80, 240, len(dates))),
            "MID": frame(dates, np.linspace(80, 160, len(dates))),
            "SLOW": frame(dates, np.linspace(80, 110, len(dates))),
        },
        tuned,
    )
    assert signals["FAST"]["entry"].any()
    assert not signals["SLOW"]["entry"].any()
    assert not (signals["FAST"]["entry"] & signals["FAST"]["exit"]).any()


def test_indicator_frame_draws_the_sleeve():
    dates = pd.bdate_range("2018-01-01", periods=160)
    bars = frame(dates, np.linspace(100, 140, len(dates)))
    plotted, kinds = MomentumSleeve().indicator_frame(bars, {})
    assert list(plotted.columns) == [
        "Weekly EMA",
        "Trail",
        "ROC",
        "ATR",
        "Market drawdown",
        "Market drawdown.limit",
    ]
    assert kinds["Weekly EMA"] == "overlay"
    assert kinds["Trail"] == "overlay"
    assert kinds["ROC"] == "oscillator"
    assert kinds["ATR"] == "oscillator"
    assert kinds["Market drawdown"] == "oscillator"
    assert plotted.index.equals(bars.index)
    assert plotted["Market drawdown.limit"].dropna().eq(-8).all()
