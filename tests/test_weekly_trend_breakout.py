"""Weekly EMA filter, Donchian breakout, loose ADX floor, and ATR sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.book import check_book_strategy, indicator_names
from engine.indicators import adx, donchian, ema
from engine.strategy import resolve_params
from strategies.medium_frequency_momentum import completed_ohlc
from strategies.weekly_trend_breakout import WeeklyTrendBreakout, prepare_signals


def params(**overrides) -> dict:
    resolved = resolve_params(WeeklyTrendBreakout().param_space(), {})
    resolved.update(overrides)
    return resolved


def frame(index, close) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000.0,
        },
        index=pd.DatetimeIndex(index, name="date"),
    )


def test_book_accepts_the_strategy():
    strategy = WeeklyTrendBreakout()
    assert check_book_strategy(strategy) == []
    assert [stage.key for stage in strategy.stages] == ["filter", "buy", "sell", "stop", "size"]
    named = [name for stage in strategy.stages for name in stage.params]
    assert len(named) == len(set(named))
    assert set(named).isdisjoint(strategy.cost_params)
    assert set(named) | set(strategy.cost_params) == set(strategy.param_space())
    searched = set(indicator_names(strategy))
    assert searched == {"ema_weeks", "adx_weeks", "adx_min", "channel_weeks", "atr_length"}
    assert searched.isdisjoint({"stop_atr", "partial_pct", "capital", "entry_rank", "max_position_inr"})


def test_account_is_20_lakh_and_a_name_is_capped_at_2_lakh():
    strategy = WeeklyTrendBreakout()
    space = strategy.param_space()
    assert space["ema_weeks"].default == 20
    assert space["channel_weeks"].default == 14
    assert space["adx_min"].default == 15
    assert space["atr_length"].default == 14
    assert space["stop_atr"].default == 4
    assert space["partial_pct"].default == 100
    assert space["capital"].default == 2_000_000
    assert space["max_position_inr"].default == 200_000
    assert space["entry_rank"].default == 0
    resolved = resolve_params(space, {})
    text = " ".join(strategy.stage_line(stage.key, resolved) for stage in strategy.stages)
    assert "20-week EMA" in text
    assert "ADX is at least 15" in text
    assert "prior 14 weeks" in text
    assert "4 entry ATRs" in text
    assert "200,000" in text
    assert "2,000,000" in text
    assert "strongest weekly rate of change" in text
    with pytest.raises(RuntimeError, match="bank"):
        strategy.target_positions(pd.DataFrame(), {})


def test_a_buy_is_a_new_high_above_the_average_and_a_sell_is_back_under_it():
    dates = pd.bdate_range("2018-01-01", periods=400)
    close = 100 + np.linspace(0, 80, len(dates))
    close[250:280] = np.linspace(close[249], close[249] - 40, 30)
    bars = frame(dates, close)
    tuned = params(ema_weeks=4, channel_weeks=4, adx_weeks=4, adx_min=10, atr_length=10)
    signals = prepare_signals(bars, tuned)
    weekly = completed_ohlc(bars, "W-FRI")
    average = ema(weekly["close"], 4)
    high = donchian(weekly, 4)["upper"]
    strength = adx(weekly, 4)["adx"]
    entries = signals.index[signals["entry"].to_numpy(dtype=bool)]
    exits = signals.index[signals["exit"].to_numpy(dtype=bool)]
    assert len(entries) > 0
    assert len(exits) > 0
    for stamp in entries:
        assert weekly.at[stamp, "close"] > average.at[stamp]
        assert weekly.at[stamp, "close"] > high.at[stamp]
        assert strength.at[stamp] >= 10
    for stamp in exits:
        assert weekly.at[stamp, "close"] < average.at[stamp]
    assert not (signals["entry"] & signals["exit"]).any()
    assert signals.index.equals(bars.index)


def test_a_falling_name_and_a_high_adx_floor_buy_nothing():
    dates = pd.bdate_range("2018-01-01", periods=300)
    falling = frame(dates, np.linspace(200, 50, len(dates)))
    assert not prepare_signals(falling, params(ema_weeks=4, channel_weeks=4, adx_weeks=4, adx_min=10))["entry"].any()
    rising = frame(dates, np.linspace(50, 200, len(dates)))
    blocked = prepare_signals(rising, params(ema_weeks=4, channel_weeks=4, adx_weeks=4, adx_min=1000))
    assert not blocked["entry"].any()


def test_indicator_frame_draws_the_five_indicators():
    dates = pd.bdate_range("2020-01-01", periods=120)
    bars = frame(dates, np.linspace(100, 130, len(dates)))
    plotted, kinds = WeeklyTrendBreakout().indicator_frame(bars, {})
    assert list(plotted.columns) == ["Weekly EMA", "Donchian high", "ADX", "ADX.minimum", "ROC", "ATR"]
    assert kinds == {
        "Weekly EMA": "overlay",
        "Donchian high": "overlay",
        "ADX": "oscillator",
        "ADX.minimum": "oscillator",
        "ROC": "oscillator",
        "ATR": "oscillator",
    }
    assert plotted.index.equals(bars.index)
    assert plotted["ADX.minimum"].dropna().eq(15).all()
