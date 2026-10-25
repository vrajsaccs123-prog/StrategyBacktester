"""Daily EMA crossover entries, the 90-day high filter, and ATR sizing."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from engine.book import check_book_strategy
from engine.indicators import atr, ema, rsi
from engine.portfolio import simulate
from engine.strategy import resolve_params
from strategies.ema_crossover import EmaCrossover, prepare_signals, within_recent_high


def params(**overrides) -> dict:
    resolved = resolve_params(EmaCrossover().param_space(), {})
    resolved.update(
        {
            "slippage_bps": 0,
            "brokerage_bps": 0,
            "brokerage_cap_inr": 0,
            "stt_buy_bps": 0,
            "stt_sell_bps": 0,
            "exchange_bps": 0,
            "sebi_bps": 0,
            "stamp_buy_bps": 0,
            "gst_pct": 0,
        }
    )
    resolved.update(overrides)
    return resolved


def frame(index, open_, high, low, close) -> pd.DataFrame:
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


def week_bars(weeks: list[tuple[float, float]], start: str = "2024-01-01") -> pd.DataFrame:
    """One Monday-Friday bar set per (week open, week close).

    The Monday open is the week open. The Friday close is the week close.
    In-between closes step from the open to the close. Each day's range is 10,
    so the ATR stays positive.
    """

    dates = pd.bdate_range(start, periods=5 * len(weeks))
    open_ = np.empty(len(dates))
    close = np.empty(len(dates))
    for index, (week_open, week_close) in enumerate(weeks):
        slot = slice(index * 5, index * 5 + 5)
        close[slot] = np.linspace(week_open, week_close, 5)
        open_[slot] = close[slot]
        open_[index * 5] = week_open
    high = np.maximum(open_, close) + 5
    low = np.minimum(open_, close) - 5
    return frame(dates, open_, high, low, close)


def test_book_accepts_the_strategy():
    strategy = EmaCrossover()
    assert check_book_strategy(strategy) == []
    assert [stage.key for stage in strategy.stages] == ["filter", "buy", "sell", "stop", "size"]
    named = [name for stage in strategy.stages for name in stage.params]
    assert len(named) == len(set(named))
    assert set(named).isdisjoint(strategy.cost_params)
    assert set(named) | set(strategy.cost_params) == set(strategy.param_space())


def test_account_is_20_lakh_and_size_uses_portfolio_over_20():
    strategy = EmaCrossover()
    space = strategy.param_space()
    assert space["fast"].default == 10
    assert space["slow"].default == 30
    assert space["atr_length"].default == 10
    assert space["rsi_length"].default == 14
    assert space["rsi_buy"].default == 50
    assert "rsi_sell" not in space
    assert space["high_days"].default == 90
    assert space["max_drop_pct"].default == 20
    assert space["capital"].default == 2_000_000
    assert space["max_position_inr"].default == 2_000_000
    assert space["risk_per_atr"].default == 1
    assert space["partial_pct"].default == 100
    resolved = resolve_params(space, {})
    text = " ".join(strategy.stage_line(stage.key, resolved) for stage in strategy.stages)
    assert "10-day EMA crosses above the 30-day EMA" in text
    assert "RSI is above 50" in text
    assert "10-day EMA crosses below the 30-day EMA" in text
    assert "within 20% of the highest high" in text
    assert "last 90 trading days" in text
    assert "100% of the shares sell there" in text
    assert "weekly" not in text.lower()
    assert "portfolio value / 20" in text
    assert "2,000,000" in text
    with pytest.raises(RuntimeError, match="bank"):
        strategy.target_positions(pd.DataFrame(), {})


def test_the_close_must_stay_within_20_percent_of_the_90_day_high():
    dates = pd.bdate_range("2024-01-01", periods=100)
    close = np.full(100, 100.0)
    high = np.full(100, 110.0)
    close[90] = 88  # exactly 20% under the 110 high
    high[90] = 88
    close[91] = 87.9
    high[91] = 87.9
    low = np.minimum(close, high) - 1
    bars = frame(dates, close, high, low, close)
    allowed = within_recent_high(bars, 90, 20)
    assert not allowed.iloc[:89].any()
    assert allowed.iloc[89]
    assert allowed.iloc[90]
    assert not allowed.iloc[91]


def test_a_cross_below_the_high_is_not_a_buy_and_the_sell_is_the_reverse_cross():
    bars = _cross_then_reverse()
    wide = dict(fast=3, slow=5, rsi_buy=-1, high_days=20)
    blocked = prepare_signals(bars, params(max_drop_pct=20, **wide))
    opened = prepare_signals(bars, params(max_drop_pct=100, **wide))
    assert opened["entry"].any()
    assert not blocked["entry"].any()
    pd.testing.assert_series_equal(opened["exit"], blocked["exit"], check_names=False)
    assert opened["exit"].any()


def test_signals_buy_the_cross_near_the_high_and_sell_when_the_ema_crosses_back():
    bars = _cross_then_reverse()
    settings = params(fast=3, slow=5, rsi_buy=-1, high_days=20, max_drop_pct=100)
    signals = prepare_signals(bars, settings)
    fast = ema(bars["close"], 3)
    slow = ema(bars["close"], 5)
    allowed = within_recent_high(bars, 20, 100)
    strength = rsi(bars["close"], int(settings["rsi_length"]))
    crossed = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    crossed_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    pd.testing.assert_series_equal(
        signals["entry"],
        (crossed & allowed & (strength > settings["rsi_buy"])).fillna(False),
        check_names=False,
    )
    pd.testing.assert_series_equal(signals["exit"], crossed_down.fillna(False), check_names=False)
    assert signals["entry"].any()
    assert signals["exit"].any()
    pd.testing.assert_series_equal(signals["atr"], atr(bars, 10), check_names=False)


def test_share_count_is_portfolio_over_20_times_price_over_the_previous_atr():
    bars = _cross_then_reverse()
    settings = params(fast=3, slow=5, rsi_buy=-1, high_days=20, max_drop_pct=100, partial_pct=0)
    signals = prepare_signals(bars, settings)
    entry_at = int(np.flatnonzero(signals["entry"].to_numpy())[0])
    result = simulate({"AAA": bars}, {"AAA": signals}, settings)
    trade = result.trades.iloc[0]
    atr_value = float(signals["atr"].iloc[entry_at])
    fill = float(bars["open"].iloc[entry_at + 1])
    budget = 2_000_000 / 20
    expected = min(math.floor(budget / atr_value), math.floor(2_000_000 / fill))
    assert trade["shares"] == expected
    assert trade["entry_time"] == bars.index[entry_at + 1]
    assert trade["reason"] == "crossover"
    # Starting portfolio / 20 × fill / ATR, in whole shares, inside the 20 lakh account.
    assert trade["shares"] == math.floor(budget / atr_value)
    assert trade["shares"] * fill < 2_000_000


def test_indicator_frame_names_the_emas_rsi_and_the_atr():
    bars = week_bars([(100, 110), (108, 120), (118, 130)])
    frame_out, kinds = EmaCrossover().indicator_frame(bars, {})
    assert list(frame_out.columns) == ["EMA 10", "EMA 30", "90-day high", "RSI", "ATR"]
    assert kinds["EMA 10"] == "overlay"
    assert kinds["EMA 30"] == "overlay"
    assert kinds["90-day high"] == "overlay"
    assert kinds["RSI"] == "oscillator"
    assert kinds["ATR"] == "oscillator"
    pd.testing.assert_series_equal(frame_out["EMA 10"], ema(bars["close"], 10), check_names=False)
    pd.testing.assert_series_equal(frame_out["EMA 30"], ema(bars["close"], 30), check_names=False)
    pd.testing.assert_series_equal(frame_out["RSI"], rsi(bars["close"], 14), check_names=False)
    pd.testing.assert_series_equal(frame_out["ATR"], atr(bars, 10), check_names=False)


def test_rsi_must_be_above_50_to_buy_and_the_sell_is_the_reverse_cross():
    bars = _cross_then_reverse()
    near = dict(fast=3, slow=5, high_days=20, max_drop_pct=100)
    daily = rsi(bars["close"], 14)
    fast = ema(bars["close"], 3)
    slow = ema(bars["close"], 5)
    crossed = ((fast > slow) & (fast.shift(1) <= slow.shift(1))).fillna(False)
    crossed_down = ((fast < slow) & (fast.shift(1) >= slow.shift(1))).fillna(False)
    # This path's cross is oversold, so the default level of 50 does not buy it.
    default = prepare_signals(bars, params(**near))
    assert crossed.any()
    assert (daily.loc[crossed] < 50).all()
    assert not default["entry"].any()
    pd.testing.assert_series_equal(default["exit"], crossed_down, check_names=False)
    opened = prepare_signals(bars, params(rsi_buy=0, **near))
    assert opened["entry"].any()
    assert (daily.loc[opened["entry"]] > 0).all()
    assert (daily.loc[opened["exit"]] <= 70).all()


def test_the_stop_sells_every_share():
    bars = _cross_then_reverse()
    settings = params(fast=3, slow=5, rsi_buy=-1, high_days=20, max_drop_pct=100, stop_atr=0.5)
    signals = prepare_signals(bars, settings)
    result = simulate({"AAA": bars}, {"AAA": signals}, settings)
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["reason"] == "stop"
    assert result.open_side is None


def _cross_then_reverse() -> pd.DataFrame:
    """Decline, two green weeks, a rally that crosses up, then a red week that crosses down.

    The path is fixed so the tests assert behaviour rather than a searched date.
    Weeks start Monday 1 Jan 2024.
    """

    dates = pd.bdate_range("2024-01-01", periods=31)
    close = np.array(
        [
            140, 132, 124, 116, 108,  # red week
            104, 100, 96, 92, 88,  # red week
            87, 87.2, 87.4, 87.6, 88.2,  # green, close 88.2
            88.05, 88.1, 88.15, 88.2, 88.4,  # green, close 88.4
            100, 115, 130, 145, 160,  # rally. The cross is Monday 29 Jan.
            150, 120, 90, 70, 55,  # red week, fast EMA back below the slow one
            50,  # next open, so the Friday exit can fill
        ],
        dtype=float,
    )
    open_ = close.copy()
    open_[0] = 148
    open_[5] = 110
    open_[10] = 86
    open_[15] = 87.6
    open_[20] = 91
    open_[25] = 170
    open_[30] = 52
    high = np.maximum(open_, close) + 5
    low = np.minimum(open_, close) - 5
    return frame(dates, open_, high, low, close)
