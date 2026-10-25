"""Three rising green months, a 5% weekly drop, a one-week hold, and a 5% stop."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from engine.book import check_book_strategy
from engine.indicators import atr
from engine.portfolio import simulate
from engine.strategy import resolve_params
from strategies.weekly_drop import WeeklyDrop, prepare_signals, prior_months_pass


def params(**overrides) -> dict:
    resolved = resolve_params(WeeklyDrop().param_space(), {})
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


def paint_months(specs: list[tuple[str, float, float]]) -> pd.DataFrame:
    """One rising or falling path per calendar month, with a wide daily range."""

    chunks = []
    for month, month_open, month_close in specs:
        start = pd.Timestamp(month + "-01")
        dates = pd.bdate_range(start, start + pd.offsets.MonthEnd(0))
        close = np.linspace(month_open, month_close, len(dates))
        open_ = np.empty(len(dates))
        open_[0] = month_open
        open_[1:] = close[:-1]
        high = np.maximum(open_, close) + 15
        low = np.minimum(open_, close) - 15
        chunks.append(frame(dates, open_, high, low, close))
    return pd.concat(chunks)


def put(bars: pd.DataFrame, day: str, open_: float, high: float, low: float, close: float) -> None:
    stamp = pd.Timestamp(day)
    bars.loc[stamp, "open"] = open_
    bars.loc[stamp, "high"] = high
    bars.loc[stamp, "low"] = low
    bars.loc[stamp, "close"] = close


def rising_book() -> pd.DataFrame:
    """October through March rise. April holds 200, then one Friday closes at 190."""

    bars = paint_months(
        [
            ("2023-10", 100, 110),
            ("2023-11", 112, 124),
            ("2023-12", 126, 138),
            ("2024-01", 140, 152),
            ("2024-02", 154, 166),
            ("2024-03", 168, 180),
            ("2024-04", 200, 200),
        ]
    )
    put(bars, "2024-03-29", 198, 214, 184, 200)
    put(bars, "2024-04-05", 200, 214, 186, 200)
    put(bars, "2024-04-12", 198, 210, 186, 190)
    for day in pd.bdate_range("2024-04-15", "2024-04-30"):
        put(bars, str(day.date()), 190, 194, 186, 190)
    return bars


def test_book_accepts_the_strategy():
    strategy = WeeklyDrop()
    assert check_book_strategy(strategy) == []
    assert [stage.key for stage in strategy.stages] == ["filter", "buy", "sell", "stop", "size"]
    named = [name for stage in strategy.stages for name in stage.params]
    assert len(named) == len(set(named))
    assert set(named).isdisjoint(strategy.cost_params)
    assert set(named) | set(strategy.cost_params) | {"stop_atr"} == set(strategy.param_space())
    space = strategy.param_space()
    assert space["months"].default == 3
    assert space["drop_pct"].default == 5
    assert space["atr_length"].default == 15
    assert space["stop_pct"].default == 5
    assert space["partial_pct"].default == 100
    assert space["capital"].default == 2_000_000
    assert space["max_position_inr"].default == 2_000_000
    assert space["risk_per_atr"].default == 1
    resolved = resolve_params(space, {})
    text = " ".join(strategy.stage_line(stage.key, resolved) for stage in strategy.stages)
    assert "3 completed months" in text
    assert "5% below" in text
    assert "5% under the fill" in text
    assert "portfolio value / 20" in text
    assert "2,000,000" in text
    with pytest.raises(RuntimeError, match="bank"):
        strategy.target_positions(pd.DataFrame(), {})


def test_three_rising_green_months_pass_and_the_current_month_is_excluded():
    bars = paint_months(
        [
            ("2023-09", 100, 110),
            ("2023-10", 112, 122),
            ("2023-11", 124, 134),
            ("2023-12", 150, 140),
            ("2024-01", 142, 155),
        ]
    )
    allowed = prior_months_pass(bars, 3)
    # December is red, and it is the current month on its own sessions, so Sep-Nov still pass.
    assert allowed.loc["2023-12-01":"2023-12-29"].all()
    assert float(bars.loc[pd.Timestamp("2023-12-29"), "close"]) < float(bars.loc[pd.Timestamp("2023-12-01"), "open"])
    # January includes that red December.
    assert not allowed.loc["2024-01-02":].any()
    # September does not yet have three earlier months.
    assert not allowed.loc["2023-09-01":"2023-09-29"].any()


def test_a_green_month_that_closes_below_the_prior_month_fails():
    bars = paint_months(
        [
            ("2023-09", 100, 110),
            ("2023-10", 112, 130),
            ("2023-11", 120, 128),
            ("2023-12", 130, 142),
        ]
    )
    allowed = prior_months_pass(bars, 3)
    assert not allowed.loc["2023-12-01":].any()


def test_a_five_percent_week_buys_the_next_open_and_sells_after_the_following_week():
    bars = rising_book()
    signals = prepare_signals(bars, params())
    assert signals.loc[pd.Timestamp("2024-04-12"), "entry"]
    assert not signals.loc[pd.Timestamp("2024-04-05"), "entry"]
    assert not signals.loc[pd.Timestamp("2024-04-15"), "entry"]
    assert signals.loc[pd.Timestamp("2024-04-19"), "exit"]
    assert not signals.loc[pd.Timestamp("2024-04-12"), "exit"]
    result = simulate({"AAA": bars}, {"AAA": signals}, params())
    trade = result.trades.iloc[0]
    assert len(result.trades) == 1
    assert trade["entry_time"] == pd.Timestamp("2024-04-15")
    assert trade["exit_time"] == pd.Timestamp("2024-04-22")
    assert trade["entry_price"] == pytest.approx(190)
    assert trade["exit_price"] == pytest.approx(190)
    assert trade["reason"] == "crossover"
    atr_value = float(signals.loc[pd.Timestamp("2024-04-12"), "atr"])
    budget = 2_000_000 / 20
    expected = min(math.floor(budget / atr_value), math.floor(2_000_000 / 190))
    assert trade["shares"] == expected
    assert trade["shares"] == math.floor(budget / atr_value)
    pd.testing.assert_series_equal(signals["atr"], atr(bars, 15), check_names=False)


def test_a_smaller_drop_is_not_a_buy():
    bars = rising_book()
    put(bars, "2024-04-12", 198, 210, 186, 190.2)
    signals = prepare_signals(bars, params())
    assert not signals["entry"].any()
    result = simulate({"AAA": bars}, {"AAA": signals}, params())
    assert result.trades.empty


def test_a_red_prior_month_blocks_the_drop():
    bars = rising_book()
    january = bars.loc["2024-01-01":"2024-01-31"]
    put(bars, str(january.index[0].date()), 160, 170, 140, 158)
    put(bars, str(january.index[-1].date()), 150, 160, 130, 140)
    signals = prepare_signals(bars, params())
    assert not signals.loc[pd.Timestamp("2024-04-12"), "entry"]


def test_the_stop_is_five_percent_under_the_fill_and_sells_every_share():
    bars = rising_book()
    put(bars, "2024-04-16", 190, 192, 170, 188)
    signals = prepare_signals(bars, params())
    result = simulate({"AAA": bars}, {"AAA": signals}, params())
    trade = result.trades.iloc[0]
    assert len(result.trades) == 1
    assert trade["reason"] == "stop"
    assert trade["entry_time"] == pd.Timestamp("2024-04-15")
    assert trade["exit_time"] == pd.Timestamp("2024-04-16")
    assert trade["exit_price"] == pytest.approx(190 * 0.95)
    assert trade["shares"] == math.floor((2_000_000 / 20) / float(signals.loc[pd.Timestamp("2024-04-12"), "atr"]))
    assert result.open_side is None


def test_a_second_down_week_sells_the_hold_and_buys_again_the_same_morning():
    bars = rising_book()
    for day in pd.bdate_range("2024-04-15", "2024-04-18"):
        put(bars, str(day.date()), 170, 174, 166, 170)
    put(bars, "2024-04-19", 170, 174, 166, 162)
    for day in pd.bdate_range("2024-04-22", "2024-04-30"):
        put(bars, str(day.date()), 162, 166, 158, 162)
    signals = prepare_signals(bars, params())
    assert bool(signals.loc[pd.Timestamp("2024-04-19"), "exit"])
    assert bool(signals.loc[pd.Timestamp("2024-04-19"), "entry"])
    result = simulate({"AAA": bars}, {"AAA": signals}, params())
    assert list(result.trades["entry_time"]) == [pd.Timestamp("2024-04-15"), pd.Timestamp("2024-04-22")]
    assert list(result.trades["exit_time"]) == [pd.Timestamp("2024-04-22"), pd.Timestamp("2024-04-29")]
    assert list(result.trades["reason"]) == ["crossover", "crossover"]
    assert result.trades["entry_price"].tolist() == pytest.approx([170, 162])


def test_a_later_close_does_not_change_an_earlier_signal():
    bars = rising_book()
    original = prepare_signals(bars, params())
    altered = bars.copy()
    altered.iloc[-1, altered.columns.get_loc("close")] = 40
    altered.iloc[-1, altered.columns.get_loc("low")] = 30
    changed = prepare_signals(altered, params())
    cutoff = pd.Timestamp("2024-04-20")
    mask = original.index < cutoff
    pd.testing.assert_series_equal(original.loc[mask, "entry"], changed.loc[mask, "entry"])
    pd.testing.assert_series_equal(original.loc[mask, "exit"], changed.loc[mask, "exit"])


def test_indicator_frame_shows_the_atr_the_weekly_return_and_the_month_filter():
    bars = rising_book()
    plotted, kinds = WeeklyDrop().indicator_frame(bars, {})
    assert list(plotted.columns) == ["ATR", "Weekly return", "Weekly return.threshold", "Month filter"]
    assert kinds["ATR"] == "oscillator"
    assert kinds["Weekly return"] == "oscillator"
    assert kinds["Weekly return.threshold"] == "oscillator"
    assert kinds["Month filter"] == "oscillator"
    pd.testing.assert_series_equal(plotted["ATR"], atr(bars, 15), check_names=False)
    assert plotted.loc[pd.Timestamp("2024-04-11"), "Weekly return"] == pytest.approx(0)
    assert plotted.loc[pd.Timestamp("2024-04-12"), "Weekly return"] == pytest.approx(-5)
    assert plotted["Weekly return.threshold"].iloc[-1] == -5
    assert plotted.loc[pd.Timestamp("2023-10-02"), "Month filter"] == 0
    assert plotted.loc[pd.Timestamp("2024-04-12"), "Month filter"] == 1
