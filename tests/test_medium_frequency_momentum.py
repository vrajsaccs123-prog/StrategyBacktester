"""Medium frequency momentum signals, support zones, and portfolio fills."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.indicators import atr, ema
from engine.metrics import compute_metrics
from engine.portfolio import affordable_shares, share_charges, simulate
from engine.strategy import resolve_params
from strategies.medium_frequency_momentum import (
    BOOK_ATR_LENGTH,
    MediumFrequencyMomentum,
    completed_ohlc,
    grid_values,
    prepare_signals,
    select_support,
    udr_passes,
    upside_downside,
    zone_distance,
)


def flat_params(**overrides) -> dict:
    params = resolve_params(MediumFrequencyMomentum().param_space(), {})
    params.update(
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
    params.update(overrides)
    return params


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


def signal_frame(bars: pd.DataFrame, entry=None, exit_=None, atr_value=10.0, udr=2.0, spread=1.0) -> pd.DataFrame:
    blank = pd.DataFrame(
        {
            "entry": False,
            "exit": False,
            "atr": atr_value,
            "udr": np.nan,
            "spread_atr": np.nan,
        },
        index=bars.index,
    )
    if entry is not None:
        blank.iloc[entry, blank.columns.get_loc("entry")] = True
        blank.iloc[entry, blank.columns.get_loc("udr")] = udr
        blank.iloc[entry, blank.columns.get_loc("spread_atr")] = spread
    if exit_ is not None:
        blank.iloc[exit_, blank.columns.get_loc("exit")] = True
    return blank


def test_udr_averages_squared_up_and_down_weeks():
    close = pd.Series([100.0, 110.0, 99.0, 120.0, 108.0])
    parts = upside_downside(close, 2)
    # Returns at index 2 and 1: +10% then -10%. Mean of squares is 0.005 / 0.005.
    assert parts["udr"].iloc[2] == pytest.approx(1.0)
    assert parts["v_up"].iloc[2] == pytest.approx(0.005)
    assert parts["v_down"].iloc[2] == pytest.approx(0.005)
    third = (120.0 / 99.0 - 1) ** 2
    assert parts["v_up"].iloc[3] == pytest.approx(third / 2)
    assert parts["udr"].iloc[3] > 1


def test_udr_with_no_down_week_is_undefined_until_the_setting_accepts_it():
    close = pd.Series([100.0, 110.0, 121.0])
    parts = upside_downside(close, 2)
    assert parts["v_down"].iloc[2] == 0
    assert pd.isna(parts["udr"].iloc[2])
    assert bool(udr_passes(close, 2, 1.0, False).iloc[2]) is False
    assert bool(udr_passes(close, 2, 1.0, True).iloc[2]) is True


def test_completed_week_waits_for_the_next_session_or_a_finished_friday():
    dates = pd.bdate_range("2024-01-05", periods=8)  # Fri 5 Jan through the next Friday, then more
    # 5 Jan 2024 is a Friday. Eight business days end on the following Tuesday.
    bars = frame(dates, [10] * 8, [11] * 8, [9] * 8, list(range(8)))
    weekly = completed_ohlc(bars, "W-FRI")
    assert weekly.index[-1] == pd.Timestamp("2024-01-12")
    assert weekly["close"].iloc[0] == 0
    assert weekly["close"].iloc[1] == 5

    friday = dates[:6]  # ends on Friday 12 Jan
    finished = completed_ohlc(frame(friday, [10] * 6, [11] * 6, [9] * 6, list(range(6))), "W-FRI")
    assert finished.index[-1] == pd.Timestamp("2024-01-12")


def test_unfinished_month_is_left_out():
    dates = pd.bdate_range("2024-01-02", "2024-02-15")
    bars = frame(dates, [10] * len(dates), [11] * len(dates), [9] * len(dates), np.arange(len(dates)))
    monthly = completed_ohlc(bars, "M")
    assert list(monthly.index) == [pd.Timestamp("2024-01-31")]
    assert monthly["close"].iloc[0] == float(dates.get_loc(pd.Timestamp("2024-01-31")))


def test_monthly_ema_uses_the_library_and_only_completed_months():
    dates = pd.bdate_range("2022-01-03", "2024-04-10")
    close = pd.Series(np.linspace(100, 160, len(dates)), index=dates)
    bars = frame(dates, close, close + 1, close - 1, close)
    monthly = completed_ohlc(bars, "M")
    library = ema(monthly["close"], 10)
    assert library.notna().sum() > 3
    # April is unfinished, so a spike there cannot move the stored monthly EMA.
    spiked = bars.copy()
    spiked.loc[spiked.index >= "2024-04-01", "close"] = 500
    spiked.loc[spiked.index >= "2024-04-01", "high"] = 500
    again = ema(completed_ohlc(spiked, "M")["close"], 10)
    pd.testing.assert_series_equal(library, again)


def test_support_groups_nearby_lows_counts_one_visit_and_breaks_ties_upward():
    dates = pd.bdate_range("2024-01-01", periods=40)
    low = np.full(40, 100.0)
    high = np.full(40, 110.0)
    close = np.full(40, 105.0)
    # Two nearby swing lows, then a lower one. Neighbours stay above each pivot.
    low[5] = 90.0
    close[5] = 90.2
    low[8] = 90.4
    close[8] = 90.4
    low[10] = 90.2
    close[10] = 90.3
    low[22] = 80.0
    close[22] = 80.2
    bars = frame(dates, close, high, low, close)
    price = float(bars["close"].iloc[30])
    params = flat_params(
        zone_width_pct=0.5 / price * 100,
        touch_rebound_pct=1.0 / price * 100,
        zone_lookback_days=90,
        swing_left=2,
        swing_right=2,
        min_touches=1,
    )
    zone = select_support(bars, 30, params)
    assert zone is not None
    assert zone.level == pytest.approx(90.1)
    assert zone.touches >= 1
    assert zone.low < zone.level < zone.high

    wide = flat_params(
        zone_width_pct=0.05 / price * 100,
        touch_rebound_pct=0.2 / price * 100,
        zone_lookback_days=90,
        swing_left=2,
        swing_right=2,
        min_touches=1,
    )
    separate = select_support(bars, 30, wide)
    assert separate is not None
    assert separate.level == pytest.approx(max(90.0, 90.2))


def test_support_ignores_a_zone_above_the_close():
    dates = pd.bdate_range("2024-01-01", periods=30)
    low = np.full(30, 100.0)
    high = np.full(30, 130.0)
    close = np.full(30, 120.0)
    low[5] = 90
    low[16] = 80
    close[20:] = 85
    bars = frame(dates, close, high, low, close)
    price = float(bars["close"].iloc[24])
    params = flat_params(
        zone_width_pct=1.0 / price * 100,
        touch_rebound_pct=0.4 / price * 100,
        min_touches=1,
        swing_left=2,
        swing_right=2,
    )
    zone = select_support(bars, 24, params)
    assert zone is not None
    assert zone.level == pytest.approx(80)
    above = select_support(bars, 24, params)
    above.low, above.high = 84, 86
    assert zone_distance(85, above) == 0


def test_several_bars_at_one_level_are_one_touch():
    dates = pd.bdate_range("2024-01-01", periods=25)
    low = np.full(25, 100.0)
    high = np.full(25, 101.0)
    close = np.full(25, 100.5)
    low[6] = 90.0
    close[6] = 90.2
    high[6] = 91.0
    for index in (7, 8, 9):
        low[index] = 90.4
        high[index] = 91.0
        close[index] = 90.6
    close[10] = 100
    high[10] = 101
    low[10] = 94
    bars = frame(dates, close, np.maximum(high, close), np.minimum(low, close), close)
    price = float(bars["close"].iloc[15])
    params = flat_params(
        zone_width_pct=1.0 / price * 100,
        touch_rebound_pct=1.0 / price * 100,
        min_touches=1,
        swing_left=2,
        swing_right=2,
        zone_lookback_days=90,
    )
    zone = select_support(bars, 15, params)
    assert zone is not None
    assert zone.level == pytest.approx(90)
    assert zone.touches == 1


def test_prepare_refuses_a_fast_average_that_is_not_faster():
    dates = pd.bdate_range("2021-01-04", periods=400)
    close = 100 + np.linspace(0, 20, 400)
    bars = frame(dates, close, close + 1, close - 1, close)
    params = flat_params(weekly_fast=8, weekly_slow=4)
    signals = prepare_signals(bars, params)
    assert not signals["entry"].any()
    pd.testing.assert_series_equal(signals["atr"], atr(bars, BOOK_ATR_LENGTH), check_names=False)


def test_a_later_close_does_not_change_an_earlier_signal():
    dates = pd.bdate_range("2020-01-02", periods=500)
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, 500)))
    bars = frame(dates, close, close * 1.01, close * 0.99, close)
    params = flat_params()
    original = prepare_signals(bars, params)
    altered = bars.copy()
    altered.iloc[-1, altered.columns.get_loc("close")] = close[-1] * 3
    altered.iloc[-1, altered.columns.get_loc("high")] = close[-1] * 3
    changed = prepare_signals(altered, params)
    cutoff = bars.index[-1] - pd.Timedelta(days=10)
    mask = original.index < cutoff
    pd.testing.assert_series_equal(original.loc[mask, "entry"], changed.loc[mask, "entry"])
    pd.testing.assert_series_equal(original.loc[mask, "exit"], changed.loc[mask, "exit"])


def test_entry_fills_the_next_open_and_a_gap_stop_uses_that_open():
    dates = pd.bdate_range("2024-06-03", periods=6)
    bars = frame(
        dates,
        open_=[999, 500, 470, 470, 470, 450],
        high=[1000, 510, 475, 470, 470, 455],
        low=[990, 500, 460, 400, 400, 440],
        close=[999, 505, 472, 410, 410, 452],
    )
    signals = signal_frame(bars, entry=0, exit_=4, atr_value=10)
    result = simulate({"AAA": bars}, {"AAA": signals}, flat_params(max_position_inr=20_000_000))
    assert result.trades["entry_time"].iloc[0] == dates[1]
    assert result.trades["entry_price"].iloc[0] == pytest.approx(500)
    assert len(result.trades) == 2
    stopped = result.trades.iloc[0]
    assert stopped["reason"] == "stop"
    assert stopped["exit_time"] == dates[2]
    assert stopped["exit_price"] == pytest.approx(470)
    assert stopped["shares"] == 20_000
    # The next day trades through the stop again and must not sell the remainder.
    assert result.trades.iloc[1]["reason"] == "crossover"
    assert result.trades.iloc[1]["exit_time"] == dates[5]
    assert result.trades.iloc[1]["exit_price"] == pytest.approx(450)
    assert result.trades.iloc[1]["shares"] == 20_000


def test_a_stop_inside_the_day_fills_at_the_stop():
    dates = pd.bdate_range("2024-06-03", periods=3)
    bars = frame(dates, [500, 500, 490], [510, 510, 495], [490, 500, 470], [500, 505, 490])
    signals = signal_frame(bars, entry=0, atr_value=10)
    result = simulate({"AAA": bars}, {"AAA": signals}, flat_params(max_position_inr=20_000_000))
    assert result.trades["exit_price"].iloc[0] == pytest.approx(480)
    assert result.trades["shares"].iloc[0] == 20_000
    assert result.open_side == "long"
    # Cash caps the buy at 40,000 shares. Half remain, marked at the day-3 close of 490.
    assert result.equity["equity"].iloc[-1] == pytest.approx(20_000_000 - 500 * 40_000 + 480 * 20_000 + 490 * 20_000)


def test_stop_is_processed_before_the_weekly_exit_on_the_same_open():
    dates = pd.bdate_range("2024-06-03", periods=4)
    bars = frame(dates, [500, 500, 500, 470], [510, 510, 510, 475], [490, 500, 500, 460], [500, 505, 505, 472])
    signals = signal_frame(bars, entry=0, exit_=2, atr_value=10)
    result = simulate({"AAA": bars}, {"AAA": signals}, flat_params(max_position_inr=20_000_000))
    assert list(result.trades["reason"]) == ["stop", "crossover"]
    assert list(result.trades["exit_price"]) == pytest.approx([470, 470])
    assert list(result.trades["shares"]) == [20_000, 20_000]
    assert result.trades["exit_time"].nunique() == 1


def test_cash_from_the_morning_sale_can_fund_the_morning_buy():
    dates = pd.bdate_range("2024-06-03", periods=4)
    held = frame(dates, [100, 100, 100, 100], [101, 101, 101, 101], [99, 99, 99, 99], [100, 100, 100, 100])
    fresh = frame(dates, [100, 100, 100, 100], [101, 101, 101, 101], [99, 99, 99, 99], [100, 100, 100, 130])
    held_sig = signal_frame(held, entry=0, exit_=2, atr_value=5)
    fresh_sig = signal_frame(fresh, entry=2, atr_value=5, udr=3)
    params = flat_params(capital=1_000_000, max_position_inr=5_000_000)
    result = simulate({"AAA": held, "BBB": fresh}, {"AAA": held_sig, "BBB": fresh_sig}, params)
    assert result.trades["symbol"].tolist() == ["AAA"]
    assert result.equity["equity"].iloc[-1] == pytest.approx(1_300_000)
    assert result.equity["position"].iloc[1] == 1
    assert result.equity["position"].iloc[-1] == 1


def test_an_unfunded_signal_is_dropped_rather_than_queued():
    dates = pd.bdate_range("2024-06-03", periods=6)
    held = frame(dates, [100] * 6, [101] * 6, [99] * 6, [100] * 6)
    fresh = frame(dates, [100] * 6, [101] * 6, [99] * 6, [150] * 6)
    held_sig = signal_frame(held, entry=0, exit_=4, atr_value=5)
    fresh_sig = signal_frame(fresh, entry=2, atr_value=5, udr=4)
    params = flat_params(capital=1_000_000, max_position_inr=5_000_000)
    result = simulate({"AAA": held, "BBB": fresh}, {"AAA": held_sig, "BBB": fresh_sig}, params)
    assert result.equity["equity"].iloc[-1] == pytest.approx(1_000_000)
    assert result.equity["position"].iloc[-1] == 0


def test_higher_udr_is_filled_when_cash_covers_only_one_name():
    dates = pd.bdate_range("2024-06-03", periods=3)
    quiet = frame(dates, [500, 500, 500], [510, 510, 510], [490, 490, 490], [500, 500, 500])
    strong = frame(dates, [500, 500, 500], [530, 530, 530], [490, 490, 490], [500, 500, 520])
    params = flat_params(capital=1_000_000, max_position_inr=1_000_000)
    result = simulate(
        {"AAA": quiet, "ZZZ": strong},
        {"AAA": signal_frame(quiet, entry=0, atr_value=10, udr=1.2), "ZZZ": signal_frame(strong, entry=0, atr_value=10, udr=4.0)},
        params,
    )
    assert result.equity["position"].iloc[-1] == 1
    # ZZZ is the name that was bought: 2,000 shares at 500, marked at 520.
    assert result.equity["equity"].iloc[-1] == pytest.approx(2000 * 520)


def test_a_second_signal_does_not_add_to_an_open_position():
    dates = pd.bdate_range("2024-06-03", periods=4)
    bars = frame(dates, [100, 100, 100, 100], [101, 101, 101, 101], [99, 99, 99, 99], [100, 100, 100, 110])
    signals = signal_frame(bars, entry=0, atr_value=10)
    signals.iloc[2, signals.columns.get_loc("entry")] = True
    signals.iloc[2, signals.columns.get_loc("udr")] = 5
    params = flat_params(capital=3_000_000, max_position_inr=5_000_000)
    result = simulate({"AAA": bars}, {"AAA": signals}, params)
    assert result.trades.empty
    assert result.equity["equity"].iloc[-1] == pytest.approx(3_000_000 - 1_500_000 + 15_000 * 110)


def test_buy_and_sell_prices_include_slippage_and_delivery_charges():
    dates = pd.bdate_range("2024-06-03", periods=3)
    bars = frame(dates, [100, 100, 100], [101, 101, 101], [99, 99, 99], [100, 100, 100])
    signals = signal_frame(bars, entry=0, exit_=1, atr_value=100)
    params = flat_params(
        slippage_bps=100,
        brokerage_bps=10,
        gst_pct=18,
        capital=10_000,
        max_position_inr=10_000,
    )
    result = simulate({"AAA": bars}, {"AAA": signals}, params)
    trade = result.trades.iloc[0]
    # 10,000 / 20 / ATR 100 = 5 shares.
    assert trade["shares"] == 5
    assert trade["entry_price"] == pytest.approx(101)
    assert trade["exit_price"] == pytest.approx(99)
    buy_cost = share_charges(101 * 5, "buy", params)
    sell_cost = share_charges(99 * 5, "sell", params)
    assert buy_cost == pytest.approx(505 * 0.001 * 1.18)
    assert trade["pnl"] == pytest.approx((99 - 101) * 5 - buy_cost - sell_cost)
    assert affordable_shares(1000, 100, flat_params(brokerage_bps=1000)) == 9


def test_share_count_uses_portfolio_over_20_and_grows_after_a_gain():
    dates = pd.bdate_range("2024-06-03", periods=6)
    bars = frame(
        dates,
        [100, 100, 200, 200, 200, 200],
        [100, 200, 200, 200, 200, 200],
        [100, 100, 200, 200, 200, 200],
        [100, 200, 200, 200, 200, 200],
    )
    signals = signal_frame(bars, entry=0, exit_=2, atr_value=100)
    signals.iloc[2, signals.columns.get_loc("entry")] = True
    signals.iloc[4, signals.columns.get_loc("exit")] = True
    result = simulate(
        {"AAA": bars},
        {"AAA": signals},
        flat_params(capital=4_000_000, max_position_inr=4_000_000, partial_pct=0),
    )
    # 40 lakh / 20 / ATR 100 = 2,000 shares. A fixed 1 lakh would buy 1,000.
    assert result.trades.iloc[0]["shares"] == 2_000
    # Sold at 200, so the account is 42 lakh. The next buy is floor(2.1 lakh / 100).
    assert result.trades.iloc[1]["shares"] == 2_100
    doubled = flat_params(capital=4_000_000, max_position_inr=4_000_000, partial_pct=0, risk_per_atr=2)
    wider = simulate({"AAA": bars}, {"AAA": signals}, doubled)
    assert wider.trades.iloc[0]["shares"] == 4_000


def test_a_halted_name_fills_on_its_next_traded_open():
    dates = pd.to_datetime(["2024-06-03", "2024-06-07"])
    bars = frame(dates, [100, 80], [101, 101], [99, 79], [100, 100])
    signals = signal_frame(bars, entry=0, atr_value=10)
    result = simulate({"AAA": bars}, {"AAA": signals}, flat_params(capital=1_000_000, max_position_inr=5_000_000))
    assert result.trades.empty
    assert result.open_side == "long"
    assert result.equity["equity"].iloc[-1] == pytest.approx(1_000_000 - 80 * 5_000 + 5_000 * 100)


def test_grid_keeps_each_parameter_on_its_own_steps():
    space = MediumFrequencyMomentum().param_space()
    assert grid_values(space["udr_min"])[:3] == [0.5, 0.6, 0.7]
    assert grid_values(space["udr_weeks"])[0] == 8
    assert grid_values(space["brokerage_cap_inr"]) == [20]


def test_target_positions_points_at_the_portfolio_runner():
    with pytest.raises(RuntimeError, match="Run"):
        MediumFrequencyMomentum().target_positions(pd.DataFrame(), {})


def test_discover_loads_the_book_strategy():
    from strategies.registry import discover

    books = [item for item in discover() if item.stages]
    assert [item.key for item in books] == [
        "ema_crossover",
        "medium_frequency_momentum",
        "medium_frequency_momentum_no_stop",
        "momentum_sleeve",
        "weekly_drop",
        "weekly_trend_breakout",
    ]
    momentum = next(item for item in books if item.key == "medium_frequency_momentum")
    assert len(momentum.stages) == 5


def test_stages_cover_filter_buy_sell_stop_and_size():
    strategy = MediumFrequencyMomentum()
    assert [stage.key for stage in strategy.stages] == ["filter", "buy", "sell", "stop", "size"]
    named = [name for stage in strategy.stages for name in stage.params]
    assert len(named) == len(set(named))
    assert set(named).isdisjoint(strategy.cost_params)
    assert set(named) | set(strategy.cost_params) == set(strategy.param_space())
    params = {name: spec.coerce(spec.default) for name, spec in strategy.param_space().items()}
    lines = [strategy.stage_line(stage.key, params) for stage in strategy.stages]
    assert all(lines)
    assert "upside/downside" in lines[0]
    assert "portfolio value / 20" in lines[-1]


def _rising_base() -> pd.DataFrame:
    dates = pd.bdate_range("2018-01-01", periods=700)
    close = np.empty(700)
    close[:200] = np.linspace(80, 140, 200)
    close[200:280] = np.linspace(140, 100, 80)
    close[280:620] = 100
    for index in range(300, 600, 15):
        close[index] = 97
        close[index + 1] = 97.5
        close[index + 2] = 103
    close[620:] = np.linspace(100, 130, 80)
    high = close + 1.5
    low = close - 1.5
    pivot = int(dates.get_loc(pd.Timestamp("2020-05-22"))) - 8
    low[pivot] = close[pivot] - 8
    high[pivot] = close[pivot]
    low[pivot + 1] = close[pivot + 1] - 1
    high[pivot + 3] = close[pivot + 3] + 6
    close[pivot + 3] = close[pivot + 3] + 4
    bars = frame(dates, close, high, np.minimum(low, close), close)
    bars["high"] = np.maximum(bars["high"].to_numpy(), bars[["open", "close"]].max(axis=1).to_numpy())
    bars["low"] = np.minimum(bars["low"].to_numpy(), bars[["open", "close"]].min(axis=1).to_numpy())
    return bars


def test_one_completed_week_can_pass_every_entry_gate():
    bars = _rising_base()
    params = flat_params(
        udr_weeks=8,
        udr_min=0.5,
        udr_undefined_passes=1,
        monthly_ema=4,
        weekly_fast=3,
        weekly_slow=6,
        zone_width_pct=3.66,
        touch_rebound_pct=1.10,
        zone_proximity_pct=7.32,
        zone_lookback_days=120,
    )
    signals = prepare_signals(bars, params)
    assert int(signals["entry"].sum()) == 1
    blocked = dict(params)
    blocked["udr_min"] = 100
    blocked["udr_undefined_passes"] = 0
    assert not prepare_signals(bars, blocked)["entry"].any()
    far = dict(params)
    far["zone_proximity_pct"] = 0.04
    assert not prepare_signals(bars, far)["entry"].any()


def test_metrics_read_the_portfolio_equity():
    dates = pd.bdate_range("2024-06-03", periods=3)
    bars = frame(dates, [100, 100, 100], [101, 101, 101], [99, 99, 99], [100, 110, 110])
    result = simulate({"AAA": bars}, {"AAA": signal_frame(bars, entry=0, atr_value=10)}, flat_params(capital=1_000_000, max_position_inr=5_000_000))
    metrics = compute_metrics(result)
    assert metrics["ending_equity"] == pytest.approx(1_050_000)
    assert metrics["trade_count"] == 0
