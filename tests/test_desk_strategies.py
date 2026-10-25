"""The eight desk strategies: book shape, coarse grids, and a few signal rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.book import check_book_strategy, indicator_names
from engine.forest import _draw
from engine.indicator_impl import ols
from engine.portfolio import simulate
from engine.strategy import resolve_params
from strategies._desk import rolling_linreg_end
from strategies.s01_trend_core import ATR_MULTS, TrendCore, prepare_signals as trend_signals
from strategies.s02_donchian_volume_breakout import DonchianVolumeBreakout
from strategies.s03_uptrend_pullback import UptrendPullback, prepare_signals as pullback_signals
from strategies.s04_macd_supertrend_swing import MacdSupertrendSwing
from strategies.s05_bollinger_range_reversion import BollingerRangeReversion
from strategies.s06_squeeze_breakout import SqueezeBreakout
from strategies.s07_ichimoku_trend import IchimokuTrend
from strategies.s08_momentum_rotation import MomentumRotation, prepare_book, rebalance_dates

BOOKS = (
    TrendCore,
    DonchianVolumeBreakout,
    UptrendPullback,
    MacdSupertrendSwing,
    BollingerRangeReversion,
    SqueezeBreakout,
    IchimokuTrend,
    MomentumRotation,
)

FIXED = {
    "stop_atr",
    "partial_pct",
    "capital",
    "entry_rank",
    "max_position_inr",
    "max_position_pct",
    "risk_per_atr",
    "atr_length",
    "slippage_bps",
    "stop_pct",
    "max_positions",
}


def quiet(strategy, **overrides) -> dict:
    params = resolve_params(strategy().param_space(), {})
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


def bars_from(close, high=None, low=None, volume=None, start="2018-01-02") -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    high = close + 1.0 if high is None else np.asarray(high, dtype=float)
    low = close - 1.0 if low is None else np.asarray(low, dtype=float)
    volume = np.full(len(close), 1_000.0) if volume is None else np.asarray(volume, dtype=float)
    index = pd.bdate_range(start, periods=len(close))
    return pd.DataFrame(
        {"open": np.r_[close[0], close[:-1]], "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


@pytest.mark.parametrize("factory", BOOKS)
def test_each_strategy_is_a_book(factory):
    strategy = factory()
    assert check_book_strategy(strategy) == []
    assert indicator_names(strategy)
    assert FIXED.isdisjoint(indicator_names(strategy))
    frame, kinds = strategy.indicator_frame(bars_from(np.linspace(100, 130, 90)), quiet(factory))
    assert set(kinds) <= set(frame.columns)
    assert set(kinds.values()) <= {"overlay", "oscillator"}


def test_coarse_grids_are_the_listed_points():
    rng = np.random.default_rng(1)
    trend = TrendCore().param_space()
    assert { _draw(rng, trend["fast"]) for _ in range(40)} == {10, 20, 30}
    assert {_draw(rng, trend["slow"]) for _ in range(40)} == {50, 100}
    assert {_draw(rng, trend["atr_mult"]) for _ in range(40)} == {0, 1, 2}
    assert ATR_MULTS == (2.5, 3.0, 4.0)

    pullback = UptrendPullback().param_space()
    assert {_draw(rng, pullback["rsi_period"]) for _ in range(40)} == {2, 3, 4}
    assert {_draw(rng, pullback["rsi_thresh"]) for _ in range(40)} == {10, 20, 30}
    assert {_draw(rng, pullback["exit_ma"]) for _ in range(30)} == {5, 10}

    bands = BollingerRangeReversion().param_space()
    assert {_draw(rng, bands["bb_std"]) for _ in range(40)} == {1.8, 2.0, 2.2}
    assert {_draw(rng, bands["chop_thresh"]) for _ in range(40)} == {0, 1, 2}
    assert {_draw(rng, bands["time_stop"]) for _ in range(40)} == {0, 1, 2}

    rotation = MomentumRotation().param_space()
    assert {_draw(rng, rotation["top_pct"]) for _ in range(40)} == {0.1, 0.2, 0.3}
    assert {_draw(rng, rotation["lookback"]) for _ in range(20)} == {126, 252}
    assert {_draw(rng, rotation["rebalance"]) for _ in range(20)} == {0, 1}


def test_trend_core_buys_the_rise_and_sells_the_break():
    rise = np.linspace(100, 280, 360)
    crash = np.linspace(280, 180, 30)
    close = np.r_[rise, crash]
    high = np.r_[rise + 0.4, crash + 0.4]
    low = np.r_[rise - 0.4, crash - 6.0]
    frame = trend_signals(bars_from(close, high, low), quiet(TrendCore))
    assert not frame["entry"].iloc[:300].any()
    assert frame["entry"].iloc[300:360].any()
    assert frame["exit"].iloc[-20:].any()
    assert np.allclose(
        frame.loc[frame["entry"], "atr"],
        frame.loc[frame["entry"], "stop_pad"],
        equal_nan=True,
    )


def test_pullback_accepts_either_trigger_and_skips_the_warmup():
    close = np.r_[np.full(260, 100.0), np.full(100, 150.0)]
    high = close + 1.0
    low = close - 1.0
    high[-1] = 200.0
    low[-1] = 140.0
    close[-1] = 150.0
    frame = pullback_signals(bars_from(close, high, low), quiet(UptrendPullback))
    assert frame["entry"].iloc[-1]
    assert not frame["entry"].iloc[:300].any()
    assert "stop_pad" in frame.columns


def test_regression_endpoint_matches_the_catalog():
    series = pd.Series(np.linspace(1, 40, 40) + np.sin(np.linspace(0, 6, 40)))
    _slope, end, _r2, _err = ols(series, 20)
    fitted = rolling_linreg_end(series.to_numpy(), 20)
    np.testing.assert_allclose(fitted, end.to_numpy(), equal_nan=True)


def test_rotation_rebalances_at_month_end_and_skips_warmup():
    dates = pd.bdate_range("2020-01-02", periods=40)
    assert rebalance_dates(dates, 0).is_month_end.any() or all(
        stamp == dates[dates.to_period("M") == stamp.to_period("M")][-1] for stamp in rebalance_dates(dates, 0)
    )
    weekly = rebalance_dates(dates, 1)
    monthly = rebalance_dates(dates, 0)
    assert len(weekly) >= len(monthly)

    index = pd.bdate_range("2019-01-02", periods=420)
    names = {}
    rng = np.random.default_rng(2)
    for offset, drift, noise in (
        ("A", 0.004, 0.002),
        ("B", 0.0035, 0.002),
        ("C", 0.004, 0.03),
        ("D", 0.003, 0.03),
        ("E", -0.001, 0.01),
        ("F", -0.001, 0.01),
        ("G", 0.0, 0.01),
        ("H", 0.0, 0.01),
    ):
        shock = rng.normal(drift, noise, len(index))
        close = 100 * np.exp(np.cumsum(shock))
        names[offset] = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000.0,
            },
            index=index,
        )
    params = quiet(MomentumRotation, lookback=126, top_pct=0.25, rebalance=0, trail_pct=20)
    signals, ratio = prepare_book(names, params)
    assert len(ratio) == len(index)
    entries = signals["A"]["entry"]
    assert not entries.iloc[:300].any()
    if entries.any():
        stamped = entries.index[entries.to_numpy()]
        expected = set(rebalance_dates(index, 0))
        assert set(stamped) <= expected
    leader = int(signals["A"]["entry"].sum() + signals["B"]["entry"].sum())
    laggard = int(signals["E"]["entry"].sum() + signals["F"]["entry"].sum())
    assert leader >= laggard


def test_size_stop_pad_percent_cap_and_slot_limit():
    dates = pd.bdate_range("2024-06-03", periods=4)
    bars = pd.DataFrame(
        {"open": [100, 100, 100, 100], "high": [101, 101, 101, 101], "low": [99, 94, 99, 99], "close": [100, 100, 100, 100], "volume": 1.0},
        index=dates,
    )
    signal = pd.DataFrame(
        {"entry": [True, False, False, False], "exit": False, "atr": 10.0, "udr": 1.0, "spread_atr": 1.0, "size": 0.5, "stop_pad": 5.0},
        index=dates,
    )
    params = quiet(TrendCore, capital=1_000_000, risk_per_atr=1, max_position_inr=5_000_000, max_position_pct=0, partial_pct=100)
    result = simulate({"AAA": bars}, {"AAA": signal}, params)
    # (1_000_000 / 20) / 10 * 0.5 = 2_500 shares. The low at 94 trades through the stop at 95.
    assert result.trades.iloc[0]["shares"] == 2500
    assert result.trades.iloc[0]["exit_price"] == pytest.approx(95)
    assert result.trades.iloc[0]["reason"] == "stop"

    capped = signal.copy()
    capped["size"] = 1.0
    wide = quiet(TrendCore, capital=1_000_000, risk_per_atr=1, max_position_inr=5_000_000, max_position_pct=10, partial_pct=100)
    capped_result = simulate({"AAA": bars}, {"AAA": capped}, wide)
    assert capped_result.trades.iloc[0]["shares"] == 1000

    other = bars.copy()
    first = signal.copy()
    first["size"] = 1.0
    first["udr"] = 1.0
    second = first.copy()
    second["udr"] = 5.0
    limited = quiet(
        TrendCore,
        capital=1_000_000,
        risk_per_atr=1,
        max_position_inr=5_000_000,
        max_position_pct=0,
        max_positions=1,
        partial_pct=100,
    )
    both = simulate({"AAA": bars, "BBB": other}, {"AAA": first, "BBB": second}, limited)
    assert list(both.trades["symbol"].unique()) == ["BBB"]

    blocked = signal.copy()
    blocked["size"] = 1.0
    blocked["stop_pad"] = np.nan
    blocked["stop_price"] = 110.0
    skipped = simulate({"AAA": bars}, {"AAA": blocked}, params)
    assert skipped.trades.empty
