"""OHLC candles, range aggregation, log scale, and zoom on equity charts."""

from __future__ import annotations

import pandas as pd

from engine.models import BacktestResult, BacktestSettings
from engine.report import (
    DOWN_COLOR,
    UP_COLOR,
    comparison_figure,
    equity_figure,
    equity_ohlc,
    ohlc_figure,
    price_figure,
    resample_ohlc,
)


def _bars(index: pd.DatetimeIndex, opens, highs, lows, closes, volumes=None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes if volumes is not None else [1.0] * len(index),
        },
        index=index,
    )


def test_week_month_quarter_and_year_keep_ohlc():
    index = pd.to_datetime(
        ["2024-01-02", "2024-01-05", "2024-03-28", "2024-04-01", "2024-06-28", "2025-01-02"]
    )
    bars = _bars(
        index,
        opens=[10, 12, 11, 20, 22, 30],
        highs=[15, 16, 14, 25, 24, 33],
        lows=[9, 11, 8, 19, 18, 29],
        closes=[12, 13, 10, 21, 23, 31],
        volumes=[1, 2, 3, 4, 5, 6],
    )

    week = resample_ohlc(bars.iloc[:2], "Week")
    assert len(week) == 1
    assert week.iloc[0][["open", "high", "low", "close", "volume"]].tolist() == [10, 16, 9, 13, 3]

    month = resample_ohlc(bars.iloc[:3], "Month")
    assert len(month) == 2
    january = month.iloc[0]
    assert january["open"] == 10
    assert january["high"] == 16
    assert january["low"] == 9
    assert january["close"] == 13

    quarter = resample_ohlc(bars.iloc[:5], "Quarter")
    assert len(quarter) == 2
    assert quarter.iloc[0][["open", "high", "low", "close"]].tolist() == [10, 16, 8, 10]
    assert quarter.iloc[1][["open", "high", "low", "close"]].tolist() == [20, 25, 18, 23]

    year = resample_ohlc(bars, "Year")
    assert len(year) == 2
    assert year.iloc[0]["open"] == 10
    assert year.iloc[0]["close"] == 23
    assert year.iloc[1]["open"] == 30
    assert year.iloc[1]["close"] == 31


def test_daily_equity_uses_previous_close_as_open():
    series = pd.Series([100.0, 110.0, 90.0], index=pd.bdate_range("2024-01-02", periods=3))
    view = equity_ohlc(series, "Day")
    assert view.iloc[1][["open", "high", "low", "close"]].tolist() == [100.0, 110.0, 100.0, 110.0]
    assert view.iloc[2][["open", "high", "low", "close"]].tolist() == [110.0, 110.0, 90.0, 90.0]
    assert view.iloc[2]["close"] < view.iloc[2]["open"]
    assert view.iloc[1]["close"] > view.iloc[1]["open"]


def test_longer_equity_range_uses_first_high_low_last():
    series = pd.Series([100.0, 130.0, 80.0, 90.0, 140.0], index=pd.bdate_range("2024-01-01", periods=5))
    view = equity_ohlc(series, "Week")
    assert len(view) == 1
    assert view.iloc[0][["open", "high", "low", "close"]].tolist() == [100.0, 140.0, 80.0, 140.0]


def test_price_and_equity_charts_are_log_zoomable_candles():
    index = pd.bdate_range("2024-01-01", periods=10)
    bars = _bars(
        index,
        opens=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        highs=[12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        lows=[9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        closes=[11, 10, 13, 12, 15, 14, 17, 16, 19, 18],
    )
    equity = pd.DataFrame(
        {
            "equity": [100, 110, 105, 120, 115, 130, 125, 140, 135, 150],
            "drawdown": [0, 0, -0.05, 0, -0.04, 0, -0.04, 0, -0.04, 0],
            "position": [1] * 10,
            "close": bars["close"].to_numpy(),
        },
        index=index,
    )
    result = BacktestResult(equity, pd.DataFrame(), BacktestSettings())

    price = price_figure(bars, result, bar_range="Week")
    assert price is not None
    candle = price.data[0]
    assert candle.type == "candlestick"
    assert candle.increasing.fillcolor == UP_COLOR
    assert candle.decreasing.fillcolor == DOWN_COLOR
    assert list(candle.open)[0] == 10
    assert price.layout.yaxis.type == "log"
    assert price.layout.yaxis2.type is None
    assert price.layout.dragmode == "zoom"
    assert price.layout.xaxis.fixedrange is not True
    assert price.layout.xaxis.rangeslider.visible is not True

    preview = ohlc_figure(bars, "Day")
    assert preview.data[0].type == "candlestick"
    assert preview.layout.yaxis.type == "log"
    assert preview.layout.xaxis.rangeslider.visible is True
    assert preview.layout.dragmode == "zoom"

    account = equity_figure(result, "Month")
    assert account.data[0].type == "candlestick"
    assert account.layout.yaxis.type == "log"
    assert account.data[0].decreasing.fillcolor == DOWN_COLOR

    compared = comparison_figure({"one": equity["equity"], "two": equity["equity"] * 2}, "Quarter")
    assert all(trace.type == "candlestick" for trace in compared.data)
    assert compared.layout.yaxis.type == "log"
    assert compared.layout.yaxis2.type == "log"


def test_non_positive_prices_stay_on_a_linear_axis():
    index = pd.bdate_range("2024-01-01", periods=2)
    bars = _bars(index, opens=[1, 1], highs=[2, 2], lows=[0, 1], closes=[1, 1])
    figure = ohlc_figure(bars, "Day")
    assert figure.layout.yaxis.type is None
