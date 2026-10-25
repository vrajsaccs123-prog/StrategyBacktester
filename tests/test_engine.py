"""CSV loading, rule signals, fills, costs, metrics, and holdout isolation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.backtest import run_backtest
from engine.data import load_csv
from engine.catalog import load_catalog
from engine.indicators import INDICATOR_DEFS, atr, compute_indicator, engine_for_catalog, rsi, sma
from engine.metrics import (
    average_hold_days,
    compute_metrics,
    drawdown_recovery,
    format_drawdown_days,
    format_metric,
    monthly_returns,
    rolling_returns,
)
from engine.models import BacktestResult, BacktestSettings
from engine.rules import apply_params, default_spec, eval_rule, extract_param_space, positions_from_flags, validate_spec
from engine.strategy import RuleStrategy
from strategies.sma_cross import SmaCross


def ohlc(opens, closes, start="2024-01-01") -> pd.DataFrame:
    opens = np.asarray(opens, dtype=float)
    closes = np.asarray(closes, dtype=float)
    high = np.maximum(opens, closes) + 0.1
    low = np.minimum(opens, closes) - 0.1
    index = pd.bdate_range(start, periods=len(opens))
    return pd.DataFrame(
        {"open": opens, "high": high, "low": low, "close": closes, "volume": 1_000.0},
        index=index,
    )


def test_sample_csv_loads():
    frame = load_csv("data/sample_ohlcv.csv")
    assert len(frame) == 800
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert frame.index.is_monotonic_increasing


def test_load_csv_aliases_and_sort(tmp_path):
    path = tmp_path / "bars.csv"
    path.write_text(
        "Date,Open,High,Low,Close,Volume\n"
        "2024-01-03,12,13,11,12.5,5\n"
        "2024-01-02,10,11,9,10.5,4\n",
        encoding="utf-8",
    )
    frame = load_csv(path)
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert frame.index[0] == pd.Timestamp("2024-01-02")
    assert frame.iloc[0]["close"] == pytest.approx(10.5)


def test_load_csv_rejects_duplicate_timestamps(tmp_path):
    path = tmp_path / "dup.csv"
    path.write_text(
        "date,open,high,low,close\n2024-01-02,1,2,1,1\n2024-01-02,1,2,1,1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_csv(path)


def test_load_csv_rejects_high_below_low(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        "date,open,high,low,close\n2024-01-02,1,1,2,1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="high below low"):
        load_csv(path)


def test_sma_matches_trailing_window():
    close = pd.Series([1, 2, 3, 4, 5], index=pd.bdate_range("2024-01-01", periods=5), dtype=float)
    values = sma(close, 3)
    assert pd.isna(values.iloc[1])
    assert values.iloc[2] == pytest.approx(2)
    assert values.iloc[4] == pytest.approx(4)


def test_donchian_uses_prior_bars_only():
    bars = ohlc([1, 2, 3, 4], [1, 2, 3, 10])
    bars["high"] = [1, 5, 4, 3]
    bars["low"] = [0.5, 1, 1, 1]
    channel = compute_indicator("donchian", bars, {"length": 2})
    # Bar 3 compares against highs on bars 1 and 2, not its own high.
    assert channel["upper"].iloc[3] == pytest.approx(5)


def test_indicators_emit_named_columns():
    bars = ohlc(np.linspace(10, 20, 80), np.linspace(10, 21, 80))
    expected = {
        "sma": ["value"],
        "ema": ["value"],
        "rsi": ["value"],
        "macd": ["macd", "signal", "hist"],
        "bollinger": ["mid", "upper", "lower"],
        "atr": ["value"],
        "stochastic": ["k", "d"],
        "adx": ["adx", "plus_di", "minus_di"],
        "donchian": ["upper", "lower", "mid"],
        "roc": ["value"],
    }
    assert set(expected) <= set(INDICATOR_DEFS)
    for kind, columns in expected.items():
        frame = compute_indicator(kind, bars, {name: spec.default for name, spec in INDICATOR_DEFS[kind].params.items()})
        assert list(frame.columns) == columns
        assert frame.iloc[-1].notna().all()


def _broad_bars(n=900) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
    opened = np.empty(n)
    opened[0] = close[0]
    opened[1:] = close[:-1]
    high = np.maximum(opened, close) * (1 + rng.uniform(0.0, 0.006, n))
    low = np.minimum(opened, close) * (1 - rng.uniform(0.0, 0.006, n))
    low[40] = high[38] + 1
    high[40] = low[40] + 2
    opened[40] = low[40] + 0.4
    close[40] = low[40] + 1.2
    high = np.maximum(high, np.maximum(opened, close))
    low = np.minimum(low, np.minimum(opened, close))
    index = pd.bdate_range("2018-01-01", periods=n)
    return pd.DataFrame(
        {"open": opened, "high": high, "low": low, "close": close, "volume": rng.uniform(1_000, 5_000, n)},
        index=index,
    )


def test_catalog_indicators_match_the_description_and_compute():
    items = load_catalog()
    text = Path("indicators.md").read_text(encoding="utf-8")
    assert f"Total indicators: {len(items)}" in text
    assert len(items) == len({item["id"] for item in items})
    for item in items:
        assert f"`{item['id']}`" in text
    assert engine_for_catalog("bbands") == "bollinger"
    assert engine_for_catalog("stoch") == "stochastic"
    assert engine_for_catalog("greeks") is None
    assert INDICATOR_DEFS["wma"].label == "Weighted Moving Average"
    assert len([item for item in items if engine_for_catalog(item["id"])]) >= 170

    bars = _broad_bars()
    for kind, definition in INDICATOR_DEFS.items():
        params = {name: spec.default for name, spec in definition.params.items()}
        frame = compute_indicator(kind, bars, params)
        assert list(frame.columns) == definition.outputs
        for column in frame.columns:
            assert np.isfinite(frame[column].to_numpy(dtype=float)).any()


def test_new_indicator_formulas_and_rule_column():
    bars = ohlc([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    wma = compute_indicator("wma", bars, {"period": 3})["value"]
    assert wma.iloc[2] == pytest.approx((1 + 4 + 9) / 6)
    assert wma.iloc[4] == pytest.approx((3 + 8 + 15) / 6)

    bars = ohlc([10, 10, 10, 10], [10, 11, 10, 12])
    bars["volume"] = [1, 1, 2, 3]
    assert list(compute_indicator("obv", bars, {})["value"]) == pytest.approx([0, 1, -1, 2])

    typical = compute_indicator("typical_price", bars, {})["value"]
    assert typical.iloc[1] == pytest.approx((bars["high"].iloc[1] + bars["low"].iloc[1] + 11) / 3)

    rising = ohlc(np.linspace(10, 20, 80), np.linspace(10, 21, 80))
    natr = compute_indicator("natr", rising, {"period": 14})["value"]
    assert natr.iloc[-1] == pytest.approx(100 * atr(rising, 14).iloc[-1] / rising["close"].iloc[-1])
    rsi2 = compute_indicator("rsi2", rising, {"period": 2})["value"]
    assert rsi2.iloc[-1] == pytest.approx(rsi(rising["close"], 2).iloc[-1])

    flat = ohlc([5, 5], [5, 8])
    flat.loc[flat.index[0], ["open", "high", "low", "close"]] = [5, 6, 4, 5]
    doji = compute_indicator("candle_doji", flat, {"body_pct": 0.1})["value"]
    assert doji.iloc[0] == 1
    assert doji.iloc[1] == 0

    spec = default_spec()
    spec["indicators"].append({"id": "wave", "type": "wma", "params": {"period": 10}})
    spec["entry"]["all"].append({"left": "wave", "op": ">", "right": "close"})
    assert validate_spec(spec) == []


def test_rule_and_or_and_cross():
    index = pd.bdate_range("2024-01-01", periods=5)
    frame = pd.DataFrame(
        {
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": [1, 2, 3, 2, 1],
            "volume": 1.0,
            "a": [1, 2, 3, 2, 1],
            "b": [2, 2, 2, 2, 2],
        },
        index=index,
    )
    cross = eval_rule(frame, {"all": [{"left": "a", "op": "crosses_above", "right": "b"}]})
    assert list(cross) == [False, False, True, False, False]
    both = eval_rule(
        frame,
        {"all": [{"left": "a", "op": ">", "right": 1}, {"left": "b", "op": "==", "right": 2}]},
    )
    assert list(both) == [False, True, True, True, False]
    either = eval_rule(
        frame,
        {"any": [{"left": "a", "op": "<", "right": 0}, {"left": "a", "op": ">", "right": 2}]},
    )
    assert list(either) == [False, False, True, False, False]


def test_default_spec_param_roundtrip():
    spec = default_spec()
    assert validate_spec(spec) == []
    space = extract_param_space(spec)
    assert "ind.fast.length" in space
    updated = apply_params(spec, {"ind.fast.length": 15})
    assert updated["indicators"][0]["params"]["length"]["value"] == 15


def test_next_bar_fill_uses_the_following_open():
    bars = ohlc([10, 10, 20, 20, 7], [10, 11, 20, 21, 7])
    target = pd.Series([0, 1, 1, 0, 0], index=bars.index)
    result = run_backtest(bars, target, BacktestSettings(initial_capital=2_000))
    trade = result.trades.iloc[0]
    assert trade["entry_time"] == bars.index[2]
    assert trade["exit_time"] == bars.index[4]
    assert trade["entry_price"] == pytest.approx(20)
    assert trade["exit_price"] == pytest.approx(7)
    assert result.equity["equity"].iloc[-1] == pytest.approx(700)


def test_sma_cross_trades_match_hand_calculation():
    closes = [10, 10, 10, 10, 14, 14, 14, 8, 8, 8]
    opens = [10, 10, 10, 10, 10, 20, 14, 14, 5, 8]
    bars = ohlc(opens, closes)
    target = SmaCross().target_positions(bars, {"fast": 2, "slow": 3})
    result = run_backtest(bars, target, BacktestSettings(initial_capital=2_000))
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["entry_time"] == bars.index[5]
    assert trade["exit_time"] == bars.index[8]
    assert trade["entry_price"] == pytest.approx(20)
    assert trade["exit_price"] == pytest.approx(5)
    assert trade["pnl"] == pytest.approx(-1_500)
    assert result.equity["equity"].iloc[-1] == pytest.approx(500)


def test_commission_reduces_profit():
    bars = ohlc([10, 10, 10, 12], [10, 10, 11, 12])
    target = pd.Series([0, 1, 1, 1], index=bars.index)
    plain = run_backtest(bars, target, BacktestSettings(initial_capital=10_000))
    charged = run_backtest(bars, target, BacktestSettings(initial_capital=10_000, commission_bps=100))
    assert charged.equity["equity"].iloc[-1] < plain.equity["equity"].iloc[-1]


def test_slippage_worsens_the_long_entry():
    bars = ohlc([10, 10, 20, 20], [10, 11, 20, 20])
    target = pd.Series([0, 1, 1, 1], index=bars.index)
    result = run_backtest(bars, target, BacktestSettings(initial_capital=2_000, slippage_bps=100))
    assert result.open_side == "long"
    # Still open, so the fill is visible once the position is closed on a longer series.
    closed = ohlc([10, 10, 20, 20, 20], [10, 11, 20, 20, 20])
    target = pd.Series([0, 1, 1, 0, 0], index=closed.index)
    result = run_backtest(closed, target, BacktestSettings(initial_capital=2_000, slippage_bps=100))
    assert result.trades.iloc[0]["entry_price"] == pytest.approx(20 * 1.01)


def test_short_round_trip_pnl():
    bars = ohlc([10, 10, 10, 9, 8], [10, 10, 9, 9, 8])
    target = pd.Series([0, -1, -1, 0, 0], index=bars.index)
    result = run_backtest(
        bars,
        target,
        BacktestSettings(initial_capital=1_000, allow_short=True),
    )
    trade = result.trades.iloc[0]
    assert trade["side"] == "short"
    assert trade["entry_time"] == bars.index[2]
    assert trade["exit_time"] == bars.index[4]
    assert trade["entry_price"] == pytest.approx(10)
    assert trade["exit_price"] == pytest.approx(8)
    assert trade["pnl"] == pytest.approx(200)
    assert result.equity["equity"].iloc[-1] == pytest.approx(1_200)


def test_long_only_setting_blocks_shorts():
    bars = ohlc([10, 10, 10], [10, 10, 10])
    target = pd.Series([-1, -1, -1], index=bars.index)
    result = run_backtest(bars, target, BacktestSettings(allow_short=False))
    assert (result.equity["position"] == 0).all()
    assert result.trades.empty


def test_rule_strategy_enters_on_cross():
    bars = ohlc([1, 1, 1, 1, 1, 1], [1, 1, 1, 5, 5, 5])
    spec = {
        "name": "Up",
        "side": "long_only",
        "indicators": [],
        "entry": {"all": [{"left": "close", "op": "crosses_above", "right": 2}]},
        "exit": {"all": []},
        "entry_short": {"all": []},
        "exit_short": {"all": []},
    }
    target = RuleStrategy(spec).target_positions(bars, {})
    assert list(target) == [0, 0, 0, 1, 1, 1]


def test_max_drawdown_on_known_equity():
    index = pd.bdate_range("2020-01-01", periods=4)
    equity = pd.DataFrame(
        {
            "equity": [100.0, 110.0, 90.0, 120.0],
            "drawdown": [0.0, 0.0, 90 / 110 - 1, 0.0],
            "position": [1, 1, 1, 1],
            "close": [1, 1, 1, 1],
        },
        index=index,
    )
    result = BacktestResult(equity, pd.DataFrame(columns=["pnl"]), BacktestSettings())
    metrics = compute_metrics(result)
    assert metrics["max_drawdown"] == pytest.approx(90 / 110 - 1)
    assert metrics["trade_count"] == 0
    assert metrics["starting_equity"] == pytest.approx(100)
    assert metrics["ending_equity"] == pytest.approx(120)
    assert metrics["total_return"] == pytest.approx(0.2)
    assert metrics["win_count"] == 0
    assert metrics["loss_count"] == 0
    assert math.isfinite(metrics["volatility"])
    assert metrics["volatility"] > 0


def test_trade_return_stats():
    index = pd.bdate_range("2020-01-01", periods=4)
    equity = pd.DataFrame(
        {
            "equity": [100.0, 110.0, 90.0, 120.0],
            "drawdown": [0.0, 0.0, 90 / 110 - 1, 0.0],
            "position": [1, 1, 1, 0],
            "close": [1, 1, 1, 1],
        },
        index=index,
    )
    trades = pd.DataFrame(
        {
            "pnl": [10.0, -5.0, 0.0],
            "return_pct": [0.10, -0.04, 0.0],
        }
    )
    metrics = compute_metrics(BacktestResult(equity, trades, BacktestSettings()))
    assert metrics["win_count"] == 1
    assert metrics["loss_count"] == 1
    assert metrics["trade_count"] == 3
    assert metrics["avg_win_return"] == pytest.approx(0.10)
    assert metrics["avg_loss_return"] == pytest.approx(-0.04)
    assert metrics["win_rate"] == pytest.approx(1 / 3)
    assert math.isnan(metrics["avg_hold_days"])


def test_drawdown_days_are_calendar_days_from_the_low_back_to_the_peak():
    index = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-13"])
    equity = pd.DataFrame(
        {
            "equity": [100.0, 80.0, 90.0, 100.0],
            "drawdown": [0.0, -0.2, -0.1, 0.0],
            "position": [1, 1, 1, 1],
            "close": [1, 1, 1, 1],
        },
        index=index,
    )
    trades = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(["2020-01-03", "2020-01-06"]),
            "exit_time": pd.to_datetime(["2020-01-06", "2020-01-16"]),
            "pnl": [1.0, -1.0],
            "return_pct": [0.01, -0.02],
        }
    )
    metrics = compute_metrics(BacktestResult(equity, trades, BacktestSettings()))
    assert metrics["drawdown_days"] == 10
    assert metrics["drawdown_recovered"] is True
    assert metrics["avg_hold_days"] == pytest.approx(6.5)
    assert format_drawdown_days(metrics) == "10 days"
    assert format_metric("avg_hold_days", metrics["avg_hold_days"]) == "6.5 days"


def test_open_drawdown_counts_through_the_last_day():
    index = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-31"])
    equity = pd.Series([100.0, 80.0, 90.0], index=index)
    days, recovered = drawdown_recovery(equity)
    assert days == 28
    assert recovered is False
    assert format_drawdown_days({"drawdown_days": days, "drawdown_recovered": recovered}) == "28 days, open"
    flat = pd.Series([100.0, 101.0, 102.0], index=index)
    assert drawdown_recovery(flat) == (0.0, True)


def test_rolling_and_monthly_returns():
    path = pd.Series(
        [100.0, 110.0, 99.0, 108.9],
        index=pd.to_datetime(["2020-01-02", "2020-01-31", "2020-02-28", "2020-03-31"]),
    )
    table = monthly_returns(path)
    assert table.loc[2020, "Jan"] == pytest.approx(0.10)
    assert table.loc[2020, "Feb"] == pytest.approx(-0.10)
    assert table.loc[2020, "Mar"] == pytest.approx(0.10)
    assert pd.isna(table.loc[2020, "Apr"])
    months = table.loc[2020, ["Jan", "Feb", "Mar"]].astype(float)
    assert float((1 + months).prod() - 1) == pytest.approx(table.loc[2020, "Year"])

    anniversary = pd.Series([100.0, 133.1], index=pd.to_datetime(["2017-06-01", "2020-06-01"]))
    rolled = rolling_returns(anniversary, 3)
    assert len(rolled) == 1
    assert rolled.iloc[0] == pytest.approx(0.10)
    assert rolling_returns(anniversary, 10).empty
    all_time = rolling_returns(anniversary, None)
    assert list(all_time.index) == list(anniversary.index)
    assert all_time.iloc[0] == pytest.approx(0.0)
    assert all_time.iloc[-1] == pytest.approx(0.331)
    early = pd.Series([100.0, 110.0], index=pd.to_datetime(["2019-06-03", "2020-06-02"]))
    assert rolling_returns(early, 1).empty
    held = average_hold_days(pd.DataFrame())
    assert math.isnan(held)
