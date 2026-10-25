"""Train/test split, random-forest search, and the book chart contract."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd

from engine.book import COST_PARAMS, check_book_strategy, indicator_names, new_strategy_source
from engine.forest import (
    book_strategies,
    calendar,
    evaluate_book,
    judge,
    optimize_book,
    ratio_score,
    repair_lengths,
    research_split,
    study_book,
    window_dates,
)
from engine.models import BacktestResult, BacktestSettings, ParamSpec
from engine.metrics import monthly_returns
from engine.report import equity_line_figure, monthly_returns_html, rolling_return_figure, style_monthly_returns, symbol_price_figure
from engine.strategy import Stage, Strategy
from strategies.ema_crossover import EmaCrossover
from strategies.medium_frequency_momentum import MediumFrequencyMomentum
from strategies.medium_frequency_momentum_no_stop import MediumFrequencyMomentumNoStop
from strategies.weekly_drop import WeeklyDrop


def _series(n: int, seed: int, drift: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2016-01-04", periods=n)
    close = 80 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    close = np.maximum(close + 6 * np.sin(np.linspace(0, 10 * np.pi, n)), 5)
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.01,
            "low": np.minimum(open_, close) * 0.99,
            "close": close,
            "volume": 1_000.0,
        },
        index=index,
    )


def _data(n: int = 180) -> dict[str, pd.DataFrame]:
    return {"AAA": _series(n, 1, 0.0008), "BBB": _series(n, 2, -0.0002)}


class TinyBook(Strategy):
    key = "tiny_book"
    name = "Tiny Book"
    stages = (
        Stage("filter", "Filter", "Close is above the slow average.", (("Trend", ("slow",)),)),
        Stage("buy", "Buy", "Fast average crosses above the slow average.", (("Cross", ("fast",)),)),
        Stage("sell", "Sell", "Fast average crosses below the slow average.", (("Cross", ()),)),
        Stage("size", "Position sizing", "Fixed risk budget.", (("Shares", ("risk_per_atr",)),)),
    )
    cost_params = COST_PARAMS

    def param_space(self):
        space = {
            "fast": ParamSpec(4, 2, 8, 1, "int"),
            "slow": ParamSpec(12, None, None, None, "int"),
            "atr_length": ParamSpec(5, None, None, None, "int"),
            "stop_atr": ParamSpec(3.0, None, None, None, "float"),
            "partial_pct": ParamSpec(100, None, None, None, "int"),
            "capital": ParamSpec(200_000, None, None, None, "int"),
            "max_position_inr": ParamSpec(100_000, None, None, None, "int"),
            "risk_per_atr": ParamSpec(20_000, None, None, None, "int"),
            "entry_rank": ParamSpec(2, None, None, None, "int"),
        }
        for name in COST_PARAMS:
            space[name] = ParamSpec(0 if name != "gst_pct" else 0, None, None, None, "float")
        return space

    def signals(self, bars, params):
        from engine.indicators import atr, sma

        fast_n = int(params["fast"])
        slow_n = int(params["slow"])
        fast = sma(bars["close"], fast_n)
        slow = sma(bars["close"], slow_n)
        width = atr(bars, int(params["atr_length"]))
        entry = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        exit_ = (fast < slow) & (fast.shift(1) >= slow.shift(1))
        if fast_n >= slow_n:
            entry = entry & False
        return pd.DataFrame(
            {"entry": entry.fillna(False), "exit": exit_.fillna(False), "atr": width},
            index=bars.index,
        )

    def indicator_frame(self, bars, params):
        from engine.indicators import sma

        fast = sma(bars["close"], int(params["fast"]))
        slow = sma(bars["close"], int(params["slow"]))
        frame = pd.DataFrame({"Fast": fast, "Slow": slow}, index=bars.index)
        return frame, {"Fast": "overlay", "Slow": "overlay"}


def test_train_window_ends_before_the_test_window():
    study = evaluate_book(TinyBook(), _data())
    assert study.train_result.equity.index[-1] <= study.split_end
    assert study.test_result.equity.index[0] >= study.test_start
    assert study.test_start > study.split_end
    assert study.optimized is False
    assert abs(study.train_score - ratio_score(study.train_metrics)) < 1e-9


def test_report_names_the_train_and_test_dates():
    data = _data()
    study = evaluate_book(TinyBook(), data)
    dates = calendar(data)
    split_end, test_start = research_split(data)
    train, test = window_dates(study)
    assert train == f"{pd.Timestamp(dates[0]):%d %b %Y} to {split_end:%d %b %Y}"
    assert test == f"{test_start:%d %b %Y} to {pd.Timestamp(dates[-1]):%d %b %Y}"


def test_search_does_not_read_the_test_window():
    data = _data()
    kwargs = dict(searched=["fast"], n_samples=4, n_folds=2, n_shortlist=1, n_candidates=12, seed=3)
    first = optimize_book(TinyBook(), data, **kwargs)
    altered = {symbol: frame.copy() for symbol, frame in data.items()}
    for frame in altered.values():
        test = frame.index >= first.test_start
        frame.loc[test, ["open", "high", "low", "close"]] = frame.loc[test, ["open", "high", "low", "close"]] * 4 + 30
    second = optimize_book(TinyBook(), altered, **kwargs)

    assert first.scored_ends
    assert all(end <= first.split_end for end in first.scored_ends)
    assert first.params["fast"] == second.params["fast"]
    assert first.optimized is True
    assert len(first.folds) == 2
    assert first.test_result.equity.index[0] >= first.test_start


def test_holds_when_the_test_ratio_stays_close():
    verdict, _ = judge(1.0, 0.8)
    assert verdict == "Holds on test"
    verdict, _ = judge(1.0, 0.4)
    assert verdict == "Decays on test"
    verdict, _ = judge(1.0, -0.2)
    assert verdict == "Does not hold"


def test_momentum_exposes_the_indicators_the_chart_draws():
    index = pd.bdate_range("2018-01-02", periods=420)
    rng = np.random.default_rng(4)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(index))))
    open_ = np.r_[close[0], close[:-1]]
    bars = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.01,
            "low": np.minimum(open_, close) * 0.99,
            "close": close,
            "volume": 1_000.0,
        },
        index=index,
    )
    strategy = MediumFrequencyMomentum()
    assert check_book_strategy(strategy) == []
    frame, kinds = strategy.indicator_frame(bars, {})
    assert list(kinds) == ["Weekly fast EMA", "Weekly slow EMA", "Monthly EMA", "UDR"]
    assert kinds["Weekly fast EMA"] == "overlay"
    assert "ATR" not in kinds
    assert frame.index.equals(bars.index)


def test_a_new_book_is_included_in_the_backtest(tmp_path):
    source = new_strategy_source("SmaBook", "sma_book", "SMA Book")
    path = tmp_path / "sma_book.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("sma_book_future", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    strategy = module.SmaBook()
    plain = Strategy()
    plain.key = "plain"
    assert book_strategies([plain, strategy]) == [strategy]
    assert indicator_names(strategy)


def test_new_strategy_source_is_a_runnable_book(tmp_path):
    source = new_strategy_source("SmaBook", "sma_book", "SMA Book")
    path = tmp_path / "sma_book.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("sma_book_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    strategy = module.SmaBook()
    assert check_book_strategy(strategy) == []
    study = evaluate_book(strategy, _data(160))
    assert study.strategy_name == "SMA Book"
    assert len(study.train_result.equity) and len(study.test_result.equity)


def test_symbol_chart_marks_train_and_test_fills():
    index = pd.bdate_range("2024-01-02", periods=6)
    bars = pd.DataFrame(
        {
            "open": [10, 11, 12, 13, 12, 11],
            "high": [11, 12, 13, 14, 13, 12],
            "low": [9, 10, 11, 12, 11, 10],
            "close": [10.5, 11.5, 12.5, 13.5, 12.5, 11.5],
            "volume": [1, 1, 1, 1, 1, 1],
        },
        index=index,
    )
    trades = pd.DataFrame(
        {
            "entry_time": [index[1], index[4]],
            "exit_time": [index[2], index[5]],
            "entry_price": [11.0, 12.0],
            "exit_price": [12.0, 11.0],
            "window": ["train", "test"],
            "reason": ["crossover", "stop"],
            "symbol": ["AAA", "AAA"],
        }
    )
    indicators = pd.DataFrame({"Fast": bars["close"].rolling(2).mean()}, index=index)
    figure = symbol_price_figure(
        bars,
        trades,
        indicators,
        {"Fast": "overlay"},
        title="AAA",
        test_start=index[4],
    )
    names = [trace.name for trace in figure.data]
    assert "Train buy" in names
    assert "Test buy" in names
    assert "Train sell" in names
    assert "Fast" in names


def test_each_book_searches_indicator_settings_only():
    ema = set(indicator_names(EmaCrossover()))
    assert ema == {
        "fast",
        "slow",
        "rsi_length",
        "rsi_buy",
        "high_days",
        "max_drop_pct",
        "atr_length",
    }
    drop = set(indicator_names(WeeklyDrop()))
    assert drop == {"months", "drop_pct", "atr_length"}
    momentum = indicator_names(MediumFrequencyMomentum())
    assert indicator_names(MediumFrequencyMomentumNoStop()) == momentum
    assert {"weekly_fast", "weekly_slow", "udr_weeks", "monthly_ema", "min_touches"} <= set(momentum)
    fixed = {"stop_atr", "partial_pct", "capital", "entry_rank", "slippage_bps", "stop_pct"}
    assert fixed.isdisjoint(momentum)
    assert fixed.isdisjoint(ema)
    assert "stop_pct" not in drop


def test_a_fast_length_is_kept_below_the_slow_length():
    space = EmaCrossover().param_space()
    repaired = repair_lengths(space, {"fast": 15, "slow": 3, "rsi_length": 14})
    assert repaired is not None
    assert repaired["fast"] < repaired["slow"]
    assert repaired["rsi_length"] == 14
    untouched = repair_lengths(space, {"fast": 10, "slow": 30})
    assert untouched["fast"] == 10 and untouched["slow"] == 30


def test_study_book_locks_the_train_choice():
    kwargs = dict(n_samples=4, n_folds=2, n_shortlist=1, n_candidates=8, seed=2)
    study = study_book(EmaCrossover(), _data(160), **kwargs)
    assert study.optimized is True
    assert set(study.searched) == set(indicator_names(EmaCrossover()))
    assert study.params["fast"] < study.params["slow"]
    assert study.params["capital"] == 2_000_000
    assert study.params["partial_pct"] == 100
    assert study.scored_ends
    assert all(end <= study.split_end for end in study.scored_ends)
    assert study.test_result.equity.index[0] >= study.test_start

    from app.shared import _param_table

    table = _param_table(study)
    fast = table[table["Parameter"] == "Fast EMA"].iloc[0]
    capital = table[table["Parameter"] == "Portfolio capital, rupees"].iloc[0]
    assert fast["Source"] == "Train choice"
    assert fast["Used"] == study.params["fast"]
    assert capital["Source"] == "File"
    assert capital["Used"] == 2_000_000


def test_study_book_does_not_read_the_test_window():
    data = _data()
    kwargs = dict(n_samples=4, n_folds=2, n_shortlist=1, n_candidates=8, seed=3)
    first = study_book(TinyBook(), data, **kwargs)
    altered = {symbol: frame.copy() for symbol, frame in data.items()}
    for frame in altered.values():
        test = frame.index >= first.test_start
        frame.loc[test, ["open", "high", "low", "close"]] = frame.loc[test, ["open", "high", "low", "close"]] * 4 + 30
    second = study_book(TinyBook(), altered, **kwargs)
    assert first.params["fast"] == second.params["fast"]
    assert first.params["slow"] == 12
    assert first.searched == ["fast"]


def test_portfolio_value_is_a_line():
    index = pd.bdate_range("2024-01-02", periods=4)
    equity = pd.DataFrame(
        {"equity": [100.0, 110.0, 105.0, 120.0], "drawdown": [0, 0, -0.05, 0], "position": [0, 1, 1, 0]},
        index=index,
    )
    figure = equity_line_figure(BacktestResult(equity, pd.DataFrame(), BacktestSettings()), title="Test portfolio value")
    assert figure.layout.title.text == "Test portfolio value"
    assert figure.data[0].name == "Portfolio"
    assert list(figure.data[0].y) == [100.0, 110.0, 105.0, 120.0]


def test_rolling_return_and_monthly_heatmap():
    series = pd.Series([0.10, -0.05], index=pd.to_datetime(["2020-01-02", "2020-06-01"]))
    figure = rolling_return_figure(series, title="Test · 1 year", color="#1d4ed8")
    assert figure.layout.title.text == "Test · 1 year"
    assert figure.layout.yaxis.tickformat == ".0%"
    assert list(figure.data[0].y) == [0.10, -0.05]
    path = pd.Series(
        [100.0, 110.0, 99.0],
        index=pd.to_datetime(["2020-01-02", "2020-01-31", "2020-02-28"]),
    )
    rendered = style_monthly_returns(monthly_returns(path)).to_html()
    assert "10.0%" in rendered
    assert "background-color: #15803d" in rendered
    assert "-10.0%" in rendered
    assert "background-color: #dc2626" in rendered
    page = monthly_returns_html(monthly_returns(path))
    assert "overflow-x: auto" in page
    assert "position: sticky" in page
    assert "10.0%" in page
