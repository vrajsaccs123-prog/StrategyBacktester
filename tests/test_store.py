"""Saved runs round-trip through SQLite."""

from __future__ import annotations

import pandas as pd

from engine.backtest import run_backtest
from engine.models import BacktestSettings
from engine.store import Store


def test_strategy_and_run_roundtrip(tmp_path):
    store = Store(tmp_path / "bank.sqlite")
    spec = {"name": "Saved", "side": "long_only", "indicators": [], "entry": {"all": []}, "exit": {"all": []}}
    strategy_id = store.save_strategy(spec)
    assert store.get_strategy(strategy_id)["name"] == "Saved"
    store.save_strategy({**spec, "name": "Renamed"}, strategy_id)
    assert store.get_strategy(strategy_id)["name"] == "Renamed"

    index = pd.bdate_range("2024-01-01", periods=4)
    bars = pd.DataFrame(
        {"open": [10, 10, 12, 12], "high": [11, 11, 13, 13], "low": [9, 9, 11, 11], "close": [10, 11, 12, 12], "volume": 1},
        index=index,
    )
    result = run_backtest(bars, pd.Series([0, 1, 1, 0], index=index), BacktestSettings(initial_capital=1_000))
    run_id = store.save_run(
        strategy_key="rule:1",
        strategy_name="Renamed",
        dataset_name="bars.csv",
        dataset_path="bars.csv",
        params={"fast": 10},
        settings=result.settings.as_dict(),
        metrics={"sharpe": float("inf"), "cagr": float("nan"), "trade_count": 1},
        trades=result.trades,
        equity=result.equity,
        spec=spec,
        role="backtest",
    )
    loaded = store.get_run(run_id)
    assert loaded["metrics"]["sharpe"] == float("inf")
    assert pd.isna(loaded["metrics"]["cagr"])
    assert loaded["equity"]["equity"].iloc[-1] == result.equity["equity"].iloc[-1]
    assert len(loaded["trades"]) == len(result.trades)
    assert list(store.list_runs())[0]["id"] == run_id
    store.delete_run(run_id)
    assert store.get_run(run_id) is None
