"""Nifty 500 daily bars used by equity backtests."""

from __future__ import annotations

import pandas as pd
import pytest

from engine.data import load_bars
from engine.nse import ADJUSTED_PATH, available, listings, load_symbol, load_universe, lookup, missing_symbols


pytestmark = pytest.mark.skipif(not ADJUSTED_PATH.exists(), reason="NSE price file is not in the repo")


def test_universe_lists_reliance_and_skips_names_without_bars():
    assert available()
    reliance = lookup("RELIANCE")
    assert reliance is not None
    assert "Reliance" in reliance.company
    symbols = {item.symbol for item in listings()}
    assert "RELIANCE" in symbols
    assert "TCS" in symbols
    assert set(missing_symbols()) >= {"DUMMYHEG"}
    assert "DUMMYHEG" not in symbols


def test_reliance_adjusted_and_exchange_prices():
    adjusted = load_symbol("RELIANCE", adjusted=True)
    raw = load_symbol("RELIANCE", adjusted=False)
    assert list(adjusted.columns) == ["open", "high", "low", "close", "volume"]
    assert adjusted.index.is_monotonic_increasing
    assert adjusted.index.equals(raw.index)
    assert len(adjusted) > 1000
    assert not adjusted["close"].equals(raw["close"])
    assert adjusted["high"].ge(adjusted["low"]).all()
    reloaded = load_bars("nse:RELIANCE:adjusted")
    assert reloaded["close"].iloc[-1] == adjusted["close"].iloc[-1]


def test_universe_batch_matches_the_single_symbol_load():
    alone = load_symbol("TCS", adjusted=True)
    batch = load_universe(adjusted=True, symbols=["tcs"])
    assert list(batch) == ["TCS"]
    pd.testing.assert_frame_equal(batch["TCS"], alone)


def test_unknown_symbol_is_rejected():
    with pytest.raises(ValueError, match="No daily bars"):
        load_symbol("NOTASYMBOL")
