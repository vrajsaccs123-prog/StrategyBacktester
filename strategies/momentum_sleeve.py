"""Momentum sleeve for a Nifty 500 book.

Five measures. A name can be bought only when its completed week closes above
a rising EMA and the equal-weight index is close to its trailing high. The
EMA and the rate of change are computed on the shared Friday calendar. The
buy is a place in the strongest rates of change. The sell is falling out of
that list, a giveback from the name's own trailing high, or the index
breaking its high. ATR sizes the shares and freezes a stop under the fill.

The book holds a handful of leaders and leaves when the market itself breaks,
which is what keeps a crash from becoming the drawdown. Slots, the rank
buffer, the trail, and the index band stay at the file values. The train
window may still choose the EMA length.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.book import COST_PARAMS, size_unit
from engine.indicators import atr, ema, roc
from engine.models import ParamSpec
from engine.strategy import Stage, Strategy, resolve_params
from strategies.medium_frequency_momentum import completed_ohlc

PARAM_LABELS = {
    "ema_weeks": "Weekly EMA length",
    "roc_weeks": "Rate of change, weeks",
    "slots": "Names bought",
    "hold_rank": "Sell when rank is worse than",
    "index_days": "Index high, sessions",
    "index_entry_pct": "Buy when the index is within this percent of its high",
    "index_exit_pct": "Sell when the index is this percent under its high",
    "trail_days": "Trailing high, sessions",
    "trail_pct": "Sell this percent under the high",
    "atr_length": "ATR length",
}

RANK_LABELS = {
    0: "Strongest rate of change",
    1: "Closest to the trailing high",
    2: "Symbol name",
}

CHOICES = {
    "entry_rank": tuple((RANK_LABELS[index], index) for index in range(3)),
}

STAGES = (
    Stage(
        "filter",
        "Filter",
        "The completed week closes above a rising EMA, and the equal-weight index is near its trailing high.",
        (
            ("Weekly trend", ("ema_weeks",)),
            ("Market", ("index_days", "index_entry_pct")),
        ),
    ),
    Stage(
        "buy",
        "Buy",
        "Buy the strongest rates of change that pass the filter. The fill is the next open.",
        (("Leaders", ("roc_weeks", "slots")),),
    ),
    Stage(
        "sell",
        "Sell",
        "Sell when the rank falls out of the wider list, the close gives back the trail, or the index breaks its high. The fill is the next open.",
        (
            ("Rank", ("hold_rank",)),
            ("Giveback", ("trail_days", "trail_pct")),
            ("Market break", ("index_exit_pct",)),
        ),
    ),
    Stage(
        "stop",
        "Stop loss",
        "The stop sells the shares. It is frozen under the entry fill, and it sells before a signal that fills on the same open.",
        (("Protective stop", ("stop_atr", "partial_pct")),),
    ),
    Stage(
        "size",
        "Position sizing",
        "Shares are the portfolio value divided by 20, times the size factor, divided by ATR, then cut by the rupee cap and by cash.",
        (
            ("Shares", ("atr_length", "risk_per_atr", "max_position_inr", "capital")),
            ("Several buys the same morning", ("entry_rank",)),
        ),
    ),
)

# Full-universe signals, reused across folds. The key is the indicator settings.
_CACHE: dict = {"key": None, "signals": {}, "ids": {}}


def stage_line(key: str, params: dict) -> str:
    """One sentence that states the current settings for a stage."""

    weeks = int(params["ema_weeks"])
    if key == "filter":
        return (
            f"The completed week closes above the rising {weeks}-week EMA, "
            f"and the equal-weight index is within {float(params['index_entry_pct']):g}% "
            f"of its {int(params['index_days'])}-session high."
        )
    if key == "buy":
        slots = int(params["slots"])
        noun = "name" if slots == 1 else "names"
        return (
            f"The {slots} strongest {int(params['roc_weeks'])}-week rates of change, "
            f"{noun} that pass the filter. The fill is the next open."
        )
    if key == "sell":
        return (
            f"Rank falls past {int(params['hold_rank'])}, the close is {float(params['trail_pct']):g}% "
            f"or more under the {int(params['trail_days'])}-session high, or the index is "
            f"{float(params['index_exit_pct']):g}% or more under its {int(params['index_days'])}-session high. "
            "The fill is the next open."
        )
    if key == "stop":
        return (
            f"A stop is frozen {float(params['stop_atr']):g} entry ATRs under the fill. "
            f"{int(params['partial_pct'])}% of the shares sell there."
        )
    if key == "size":
        rank = {
            0: "the strongest rate of change",
            1: "whoever is closest to the trailing high",
            2: "symbol name",
        }[int(params["entry_rank"])]
        unit = size_unit(params)
        return (
            f"Shares are floor({unit} / {int(params['atr_length'])}-day ATR), "
            f"a position of {unit} × price / ATR, "
            f"capped at ₹{int(params['max_position_inr']):,}. "
            f"Starting cash is ₹{int(params['capital']):,}. "
            f"Same-morning buys are filled by {rank}."
        )
    raise KeyError(key)


def _weekly(bars: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    finished = completed_ohlc(bars, "W-FRI")
    if finished.empty:
        empty = pd.Series(dtype=float)
        return finished, empty, empty
    average = ema(finished["close"], int(params["ema_weeks"]))
    momentum = roc(finished["close"], int(params["roc_weeks"]))
    return finished, average, momentum


def _trail(bars: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
    lookback = int(params["trail_days"])
    peak = bars["high"].rolling(lookback, min_periods=lookback).max().shift(1)
    deep = bars["close"] <= peak * (1.0 - float(params["trail_pct"]) / 100.0)
    return peak, deep.fillna(False)


def _market(closes: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Equal-weight index, and whether it is near its high or through the exit."""

    index = (1.0 + closes.pct_change().mean(axis=1).fillna(0.0)).cumprod()
    peak = index.rolling(int(params["index_days"]), min_periods=int(params["index_days"])).max().shift(1)
    near = index >= peak * (1.0 - float(params["index_entry_pct"]) / 100.0)
    broken = index <= peak * (1.0 - float(params["index_exit_pct"]) / 100.0)
    return index / peak - 1.0, near.fillna(False), broken.fillna(False)


def _frame(index: pd.DatetimeIndex, entry: pd.Series, exit_: pd.Series, width: pd.Series, momentum: pd.Series, peak: pd.Series, close: pd.Series) -> pd.DataFrame:
    distance = (close / peak - 1.0).replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(
        {
            "entry": entry.to_numpy(dtype=bool) & ~exit_.to_numpy(dtype=bool),
            "exit": exit_.to_numpy(dtype=bool),
            "atr": width.to_numpy(dtype=float),
            "udr": momentum.reindex(index).ffill().to_numpy(dtype=float),
            "spread_atr": (-distance).to_numpy(dtype=float),
        },
        index=index,
    )


def flat_signals(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    """No orders. Used for a name that cannot be ranked with the book."""

    index = bars.index
    width = atr(bars, int(params["atr_length"])) if len(bars) else pd.Series(dtype=float, index=index)
    return pd.DataFrame(
        {"entry": False, "exit": False, "atr": width, "udr": np.nan, "spread_atr": np.nan},
        index=index,
    )


def single_signals(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    """One name, used when the book is not available. The market filter is not applied."""

    index = bars.index
    width = atr(bars, int(params["atr_length"])) if len(bars) else pd.Series(dtype=float, index=index)
    empty = pd.DataFrame(
        {"entry": False, "exit": False, "atr": width, "udr": np.nan, "spread_atr": np.nan},
        index=index,
    )
    if bars.empty:
        return empty
    weekly, average, momentum = _weekly(bars, params)
    peak, deep = _trail(bars, params)
    if weekly.empty:
        return empty
    trend = ((weekly["close"] > average) & (average > average.shift(1))).fillna(False)
    ripe = (momentum > 0).fillna(False)
    setup = (trend & ripe).astype(float)
    own = weekly.index.intersection(index)
    turned = setup.astype(bool) & ~setup.astype(bool).shift(1, fill_value=False)
    entry = pd.Series(False, index=index)
    exit_ = pd.Series(False, index=index)
    entry.loc[own] = turned.reindex(own).fillna(False).to_numpy()
    broke = (weekly["close"] < average).fillna(False)
    exit_.loc[own] = broke.reindex(own).fillna(False).to_numpy()
    exit_.loc[deep.to_numpy()] = True
    return _frame(index, entry, exit_, width, momentum, peak, bars["close"])


def book_signals(bars_by_symbol: dict[str, pd.DataFrame], params: dict) -> dict[str, pd.DataFrame]:
    """Cross-sectional leaders. Rank uses only information known on that week.

    The EMA and the rate of change run on the shared Friday calendar. A week
    with no trade is a gap in that column, so every name is measured over the
    same dates.
    """

    weekly: dict[str, pd.DataFrame] = {}
    for symbol in sorted(bars_by_symbol):
        finished = completed_ohlc(bars_by_symbol[symbol], "W-FRI")
        if len(finished) < 80:
            continue
        weekly[symbol] = finished
    if not weekly:
        return {symbol: flat_signals(bars, params) for symbol, bars in bars_by_symbol.items()}
    closes = pd.DataFrame({symbol: frame["close"] for symbol, frame in weekly.items()})
    average = closes.apply(lambda column: ema(column, int(params["ema_weeks"])))
    momentum = closes.apply(lambda column: roc(column, int(params["roc_weeks"])))
    trend = ((closes > average) & (average > average.shift(1))).fillna(False).astype(bool)
    ranked = momentum.where(trend).rank(axis=1, ascending=False, method="first")
    chosen = ((ranked <= int(params["slots"])).fillna(False) & trend).astype(bool)
    leave = ((ranked > int(params["hold_rank"])).fillna(True) | ~trend).astype(bool)
    daily_closes = pd.DataFrame({symbol: bars_by_symbol[symbol]["close"] for symbol in weekly})
    _drawdown, near, broken = _market(daily_closes, params)
    _CACHE["market"] = _drawdown
    _CACHE["market_key"] = (
        int(params["index_days"]),
        float(params["index_entry_pct"]),
        float(params["index_exit_pct"]),
    )
    signals = {}
    for symbol, finished in weekly.items():
        daily = bars_by_symbol[symbol]
        own = finished.index.intersection(daily.index)
        entry = pd.Series(False, index=daily.index)
        exit_ = pd.Series(False, index=daily.index)
        entry.loc[own] = chosen.reindex(own)[symbol].fillna(False).to_numpy()
        exit_.loc[own] = leave.reindex(own)[symbol].fillna(True).to_numpy()
        allowed = near.reindex(daily.index).fillna(False)
        off = broken.reindex(daily.index).fillna(False)
        entry.loc[~allowed.to_numpy()] = False
        exit_.loc[off.to_numpy()] = True
        peak, deep = _trail(daily, params)
        exit_.loc[deep.to_numpy()] = True
        width = atr(daily, int(params["atr_length"]))
        signals[symbol] = _frame(daily.index, entry, exit_, width, momentum[symbol], peak, daily["close"])
    for symbol, bars in bars_by_symbol.items():
        signals.setdefault(symbol, flat_signals(bars, params))
    return signals


def _identity(bars: pd.DataFrame) -> tuple:
    close = bars["close"]
    probe = close.iloc[min(100, len(close) - 1)]
    return (bars.index[0].value, round(float(close.iloc[0]), 4), round(float(probe), 4))


def _param_key(params: dict) -> tuple:
    names = (
        "ema_weeks",
        "roc_weeks",
        "slots",
        "hold_rank",
        "index_days",
        "index_entry_pct",
        "index_exit_pct",
        "trail_days",
        "trail_pct",
        "atr_length",
    )
    return tuple(params[name] for name in names)


def _cached(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Slice the full-universe signals down to the bars that were asked for."""

    key = _param_key(params)
    if _CACHE["key"] != key:
        from engine.nse import listings, load_universe

        symbols = [item.symbol for item in listings()]
        data = load_universe(adjusted=True, symbols=symbols)
        _CACHE["key"] = key
        _CACHE["signals"] = book_signals(data, params)
        _CACHE["ids"] = {_identity(frame): symbol for symbol, frame in data.items()}
    symbol = _CACHE["ids"].get(_identity(bars))
    if symbol is None or symbol not in _CACHE["signals"]:
        return single_signals(bars, params)
    frame = _CACHE["signals"][symbol].reindex(bars.index)
    frame["entry"] = frame["entry"].fillna(False).astype(bool)
    frame["exit"] = frame["exit"].fillna(False).astype(bool)
    return frame


def _ensure_market(params: dict) -> pd.Series:
    """Equal-weight drawdown from its high. Built once and reused for every chart."""

    key = (int(params["index_days"]), float(params["index_entry_pct"]), float(params["index_exit_pct"]))
    if _CACHE.get("market_key") == key and _CACHE.get("market") is not None:
        return _CACHE["market"]
    from engine.nse import listings, load_universe

    symbols = [item.symbol for item in listings()]
    data = load_universe(adjusted=True, symbols=symbols)
    closes = pd.DataFrame({symbol: frame["close"] for symbol, frame in data.items()})
    drawdown, _near, _broken = _market(closes, params)
    _CACHE["market"] = drawdown
    _CACHE["market_key"] = key
    return drawdown


def _chart(bars: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, dict[str, str]]:
    index = bars.index
    weekly, average, momentum = _weekly(bars, params) if len(bars) else (pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float))
    peak, _deep = _trail(bars, params) if len(bars) else (pd.Series(dtype=float, index=index), None)
    width = atr(bars, int(params["atr_length"])) if len(bars) else pd.Series(dtype=float, index=index)

    def forward(series: pd.Series) -> pd.Series:
        clean = series.dropna()
        if clean.empty:
            return pd.Series(np.nan, index=index, dtype=float)
        return clean.reindex(index.union(clean.index)).sort_index().ffill().reindex(index)

    market = _ensure_market(params).reindex(index) if len(bars) >= 400 else pd.Series(np.nan, index=index, dtype=float)
    frame = pd.DataFrame(
        {
            "Weekly EMA": forward(average),
            "Trail": peak * (1.0 - float(params["trail_pct"]) / 100.0),
            "ROC": forward(momentum),
            "ATR": width,
            "Market drawdown": market,
            "Market drawdown.limit": -float(params["index_exit_pct"]),
        },
        index=index,
    )
    kinds = {
        "Weekly EMA": "overlay",
        "Trail": "overlay",
        "ROC": "oscillator",
        "ATR": "oscillator",
        "Market drawdown": "oscillator",
        "Market drawdown.limit": "oscillator",
    }
    return frame, kinds


class MomentumSleeve(Strategy):
    key = "momentum_sleeve"
    name = "Momentum sleeve"
    portfolio = True
    stages = STAGES
    cost_params = COST_PARAMS
    param_labels = PARAM_LABELS
    param_choices = CHOICES

    def param_space(self) -> dict[str, ParamSpec]:
        return {
            "ema_weeks": ParamSpec(30, 30, 40, 10, "int"),
            "roc_weeks": ParamSpec(52, None, None, None, "int"),
            "slots": ParamSpec(5, None, None, None, "int"),
            "hold_rank": ParamSpec(10, None, None, None, "int"),
            "index_days": ParamSpec(120, None, None, None, "int"),
            "index_entry_pct": ParamSpec(2, None, None, None, "float"),
            "index_exit_pct": ParamSpec(8, None, None, None, "float"),
            "trail_days": ParamSpec(80, None, None, None, "int"),
            "trail_pct": ParamSpec(16, None, None, None, "float"),
            "atr_length": ParamSpec(14, None, None, None, "int"),
            "stop_atr": ParamSpec(4.0, None, None, None, "float"),
            "partial_pct": ParamSpec(100, None, None, None, "int"),
            "entry_rank": ParamSpec(0, 0, 2, 1, "int"),
            "capital": ParamSpec(2_000_000, None, None, None, "int"),
            "max_position_inr": ParamSpec(200_000, None, None, None, "int"),
            "risk_per_atr": ParamSpec(1, None, None, None, "float"),
            "slippage_bps": ParamSpec(5, None, None, None, "float"),
            "brokerage_bps": ParamSpec(0, None, None, None, "float"),
            "brokerage_cap_inr": ParamSpec(20, None, None, None, "float"),
            "stt_buy_bps": ParamSpec(10, None, None, None, "float"),
            "stt_sell_bps": ParamSpec(10, None, None, None, "float"),
            "exchange_bps": ParamSpec(0.297, None, None, None, "float"),
            "sebi_bps": ParamSpec(0.01, None, None, None, "float"),
            "stamp_buy_bps": ParamSpec(1.5, None, None, None, "float"),
            "gst_pct": ParamSpec(18, None, None, None, "float"),
        }

    def target_positions(self, bars: pd.DataFrame, params: dict, start: int = 0) -> pd.Series:
        raise RuntimeError("Momentum sleeve trades the book. Run it from the bank.")

    def signals(self, bars: pd.DataFrame, params: dict) -> pd.DataFrame:
        resolved = resolve_params(self.param_space(), params)
        if len(bars) < 120:
            return flat_signals(bars, resolved)
        return _cached(bars, resolved)

    def indicator_frame(self, bars: pd.DataFrame, params: dict):
        return _chart(bars, resolve_params(self.param_space(), params))

    def stage_line(self, key: str, params: dict) -> str:
        return stage_line(key, params)
