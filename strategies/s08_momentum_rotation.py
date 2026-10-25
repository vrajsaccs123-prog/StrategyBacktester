"""Cross-sectional momentum rotation on the Nifty 500.

On the last session of each month, or every second week, rank names by 12-1
momentum. The candidate pool is the top 40% by that momentum. The book buys
the lowest-volatility slice of that pool, sized to `top_pct` of the names with
a score. Weights are inverse volatility, capped at 8% of equity, and the rest
stays in cash. A name that leaves the top 40% is sold at the next rebalance.
If the equal-weight index closes under its 200-day average, the rebalance
goes to cash. The price file has no separate index, so the index is the
equal-weight book.

There is no other trading during the month except a 20% trailing stop, checked
on the close and filled on the next open. The frozen stop is also 20% under
the fill. The signal ATR column is the close and the size column is the weight
times 20, so shares are floor((portfolio value / 20) × weight × 20 / close).
The first 300 bars of a name are not bought.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.book import COST_PARAMS
from engine.indicators import sma
from engine.models import ParamSpec
from engine.strategy import Stage, Strategy, resolve_params
from strategies._desk import WARMUP, book_space, cap_sentence, choice, flat, pack, rank_line, rising_max

POOL = 0.40
SKIP = 21
WEIGHT_CAP = 0.08
REBALANCE_LABELS = ("the last session of the month", "every second week")

PARAM_LABELS = {
    "top_pct": "Share of names held",
    "lookback": "Momentum lookback, sessions",
    "rebalance": "Rebalance (0 = month-end, 1 = every second week)",
    "trail_pct": "Trailing stop, percent",
    "atr_length": "ATR length",
    "stop_pct": "Stop under the fill, percent",
}

RANK_LABELS = {
    0: "Strongest 12-1 momentum",
    1: "Lowest volatility",
    2: "Symbol name",
}

STAGES = (
    Stage(
        "filter",
        "Filter",
        "The equal-weight index is at or above its 200-day average. The first 300 bars are not bought.",
        (("Market", ("lookback",)),),
    ),
    Stage(
        "buy",
        "Buy",
        "At the rebalance, buy the lowest-volatility names inside the top 40% by 12-1 momentum. The fill is the next open.",
        (("Rotation", ("top_pct", "rebalance")),),
    ),
    Stage(
        "sell",
        "Sell",
        "Sell a name that leaves the top 40%, the whole book when the index is under its average, or a 20% trail. The fill is the next open.",
        (("Exit", ("trail_pct",)),),
    ),
    Stage(
        "stop",
        "Stop loss",
        "A stop 20% under the fill sells the whole position.",
        (("Protective stop", ("stop_pct", "partial_pct")),),
    ),
    Stage(
        "size",
        "Position sizing",
        "Inverse-volatility weights, capped at 8% of equity. The remainder stays in cash.",
        (
            ("Shares", ("atr_length", "risk_per_atr", "max_position_pct", "max_position_inr", "capital")),
            ("Several buys the same morning", ("entry_rank",)),
        ),
    ),
)

_CACHE: dict = {"key": None, "signals": {}, "ids": {}, "ratio": None, "frames": None}


def stage_line(key: str, params: dict) -> str:
    schedule = REBALANCE_LABELS[int(params["rebalance"])] if int(params["rebalance"]) in (0, 1) else REBALANCE_LABELS[0]
    if key == "filter":
        return (
            f"The equal-weight index is at or above its 200-day average. "
            f"Momentum skips {SKIP} sessions and looks back {int(params['lookback'])}. "
            f"The first {WARMUP} bars of a name are not bought."
        )
    if key == "buy":
        return (
            f"On {schedule}, keep the top {POOL:.0%} by 12-1 momentum, then hold the lowest-volatility "
            f"{float(params['top_pct']):.0%} of the scored names. The fill is the next open."
        )
    if key == "sell":
        return (
            f"A name that leaves the top {POOL:.0%} is sold at the next rebalance. "
            f"The book goes to cash when the index is under its 200-day average. "
            f"A close {float(params['trail_pct']):g}% under the highest close since the name re-entered the pool "
            "sells at the next open."
        )
    if key == "stop":
        return (
            f"The stop is {float(params['stop_pct']):g}% under the fill. "
            f"{int(params['partial_pct'])}% of the shares sell there."
        )
    if key == "size":
        return (
            "Weights are inverse volatility, capped at 8% of equity, with the remainder in cash. "
            f"The signal ATR is the close and the size column is the weight times 20, "
            f"with size factor {float(params['risk_per_atr']):g}. "
            f"{cap_sentence(params)} "
            f"Same-morning buys are filled by {rank_line(params, RANK_LABELS)}."
        )
    raise KeyError(key)


def rebalance_dates(index: pd.DatetimeIndex, code: int) -> pd.DatetimeIndex:
    """Last session of each month, or the last session of every second week."""

    dates = pd.DatetimeIndex(pd.to_datetime(index)).sort_values().unique()
    if len(dates) == 0:
        return dates
    series = pd.Series(1, index=dates)
    if int(code) == 1:
        ends = series.groupby(dates.to_period("W-FRI")).tail(1)
        ends = ends.iloc[1::2]
    else:
        ends = series.groupby(dates.to_period("M")).tail(1)
    return pd.DatetimeIndex(ends.index)


def _equal_weight(closes: pd.DataFrame) -> pd.Series:
    return (1.0 + closes.pct_change().mean(axis=1).fillna(0.0)).cumprod()


def prepare_book(data: dict[str, pd.DataFrame], params: dict) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """Signals for every name. Rank and the index use only information known that session."""

    usable = {symbol: bars for symbol, bars in data.items() if bars is not None and not bars.empty}
    if not usable:
        return {}, pd.Series(dtype=float)
    closes = pd.DataFrame({symbol: bars["close"].astype(float) for symbol, bars in usable.items()}).sort_index()
    lookback = int(params["lookback"])
    momentum = closes.shift(SKIP) / closes.shift(lookback).replace(0, np.nan) - 1.0
    volatility = closes.pct_change().rolling(lookback, min_periods=lookback).std(ddof=1)
    valid = momentum.notna() & volatility.notna() & (volatility > 0)
    counted = valid.sum(axis=1).astype(float)
    pool_n = np.ceil(counted * POOL).clip(lower=1.0)
    momentum_rank = momentum.where(valid).rank(axis=1, ascending=False, method="first")
    in_pool = momentum_rank.le(pool_n, axis=0) & valid
    target = np.ceil(counted * float(params["top_pct"])).clip(lower=1.0)
    volatility_rank = volatility.where(in_pool).rank(axis=1, ascending=True, method="first")
    pool_count = in_pool.sum(axis=1).astype(float)
    take = pd.Series(np.maximum(np.minimum(target.to_numpy(), pool_count.to_numpy()), 1.0), index=closes.index)
    selected = volatility_rank.le(take, axis=0) & in_pool
    inverse = (1.0 / volatility).where(selected)
    weights = inverse.div(inverse.sum(axis=1).replace(0, np.nan), axis=0).clip(upper=WEIGHT_CAP)

    index = _equal_weight(closes)
    average = sma(index, 200)
    ratio = index / average.replace(0, np.nan)
    risk_on = (index >= average).fillna(False).to_numpy(dtype=bool)
    marks = rebalance_dates(closes.index, int(params["rebalance"]))
    rebalance = closes.index.isin(marks)

    held = np.array(in_pool.to_numpy(dtype=float), copy=True)
    held[~rebalance, :] = np.nan
    regime = pd.DataFrame(held, index=closes.index, columns=closes.columns).ffill().fillna(0.0).to_numpy(dtype=float) > 0.5

    trail = float(params["trail_pct"]) / 100.0
    selected_v = selected.to_numpy(dtype=bool)
    momentum_v = momentum.to_numpy(dtype=float)
    volatility_v = volatility.to_numpy(dtype=float)
    weight_v = weights.to_numpy(dtype=float)
    close_v = closes.to_numpy(dtype=float)
    columns = {symbol: position for position, symbol in enumerate(closes.columns)}
    signals: dict[str, pd.DataFrame] = {}
    for symbol, bars in usable.items():
        column = columns[symbol]
        loc = closes.index.get_indexer(bars.index)
        if np.any(loc < 0):
            signals[symbol] = flat(bars, bars["close"].to_numpy(dtype=float))
            continue
        price = close_v[loc, column]
        member = regime[loc, column]
        peak = rising_max(price, member)
        giveback = np.isfinite(peak) & (price < peak * (1.0 - trail))
        scheduled = rebalance[loc]
        allowed = risk_on[loc]
        exit_ = (~member) | giveback | (scheduled & ~allowed)
        weight = weight_v[loc, column]
        entry = selected_v[loc, column] & scheduled & allowed & ~giveback & np.isfinite(weight) & (weight > 0) & ~exit_
        signals[symbol] = pack(
            bars.index,
            entry,
            exit_,
            price,
            momentum_v[loc, column],
            -volatility_v[loc, column],
            size=weight * 20.0,
        )
    for symbol, bars in data.items():
        signals.setdefault(symbol, flat(bars))
    return signals, ratio


def _param_key(params: dict) -> tuple:
    return (
        round(float(params["top_pct"]), 8),
        int(params["lookback"]),
        int(params["rebalance"]),
        round(float(params["trail_pct"]), 8),
    )


def _identity(bars: pd.DataFrame) -> tuple:
    close = bars["close"].astype(float)
    probe = min(20, len(close) - 1)
    sample = round(float(close.iloc[: probe + 1].sum()), 2)
    return (
        int(pd.Timestamp(bars.index[0]).value),
        round(float(close.iloc[0]), 4),
        probe,
        round(float(close.iloc[probe]), 4),
        sample,
    )


def _universe() -> dict[str, pd.DataFrame]:
    if _CACHE["frames"] is None:
        from engine.nse import listings, load_universe

        symbols = [item.symbol for item in listings()]
        _CACHE["frames"] = load_universe(adjusted=True, symbols=symbols)
    return _CACHE["frames"]


def _cached(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    key = _param_key(params)
    if _CACHE["key"] != key:
        signals, ratio = prepare_book(_universe(), params)
        _CACHE["key"] = key
        _CACHE["signals"] = signals
        _CACHE["ratio"] = ratio
        _CACHE["ids"] = {_identity(frame): symbol for symbol, frame in _universe().items()}
    symbol = _CACHE["ids"].get(_identity(bars))
    if symbol is None or symbol not in _CACHE["signals"]:
        return flat(bars, bars["close"].to_numpy(dtype=float) if len(bars) else None)
    frame = _CACHE["signals"][symbol].reindex(bars.index)
    frame["entry"] = frame["entry"].fillna(False).astype(bool)
    frame["exit"] = frame["exit"].fillna(False).astype(bool)
    return frame


def _ratio(params: dict) -> pd.Series:
    key = _param_key(params)
    if _CACHE["key"] == key and _CACHE["ratio"] is not None:
        return _CACHE["ratio"]
    _closes = pd.DataFrame({symbol: frame["close"].astype(float) for symbol, frame in _universe().items()})
    index = _equal_weight(_closes)
    average = sma(index, 200)
    return index / average.replace(0, np.nan)


def chart_frame(bars: pd.DataFrame, params: dict):
    lookback = int(params["lookback"])
    if bars.empty:
        frame = pd.DataFrame(columns=["Momentum", "Volatility", "Index / SMA200"], index=bars.index)
    else:
        close = bars["close"].astype(float)
        momentum = close.shift(SKIP) / close.shift(lookback).replace(0, np.nan) - 1.0
        volatility = close.pct_change().rolling(lookback, min_periods=lookback).std(ddof=1)
        ratio = pd.Series(np.nan, index=bars.index, dtype=float)
        if len(bars) >= WARMUP:
            ratio = _ratio(params).reindex(bars.index)
        frame = pd.DataFrame(
            {"Momentum": momentum, "Volatility": volatility, "Index / SMA200": ratio},
            index=bars.index,
        )
    kinds = {"Momentum": "oscillator", "Volatility": "oscillator", "Index / SMA200": "oscillator"}
    return frame, kinds


class MomentumRotation(Strategy):
    key = "s08_momentum_rotation"
    name = "Cross-Sectional Momentum Rotation"
    portfolio = True
    stages = STAGES
    cost_params = COST_PARAMS
    param_labels = PARAM_LABELS
    param_choices = {"entry_rank": tuple((RANK_LABELS[index], index) for index in range(3))}

    def param_space(self) -> dict[str, ParamSpec]:
        return book_space(
            {
                "top_pct": ParamSpec(0.2, 0.1, 0.3, 0.1, "float"),
                "lookback": ParamSpec(252, 126, 252, 126, "int"),
                "rebalance": choice(0, 2),
                "trail_pct": ParamSpec(20, None, None, None, "float"),
            },
            risk_per_atr=1.0,
            stop_atr=1.0,
            stop_pct=20.0,
            max_position_pct=8.0,
        )

    def target_positions(self, bars: pd.DataFrame, params: dict, start: int = 0) -> pd.Series:
        raise RuntimeError("Momentum rotation trades the book. Run it from the bank.")

    def signals(self, bars: pd.DataFrame, params: dict) -> pd.DataFrame:
        resolved = resolve_params(self.param_space(), params)
        if len(bars) < WARMUP:
            return flat(bars, bars["close"].to_numpy(dtype=float) if len(bars) else None)
        return _cached(bars, resolved)

    def indicator_frame(self, bars: pd.DataFrame, params: dict):
        return chart_frame(bars, resolve_params(self.param_space(), params))

    def stage_line(self, key: str, params: dict) -> str:
        return stage_line(key, params)
