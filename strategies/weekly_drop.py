"""Buy a completed week that fell 5%, hold it for the next week, stop 5%.

Filter: the three completed months before the current month are green, and
each of those closes is above the previous close. The month that contains the
bar is not one of the three, including on the session that completes it.

The buy is known at the down week's close and filled on the next open. The
sell is the open after the following completed week. The stop is 5% under the
fill and sells the whole position. Shares are floor((portfolio value / 20) /
15-day ATR), so the rupee position is that unit times price divided by that
ATR. The account starts at 20 lakh.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.book import COST_PARAMS, size_unit
from engine.indicators import atr
from engine.models import ParamSpec
from engine.strategy import Stage, Strategy, resolve_params
from strategies.medium_frequency_momentum import completed_ohlc

PARAM_LABELS = {
    "months": "Completed months",
    "drop_pct": "Weekly drop, percent",
    "atr_length": "ATR length",
    "stop_pct": "Stop below the fill, percent",
}

RANK_LABELS = {
    0: "Deepest weekly drop",
    1: "Largest drop in ATRs",
    2: "Symbol name",
}

CHOICES = {
    "entry_rank": tuple((RANK_LABELS[index], index) for index in range(3)),
}

STAGES = (
    Stage(
        "filter",
        "Filter",
        "The three completed months before this month are green, and each close is above the previous close.",
        (("Monthly candles", ("months",)),),
    ),
    Stage(
        "buy",
        "Buy",
        "The completed week closes at least 5% below the prior week's close. The fill is the next open.",
        (("Weekly drop", ("drop_pct",)),),
    ),
    Stage(
        "sell",
        "Sell",
        "The open after the following completed week sells the shares bought for that week.",
        (("One week", ()),),
    ),
    Stage(
        "stop",
        "Stop loss",
        "The stop is 5% under the fill. The whole position sells at the stop.",
        (("Protective stop", ("stop_pct", "partial_pct")),),
    ),
    Stage(
        "size",
        "Position sizing",
        "Shares are the portfolio value divided by 20, times the size factor, divided by the 15-day ATR, then cut by the rupee cap and by cash.",
        (
            ("Shares", ("atr_length", "risk_per_atr", "max_position_inr", "capital")),
            ("Several buys the same morning", ("entry_rank",)),
        ),
    ),
)


def stage_line(key: str, params: dict) -> str:
    """One sentence that states the current settings for a stage."""

    months = int(params["months"])
    drop = float(params["drop_pct"])
    if key == "filter":
        return (
            f"The {months} completed months before this month are green candles, "
            "each close above its open, and each close above the previous close."
        )
    if key == "buy":
        return (
            f"The completed week closes at least {drop:g}% below the prior completed week's close. "
            "The fill is the next open."
        )
    if key == "sell":
        return "The open after the following completed week sells the shares bought for that week."
    if key == "stop":
        return (
            f"The stop is {float(params['stop_pct']):g}% under the fill. "
            f"{int(params['partial_pct'])}% of the shares sell at the stop."
        )
    if key == "size":
        rank = {
            0: "the deepest weekly drop",
            1: "the largest drop in ATRs",
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


def _hold_forward(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Carry a completed-week value onto later sessions. Nothing is filled ahead of its close."""

    clean = series.sort_index().dropna()
    if clean.empty:
        return pd.Series(np.nan, index=index, dtype=float)
    combined = clean.reindex(index.union(clean.index)).sort_index().ffill()
    return combined.reindex(index)


def prior_months_pass(bars: pd.DataFrame, months: int) -> pd.Series:
    """True when the completed months before this month are green and their closes rise.

    Green means the month closes above its open. The month that contains the bar
    is excluded, including on the session that completes it.
    """

    allowed = pd.Series(False, index=bars.index)
    count = int(months)
    if bars.empty or count < 1:
        return allowed
    monthly = completed_ohlc(bars, "M")
    if len(monthly) < count:
        return allowed
    opened = monthly["open"].to_numpy(dtype=float)
    closed = monthly["close"].to_numpy(dtype=float)
    green = closed > opened
    passes = np.zeros(len(monthly), dtype=bool)
    for index in range(count, len(monthly)):
        window = slice(index - count, index)
        if not bool(green[window].all()):
            continue
        earlier = closed[index - count : index - 1]
        later = closed[index - count + 1 : index]
        passes[index] = bool(np.all(later > earlier))
    month_periods = monthly.index.to_period("M")
    lookup = pd.Series(passes, index=month_periods)
    bar_periods = bars.index.to_period("M")
    values = pd.Series(bar_periods, index=bars.index).map(lookup).fillna(False).to_numpy(dtype=bool, copy=True)
    earlier = closed[-count:-1]
    later = closed[-count + 1 :]
    pending = bool(green[-count:].all() and np.all(later > earlier))
    values[np.asarray(bar_periods > month_periods[-1])] = pending
    return pd.Series(values, index=bars.index)


def prepare_signals(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Entry and exit flags on the daily index. Flags are known at that close."""

    index = bars.index
    length = int(params["atr_length"])
    width = atr(bars, length) if len(bars) else pd.Series(dtype=float, index=index)
    out = pd.DataFrame(
        {
            "entry": False,
            "exit": False,
            "atr": width,
            "udr": np.nan,
            "spread_atr": np.nan,
        },
        index=index,
    )
    if bars.empty or float(params["drop_pct"]) <= 0:
        return out
    weekly = completed_ohlc(bars, "W-FRI")
    if len(weekly) < 2:
        return out
    change = weekly["close"] / weekly["close"].shift(1) - 1
    fell = change <= -(float(params["drop_pct"]) / 100.0)
    month_ok = prior_months_pass(bars, int(params["months"])).reindex(weekly.index).fillna(False).astype(bool)
    entry_week = fell.fillna(False).astype(bool) & month_ok
    exit_week = entry_week.shift(1, fill_value=False).astype(bool)
    chosen = entry_week[entry_week].index
    out.loc[chosen, "entry"] = True
    out.loc[exit_week[exit_week].index, "exit"] = True
    if len(chosen):
        drop = -change.loc[chosen]
        out.loc[chosen, "udr"] = drop.to_numpy(dtype=float)
        span = width.reindex(chosen).to_numpy(dtype=float)
        moved = drop.to_numpy(dtype=float) * weekly.loc[chosen, "close"].to_numpy(dtype=float)
        out.loc[chosen, "spread_atr"] = moved / span
    return out


def chart_frame(bars: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, dict[str, str]]:
    """The 15-day ATR, the completed week's percent change, and the month filter."""

    index = bars.index
    length = int(params["atr_length"])
    width = atr(bars, length) if len(bars) else pd.Series(dtype=float, index=index)
    weekly = completed_ohlc(bars, "W-FRI") if len(bars) else pd.DataFrame()
    if len(weekly) >= 2:
        change = (weekly["close"] / weekly["close"].shift(1) - 1.0) * 100.0
    else:
        change = pd.Series(dtype=float)
    frame = pd.DataFrame(
        {
            "ATR": width,
            "Weekly return": _hold_forward(change, index),
            "Weekly return.threshold": -float(params["drop_pct"]),
            "Month filter": prior_months_pass(bars, int(params["months"])).astype(float) if len(bars) else pd.Series(dtype=float, index=index),
        },
        index=index,
    )
    kinds = {
        "ATR": "oscillator",
        "Weekly return": "oscillator",
        "Weekly return.threshold": "oscillator",
        "Month filter": "oscillator",
    }
    return frame, kinds


class WeeklyDrop(Strategy):
    key = "weekly_drop"
    name = "Weekly drop"
    portfolio = True
    stages = STAGES
    cost_params = COST_PARAMS
    param_labels = PARAM_LABELS
    param_choices = CHOICES

    def param_space(self) -> dict[str, ParamSpec]:
        # stop_atr stays because every book strategy exposes it. This stop is a percent of the fill.
        return {
            "months": ParamSpec(3, 2, 6, 1, "int"),
            "drop_pct": ParamSpec(5, 3, 12, 1, "float"),
            "atr_length": ParamSpec(15, 5, 30, 1, "int"),
            "stop_pct": ParamSpec(5, None, None, None, "float"),
            "stop_atr": ParamSpec(0, None, None, None, "float"),
            "partial_pct": ParamSpec(100, None, None, None, "int"),
            "entry_rank": ParamSpec(0, 0, 2, 1, "int"),
            "capital": ParamSpec(2_000_000, None, None, None, "int"),
            "max_position_inr": ParamSpec(2_000_000, None, None, None, "int"),
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
        raise RuntimeError("Weekly drop trades the book. Run it from the bank.")

    def signals(self, bars: pd.DataFrame, params: dict) -> pd.DataFrame:
        return prepare_signals(bars, resolve_params(self.param_space(), params))

    def indicator_frame(self, bars: pd.DataFrame, params: dict):
        return chart_frame(bars, resolve_params(self.param_space(), params))

    def stage_line(self, key: str, params: dict) -> str:
        return stage_line(key, params)
