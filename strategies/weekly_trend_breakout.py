"""Weekly trend breakout for a Nifty 500 book.

Five indicators, and only two of them can block a buy. The completed week
must close above its EMA and above the prior Donchian high, with ADX at or
above a low floor so a dead range does not qualify. Rate of change does not
block a buy. It only decides which breakout is filled first when several names
signal on the same morning. ATR sizes the shares and places the stop.

The sell is the week that closes back under the EMA. The stop is wide, four
entry ATRs, so an ordinary week does not knock out a trend the EMA is still
holding. The account starts at 20 lakh. Each name is capped at 2 lakh, so the
book can hold about ten trends at once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.book import COST_PARAMS, size_unit
from engine.indicators import adx, atr, donchian, ema, roc
from engine.models import ParamSpec
from engine.strategy import Stage, Strategy, resolve_params
from strategies.medium_frequency_momentum import completed_ohlc

PARAM_LABELS = {
    "ema_weeks": "Weekly EMA length",
    "adx_weeks": "Weekly ADX length",
    "adx_min": "ADX minimum",
    "channel_weeks": "Donchian weeks",
    "atr_length": "ATR length",
}

RANK_LABELS = {
    0: "Strongest weekly rate of change",
    1: "Highest ADX",
    2: "Symbol name",
}

CHOICES = {
    "entry_rank": tuple((RANK_LABELS[index], index) for index in range(3)),
}

STAGES = (
    Stage(
        "filter",
        "Filter",
        "The completed week closes above its EMA, and ADX is at least a modest floor.",
        (
            ("Weekly trend", ("ema_weeks",)),
            ("Trend strength", ("adx_weeks", "adx_min")),
        ),
    ),
    Stage(
        "buy",
        "Buy",
        "The completed week closes above the highest high of the prior weeks. The fill is the next open.",
        (("Weekly breakout", ("channel_weeks",)),),
    ),
    Stage(
        "sell",
        "Sell",
        "The completed week closes back under the EMA. The fill is the next open.",
        (("Trend break", ()),),
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


def stage_line(key: str, params: dict) -> str:
    """One sentence that states the current settings for a stage."""

    weeks = int(params["ema_weeks"])
    channel = int(params["channel_weeks"])
    if key == "filter":
        return (
            f"The completed week closes above the {weeks}-week EMA, "
            f"and {int(params['adx_weeks'])}-week ADX is at least {float(params['adx_min']):g}."
        )
    if key == "buy":
        return (
            f"The completed week closes above the highest high of the prior {channel} weeks. "
            "The fill is the next open."
        )
    if key == "sell":
        return (
            f"The completed week closes below the {weeks}-week EMA. "
            "The fill is the next open."
        )
    if key == "stop":
        return (
            f"A stop is frozen {float(params['stop_atr']):g} entry ATRs under the fill. "
            f"{int(params['partial_pct'])}% of the shares sell there."
        )
    if key == "size":
        rank = {
            0: "the strongest weekly rate of change",
            1: "the highest ADX",
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


def _weekly_views(bars: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Completed-week EMA, prior Donchian high, ADX, and rate of change."""

    weekly = completed_ohlc(bars, "W-FRI")
    if weekly.empty:
        empty = pd.Series(dtype=float)
        return weekly, empty, empty, empty, empty
    average = ema(weekly["close"], int(params["ema_weeks"]))
    high = donchian(weekly, int(params["channel_weeks"]))["upper"]
    strength = adx(weekly, int(params["adx_weeks"]))["adx"]
    momentum = roc(weekly["close"], int(params["channel_weeks"]))
    return weekly, average, high, strength, momentum


def prepare_signals(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Entry and exit flags on the daily index. Flags are known at that week's close."""

    index = bars.index
    width = atr(bars, int(params["atr_length"])) if len(bars) else pd.Series(dtype=float, index=index)
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
    if bars.empty:
        return out
    weekly, average, high, strength, momentum = _weekly_views(bars, params)
    if weekly.empty:
        return out
    close = weekly["close"]
    entry_week = (close > average) & (close > high) & (strength >= float(params["adx_min"]))
    exit_week = close < average
    entry_week = entry_week.fillna(False)
    exit_week = exit_week.fillna(False)
    entries = entry_week.index[entry_week.to_numpy(dtype=bool)].intersection(index)
    exits = exit_week.index[exit_week.to_numpy(dtype=bool)].intersection(index)
    out.loc[entries, "entry"] = True
    out.loc[exits, "exit"] = True
    if len(entries):
        out.loc[entries, "udr"] = momentum.reindex(entries).to_numpy(dtype=float)
        out.loc[entries, "spread_atr"] = strength.reindex(entries).to_numpy(dtype=float)
    return out


def chart_frame(bars: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, dict[str, str]]:
    """The weekly EMA and Donchian high on the candles, with ADX, rate of change, and ATR."""

    index = bars.index
    weekly, average, high, strength, momentum = _weekly_views(bars, params) if len(bars) else (
        pd.DataFrame(),
        pd.Series(dtype=float),
        pd.Series(dtype=float),
        pd.Series(dtype=float),
        pd.Series(dtype=float),
    )
    frame = pd.DataFrame(
        {
            "Weekly EMA": _hold_forward(average, index),
            "Donchian high": _hold_forward(high, index),
            "ADX": _hold_forward(strength, index),
            "ADX.minimum": float(params["adx_min"]),
            "ROC": _hold_forward(momentum, index),
            "ATR": atr(bars, int(params["atr_length"])) if len(bars) else pd.Series(dtype=float, index=index),
        },
        index=index,
    )
    kinds = {
        "Weekly EMA": "overlay",
        "Donchian high": "overlay",
        "ADX": "oscillator",
        "ADX.minimum": "oscillator",
        "ROC": "oscillator",
        "ATR": "oscillator",
    }
    return frame, kinds


class WeeklyTrendBreakout(Strategy):
    key = "weekly_trend_breakout"
    name = "Weekly trend breakout"
    portfolio = True
    stages = STAGES
    cost_params = COST_PARAMS
    param_labels = PARAM_LABELS
    param_choices = CHOICES

    def param_space(self) -> dict[str, ParamSpec]:
        return {
            "ema_weeks": ParamSpec(20, 8, 40, 4, "int"),
            "adx_weeks": ParamSpec(14, 8, 20, 2, "int"),
            "adx_min": ParamSpec(15, 10, 25, 5, "float"),
            "channel_weeks": ParamSpec(14, 6, 26, 4, "int"),
            "atr_length": ParamSpec(14, 10, 20, 2, "int"),
            "stop_atr": ParamSpec(4.0, 0.5, 5.0, 0.5, "float"),
            "partial_pct": ParamSpec(100, 0, 100, 10, "int"),
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
        raise RuntimeError("Weekly trend breakout trades the book. Run it from the bank.")

    def signals(self, bars: pd.DataFrame, params: dict) -> pd.DataFrame:
        return prepare_signals(bars, resolve_params(self.param_space(), params))

    def indicator_frame(self, bars: pd.DataFrame, params: dict):
        return chart_frame(bars, resolve_params(self.param_space(), params))

    def stage_line(self, key: str, params: dict) -> str:
        return stage_line(key, params)
