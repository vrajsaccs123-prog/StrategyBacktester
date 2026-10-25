"""Daily 10/30 EMA crossover, only in names near their 90-day high.

Buy when the fast EMA crosses above the slow EMA, the 14-day RSI is above 50,
and the close is within 20% of the highest high of the last 90 trading days.
Sell when the fast EMA crosses back below the slow EMA.
The stop sells every share 2 ATRs under the fill.

Shares are floor((portfolio value / 20) / 10-day ATR), so the rupee position
is that unit times price divided by that ATR. The account starts at 20 lakh.
"""

from __future__ import annotations

import pandas as pd

from engine.book import COST_PARAMS, size_unit
from engine.indicators import atr, ema, rsi
from engine.models import ParamSpec
from engine.strategy import Stage, Strategy, resolve_params

PARAM_LABELS = {
    "fast": "Fast EMA",
    "slow": "Slow EMA",
    "atr_length": "ATR length",
    "rsi_length": "RSI length",
    "rsi_buy": "RSI above, for a buy",
    "high_days": "Trading days to the high",
    "max_drop_pct": "Max drop from that high, percent",
}

RANK_LABELS = {
    0: "Widest EMA gap",
    1: "Widest EMA gap per ATR",
    2: "Symbol name",
}

CHOICES = {
    "entry_rank": tuple((RANK_LABELS[index], index) for index in range(3)),
}

STAGES = (
    Stage(
        "filter",
        "Filter",
        "The close is within 20% of the highest high in the last 90 trading days.",
        (("Near the high", ("high_days", "max_drop_pct")),),
    ),
    Stage(
        "buy",
        "Buy",
        "The fast EMA crosses above the slow EMA while RSI is above 50. The fill is the next open.",
        (("Daily cross", ("fast", "slow", "rsi_length", "rsi_buy")),),
    ),
    Stage(
        "sell",
        "Sell",
        "The fast EMA crosses below the slow EMA. The fill is the next open.",
        (("Reversed cross", ()),),
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
        "Shares are the portfolio value divided by 20, times the size factor, divided by the 10-day ATR, then cut by the rupee cap and by cash.",
        (
            ("Shares", ("atr_length", "risk_per_atr", "max_position_inr", "capital")),
            ("Several buys the same morning", ("entry_rank",)),
        ),
    ),
)


def stage_line(key: str, params: dict) -> str:
    """One sentence that states the current settings for a stage."""

    fast = int(params["fast"])
    slow = int(params["slow"])
    if key == "filter":
        return (
            f"The close is within {float(params['max_drop_pct']):g}% of the highest high "
            f"in the last {int(params['high_days'])} trading days."
        )
    if key == "buy":
        return (
            f"The {fast}-day EMA crosses above the {slow}-day EMA, "
            f"and the {int(params['rsi_length'])}-day RSI is above {float(params['rsi_buy']):g}. "
            "The fill is the next open."
        )
    if key == "sell":
        return (
            f"The {fast}-day EMA crosses below the {slow}-day EMA. "
            "The fill is the next open."
        )
    if key == "stop":
        return (
            f"A stop is frozen {float(params['stop_atr']):g} entry ATRs under the fill. "
            f"{int(params['partial_pct'])}% of the shares sell there."
        )
    if key == "size":
        rank = {
            0: "the widest EMA gap",
            1: "the widest EMA gap per ATR",
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


def within_recent_high(bars: pd.DataFrame, days: int, max_drop_pct: float) -> pd.Series:
    """True when the close has not fallen more than max_drop_pct from the highest high.

    The high is the highest high of the last `days` trading sessions, including
    the signal day. Fewer sessions than that do not pass. A drop of exactly
    max_drop_pct still passes.
    """

    allowed = pd.Series(False, index=bars.index)
    if bars.empty or int(days) < 1:
        return allowed
    peak = bars["high"].rolling(int(days), min_periods=int(days)).max()
    limit = peak * (1 - float(max_drop_pct) / 100)
    return (bars["close"] >= limit) & peak.gt(0)


def prepare_signals(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Entry and exit flags on the daily index. Flags are known at that close."""

    index = bars.index
    width = atr(bars, int(params["atr_length"])) if len(bars) else pd.Series(dtype=float, index=index)
    fast_n = int(params["fast"])
    slow_n = int(params["slow"])
    fast = ema(bars["close"], fast_n) if len(bars) else pd.Series(dtype=float, index=index)
    slow = ema(bars["close"], slow_n) if len(bars) else pd.Series(dtype=float, index=index)
    price = bars["close"] if len(bars) else pd.Series(dtype=float, index=index)
    out = pd.DataFrame(
        {
            "entry": False,
            "exit": False,
            "atr": width,
            "udr": (fast - slow) / price.where(price > 0),
            "spread_atr": (fast - slow) / width.where(width > 0),
        },
        index=index,
    )
    if bars.empty or fast_n >= slow_n:
        return out
    strength = rsi(bars["close"], int(params["rsi_length"]))
    crossed = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    crossed_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    strong = strength > float(params["rsi_buy"])
    near_high = within_recent_high(bars, int(params["high_days"]), float(params["max_drop_pct"]))
    out["entry"] = (crossed & near_high & strong).fillna(False)
    out["exit"] = crossed_down.fillna(False)
    return out


def chart_frame(bars: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, dict[str, str]]:
    """Daily EMAs and the lookback high on the candles, with RSI and the ATR."""

    fast_n = int(params["fast"])
    slow_n = int(params["slow"])
    fast_name = f"EMA {fast_n}"
    slow_name = f"EMA {slow_n}"
    if fast_name == slow_name:
        fast_name = f"Fast EMA {fast_n}"
        slow_name = f"Slow EMA {slow_n}"
    high_name = f"{int(params['high_days'])}-day high"
    empty = pd.Series(dtype=float)
    window = int(params["high_days"])
    peak = bars["high"].rolling(window, min_periods=window).max() if len(bars) else empty
    frame = pd.DataFrame(
        {
            fast_name: ema(bars["close"], fast_n) if len(bars) else empty,
            slow_name: ema(bars["close"], slow_n) if len(bars) else empty,
            high_name: peak,
            "RSI": rsi(bars["close"], int(params["rsi_length"])) if len(bars) else empty,
            "ATR": atr(bars, int(params["atr_length"])) if len(bars) else empty,
        },
        index=bars.index,
    )
    kinds = {
        fast_name: "overlay",
        slow_name: "overlay",
        high_name: "overlay",
        "RSI": "oscillator",
        "ATR": "oscillator",
    }
    return frame, kinds


class EmaCrossover(Strategy):
    key = "ema_crossover"
    name = "EMA Crossover"
    portfolio = True
    stages = STAGES
    cost_params = COST_PARAMS
    param_labels = PARAM_LABELS
    param_choices = CHOICES

    def param_space(self) -> dict[str, ParamSpec]:
        return {
            "fast": ParamSpec(10, 5, 30, 1, "int"),
            "slow": ParamSpec(30, 20, 80, 5, "int"),
            "rsi_length": ParamSpec(14, 2, 30, 1, "int"),
            "rsi_buy": ParamSpec(50, 40, 70, 1, "float"),
            "high_days": ParamSpec(90, 20, 250, 5, "int"),
            "max_drop_pct": ParamSpec(20, 5, 50, 5, "float"),
            "atr_length": ParamSpec(10, 5, 30, 1, "int"),
            "stop_atr": ParamSpec(2.0, 0.5, 5.0, 0.5, "float"),
            "partial_pct": ParamSpec(100, 0, 100, 10, "int"),
            "entry_rank": ParamSpec(2, 0, 2, 1, "int"),
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
        raise RuntimeError("EMA crossover trades the book. Run it from the bank.")

    def signals(self, bars: pd.DataFrame, params: dict) -> pd.DataFrame:
        return prepare_signals(bars, resolve_params(self.param_space(), params))

    def indicator_frame(self, bars: pd.DataFrame, params: dict):
        return chart_frame(bars, resolve_params(self.param_space(), params))

    def stage_line(self, key: str, params: dict) -> str:
        return stage_line(key, params)
