"""Adaptive Trend Core. Trend following on the Nifty 500.

Buy when the fast EMA is above the slow EMA, ADX(14) is above 20, and the
close is at or above the chandelier. The chandelier is the 22-day high minus
the ATR multiple. It only rises while the fast EMA stays above the slow EMA,
and it resets when that trend ends. Sell when the close is under that line or
the fast EMA crosses back under the slow EMA. The fill is the next open.

The signal ATR is the multiple times the 14-day ATR. The size factor is 0.2,
so shares risk 1% of equity over that distance. The stop is the same distance
under the fill. The first 300 bars are not bought. One position is capped at
25% of equity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.book import COST_PARAMS
from engine.indicators import adx, atr, ema
from engine.models import ParamSpec
from engine.strategy import Stage, Strategy, resolve_params
from strategies._desk import WARMUP, book_space, cap_sentence, choice, pack, pick, rank_line, rising_max

ATR_MULTS = (2.5, 3.0, 4.0)
CHANDELIER = 22

PARAM_LABELS = {
    "fast": "Fast EMA",
    "slow": "Slow EMA",
    "atr_mult": "ATR multiple (0 = 2.5, 1 = 3, 2 = 4)",
    "adx_min": "ADX minimum",
    "atr_length": "ATR length",
}

RANK_LABELS = {
    0: "Strongest ADX",
    1: "Widest EMA gap per ATR",
    2: "Symbol name",
}

STAGES = (
    Stage(
        "filter",
        "Filter",
        "ADX(14) is above 20, and the first 300 bars of the name are not bought.",
        (("Trend strength", ("adx_min",)),),
    ),
    Stage(
        "buy",
        "Buy",
        "The fast EMA is above the slow EMA and the close is at or above the chandelier. The fill is the next open.",
        (("Trend", ("fast", "slow")),),
    ),
    Stage(
        "sell",
        "Sell",
        "The close is under the chandelier, or the fast EMA is under the slow EMA. The fill is the next open.",
        (("Chandelier", ("atr_mult",)),),
    ),
    Stage(
        "stop",
        "Stop loss",
        "The stop is the ATR multiple under the fill and sells the whole position.",
        (("Protective stop", ("stop_atr", "partial_pct")),),
    ),
    Stage(
        "size",
        "Position sizing",
        "Shares risk 1% of equity over the chandelier distance, capped at 25% of equity.",
        (
            ("Shares", ("atr_length", "risk_per_atr", "max_position_pct", "max_position_inr", "capital")),
            ("Several buys the same morning", ("entry_rank",)),
        ),
    ),
)


def _multiple(params: dict) -> float:
    return float(pick(params["atr_mult"], ATR_MULTS))


def stage_line(key: str, params: dict) -> str:
    fast = int(params["fast"])
    slow = int(params["slow"])
    mult = _multiple(params)
    if key == "filter":
        return (
            f"ADX(14) is above {float(params['adx_min']):g}. "
            f"The first {WARMUP} bars of a name are not bought."
        )
    if key == "buy":
        return (
            f"The {fast}-day EMA is above the {slow}-day EMA, ADX is above the filter, "
            f"and the close is at or above the chandelier. The fill is the next open."
        )
    if key == "sell":
        return (
            f"The close is under the {CHANDELIER}-day high minus {mult:g} ATRs, "
            f"or the {fast}-day EMA is under the {slow}-day EMA. "
            "That chandelier only rises while the fast EMA stays above the slow EMA. "
            "The fill is the next open."
        )
    if key == "stop":
        return (
            f"The stop is {mult:g} times the {int(params['atr_length'])}-day ATR under the fill. "
            f"The signal ATR is that distance and the file stop multiple is {float(params['stop_atr']):g}. "
            f"{int(params['partial_pct'])}% of the shares sell there."
        )
    if key == "size":
        return (
            f"Shares are floor((portfolio value / 20) × {float(params['risk_per_atr']):g} "
            f"/ ({mult:g} × {int(params['atr_length'])}-day ATR)), which is 1% of equity over the stop. "
            f"{cap_sentence(params)} "
            f"Same-morning buys are filled by {rank_line(params, RANK_LABELS)}."
        )
    raise KeyError(key)


def _lines(bars: pd.DataFrame, params: dict):
    fast_n = int(params["fast"])
    slow_n = int(params["slow"])
    mult = _multiple(params)
    width = atr(bars, int(params["atr_length"]))
    fast = ema(bars["close"], fast_n)
    slow = ema(bars["close"], slow_n)
    strength = adx(bars, int(params["atr_length"]))["adx"]
    high = bars["high"].rolling(CHANDELIER, min_periods=CHANDELIER).max()
    raw = high - mult * width
    trend = np.array(((fast > slow) & (fast_n < slow_n)).fillna(False).to_numpy(dtype=bool), copy=True)
    if len(trend) > WARMUP:
        trend[:WARMUP] = False
    else:
        trend[:] = False
    ratchet = rising_max(raw.to_numpy(dtype=float), trend)
    return fast, slow, width, strength, raw, ratchet, mult


def prepare_signals(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    index = bars.index
    if bars.empty:
        return pack(index, [], [], [], [], [])
    fast, slow, width, strength, _raw, ratchet, mult = _lines(bars, params)
    close = bars["close"].to_numpy(dtype=float)
    adx_min = float(params["adx_min"])
    trend = ((fast > slow) & (int(params["fast"]) < int(params["slow"]))).fillna(False).to_numpy(dtype=bool)
    ema_exit = ((fast < slow) & (int(params["fast"]) < int(params["slow"]))).fillna(False).to_numpy(dtype=bool)
    under = np.isfinite(ratchet) & (close < ratchet)
    exit_ = ema_exit | under
    strong = strength.to_numpy(dtype=float) > adx_min
    entry = trend & strong & np.isfinite(ratchet) & (close >= ratchet) & ~exit_
    distance = (mult * width).to_numpy(dtype=float)
    gap = ((fast - slow) / width.replace(0, np.nan)).to_numpy(dtype=float)
    return pack(index, entry, exit_, distance, strength.to_numpy(dtype=float), gap, stop_pad=distance)


def chart_frame(bars: pd.DataFrame, params: dict):
    fast_n = int(params["fast"])
    slow_n = int(params["slow"])
    mult = _multiple(params)
    fast_name = f"EMA {fast_n}"
    slow_name = f"EMA {slow_n}"
    if fast_name == slow_name:
        fast_name = f"Fast EMA {fast_n}"
        slow_name = f"Slow EMA {slow_n}"
    stop_name = f"Chandelier {mult:g}"
    if bars.empty:
        frame = pd.DataFrame(columns=[fast_name, slow_name, stop_name, "ADX", "ATR"], index=bars.index)
    else:
        fast, slow, width, strength, _raw, ratchet, _mult = _lines(bars, params)
        frame = pd.DataFrame(
            {fast_name: fast, slow_name: slow, stop_name: ratchet, "ADX": strength, "ATR": width},
            index=bars.index,
        )
    kinds = {fast_name: "overlay", slow_name: "overlay", stop_name: "overlay", "ADX": "oscillator", "ATR": "oscillator"}
    return frame, kinds


class TrendCore(Strategy):
    key = "s01_trend_core"
    name = "Adaptive Trend Core"
    portfolio = True
    stages = STAGES
    cost_params = COST_PARAMS
    param_labels = PARAM_LABELS
    param_choices = {"entry_rank": tuple((RANK_LABELS[index], index) for index in range(3))}

    def param_space(self) -> dict[str, ParamSpec]:
        return book_space(
            {
                "fast": ParamSpec(20, 10, 30, 10, "int"),
                "slow": ParamSpec(50, 50, 100, 50, "int"),
                "atr_mult": choice(1, len(ATR_MULTS)),
                "adx_min": ParamSpec(20, None, None, None, "float"),
            },
            risk_per_atr=0.2,
            stop_atr=1.0,
            max_position_pct=25.0,
        )

    def target_positions(self, bars: pd.DataFrame, params: dict, start: int = 0) -> pd.Series:
        raise RuntimeError("Adaptive Trend Core trades the book. Run it from the bank.")

    def signals(self, bars: pd.DataFrame, params: dict) -> pd.DataFrame:
        return prepare_signals(bars, resolve_params(self.param_space(), params))

    def indicator_frame(self, bars: pd.DataFrame, params: dict):
        return chart_frame(bars, resolve_params(self.param_space(), params))

    def stage_line(self, key: str, params: dict) -> str:
        return stage_line(key, params)
