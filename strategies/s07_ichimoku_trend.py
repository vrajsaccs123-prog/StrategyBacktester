"""Ichimoku trend rider. Slow trend following, long only.

Buy when the close is above the cloud and Tenkan crosses above Kijun, with
ADX above its minimum. The cloud at this bar was computed 26 bars earlier.
Chikou is not used. Sell when the close is under Kijun for two bars, or the
close falls back into the cloud. The fill is the next open.

The stop is Kijun minus one ATR on the signal bar. The signal ATR is the close
minus that stop, and the size factor is 0.2, so the position risks 1% of equity
over that distance. At most 8 positions are open. The first 300 bars are not
bought. One position is also capped at 25% of equity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.book import COST_PARAMS
from engine.indicators import adx, atr, compute_indicator
from engine.models import ParamSpec
from engine.strategy import Stage, Strategy, resolve_params
from strategies._desk import WARMUP, book_space, cap_sentence, choice, pack, pick, rank_line

ADX_LEVELS = (15.0, 18.0, 20.0)

PARAM_LABELS = {
    "tenkan": "Tenkan length",
    "kijun": "Kijun length",
    "adx_min": "ADX minimum (0 = 15, 1 = 18, 2 = 20)",
    "atr_length": "ATR length",
}

RANK_LABELS = {
    0: "Strongest ADX",
    1: "Furthest above the cloud",
    2: "Symbol name",
}

STAGES = (
    Stage(
        "filter",
        "Filter",
        "The close is above the cloud and ADX is above its minimum. The first 300 bars are not bought.",
        (("Cloud", ("adx_min",)),),
    ),
    Stage(
        "buy",
        "Buy",
        "Tenkan crosses above Kijun while the close is above the cloud. The fill is the next open.",
        (("Cross", ("tenkan", "kijun")),),
    ),
    Stage(
        "sell",
        "Sell",
        "The close is under Kijun for two bars, or it falls into the cloud. The fill is the next open.",
        (("Giveback", ()),),
    ),
    Stage(
        "stop",
        "Stop loss",
        "The stop is Kijun minus one ATR on the signal bar.",
        (("Protective stop", ("stop_atr", "partial_pct")),),
    ),
    Stage(
        "size",
        "Position sizing",
        "Each position risks 1% of equity over the distance to Kijun minus one ATR. At most 8 are open.",
        (
            ("Shares", ("atr_length", "risk_per_atr", "max_position_pct", "max_positions", "max_position_inr", "capital")),
            ("Several buys the same morning", ("entry_rank",)),
        ),
    ),
)


def _adx_min(params: dict) -> float:
    return float(pick(params["adx_min"], ADX_LEVELS))


def stage_line(key: str, params: dict) -> str:
    tenkan = int(params["tenkan"])
    kijun = int(params["kijun"])
    if key == "filter":
        return (
            f"The close is above the cloud, and ADX(14) is above {_adx_min(params):g}. "
            f"The first {WARMUP} bars of a name are not bought."
        )
    if key == "buy":
        return (
            f"Tenkan({tenkan}) crosses above Kijun({kijun}) while the close is above the cloud. "
            "Senkou spans are the values plotted on this bar. Chikou is not used. The fill is the next open."
        )
    if key == "sell":
        return (
            f"The close is under Kijun({kijun}) for two bars, or the close is back inside the cloud. "
            "The fill is the next open."
        )
    if key == "stop":
        return (
            f"The stop is Kijun minus the {int(params['atr_length'])}-day ATR on the signal bar. "
            f"{int(params['partial_pct'])}% of the shares sell there."
        )
    if key == "size":
        return (
            f"Shares are floor((portfolio value / 20) × {float(params['risk_per_atr']):g} "
            "/ (close − stop)), which is 1% of equity over that distance. "
            f"{cap_sentence(params)} "
            f"Same-morning buys are filled by {rank_line(params, RANK_LABELS)}."
        )
    raise KeyError(key)


def _cloud(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    return compute_indicator(
        "ichimoku",
        bars,
        {
            "tenkan": int(params["tenkan"]),
            "kijun": int(params["kijun"]),
            "senkou_b": 52,
            "displacement": 26,
        },
    )


def prepare_signals(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    index = bars.index
    if bars.empty:
        return pack(index, [], [], [], [], [])
    if int(params["tenkan"]) >= int(params["kijun"]):
        return pack(index, np.zeros(len(bars), dtype=bool), np.zeros(len(bars), dtype=bool), np.full(len(bars), np.nan), np.full(len(bars), np.nan), np.full(len(bars), np.nan))
    cloud = _cloud(bars, params)
    width = atr(bars, int(params["atr_length"]))
    strength = adx(bars, int(params["atr_length"]))["adx"]
    tenkan = cloud["tenkan"]
    kijun = cloud["kijun"]
    top = pd.concat([cloud["senkou_a"], cloud["senkou_b"]], axis=1).max(axis=1)
    close = bars["close"]
    crossed = ((tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))).fillna(False)
    above = (close > top).fillna(False)
    strong = (strength > _adx_min(params)).fillna(False)
    under_kijun = ((close < kijun) & (close.shift(1) < kijun.shift(1))).fillna(False)
    inside = (close < top).fillna(False)
    exit_ = (under_kijun | inside).to_numpy(dtype=bool)
    stop_price = (kijun - width).to_numpy(dtype=float)
    distance = (close - (kijun - width)).to_numpy(dtype=float)
    entry = (crossed & above & strong).to_numpy(dtype=bool) & ~exit_ & np.isfinite(distance) & (distance > 0)
    gap = ((close - top) / width.replace(0, np.nan)).to_numpy(dtype=float)
    return pack(
        index,
        entry,
        exit_,
        distance,
        strength.to_numpy(dtype=float),
        gap,
        stop_price=stop_price,
    )


def chart_frame(bars: pd.DataFrame, params: dict):
    tenkan_name = f"Tenkan {int(params['tenkan'])}"
    kijun_name = f"Kijun {int(params['kijun'])}"
    if bars.empty:
        columns = [tenkan_name, kijun_name, "Senkou A", "Senkou B", "ADX"]
        frame = pd.DataFrame(columns=columns, index=bars.index)
    else:
        cloud = _cloud(bars, params)
        strength = adx(bars, int(params["atr_length"]))["adx"]
        frame = pd.DataFrame(
            {
                tenkan_name: cloud["tenkan"],
                kijun_name: cloud["kijun"],
                "Senkou A": cloud["senkou_a"],
                "Senkou B": cloud["senkou_b"],
                "ADX": strength,
            },
            index=bars.index,
        )
    kinds = {
        tenkan_name: "overlay",
        kijun_name: "overlay",
        "Senkou A": "overlay",
        "Senkou B": "overlay",
        "ADX": "oscillator",
    }
    return frame, kinds


class IchimokuTrend(Strategy):
    key = "s07_ichimoku_trend"
    name = "Ichimoku Trend Rider"
    portfolio = True
    stages = STAGES
    cost_params = COST_PARAMS
    param_labels = PARAM_LABELS
    param_choices = {"entry_rank": tuple((RANK_LABELS[index], index) for index in range(3))}

    def param_space(self) -> dict[str, ParamSpec]:
        return book_space(
            {
                "tenkan": ParamSpec(9, 9, 12, 3, "int"),
                "kijun": ParamSpec(26, 26, 34, 8, "int"),
                "adx_min": choice(1, len(ADX_LEVELS)),
            },
            risk_per_atr=0.2,
            stop_atr=1.0,
            max_position_pct=25.0,
            max_positions=8,
        )

    def target_positions(self, bars: pd.DataFrame, params: dict, start: int = 0) -> pd.Series:
        raise RuntimeError("Ichimoku trend trades the book. Run it from the bank.")

    def signals(self, bars: pd.DataFrame, params: dict) -> pd.DataFrame:
        return prepare_signals(bars, resolve_params(self.param_space(), params))

    def indicator_frame(self, bars: pd.DataFrame, params: dict):
        return chart_frame(bars, resolve_params(self.param_space(), params))

    def stage_line(self, key: str, params: dict) -> str:
        return stage_line(key, params)
