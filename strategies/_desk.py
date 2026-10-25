"""Shared pieces for the eight desk strategies.

The book sizes shares as portfolio value / 20, times the size factor, divided
by the signal ATR, then the rupee cap and cash. These strategies put the risk
distance in that ATR column. A factor of 0.2 is then 1% of equity per share of
distance. `size` scales the share count without moving the stop. `stop_pad` is
a distance under the fill. `stop_price` is an absolute stop. The first 300 bars
of each name are not bought.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.book import COST_PARAMS
from engine.models import ParamSpec

WARMUP = 300
ATR_LENGTH = 14

COST_SPECS = {
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


def choice(default_index: int, count: int) -> ParamSpec:
    """Integer selector. The train window only tries the listed points."""

    return ParamSpec(int(default_index), 0, int(count) - 1, 1, "int")


def pick(code, values):
    index = int(round(float(code)))
    if index < 0 or index >= len(values):
        index = 0
    return values[index]


def book_space(indicators: dict[str, ParamSpec], **account) -> dict[str, ParamSpec]:
    merged = account_specs(**account)
    merged.update(indicators)
    merged.update(COST_SPECS)
    return merged


def account_specs(
    *,
    risk_per_atr: float,
    stop_atr: float = 1.0,
    max_position_pct: float = 25.0,
    max_positions: int | None = None,
    stop_pct: float | None = None,
    entry_rank: int = 0,
    max_position_inr: int = 50_000_000,
    capital: int = 2_000_000,
) -> dict[str, ParamSpec]:
    space = {
        "atr_length": ParamSpec(ATR_LENGTH, None, None, None, "int"),
        "stop_atr": ParamSpec(float(stop_atr), None, None, None, "float"),
        "partial_pct": ParamSpec(100, None, None, None, "int"),
        "entry_rank": ParamSpec(int(entry_rank), 0, 2, 1, "int"),
        "capital": ParamSpec(int(capital), None, None, None, "int"),
        "max_position_inr": ParamSpec(int(max_position_inr), None, None, None, "int"),
        "max_position_pct": ParamSpec(float(max_position_pct), None, None, None, "float"),
        "risk_per_atr": ParamSpec(float(risk_per_atr), None, None, None, "float"),
    }
    if max_positions is not None:
        space["max_positions"] = ParamSpec(int(max_positions), None, None, None, "int")
    if stop_pct is not None:
        space["stop_pct"] = ParamSpec(float(stop_pct), None, None, None, "float")
    return space


def pack(
    index,
    entry,
    exit_,
    distance,
    udr,
    spread,
    size=None,
    stop_pad=None,
    stop_price=None,
) -> pd.DataFrame:
    """One signal frame. Buys before bar 300, and buys with no risk distance, are removed."""

    frame = pd.DataFrame(
        {
            "entry": np.asarray(entry, dtype=bool),
            "exit": np.asarray(exit_, dtype=bool),
            "atr": np.asarray(distance, dtype=float),
            "udr": np.asarray(udr, dtype=float),
            "spread_atr": np.asarray(spread, dtype=float),
        },
        index=index,
    )
    if size is not None:
        frame["size"] = np.asarray(size, dtype=float)
    if stop_pad is not None:
        frame["stop_pad"] = np.asarray(stop_pad, dtype=float)
    if stop_price is not None:
        frame["stop_price"] = np.asarray(stop_price, dtype=float)
    allowed = np.isfinite(frame["atr"].to_numpy(dtype=float)) & (frame["atr"].to_numpy(dtype=float) > 0)
    if len(frame) > WARMUP:
        allowed[:WARMUP] = False
    else:
        allowed[:] = False
    frame["entry"] = frame["entry"].to_numpy(dtype=bool) & allowed
    return frame


def flat(bars: pd.DataFrame, distance=None) -> pd.DataFrame:
    count = len(bars)
    width = np.full(count, np.nan) if distance is None else np.asarray(distance, dtype=float)
    return pack(
        bars.index,
        np.zeros(count, dtype=bool),
        np.zeros(count, dtype=bool),
        width,
        np.full(count, np.nan),
        np.full(count, np.nan),
    )


def rising_max(level: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Highest finite level seen while `active` stays on. A gap in `active` resets it."""

    out = np.full(len(level), np.nan)
    current = np.nan
    for i in range(len(level)):
        if not active[i]:
            current = np.nan
            continue
        if np.isfinite(level[i]):
            current = level[i] if not np.isfinite(current) else max(current, float(level[i]))
        if np.isfinite(current):
            out[i] = current
    return out


def run_length(flags: np.ndarray) -> np.ndarray:
    out = np.zeros(len(flags), dtype=int)
    count = 0
    for i, flag in enumerate(np.asarray(flags, dtype=bool)):
        count = count + 1 if flag else 0
        out[i] = count
    return out


def rolling_linreg_end(values: np.ndarray, length: int) -> np.ndarray:
    """Fitted value at the last point of each window. Matches the catalog regression endpoint."""

    series = np.asarray(values, dtype=float)
    n = int(length)
    out = np.full(len(series), np.nan)
    if n < 2 or len(series) < n:
        return out
    x = np.arange(n, dtype=float)
    x_mean = (n - 1) / 2.0
    var_x = float(np.sum((x - x_mean) ** 2))
    coef = (1.0 / n) + x_mean * (x - x_mean) / var_x
    finite = np.isfinite(series)
    clean = np.where(finite, series, 0.0)
    fitted = np.convolve(clean, coef[::-1], mode="valid")
    missing = np.convolve((~finite).astype(float), np.ones(n), mode="valid")
    fitted[missing > 0] = np.nan
    out[n - 1 :] = fitted
    return out


def rank_line(params: dict, labels: dict[int, str]) -> str:
    return labels[int(params["entry_rank"])]


def money_flow(bars: pd.DataFrame, length: int) -> pd.Series:
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    close = bars["close"].astype(float)
    volume = bars["volume"].astype(float)
    typical = (high + low + close) / 3.0
    flow = typical * volume
    delta = typical.diff()
    window = int(length)
    positive = flow.where(delta > 0, 0.0).rolling(window, min_periods=window).sum()
    negative = flow.where(delta < 0, 0.0).rolling(window, min_periods=window).sum()
    ratio = positive / negative.replace(0, np.nan)
    values = 100 - 100 / (1 + ratio)
    values = values.where(negative != 0, 100.0)
    values = values.mask((positive == 0) & (negative == 0))
    return values


def chaikin(bars: pd.DataFrame, length: int) -> pd.Series:
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    close = bars["close"].astype(float)
    volume = bars["volume"].astype(float)
    span = (high - low).replace(0, np.nan)
    multiplier = ((close - low) - (high - close)) / span
    window = int(length)
    return (multiplier * volume).rolling(window, min_periods=window).sum() / volume.rolling(
        window, min_periods=window
    ).sum().replace(0, np.nan)


def relative_volume(volume: pd.Series, length: int) -> pd.Series:
    """Today's volume divided by the mean of the previous `length` sessions."""

    base = volume.shift(1).rolling(int(length), min_periods=int(length)).mean()
    return volume / base.replace(0, np.nan)


def cap_sentence(params: dict) -> str:
    slots = params.get("max_positions")
    slot_text = f" At most {int(slots)} positions are open." if slots else ""
    return (
        f"A position is capped at {float(params['max_position_pct']):g}% of equity "
        f"and at ₹{int(params['max_position_inr']):,}."
        f"{slot_text} Starting cash is ₹{int(params['capital']):,}."
    )


COST_PARAMS = COST_PARAMS
