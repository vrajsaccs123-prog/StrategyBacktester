"""Registry and shared math for catalog indicators.

Built-in SMA, EMA, RSI, MACD, Bollinger, ATR, Stochastic, ADX, Donchian, and ROC
stay in engine.indicators. Everything else registers here and is loaded from
engine.library.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.models import ParamSpec


class LibIndicator:
    def __init__(self, compute, params, outputs, plot, output_plots):
        self.compute = compute
        self.params = params
        self.outputs = outputs
        self.plot = plot
        self.output_plots = output_plots


LIBRARY: dict[str, LibIndicator] = {}


def add(kind, params, outputs, plot, fn, output_plots=None):
    if kind in LIBRARY:
        raise RuntimeError(f"Duplicate indicator {kind}")
    LIBRARY[kind] = LibIndicator(fn, params, outputs, plot, output_plots or {})


def I(default, low=1, high=500):
    return ParamSpec(int(default), int(low), int(high), 1, "int")


def F(default, low, high, step):
    return ParamSpec(float(default), float(low), float(high), float(step), "float")


def N(raw, name, default, minimum=1):
    number = int(round(float(raw.get(name, default))))
    if number < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return number


def G(raw, name, default):
    return float(raw.get(name, default))


def ohlcv(bars: pd.DataFrame):
    return (
        bars["open"].astype(float),
        bars["high"].astype(float),
        bars["low"].astype(float),
        bars["close"].astype(float),
        bars["volume"].astype(float),
    )


def V(series) -> pd.DataFrame:
    return pd.DataFrame({"value": series})


def sma(series: pd.Series, length: int) -> pd.Series:
    from engine.indicators import sma as fn

    return fn(series, int(length))


def ema(series: pd.Series, length: int) -> pd.Series:
    from engine.indicators import ema as fn

    return fn(series, int(length))


def rsi(series: pd.Series, length: int) -> pd.Series:
    from engine.indicators import rsi as fn

    return fn(series, int(length))


def atr(bars: pd.DataFrame, length: int) -> pd.Series:
    from engine.indicators import atr as fn

    return fn(bars, int(length))


def macd_frame(close: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
    from engine.indicators import macd as fn

    return fn(close, int(fast), int(slow), int(signal))


def bollinger_frame(close: pd.Series, length: int, std: float) -> pd.DataFrame:
    from engine.indicators import bollinger as fn

    return fn(close, int(length), float(std))


def adx_frame(bars: pd.DataFrame, length: int) -> pd.DataFrame:
    from engine.indicators import adx as fn

    return fn(bars, int(length))


def wilder(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


def wma(series: pd.Series, length: int) -> pd.Series:
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    weights = np.arange(1, length + 1, dtype=float)
    denom = weights.sum()
    for i in range(length - 1, len(values)):
        window = values[i - length + 1 : i + 1]
        if np.isfinite(window).all():
            out[i] = np.dot(window, weights) / denom
    return pd.Series(out, index=series.index)


def true_range(bars: pd.DataFrame) -> pd.Series:
    high, low, close = bars["high"].astype(float), bars["low"].astype(float), bars["close"].astype(float)
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    if len(tr):
        tr.iloc[0] = float(high.iloc[0] - low.iloc[0])
    return tr


def rsi_from_delta(delta: pd.Series, length: int) -> pd.Series:
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = wilder(gain, length)
    avg_loss = wilder(loss, length)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    values = 100 - (100 / (1 + rs))
    values = values.where(avg_loss != 0, 100.0)
    values = values.where(~((avg_gain == 0) & (avg_loss == 0)))
    return values


def ols(series: pd.Series, length: int):
    values = series.to_numpy(dtype=float)
    n = int(length)
    if n < 2:
        raise ValueError("period must be at least 2")
    slope = np.full(len(values), np.nan)
    end = np.full(len(values), np.nan)
    r2 = np.full(len(values), np.nan)
    stderr = np.full(len(values), np.nan)
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    var_x = np.sum((x - x_mean) ** 2)
    for i in range(n - 1, len(values)):
        window = values[i - n + 1 : i + 1]
        if not np.isfinite(window).all():
            continue
        y_mean = window.mean()
        b = np.sum((x - x_mean) * (window - y_mean)) / var_x
        a = y_mean - b * x_mean
        fitted = a + b * x
        resid = window - fitted
        ss_tot = np.sum((window - y_mean) ** 2)
        slope[i] = b
        end[i] = a + b * (n - 1)
        r2[i] = 1.0 if ss_tot == 0 else 1 - np.sum(resid**2) / ss_tot
        stderr[i] = 0.0 if n <= 2 else np.sqrt(np.sum(resid**2) / (n - 2))
    index = series.index
    return (
        pd.Series(slope, index=index),
        pd.Series(end, index=index),
        pd.Series(r2, index=index),
        pd.Series(stderr, index=index),
    )


def percent_rank(series: pd.Series, length: int) -> pd.Series:
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for i in range(length - 1, len(values)):
        window = values[i - length + 1 : i + 1]
        if np.isfinite(window).all():
            out[i] = np.mean(window < window[-1]) * 100
    return pd.Series(out, index=series.index)


def day_key(bars: pd.DataFrame) -> pd.Series:
    if isinstance(bars.index, pd.DatetimeIndex):
        return pd.Series(bars.index.normalize(), index=bars.index)
    return pd.Series(0, index=bars.index)


def median_bar_seconds(bars: pd.DataFrame) -> float | None:
    if not isinstance(bars.index, pd.DatetimeIndex) or len(bars.index) < 3:
        return None
    delta = bars.index.to_series().diff().dt.total_seconds().median()
    if pd.isna(delta):
        return None
    return float(delta)


def is_daily(bars: pd.DataFrame) -> bool:
    seconds = median_bar_seconds(bars)
    return seconds is None or seconds >= 20 * 3600


def confirmed_swings(high: np.ndarray, low: np.ndarray, left: int, right: int):
    highs = []
    lows = []
    n = len(high)
    for i in range(left, n - right):
        if high[i] > np.max(high[i - left : i]) and high[i] > np.max(high[i + 1 : i + right + 1]):
            highs.append((i + right, float(high[i])))
        if low[i] < np.min(low[i - left : i]) and low[i] < np.min(low[i + 1 : i + right + 1]):
            lows.append((i + right, float(low[i])))
    return highs, lows


def profile_levels(prices, weights, bins: int):
    prices = np.asarray(prices, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(prices) & np.isfinite(weights) & (weights > 0)
    prices = prices[mask]
    weights = weights[mask]
    if len(prices) == 0:
        return np.nan, np.nan, np.nan
    lo = float(prices.min())
    hi = float(prices.max())
    if hi <= lo:
        return lo, hi, lo
    hist, edges = np.histogram(prices, bins=int(bins), range=(lo, hi), weights=weights)
    total = float(hist.sum())
    if total <= 0:
        return np.nan, np.nan, np.nan
    poc_i = int(np.argmax(hist))
    acc = float(hist[poc_i])
    lo_i = hi_i = poc_i
    while acc < 0.7 * total and (lo_i > 0 or hi_i < len(hist) - 1):
        left = float(hist[lo_i - 1]) if lo_i > 0 else -1.0
        right = float(hist[hi_i + 1]) if hi_i < len(hist) - 1 else -1.0
        if right >= left:
            hi_i += 1
            acc += float(hist[hi_i])
        else:
            lo_i -= 1
            acc += float(hist[lo_i])
    poc = (edges[poc_i] + edges[poc_i + 1]) / 2
    return float(poc), float(edges[hi_i + 1]), float(edges[lo_i])


def hilbert(close: pd.Series, fast_limit=0.5, slow_limit=0.05) -> dict[str, pd.Series]:
    price = close.to_numpy(dtype=float)
    n = len(price)
    smooth = np.zeros(n)
    detrend = np.zeros(n)
    i1 = np.zeros(n)
    q1 = np.zeros(n)
    i2 = np.zeros(n)
    q2 = np.zeros(n)
    re = np.zeros(n)
    im = np.zeros(n)
    period = np.full(n, np.nan)
    smooth_period = np.full(n, np.nan)
    phase = np.zeros(n)
    raw_delta = np.zeros(n)
    mama = np.zeros(n)
    fama = np.zeros(n)

    def lag(arr, i, k):
        return arr[i - k] if i >= k else 0.0

    for i in range(n):
        p0 = price[i] if np.isfinite(price[i]) else (price[i - 1] if i else 0.0)
        smooth[i] = (4 * p0 + 3 * lag(price, i, 1) + 2 * lag(price, i, 2) + lag(price, i, 3)) / 10
        adj = 0.075 * (period[i - 1] if i and np.isfinite(period[i - 1]) else 0.0) + 0.54

        def component(src, idx):
            return (0.0962 * lag(src, idx, 0) + 0.5769 * lag(src, idx, 2) - 0.5769 * lag(src, idx, 4) - 0.0962 * lag(src, idx, 6)) * adj

        detrend[i] = component(smooth, i)
        q1[i] = component(detrend, i)
        i1[i] = lag(detrend, i, 3)
        ji = component(i1, i)
        jq = component(q1, i)
        i2[i] = 0.2 * (i1[i] - jq) + 0.8 * lag(i2, i, 1)
        q2[i] = 0.2 * (q1[i] + ji) + 0.8 * lag(q2, i, 1)
        re[i] = 0.2 * (i2[i] * lag(i2, i, 1) + q2[i] * lag(q2, i, 1)) + 0.8 * lag(re, i, 1)
        im[i] = 0.2 * (i2[i] * lag(q2, i, 1) - q2[i] * lag(i2, i, 1)) + 0.8 * lag(im, i, 1)
        prev = period[i - 1] if i and np.isfinite(period[i - 1]) and period[i - 1] > 0 else 15.0
        if re[i] != 0 or im[i] != 0:
            angle = np.degrees(np.arctan2(im[i], re[i]))
            per = prev if abs(angle) < 1e-6 else abs(360.0 / angle)
        else:
            per = prev
        if not np.isfinite(per) or per <= 0:
            per = prev
        per = min(max(per, 0.67 * prev), 1.5 * prev)
        per = min(max(per, 6.0), 50.0)
        period[i] = 0.2 * per + 0.8 * prev
        prev_smooth = smooth_period[i - 1] if i and np.isfinite(smooth_period[i - 1]) else period[i]
        smooth_period[i] = 0.33 * period[i] + 0.67 * prev_smooth
        if i1[i] != 0:
            phase[i] = np.degrees(np.arctan(q1[i] / i1[i]))
        elif i:
            phase[i] = phase[i - 1]
        raw_delta[i] = (phase[i - 1] if i else phase[i]) - phase[i]
        delta = raw_delta[i] if raw_delta[i] >= 1 else 1.0
        alpha = fast_limit / delta
        alpha = min(max(alpha, slow_limit), fast_limit)
        prev_mama = mama[i - 1] if i else p0
        mama[i] = alpha * p0 + (1 - alpha) * prev_mama
        prev_fama = fama[i - 1] if i else p0
        fama[i] = 0.5 * alpha * mama[i] + (1 - 0.5 * alpha) * prev_fama

    trend = np.full(n, np.nan)
    dc = np.clip(np.round(np.nan_to_num(smooth_period, nan=10.0)), 2, 50).astype(int)
    for i in range(n):
        width = int(dc[i])
        if i + 1 >= width and np.isfinite(price[i - width + 1 : i + 1]).all():
            trend[i] = np.mean(price[i - width + 1 : i + 1])
    weights = np.arange(1, 5, dtype=float)
    weights /= weights.sum()
    trend_s = trend.copy()
    for i in range(3, n):
        window = trend[i - 3 : i + 1]
        if np.isfinite(window).all():
            trend_s[i] = np.dot(window, weights)

    index = close.index

    def series(values):
        return pd.Series(values, index=index, dtype=float)

    return {
        "period": series(smooth_period),
        "sine": series(np.sin(np.radians(phase))),
        "leadsine": series(np.sin(np.radians(phase + 45))),
        "mama": series(mama),
        "fama": series(fama),
        "trendline": series(trend_s),
        "trendmode": series(np.where(raw_delta < 1, 1.0, 0.0)),
    }


def _load() -> None:
    from engine import library  # noqa: F401


_load()
