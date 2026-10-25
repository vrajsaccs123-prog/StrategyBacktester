"""Moving averages, trend, and momentum indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.indicator_impl import (
    F,
    G,
    I,
    N,
    V,
    adx_frame,
    add,
    atr,
    bollinger_frame,
    ema,
    macd_frame,
    ohlcv,
    ols,
    percent_rank,
    rsi,
    rsi_from_delta,
    sma,
    true_range,
    wilder,
    wma,
)


def _c(bars):
    return bars["close"].astype(float)


def _hlc(bars):
    return bars["high"].astype(float), bars["low"].astype(float), bars["close"].astype(float)


def _dema(bars, raw):
    n = N(raw, "period", 20)
    line = ema(_c(bars), n)
    return V(2 * line - ema(line, n))


def _tema(bars, raw):
    n = N(raw, "period", 20)
    close = _c(bars)
    e1 = ema(close, n)
    e2 = ema(e1, n)
    e3 = ema(e2, n)
    return V(3 * e1 - 3 * e2 + e3)


def _wma_ind(bars, raw):
    return V(wma(_c(bars), N(raw, "period", 20)))


def _hma(bars, raw):
    n = N(raw, "period", 20)
    close = _c(bars)
    half = max(1, n // 2)
    root = max(1, int(round(n**0.5)))
    return V(wma(2 * wma(close, half) - wma(close, n), root))


def _smma(bars, raw):
    return V(wilder(_c(bars), N(raw, "period", 14)))


def _kama(bars, raw):
    n = N(raw, "period", 10)
    fast = N(raw, "fast", 2)
    slow = N(raw, "slow", 30)
    close = _c(bars).to_numpy(dtype=float)
    out = np.full(len(close), np.nan)
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    for i in range(n, len(close)):
        window = close[i - n : i + 1]
        change = abs(window[-1] - window[0])
        volatility = np.abs(np.diff(window)).sum()
        er = 0.0 if volatility == 0 else change / volatility
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        prev = close[i] if not np.isfinite(out[i - 1]) else out[i - 1]
        out[i] = prev + sc * (close[i] - prev)
    return V(pd.Series(out, index=bars.index))


def _vidya(bars, raw):
    n = N(raw, "period", 14)
    cmo_n = N(raw, "cmo_period", 9)
    close = _c(bars)
    delta = close.diff()
    up = delta.clip(lower=0).rolling(cmo_n, min_periods=cmo_n).sum()
    down = (-delta.clip(upper=0)).rolling(cmo_n, min_periods=cmo_n).sum()
    cmo = ((up - down) / (up + down).replace(0, np.nan)).abs()
    alpha = 2 / (n + 1)
    values = close.to_numpy(dtype=float)
    scale = cmo.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        if not np.isfinite(values[i]) or not np.isfinite(scale[i]):
            continue
        prev = values[i] if not np.isfinite(out[i - 1] if i else np.nan) else out[i - 1]
        k = alpha * scale[i]
        out[i] = k * values[i] + (1 - k) * prev
    return V(pd.Series(out, index=bars.index))


def _alma(bars, raw):
    n = N(raw, "period", 9)
    offset = G(raw, "offset", 0.85)
    sigma = G(raw, "sigma", 6)
    if sigma == 0:
        raise ValueError("sigma must not be 0")
    m = offset * (n - 1)
    s = n / sigma
    slots = np.arange(n, dtype=float)
    weights = np.exp(-((slots - m) ** 2) / (2 * s * s))
    weights /= weights.sum()
    values = _c(bars).to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for i in range(n - 1, len(values)):
        window = values[i - n + 1 : i + 1]
        if np.isfinite(window).all():
            out[i] = np.dot(window, weights)
    return V(pd.Series(out, index=bars.index))


def _zlema(bars, raw):
    n = N(raw, "period", 20)
    close = _c(bars)
    lag = (n - 1) // 2
    return V(ema(2 * close - close.shift(lag), n))


def _t3(bars, raw):
    n = N(raw, "period", 5)
    v = G(raw, "v_factor", 0.7)
    close = _c(bars)
    e1 = ema(close, n)
    e2 = ema(e1, n)
    e3 = ema(e2, n)
    e4 = ema(e3, n)
    e5 = ema(e4, n)
    e6 = ema(e5, n)
    c1 = -(v**3)
    c2 = 3 * v**2 + 3 * v**3
    c3 = -6 * v**2 - 3 * v - 3 * v**3
    c4 = 1 + 3 * v + v**3 + 3 * v**2
    return V(c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3)


def _trima(bars, raw):
    n = N(raw, "period", 20)
    inner = int(np.ceil((n + 1) / 2))
    close = _c(bars)
    return V(sma(sma(close, inner), inner))


def _vwma(bars, raw):
    n = N(raw, "period", 20)
    _o, _h, _l, close, volume = ohlcv(bars)
    denom = volume.rolling(n, min_periods=n).sum().replace(0, np.nan)
    return V((close * volume).rolling(n, min_periods=n).sum() / denom)


def _lsma(bars, raw):
    _slope, end, _r2, _err = ols(_c(bars), N(raw, "period", 25))
    return V(end)


def _mcginley(bars, raw):
    n = N(raw, "period", 14)
    values = _c(bars).to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    k = 0.6
    for i, price in enumerate(values):
        if not np.isfinite(price):
            continue
        if not np.isfinite(out[i - 1] if i else np.nan) or out[i - 1] == 0:
            out[i] = price
            continue
        prev = out[i - 1]
        out[i] = prev + (price - prev) / (k * n * (price / prev) ** 4)
    return V(pd.Series(out, index=bars.index))


def _frama(bars, raw):
    n = N(raw, "period", 16)
    n = n if n % 2 == 0 else n - 1
    n = max(n, 2)
    half = n // 2
    high, low, close = _hlc(bars)
    n1 = (high.rolling(half, min_periods=half).max() - low.rolling(half, min_periods=half).min()).shift(half)
    n2 = high.rolling(half, min_periods=half).max() - low.rolling(half, min_periods=half).min()
    n3 = high.rolling(n, min_periods=n).max() - low.rolling(n, min_periods=n).min()
    dimension = (np.log(n1 + n2) - np.log(n3.replace(0, np.nan))) / np.log(2)
    alpha = np.exp(-4.6 * (dimension - 1)).clip(0.01, 1)
    values = close.to_numpy(dtype=float)
    coeffs = alpha.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for i, price in enumerate(values):
        if not np.isfinite(price) or not np.isfinite(coeffs[i]):
            continue
        prev = price if not np.isfinite(out[i - 1] if i else np.nan) else out[i - 1]
        out[i] = coeffs[i] * price + (1 - coeffs[i]) * prev
    return V(pd.Series(out, index=bars.index))


def _jma(bars, raw):
    n = N(raw, "period", 14)
    phase = G(raw, "phase", 0)
    phase_ratio = phase / 100 + 1.5
    beta = 0.45 * (n - 1) / (0.45 * (n - 1) + 2)
    alpha = beta**2
    values = _c(bars).to_numpy(dtype=float)
    e0 = np.zeros(len(values))
    e1 = np.zeros(len(values))
    e2 = np.zeros(len(values))
    out = np.zeros(len(values))
    for i, price in enumerate(values):
        if not np.isfinite(price):
            out[i] = np.nan
            continue
        prev_e0 = e0[i - 1] if i else price
        prev_e1 = e1[i - 1] if i else 0.0
        prev_e2 = e2[i - 1] if i else 0.0
        prev = out[i - 1] if i else price
        e0[i] = (1 - alpha) * price + alpha * prev_e0
        e1[i] = (price - e0[i]) * (1 - beta) + beta * prev_e1
        e2[i] = (e0[i] + phase_ratio * e1[i] - prev) * (1 - alpha) ** 2 + alpha**2 * prev_e2
        out[i] = e2[i] + prev
    return V(pd.Series(out, index=bars.index))


def _envelope(bars, raw):
    n = N(raw, "period", 20)
    pct = G(raw, "percent", 2.5)
    mid = sma(_c(bars), n)
    return pd.DataFrame({"upper": mid * (1 + pct / 100), "middle": mid, "lower": mid * (1 - pct / 100)})


def _ppo(bars, raw):
    fast, slow, signal_n = N(raw, "fast", 12), N(raw, "slow", 26), N(raw, "signal", 9)
    close = _c(bars)
    fast_e, slow_e = ema(close, fast), ema(close, slow)
    line = 100 * (fast_e - slow_e) / slow_e.replace(0, np.nan)
    signal = ema(line, signal_n)
    return pd.DataFrame({"ppo": line, "signal": signal, "histogram": line - signal})


def _adxr(bars, raw):
    n = N(raw, "period", 14)
    line = adx_frame(bars, n)["adx"]
    return V((line + line.shift(n)) / 2)


def _dmi(bars, raw):
    frame = adx_frame(bars, N(raw, "period", 14))
    return pd.DataFrame({"plus_di": frame["plus_di"], "minus_di": frame["minus_di"]})


def _aroon_pair(high, low, n):
    def since_extreme(is_high):
        def stat(window):
            pos = np.argmax(window) if is_high else np.argmin(window)
            return (n - 1) - pos

        return (high if is_high else low).rolling(n, min_periods=n).apply(stat, raw=True)

    up = 100 * (n - since_extreme(True)) / n
    down = 100 * (n - since_extreme(False)) / n
    return up, down


def _aroon(bars, raw):
    n = N(raw, "period", 25)
    high, low, _close = _hlc(bars)
    up, down = _aroon_pair(high, low, n)
    return pd.DataFrame({"aroon_up": up, "aroon_down": down})


def _aroon_osc(bars, raw):
    n = N(raw, "period", 25)
    high, low, _close = _hlc(bars)
    up, down = _aroon_pair(high, low, n)
    return V(up - down)


def _psar(bars, raw):
    start, step, limit = G(raw, "af_start", 0.02), G(raw, "af_step", 0.02), G(raw, "af_max", 0.2)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    n = len(high)
    out = np.full(n, np.nan)
    if n == 0:
        return V(pd.Series(out, index=bars.index))
    bull = True
    af = start
    ep = high[0]
    out[0] = low[0]
    for i in range(1, n):
        out[i] = out[i - 1] + af * (ep - out[i - 1])
        if bull:
            out[i] = min(out[i], low[i - 1], low[i - 2] if i > 1 else low[i - 1])
            if low[i] < out[i]:
                bull = False
                out[i] = ep
                ep = low[i]
                af = start
            elif high[i] > ep:
                ep = high[i]
                af = min(af + step, limit)
        else:
            out[i] = max(out[i], high[i - 1], high[i - 2] if i > 1 else high[i - 1])
            if high[i] > out[i]:
                bull = True
                out[i] = ep
                ep = high[i]
                af = start
            elif low[i] < ep:
                ep = low[i]
                af = min(af + step, limit)
    return V(pd.Series(out, index=bars.index))


def _supertrend(bars, raw):
    n = N(raw, "period", 10)
    mult = G(raw, "multiplier", 3)
    high, low, close = _hlc(bars)
    hl2 = (high + low) / 2
    width = atr(bars, n)
    basic_u = (hl2 + mult * width).to_numpy(dtype=float)
    basic_l = (hl2 - mult * width).to_numpy(dtype=float)
    close_v = close.to_numpy(dtype=float)
    size = len(close_v)
    final_u = np.full(size, np.nan)
    final_l = np.full(size, np.nan)
    direction = np.full(size, np.nan)
    line = np.full(size, np.nan)
    for i in range(size):
        if not np.isfinite(basic_u[i]) or not np.isfinite(basic_l[i]):
            continue
        if i == 0 or not np.isfinite(final_u[i - 1]):
            final_u[i] = basic_u[i]
            final_l[i] = basic_l[i]
            direction[i] = 1
            line[i] = final_l[i]
            continue
        final_u[i] = basic_u[i] if basic_u[i] < final_u[i - 1] or close_v[i - 1] > final_u[i - 1] else final_u[i - 1]
        final_l[i] = basic_l[i] if basic_l[i] > final_l[i - 1] or close_v[i - 1] < final_l[i - 1] else final_l[i - 1]
        if direction[i - 1] >= 0 and close_v[i] < final_l[i]:
            direction[i] = -1
            line[i] = final_u[i]
        elif direction[i - 1] < 0 and close_v[i] > final_u[i]:
            direction[i] = 1
            line[i] = final_l[i]
        elif direction[i - 1] >= 0:
            direction[i] = 1
            line[i] = final_l[i]
        else:
            direction[i] = -1
            line[i] = final_u[i]
    return pd.DataFrame(
        {"supertrend": pd.Series(line, index=bars.index), "direction": pd.Series(direction, index=bars.index)}
    )


def _ichimoku(bars, raw):
    tenkan_n = N(raw, "tenkan", 9)
    kijun_n = N(raw, "kijun", 26)
    span_n = N(raw, "senkou_b", 52)
    shift = N(raw, "displacement", 26)
    high, low, close = _hlc(bars)

    def mid(length):
        return (high.rolling(length, min_periods=length).max() + low.rolling(length, min_periods=length).min()) / 2

    tenkan = mid(tenkan_n)
    kijun = mid(kijun_n)
    senkou_a = ((tenkan + kijun) / 2).shift(shift)
    senkou_b = mid(span_n).shift(shift)
    chikou = close.shift(shift)
    return pd.DataFrame(
        {"tenkan": tenkan, "kijun": kijun, "senkou_a": senkou_a, "senkou_b": senkou_b, "chikou": chikou}
    )


def _vortex(bars, raw):
    n = N(raw, "period", 14)
    high, low, _close = _hlc(bars)
    plus = (high - low.shift(1)).abs().rolling(n, min_periods=n).sum()
    minus = (low - high.shift(1)).abs().rolling(n, min_periods=n).sum()
    tr = true_range(bars).rolling(n, min_periods=n).sum().replace(0, np.nan)
    return pd.DataFrame({"vi_plus": plus / tr, "vi_minus": minus / tr})


def _mass(bars, raw):
    ema_n = N(raw, "ema_period", 9)
    sum_n = N(raw, "sum_period", 25)
    span = bars["high"].astype(float) - bars["low"].astype(float)
    e1 = ema(span, ema_n)
    ratio = e1 / ema(e1, ema_n).replace(0, np.nan)
    return V(ratio.rolling(sum_n, min_periods=sum_n).sum())


def _chop(bars, raw):
    n = N(raw, "period", 14)
    high, low, _close = _hlc(bars)
    span = (high.rolling(n, min_periods=n).max() - low.rolling(n, min_periods=n).min()).replace(0, np.nan)
    total = true_range(bars).rolling(n, min_periods=n).sum()
    return V(100 * np.log10(total / span) / np.log10(n))


def _vhf(bars, raw):
    n = N(raw, "period", 28)
    close = _c(bars)
    path = close.diff().abs().rolling(n, min_periods=n).sum().replace(0, np.nan)
    return V((close.rolling(n, min_periods=n).max() - close.rolling(n, min_periods=n).min()) / path)


def _trix(bars, raw):
    n = N(raw, "period", 15)
    close = _c(bars)
    e3 = ema(ema(ema(close, n), n), n)
    return V(100 * (e3 - e3.shift(1)) / e3.shift(1).replace(0, np.nan))


def _cci(bars, raw):
    n = N(raw, "period", 20)
    high, low, close = _hlc(bars)
    tp = (high + low + close) / 3

    def mad(window):
        return np.mean(np.abs(window - window.mean()))

    mean = sma(tp, n)
    deviation = tp.rolling(n, min_periods=n).apply(mad, raw=True).replace(0, np.nan)
    return V((tp - mean) / (0.015 * deviation))


def _stoch_rsi(bars, raw):
    rsi_n = N(raw, "rsi_period", 14)
    stoch_n = N(raw, "stoch_period", 14)
    smooth = N(raw, "k", 3)
    d_n = N(raw, "d", 3)
    line = rsi(_c(bars), rsi_n)
    lo = line.rolling(stoch_n, min_periods=stoch_n).min()
    hi = line.rolling(stoch_n, min_periods=stoch_n).max()
    k = 100 * (line - lo) / (hi - lo).replace(0, np.nan)
    k = k.rolling(smooth, min_periods=smooth).mean()
    d = k.rolling(d_n, min_periods=d_n).mean()
    return pd.DataFrame({"k": k, "d": d})


def _williams(bars, raw):
    n = N(raw, "period", 14)
    high, low, close = _hlc(bars)
    hh = high.rolling(n, min_periods=n).max()
    ll = low.rolling(n, min_periods=n).min()
    return V(-100 * (hh - close) / (hh - ll).replace(0, np.nan))


def _mom(bars, raw):
    n = N(raw, "period", 10)
    close = _c(bars)
    return V(close - close.shift(n))


def _cmo(bars, raw):
    n = N(raw, "period", 14)
    delta = _c(bars).diff()
    up = delta.clip(lower=0).rolling(n, min_periods=n).sum()
    down = (-delta.clip(upper=0)).rolling(n, min_periods=n).sum()
    return V(100 * (up - down) / (up + down).replace(0, np.nan))


def _tsi(bars, raw):
    long_n, short_n, signal_n = N(raw, "long", 25), N(raw, "short", 13), N(raw, "signal", 7)
    delta = _c(bars).diff()
    line = 100 * ema(ema(delta, long_n), short_n) / ema(ema(delta.abs(), long_n), short_n).replace(0, np.nan)
    return pd.DataFrame({"tsi": line, "signal": ema(line, signal_n)})


def _uo(bars, raw):
    p1, p2, p3 = N(raw, "p1", 7), N(raw, "p2", 14), N(raw, "p3", 28)
    high, low, close = _hlc(bars)
    prev = close.shift(1)
    bp = close - np.minimum(low, prev)
    tr = np.maximum(high, prev) - np.minimum(low, prev)

    def avg(length):
        return bp.rolling(length, min_periods=length).sum() / tr.rolling(length, min_periods=length).sum().replace(0, np.nan)

    return V(100 * (4 * avg(p1) + 2 * avg(p2) + avg(p3)) / 7)


def _ao(bars, raw):
    fast, slow = N(raw, "fast", 5), N(raw, "slow", 34)
    median = (bars["high"].astype(float) + bars["low"].astype(float)) / 2
    return V(sma(median, fast) - sma(median, slow))


def _ac(bars, raw):
    signal_n = N(raw, "ao_signal", 5)
    median = (bars["high"].astype(float) + bars["low"].astype(float)) / 2
    ao = sma(median, 5) - sma(median, 34)
    return V(ao - sma(ao, signal_n))


def _dpo(bars, raw):
    n = N(raw, "period", 20)
    close = _c(bars)
    return V(close.shift(n // 2 + 1) - sma(close, n))


def _kst(bars, raw):
    rocs = [N(raw, "r1", 10), N(raw, "r2", 15), N(raw, "r3", 20), N(raw, "r4", 30)]
    smas = [N(raw, "s1", 10), N(raw, "s2", 10), N(raw, "s3", 10), N(raw, "s4", 15)]
    close = _c(bars)
    line = 0
    for weight, (roc_n, sma_n) in enumerate(zip(rocs, smas), start=1):
        roc = (close / close.shift(roc_n) - 1) * 100
        line = line + weight * sma(roc, sma_n)
    signal = sma(line, N(raw, "signal", 9))
    return pd.DataFrame({"kst": line, "signal": signal})


def _coppock(bars, raw):
    close = _c(bars)
    roc = (close / close.shift(N(raw, "roc1", 14)) - 1) * 100 + (close / close.shift(N(raw, "roc2", 11)) - 1) * 100
    return V(wma(roc, N(raw, "wma", 10)))


def _connors(bars, raw):
    rsi_n, streak_n, rank_n = N(raw, "rsi", 3), N(raw, "streak", 2), N(raw, "rank", 100)
    close = _c(bars)
    values = close.to_numpy(dtype=float)
    streak = np.zeros(len(values))
    for i in range(1, len(values)):
        if values[i] > values[i - 1]:
            streak[i] = streak[i - 1] + 1 if streak[i - 1] > 0 else 1
        elif values[i] < values[i - 1]:
            streak[i] = streak[i - 1] - 1 if streak[i - 1] < 0 else -1
    streak_s = pd.Series(streak, index=bars.index)
    ranked = percent_rank(close.pct_change(), rank_n)
    return V((rsi(close, rsi_n) + rsi(streak_s, streak_n) + ranked) / 3)


def _rvi(bars, raw):
    n = N(raw, "period", 10)
    opened, high, low, close, _volume = ohlcv(bars)

    def swma(series):
        return (series + 2 * series.shift(1) + 2 * series.shift(2) + series.shift(3)) / 6

    line = sma(swma(close - opened), n) / sma(swma((high - low).replace(0, np.nan)), n).replace(0, np.nan)
    return pd.DataFrame({"rvi": line, "signal": swma(line)})


def _rmi(bars, raw):
    return V(rsi_from_delta(_c(bars).diff(N(raw, "momentum", 5)), N(raw, "period", 14)))


def _stoch_smooth(series, length):
    lo = series.rolling(length, min_periods=length).min()
    hi = series.rolling(length, min_periods=length).max()
    k = 100 * (series - lo) / (hi - lo).replace(0, np.nan)
    values = k.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    prev = np.nan
    for i, value in enumerate(values):
        if not np.isfinite(value):
            continue
        prev = value if not np.isfinite(prev) else prev + 0.5 * (value - prev)
        out[i] = prev
    return pd.Series(out, index=series.index)


def _stc(bars, raw):
    fast, slow, cycle = N(raw, "fast", 23), N(raw, "slow", 50), N(raw, "cycle", 10)
    close = _c(bars)
    macd = ema(close, fast) - ema(close, slow)
    return V(_stoch_smooth(_stoch_smooth(macd, cycle), cycle))


def _pmo(bars, raw):
    close = _c(bars)
    roc1 = close.pct_change() * 100
    line = ema(ema(10 * roc1, N(raw, "s1", 35)), N(raw, "s2", 20))
    return pd.DataFrame({"pmo": line, "signal": ema(line, N(raw, "signal", 10))})


def _fisher(bars, raw):
    n = N(raw, "period", 10)
    high, low, _close = _hlc(bars)
    mid = ((high + low) / 2).to_numpy(dtype=float)
    hh = high.rolling(n, min_periods=n).max().to_numpy(dtype=float)
    ll = low.rolling(n, min_periods=n).min().to_numpy(dtype=float)
    x = np.zeros(len(mid))
    fish = np.full(len(mid), np.nan)
    for i in range(len(mid)):
        if not np.isfinite(hh[i]) or not np.isfinite(ll[i]) or hh[i] == ll[i]:
            continue
        raw_x = 0.66 * ((mid[i] - ll[i]) / (hh[i] - ll[i]) - 0.5) + 0.67 * (x[i - 1] if i else 0.0)
        raw_x = min(max(raw_x, -0.999), 0.999)
        x[i] = raw_x
        prev = fish[i - 1] if i and np.isfinite(fish[i - 1]) else 0.0
        fish[i] = 0.5 * np.log((1 + raw_x) / (1 - raw_x)) + 0.5 * prev
    index = bars.index
    line = pd.Series(fish, index=index)
    return pd.DataFrame({"fisher": line, "trigger": line.shift(1)})


def _inertia(bars, raw):
    std_n = N(raw, "rvi_period", 14)
    reg_n = N(raw, "reg_period", 20)
    close = _c(bars)
    deviation = close.rolling(10, min_periods=10).std(ddof=0)
    up = deviation.where(close.diff() > 0, 0.0)
    down = deviation.where(close.diff() <= 0, 0.0)
    up_s = wilder(up.fillna(0), std_n)
    down_s = wilder(down.fillna(0), std_n)
    rvi = 100 * up_s / (up_s + down_s).replace(0, np.nan)
    _slope, end, _r2, _err = ols(rvi, reg_n)
    return V(end)


def _squeeze(bars, raw):
    bb_n, kc_n = N(raw, "bb", 20), N(raw, "kc", 20)
    bb_mult, kc_mult = G(raw, "bb_mult", 2), G(raw, "kc_mult", 1.5)
    high, low, close = _hlc(bars)
    bands = bollinger_frame(close, bb_n, bb_mult)
    mid = sma(close, kc_n)
    width = atr(bars, kc_n)
    kc_upper = mid + kc_mult * width
    kc_lower = mid - kc_mult * width
    squeeze = ((bands["lower"] > kc_lower) & (bands["upper"] < kc_upper)).astype(float)
    basis = ((high.rolling(kc_n, min_periods=kc_n).max() + low.rolling(kc_n, min_periods=kc_n).min()) / 2 + mid) / 2
    _slope, end, _r2, _err = ols(close - basis, kc_n)
    return pd.DataFrame({"momentum": end, "squeeze_on": squeeze})


def _elder_ray(bars, raw):
    n = N(raw, "period", 13)
    high, low, close = _hlc(bars)
    basis = ema(close, n)
    return pd.DataFrame({"bull_power": high - basis, "bear_power": low - basis})


def _elder_impulse(bars, raw):
    basis = ema(_c(bars), N(raw, "ema", 13))
    hist = macd_frame(_c(bars), N(raw, "fast", 12), N(raw, "slow", 26), N(raw, "signal", 9))["hist"]
    up = (basis > basis.shift(1)) & (hist > hist.shift(1))
    down = (basis < basis.shift(1)) & (hist < hist.shift(1))
    return V(pd.Series(np.where(up.fillna(False), 1.0, np.where(down.fillna(False), -1.0, 0.0)), index=bars.index))


def _bop(bars, raw):
    opened, high, low, close, _volume = ohlcv(bars)
    raw_bop = (close - opened) / (high - low).replace(0, np.nan)
    return V(sma(raw_bop, N(raw, "smooth", 14)))


def _qstick(bars, raw):
    opened, _h, _l, close, _v = ohlcv(bars)
    return V(sma(close - opened, N(raw, "period", 8)))


add("wma", {"period": I(20, 1, 400)}, ["value"], "overlay", _wma_ind)
add("dema", {"period": I(20, 1, 400)}, ["value"], "overlay", _dema)
add("tema", {"period": I(20, 1, 400)}, ["value"], "overlay", _tema)
add("hma", {"period": I(20, 1, 400)}, ["value"], "overlay", _hma)
add("smma", {"period": I(14, 1, 400)}, ["value"], "overlay", _smma)
add("kama", {"period": I(10, 1, 400), "fast": I(2, 1, 50), "slow": I(30, 2, 200)}, ["value"], "overlay", _kama)
add("vidya", {"period": I(14, 1, 400), "cmo_period": I(9, 1, 100)}, ["value"], "overlay", _vidya)
add("alma", {"period": I(9, 2, 200), "offset": F(0.85, 0, 1, 0.05), "sigma": F(6, 1, 20, 0.5)}, ["value"], "overlay", _alma)
add("zlema", {"period": I(20, 1, 400)}, ["value"], "overlay", _zlema)
add("t3", {"period": I(5, 1, 100), "v_factor": F(0.7, 0, 1, 0.05)}, ["value"], "overlay", _t3)
add("trima", {"period": I(20, 1, 400)}, ["value"], "overlay", _trima)
add("vwma", {"period": I(20, 1, 400)}, ["value"], "overlay", _vwma)
add("lsma", {"period": I(25, 2, 400)}, ["value"], "overlay", _lsma)
add("mcginley", {"period": I(14, 1, 400)}, ["value"], "overlay", _mcginley)
add("frama", {"period": I(16, 2, 200)}, ["value"], "overlay", _frama)
add("jma", {"period": I(14, 1, 200), "phase": I(0, -100, 100)}, ["value"], "overlay", _jma)
add("ma_envelope", {"period": I(20, 1, 400), "percent": F(2.5, 0.1, 20, 0.1)}, ["upper", "middle", "lower"], "overlay", _envelope)
add("ppo", {"fast": I(12, 2, 100), "slow": I(26, 3, 200), "signal": I(9, 2, 50)}, ["ppo", "signal", "histogram"], "oscillator", _ppo)
add("adxr", {"period": I(14, 2, 100)}, ["value"], "oscillator", _adxr)
add("dmi", {"period": I(14, 2, 100)}, ["plus_di", "minus_di"], "oscillator", _dmi)
add("aroon", {"period": I(25, 2, 200)}, ["aroon_up", "aroon_down"], "oscillator", _aroon)
add("aroon_osc", {"period": I(25, 2, 200)}, ["value"], "oscillator", _aroon_osc)
add("psar", {"af_start": F(0.02, 0.001, 0.2, 0.01), "af_step": F(0.02, 0.001, 0.2, 0.01), "af_max": F(0.2, 0.02, 1, 0.01)}, ["value"], "overlay", _psar)
add("supertrend", {"period": I(10, 2, 100), "multiplier": F(3, 0.5, 10, 0.1)}, ["supertrend", "direction"], "overlay", _supertrend, {"direction": "oscillator"})
add("ichimoku", {"tenkan": I(9, 1, 100), "kijun": I(26, 1, 200), "senkou_b": I(52, 1, 400), "displacement": I(26, 1, 100)}, ["tenkan", "kijun", "senkou_a", "senkou_b", "chikou"], "overlay", _ichimoku)
add("vortex", {"period": I(14, 2, 100)}, ["vi_plus", "vi_minus"], "oscillator", _vortex)
add("mass_index", {"ema_period": I(9, 1, 50), "sum_period": I(25, 2, 100)}, ["value"], "oscillator", _mass)
add("choppiness", {"period": I(14, 2, 100)}, ["value"], "oscillator", _chop)
add("vhf", {"period": I(28, 2, 200)}, ["value"], "oscillator", _vhf)
add("trix", {"period": I(15, 1, 100)}, ["value"], "oscillator", _trix)
add("cci", {"period": I(20, 2, 200)}, ["value"], "oscillator", _cci)
add("stoch_rsi", {"rsi_period": I(14, 2, 100), "stoch_period": I(14, 2, 100), "k": I(3, 1, 20), "d": I(3, 1, 20)}, ["k", "d"], "oscillator", _stoch_rsi)
add("williams_r", {"period": I(14, 2, 100)}, ["value"], "oscillator", _williams)
add("mom", {"period": I(10, 1, 200)}, ["value"], "oscillator", _mom)
add("cmo", {"period": I(14, 2, 100)}, ["value"], "oscillator", _cmo)
add("tsi", {"long": I(25, 2, 100), "short": I(13, 2, 50), "signal": I(7, 1, 40)}, ["tsi", "signal"], "oscillator", _tsi)
add("ultimate_osc", {"p1": I(7, 1, 50), "p2": I(14, 2, 100), "p3": I(28, 3, 200)}, ["value"], "oscillator", _uo)
add("awesome_osc", {"fast": I(5, 1, 50), "slow": I(34, 2, 200)}, ["value"], "oscillator", _ao)
add("accel_osc", {"ao_signal": I(5, 1, 40)}, ["value"], "oscillator", _ac)
add("dpo", {"period": I(20, 2, 200)}, ["value"], "oscillator", _dpo)
add("kst", {"r1": I(10, 1, 100), "r2": I(15, 1, 100), "r3": I(20, 1, 150), "r4": I(30, 1, 200), "s1": I(10, 1, 50), "s2": I(10, 1, 50), "s3": I(10, 1, 50), "s4": I(15, 1, 50), "signal": I(9, 1, 40)}, ["kst", "signal"], "oscillator", _kst)
add("coppock", {"roc1": I(14, 1, 100), "roc2": I(11, 1, 100), "wma": I(10, 1, 50)}, ["value"], "oscillator", _coppock)
add("connors_rsi", {"rsi": I(3, 2, 20), "streak": I(2, 2, 20), "rank": I(100, 5, 300)}, ["value"], "oscillator", _connors)
add("rvi", {"period": I(10, 1, 100)}, ["rvi", "signal"], "oscillator", _rvi)
add("rmi", {"period": I(14, 2, 100), "momentum": I(5, 1, 50)}, ["value"], "oscillator", _rmi)
add("stc", {"fast": I(23, 2, 100), "slow": I(50, 3, 200), "cycle": I(10, 2, 50)}, ["value"], "oscillator", _stc)
add("pmo", {"s1": I(35, 2, 100), "s2": I(20, 2, 80), "signal": I(10, 1, 40)}, ["pmo", "signal"], "oscillator", _pmo)
add("fisher", {"period": I(10, 2, 100)}, ["fisher", "trigger"], "oscillator", _fisher)
add("inertia", {"rvi_period": I(14, 2, 50), "reg_period": I(20, 2, 100)}, ["value"], "oscillator", _inertia)
add("squeeze_mom", {"bb": I(20, 2, 100), "kc": I(20, 2, 100), "bb_mult": F(2, 0.5, 4, 0.1), "kc_mult": F(1.5, 0.5, 4, 0.1)}, ["momentum", "squeeze_on"], "oscillator", _squeeze)
add("elder_ray", {"period": I(13, 2, 100)}, ["bull_power", "bear_power"], "oscillator", _elder_ray)
add("elder_impulse", {"ema": I(13, 2, 100), "fast": I(12, 2, 50), "slow": I(26, 3, 100), "signal": I(9, 2, 40)}, ["value"], "oscillator", _elder_impulse)
add("bop", {"smooth": I(14, 1, 100)}, ["value"], "oscillator", _bop)
add("qstick", {"period": I(8, 1, 100)}, ["value"], "oscillator", _qstick)
