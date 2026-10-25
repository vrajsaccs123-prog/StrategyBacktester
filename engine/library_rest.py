"""Volatility, volume, levels, statistics, cycles, structure, candles, and risk."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.indicator_impl import (
    F,
    G,
    I,
    N,
    V,
    add,
    atr,
    bollinger_frame,
    confirmed_swings,
    day_key,
    ema,
    hilbert,
    is_daily,
    ohlcv,
    ols,
    percent_rank,
    profile_levels,
    rsi,
    sma,
    true_range,
    wilder,
)


def _c(bars):
    return bars["close"].astype(float)


def _returns(close):
    return np.log(close / close.shift(1).replace(0, np.nan))


def _chaikin_vol(bars, raw):
    ema_n, roc_n = N(raw, "ema", 10), N(raw, "roc", 10)
    span = ema(bars["high"].astype(float) - bars["low"].astype(float), ema_n)
    prev = span.shift(roc_n).replace(0, np.nan)
    return V(100 * (span - prev) / prev)


def _tr(bars, _raw):
    return V(true_range(bars))


def _natr(bars, raw):
    return V(100 * atr(bars, N(raw, "period", 14)) / _c(bars).replace(0, np.nan))


def _bb_percent(bars, raw):
    n, std = N(raw, "period", 20), G(raw, "stddev", 2)
    bands = bollinger_frame(_c(bars), n, std)
    width = (bands["upper"] - bands["lower"]).replace(0, np.nan)
    return V((_c(bars) - bands["lower"]) / width)


def _bb_width(bars, raw):
    bands = bollinger_frame(_c(bars), N(raw, "period", 20), G(raw, "stddev", 2))
    return V((bands["upper"] - bands["lower"]) / bands["mid"].replace(0, np.nan))


def _keltner(bars, raw):
    mid = ema(_c(bars), N(raw, "ema", 20))
    width = G(raw, "mult", 2) * atr(bars, N(raw, "atr", 10))
    return pd.DataFrame({"upper": mid + width, "middle": mid, "lower": mid - width})


def _starc(bars, raw):
    mid = sma(_c(bars), N(raw, "sma", 6))
    width = G(raw, "mult", 2) * atr(bars, N(raw, "atr", 15))
    return pd.DataFrame({"upper": mid + width, "middle": mid, "lower": mid - width})


def _chandelier(bars, raw):
    n, mult = N(raw, "period", 22), G(raw, "mult", 3)
    width = mult * atr(bars, n)
    high, low, _close = bars["high"].astype(float), bars["low"].astype(float), _c(bars)
    return pd.DataFrame(
        {
            "long_exit": high.rolling(n, min_periods=n).max() - width,
            "short_exit": low.rolling(n, min_periods=n).min() + width,
        }
    )


def _hist_vol(bars, raw):
    n = N(raw, "period", 20)
    annual = G(raw, "annualization", 252)
    return V(_returns(_c(bars)).rolling(n, min_periods=n).std(ddof=1) * np.sqrt(annual))


def _stddev(bars, raw):
    n = N(raw, "period", 20)
    return V(_c(bars).rolling(n, min_periods=n).std(ddof=0))


def _rvi_vol(bars, raw):
    std_n, smooth = N(raw, "stddev", 10), N(raw, "smooth", 14)
    close = _c(bars)
    deviation = close.rolling(std_n, min_periods=std_n).std(ddof=0)
    up = wilder(deviation.where(close.diff() > 0, 0.0).fillna(0), smooth)
    down = wilder(deviation.where(close.diff() <= 0, 0.0).fillna(0), smooth)
    return V(100 * up / (up + down).replace(0, np.nan))


def _ulcer(bars, raw):
    n = N(raw, "period", 14)
    close = _c(bars)
    peak = close.rolling(n, min_periods=n).max().replace(0, np.nan)
    drawdown = (close - peak) / peak * 100
    return V(np.sqrt((drawdown**2).rolling(n, min_periods=n).mean()))


def _range_vol(term, period):
    return np.sqrt(term.rolling(period, min_periods=period).mean().clip(lower=0))


def _parkinson(bars, raw):
    n = N(raw, "period", 20)
    high, low = bars["high"].astype(float), bars["low"].astype(float)
    term = np.log(high / low.replace(0, np.nan)) ** 2
    return V(np.sqrt(term.rolling(n, min_periods=n).sum() / (4 * n * np.log(2))))


def _garman(bars, raw):
    n = N(raw, "period", 20)
    opened, high, low, close, _v = ohlcv(bars)
    term = 0.5 * np.log(high / low.replace(0, np.nan)) ** 2 - (2 * np.log(2) - 1) * np.log(close / opened.replace(0, np.nan)) ** 2
    return V(_range_vol(term, n))


def _rogers(bars, raw):
    n = N(raw, "period", 20)
    opened, high, low, close, _v = ohlcv(bars)
    term = np.log(high / close.replace(0, np.nan)) * np.log(high / opened.replace(0, np.nan))
    term = term + np.log(low / close.replace(0, np.nan)) * np.log(low / opened.replace(0, np.nan))
    return V(_range_vol(term, n))


def _yang(bars, raw):
    n = N(raw, "period", 20, minimum=2)
    opened, high, low, close, _v = ohlcv(bars)
    overnight = np.log(opened / close.shift(1).replace(0, np.nan))
    open_close = np.log(close / opened.replace(0, np.nan))
    rs = np.log(high / close.replace(0, np.nan)) * np.log(high / opened.replace(0, np.nan))
    rs = rs + np.log(low / close.replace(0, np.nan)) * np.log(low / opened.replace(0, np.nan))
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    variance = overnight.rolling(n, min_periods=n).var(ddof=1)
    variance = variance + k * open_close.rolling(n, min_periods=n).var(ddof=1)
    variance = variance + (1 - k) * rs.rolling(n, min_periods=n).mean()
    return V(np.sqrt(variance.clip(lower=0)))


def _mfm(bars):
    opened, high, low, close, volume = ohlcv(bars)
    return ((close - low) - (high - close)) / (high - low).replace(0, np.nan), volume, opened, high, low, close


def _obv(bars, _raw):
    _o, _h, _l, close, volume = ohlcv(bars)
    return V((np.sign(close.diff()).fillna(0) * volume).cumsum())


def _ad_line(bars, _raw):
    mfm, volume, *_rest = _mfm(bars)
    return V((mfm.fillna(0) * volume).cumsum())


def _cmf(bars, raw):
    n = N(raw, "period", 20)
    mfm, volume, *_rest = _mfm(bars)
    return V((mfm * volume).rolling(n, min_periods=n).sum() / volume.rolling(n, min_periods=n).sum().replace(0, np.nan))


def _chaikin_osc(bars, raw):
    mfm, volume, *_rest = _mfm(bars)
    line = (mfm.fillna(0) * volume).cumsum()
    return V(ema(line, N(raw, "fast", 3)) - ema(line, N(raw, "slow", 10)))


def _mfi(bars, raw):
    n = N(raw, "period", 14)
    _o, high, low, close, volume = ohlcv(bars)
    tp = (high + low + close) / 3
    flow = tp * volume
    delta = tp.diff()
    pos = flow.where(delta > 0, 0.0).rolling(n, min_periods=n).sum()
    neg = flow.where(delta < 0, 0.0).rolling(n, min_periods=n).sum()
    mfi = 100 - 100 / (1 + pos / neg.replace(0, np.nan))
    mfi = mfi.where(neg != 0, 100.0)
    mfi = mfi.where(~((pos == 0) & (neg == 0)))
    return V(mfi)


def _session_sums(bars):
    _o, high, low, close, volume = ohlcv(bars)
    tp = (high + low + close) / 3
    key = day_key(bars)
    sum_v = volume.groupby(key).cumsum()
    sum_pv = (tp * volume).groupby(key).cumsum()
    sum_p2 = (volume * tp * tp).groupby(key).cumsum()
    return tp, sum_v, sum_pv, sum_p2


def _vwap(bars, _raw):
    _tp, sum_v, sum_pv, _p2 = _session_sums(bars)
    return V(sum_pv / sum_v.replace(0, np.nan))


def _avwap(bars, raw):
    anchor = min(N(raw, "anchor_index", 0, minimum=0), max(len(bars) - 1, 0))
    _o, high, low, close, volume = ohlcv(bars)
    tp = (high + low + close) / 3
    pv = tp * volume
    vol = volume.copy()
    if anchor:
        pv = pv.copy()
        pv.iloc[:anchor] = 0
        vol.iloc[:anchor] = 0
    line = pv.cumsum() / vol.cumsum().replace(0, np.nan)
    if anchor:
        line.iloc[:anchor] = np.nan
    return V(line)


def _vwap_bands(bars, raw):
    m1, m2 = G(raw, "mult1", 1), G(raw, "mult2", 2)
    _tp, sum_v, sum_pv, sum_p2 = _session_sums(bars)
    vwap = sum_pv / sum_v.replace(0, np.nan)
    var = (sum_p2 / sum_v.replace(0, np.nan) - vwap**2).clip(lower=0)
    sigma = np.sqrt(var)
    return pd.DataFrame(
        {"upper1": vwap + m1 * sigma, "lower1": vwap - m1 * sigma, "upper2": vwap + m2 * sigma, "lower2": vwap - m2 * sigma}
    )


def _eom(bars, raw):
    n = N(raw, "period", 14)
    scale = G(raw, "scale", 1e8)
    if scale == 0:
        raise ValueError("scale must not be 0")
    high, low, volume = bars["high"].astype(float), bars["low"].astype(float), bars["volume"].astype(float)
    distance = (high + low) / 2 - (high.shift(1) + low.shift(1)) / 2
    box = (volume / scale) / (high - low).replace(0, np.nan)
    return V(sma(distance / box.replace(0, np.nan), n))


def _force(bars, raw):
    close, volume = _c(bars), bars["volume"].astype(float)
    return V(ema((close - close.shift(1)) * volume, N(raw, "period", 13)))


def _signed_volume_index(bars, rising: bool):
    close = _c(bars).to_numpy(dtype=float)
    volume = bars["volume"].to_numpy(dtype=float)
    out = np.full(len(close), np.nan)
    if len(close) == 0:
        return V(pd.Series(out, index=bars.index))
    out[0] = 1000.0
    for i in range(1, len(close)):
        update = volume[i] > volume[i - 1] if rising else volume[i] < volume[i - 1]
        if update and close[i - 1] != 0 and np.isfinite(close[i]):
            out[i] = out[i - 1] * (1 + (close[i] - close[i - 1]) / close[i - 1])
        else:
            out[i] = out[i - 1]
    return V(pd.Series(out, index=bars.index))


def _nvi(bars, _raw):
    return _signed_volume_index(bars, False)


def _pvi(bars, _raw):
    return _signed_volume_index(bars, True)


def _pvt(bars, _raw):
    close, volume = _c(bars), bars["volume"].astype(float)
    return V((volume * close.pct_change()).fillna(0).cumsum())


def _vroc(bars, raw):
    volume = bars["volume"].astype(float)
    n = N(raw, "period", 14)
    return V(100 * (volume - volume.shift(n)) / volume.shift(n).replace(0, np.nan))


def _kvo(bars, raw):
    fast, slow, signal_n = N(raw, "fast", 34), N(raw, "slow", 55), N(raw, "signal", 13)
    high, low, close, volume = bars["high"].astype(float), bars["low"].astype(float), _c(bars), bars["volume"].astype(float)
    trend = np.sign((high + low + close).diff()).replace(0, np.nan).ffill().fillna(1)
    dm = high - low
    dm_v = dm.to_numpy(dtype=float)
    trend_v = trend.to_numpy(dtype=float)
    cm = np.zeros(len(dm_v))
    cm[0] = dm_v[0]
    for i in range(1, len(dm_v)):
        cm[i] = cm[i - 1] + dm_v[i] if trend_v[i] == trend_v[i - 1] else dm_v[i] + dm_v[i - 1]
    cm_s = pd.Series(cm, index=bars.index).replace(0, np.nan)
    force = volume * trend * (2 * dm / cm_s - 1).abs() * 100
    line = ema(force, fast) - ema(force, slow)
    return pd.DataFrame({"kvo": line, "signal": ema(line, signal_n)})


def _vol_osc(bars, raw):
    volume = bars["volume"].astype(float)
    fast, slow = ema(volume, N(raw, "fast", 5)), ema(volume, N(raw, "slow", 10))
    return V(100 * (fast - slow) / slow.replace(0, np.nan))


def _twiggs(bars, raw):
    n = N(raw, "period", 21)
    high, low, close, volume = bars["high"].astype(float), bars["low"].astype(float), _c(bars), bars["volume"].astype(float)
    true_high = np.maximum(high, close.shift(1))
    true_low = np.minimum(low, close.shift(1))
    ad = ((close - true_low) - (true_high - close)) / (true_high - true_low).replace(0, np.nan)
    return V(ema(ad * volume, n) / ema(volume, n).replace(0, np.nan))


def _rolling_profile(bars, lookback, bins, weight_from_volume: bool):
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    volume = bars["volume"].to_numpy(dtype=float)
    typical = (high + low + close) / 3
    poc = np.full(len(close), np.nan)
    vah = np.full(len(close), np.nan)
    val = np.full(len(close), np.nan)
    for i in range(lookback - 1, len(close)):
        prices = typical[i - lookback + 1 : i + 1]
        weights = volume[i - lookback + 1 : i + 1] if weight_from_volume else np.ones(lookback)
        poc[i], vah[i], val[i] = profile_levels(prices, weights, bins)
    index = bars.index
    return pd.DataFrame(
        {"poc": pd.Series(poc, index=index), "vah": pd.Series(vah, index=index), "val": pd.Series(val, index=index)}
    )


def _vol_profile(bars, raw):
    return _rolling_profile(bars, N(raw, "lookback", 100), N(raw, "bins", 24), True)


def _mfi_bw(bars, _raw):
    high, low, volume = bars["high"].astype(float), bars["low"].astype(float), bars["volume"].astype(float)
    return V((high - low) / volume.replace(0, np.nan))


def _prior_hlc(bars):
    high, low, close = bars["high"].astype(float), bars["low"].astype(float), _c(bars)
    if is_daily(bars):
        return high.shift(1), low.shift(1), close.shift(1)
    grouped = pd.DataFrame({"high": high, "low": low, "close": close})
    daily = grouped.groupby(day_key(bars)).agg({"high": "max", "low": "min", "close": "last"}).shift(1)
    keys = pd.DatetimeIndex(bars.index.normalize())
    mapped = daily.reindex(keys)
    mapped.index = bars.index
    return mapped["high"], mapped["low"], mapped["close"]


def _pivot_classic(bars, _raw):
    high, low, close = _prior_hlc(bars)
    pp = (high + low + close) / 3
    span = high - low
    return pd.DataFrame(
        {
            "pp": pp,
            "r1": 2 * pp - low,
            "r2": pp + span,
            "r3": pp + 2 * span,
            "s1": 2 * pp - high,
            "s2": pp - span,
            "s3": pp - 2 * span,
        }
    )


def _pivot_fib(bars, _raw):
    high, low, close = _prior_hlc(bars)
    pp = (high + low + close) / 3
    span = high - low
    return pd.DataFrame(
        {
            "pp": pp,
            "r1": pp + 0.382 * span,
            "r2": pp + 0.618 * span,
            "r3": pp + span,
            "s1": pp - 0.382 * span,
            "s2": pp - 0.618 * span,
            "s3": pp - span,
        }
    )


def _pivot_camarilla(bars, _raw):
    high, low, close = _prior_hlc(bars)
    span = (high - low) * 1.1
    frame = {f"r{level}": close + span / divisor for level, divisor in ((1, 12), (2, 6), (3, 4), (4, 2))}
    frame.update({f"s{level}": close - span / divisor for level, divisor in ((1, 12), (2, 6), (3, 4), (4, 2))})
    return pd.DataFrame(frame)


def _pivot_woodie(bars, _raw):
    high, low, close = _prior_hlc(bars)
    pp = (high + low + 2 * close) / 4
    span = high - low
    return pd.DataFrame({"pp": pp, "r1": 2 * pp - low, "r2": pp + span, "s1": 2 * pp - high, "s2": pp - span})


def _pivot_demark(bars, _raw):
    opened = bars["open"].astype(float)
    high, low, close = _prior_hlc(bars)
    prev_open = opened.shift(1) if is_daily(bars) else opened.groupby(day_key(bars)).transform("first").groupby(day_key(bars)).shift(1)
    if not is_daily(bars):
        # Previous session's open is the first open of the prior day, already shifted via group.
        prev_open = opened.groupby(day_key(bars)).transform("first")
        prev_open = prev_open.groupby(day_key(bars)).shift(1)
        # groupby.shift on a transformed series is awkward; use the daily open.
        daily_open = opened.groupby(day_key(bars)).first().shift(1)
        prev_open = daily_open.reindex(pd.DatetimeIndex(bars.index.normalize()))
        prev_open.index = bars.index
    x = np.where(close < prev_open, high + 2 * low + close, np.where(close > prev_open, 2 * high + low + close, high + low + 2 * close))
    x = pd.Series(x, index=bars.index, dtype=float)
    return pd.DataFrame({"pp": x / 4, "r1": x / 2 - low, "s1": x / 2 - high})


def _fib_levels(bars, raw, ratios, names, extension: bool):
    n = N(raw, "period", 100)
    high = bars["high"].astype(float).rolling(n, min_periods=n).max()
    low = bars["low"].astype(float).rolling(n, min_periods=n).min()
    span = high - low
    frame = {}
    for name, ratio in zip(names, ratios):
        frame[name] = low + ratio * span if extension else high - ratio * span
    return pd.DataFrame(frame)


def _fib_retrace(bars, raw):
    return _fib_levels(bars, raw, [0.236, 0.382, 0.5, 0.618, 0.786], ["r236", "r382", "r500", "r618", "r786"], False)


def _fib_extension(bars, raw):
    return _fib_levels(bars, raw, [1.0, 1.272, 1.618, 2.0, 2.618], ["x100", "x127", "x162", "x200", "x262"], True)


def _swing(bars, raw):
    left, right = N(raw, "left", 2), N(raw, "right", 2)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    highs, lows = confirmed_swings(high, low, left, right)
    swing_high = pd.Series(np.nan, index=bars.index)
    swing_low = pd.Series(np.nan, index=bars.index)
    for idx, price in highs:
        swing_high.iloc[idx] = price
    for idx, price in lows:
        swing_low.iloc[idx] = price
    return pd.DataFrame({"swing_high": swing_high.ffill(), "swing_low": swing_low.ffill()})


def _zigzag(bars, raw):
    dev = G(raw, "deviation_pct", 5) / 100
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    out = np.full(len(close), np.nan)
    if len(close) == 0:
        return V(pd.Series(out, index=bars.index))
    last = close[0]
    mode = 0
    extreme = close[0]
    out[0] = close[0]
    for i in range(1, len(close)):
        if mode == 0:
            if high[i] >= last * (1 + dev):
                mode = 1
                extreme = high[i]
            elif low[i] <= last * (1 - dev):
                mode = -1
                extreme = low[i]
            out[i] = last
            continue
        if mode == 1:
            if high[i] > extreme:
                extreme = high[i]
            if extreme > 0 and low[i] <= extreme * (1 - dev):
                last = extreme
                mode = -1
                extreme = low[i]
            out[i] = last
        else:
            if low[i] < extreme:
                extreme = low[i]
            if extreme != 0 and high[i] >= extreme * (1 + dev):
                last = extreme
                mode = 1
                extreme = high[i]
            out[i] = last
    return V(pd.Series(out, index=bars.index))


def _regression(bars, raw):
    _slope, end, _r2, err = ols(_c(bars), N(raw, "period", 100))
    mult = G(raw, "mult", 2)
    return pd.DataFrame({"upper": end + mult * err, "middle": end, "lower": end - mult * err})


def _price_channel(bars, raw):
    n = N(raw, "period", 20)
    high, low = bars["high"].astype(float), bars["low"].astype(float)
    return pd.DataFrame(
        {"upper": high.rolling(n, min_periods=n).max(), "lower": low.rolling(n, min_periods=n).min()}
    )


def _typical(bars, _raw):
    high, low, close = bars["high"].astype(float), bars["low"].astype(float), _c(bars)
    return V((high + low + close) / 3)


def _median(bars, _raw):
    return V((bars["high"].astype(float) + bars["low"].astype(float)) / 2)


def _weighted_close(bars, _raw):
    high, low, close = bars["high"].astype(float), bars["low"].astype(float), _c(bars)
    return V((high + low + 2 * close) / 4)


def _ohlc4(bars, _raw):
    opened, high, low, close, _v = ohlcv(bars)
    return V((opened + high + low + close) / 4)


def _heikin(bars, _raw):
    opened, high, low, close, _v = ohlcv(bars)
    ha_close = (opened + high + low + close) / 4
    values_o = np.empty(len(bars), dtype=float)
    values_c = ha_close.to_numpy(dtype=float, copy=True)
    if len(values_o):
        values_o[0] = (float(opened.iloc[0]) + float(close.iloc[0])) / 2
        for i in range(1, len(values_o)):
            values_o[i] = (values_o[i - 1] + values_c[i - 1]) / 2
    ha_open = pd.Series(values_o, index=bars.index)
    ha_high = pd.concat([high, ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([low, ha_open, ha_close], axis=1).min(axis=1)
    return pd.DataFrame({"ha_open": ha_open, "ha_high": ha_high, "ha_low": ha_low, "ha_close": ha_close})


def _log_return(bars, raw):
    n = N(raw, "period", 1)
    close = _c(bars)
    return V(np.log(close / close.shift(n).replace(0, np.nan)))


def _pct_return(bars, raw):
    close = _c(bars)
    return V(close / close.shift(N(raw, "period", 1)).replace(0, np.nan) - 1)


def _gap(bars, _raw):
    opened, _h, _l, close, _v = ohlcv(bars)
    gap = opened - close.shift(1)
    return pd.DataFrame({"gap": gap, "gap_pct": gap / close.shift(1).replace(0, np.nan)})


def _linreg_slope(bars, raw):
    slope, _end, _r2, _err = ols(_c(bars), N(raw, "period", 14))
    return V(slope)


def _linreg_r2(bars, raw):
    _slope, _end, r2, _err = ols(_c(bars), N(raw, "period", 14))
    return V(r2)


def _linreg_angle(bars, raw):
    slope, _end, _r2, _err = ols(_c(bars), N(raw, "period", 14))
    return V(np.degrees(np.arctan(slope)))


def _zscore(bars, raw):
    n = N(raw, "period", 20)
    close = _c(bars)
    deviation = close.rolling(n, min_periods=n).std(ddof=0).replace(0, np.nan)
    return V((close - sma(close, n)) / deviation)


def _moment(close, period, power):
    returns = close.pct_change()

    def stat(window):
        mu = window.mean()
        sigma = window.std(ddof=0)
        if not np.isfinite(sigma) or sigma == 0:
            return np.nan
        return np.mean(((window - mu) / sigma) ** power)

    return returns.rolling(period, min_periods=period).apply(stat, raw=True)


def _skew(bars, raw):
    return V(_moment(_c(bars), N(raw, "period", 30), 3))


def _kurt(bars, raw):
    return V(_moment(_c(bars), N(raw, "period", 30), 4))


def _hurst_window(window):
    window = np.asarray(window, dtype=float)
    window = window[np.isfinite(window)]
    n = len(window)
    if n < 16:
        return np.nan
    sizes = []
    values = []
    for size in (n // 8, n // 4, n // 2, n):
        if size < 8:
            continue
        chunks = n // size
        scores = []
        for k in range(chunks):
            chunk = window[k * size : (k + 1) * size]
            dev = np.cumsum(chunk - chunk.mean())
            r = dev.max() - dev.min()
            s = chunk.std(ddof=1)
            if s > 0 and r > 0:
                scores.append(r / s)
        if scores:
            sizes.append(size)
            values.append(float(np.mean(scores)))
    if len(sizes) < 2:
        return np.nan
    slope, _intercept = np.polyfit(np.log(sizes), np.log(values), 1)
    return float(slope)


def _hurst(bars, raw):
    n = N(raw, "period", 100)
    return V(_returns(_c(bars)).rolling(n, min_periods=n).apply(_hurst_window, raw=True))


def _autocorr(bars, raw):
    lag, n = N(raw, "lag", 1), N(raw, "period", 50)
    returns = _c(bars).pct_change()
    return V(returns.rolling(n, min_periods=n).corr(returns.shift(lag)))


def _prank(bars, raw):
    return V(percent_rank(_c(bars), N(raw, "period", 100)))


def _er(bars, raw):
    n = N(raw, "period", 10)
    close = _c(bars)
    path = close.diff().abs().rolling(n, min_periods=n).sum().replace(0, np.nan)
    return V((close - close.shift(n)).abs() / path)


def _fdi(bars, raw):
    n = N(raw, "period", 30)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    out = np.full(len(high), np.nan)
    for i in range(n - 1, len(high)):
        segment_h = high[i - n + 1 : i + 1]
        segment_l = low[i - n + 1 : i + 1]
        price = (segment_h + segment_l) / 2
        span = price.max() - price.min()
        if span <= 0:
            continue
        norm = (price - price.min()) / span
        length = np.abs(np.diff(norm)).sum()
        if length <= 0:
            out[i] = 1.0
        else:
            out[i] = 1 + (np.log(length) + np.log(2)) / np.log(2 * n)
    return V(pd.Series(out, index=bars.index))


def _super_smoother(bars, raw):
    n = N(raw, "period", 10)
    a1 = np.exp(-np.sqrt(2) * np.pi / n)
    c2 = 2 * a1 * np.cos(np.sqrt(2) * np.pi / n)
    c3 = -(a1**2)
    c1 = 1 - c2 - c3
    values = _c(bars).to_numpy(dtype=float)
    out = np.zeros(len(values))
    for i, price in enumerate(values):
        prev = values[i - 1] if i else price
        ss1 = out[i - 1] if i else price
        ss2 = out[i - 2] if i > 1 else price
        out[i] = c1 * (price + prev) / 2 + c2 * ss1 + c3 * ss2
    return V(pd.Series(out, index=bars.index))


def _cyber(bars, raw):
    alpha = G(raw, "alpha", 0.07)
    price = _c(bars)
    smooth = (price + 2 * price.shift(1) + 2 * price.shift(2) + price.shift(3)) / 6
    values = smooth.to_numpy(dtype=float)
    cycle = np.zeros(len(values))
    for i in range(len(values)):
        if i < 2 or not np.isfinite(values[i]) or not np.isfinite(values[i - 1]) or not np.isfinite(values[i - 2]):
            cycle[i] = 0.0
            continue
        cycle[i] = (1 - alpha / 2) ** 2 * (values[i] - 2 * values[i - 1] + values[i - 2])
        cycle[i] += 2 * (1 - alpha) * cycle[i - 1] - (1 - alpha) ** 2 * cycle[i - 2]
    line = pd.Series(cycle, index=bars.index)
    return pd.DataFrame({"cycle": line, "signal": line.shift(1)})


def _itrend(bars, raw):
    alpha = G(raw, "alpha", 0.07)
    values = _c(bars).to_numpy(dtype=float)
    out = np.zeros(len(values))
    for i, price in enumerate(values):
        p1 = values[i - 1] if i else price
        p2 = values[i - 2] if i > 1 else price
        it1 = out[i - 1] if i else price
        it2 = out[i - 2] if i > 1 else price
        out[i] = (alpha - alpha**2 / 4) * price + 0.5 * alpha**2 * p1 - (alpha - 0.75 * alpha**2) * p2
        out[i] += 2 * (1 - alpha) * it1 - (1 - alpha) ** 2 * it2
    return V(pd.Series(out, index=bars.index))


def _cg(bars, raw):
    n = N(raw, "period", 10)
    values = _c(bars).to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for i in range(n - 1, len(values)):
        window = values[i - n + 1 : i + 1]
        if not np.isfinite(window).all() or window.sum() == 0:
            continue
        weights = np.arange(n, 0, -1, dtype=float)
        out[i] = -np.dot(weights, window) / window.sum()
    return V(pd.Series(out, index=bars.index))


def _from_hilbert(key):
    def compute(bars, raw):
        state = hilbert(_c(bars), G(raw, "fast_limit", 0.5) if "fast_limit" in raw or key == "mama" else 0.5, G(raw, "slow_limit", 0.05) if key == "mama" else 0.05)
        if key == "mama":
            return pd.DataFrame({"mama": state["mama"], "fama": state["fama"]})
        if key == "cycle_ht_dcperiod":
            return V(state["period"])
        if key == "ht_sine":
            return pd.DataFrame({"sine": state["sine"], "leadsine": state["leadsine"]})
        if key == "ht_trendmode":
            return V(state["trendmode"])
        return V(state["trendline"])

    return compute


def _opening_range(bars, raw):
    minutes = N(raw, "minutes", 30)
    high, low = bars["high"].astype(float), bars["low"].astype(float)
    if is_daily(bars) or not isinstance(bars.index, pd.DatetimeIndex):
        return pd.DataFrame({"orh": high, "orl": low})
    orh = np.full(len(bars), np.nan)
    orl = np.full(len(bars), np.nan)
    days = bars.index.normalize()
    for day in pd.unique(days):
        slots = np.flatnonzero(days == day)
        start = bars.index[slots[0]]
        deadline = start + pd.Timedelta(minutes=minutes)
        run_high = np.nan
        run_low = np.nan
        frozen_high = np.nan
        frozen_low = np.nan
        closed = False
        for idx in slots:
            if bars.index[idx] <= deadline:
                run_high = high.iloc[idx] if not np.isfinite(run_high) else max(run_high, high.iloc[idx])
                run_low = low.iloc[idx] if not np.isfinite(run_low) else min(run_low, low.iloc[idx])
                orh[idx] = run_high
                orl[idx] = run_low
            else:
                if not closed:
                    frozen_high, frozen_low = run_high, run_low
                    closed = True
                orh[idx] = frozen_high
                orl[idx] = frozen_low
    return pd.DataFrame({"orh": pd.Series(orh, index=bars.index), "orl": pd.Series(orl, index=bars.index)})


def _prev_day(bars, _raw):
    high, low, close = _prior_hlc(bars)
    return pd.DataFrame({"pdh": high, "pdl": low, "pdc": close})


def _gap_fill(bars, _raw):
    opened, high, low, close, _v = ohlcv(bars)
    if is_daily(bars):
        prior = close.shift(1)
        gap = opened - prior
        filled = ((gap > 0) & (low <= prior)) | ((gap < 0) & (high >= prior)) | (gap.abs() <= 1e-12)
        return pd.DataFrame({"gap_open": gap, "filled": filled.astype(float)})
    _pdh, _pdl, pdc = _prior_hlc(bars)
    first = opened.groupby(day_key(bars)).transform("first")
    gap = first - pdc
    crossed = ((gap > 0) & (low <= pdc)) | ((gap < 0) & (high >= pdc)) | (gap.abs() <= 1e-12)
    filled = crossed.groupby(day_key(bars)).cummax().astype(float)
    return pd.DataFrame({"gap_open": gap, "filled": filled})


def _order_block(bars, raw):
    lookback = N(raw, "lookback", 10)
    opened, high, low, close, _v = ohlcv(bars)
    width = atr(bars, 14).to_numpy(dtype=float)
    o = opened.to_numpy(dtype=float)
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)
    zone_high = np.full(len(c), np.nan)
    zone_low = np.full(len(c), np.nan)
    cur_high = np.nan
    cur_low = np.nan
    for i in range(lookback, len(c)):
        if not np.isfinite(width[i]):
            zone_high[i] = cur_high
            zone_low[i] = cur_low
            continue
        move = c[i] - c[i - lookback]
        if move > width[i]:
            for j in range(i - 1, i - lookback - 1, -1):
                if c[j] < o[j]:
                    cur_high, cur_low = h[j], l[j]
                    break
        elif move < -width[i]:
            for j in range(i - 1, i - lookback - 1, -1):
                if c[j] > o[j]:
                    cur_high, cur_low = h[j], l[j]
                    break
        zone_high[i] = cur_high
        zone_low[i] = cur_low
    return pd.DataFrame(
        {"zone_high": pd.Series(zone_high, index=bars.index), "zone_low": pd.Series(zone_low, index=bars.index)}
    )


def _fvg(bars, _raw):
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    top = np.full(len(high), np.nan)
    bottom = np.full(len(high), np.nan)
    cur_top = np.nan
    cur_bottom = np.nan
    bullish = False
    for i in range(2, len(high)):
        if low[i] > high[i - 2]:
            cur_top, cur_bottom, bullish = low[i], high[i - 2], True
        elif high[i] < low[i - 2]:
            cur_top, cur_bottom, bullish = low[i - 2], high[i], False
        if np.isfinite(cur_top):
            if bullish and low[i] <= cur_bottom:
                cur_top = cur_bottom = np.nan
            elif not bullish and high[i] >= cur_top:
                cur_top = cur_bottom = np.nan
        top[i] = cur_top
        bottom[i] = cur_bottom
    return pd.DataFrame(
        {"fvg_top": pd.Series(top, index=bars.index), "fvg_bottom": pd.Series(bottom, index=bars.index)}
    )


def _bos(bars, raw):
    swing = N(raw, "swing", 5)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    highs, lows = confirmed_swings(high, low, swing, swing)
    bos = np.zeros(len(close))
    choch = np.zeros(len(close))
    events = [(idx, price, 1) for idx, price in highs] + [(idx, price, -1) for idx, price in lows]
    events.sort()
    cursor = 0
    last_high = np.nan
    last_low = np.nan
    high_open = False
    low_open = False
    trend = 0
    for i in range(len(close)):
        while cursor < len(events) and events[cursor][0] == i:
            _idx, price, side = events[cursor]
            if side == 1:
                last_high = price
                high_open = True
            else:
                last_low = price
                low_open = True
            cursor += 1
        if high_open and np.isfinite(last_high) and close[i] > last_high:
            if trend < 0:
                choch[i] = 1
            else:
                bos[i] = 1
            trend = 1
            high_open = False
        elif low_open and np.isfinite(last_low) and close[i] < last_low:
            if trend > 0:
                choch[i] = -1
            else:
                bos[i] = -1
            trend = -1
            low_open = False
    return pd.DataFrame({"bos": pd.Series(bos, index=bars.index), "choch": pd.Series(choch, index=bars.index)})


def _adr(bars, raw):
    high, low = bars["high"].astype(float), bars["low"].astype(float)
    return V(sma(high - low, N(raw, "period", 14)))


def _rvol(bars, raw):
    volume = bars["volume"].astype(float)
    return V(volume / sma(volume, N(raw, "period", 20)).replace(0, np.nan))


def _tpo(bars, raw):
    return _rolling_profile(bars, max(N(raw, "period_minutes", 30), 8), 24, False)


def _parts(bars):
    opened, high, low, close, _v = ohlcv(bars)
    body = (close - opened).abs()
    upper = high - np.maximum(opened, close)
    lower = np.minimum(opened, close) - low
    bull = close > opened
    bear = close < opened
    return opened, high, low, close, body, upper, lower, bull, bear


def _dir(bull, bear):
    return V(pd.Series(np.where(bull.fillna(False), 1.0, np.where(bear.fillna(False), -1.0, 0.0)), index=bull.index))


def _flag(mask):
    return V(mask.fillna(False).astype(float))


def _doji(bars, raw):
    _o, high, low, _c, body, _u, _l, _bull, _bear = _parts(bars)
    pct = G(raw, "body_pct", 0.1)
    return _flag(body <= pct * (high - low))


def _hammer(bars, _raw):
    _o, _h, _l, close, body, _u, lower, _bull, _bear = _parts(bars)
    _o, _h, _l, _c, _body, upper, _lower, _bull, _bear = _parts(bars)
    shape = (lower >= 2 * body) & (upper <= body) & (body > 0)
    down = close < close.shift(5)
    up = close > close.shift(5)
    return _dir(shape & down, shape & up)


def _inv_hammer(bars, _raw):
    _o, _h, _l, close, body, upper, lower, _bull, _bear = _parts(bars)
    shape = (upper >= 2 * body) & (lower <= body) & (body > 0)
    down = close < close.shift(5)
    up = close > close.shift(5)
    return _dir(shape & down, shape & up)


def _engulfing(bars, _raw):
    opened, _h, _l, close, _body, _u, _lwr, bull, bear = _parts(bars)
    bullish = bear.shift(1, fill_value=False) & bull & (opened <= close.shift(1)) & (close >= opened.shift(1))
    bearish = bull.shift(1, fill_value=False) & bear & (opened >= close.shift(1)) & (close <= opened.shift(1))
    return _dir(bullish, bearish)


def _harami(bars, _raw):
    opened, _h, _l, close, body, _u, _lwr, bull, bear = _parts(bars)
    top = np.maximum(opened, close)
    bot = np.minimum(opened, close)
    inside = (top <= top.shift(1)) & (bot >= bot.shift(1)) & (body < body.shift(1))
    return _dir(inside & bull & bear.shift(1, fill_value=False), inside & bear & bull.shift(1, fill_value=False))


def _piercing(bars, _raw):
    opened, _h, _l, close, _body, _u, _lwr, bull, bear = _parts(bars)
    midpoint = (opened.shift(1) + close.shift(1)) / 2
    bullish = bear.shift(1, fill_value=False) & bull & (opened < close.shift(1)) & (close > midpoint) & (close < opened.shift(1))
    bearish = bull.shift(1, fill_value=False) & bear & (opened > close.shift(1)) & (close < midpoint) & (close > opened.shift(1))
    return _dir(bullish, bearish)


def _star(bars, _raw):
    opened, _h, _l, close, body, _u, _lwr, bull, bear = _parts(bars)
    span = (bars["high"].astype(float) - bars["low"].astype(float)).replace(0, np.nan)
    large = body.shift(2) > 0.5 * span.shift(2)
    small = body.shift(1) <= 0.3 * span.shift(1)
    midpoint = (opened.shift(2) + close.shift(2)) / 2
    morning = large & bear.shift(2, fill_value=False) & small & bull & (close > midpoint)
    evening = large & bull.shift(2, fill_value=False) & small & bear & (close < midpoint)
    return _dir(morning, evening)


def _soldiers(bars, _raw):
    opened, high, low, close, body, upper, lower, bull, bear = _parts(bars)
    span = (high - low).replace(0, np.nan)
    strong_up = bull & (body > 0.5 * span) & (upper <= body)
    strong_down = bear & (body > 0.5 * span) & (lower <= body)
    higher = (close > close.shift(1)) & (close.shift(1) > close.shift(2))
    lower_c = (close < close.shift(1)) & (close.shift(1) < close.shift(2))
    inside_prev = (opened <= np.maximum(opened.shift(1), close.shift(1))) & (opened >= np.minimum(opened.shift(1), close.shift(1)))
    inside_prev2 = (opened.shift(1) <= np.maximum(opened.shift(2), close.shift(2))) & (
        opened.shift(1) >= np.minimum(opened.shift(2), close.shift(2))
    )
    up = strong_up & strong_up.shift(1, fill_value=False) & strong_up.shift(2, fill_value=False) & higher & inside_prev & inside_prev2
    down = strong_down & strong_down.shift(1, fill_value=False) & strong_down.shift(2, fill_value=False) & lower_c & inside_prev & inside_prev2
    return _dir(up, down)


def _marubozu(bars, _raw):
    _o, high, low, _c, _body, upper, lower, bull, bear = _parts(bars)
    span = (high - low).replace(0, np.nan)
    shape = (upper <= 0.05 * span) & (lower <= 0.05 * span) & ((high - low) > 0)
    return _dir(shape & bull, shape & bear)


def _spinning(bars, _raw):
    _o, high, low, _c, body, upper, lower, _bull, _bear = _parts(bars)
    span = high - low
    return _flag((body <= 0.3 * span) & (upper > body) & (lower > body) & (span > 0))


def _dragonfly(bars, raw):
    _o, high, low, _c, body, upper, lower, _bull, _bear = _parts(bars)
    pct = G(raw, "body_pct", 0.1) if "body_pct" in raw else 0.1
    doji = body <= pct * (high - low)
    return _dir(doji & (lower >= 2 * (high - low).clip(lower=1e-12) * 0 + lower) & (lower > upper * 2) & (upper <= body + 1e-12), doji & (upper > lower * 2) & (lower <= body + 1e-12))


def _tweezer(bars, raw):
    tol = G(raw, "tol", 0.001)
    high, low, close = bars["high"].astype(float), bars["low"].astype(float), _c(bars)
    top = ((high - high.shift(1)).abs() <= tol * high) & (close < close.shift(2))
    bottom = ((low - low.shift(1)).abs() <= tol * low) & (close > close.shift(2))
    return _dir(bottom, top)


def _inside_outside(bars, _raw):
    high, low = bars["high"].astype(float), bars["low"].astype(float)
    inside = (high <= high.shift(1)) & (low >= low.shift(1))
    outside = (high > high.shift(1)) & (low < low.shift(1))
    return pd.DataFrame({"inside": inside.fillna(False).astype(float), "outside": outside.fillna(False).astype(float)})


def _three_inside(bars, _raw):
    harami = _harami(bars, {})["value"]
    close = _c(bars)
    opened = bars["open"].astype(float)
    bullish = (harami.shift(1) > 0) & (close > np.maximum(opened.shift(2), close.shift(2)))
    bearish = (harami.shift(1) < 0) & (close < np.minimum(opened.shift(2), close.shift(2)))
    return _dir(bullish, bearish)


def _kicker(bars, _raw):
    opened, _h, _l, close, _body, _u, _lwr, bull, bear = _parts(bars)
    bullish = bear.shift(1, fill_value=False) & bull & (opened >= opened.shift(1)) & (close > opened)
    bearish = bull.shift(1, fill_value=False) & bear & (opened <= opened.shift(1)) & (close < opened)
    return _dir(bullish, bearish)


def _breakout(bars, raw):
    n = N(raw, "period", 20)
    high, low, close = bars["high"].astype(float), bars["low"].astype(float), _c(bars)
    prior_high = high.shift(1).rolling(n, min_periods=n).max()
    prior_low = low.shift(1).rolling(n, min_periods=n).min()
    signal = pd.Series(np.where(close > prior_high, 1.0, np.where(close < prior_low, -1.0, 0.0)), index=bars.index)
    return V(signal.where(prior_high.notna(), 0.0))


def _td_sequential(bars, raw):
    setup_len = N(raw, "setup", 9)
    countdown_len = N(raw, "countdown", 13)
    close = bars["close"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    setup = np.zeros(len(close))
    countdown = np.zeros(len(close))
    direction = 0
    count = 0
    cd_dir = 0
    cd_count = 0
    for i in range(4, len(close)):
        buy = close[i] < close[i - 4]
        sell = close[i] > close[i - 4]
        if buy and not sell:
            count = count + 1 if direction == -1 else 1
            direction = -1
        elif sell and not buy:
            count = count + 1 if direction == 1 else 1
            direction = 1
        else:
            count = 0
            direction = 0
        setup[i] = count * direction
        if count == setup_len:
            cd_dir = direction
            cd_count = 0
            count = 0
            direction = 0
        if cd_dir == -1 and i >= 2 and close[i] <= low[i - 2]:
            cd_count += 1
        elif cd_dir == 1 and i >= 2 and close[i] >= high[i - 2]:
            cd_count += 1
        countdown[i] = cd_count * cd_dir
        if cd_count >= countdown_len:
            cd_dir = 0
            cd_count = 0
    return pd.DataFrame({"setup": pd.Series(setup, index=bars.index), "countdown": pd.Series(countdown, index=bars.index)})


def _td_setup(bars, raw):
    lookback = N(raw, "lookback", 4)
    close = bars["close"].to_numpy(dtype=float)
    out = np.zeros(len(close))
    count = 0
    for i in range(lookback, len(close)):
        if close[i] < close[i - lookback]:
            count = count + 1 if count > 0 else 1
        elif close[i] > close[i - lookback]:
            count = count - 1 if count < 0 else -1
        else:
            count = 0
        out[i] = count
    return V(pd.Series(out, index=bars.index))


def _streak(bars, _raw):
    close = bars["close"].to_numpy(dtype=float)
    out = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            out[i] = out[i - 1] + 1 if out[i - 1] > 0 else 1
        elif close[i] < close[i - 1]:
            out[i] = out[i - 1] - 1 if out[i - 1] < 0 else -1
    return V(pd.Series(out, index=bars.index))


def _ibs(bars, _raw):
    high, low, close = bars["high"].astype(float), bars["low"].astype(float), _c(bars)
    return V((close - low) / (high - low).replace(0, np.nan))


def _pct_ma(bars, raw):
    close = _c(bars)
    average = sma(close, N(raw, "period", 50))
    return V(100 * (close - average) / average.replace(0, np.nan))


def _rsi2(bars, raw):
    return V(rsi(_c(bars), N(raw, "period", 2)))


def _kalman(bars, raw):
    q, r = G(raw, "q", 1e-5), G(raw, "r", 0.001)
    values = _c(bars).to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) == 0:
        return V(pd.Series(out, index=bars.index))
    x = values[0]
    p = 1.0
    for i, measurement in enumerate(values):
        if not np.isfinite(measurement):
            out[i] = x
            continue
        p = p + q
        gain = p / (p + r) if p + r else 0.0
        x = x + gain * (measurement - x)
        p = (1 - gain) * p
        out[i] = x
    return V(pd.Series(out, index=bars.index))


def _sharpe(bars, raw):
    n = N(raw, "period", 60)
    rf = G(raw, "rf", 0) / 252
    returns = _c(bars).pct_change()
    deviation = returns.rolling(n, min_periods=n).std(ddof=1).replace(0, np.nan)
    return V((returns.rolling(n, min_periods=n).mean() - rf) / deviation * np.sqrt(252))


def _sortino(bars, raw):
    n = N(raw, "period", 60)
    mar = G(raw, "mar", 0) / 252
    returns = _c(bars).pct_change()
    excess = returns - mar
    downside = np.sqrt((np.minimum(excess, 0) ** 2).rolling(n, min_periods=n).mean()).replace(0, np.nan)
    return V(excess.rolling(n, min_periods=n).mean() / downside * np.sqrt(252))


def _drawdown(bars, _raw):
    close = _c(bars)
    peak = close.cummax().replace(0, np.nan)
    dd = (close - peak) / peak
    return pd.DataFrame({"dd": dd, "max_dd": dd.cummin()})


def _var(bars, raw):
    n = N(raw, "period", 250)
    confidence = min(max(G(raw, "confidence", 0.95), 0.5), 0.999)
    returns = _c(bars).pct_change()
    return V(-returns.rolling(n, min_periods=n).quantile(1 - confidence))


def _cvar(bars, raw):
    n = N(raw, "period", 250)
    confidence = min(max(G(raw, "confidence", 0.95), 0.5), 0.999)

    def stat(window):
        threshold = np.quantile(window, 1 - confidence)
        tail = window[window <= threshold]
        if len(tail) == 0:
            return np.nan
        return -float(tail.mean())

    return V(_c(bars).pct_change().rolling(n, min_periods=n).apply(stat, raw=True))


def _calmar(bars, raw):
    years = N(raw, "years", 3)
    width = max(int(round(years * 252)), 2)
    values = _c(bars).to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for i in range(width - 1, len(values)):
        window = values[i - width + 1 : i + 1]
        if not np.isfinite(window).all() or window[0] <= 0 or window[-1] <= 0:
            continue
        cagr = (window[-1] / window[0]) ** (252 / (width - 1)) - 1
        peak = np.maximum.accumulate(window)
        drawdown = (window - peak) / np.where(peak == 0, np.nan, peak)
        max_dd = float(np.nanmin(drawdown))
        if max_dd < 0:
            out[i] = cagr / abs(max_dd)
    return V(pd.Series(out, index=bars.index))


def _vol_target(bars, raw):
    n = N(raw, "period", 20)
    target = G(raw, "target_vol", 0.15)
    realized = _c(bars).pct_change().rolling(n, min_periods=n).std(ddof=1).replace(0, np.nan) * np.sqrt(252)
    return V(target / realized)


def _atr_position(bars, raw):
    width = atr(bars, N(raw, "period", 14)).replace(0, np.nan)
    shares = G(raw, "equity", 100000) * G(raw, "risk_pct", 0.01) / (G(raw, "atr_mult", 2) * width)
    return V(shares)


def _seasonality(bars, _raw):
    close = _c(bars)
    if not isinstance(bars.index, pd.DatetimeIndex):
        return V(pd.Series(np.nan, index=bars.index))
    returns = close.pct_change()
    bucket = pd.Series(bars.index.day_of_week, index=bars.index)
    average = returns.groupby(bucket).transform(lambda series: series.expanding().mean().shift(1))
    return V(average)


def _time_filter(bars, raw):
    start = N(raw, "start_hour", 9, minimum=0) * 60 + N(raw, "start_minute", 15, minimum=0)
    end = N(raw, "end_hour", 15, minimum=0) * 60 + N(raw, "end_minute", 30, minimum=0)
    if is_daily(bars) or not isinstance(bars.index, pd.DatetimeIndex):
        return V(pd.Series(1.0, index=bars.index))
    minutes = bars.index.hour * 60 + bars.index.minute
    if start <= end:
        flag = (minutes >= start) & (minutes <= end)
    else:
        flag = (minutes >= start) | (minutes <= end)
    return V(pd.Series(flag.astype(float), index=bars.index))


def _mom_12_1(bars, raw):
    close = _c(bars)
    skip, lookback = N(raw, "skip", 21, minimum=0), N(raw, "lookback", 252)
    return V(close.shift(skip) / close.shift(lookback).replace(0, np.nan) - 1)


def _low_vol(bars, raw):
    n = N(raw, "period", 252)
    return V(_c(bars).pct_change().rolling(n, min_periods=n).std(ddof=1) * np.sqrt(252))


_PIVOT = ["pp", "r1", "r2", "r3", "s1", "s2", "s3"]

add("chaikin_vol", {"ema": I(10, 1, 100), "roc": I(10, 1, 100)}, ["value"], "oscillator", _chaikin_vol)
add("tr", {}, ["value"], "oscillator", _tr)
add("natr", {"period": I(14, 2, 100)}, ["value"], "oscillator", _natr)
add("bb_percent_b", {"period": I(20, 2, 400), "stddev": F(2, 0.5, 4, 0.1)}, ["value"], "oscillator", _bb_percent)
add("bb_width", {"period": I(20, 2, 400), "stddev": F(2, 0.5, 4, 0.1)}, ["value"], "oscillator", _bb_width)
add("keltner", {"ema": I(20, 2, 400), "atr": I(10, 2, 100), "mult": F(2, 0.5, 6, 0.1)}, ["upper", "middle", "lower"], "overlay", _keltner)
add("starc", {"sma": I(6, 2, 100), "atr": I(15, 2, 100), "mult": F(2, 0.5, 6, 0.1)}, ["upper", "middle", "lower"], "overlay", _starc)
add("chandelier", {"period": I(22, 2, 200), "mult": F(3, 0.5, 8, 0.1)}, ["long_exit", "short_exit"], "overlay", _chandelier)
add("hist_vol", {"period": I(20, 2, 200), "annualization": I(252, 1, 365)}, ["value"], "oscillator", _hist_vol)
add("stddev", {"period": I(20, 2, 200)}, ["value"], "oscillator", _stddev)
add("rvi_vol", {"stddev": I(10, 2, 50), "smooth": I(14, 2, 50)}, ["value"], "oscillator", _rvi_vol)
add("ulcer", {"period": I(14, 2, 200)}, ["value"], "oscillator", _ulcer)
add("parkinson", {"period": I(20, 2, 200)}, ["value"], "oscillator", _parkinson)
add("garman_klass", {"period": I(20, 2, 200)}, ["value"], "oscillator", _garman)
add("rogers_satchell", {"period": I(20, 2, 200)}, ["value"], "oscillator", _rogers)
add("yang_zhang", {"period": I(20, 2, 200)}, ["value"], "oscillator", _yang)
add("obv", {}, ["value"], "oscillator", _obv)
add("ad_line", {}, ["value"], "oscillator", _ad_line)
add("cmf", {"period": I(20, 2, 200)}, ["value"], "oscillator", _cmf)
add("chaikin_osc", {"fast": I(3, 1, 40), "slow": I(10, 2, 80)}, ["value"], "oscillator", _chaikin_osc)
add("mfi", {"period": I(14, 2, 100)}, ["value"], "oscillator", _mfi)
add("vwap", {}, ["value"], "overlay", _vwap)
add("avwap", {"anchor_index": I(0, 0, 10000)}, ["value"], "overlay", _avwap)
add("vwap_bands", {"mult1": F(1, 0.5, 5, 0.1), "mult2": F(2, 0.5, 6, 0.1)}, ["upper1", "lower1", "upper2", "lower2"], "overlay", _vwap_bands)
add("eom", {"period": I(14, 1, 100), "scale": F(1e8, 1e4, 1e12, 1e6)}, ["value"], "oscillator", _eom)
add("force_index", {"period": I(13, 1, 100)}, ["value"], "oscillator", _force)
add("nvi", {}, ["value"], "oscillator", _nvi)
add("pvi", {}, ["value"], "oscillator", _pvi)
add("pvt", {}, ["value"], "oscillator", _pvt)
add("vroc", {"period": I(14, 1, 200)}, ["value"], "oscillator", _vroc)
add("kvo", {"fast": I(34, 2, 100), "slow": I(55, 3, 200), "signal": I(13, 1, 40)}, ["kvo", "signal"], "oscillator", _kvo)
add("vol_osc", {"fast": I(5, 1, 50), "slow": I(10, 2, 100)}, ["value"], "oscillator", _vol_osc)
add("twiggs", {"period": I(21, 2, 100)}, ["value"], "oscillator", _twiggs)
add("vol_profile", {"bins": I(24, 4, 80), "lookback": I(100, 10, 400)}, ["poc", "vah", "val"], "overlay", _vol_profile)
add("mfi_bw", {}, ["value"], "oscillator", _mfi_bw)
add("pivot_classic", {}, _PIVOT, "overlay", _pivot_classic)
add("pivot_fib", {}, _PIVOT, "overlay", _pivot_fib)
add("pivot_camarilla", {}, ["r1", "r2", "r3", "r4", "s1", "s2", "s3", "s4"], "overlay", _pivot_camarilla)
add("pivot_woodie", {}, ["pp", "r1", "r2", "s1", "s2"], "overlay", _pivot_woodie)
add("pivot_demark", {}, ["pp", "r1", "s1"], "overlay", _pivot_demark)
add("fib_retrace", {"period": I(100, 5, 400)}, ["r236", "r382", "r500", "r618", "r786"], "overlay", _fib_retrace)
add("fib_extension", {"period": I(100, 5, 400)}, ["x100", "x127", "x162", "x200", "x262"], "overlay", _fib_extension)
add("swing_hl", {"left": I(2, 1, 20), "right": I(2, 1, 20)}, ["swing_high", "swing_low"], "overlay", _swing)
add("zigzag", {"deviation_pct": F(5, 0.5, 30, 0.5)}, ["value"], "overlay", _zigzag)
add("regression_channel", {"period": I(100, 5, 400), "mult": F(2, 0.5, 4, 0.1)}, ["upper", "middle", "lower"], "overlay", _regression)
add("price_channel", {"period": I(20, 2, 400)}, ["upper", "lower"], "overlay", _price_channel)
add("typical_price", {}, ["value"], "overlay", _typical)
add("median_price", {}, ["value"], "overlay", _median)
add("weighted_close", {}, ["value"], "overlay", _weighted_close)
add("ohlc4", {}, ["value"], "overlay", _ohlc4)
add("heikin_ashi", {}, ["ha_open", "ha_high", "ha_low", "ha_close"], "overlay", _heikin)
add("log_return", {"period": I(1, 1, 200)}, ["value"], "oscillator", _log_return)
add("pct_return", {"period": I(1, 1, 200)}, ["value"], "oscillator", _pct_return)
add("gap", {}, ["gap", "gap_pct"], "oscillator", _gap)
add("linreg_slope", {"period": I(14, 2, 200)}, ["value"], "oscillator", _linreg_slope)
add("linreg_r2", {"period": I(14, 2, 200)}, ["value"], "oscillator", _linreg_r2)
add("linreg_angle", {"period": I(14, 2, 200)}, ["value"], "oscillator", _linreg_angle)
add("zscore", {"period": I(20, 2, 200)}, ["value"], "oscillator", _zscore)
add("skew", {"period": I(30, 5, 200)}, ["value"], "oscillator", _skew)
add("kurt", {"period": I(30, 5, 200)}, ["value"], "oscillator", _kurt)
add("hurst", {"period": I(100, 32, 400)}, ["value"], "oscillator", _hurst)
add("autocorr", {"lag": I(1, 1, 20), "period": I(50, 10, 200)}, ["value"], "oscillator", _autocorr)
add("percent_rank", {"period": I(100, 5, 400)}, ["value"], "oscillator", _prank)
add("er", {"period": I(10, 2, 100)}, ["value"], "oscillator", _er)
add("fractal_dim", {"period": I(30, 8, 200)}, ["value"], "oscillator", _fdi)
add("super_smoother", {"period": I(10, 2, 100)}, ["value"], "overlay", _super_smoother)
add("cyber_cycle", {"alpha": F(0.07, 0.01, 0.5, 0.01)}, ["cycle", "signal"], "oscillator", _cyber)
add("itrend", {"alpha": F(0.07, 0.01, 0.5, 0.01)}, ["value"], "overlay", _itrend)
add("stoch_cg", {"period": I(10, 2, 100)}, ["value"], "oscillator", _cg)
add("cycle_ht_dcperiod", {}, ["value"], "oscillator", _from_hilbert("cycle_ht_dcperiod"))
add("ht_sine", {}, ["sine", "leadsine"], "oscillator", _from_hilbert("ht_sine"))
add("ht_trendmode", {}, ["value"], "oscillator", _from_hilbert("ht_trendmode"))
add("ht_trendline", {}, ["value"], "overlay", _from_hilbert("ht_trendline"))
add("mama", {"fast_limit": F(0.5, 0.01, 0.99, 0.01), "slow_limit": F(0.05, 0.01, 0.99, 0.01)}, ["mama", "fama"], "overlay", _from_hilbert("mama"))
add("opening_range", {"minutes": I(30, 1, 240)}, ["orh", "orl"], "overlay", _opening_range)
add("prev_day_levels", {}, ["pdh", "pdl", "pdc"], "overlay", _prev_day)
add("gap_fill", {}, ["gap_open", "filled"], "oscillator", _gap_fill)
add("order_block", {"lookback": I(10, 2, 50)}, ["zone_high", "zone_low"], "overlay", _order_block)
add("fvg", {}, ["fvg_top", "fvg_bottom"], "overlay", _fvg)
add("bos_choch", {"swing": I(5, 1, 20)}, ["bos", "choch"], "oscillator", _bos)
add("adr", {"period": I(14, 1, 100)}, ["value"], "oscillator", _adr)
add("rel_volume", {"period": I(20, 2, 200)}, ["value"], "oscillator", _rvol)
add("market_profile_tpo", {"period_minutes": I(30, 8, 240)}, ["poc", "vah", "val"], "overlay", _tpo)
add("candle_doji", {"body_pct": F(0.1, 0.01, 0.5, 0.01)}, ["value"], "oscillator", _doji)
add("candle_hammer", {}, ["value"], "oscillator", _hammer)
add("candle_inv_hammer", {}, ["value"], "oscillator", _inv_hammer)
add("candle_engulfing", {}, ["value"], "oscillator", _engulfing)
add("candle_harami", {}, ["value"], "oscillator", _harami)
add("candle_piercing", {}, ["value"], "oscillator", _piercing)
add("candle_star", {}, ["value"], "oscillator", _star)
add("candle_soldiers", {}, ["value"], "oscillator", _soldiers)
add("candle_marubozu", {}, ["value"], "oscillator", _marubozu)
add("candle_spinning", {}, ["value"], "oscillator", _spinning)
add("candle_dragonfly", {"body_pct": F(0.1, 0.01, 0.5, 0.01)}, ["value"], "oscillator", _dragonfly)
add("candle_tweezer", {"tol": F(0.001, 0.0001, 0.05, 0.0005)}, ["value"], "oscillator", _tweezer)
add("candle_inside_outside", {}, ["inside", "outside"], "oscillator", _inside_outside)
add("candle_three_inside", {}, ["value"], "oscillator", _three_inside)
add("candle_kicker", {}, ["value"], "oscillator", _kicker)
add("chart_breakout", {"period": I(20, 2, 200)}, ["value"], "oscillator", _breakout)
add("td_sequential", {"setup": I(9, 2, 20), "countdown": I(13, 2, 30)}, ["setup", "countdown"], "oscillator", _td_sequential)
add("td_setup", {"lookback": I(4, 1, 20)}, ["value"], "oscillator", _td_setup)
add("consec_bars", {}, ["value"], "oscillator", _streak)
add("ibs", {}, ["value"], "oscillator", _ibs)
add("pct_from_ma", {"period": I(50, 2, 400)}, ["value"], "oscillator", _pct_ma)
add("rsi2", {"period": I(2, 2, 20)}, ["value"], "oscillator", _rsi2)
add("kalman_filter", {"q": F(1e-5, 1e-8, 1, 1e-5), "r": F(0.001, 1e-6, 10, 0.001)}, ["value"], "overlay", _kalman)
add("sharpe_roll", {"period": I(60, 5, 400), "rf": F(0, 0, 0.2, 0.01)}, ["value"], "oscillator", _sharpe)
add("sortino_roll", {"period": I(60, 5, 400), "mar": F(0, 0, 0.2, 0.01)}, ["value"], "oscillator", _sortino)
add("drawdown", {}, ["dd", "max_dd"], "oscillator", _drawdown)
add("var_hist", {"period": I(250, 20, 600), "confidence": F(0.95, 0.8, 0.99, 0.01)}, ["value"], "oscillator", _var)
add("cvar", {"period": I(250, 20, 600), "confidence": F(0.95, 0.8, 0.99, 0.01)}, ["value"], "oscillator", _cvar)
add("calmar", {"years": I(3, 1, 10)}, ["value"], "oscillator", _calmar)
add("vol_target", {"target_vol": F(0.15, 0.01, 1, 0.01), "period": I(20, 5, 200)}, ["value"], "oscillator", _vol_target)
add("atr_position", {"risk_pct": F(0.01, 0.001, 0.1, 0.001), "atr_mult": F(2, 0.5, 6, 0.1), "period": I(14, 2, 100), "equity": F(100000, 1000, 1e7, 1000)}, ["value"], "oscillator", _atr_position)
add("seasonality_dow", {}, ["value"], "oscillator", _seasonality)
add("time_filter", {"start_hour": I(9, 0, 23), "start_minute": I(15, 0, 59), "end_hour": I(15, 0, 23), "end_minute": I(30, 0, 59)}, ["value"], "oscillator", _time_filter)
add("momentum_12_1", {"lookback": I(252, 40, 600), "skip": I(21, 0, 60)}, ["value"], "oscillator", _mom_12_1)
add("low_vol_factor", {"period": I(252, 20, 600)}, ["value"], "oscillator", _low_vol)
