"""Price indicators used by rule strategies and the chart."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.models import ParamSpec

OVERLAY_TYPES = {"sma", "ema", "bollinger", "donchian"}


def _wilder(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


def _length(params: dict, name: str, default: int) -> int:
    value = params.get(name, default)
    if isinstance(value, dict):
        value = value.get("value", default)
    length = int(round(float(value)))
    if length < 1:
        raise ValueError(f"{name} must be at least 1")
    return length


def sma(close: pd.Series, length: int) -> pd.Series:
    return close.rolling(length, min_periods=length).mean()


def ema(close: pd.Series, length: int) -> pd.Series:
    return close.ewm(span=length, min_periods=length, adjust=False).mean()


def rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = _wilder(gain, length)
    avg_loss = _wilder(loss, length)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    values = 100 - (100 / (1 + rs))
    values = values.where(avg_loss != 0, 100.0)
    values = values.where(~((avg_gain == 0) & (avg_loss == 0)))
    return values


def macd(close: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    signal_line = line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": signal_line, "hist": line - signal_line})


def bollinger(close: pd.Series, length: int, std_mult: float) -> pd.DataFrame:
    mid = sma(close, length)
    deviation = close.rolling(length, min_periods=length).std(ddof=0)
    return pd.DataFrame(
        {
            "mid": mid,
            "upper": mid + std_mult * deviation,
            "lower": mid - std_mult * deviation,
        }
    )


def atr(bars: pd.DataFrame, length: int) -> pd.Series:
    prev_close = bars["close"].shift(1)
    true_range = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - prev_close).abs(),
            (bars["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return _wilder(true_range, length)


def stochastic(bars: pd.DataFrame, k_length: int, d_length: int) -> pd.DataFrame:
    lowest = bars["low"].rolling(k_length, min_periods=k_length).min()
    highest = bars["high"].rolling(k_length, min_periods=k_length).max()
    span = (highest - lowest).replace(0, np.nan)
    k = 100 * (bars["close"] - lowest) / span
    d = k.rolling(d_length, min_periods=d_length).mean()
    return pd.DataFrame({"k": k, "d": d})


def adx(bars: pd.DataFrame, length: int) -> pd.DataFrame:
    up_move = bars["high"].diff()
    down_move = -bars["low"].diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=bars.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=bars.index,
    )
    true_range = atr(bars, length)
    plus_di = 100 * _wilder(plus_dm, length) / true_range.replace(0, np.nan)
    minus_di = 100 * _wilder(minus_dm, length) / true_range.replace(0, np.nan)
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denom
    return pd.DataFrame({"adx": _wilder(dx, length), "plus_di": plus_di, "minus_di": minus_di})


def donchian(bars: pd.DataFrame, length: int) -> pd.DataFrame:
    # Prior window only, so a breakout can compare this bar's close to the channel.
    upper = bars["high"].rolling(length, min_periods=length).max().shift(1)
    lower = bars["low"].rolling(length, min_periods=length).min().shift(1)
    return pd.DataFrame({"upper": upper, "lower": lower, "mid": (upper + lower) / 2})


def roc(close: pd.Series, length: int) -> pd.Series:
    return (close / close.shift(length) - 1) * 100


def _plain(params: dict) -> dict:
    clean = {}
    for key, value in params.items():
        clean[key] = value.get("value", value) if isinstance(value, dict) else value
    return clean


def compute_indicator(kind: str, bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Return one column per output. Single-output indicators use column 'value'."""

    raw = _plain(params)
    if kind in LIBRARY:
        lib = LIBRARY[kind]
        frame = lib.compute(bars, raw).replace([np.inf, -np.inf], np.nan)
        if list(frame.columns) != list(lib.outputs):
            raise ValueError(f"{kind} returned {list(frame.columns)}, expected {list(lib.outputs)}")
        return frame
    close = bars["close"]
    if kind == "sma":
        return pd.DataFrame({"value": sma(close, _length(raw, "length", 20))})
    if kind == "ema":
        return pd.DataFrame({"value": ema(close, _length(raw, "length", 20))})
    if kind == "rsi":
        return pd.DataFrame({"value": rsi(close, _length(raw, "length", 14))})
    if kind == "macd":
        return macd(
            close,
            _length(raw, "fast", 12),
            _length(raw, "slow", 26),
            _length(raw, "signal", 9),
        )
    if kind == "bollinger":
        std_mult = float(raw.get("std", 2.0))
        return bollinger(close, _length(raw, "length", 20), std_mult)
    if kind == "atr":
        return pd.DataFrame({"value": atr(bars, _length(raw, "length", 14))})
    if kind == "stochastic":
        return stochastic(bars, _length(raw, "k", 14), _length(raw, "d", 3))
    if kind == "adx":
        return adx(bars, _length(raw, "length", 14))
    if kind == "donchian":
        return donchian(bars, _length(raw, "length", 20))
    if kind == "roc":
        return pd.DataFrame({"value": roc(close, _length(raw, "length", 12))})
    raise ValueError(f"Unknown indicator: {kind}")


class IndicatorDef:
    def __init__(
        self,
        label: str,
        params: dict[str, ParamSpec],
        outputs: list[str],
        kind: str,
        category: str = "",
        definition: str = "",
        formula: str = "",
        inputs: list[str] | None = None,
        output_plots: dict[str, str] | None = None,
        catalog_id: str = "",
        note: str = "",
    ):
        self.label = label
        self.params = params
        self.outputs = outputs
        self.plot = kind
        self.category = category
        self.definition = definition
        self.formula = formula
        self.inputs = inputs or []
        self.output_plots = output_plots or {}
        self.catalog_id = catalog_id
        self.note = note


def _int_param(default: int, low: int, high: int) -> ParamSpec:
    return ParamSpec(default, low, high, 1, "int")


INDICATOR_DEFS: dict[str, IndicatorDef] = {
    "sma": IndicatorDef("SMA", {"length": _int_param(20, 2, 400)}, ["value"], "overlay"),
    "ema": IndicatorDef("EMA", {"length": _int_param(20, 2, 400)}, ["value"], "overlay"),
    "rsi": IndicatorDef("RSI", {"length": _int_param(14, 2, 100)}, ["value"], "oscillator"),
    "macd": IndicatorDef(
        "MACD",
        {
            "fast": _int_param(12, 2, 100),
            "slow": _int_param(26, 3, 200),
            "signal": _int_param(9, 2, 50),
        },
        ["macd", "signal", "hist"],
        "oscillator",
    ),
    "bollinger": IndicatorDef(
        "Bollinger Bands",
        {"length": _int_param(20, 2, 400), "std": ParamSpec(2.0, 0.5, 4.0, 0.1, "float")},
        ["mid", "upper", "lower"],
        "overlay",
    ),
    "atr": IndicatorDef("ATR", {"length": _int_param(14, 2, 100)}, ["value"], "oscillator"),
    "stochastic": IndicatorDef(
        "Stochastic",
        {"k": _int_param(14, 2, 100), "d": _int_param(3, 2, 20)},
        ["k", "d"],
        "oscillator",
    ),
    "adx": IndicatorDef("ADX", {"length": _int_param(14, 2, 100)}, ["adx", "plus_di", "minus_di"], "oscillator"),
    "donchian": IndicatorDef("Donchian", {"length": _int_param(20, 2, 400)}, ["upper", "lower", "mid"], "overlay"),
    "roc": IndicatorDef("ROC", {"length": _int_param(12, 1, 200)}, ["value"], "oscillator"),
}


from engine.indicator_impl import LIBRARY  # noqa: E402

CATALOG_ALIASES = {"bbands": "bollinger", "stoch": "stochastic"}
BUILDER_NOTES = {
    "sma": "In the builder the period is called length.",
    "ema": "In the builder the period is called length.",
    "rsi": "In the builder the period is called length.",
    "macd": "The histogram column is named hist.",
    "bollinger": "Parameters are length and std. The center line is mid.",
    "atr": "In the builder the period is called length.",
    "stochastic": "k is the lookback and d smooths %K. %K itself is not pre-smoothed.",
    "adx": "In the builder the period is called length.",
    "donchian": "length is the period. The channel is shifted one bar so the current close can break out of it.",
    "roc": "In the builder the period is called length. The value is a percent.",
}


for _kind, _lib in LIBRARY.items():
    INDICATOR_DEFS[_kind] = IndicatorDef(_kind, _lib.params, _lib.outputs, _lib.plot, output_plots=_lib.output_plots)


def _attach_catalog() -> None:
    from engine.catalog import load_catalog

    for item in load_catalog():
        key = item["id"] if item["id"] in INDICATOR_DEFS else CATALOG_ALIASES.get(item["id"])
        if key not in INDICATOR_DEFS or INDICATOR_DEFS[key].catalog_id:
            continue
        definition = INDICATOR_DEFS[key]
        definition.label = item["name"]
        definition.category = item["category"]
        definition.definition = item["definition"]
        definition.formula = item["formula"]
        definition.inputs = list(item["inputs"])
        definition.catalog_id = item["id"]
        definition.note = BUILDER_NOTES.get(key, "")


_attach_catalog()


def engine_for_catalog(catalog_id: str) -> str | None:
    if catalog_id in INDICATOR_DEFS:
        return catalog_id
    aliased = CATALOG_ALIASES.get(catalog_id)
    if aliased in INDICATOR_DEFS:
        return aliased
    return None


def indicator_categories() -> list[str]:
    from engine.catalog import catalog_categories

    present = {definition.category for definition in INDICATOR_DEFS.values() if definition.category}
    return [category for category in catalog_categories() if category in present]


def output_names(indicator_id: str, kind: str) -> list[str]:
    outputs = INDICATOR_DEFS[kind].outputs
    if outputs == ["value"]:
        return [indicator_id]
    return [f"{indicator_id}.{name}" for name in outputs]


def build_feature_frame(bars: pd.DataFrame, indicators: list[dict]) -> pd.DataFrame:
    frame = bars[["open", "high", "low", "close", "volume"]].copy()
    for spec in indicators:
        computed = compute_indicator(spec["type"], bars, spec.get("params", {}))
        names = output_names(spec["id"], spec["type"])
        if len(names) != len(computed.columns):
            raise ValueError(f"Output mismatch for {spec['id']}")
        for column, name in zip(computed.columns, names):
            frame[name] = computed[column]
    return frame


def indicator_plot_frame(bars: pd.DataFrame, indicators: list[dict]) -> tuple[pd.DataFrame, dict[str, str]]:
    columns = {}
    kinds = {}
    for spec in indicators:
        computed = compute_indicator(spec["type"], bars, spec.get("params", {}))
        names = output_names(spec["id"], spec["type"])
        definition = INDICATOR_DEFS[spec["type"]]
        for column, name in zip(computed.columns, names):
            columns[name] = computed[column]
            kinds[name] = definition.output_plots.get(column, definition.plot)
    if not columns:
        return pd.DataFrame(index=bars.index), {}
    return pd.DataFrame(columns, index=bars.index), kinds
