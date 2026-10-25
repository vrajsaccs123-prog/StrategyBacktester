"""Medium frequency momentum for a Nifty 500 book.

The book has five stages. Filter: 13-week upside/downside ratio above a
threshold, and a rising completed-month EMA. Buy: a 4/8 weekly EMA cross, with
price near the busiest support zone of the prior 90 days. Sell: the fast weekly
EMA below the slow one. Stop: half the shares use a frozen 2-ATR stop. Size:
floor((portfolio value / 20) × size factor / ATR), capped per name and by cash.

EMA comes from the shared indicator library. The book still reads a fixed
Wilder ATR for the stop and the share count; that series is not an entry or
exit rule. Swing lows use the same confirmed-pivot rule as the swing
indicator. Weekly upside/downside ratio and the support zone are not in that
catalog.

Set the universe, then edit each stage on Strategy, then run the book.
It does not fit the one-symbol backtest, which allows one full position
sized as a percent of equity.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from engine.book import size_unit
from engine.indicator_impl import confirmed_swings
from engine.indicators import atr, ema
from engine.models import ParamSpec
from engine.strategy import Stage, Strategy, resolve_params

# Wilder length frozen for the book's stop and share count. Not a signal parameter.
BOOK_ATR_LENGTH = 10

PARAM_LABELS = {
    "udr_weeks": "UDR weeks",
    "udr_min": "UDR minimum",
    "udr_undefined_passes": "UDR when there is no down week",
    "monthly_ema": "Monthly EMA length",
    "weekly_fast": "Weekly fast EMA",
    "weekly_slow": "Weekly slow EMA",
    "stop_atr": "Stop distance, in entry ATRs",
    "partial_pct": "Shares sold at the stop, percent",
    "zone_lookback_days": "Support lookback, calendar days",
    "swing_left": "Swing bars before the low",
    "swing_right": "Swing bars after the low",
    "zone_width_pct": "Zone width, percent of price",
    "touch_rebound_pct": "Rebound for a touch, percent of price",
    "zone_proximity_pct": "Entry distance to the zone, percent of price",
    "min_touches": "Minimum touches",
    "entry_rank": "Who gets the cash first",
    "capital": "Portfolio capital, rupees",
    "max_position_inr": "Max position, rupees",
    "risk_per_atr": "Factor on portfolio value / 20",
    "slippage_bps": "Slippage, bps",
    "brokerage_bps": "Brokerage, bps",
    "brokerage_cap_inr": "Brokerage cap per order, rupees",
    "stt_buy_bps": "STT on buys, bps",
    "stt_sell_bps": "STT on sells, bps",
    "exchange_bps": "Exchange charge, bps",
    "sebi_bps": "SEBI charge, bps",
    "stamp_buy_bps": "Stamp duty on buys, bps",
    "gst_pct": "GST on brokerage and exchange charges, percent",
}

PARAM_HELP = {
    "udr_weeks": "Completed weekly returns in V+ and V−. The pass is V+ / V− above the minimum.",
    "udr_min": "Strictly greater than this. 1 means up weeks outweigh down weeks.",
    "udr_undefined_passes": "0 rejects the name when every week in the window was flat or up. 1 accepts it.",
    "monthly_ema": "Completed months only. The pass is this EMA above its prior completed value.",
    "weekly_fast": "Crosses above the slow average on a completed week.",
    "weekly_slow": "Must stay longer than the fast average, or the name produces no entries.",
    "stop_atr": "Stop = entry fill minus this many frozen ATRs. The second half is not stopped out.",
    "partial_pct": "Percent of the original shares sold at the stop, rounded down. The stop fires once.",
    "zone_lookback_days": "Swing lows are taken from this many calendar days before the signal.",
    "swing_left": "A low must be below this many earlier lows.",
    "swing_right": "And this many later lows. It counts only after those later bars have closed.",
    "zone_width_pct": "Adjacent swing lows within this percent of the signal close share a zone. The band is the median plus or minus this width.",
    "touch_rebound_pct": "A visit counts once, after the close rebounds this percent of price above the top of the band.",
    "zone_proximity_pct": "The signal close must be this percent of price from the zone band, or inside it.",
    "min_touches": "Zones with fewer completed visits are ignored.",
    "entry_rank": "0 fills the highest UDR first. 1 fills the widest weekly EMA gap as a fraction of price. 2 fills in symbol order.",
    "capital": "Starting cash. Sales on the open fund buys on that same open.",
    "max_position_inr": "Share count cannot make the filled position larger than this.",
    "risk_per_atr": "Multiplies portfolio value / 20. Share count is floor(that / ATR), before the position cap and cash.",
    "slippage_bps": "Buys pay more and sells receive less, including stop fills.",
    "brokerage_bps": "Percent brokerage. With a cap, the order pays the smaller of the percent and the cap.",
    "brokerage_cap_inr": "Flat rupee charge when brokerage bps is 0. 0 means no cap on a percent brokerage.",
    "stt_buy_bps": "Delivery STT on the buy notional. Default is 0.1 percent.",
    "stt_sell_bps": "Delivery STT on the sell notional. Default is 0.1 percent.",
    "exchange_bps": "NSE equity transaction charge. Default is 0.00297 percent.",
    "sebi_bps": "SEBI turnover charge. Default is 10 rupees per crore.",
    "stamp_buy_bps": "Stamp duty on buys. Default is 0.015 percent. Sells are not charged.",
    "gst_pct": "Applied to brokerage, exchange, and SEBI charges. Not to STT or stamp.",
}

STAGES = (
    Stage(
        "filter",
        "Filter",
        "A name can be bought only when every condition in this stage passes.",
        (
            ("Upside/downside ratio", ("udr_weeks", "udr_min", "udr_undefined_passes")),
            ("Monthly trend", ("monthly_ema",)),
        ),
    ),
    Stage(
        "buy",
        "Buy",
        "A buy that passes the filter is filled on the next traded open.",
        (
            ("Weekly cross", ("weekly_fast", "weekly_slow")),
            (
                "Support zone",
                (
                    "zone_lookback_days",
                    "swing_left",
                    "swing_right",
                    "zone_width_pct",
                    "touch_rebound_pct",
                    "zone_proximity_pct",
                    "min_touches",
                ),
            ),
        ),
    ),
    Stage(
        "sell",
        "Sell",
        "A sell closes whatever shares are still open, on the next traded open.",
        (("Weekly exit", ()),),
    ),
    Stage(
        "stop",
        "Stop loss",
        "The stop is frozen from the entry fill. On an open where a stop and a sell both fill, the stop sells first.",
        (("Protective stop", ("stop_atr", "partial_pct")),),
    ),
    Stage(
        "size",
        "Position sizing",
        "Share count is the portfolio value divided by 20, times the size factor, divided by ATR, then cut by the rupee cap and by cash. An order that still does not fit is dropped.",
        (
            ("Shares", ("risk_per_atr", "max_position_inr", "capital")),
            ("Several buys the same morning", ("entry_rank",)),
        ),
    ),
)

COST_PARAMS = (
    "slippage_bps",
    "brokerage_bps",
    "brokerage_cap_inr",
    "stt_buy_bps",
    "stt_sell_bps",
    "exchange_bps",
    "sebi_bps",
    "stamp_buy_bps",
    "gst_pct",
)

RANK_LABELS = {
    0: "Highest UDR",
    1: "Widest weekly EMA gap",
    2: "Symbol name",
}

CHOICES = {
    "udr_undefined_passes": (
        ("Fail the filter", 0),
        ("Pass the filter", 1),
    ),
    "entry_rank": tuple((RANK_LABELS[index], index) for index in range(3)),
}


def stage_line(key: str, params: dict) -> str:
    """One sentence that states the current settings for a stage."""

    if key == "filter":
        undefined = "passes" if int(params["udr_undefined_passes"]) else "fails"
        return (
            f"{int(params['udr_weeks'])}-week upside/downside ratio above {float(params['udr_min']):g}, "
            f"and the {int(params['monthly_ema'])}-month EMA rising. "
            f"A name with no down week {undefined}."
        )
    if key == "buy":
        touches = int(params["min_touches"])
        touch_word = "touch" if touches == 1 else "touches"
        return (
            f"{int(params['weekly_fast'])}/{int(params['weekly_slow'])} weekly EMA cross, "
            f"close within {float(params['zone_proximity_pct']):g}% of a support zone "
            f"from the prior {int(params['zone_lookback_days'])} days "
            f"with at least {touches} {touch_word}."
        )
    if key == "sell":
        return "Fast weekly EMA below the slow one. Filled on the next open."
    if key == "stop":
        return (
            f"Stop is {float(params['stop_atr']):g} entry ATRs under the fill. "
            f"{int(params['partial_pct'])}% of the original shares sell at the stop. "
            "The rest wait for the sell."
        )
    if key == "size":
        rank = {
            0: "the highest UDR",
            1: "the widest weekly EMA gap",
            2: "symbol name",
        }[int(params["entry_rank"])]
        unit = size_unit(params)
        return (
            f"Shares are floor({unit} / ATR), "
            f"a position of {unit} × price / ATR, "
            f"capped at ₹{int(params['max_position_inr']):,}. "
            f"Starting cash is ₹{int(params['capital']):,}. "
            f"Same-morning buys are filled by {rank}."
        )
    raise KeyError(key)


class SupportZone:
    def __init__(self, level: float, low: float, high: float, touches: int):
        self.level = float(level)
        self.low = float(low)
        self.high = float(high)
        self.touches = int(touches)


def grid_values(spec: ParamSpec) -> list:
    """Each searchable step, so one parameter can be tested while the others stay put."""

    if not spec.searchable or spec.step is None or float(spec.step) <= 0:
        return [spec.coerce(spec.default)]
    count = int(round((float(spec.max) - float(spec.min)) / float(spec.step)))
    values = []
    for step_index in range(count + 1):
        raw = round(float(spec.min) + step_index * float(spec.step), 8)
        if raw > float(spec.max) + 1e-8:
            break
        coerced = spec.coerce(raw)
        if not values or values[-1] != coerced:
            values.append(coerced)
    return values or [spec.coerce(spec.default)]


def _naive(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp


def _period_finished(last_ts: pd.Timestamp, period, has_later: bool) -> bool:
    """A trailing week or month is finished when the next session would leave it."""

    if has_later:
        return True
    end = _naive(period.end_time).normalize()
    last = _naive(last_ts).normalize()
    if last >= end:
        return True
    nxt = last + pd.offsets.BDay(1)
    return nxt.to_period(period.freq) != period


def completed_ohlc(bars: pd.DataFrame, freq: str) -> pd.DataFrame:
    """OHLC of finished weeks (`W-FRI`) or months (`M`), indexed by the last session.

    The current unfinished period is omitted. A historical period is kept once a
    later period exists, including when its last session was a holiday.
    """

    columns = ["open", "high", "low", "close"]
    if freq not in {"W-FRI", "M"}:
        raise ValueError(f"Unsupported period: {freq}")
    if bars.empty:
        return pd.DataFrame(columns=columns)
    ordered = bars.sort_index()
    keys = ordered.index.to_period(freq)
    chunks = [(period, chunk) for period, chunk in ordered.groupby(keys, sort=True)]
    rows = []
    for index, (period, chunk) in enumerate(chunks):
        last = chunk.index[-1]
        if not _period_finished(last, period, index < len(chunks) - 1):
            continue
        rows.append(
            {
                "date": last,
                "open": float(chunk["open"].iloc[0]),
                "high": float(chunk["high"].max()),
                "low": float(chunk["low"].min()),
                "close": float(chunk["close"].iloc[-1]),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows).set_index("date")
    frame.index.name = "date"
    return frame


def upside_downside(close: pd.Series, weeks: int) -> pd.DataFrame:
    """V+, V−, and UDR from completed-period closes.

    V− of zero leaves UDR undefined (NaN) instead of a number.
    """

    length = int(weeks)
    if length < 1:
        raise ValueError("UDR weeks must be at least 1")
    returns = close.astype(float) / close.astype(float).shift(1) - 1
    up = np.square(returns.clip(lower=0))
    down = np.square(returns.clip(upper=0))
    v_up = up.rolling(length, min_periods=length).mean()
    v_down = down.rolling(length, min_periods=length).mean()
    ratio = v_up / v_down.where(v_down > 0)
    return pd.DataFrame({"v_up": v_up, "v_down": v_down, "udr": ratio})


def udr_passes(close: pd.Series, weeks: int, minimum: float, undefined_passes: bool) -> pd.Series:
    parts = upside_downside(close, weeks)
    passed = parts["udr"] > float(minimum)
    if undefined_passes:
        undefined = parts["v_down"].eq(0) & parts["v_up"].notna()
        passed = passed | undefined
    return passed.fillna(False)


def swing_lows(bars: pd.DataFrame, left: int, right: int) -> list[tuple[int, int, float]]:
    """Confirmed swing lows as (pivot bar, confirmation bar, price)."""

    left_n = int(left)
    right_n = int(right)
    if left_n < 1 or right_n < 1:
        raise ValueError("Swing bars must be at least 1")
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    _, lows = confirmed_swings(high, low, left_n, right_n)
    return [(confirm - right_n, confirm, price) for confirm, price in lows]


def select_support(
    bars: pd.DataFrame,
    asof: int,
    params: dict,
    swings: list[tuple[int, int, float]] | None = None,
) -> SupportZone | None:
    """Zone with the most completed touches. Ties use the higher level below the close.

    Swing lows must sit in the prior lookback and already be confirmed. The band
    around the median is one zone-width on either side. Width and rebound are
    percents of the close. A touch is one visit that later closes at least one
    rebound above the band.
    """

    if asof < 0 or asof >= len(bars):
        return None
    price_now = float(bars["close"].iloc[asof])
    if not math.isfinite(price_now) or price_now <= 0:
        return None
    width = float(params["zone_width_pct"]) / 100.0 * price_now
    rebound = float(params["touch_rebound_pct"]) / 100.0 * price_now
    if width < 0 or rebound < 0:
        return None
    if swings is None:
        swings = swing_lows(bars, int(params["swing_left"]), int(params["swing_right"]))

    asof_date = _naive(bars.index[asof])
    start = asof_date - pd.Timedelta(days=int(params["zone_lookback_days"]))
    prices = []
    for pivot, confirm, price in swings:
        if confirm > asof or pivot < 0:
            continue
        pivot_date = _naive(bars.index[pivot])
        if pivot_date < start or pivot_date >= asof_date:
            continue
        prices.append(float(price))
    if not prices:
        return None

    prices.sort()
    clusters: list[list[float]] = [[prices[0]]]
    for price in prices[1:]:
        if price - clusters[-1][-1] <= width:
            clusters[-1].append(price)
        else:
            clusters.append([price])

    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    begin = int(bars.index.searchsorted(start))
    required = int(params["min_touches"])
    best: SupportZone | None = None
    for cluster in clusters:
        level = float(np.median(cluster))
        if not level < price_now:
            continue
        band_low = level - width
        band_high = level + width
        touches = _count_touches(high, low, close, begin, asof + 1, band_low, band_high, rebound)
        if touches < required:
            continue
        zone = SupportZone(level, band_low, band_high, touches)
        if best is None or zone.touches > best.touches or (zone.touches == best.touches and zone.level > best.level):
            best = zone
    return best


def _count_touches(high, low, close, begin: int, end: int, band_low: float, band_high: float, rebound: float) -> int:
    touches = 0
    in_visit = False
    goal = band_high + rebound
    for index in range(begin, end):
        if not in_visit:
            overlaps = low[index] <= band_high and high[index] >= band_low
            if not overlaps:
                continue
            in_visit = True
        if close[index] >= goal:
            touches += 1
            in_visit = False
    return touches


def zone_distance(price: float, zone: SupportZone) -> float:
    if zone.low <= price <= zone.high:
        return 0.0
    if price < zone.low:
        return zone.low - price
    return price - zone.high


def _hold_forward(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Carry a completed-period value onto later sessions. Nothing is filled ahead of its close."""

    clean = series.sort_index().dropna()
    if clean.empty:
        return pd.Series(np.nan, index=index, dtype=float)
    combined = clean.reindex(index.union(clean.index)).sort_index().ffill()
    return combined.reindex(index)


def chart_frame(bars: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, dict[str, str]]:
    """Weekly EMAs, the monthly EMA, and UDR on the daily index."""

    index = bars.index
    weekly = completed_ohlc(bars, "W-FRI")
    monthly = completed_ohlc(bars, "M")
    fast = ema(weekly["close"], int(params["weekly_fast"])) if len(weekly) else pd.Series(dtype=float)
    slow = ema(weekly["close"], int(params["weekly_slow"])) if len(weekly) else pd.Series(dtype=float)
    month = ema(monthly["close"], int(params["monthly_ema"])) if len(monthly) else pd.Series(dtype=float)
    ratio = upside_downside(weekly["close"], int(params["udr_weeks"]))["udr"] if len(weekly) else pd.Series(dtype=float)
    frame = pd.DataFrame(
        {
            "Weekly fast EMA": _hold_forward(fast, index),
            "Weekly slow EMA": _hold_forward(slow, index),
            "Monthly EMA": _hold_forward(month, index),
            "UDR": _hold_forward(ratio, index),
        },
        index=index,
    )
    kinds = {
        "Weekly fast EMA": "overlay",
        "Weekly slow EMA": "overlay",
        "Monthly EMA": "overlay",
        "UDR": "oscillator",
    }
    return frame, kinds


def prepare_signals(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Entry and exit flags on the daily index. Flags are known at that close."""

    index = bars.index
    out = pd.DataFrame(
        {
            "entry": False,
            "exit": False,
            "atr": atr(bars, BOOK_ATR_LENGTH),
            "udr": np.nan,
            "spread_atr": np.nan,
        },
        index=index,
    )
    fast_n = int(params["weekly_fast"])
    slow_n = int(params["weekly_slow"])
    if fast_n >= slow_n or bars.empty:
        return out

    weekly = completed_ohlc(bars, "W-FRI")
    monthly = completed_ohlc(bars, "M")
    if len(weekly) < slow_n + 1 or len(monthly) < int(params["monthly_ema"]) + 1:
        return out

    fast = ema(weekly["close"], fast_n)
    slow = ema(weekly["close"], slow_n)
    spread = fast - slow
    crossed = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    bearish = fast < slow
    passed = udr_passes(
        weekly["close"],
        int(params["udr_weeks"]),
        float(params["udr_min"]),
        bool(int(params["udr_undefined_passes"])),
    )
    month_ema = ema(monthly["close"], int(params["monthly_ema"]))
    rising = month_ema > month_ema.shift(1)
    mapped = pd.Series(np.nan, index=index, dtype=float)
    common = rising.index.intersection(index)
    mapped.loc[common] = rising.loc[common].astype(float)
    rising_at = mapped.ffill().reindex(weekly.index) > 0.5

    ratio = upside_downside(weekly["close"], int(params["udr_weeks"]))
    entry_week = crossed.fillna(False) & passed.fillna(False) & rising_at.fillna(False)
    swings = swing_lows(bars, int(params["swing_left"]), int(params["swing_right"]))
    proximity = float(params["zone_proximity_pct"]) / 100.0
    for timestamp in weekly.index[entry_week.to_numpy(dtype=bool)]:
        loc = index.get_loc(timestamp)
        if isinstance(loc, slice) or not isinstance(loc, (int, np.integer)):
            continue
        price = float(bars.at[timestamp, "close"])
        if not math.isfinite(price) or price <= 0:
            continue
        zone = select_support(bars, int(loc), params, swings)
        if zone is None:
            continue
        if zone_distance(price, zone) > proximity * price:
            continue
        udr_value = ratio.at[timestamp, "udr"]
        out.at[timestamp, "entry"] = True
        out.at[timestamp, "udr"] = float(udr_value) if np.isfinite(udr_value) else 1e6
        out.at[timestamp, "spread_atr"] = float(spread.at[timestamp] / price)

    for timestamp in weekly.index[bearish.fillna(False).to_numpy(dtype=bool)]:
        out.at[timestamp, "exit"] = True
    return out


class MediumFrequencyMomentum(Strategy):
    key = "medium_frequency_momentum"
    name = "Medium frequency momentum"
    portfolio = True
    stages = STAGES
    cost_params = COST_PARAMS
    param_labels = PARAM_LABELS
    param_help = PARAM_HELP
    param_choices = CHOICES

    def param_space(self) -> dict[str, ParamSpec]:
        return {
            "udr_weeks": ParamSpec(13, 8, 26, 1, "int"),
            "udr_min": ParamSpec(1.0, 0.5, 3.0, 0.1, "float"),
            "udr_undefined_passes": ParamSpec(0, 0, 1, 1, "int"),
            "monthly_ema": ParamSpec(10, 4, 18, 1, "int"),
            "weekly_fast": ParamSpec(4, 2, 12, 1, "int"),
            "weekly_slow": ParamSpec(8, 4, 30, 1, "int"),
            "stop_atr": ParamSpec(2.0, 0.5, 5.0, 0.5, "float"),
            "partial_pct": ParamSpec(50, 0, 100, 10, "int"),
            "zone_lookback_days": ParamSpec(90, 30, 180, 10, "int"),
            "swing_left": ParamSpec(2, 1, 5, 1, "int"),
            "swing_right": ParamSpec(2, 1, 5, 1, "int"),
            "zone_width_pct": ParamSpec(1.0, 0.0, 4.0, 0.1, "float"),
            "touch_rebound_pct": ParamSpec(1.0, 0.0, 5.0, 0.1, "float"),
            "zone_proximity_pct": ParamSpec(0.5, 0.0, 3.0, 0.1, "float"),
            "min_touches": ParamSpec(1, 1, 6, 1, "int"),
            "entry_rank": ParamSpec(0, 0, 2, 1, "int"),
            "capital": ParamSpec(20_000_000, 10_000_000, 50_000_000, 10_000_000, "int"),
            "max_position_inr": ParamSpec(1_000_000, 500_000, 2_000_000, 500_000, "int"),
            "risk_per_atr": ParamSpec(1, None, None, None, "float"),
            "slippage_bps": ParamSpec(5, 0, 25, 1, "float"),
            "brokerage_bps": ParamSpec(0, 0, 10, 1, "float"),
            "brokerage_cap_inr": ParamSpec(20, None, None, None, "float"),
            "stt_buy_bps": ParamSpec(10, 0, 20, 1, "float"),
            "stt_sell_bps": ParamSpec(10, 0, 20, 1, "float"),
            "exchange_bps": ParamSpec(0.297, None, None, None, "float"),
            "sebi_bps": ParamSpec(0.01, None, None, None, "float"),
            "stamp_buy_bps": ParamSpec(1.5, None, None, None, "float"),
            "gst_pct": ParamSpec(18, None, None, None, "float"),
        }

    def target_positions(self, bars: pd.DataFrame, params: dict, start: int = 0) -> pd.Series:
        raise RuntimeError("Medium frequency momentum trades a portfolio. Run it from Study.")

    def signals(self, bars: pd.DataFrame, params: dict) -> pd.DataFrame:
        return prepare_signals(bars, resolve_params(self.param_space(), params))

    def indicator_frame(self, bars: pd.DataFrame, params: dict):
        return chart_frame(bars, resolve_params(self.param_space(), params))

    def stage_line(self, key: str, params: dict) -> str:
        return stage_line(key, params)

    def run(
        self,
        data: dict[str, pd.DataFrame],
        params: dict | None = None,
        start=None,
        end=None,
        progress=None,
    ):
        from engine.book import run_book

        return run_book(self, data, params, start=start, end=end, progress=progress)

    def sweep(self, data: dict[str, pd.DataFrame], params: dict, name: str, start=None, end=None, progress=None):
        from engine.metrics import compute_metrics

        spec = self.param_space()[name]
        rows = []
        values = grid_values(spec)
        base = resolve_params(self.param_space(), params)
        for index, value in enumerate(values, start=1):
            trial = dict(base)
            trial[name] = value
            result = self.run(data, trial, start=start, end=end)
            row = {"value": value, **compute_metrics(result)}
            rows.append(row)
            if progress is not None:
                progress(index, len(values))
        return pd.DataFrame(rows)
