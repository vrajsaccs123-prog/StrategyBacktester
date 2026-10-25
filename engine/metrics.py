"""Performance statistics from a backtest equity curve and closed trades."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from engine.models import BacktestResult

METRIC_LABELS = {
    "total_return": "Total return",
    "cagr": "CAGR",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "volatility": "Standard deviation",
    "max_drawdown": "Max drawdown",
    "drawdown_days": "Drawdown days",
    "avg_hold_days": "Average days in trade",
    "calmar": "Calmar",
    "win_rate": "Win rate",
    "avg_win_return": "Average winning trade",
    "avg_loss_return": "Average losing trade",
    "profit_factor": "Profit factor",
    "exposure": "Exposure",
    "trade_count": "Trades",
    "win_count": "Winning trades",
    "loss_count": "Losing trades",
    "starting_equity": "Investment",
    "ending_equity": "Ending equity",
}

PERCENT_METRICS = {
    "total_return",
    "cagr",
    "max_drawdown",
    "win_rate",
    "exposure",
    "volatility",
    "avg_win_return",
    "avg_loss_return",
}
RATIO_METRICS = {"sharpe", "sortino", "calmar", "profit_factor"}
COUNT_METRICS = {"trade_count", "win_count", "loss_count"}
MONEY_METRICS = {"starting_equity", "ending_equity"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ROLLING_WINDOWS = (
    ("1 year", 1),
    ("3 years", 3),
    ("5 years", 5),
    ("7 years", 7),
    ("10 years", 10),
    ("All time", None),
)


def periods_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        return 252.0
    delta = pd.Series(index).diff().median()
    if pd.isna(delta) or delta <= pd.Timedelta(0):
        return 252.0
    if delta >= pd.Timedelta(days=1) * 0.9 and delta <= pd.Timedelta(days=3):
        return 252.0
    return float(pd.Timedelta(days=365.25) / delta)


def _finite(value: float) -> float:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return float("nan")
    return float(value)


def _equity_series(values: pd.Series) -> pd.Series:
    series = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    if getattr(series.index, "tz", None) is not None:
        series.index = series.index.tz_localize(None)
    series = series.sort_index()
    return series[~series.index.duplicated(keep="last")]


def drawdown_recovery(values: pd.Series) -> tuple[float, bool]:
    """Calendar days from the max-drawdown low back to the peak before it.

    The second value is false when the series is still below that peak.
    Days then run through the last date. A series that never declined
    recovers in zero days.
    """

    series = _equity_series(values)
    if series.empty:
        return float("nan"), False
    peak = series.cummax()
    if float(peak.max()) <= 0:
        return float("nan"), False
    drawdown = series / peak.where(peak > 0) - 1
    finite = drawdown.dropna()
    if finite.empty:
        return float("nan"), False
    worst = float(finite.min())
    if worst >= -1e-12:
        return 0.0, True
    trough_at = pd.Timestamp(finite.idxmin())
    peak_value = float(peak.loc[trough_at])
    later = series.loc[trough_at:].iloc[1:]
    regained = later[later >= peak_value - 1e-6]
    if len(regained):
        end = pd.Timestamp(regained.index[0])
        return float((end - trough_at).days), True
    end = pd.Timestamp(series.index[-1])
    return float((end - trough_at).days), False


def average_hold_days(trades: pd.DataFrame | None) -> float:
    """Mean calendar days from entry fill to exit fill."""

    if trades is None or len(trades) == 0:
        return float("nan")
    if "entry_time" not in trades.columns or "exit_time" not in trades.columns:
        return float("nan")
    held = (pd.to_datetime(trades["exit_time"]) - pd.to_datetime(trades["entry_time"])).dt.total_seconds() / 86400.0
    held = pd.to_numeric(held, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if held.empty:
        return float("nan")
    return float(held.mean())


def rolling_returns(values: pd.Series, years: int | None) -> pd.Series:
    """Trailing annualized return, or the return from the first point.

    A fixed span uses the last price on or before the date that many
    calendar years earlier, then annualizes that growth. Points before
    a full span are omitted. All time is the change from the first price.
    """

    series = _equity_series(values)
    if series.empty or float(series.iloc[0]) <= 0:
        return pd.Series(dtype=float)
    if years is None:
        return (series / float(series.iloc[0]) - 1).astype(float)
    span = int(years)
    if span < 1:
        raise ValueError("Rolling window must be at least 1 year")
    left = pd.DataFrame({"asof": series.index, "now": series.to_numpy()})
    left["lookback"] = left["asof"] - pd.DateOffset(years=span)
    right = pd.DataFrame({"lookback": series.index, "then": series.to_numpy()}).sort_values("lookback")
    merged = pd.merge_asof(left.sort_values("lookback"), right, on="lookback", direction="backward")
    now = merged["now"].to_numpy(dtype=float)
    then = merged["then"].to_numpy(dtype=float)
    valid = np.isfinite(now) & np.isfinite(then) & (now > 0) & (then > 0)
    annualized = np.full(len(merged), np.nan)
    annualized[valid] = (now[valid] / then[valid]) ** (1 / span) - 1
    result = pd.Series(annualized, index=pd.DatetimeIndex(merged["asof"]))
    return result.sort_index().dropna()


def monthly_returns(values: pd.Series) -> pd.DataFrame:
    """Month columns and a year column. Each value is a portfolio change.

    The first month and the first year start from the first portfolio value.
    Later periods start from the previous period's last value, so the months
    in a year compound to that year's return.
    """

    columns = MONTHS + ["Year"]
    series = _equity_series(values)
    if len(series) < 2 or float(series.iloc[0]) <= 0:
        return pd.DataFrame(columns=columns)
    start = float(series.iloc[0])
    month_end = series.resample("ME").last().dropna()
    year_end = series.resample("YE").last().dropna()
    if month_end.empty or year_end.empty:
        return pd.DataFrame(columns=columns)
    month_base = month_end.shift(1)
    month_base.iloc[0] = start
    monthly = (month_end / month_base - 1).replace([np.inf, -np.inf], np.nan)
    year_base = year_end.shift(1)
    year_base.iloc[0] = start
    yearly = (year_end / year_base - 1).replace([np.inf, -np.inf], np.nan)
    years = sorted(set(monthly.index.year) | set(yearly.index.year))
    table = pd.DataFrame(index=pd.Index(years, dtype=int), columns=columns, dtype=float)
    for stamp, value in monthly.items():
        if pd.notna(value):
            table.loc[int(stamp.year), MONTHS[int(stamp.month) - 1]] = float(value)
    for stamp, value in yearly.items():
        if pd.notna(value):
            table.loc[int(stamp.year), "Year"] = float(value)
    return table


def format_drawdown_days(metrics: dict) -> str:
    value = metrics.get("drawdown_days")
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "—"
    text = f"{int(round(float(value))):,} days"
    if not metrics.get("drawdown_recovered", False):
        return f"{text}, open"
    return text


def compute_metrics(result: BacktestResult) -> dict:
    equity = result.equity["equity"]
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    total_return = end / start - 1 if start > 0 else float("nan")

    elapsed = equity.index[-1] - equity.index[0]
    years = elapsed.total_seconds() / (365.25 * 24 * 3600) if hasattr(elapsed, "total_seconds") else 0.0
    if years > 0 and start > 0 and end > 0:
        cagr = (end / start) ** (1 / years) - 1
    else:
        cagr = float("nan")

    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    scale = math.sqrt(periods_per_year(equity.index))
    deviation = float(returns.std(ddof=1)) if len(returns) > 1 else float("nan")
    if len(returns) > 1 and deviation > 0:
        sharpe = float(returns.mean() / deviation * scale)
        volatility = float(deviation * scale)
    elif len(returns) > 1:
        sharpe = float("nan")
        volatility = 0.0
    else:
        sharpe = float("nan")
        volatility = float("nan")
    downside = math.sqrt(float((returns.clip(upper=0) ** 2).mean())) if len(returns) else 0.0
    if len(returns) > 1 and downside > 0:
        sortino = float(returns.mean() / downside * scale)
    else:
        sortino = float("nan")

    max_drawdown = float(result.equity["drawdown"].min()) if len(result.equity) else float("nan")
    drawdown_days, drawdown_recovered = drawdown_recovery(equity)
    if math.isfinite(cagr) and math.isfinite(max_drawdown) and max_drawdown < 0:
        calmar = cagr / abs(max_drawdown)
    else:
        calmar = float("nan")

    trades = result.trades
    trade_count = int(len(trades)) if trades is not None else 0
    win_count = 0
    loss_count = 0
    avg_win_return = float("nan")
    avg_loss_return = float("nan")
    if trade_count and "pnl" in trades.columns:
        pnl = pd.to_numeric(trades["pnl"], errors="coerce")
        win_mask = pnl > 0
        loss_mask = pnl < 0
        win_count = int(win_mask.sum())
        loss_count = int(loss_mask.sum())
        wins = pnl[win_mask]
        losses = pnl[loss_mask]
        win_rate = float(win_mask.mean())
        gross_loss = float(losses.sum())
        if gross_loss < 0:
            profit_factor = float(wins.sum() / abs(gross_loss))
        elif float(wins.sum()) > 0:
            profit_factor = float("inf")
        else:
            profit_factor = float("nan")
        if "return_pct" in trades.columns:
            trade_returns = pd.to_numeric(trades["return_pct"], errors="coerce")
            if win_count:
                avg_win_return = float(trade_returns[win_mask].mean())
            if loss_count:
                avg_loss_return = float(trade_returns[loss_mask].mean())
    else:
        win_rate = float("nan")
        profit_factor = float("nan")

    exposure = float((result.equity["position"] != 0).mean())
    avg_hold = average_hold_days(trades)
    return {
        "total_return": _finite(total_return),
        "cagr": _finite(cagr),
        "sharpe": _finite(sharpe),
        "sortino": _finite(sortino),
        "volatility": _finite(volatility),
        "max_drawdown": _finite(max_drawdown),
        "drawdown_days": int(round(drawdown_days)) if math.isfinite(drawdown_days) else float("nan"),
        "drawdown_recovered": bool(drawdown_recovered) if math.isfinite(drawdown_days) else False,
        "avg_hold_days": _finite(avg_hold),
        "calmar": _finite(calmar),
        "win_rate": _finite(win_rate),
        "avg_win_return": _finite(avg_win_return),
        "avg_loss_return": _finite(avg_loss_return),
        "profit_factor": profit_factor if math.isfinite(profit_factor) else profit_factor,
        "exposure": _finite(exposure),
        "trade_count": trade_count,
        "win_count": win_count,
        "loss_count": loss_count,
        "starting_equity": start,
        "ending_equity": end,
    }


def format_metric(key: str, value) -> str:
    if key in COUNT_METRICS:
        try:
            return f"{int(value):,}"
        except (TypeError, ValueError):
            return "—"
    if key == "drawdown_days":
        try:
            return f"{int(round(float(value))):,} days"
        except (TypeError, ValueError):
            return "—"
    if key == "avg_hold_days":
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return "—"
        return f"{float(value):.1f} days"
    if isinstance(value, float) and math.isinf(value):
        return "∞"
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    if key in MONEY_METRICS:
        return f"{value:,.2f}"
    if key in PERCENT_METRICS:
        return f"{value:.2%}"
    if key in RATIO_METRICS:
        return f"{value:.2f}"
    return f"{value:.4f}"


def objective_value(metrics: dict, objective: str, max_dd_cap: float | None = None) -> float:
    """Score used by the optimizer. Infeasible trials get a large penalty."""

    if objective == "cagr" and max_dd_cap is not None:
        drawdown = metrics.get("max_drawdown", float("nan"))
        if not math.isfinite(drawdown) or drawdown < -abs(max_dd_cap):
            return -1e9
    value = metrics.get(objective, float("nan"))
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return -1e9
    return float(value)
