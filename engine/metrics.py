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
    return {
        "total_return": _finite(total_return),
        "cagr": _finite(cagr),
        "sharpe": _finite(sharpe),
        "sortino": _finite(sortino),
        "volatility": _finite(volatility),
        "max_drawdown": _finite(max_drawdown),
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
