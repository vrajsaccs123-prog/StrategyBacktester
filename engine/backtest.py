"""Next-bar-open backtest. A signal at the close of bar t fills at the open of bar t+1."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.models import BacktestResult, BacktestSettings

TRADE_COLUMNS = [
    "entry_time",
    "exit_time",
    "side",
    "entry_price",
    "exit_price",
    "shares",
    "pnl",
    "return_pct",
]


def run_backtest(bars: pd.DataFrame, target: pd.Series, settings: BacktestSettings) -> BacktestResult:
    settings.validate()
    if len(bars) == 0:
        raise ValueError("No bars to backtest")
    if len(target) != len(bars):
        raise ValueError("Target positions must line up with bars")

    opens = bars["open"].to_numpy(dtype=float)
    closes = bars["close"].to_numpy(dtype=float)
    index = bars.index
    desired = target.reindex(index).fillna(0).astype(int).to_numpy()
    if not settings.allow_short:
        desired = np.clip(desired, 0, 1)

    cash = float(settings.initial_capital)
    shares = 0.0
    position = 0
    open_trade: dict | None = None
    trades: list[dict] = []
    equity_values = np.empty(len(bars), dtype=float)
    positions = np.empty(len(bars), dtype=int)

    slip = settings.slippage_bps / 10_000.0
    commission = settings.commission_bps / 10_000.0

    def fill_price(raw_open: float, direction: int) -> float:
        # Buying pays the higher price. Selling receives the lower price.
        return raw_open * (1 + slip * direction)

    def close_position(bar: int) -> None:
        nonlocal cash, shares, position, open_trade
        if position == 0 or open_trade is None:
            return
        direction = -1 if shares > 0 else 1
        price = fill_price(opens[bar], direction)
        exit_commission = abs(shares * price) * commission
        cash += shares * price - exit_commission
        quantity = abs(open_trade["shares"])
        if open_trade["side"] == "long":
            pnl = (price - open_trade["entry_price"]) * quantity - open_trade["entry_commission"] - exit_commission
        else:
            pnl = (open_trade["entry_price"] - price) * quantity - open_trade["entry_commission"] - exit_commission
        notional = open_trade["entry_price"] * quantity
        trades.append(
            {
                "entry_time": open_trade["entry_time"],
                "exit_time": index[bar],
                "side": open_trade["side"],
                "entry_price": open_trade["entry_price"],
                "exit_price": price,
                "shares": quantity,
                "pnl": pnl,
                "return_pct": pnl / notional if notional else np.nan,
            }
        )
        shares = 0.0
        position = 0
        open_trade = None

    def open_position(bar: int, side: int) -> None:
        nonlocal cash, shares, position, open_trade
        direction = 1 if side > 0 else -1
        price = fill_price(opens[bar], direction)
        equity_now = cash + shares * price
        allocation = equity_now * settings.position_pct
        if side > 0:
            quantity = allocation / (price * (1 + commission))
            entry_commission = quantity * price * commission
            cash -= quantity * price + entry_commission
            shares = quantity
        else:
            quantity = allocation / price
            entry_commission = quantity * price * commission
            cash += quantity * price - entry_commission
            shares = -quantity
        position = side
        open_trade = {
            "entry_time": index[bar],
            "entry_price": price,
            "shares": shares,
            "entry_commission": entry_commission,
            "side": "long" if side > 0 else "short",
        }

    for i in range(len(bars)):
        if i > 0 and int(desired[i - 1]) != position:
            target_side = int(desired[i - 1])
            if position != 0:
                close_position(i)
            if target_side != 0:
                open_position(i, target_side)
        equity_values[i] = cash + shares * closes[i]
        positions[i] = position

    peak = np.maximum.accumulate(equity_values)
    drawdown = equity_values / peak - 1
    equity = pd.DataFrame(
        {
            "equity": equity_values,
            "drawdown": drawdown,
            "position": positions,
            "close": closes,
        },
        index=index,
    )
    trade_frame = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    open_side = open_trade["side"] if open_trade else None
    return BacktestResult(equity=equity, trades=trade_frame, settings=settings, open_side=open_side)
