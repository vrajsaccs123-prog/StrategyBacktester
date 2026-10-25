"""Portfolio backtest for signals that fill on the next traded open.

Exits run before entries, so cash from a sale can fund a buy the same morning.
A name that exits and buys on the same close does both: the sale fills first.
A signal that still cannot be funded is dropped. It is not kept for a later day.
Stops use that day's range: a gap through the stop fills at the open, otherwise
at the stop. When a stop and a weekly exit fill on the same open, the stop
reduces the position first and the exit sells whatever remains.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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
    "symbol",
    "reason",
]


def _rate(params: dict, name: str) -> float:
    return float(params[name]) / 10_000.0


def brokerage(notional: float, params: dict) -> float:
    """Percent brokerage, a flat rupee cap, or the smaller of the two."""

    percent = max(notional, 0.0) * _rate(params, "brokerage_bps")
    cap = float(params["brokerage_cap_inr"])
    if cap > 0 and float(params["brokerage_bps"]) > 0:
        return min(percent, cap)
    if cap > 0:
        return cap
    return percent


def share_charges(notional: float, side: str, params: dict) -> float:
    """Brokerage, STT, exchange, SEBI, stamp, and GST for one fill."""

    if side not in {"buy", "sell"}:
        raise ValueError(f"Unknown trade side: {side}")
    broker = brokerage(notional, params)
    exchange = max(notional, 0.0) * _rate(params, "exchange_bps")
    sebi = max(notional, 0.0) * _rate(params, "sebi_bps")
    gst = float(params["gst_pct"]) / 100.0 * (broker + exchange + sebi)
    stt_name = "stt_buy_bps" if side == "buy" else "stt_sell_bps"
    stt = max(notional, 0.0) * _rate(params, stt_name)
    stamp = max(notional, 0.0) * _rate(params, "stamp_buy_bps") if side == "buy" else 0.0
    return broker + exchange + sebi + gst + stt + stamp


def validate_params(params: dict) -> None:
    if float(params["capital"]) <= 0:
        raise ValueError("Capital must be positive")
    if float(params["max_position_inr"]) <= 0:
        raise ValueError("Position cap must be positive")
    if float(params["risk_per_atr"]) <= 0:
        raise ValueError("Size factor must be positive")
    if not 0 <= int(params["partial_pct"]) <= 100:
        raise ValueError("Partial stop percent must be between 0 and 100")
    if "stop_pct" in params and not 0 <= float(params["stop_pct"]) <= 100:
        raise ValueError("Stop percent must be between 0 and 100")
    for name in (
        "slippage_bps",
        "brokerage_bps",
        "brokerage_cap_inr",
        "stt_buy_bps",
        "stt_sell_bps",
        "exchange_bps",
        "sebi_bps",
        "stamp_buy_bps",
        "gst_pct",
    ):
        if float(params[name]) < 0:
            raise ValueError(f"{name} cannot be negative")


def affordable_shares(cash: float, price: float, params: dict) -> int:
    """Largest whole share count whose notional plus buy charges fit in cash."""

    if cash <= 0 or price <= 0:
        return 0
    lo = 0
    hi = int(math.floor(cash / price))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        cost = price * mid + share_charges(price * mid, "buy", params)
        if cost <= cash + 1e-6:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _partial_shares(original: int, percent: int) -> int:
    return int(math.floor(original * int(percent) / 100.0))


@dataclass
class _Order:
    kind: str
    signal_day: pd.Timestamp
    atr: float
    udr: float
    spread_atr: float
    size: float = 1.0
    stop_pad: float = float("nan")
    stop_price: float = float("nan")


@dataclass
class _Position:
    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    stop: float
    original_shares: int
    shares: int
    cost_per_share: float
    partial_done: bool
    last_close: float


@dataclass
class _Book:
    index: pd.DatetimeIndex
    loc: dict[pd.Timestamp, int]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    entry: np.ndarray
    exit: np.ndarray
    atr: np.ndarray
    udr: np.ndarray
    spread: np.ndarray
    size: np.ndarray
    stop_pad: np.ndarray
    stop_price: np.ndarray


def _books(bars_by_symbol: dict[str, pd.DataFrame], signals_by_symbol: dict[str, pd.DataFrame]) -> dict[str, _Book]:
    books = {}
    for symbol, bars in bars_by_symbol.items():
        if bars.empty:
            continue
        signal = signals_by_symbol.get(symbol)
        if signal is None:
            signal = pd.DataFrame(index=bars.index)
        signal = signal.reindex(bars.index)
        entry = signal["entry"].fillna(False).to_numpy(dtype=bool) if "entry" in signal else np.zeros(len(bars), dtype=bool)
        exit_ = signal["exit"].fillna(False).to_numpy(dtype=bool) if "exit" in signal else np.zeros(len(bars), dtype=bool)
        blank = np.full(len(bars), np.nan)
        books[symbol] = _Book(
            index=bars.index,
            loc={pd.Timestamp(ts): i for i, ts in enumerate(bars.index)},
            open=bars["open"].to_numpy(dtype=float),
            high=bars["high"].to_numpy(dtype=float),
            low=bars["low"].to_numpy(dtype=float),
            close=bars["close"].to_numpy(dtype=float),
            entry=entry,
            exit=exit_,
            atr=signal["atr"].to_numpy(dtype=float) if "atr" in signal else np.full(len(bars), np.nan),
            udr=signal["udr"].to_numpy(dtype=float) if "udr" in signal else np.full(len(bars), np.nan),
            spread=signal["spread_atr"].to_numpy(dtype=float) if "spread_atr" in signal else np.full(len(bars), np.nan),
            size=signal["size"].to_numpy(dtype=float) if "size" in signal else blank,
            stop_pad=signal["stop_pad"].to_numpy(dtype=float) if "stop_pad" in signal else blank,
            stop_price=signal["stop_price"].to_numpy(dtype=float) if "stop_price" in signal else blank,
        )
    return books


def _order(kind: str, day: pd.Timestamp, book: _Book, loc: int) -> _Order:
    return _Order(
        kind,
        day,
        float(book.atr[loc]),
        float(book.udr[loc]),
        float(book.spread[loc]),
        float(book.size[loc]),
        float(book.stop_pad[loc]),
        float(book.stop_price[loc]),
    )


def _rank_key(symbol: str, order: _Order, mode: int):
    if mode == 2:
        return (symbol,)
    score = order.spread_atr if mode == 1 else order.udr
    if not math.isfinite(score):
        score = -1e300
    return (-score, symbol)


def simulate(
    bars_by_symbol: dict[str, pd.DataFrame],
    signals_by_symbol: dict[str, pd.DataFrame],
    params: dict,
) -> BacktestResult:
    validate_params(params)
    books = _books(bars_by_symbol, signals_by_symbol)
    if not books:
        raise ValueError("No bars to backtest")

    dates = None
    for book in books.values():
        dates = book.index if dates is None else dates.union(book.index)
    dates = dates.sort_values()

    cash = float(params["capital"])
    slip = _rate(params, "slippage_bps")
    rank_mode = int(params["entry_rank"])
    positions: dict[str, _Position] = {}
    orders: dict[str, _Order] = {}
    reentries: dict[str, _Order] = {}
    trades: list[dict] = []
    equity_rows = []

    def sell(pos: _Position, raw_price: float, quantity: int, reason: str, when: pd.Timestamp) -> None:
        nonlocal cash
        quantity = min(int(quantity), pos.shares)
        if quantity < 1:
            return
        price = raw_price * (1 - slip)
        notional = price * quantity
        cost = share_charges(notional, "sell", params)
        cash += notional - cost
        pnl = (price - pos.entry_price) * quantity - pos.cost_per_share * quantity - cost
        basis = pos.entry_price * quantity
        trades.append(
            {
                "entry_time": pos.entry_time,
                "exit_time": when,
                "side": "long",
                "entry_price": pos.entry_price,
                "exit_price": price,
                "shares": quantity,
                "pnl": pnl,
                "return_pct": pnl / basis if basis else np.nan,
                "symbol": pos.symbol,
                "reason": reason,
            }
        )
        pos.shares -= quantity
        if pos.shares <= 0:
            positions.pop(pos.symbol, None)

    def stop_quantity(pos: _Position) -> int:
        return min(_partial_shares(pos.original_shares, int(params["partial_pct"])), pos.shares)

    def fill_exit(symbol: str, day: pd.Timestamp, book: _Book, loc: int) -> None:
        pos = positions.get(symbol)
        orders.pop(symbol, None)
        if pos is None:
            return
        raw_open = float(book.open[loc])
        if not pos.partial_done and raw_open <= pos.stop:
            qty = stop_quantity(pos)
            if qty >= 1:
                sell(pos, raw_open, qty, "stop", day)
            if symbol in positions:
                positions[symbol].partial_done = True
                pos = positions[symbol]
            else:
                pos = None
        if pos is not None and pos.shares > 0:
            sell(pos, raw_open, pos.shares, "crossover", day)

    def portfolio_value(day: pd.Timestamp) -> float:
        """Cash plus open shares marked at today's open."""

        total = cash
        for held, pos in positions.items():
            if pos.shares <= 0:
                continue
            book = books[held]
            mark = float(pos.last_close)
            held_loc = book.loc.get(day)
            if held_loc is not None:
                opened = float(book.open[held_loc])
                if math.isfinite(opened) and opened > 0:
                    mark = opened
            if math.isfinite(mark) and mark > 0:
                total += pos.shares * mark
        return total

    def fill_entry(symbol: str, day: pd.Timestamp, book: _Book, loc: int, order: _Order) -> None:
        nonlocal cash
        orders.pop(symbol, None)
        if symbol in positions:
            return
        atr_value = float(order.atr)
        if not math.isfinite(atr_value) or atr_value <= 0:
            return
        price = float(book.open[loc]) * (1 + slip)
        if not math.isfinite(price) or price <= 0:
            return
        limit = int(params.get("max_positions") or 0)
        if limit > 0 and sum(pos.shares > 0 for pos in positions.values()) >= limit:
            return
        size_mult = float(order.size)
        if math.isfinite(size_mult) and size_mult <= 0:
            return
        if not math.isfinite(size_mult):
            size_mult = 1.0
        # Portfolio / 20, times the strategy's size factor, then divided by ATR.
        unit = portfolio_value(day) / 20.0 * float(params["risk_per_atr"])
        if not math.isfinite(unit) or unit <= 0:
            return
        risk_shares = int(math.floor(unit / atr_value * size_mult))
        cap_shares = int(math.floor(float(params["max_position_inr"]) / price))
        position_pct = float(params.get("max_position_pct") or 0)
        if position_pct > 0:
            room = portfolio_value(day) * position_pct / 100.0
            cap_shares = min(cap_shares, int(math.floor(room / price)))
        shares = min(risk_shares, cap_shares, affordable_shares(cash, price, params))
        if shares < 1:
            return
        notional = price * shares
        cost = share_charges(notional, "buy", params)
        if notional + cost > cash + 1e-6:
            return
        level = float(order.stop_price)
        pad = float(order.stop_pad)
        if math.isfinite(level) and level > 0:
            if level >= price:
                return
            stop = level
        elif math.isfinite(pad) and pad > 0:
            stop = price - pad
        else:
            stop_pct = float(params.get("stop_pct", 0) or 0)
            if stop_pct > 0:
                stop = price * (1.0 - stop_pct / 100.0)
            else:
                stop = price - float(params["stop_atr"]) * atr_value
        cash -= notional + cost
        positions[symbol] = _Position(
            symbol=symbol,
            entry_time=day,
            entry_price=price,
            stop=stop,
            original_shares=shares,
            shares=shares,
            cost_per_share=cost / shares,
            partial_done=False,
            last_close=float(book.close[loc]),
        )

    def maybe_stop(symbol: str, day: pd.Timestamp, book: _Book, loc: int) -> None:
        pos = positions.get(symbol)
        if pos is None or pos.partial_done:
            return
        raw_open = float(book.open[loc])
        if raw_open <= pos.stop:
            raw = raw_open
        elif float(book.low[loc]) <= pos.stop:
            raw = pos.stop
        else:
            return
        qty = stop_quantity(pos)
        if qty >= 1:
            sell(pos, raw, qty, "stop", day)
        if symbol in positions:
            positions[symbol].partial_done = True

    for day in dates:
        day = pd.Timestamp(day)
        for symbol, book in books.items():
            loc = book.loc.get(day)
            order = orders.get(symbol)
            if loc is None or order is None or order.kind != "exit" or day <= order.signal_day:
                continue
            fill_exit(symbol, day, book, loc)

        released = []
        for symbol, order in reentries.items():
            if day <= order.signal_day or symbol in positions:
                continue
            book = books[symbol]
            if book.loc.get(day) is None or symbol in orders:
                continue
            released.append(symbol)
        for symbol in released:
            orders[symbol] = reentries.pop(symbol)

        pending = []
        for symbol, book in books.items():
            loc = book.loc.get(day)
            order = orders.get(symbol)
            if loc is None or order is None or order.kind != "entry" or day <= order.signal_day:
                continue
            if symbol in positions:
                orders.pop(symbol, None)
                continue
            pending.append(symbol)
        pending.sort(key=lambda symbol: _rank_key(symbol, orders[symbol], rank_mode))
        for symbol in pending:
            book = books[symbol]
            fill_entry(symbol, day, book, book.loc[day], orders[symbol])

        for symbol, book in books.items():
            loc = book.loc.get(day)
            if loc is None or symbol not in positions:
                continue
            maybe_stop(symbol, day, book, loc)

        for symbol, book in books.items():
            loc = book.loc.get(day)
            if loc is None:
                continue
            held = symbol in positions
            if book.exit[loc] and held:
                orders[symbol] = _order("exit", day, book, loc)
                if book.entry[loc]:
                    reentries[symbol] = _order("entry", day, book, loc)
            elif book.entry[loc] and not held:
                orders[symbol] = _order("entry", day, book, loc)

        total = cash
        open_names = 0
        for symbol, pos in positions.items():
            book = books[symbol]
            loc = book.loc.get(day)
            if loc is not None:
                pos.last_close = float(book.close[loc])
            total += pos.shares * pos.last_close
            if pos.shares > 0:
                open_names += 1
        equity_rows.append((day, total, open_names))

    equity_values = np.array([row[1] for row in equity_rows], dtype=float)
    peak = np.maximum.accumulate(equity_values) if len(equity_values) else equity_values
    drawdown = equity_values / peak - 1 if len(equity_values) else equity_values
    equity = pd.DataFrame(
        {
            "equity": equity_values,
            "drawdown": drawdown,
            "position": [row[2] for row in equity_rows],
            "close": equity_values,
        },
        index=pd.DatetimeIndex([row[0] for row in equity_rows], name="date"),
    )
    settings = BacktestSettings(
        initial_capital=float(params["capital"]),
        position_pct=1.0,
        commission_bps=float(params["brokerage_bps"]),
        slippage_bps=float(params["slippage_bps"]),
        allow_short=False,
    )
    trade_frame = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    open_side = "long" if positions else None
    return BacktestResult(equity=equity, trades=trade_frame, settings=settings, open_side=open_side)
