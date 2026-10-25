"""Nifty 500 daily bars used for equity backtests.

The adjusted parquet already carries both exchange prices and split/bonus-adjusted
prices, so one file serves both views. A symbol is one series: EQ when that
series exists, otherwise the series with the most rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
NSE_ROOT = ROOT / "nse500" / "daily-ohlcv"
ADJUSTED_PATH = NSE_ROOT / "adjusted" / "price_adjusted_ohlcv.parquet"
UNIVERSE_DIR = NSE_ROOT / "raw" / "universe"
DEFAULT_SYMBOL = "RELIANCE"

COLUMNS = [
    "date",
    "symbol",
    "series",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "adj_volume",
]


@dataclass(frozen=True)
class Listing:
    symbol: str
    company: str
    industry: str
    isin: str

    @property
    def label(self) -> str:
        return f"{self.symbol} — {self.company}"


def available() -> bool:
    return ADJUSTED_PATH.exists() and _universe_path() is not None


def dataset_ref(symbol: str, adjusted: bool = True) -> str:
    basis = "adjusted" if adjusted else "raw"
    return f"nse:{symbol}:{basis}"


def parse_ref(path: str) -> tuple[str, bool] | None:
    if not path.startswith("nse:"):
        return None
    _, symbol, basis = path.split(":", 2)
    if not symbol:
        raise ValueError(f"NSE dataset path is missing a symbol: {path}")
    return symbol, basis != "raw"


def listings() -> tuple[Listing, ...]:
    return _listings()


def missing_symbols() -> tuple[str, ...]:
    return _missing()


def lookup(symbol: str) -> Listing | None:
    for item in listings():
        if item.symbol == symbol:
            return item
    return None


def load_symbol(symbol: str, adjusted: bool = True) -> pd.DataFrame:
    """Return one symbol's daily OHLCV, indexed by date.

    Adjusted prices undo splits and bonuses. Volume is scaled the other way
    so rupee turnover stays consistent with the adjusted price.
    """

    symbol = symbol.strip().upper()
    frames = load_universe(adjusted=adjusted, symbols=[symbol])
    if symbol not in frames:
        raise ValueError(f"No daily bars for {symbol}")
    return frames[symbol]


def load_universe(adjusted: bool = True, symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Return daily OHLCV for each requested symbol.

    `symbols=None` reads every name in the price file. Names are matched
    after trimming and uppercasing. A symbol with no rows is omitted when
    more than one name was requested.
    """

    if not ADJUSTED_PATH.exists():
        raise FileNotFoundError(f"NSE prices not found: {ADJUSTED_PATH}")

    wanted: list[str] | None
    if symbols is None:
        wanted = None
        filters = None
    else:
        wanted = sorted({item.strip().upper() for item in symbols if item and item.strip()})
        if not wanted:
            return {}
        filters = [("symbol", "in", wanted)]

    table = pq.read_table(ADJUSTED_PATH, columns=COLUMNS, filters=filters)
    raw = table.to_pandas()
    if raw.empty:
        if wanted is not None and len(wanted) == 1:
            raise ValueError(f"No daily bars for {wanted[0]}")
        return {}

    frames: dict[str, pd.DataFrame] = {}
    for name, chunk in raw.groupby(raw["symbol"].astype(str), sort=True):
        frames[str(name)] = _ohlcv_frame(chunk, str(name), adjusted)
    if wanted is not None and len(wanted) == 1 and wanted[0] not in frames:
        raise ValueError(f"No daily bars for {wanted[0]}")
    return frames


def _ohlcv_frame(raw: pd.DataFrame, symbol: str, adjusted: bool) -> pd.DataFrame:
    series_names = set(raw["series"].astype(str))
    if "EQ" in series_names:
        raw = raw[raw["series"].astype(str) == "EQ"]
    else:
        chosen = raw["series"].astype(str).value_counts().index[0]
        raw = raw[raw["series"].astype(str) == chosen]

    if adjusted:
        frame = pd.DataFrame(
            {
                "open": raw["adj_open"],
                "high": raw["adj_high"],
                "low": raw["adj_low"],
                "close": raw["adj_close"],
                "volume": raw["adj_volume"],
            }
        )
    else:
        frame = pd.DataFrame(
            {
                "open": raw["open"],
                "high": raw["high"],
                "low": raw["low"],
                "close": raw["close"],
                "volume": raw["volume"],
            }
        )

    frame.index = pd.to_datetime(raw["date"], errors="coerce")
    if frame.index.isna().any():
        raise ValueError(f"{symbol} has dates that are not timestamps")
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    bad = int(frame[["open", "high", "low", "close"]].isna().any(axis=1).sum())
    if bad:
        raise ValueError(f"{symbol} has non-numeric OHLC in {bad} row(s)")

    frame = frame.sort_index()
    duplicates = frame.index[frame.index.duplicated(keep=False)].unique()
    if len(duplicates):
        sample = ", ".join(str(item.date()) for item in list(duplicates)[:5])
        raise ValueError(f"{symbol} has duplicate dates: {sample}")
    invalid = frame["high"] < frame["low"]
    if bool(invalid.any()):
        raise ValueError(f"{symbol} has high below low in {int(invalid.sum())} row(s)")

    frame.index.name = "date"
    return frame


def _universe_path() -> Path | None:
    named = UNIVERSE_DIR / "ind_nifty500list.csv"
    if named.exists():
        return named
    matches = sorted(UNIVERSE_DIR.glob("ind_nifty500list_*.csv"))
    return matches[-1] if matches else None


@lru_cache(maxsize=1)
def _symbols_in_parquet() -> frozenset[str]:
    table = pq.read_table(ADJUSTED_PATH, columns=["symbol"])
    values = pc.unique(table.column("symbol")).to_pylist()
    return frozenset(str(value) for value in values)


@lru_cache(maxsize=1)
def _listings() -> tuple[Listing, ...]:
    path = _universe_path()
    if path is None or not ADJUSTED_PATH.exists():
        return ()
    frame = pd.read_csv(path)
    present = _symbols_in_parquet()
    rows: list[Listing] = []
    for record in frame.to_dict("records"):
        symbol = str(record["Symbol"]).strip()
        if symbol not in present:
            continue
        rows.append(
            Listing(
                symbol=symbol,
                company=str(record["Company Name"]).strip(),
                industry=str(record["Industry"]).strip(),
                isin=str(record["ISIN Code"]).strip(),
            )
        )
    rows.sort(key=lambda item: item.symbol)
    return tuple(rows)


@lru_cache(maxsize=1)
def _missing() -> tuple[str, ...]:
    path = _universe_path()
    if path is None or not ADJUSTED_PATH.exists():
        return ()
    frame = pd.read_csv(path)
    present = _symbols_in_parquet()
    missing = [str(symbol).strip() for symbol in frame["Symbol"] if str(symbol).strip() not in present]
    return tuple(sorted(set(missing)))
