"""Load and validate OHLCV CSV files."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REQUIRED = ("open", "high", "low", "close")
ALIASES = {
    "date": ("date", "datetime", "timestamp", "time", "dt"),
    "open": ("open", "o"),
    "high": ("high", "h"),
    "low": ("low", "l"),
    "close": ("close", "c", "adjclose", "adjustedclose"),
    "volume": ("volume", "vol", "v"),
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def _pick(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    lookup = {_norm(column): column for column in columns}
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    return None


def load_bars(path: str | Path) -> pd.DataFrame:
    """Load either a CSV path or an ``nse:SYMBOL:adjusted`` reference."""

    from engine.nse import load_symbol, parse_ref

    parsed = parse_ref(str(path))
    if parsed is not None:
        symbol, adjusted = parsed
        return load_symbol(symbol, adjusted=adjusted)
    return load_csv(path)


def load_csv(path: str | Path) -> pd.DataFrame:
    """Return a sorted OHLCV frame indexed by timestamp.

    Required columns, matched case-insensitively: date, open, high, low, close.
    Volume is optional and filled with zeros when missing.
    """

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"CSV not found: {source}")

    raw = pd.read_csv(source, encoding="utf-8-sig")
    if raw.empty:
        raise ValueError("CSV has no rows")

    date_col = _pick(list(raw.columns), ALIASES["date"])
    if date_col is None:
        raise ValueError("CSV needs a date column (date, datetime, or timestamp)")

    renamed = {}
    for canonical in ("open", "high", "low", "close", "volume"):
        found = _pick(list(raw.columns), ALIASES[canonical])
        if found is not None:
            renamed[found] = canonical

    missing = [name for name in REQUIRED if name not in renamed.values()]
    if missing:
        raise ValueError(f"CSV is missing columns: {', '.join(missing)}")

    frame = raw.rename(columns=renamed)
    frame.index = pd.to_datetime(frame[date_col], errors="coerce")
    if frame.index.isna().any():
        raise ValueError("Date column contains values that are not timestamps")

    for column in REQUIRED:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    bad = int(frame[list(REQUIRED)].isna().any(axis=1).sum())
    if bad:
        raise ValueError(f"OHLC columns contain non-numeric values in {bad} row(s)")

    if "volume" not in frame.columns:
        frame["volume"] = 0.0
    else:
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)

    frame = frame[["open", "high", "low", "close", "volume"]].copy()
    frame = frame.sort_index()
    duplicates = frame.index[frame.index.duplicated(keep=False)].unique()
    if len(duplicates):
        sample = ", ".join(str(item) for item in list(duplicates)[:5])
        raise ValueError(f"Duplicate timestamps: {sample}")

    invalid = frame["high"] < frame["low"]
    if bool(invalid.any()):
        raise ValueError(f"{int(invalid.sum())} row(s) have high below low")

    frame.index.name = "date"
    return frame
