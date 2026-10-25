"""SQLite store for rule strategies and backtest runs."""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from engine.backtest import TRADE_COLUMNS


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ready(value):
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return None
        return "inf"
    return value


def dump_metrics(metrics: dict) -> str:
    return json.dumps({key: _ready(value) for key, value in metrics.items()})


def load_metrics(payload: str) -> dict:
    raw = json.loads(payload)
    loaded = {}
    for key, value in raw.items():
        if value == "inf":
            loaded[key] = float("inf")
        elif value is None and key != "trade_count":
            loaded[key] = float("nan")
        else:
            loaded[key] = value
    return loaded


def dump_trades(trades: pd.DataFrame) -> str:
    if trades.empty:
        return "[]"
    rows = []
    for record in trades.to_dict(orient="records"):
        rows.append(
            {
                "entry_time": pd.Timestamp(record["entry_time"]).isoformat(),
                "exit_time": pd.Timestamp(record["exit_time"]).isoformat(),
                "side": record["side"],
                "entry_price": float(record["entry_price"]),
                "exit_price": float(record["exit_price"]),
                "shares": float(record["shares"]),
                "pnl": float(record["pnl"]),
                "return_pct": None if pd.isna(record["return_pct"]) else float(record["return_pct"]),
            }
        )
    return json.dumps(rows)


def load_trades(payload: str) -> pd.DataFrame:
    rows = json.loads(payload or "[]")
    if not rows:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    frame = pd.DataFrame(rows)
    frame["entry_time"] = pd.to_datetime(frame["entry_time"])
    frame["exit_time"] = pd.to_datetime(frame["exit_time"])
    return frame[TRADE_COLUMNS]


def dump_equity(equity: pd.DataFrame) -> str:
    rows = []
    for timestamp, row in equity.iterrows():
        rows.append(
            {
                "t": pd.Timestamp(timestamp).isoformat(),
                "equity": float(row["equity"]),
                "drawdown": float(row["drawdown"]),
                "position": int(row["position"]),
                "close": float(row["close"]),
            }
        )
    return json.dumps(rows)


def load_equity(payload: str) -> pd.DataFrame:
    rows = json.loads(payload or "[]")
    if not rows:
        return pd.DataFrame(columns=["equity", "drawdown", "position", "close"])
    frame = pd.DataFrame(rows)
    frame.index = pd.to_datetime(frame.pop("t"))
    frame.index.name = "date"
    return frame


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS strategies (
                    id INTEGER PRIMARY KEY,
                    key TEXT NOT NULL,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY,
                    strategy_key TEXT,
                    strategy_name TEXT,
                    dataset_name TEXT,
                    dataset_path TEXT,
                    date_start TEXT,
                    date_end TEXT,
                    params_json TEXT,
                    settings_json TEXT,
                    metrics_json TEXT,
                    trades_json TEXT,
                    equity_json TEXT,
                    spec_json TEXT,
                    role TEXT,
                    group_id TEXT,
                    notes TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def save_strategy(self, spec: dict, strategy_id: int | None = None) -> int:
        name = str(spec.get("name", "Strategy")).strip() or "Strategy"
        payload = json.dumps(spec)
        with self._connect() as conn:
            if strategy_id:
                conn.execute(
                    "UPDATE strategies SET name=?, spec_json=?, key=? WHERE id=?",
                    (name, payload, f"rule:{strategy_id}", strategy_id),
                )
                return int(strategy_id)
            cursor = conn.execute(
                "INSERT INTO strategies (key, name, kind, spec_json, created_at) VALUES (?, ?, 'rule', ?, ?)",
                ("pending", name, payload, _now()),
            )
            new_id = int(cursor.lastrowid)
            conn.execute("UPDATE strategies SET key=? WHERE id=?", (f"rule:{new_id}", new_id))
            return new_id

    def list_strategies(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM strategies ORDER BY created_at DESC, id DESC").fetchall()
        return [dict(row) for row in rows]

    def get_strategy(self, strategy_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM strategies WHERE id=?", (strategy_id,)).fetchone()
        return dict(row) if row else None

    def delete_strategy(self, strategy_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM strategies WHERE id=?", (strategy_id,))

    def save_run(
        self,
        *,
        strategy_key: str,
        strategy_name: str,
        dataset_name: str,
        dataset_path: str,
        params: dict,
        settings: dict,
        metrics: dict,
        trades: pd.DataFrame,
        equity: pd.DataFrame,
        spec: dict | None,
        role: str,
        group_id: str | None = None,
        notes: str | None = None,
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO runs (
                    strategy_key, strategy_name, dataset_name, dataset_path,
                    date_start, date_end, params_json, settings_json, metrics_json,
                    trades_json, equity_json, spec_json, role, group_id, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    strategy_key,
                    strategy_name,
                    dataset_name,
                    dataset_path,
                    pd.Timestamp(equity.index[0]).isoformat(),
                    pd.Timestamp(equity.index[-1]).isoformat(),
                    json.dumps(params),
                    json.dumps(settings),
                    dump_metrics(metrics),
                    dump_trades(trades),
                    dump_equity(equity),
                    json.dumps(spec) if spec is not None else None,
                    role,
                    group_id,
                    notes,
                    _now(),
                ),
            )
            return int(cursor.lastrowid)

    def list_runs(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, strategy_key, strategy_name, dataset_name, dataset_path,
                       date_start, date_end, params_json, settings_json, metrics_json,
                       spec_json, role, group_id, notes, created_at
                FROM runs
                ORDER BY created_at DESC, id DESC
                """
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["metrics"] = load_metrics(item.pop("metrics_json"))
            result.append(item)
        return result

    def get_run(self, run_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["metrics"] = load_metrics(item["metrics_json"])
        item["params"] = json.loads(item["params_json"])
        item["settings"] = json.loads(item["settings_json"])
        item["spec"] = json.loads(item["spec_json"]) if item["spec_json"] else None
        item["trades"] = load_trades(item["trades_json"])
        item["equity"] = load_equity(item["equity_json"])
        return item

    def delete_run(self, run_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM runs WHERE id=?", (run_id,))
