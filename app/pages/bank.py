import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
ROOT = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

import shared
from engine.forest import tagged_trades
from engine.metrics import format_metric

shared.boot()
st.title("Bank")

for message in shared.load_errors():
    st.warning(message)

books = shared.book_strategies()
if not books:
    st.error("No strategy is loaded. Write one on Code.")
    st.stop()

reports = {item.key: shared.load_report(item.key) for item in books}
rows = []
for item in books:
    report = reports[item.key]
    if report is None:
        continue
    trades = tagged_trades(report)
    rows.append(
        {
            "Strategy": item.name,
            "Report": report.verdict,
            "Train CAGR": format_metric("cagr", report.train_metrics.get("cagr")),
            "Test CAGR": format_metric("cagr", report.test_metrics.get("cagr")),
            "Train max drawdown": format_metric("max_drawdown", report.train_metrics.get("max_drawdown")),
            "Test max drawdown": format_metric("max_drawdown", report.test_metrics.get("max_drawdown")),
            "Trades": f"{len(trades):,}",
        }
    )
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

ready = [item for item in books if reports[item.key] is not None]
if not ready:
    st.stop()

names = [item.name for item in ready]
current = shared.selected_strategy()
picked = st.selectbox(
    "Open a report",
    names,
    index=names.index(current.name) if current and current.name in names else 0,
)
strategy = next(item for item in ready if item.name == picked)
shared.select_strategy(strategy)
shared.render_dashboard(reports[strategy.key])
