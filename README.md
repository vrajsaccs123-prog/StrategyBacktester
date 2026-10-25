# Strategy desk

A bank of Nifty 500 strategies. Each strategy is code: a filter, a buy, a sell, a size, and the costs. Its report chooses the indicator parameters on the first 70% of sessions, then scores the last 30% once with that locked set. The report includes the charts of the fills.

The contract for a new strategy, and for how the desk has to show it, is in `CONTEXT.md`.

## Setup

Python 3.11 or newer.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/Home.py
```

Tests:

```powershell
pytest
```

## Pages

Bank lists every strategy and opens its saved report. Code edits a strategy file. Indicators is a study of every indicator in `indicators.json`: what it is, how it works, and why it would be used.

Prices are the adjusted Nifty 500 series in `nse500/daily-ohlcv`. A signal at the close fills on the next open. Stops use that day's range. Same-morning buys that do not fit are dropped.
