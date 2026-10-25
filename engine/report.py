"""Plotly charts and a standalone HTML report."""

from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engine.metrics import METRIC_LABELS, MONTHS, format_drawdown_days, format_metric
from engine.models import BacktestResult, CompletedRun

CHART_RANGES = ("Day", "Week", "Month", "Quarter", "Year")
UP_COLOR = "#15803d"
DOWN_COLOR = "#dc2626"
PLOTLY_CONFIG = {"scrollZoom": True, "displayModeBar": True, "displaylogo": False}
_RANGE_RULES = {
    "Day": None,
    "Week": "W-FRI",
    "Month": "ME",
    "Quarter": "QE",
    "Year": "YE",
}


def _groups(columns: list[str]) -> list[tuple[str, list[str]]]:
    grouped: dict[str, list[str]] = {}
    for column in columns:
        grouped.setdefault(column.split(".")[0], []).append(column)
    return list(grouped.items())


def resample_ohlc(frame: pd.DataFrame, bar_range: str) -> pd.DataFrame:
    """Collapse daily bars into one OHLC candle for the chosen range."""
    if bar_range not in _RANGE_RULES:
        raise ValueError(f"Unknown range: {bar_range}")
    ordered = frame.copy()
    if not isinstance(ordered.index, pd.DatetimeIndex):
        ordered.index = pd.to_datetime(ordered.index)
    ordered = ordered.sort_index()
    ordered = ordered[~ordered.index.duplicated(keep="last")]
    rule = _RANGE_RULES[bar_range]
    if rule is None:
        return ordered
    aggregations = {}
    for column in ordered.columns:
        if column == "open":
            aggregations[column] = "first"
        elif column == "high":
            aggregations[column] = "max"
        elif column == "low":
            aggregations[column] = "min"
        elif column == "volume":
            aggregations[column] = "sum"
        else:
            aggregations[column] = "last"
    grouped = ordered.resample(rule).agg(aggregations)
    required = [column for column in ("open", "high", "low", "close") if column in grouped.columns]
    if required:
        grouped = grouped.dropna(subset=required, how="any")
    return grouped


def equity_ohlc(values: pd.Series, bar_range: str) -> pd.DataFrame:
    """Build OHLC from an account series.

    A daily candle opens at the previous close, so a down day is red.
    A longer candle uses the first, highest, lowest, and last value in the period.
    """
    if bar_range not in _RANGE_RULES:
        raise ValueError(f"Unknown range: {bar_range}")
    series = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    series = series.sort_index()
    series = series[~series.index.duplicated(keep="last")]
    if series.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    rule = _RANGE_RULES[bar_range]
    if rule is None:
        opened = series.shift(1)
        opened.iloc[0] = float(series.iloc[0])
        return pd.DataFrame(
            {
                "open": opened,
                "high": pd.concat([opened, series], axis=1).max(axis=1),
                "low": pd.concat([opened, series], axis=1).min(axis=1),
                "close": series,
            },
            index=series.index,
        )
    grouped = series.resample(rule).agg(["first", "max", "min", "last"])
    grouped.columns = ["open", "high", "low", "close"]
    return grouped.dropna(how="any")


def _candles(view: pd.DataFrame, name: str) -> go.Candlestick:
    return go.Candlestick(
        x=view.index,
        open=view["open"],
        high=view["high"],
        low=view["low"],
        close=view["close"],
        name=name,
        increasing=dict(line=dict(color=UP_COLOR, width=1), fillcolor=UP_COLOR),
        decreasing=dict(line=dict(color=DOWN_COLOR, width=1), fillcolor=DOWN_COLOR),
        hovertemplate=(
            "%{x|%d %b %Y}<br>"
            "Open: %{open:,.2f}<br>"
            "High: %{high:,.2f}<br>"
            "Low: %{low:,.2f}<br>"
            "Close: %{close:,.2f}"
            "<extra>" + name + "</extra>"
        ),
    )


def _use_log(figure: go.Figure, lows, row: int | None = None, col: int = 1) -> None:
    clean = pd.to_numeric(pd.Series(lows), errors="coerce").replace([float("inf"), float("-inf")], pd.NA).dropna()
    if len(clean) and float(clean.min()) > 0:
        if row is None:
            figure.update_yaxes(type="log")
        else:
            figure.update_yaxes(type="log", row=row, col=col)


def _zoomable(figure: go.Figure, bar_range: str, rangeslider: bool) -> None:
    figure.update_layout(dragmode="zoom", hovermode="closest", uirevision=bar_range)
    figure.update_xaxes(fixedrange=False, rangeslider_visible=False)
    figure.update_yaxes(fixedrange=False)
    if rangeslider:
        figure.update_layout(xaxis_rangeslider_visible=True)


def ohlc_figure(bars: pd.DataFrame, bar_range: str = "Day", title: str | None = None) -> go.Figure:
    view = resample_ohlc(bars, bar_range)
    figure = go.Figure()
    if not view.empty:
        figure.add_trace(_candles(view, "Price"))
        _use_log(figure, view["low"])
    figure.update_layout(
        title=title,
        height=480,
        margin=dict(l=48, r=16, t=40 if title else 16, b=24),
        yaxis_title="Price",
        legend_orientation="h",
    )
    _zoomable(figure, bar_range, rangeslider=True)
    return figure


def price_figure(
    bars: pd.DataFrame | None,
    result: BacktestResult,
    indicators: pd.DataFrame | None = None,
    kinds: dict[str, str] | None = None,
    bar_range: str = "Day",
) -> go.Figure | None:
    if bars is None:
        return None
    view = bars.reindex(result.equity.index)
    if view[["open", "high", "low", "close"]].isna().any(axis=1).all():
        return None
    view = view.dropna(subset=["open", "high", "low", "close"])
    kinds = kinds or {}
    if indicators is not None and len(indicators.columns):
        extra = [column for column in indicators.columns if column not in view.columns]
        if extra:
            view = view.join(indicators.reindex(view.index)[extra], how="left")
    view = resample_ohlc(view, bar_range)
    if view.empty:
        return None

    overlay = [column for column, kind in kinds.items() if kind == "overlay" and column in view.columns]
    oscillators = [column for column, kind in kinds.items() if kind == "oscillator" and column in view.columns]
    panels = _groups(oscillators)
    rows = 2 + len(panels)
    if panels:
        osc_share = 0.28 / len(panels)
        heights = [0.58, 0.14] + [osc_share] * len(panels)
    else:
        heights = [0.82, 0.18]

    figure = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=heights,
    )
    figure.add_trace(_candles(view, "Price"), row=1, col=1)
    for column in overlay:
        figure.add_trace(
            go.Scatter(x=view.index, y=view[column], name=column, mode="lines"),
            row=1,
            col=1,
        )
    _add_trade_markers(figure, result.trades)
    figure.add_trace(
        go.Bar(x=view.index, y=view["volume"], name="Volume", marker_color="#a8a29e"),
        row=2,
        col=1,
    )
    for offset, (name, columns) in enumerate(panels, start=3):
        for column in columns:
            figure.add_trace(
                go.Scatter(x=view.index, y=view[column], name=column, mode="lines"),
                row=offset,
                col=1,
            )
        figure.update_yaxes(title_text=name, row=offset, col=1)

    _use_log(figure, view["low"], row=1)
    figure.update_layout(
        height=640 + 140 * len(panels),
        legend_orientation="h",
        margin=dict(l=40, r=20, t=40, b=30),
        title=f"Price and trades · {bar_range}",
        yaxis_title="Price",
    )
    _zoomable(figure, bar_range, rangeslider=False)
    return figure


def _add_trade_markers(figure: go.Figure, trades: pd.DataFrame) -> None:
    if trades.empty:
        return
    styles = {
        ("long", "entry_time", "entry_price"): ("Buy", "triangle-up", "#15803d"),
        ("long", "exit_time", "exit_price"): ("Sell", "triangle-down", "#b91c1c"),
        ("short", "entry_time", "entry_price"): ("Short", "diamond", "#c2410c"),
        ("short", "exit_time", "exit_price"): ("Cover", "diamond-open", "#1d4ed8"),
    }
    for side in ("long", "short"):
        subset = trades[trades["side"] == side]
        if subset.empty:
            continue
        for (row_side, time_col, price_col), (label, symbol, color) in styles.items():
            if row_side != side:
                continue
            figure.add_trace(
                go.Scatter(
                    x=subset[time_col],
                    y=subset[price_col],
                    mode="markers",
                    name=label,
                    marker=dict(symbol=symbol, size=11, color=color, line=dict(width=1, color="#1c1917")),
                ),
                row=1,
                col=1,
            )


def equity_figure(result: BacktestResult, bar_range: str = "Day") -> go.Figure:
    view = equity_ohlc(result.equity["equity"], bar_range)
    figure = go.Figure()
    if not view.empty:
        figure.add_trace(_candles(view, "Equity"))
        _use_log(figure, view["low"])
    figure.update_layout(
        title=f"Equity · {bar_range}",
        height=420,
        margin=dict(l=48, r=16, t=40, b=24),
        yaxis_title="Rupees",
        legend_orientation="h",
    )
    _zoomable(figure, bar_range, rangeslider=True)
    return figure


def equity_line_figure(result: BacktestResult, title: str = "Portfolio value", color: str = "#1d4ed8") -> go.Figure:
    """Portfolio value through the selected window."""

    series = pd.to_numeric(result.equity["equity"], errors="coerce").dropna().astype(float)
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    series = series.sort_index()
    series = series[~series.index.duplicated(keep="last")]
    figure = go.Figure()
    if not series.empty:
        figure.add_trace(
            go.Scatter(
                x=series.index,
                y=series,
                name="Portfolio",
                mode="lines",
                line=dict(color=color, width=2),
                fill="tozeroy",
                fillcolor="rgba(29, 78, 216, 0.08)" if color == "#1d4ed8" else "rgba(68, 64, 60, 0.08)",
                hovertemplate="%{x|%d %b %Y}<br>%{y:,.2f}<extra>Portfolio</extra>",
            )
        )
    figure.update_layout(
        title=title,
        height=420,
        margin=dict(l=48, r=16, t=40, b=24),
        yaxis_title="Rupees",
        legend_orientation="h",
        dragmode="zoom",
        hovermode="x unified",
    )
    figure.update_xaxes(fixedrange=False)
    figure.update_yaxes(fixedrange=False, tickformat=",.0f")
    return figure


def rolling_return_figure(series: pd.Series, title: str, color: str = "#1d4ed8") -> go.Figure:
    """Trailing or all-time return for the selected window."""

    figure = go.Figure()
    if series is not None and len(series):
        fill = "rgba(29, 78, 216, 0.12)" if color == "#1d4ed8" else "rgba(68, 64, 60, 0.12)"
        figure.add_trace(
            go.Scatter(
                x=series.index,
                y=series.to_numpy(dtype=float),
                name="Return",
                mode="lines",
                line=dict(color=color, width=2),
                fill="tozeroy",
                fillcolor=fill,
                hovertemplate="%{x|%d %b %Y}<br>%{y:.2%}<extra>Return</extra>",
            )
        )
        figure.add_hline(y=0, line_width=1, line_color="#a8a29e")
    figure.update_layout(
        title=title,
        height=360,
        margin=dict(l=48, r=16, t=40, b=24),
        yaxis_title="Return",
        legend_orientation="h",
        dragmode="zoom",
        hovermode="x unified",
        uirevision=title,
    )
    figure.update_xaxes(fixedrange=False)
    figure.update_yaxes(fixedrange=False, tickformat=".0%")
    return figure


def _mix(start: str, end: str, tone: float) -> str:
    def parts(color: str) -> tuple[int, int, int]:
        color = color.lstrip("#")
        return tuple(int(color[index : index + 2], 16) for index in (0, 2, 4))

    origin = parts(start)
    target = parts(end)
    mixed = tuple(round(origin[index] + (target[index] - origin[index]) * tone) for index in range(3))
    return "#{:02x}{:02x}{:02x}".format(*mixed)


def _heat_colors(value: float) -> tuple[str, str]:
    cap = 0.10
    if value > 0:
        tone = min(value / cap, 1.0)
        background = _mix("#dcfce7", "#15803d", tone)
        foreground = "#f0fdf4" if tone >= 0.55 else "#14532d"
    elif value < 0:
        tone = min(abs(value) / cap, 1.0)
        background = _mix("#fee2e2", "#dc2626", tone)
        foreground = "#fef2f2" if tone >= 0.55 else "#7f1d1d"
    else:
        background, foreground = "#f5f5f4", "#44403c"
    return background, foreground


def _heat_css(value, edge: bool = False) -> str:
    if pd.isna(value):
        css = "background-color: #fafaf9"
    else:
        background, foreground = _heat_colors(float(value))
        css = f"background-color: {background}; color: {foreground}"
    if edge:
        css += "; font-weight: 700; border-left: 2px solid #a8a29e"
    return css


def style_monthly_returns(table: pd.DataFrame):
    """Green cells for gains and red cells for losses. The year is the last column."""

    view = table.copy()
    view.index = [str(int(year)) for year in view.index]
    view.index.name = None
    return (
        view.style.format("{:.1%}", na_rep="")
        .map(lambda value: _heat_css(value), subset=MONTHS)
        .map(lambda value: _heat_css(value, edge=True), subset=["Year"])
    )


def monthly_returns_html(table: pd.DataFrame) -> str:
    """A scrollable heatmap. Narrow windows can move sideways to the year column."""

    inner = style_monthly_returns(table).to_html()
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  html, body {{ margin: 0; padding: 0; background: transparent; color: #1c1917; font-family: "Source Sans Pro", sans-serif; }}
  .wrap {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; font-size: 0.82rem; min-width: 760px; width: 100%; font-variant-numeric: tabular-nums; }}
  th, td {{ border: 1px solid #e7e5e4; padding: 0.35rem 0.45rem; text-align: right; white-space: nowrap; }}
  thead th {{ background: #f5f5f4; font-weight: 600; }}
  tbody th {{ background: #fafaf9; text-align: left; font-weight: 600; position: sticky; left: 0; z-index: 1; }}
  thead th:first-child {{ position: sticky; left: 0; z-index: 2; }}
</style></head>
<body><div class="wrap">{inner}</div></body></html>"""


def drawdown_figure(result: BacktestResult) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=result.equity.index,
            y=result.equity["drawdown"],
            name="Drawdown",
            mode="lines",
            fill="tozeroy",
            line=dict(color="#b91c1c", width=1),
        )
    )
    figure.update_layout(
        title="Drawdown",
        height=280,
        margin=dict(l=48, r=16, t=40, b=24),
        yaxis_tickformat=".0%",
        dragmode="zoom",
    )
    figure.update_xaxes(fixedrange=False)
    figure.update_yaxes(fixedrange=False)
    return figure


def comparison_figure(series: dict[str, pd.Series], bar_range: str = "Day") -> go.Figure:
    names = list(series)
    rows = max(len(names), 1)
    spacing = min(0.04, 0.9 / max(rows - 1, 1))
    figure = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=spacing,
        subplot_titles=names,
    )
    for row, name in enumerate(names, start=1):
        values = series[name].dropna()
        if values.empty:
            continue
        base = float(values.iloc[0])
        rebased = values / base if base else values
        view = equity_ohlc(rebased, bar_range)
        if view.empty:
            continue
        figure.add_trace(_candles(view, name), row=row, col=1)
        _use_log(figure, view["low"], row=row)
    figure.update_layout(
        title=f"Growth of 1 · {bar_range}",
        height=max(420, 300 * rows),
        showlegend=False,
        margin=dict(l=48, r=16, t=48, b=24),
    )
    figure.update_annotations(font_size=12)
    _zoomable(figure, bar_range, rangeslider=False)
    return figure


def symbol_price_figure(
    bars: pd.DataFrame,
    trades: pd.DataFrame,
    indicators: pd.DataFrame | None = None,
    kinds: dict[str, str] | None = None,
    bar_range: str = "Day",
    title: str = "Price",
    test_start: pd.Timestamp | None = None,
) -> go.Figure:
    """Candles for one name, with the strategy's indicators and buy and sell arrows."""

    view = bars.dropna(subset=["open", "high", "low", "close"])
    kinds = kinds or {}
    if indicators is not None and len(indicators.columns):
        extra = [column for column in indicators.columns if column not in view.columns]
        if extra:
            view = view.join(indicators.reindex(view.index)[extra], how="left")
    view = resample_ohlc(view, bar_range)
    overlay = [column for column, kind in kinds.items() if kind == "overlay" and column in view.columns]
    oscillators = [column for column, kind in kinds.items() if kind == "oscillator" and column in view.columns]
    panels = _groups(oscillators)
    rows = 2 + len(panels)
    if panels:
        osc_share = 0.28 / len(panels)
        heights = [0.58, 0.14] + [osc_share] * len(panels)
    else:
        heights = [0.82, 0.18]

    figure = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=heights,
    )
    if not view.empty:
        figure.add_trace(_candles(view, "Price"), row=1, col=1)
        for column in overlay:
            figure.add_trace(
                go.Scatter(x=view.index, y=view[column], name=column, mode="lines", connectgaps=False),
                row=1,
                col=1,
            )
        _use_log(figure, view["low"], row=1)
        volume = view["volume"] if "volume" in view.columns else pd.Series(0.0, index=view.index)
        figure.add_trace(
            go.Bar(x=view.index, y=volume, name="Volume", marker_color="#a8a29e"),
            row=2,
            col=1,
        )
    _add_book_markers(figure, trades)
    for offset, (name, columns) in enumerate(panels, start=3):
        for column in columns:
            figure.add_trace(
                go.Scatter(x=view.index, y=view[column], name=column, mode="lines", connectgaps=False),
                row=offset,
                col=1,
            )
        figure.update_yaxes(title_text=name, row=offset, col=1)
    if test_start is not None and not view.empty and view.index[-1] >= pd.Timestamp(test_start):
        figure.add_vrect(
            x0=pd.Timestamp(test_start),
            x1=view.index[-1],
            fillcolor="#1d4ed8",
            opacity=0.06,
            line_width=0,
            layer="below",
            row=1,
            col=1,
        )
    figure.update_layout(
        height=640 + 140 * len(panels),
        legend_orientation="h",
        margin=dict(l=40, r=20, t=40, b=30),
        title=title,
        yaxis_title="Price",
    )
    _zoomable(figure, bar_range, rangeslider=False)
    return figure


def _add_book_markers(figure: go.Figure, trades: pd.DataFrame) -> None:
    if trades is None or trades.empty:
        return
    work = trades.copy()
    if "window" not in work.columns:
        work["window"] = "train"
    styles = {
        ("train", "entry_time", "entry_price"): ("Train buy", "triangle-up", "#15803d"),
        ("train", "exit_time", "exit_price"): ("Train sell", "triangle-down", "#b91c1c"),
        ("test", "entry_time", "entry_price"): ("Test buy", "triangle-up", "#1d4ed8"),
        ("test", "exit_time", "exit_price"): ("Test sell", "triangle-down", "#c2410c"),
    }
    for (window, time_col, price_col), (label, symbol, color) in styles.items():
        subset = work[work["window"] == window]
        if subset.empty:
            continue
        custom = subset["reason"] if "reason" in subset.columns else pd.Series([""] * len(subset), index=subset.index)
        figure.add_trace(
            go.Scatter(
                x=subset[time_col],
                y=subset[price_col],
                mode="markers",
                name=label,
                marker=dict(symbol=symbol, size=12, color=color, line=dict(width=1, color="#1c1917")),
                customdata=custom,
                hovertemplate="%{x|%d %b %Y}<br>" + label + " @ %{y:,.2f}<br>%{customdata}<extra></extra>",
            ),
            row=1,
            col=1,
        )


def report_figures(run: CompletedRun, bars: pd.DataFrame | None, bar_range: str = "Day") -> list[go.Figure]:
    figures = []
    price = price_figure(bars, run.result, run.indicators, run.indicator_kinds, bar_range)
    if price is not None:
        figures.append(price)
    figures.append(equity_figure(run.result, bar_range))
    figures.append(drawdown_figure(run.result))
    return figures


def report_html(title: str, run: CompletedRun, bars: pd.DataFrame | None, subtitle: str = "", bar_range: str = "Day") -> str:
    def _shown(key: str) -> str:
        if key == "drawdown_days":
            return format_drawdown_days(run.metrics)
        return format_metric(key, run.metrics[key])

    rows = "".join(
        f"<tr><td>{html.escape(METRIC_LABELS[key])}</td><td>{html.escape(_shown(key))}</td></tr>"
        for key in METRIC_LABELS
        if key in run.metrics
    )
    charts = "\n".join(
        figure.to_html(full_html=False, include_plotlyjs="cdn", config=PLOTLY_CONFIG)
        for figure in report_figures(run, bars, bar_range)
    )
    trades = run.result.trades
    if trades.empty:
        trade_table = "<p>No closed trades.</p>"
    else:
        pretty = trades.copy()
        pretty["entry_time"] = pretty["entry_time"].map(lambda value: pd.Timestamp(value).strftime("%Y-%m-%d %H:%M"))
        pretty["exit_time"] = pretty["exit_time"].map(lambda value: pd.Timestamp(value).strftime("%Y-%m-%d %H:%M"))
        trade_table = pretty.to_html(index=False, float_format=lambda value: f"{value:,.4f}")
    open_note = ""
    if run.result.open_side:
        open_note = f"<p>A {html.escape(run.result.open_side)} position was still open on the last bar.</p>"
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body {{ font-family: Georgia, serif; margin: 2rem auto; max-width: 1100px; color: #1c1917; }}
table {{ border-collapse: collapse; margin: 1rem 0 2rem; }}
td, th {{ border-bottom: 1px solid #d6d3d1; padding: 0.35rem 0.7rem; text-align: left; }}
</style></head>
<body>
<h1>{html.escape(title)}</h1>
<p>{html.escape(subtitle)}</p>
{open_note}
<table>{rows}</table>
{charts}
<h2>Trades</h2>
{trade_table}
</body></html>"""
