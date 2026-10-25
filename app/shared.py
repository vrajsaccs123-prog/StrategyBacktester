"""Bank state, saved backtest reports, and the report view."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.book import param_label
from engine.forest import StudyResult, calendar, tagged_trades, window_dates
from engine.metrics import ROLLING_WINDOWS, compute_metrics, format_drawdown_days, format_metric, monthly_returns, rolling_returns
from engine.report import CHART_RANGES, PLOTLY_CONFIG, equity_line_figure, monthly_returns_html, rolling_return_figure, symbol_price_figure
from engine.strategy import Strategy, resolve_params
from strategies.registry import discover

STRATEGIES = ROOT / "strategies"
REPORTS = ROOT / "data" / "reports"
WINDOW_STATS = (
    ("cagr", "CAGR"),
    ("total_return", "Absolute return"),
    ("starting_equity", "Investment"),
    ("ending_equity", "Closing amount"),
    ("win_count", "Winning trades"),
    ("loss_count", "Losing trades"),
    ("avg_win_return", "Avg winning return"),
    ("avg_loss_return", "Avg losing return"),
    ("avg_hold_days", "Avg days in trade"),
    ("max_drawdown", "Max drawdown"),
    ("drawdown_days", "Drawdown days"),
    ("sortino", "Sortino"),
    ("sharpe", "Sharpe"),
    ("volatility", "Standard deviation"),
)


def boot() -> None:
    st.session_state.setdefault("strategy_key", None)
    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] {
            background: #f5f5f4;
            border: 1px solid #e7e5e4;
            padding: 0.55rem 0.75rem;
            border-radius: 6px;
        }
        div[data-testid="stElementContainer"]:has(iframe[title="chart-swipe"]),
        div[data-testid="stElementContainer"]:has(iframe[title="st.iframe"]) {
            height: 0 !important;
            min-height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
        }
        textarea {
            font-family: Consolas, "Courier New", monospace !important;
            font-size: 0.84rem !important;
            line-height: 1.45 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def book_strategies() -> list[Strategy]:
    return [item for item in discover() if getattr(item, "stages", None)]


def load_errors() -> list[str]:
    from strategies.registry import load_errors as errors

    discover()
    return list(errors)


def selected_strategy() -> Strategy | None:
    books = book_strategies()
    if not books:
        return None
    key = st.session_state.get("strategy_key")
    for item in books:
        if item.key == key:
            return item
    st.session_state["strategy_key"] = books[0].key
    return books[0]


def select_strategy(strategy: Strategy) -> None:
    st.session_state["strategy_key"] = strategy.key


@st.cache_resource(show_spinner="Loading the Nifty 500")
def universe() -> dict[str, pd.DataFrame]:
    from engine.nse import available, listings, load_universe

    if not available():
        raise FileNotFoundError("The Nifty 500 price file is missing from nse500/daily-ohlcv/adjusted.")
    symbols = [item.symbol for item in listings()]
    data = load_universe(adjusted=True, symbols=symbols)
    if not data:
        raise ValueError("No listed name has daily bars")
    return data


def universe_caption(data: dict[str, pd.DataFrame]) -> str:
    dates = calendar(data)
    return (
        f"{len(data):,} names · adjusted prices · "
        f"{pd.Timestamp(dates[0]):%d %b %Y} to {pd.Timestamp(dates[-1]):%d %b %Y} · "
        f"{len(dates):,} sessions"
    )


def strategy_files() -> list[Path]:
    skipped = {"registry.py", "__init__.py"}
    return sorted(
        path
        for path in STRATEGIES.glob("*.py")
        if path.name not in skipped and not path.name.startswith("_")
    )


def editor_files() -> list[Path]:
    """Book strategies, plus a file that failed to load. One-symbol modules stay off the desk."""

    discover()
    books = {path.name for path in (file_for(item) for item in book_strategies()) if path is not None}
    failed = set()
    for message in load_errors():
        failed.add(message.split(":", 1)[0].strip())
    loaded = {item.__class__.__module__.rsplit(".", 1)[-1] for item in discover()}
    shown = []
    for path in strategy_files():
        if path.name in books or path.stem in failed or path.stem not in loaded:
            shown.append(path)
    return shown


def file_for(strategy: Strategy) -> Path | None:
    module = strategy.__class__.__module__.rsplit(".", 1)[-1]
    path = STRATEGIES / f"{module}.py"
    return path if path.exists() else None


def show_chart(figure) -> None:
    st.plotly_chart(figure, width="stretch", config=PLOTLY_CONFIG)


def _stage_sentence(strategy: Strategy, key: str, params: dict) -> str:
    try:
        return strategy.stage_line(key, params)
    except Exception:
        for stage in strategy.stages:
            if stage.key == key:
                return stage.summary
        return ""


def show_stages(strategy: Strategy) -> None:
    params = resolve_params(strategy.param_space(), None)
    for stage in strategy.stages:
        st.markdown(f"**{stage.title}.** {_stage_sentence(strategy, stage.key, params)}")
    if strategy.cost_params:
        labels = ", ".join(param_label(strategy, name) for name in strategy.cost_params)
        st.markdown(f"**Costs.** {labels}.")


def render_dashboard(study: StudyResult) -> None:
    window = st.segmented_control(
        "Results",
        ["Test", "Train"],
        default="Test",
        key=f"report_window_{study.strategy_key}",
        required=True,
    )
    st.subheader(study.strategy_name)
    train_dates, test_dates = window_dates(study)
    st.markdown(
        f"{study.universe_size:,} names  \n"
        f"**Train** {train_dates}  \n"
        f"**Test** {test_dates}"
    )
    if study.verdict == "Holds on test":
        st.success(f"{study.verdict}. {study.verdict_text}")
    elif study.verdict == "Does not hold":
        st.error(f"{study.verdict}. {study.verdict_text}")
    elif study.verdict == "No read":
        st.info(f"{study.verdict}. {study.verdict_text}")
    else:
        st.warning(f"{study.verdict}. {study.verdict_text}")
    st.caption("Holds means the test ratio is at least 75% of the train ratio. The test window is the last 30% of sessions.")

    result = study.test_result if window == "Test" else study.train_result
    span = test_dates if window == "Test" else train_dates
    render_charts(study, window, span)

    st.markdown(f"**{window} performance**")
    st.caption(span)
    if result.equity is None or result.equity.empty:
        st.write("This window has no portfolio value.")
    else:
        metrics = compute_metrics(result)
        _window_dashboard(metrics)
        st.caption(
            "Drawdown days count calendar days from the max-drawdown low until the portfolio regains the peak from before that decline. "
            "An open count has not regained that peak. "
            "Average days in a trade are calendar days from the entry fill to the exit fill. "
            "Standard deviation is the annualized volatility of daily portfolio returns."
        )
        color = "#1d4ed8" if window == "Test" else "#44403c"
        _rolling_section(result, window, study.strategy_key, color)
        _monthly_section(result)
        show_chart(equity_line_figure(result, title=f"{window} portfolio value", color=color))

    _trade_section(_window_trades(study, window))

    st.markdown("**Parameters**")
    if study.optimized and study.searched:
        st.caption(
            "Indicator settings were chosen on the train window. "
            "The test window was scored once with this set and did not change it. "
            "Capital, costs, the stop, and entry rank stay at the file values."
        )
    st.dataframe(_param_table(study), width="stretch", hide_index=True)


def _window_dashboard(metrics: dict) -> None:
    for start in range(0, len(WINDOW_STATS), 4):
        columns = st.columns(4)
        for column, (key, label) in zip(columns, WINDOW_STATS[start : start + 4]):
            if key == "drawdown_days":
                shown = format_drawdown_days(metrics)
            else:
                shown = format_metric(key, metrics.get(key))
            column.metric(label, shown)


def _rolling_section(result, window: str, strategy_key: str, color: str) -> None:
    st.markdown("**Rolling return**")
    labels = [label for label, _ in ROLLING_WINDOWS]
    choice = st.segmented_control(
        "Rolling return horizon",
        labels,
        default="1 year",
        key=f"rolling_return_{strategy_key}_{window.lower()}",
        required=True,
        label_visibility="collapsed",
        width="stretch",
        wrap=True,
    )
    years = dict(ROLLING_WINDOWS)[choice]
    series = rolling_returns(result.equity["equity"], years)
    if years is None:
        st.caption("Each point is the return from the first day of this window.")
    else:
        st.caption(
            f"Each point is the annualized return over the previous {choice}. "
            f"The line starts once this window contains {choice}."
        )
    if series.empty:
        st.write(f"This window is shorter than {choice}, so there is no rolling return for that span.")
        return
    title = f"{window} · all-time return" if years is None else f"{window} · {choice} rolling return"
    show_chart(rolling_return_figure(series, title=title, color=color))


def _monthly_section(result) -> None:
    st.markdown("**Monthly returns**")
    st.caption("Each cell is the change in portfolio value that month. The last column is the change for that year. Green is a gain. Red is a loss.")
    table = monthly_returns(result.equity["equity"])
    if table.empty:
        st.write("No monthly returns in this window.")
        return
    st.iframe(monthly_returns_html(table), height=34 * (len(table) + 1) + 18)


def _window_trades(study: StudyResult, window: str) -> pd.DataFrame:
    trades = tagged_trades(study)
    if trades.empty or "window" not in trades.columns:
        return trades.iloc[0:0].copy() if not trades.empty else trades
    return trades[trades["window"] == window.lower()].copy()


def _present_trades(trades: pd.DataFrame) -> pd.DataFrame:
    work = trades.copy()
    view = pd.DataFrame(index=work.index)
    view["Symbol"] = work["symbol"].astype(str) if "symbol" in work.columns else ""
    view["Entry"] = pd.to_datetime(work["entry_time"]).dt.strftime("%d %b %Y")
    view["Exit"] = pd.to_datetime(work["exit_time"]).dt.strftime("%d %b %Y")
    view["Entry price"] = pd.to_numeric(work["entry_price"], errors="coerce").map(lambda value: "—" if pd.isna(value) else f"{value:,.2f}")
    view["Exit price"] = pd.to_numeric(work["exit_price"], errors="coerce").map(lambda value: "—" if pd.isna(value) else f"{value:,.2f}")
    view["Shares"] = pd.to_numeric(work["shares"], errors="coerce").map(_share_text)
    view["Pnl"] = pd.to_numeric(work["pnl"], errors="coerce").map(lambda value: "—" if pd.isna(value) else f"{value:,.2f}")
    view["Return"] = pd.to_numeric(work["return_pct"], errors="coerce").map(lambda value: "—" if pd.isna(value) else f"{value:.2%}")
    view["Reason"] = work["reason"].astype(str) if "reason" in work.columns else ""
    return view.reset_index(drop=True)


def _share_text(value) -> str:
    if pd.isna(value):
        return "—"
    if float(value) == int(value):
        return f"{int(value):,}"
    return f"{value:,.2f}"


def _trade_section(trades: pd.DataFrame) -> None:
    st.markdown("**Trades**")
    if trades.empty:
        st.write("No closed trades in this window.")
        return
    latest = trades.sort_values(["exit_time", "symbol"], ascending=[False, True])
    preview = latest.head(5)
    st.caption(f"Latest {len(preview):,} of {len(trades):,} closed trades.")
    st.dataframe(_present_trades(preview), width="stretch", hide_index=True)
    if len(trades) > 5:
        with st.expander(f"All {len(trades):,} trades"):
            chronological = trades.sort_values(["entry_time", "symbol"])
            st.dataframe(_present_trades(chronological), width="stretch", hide_index=True)


def _param_table(study: StudyResult) -> pd.DataFrame:
    strategy = None
    for item in book_strategies():
        if item.key == study.strategy_key:
            strategy = item
            break
    if strategy is None:
        rows = [{"Parameter": name, "Used": value} for name, value in study.params.items()]
        return pd.DataFrame(rows)
    defaults = resolve_params(strategy.param_space(), None)
    space = strategy.param_space()
    rows = []
    seen: set[str] = set()
    chosen = set(study.searched)
    groups = [(stage.title, stage.params) for stage in strategy.stages]
    groups.append(("Costs", tuple(strategy.cost_params)))
    for title, names in groups:
        for name in names:
            if name in seen or name not in space:
                continue
            seen.add(name)
            row = {
                "Stage": title,
                "Parameter": param_label(strategy, name),
                "Used": study.params.get(name, defaults.get(name)),
            }
            if study.optimized:
                row["Source"] = "Train choice" if name in chosen else "File"
            rows.append(row)
    return pd.DataFrame(rows)


def _report_file(key: str) -> Path:
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in key)
    return REPORTS / f"{safe}.pkl"


def save_report(study: StudyResult) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    _report_file(study.strategy_key).write_bytes(pickle.dumps(study, protocol=pickle.HIGHEST_PROTOCOL))


def load_report(key: str) -> StudyResult | None:
    path = _report_file(key)
    if not path.exists():
        return None
    return pickle.loads(path.read_bytes())


def render_charts(study: StudyResult, window: str, span: str) -> None:
    """One name at a time, with the indicators and the buy and sell arrows."""

    from engine.nse import load_symbol

    trades = _window_trades(study, window)
    symbols = []
    if not trades.empty and "symbol" in trades.columns:
        symbols = sorted(str(symbol) for symbol in trades["symbol"].dropna().unique())
    st.markdown("**Charts**")
    st.caption(f"{window} {span}")
    if not symbols:
        st.write("No closed trade to mark in this window.")
        return
    state_key = f"chart_symbol_{study.strategy_key}_{window.lower()}"
    if st.session_state.get(state_key) not in symbols:
        st.session_state[state_key] = symbols[0]
    index = symbols.index(st.session_state[state_key])

    def _shift(step: int, names: list[str], key: str) -> None:
        current = st.session_state.get(key)
        place = names.index(current) if current in names else 0
        st.session_state[key] = names[min(max(place + step, 0), len(names) - 1)]

    bar_range = st.segmented_control(
        "Range",
        list(CHART_RANGES),
        default="Day",
        key=f"chart_range_{study.strategy_key}",
    ) or "Day"
    st.markdown('<span id="chart-nav-anchor"></span>', unsafe_allow_html=True)
    previous, picker, following = st.columns([1, 6, 1])
    previous.button(
        "Previous",
        disabled=index == 0,
        width="stretch",
        key=f"prev_{study.strategy_key}_{window.lower()}",
        on_click=_shift,
        args=(-1, symbols, state_key),
    )
    picker.selectbox(
        "Name",
        symbols,
        key=state_key,
        format_func=lambda name: f"{symbols.index(name) + 1} / {len(symbols)} · {name}",
    )
    following.button(
        "Next",
        disabled=index >= len(symbols) - 1,
        width="stretch",
        key=f"next_{study.strategy_key}_{window.lower()}",
        on_click=_shift,
        args=(1, symbols, state_key),
    )
    symbol = st.session_state[state_key]
    if window == "Train":
        st.caption("Swipe the chart left or right to change name. Green arrows are buys. Red arrows are sells. The blue band is the test window.")
    else:
        st.caption("Swipe the chart left or right to change name. Blue arrows are buys. Orange arrows are sells. The blue band is the test window.")
    try:
        bars = load_symbol(symbol, adjusted=True)
    except Exception as exc:
        st.error(str(exc))
        return
    strategy = next((item for item in book_strategies() if item.key == study.strategy_key), None)
    indicators = pd.DataFrame(index=bars.index)
    kinds: dict[str, str] = {}
    if strategy is not None:
        try:
            indicators, kinds = strategy.indicator_frame(bars, resolve_params(strategy.param_space(), study.params))
        except Exception as exc:
            st.warning(f"Indicators could not be drawn: {exc}")
    name_trades = trades[trades["symbol"].astype(str) == symbol] if "symbol" in trades.columns else trades.iloc[0:0]
    st.markdown('<span id="stock-chart-anchor"></span>', unsafe_allow_html=True)
    figure = symbol_price_figure(
        bars,
        name_trades,
        indicators,
        kinds,
        bar_range,
        title=f"{symbol} · {window} · {bar_range}",
        test_start=study.test_start,
    )
    show_chart(figure)
    _bind_chart_swipe()
    st.markdown(f"**Trades in {symbol}**")
    if name_trades.empty:
        st.write("No closed trade on this name in this window.")
    else:
        st.dataframe(_present_trades(name_trades.sort_values("entry_time")), width="stretch", hide_index=True)


def _bind_chart_swipe() -> None:
    """A horizontal swipe on the stock chart steps to the previous or next name."""

    st.iframe(
        """
        <script>
        (function () {
          const doc = window.parent.document;
          if (doc.__deskChartSwipe) return;
          doc.__deskChartSwipe = true;
          let last = 0;
          let start = null;

          function stockChart() {
            const anchor = doc.getElementById("stock-chart-anchor");
            if (!anchor) return null;
            const origin = anchor.getBoundingClientRect().top;
            const charts = [...doc.querySelectorAll('[data-testid="stPlotlyChart"]')]
              .filter((chart) => chart.getBoundingClientRect().top >= origin - 8)
              .sort((a, b) => a.getBoundingClientRect().top - b.getBoundingClientRect().top);
            return charts[0] || null;
          }

          function press(label) {
            const now = Date.now();
            if (now - last < 400) return;
            const anchor = doc.getElementById("chart-nav-anchor");
            if (!anchor) return;
            const top = anchor.getBoundingClientRect().top;
            const button = [...doc.querySelectorAll("button")].find((node) => {
              const text = (node.innerText || "").replace(/\\s+/g, " ").trim();
              if (text !== label) return false;
              const y = node.getBoundingClientRect().top;
              return y >= top - 20 && y < top + 180;
            });
            if (!button || button.disabled) return;
            last = now;
            button.click();
          }

          function onStart(event) {
            if (event.pointerType === "mouse") {
              start = null;
              return;
            }
            const chart = stockChart();
            const point = event.touches ? event.touches[0] : event;
            if (!chart || !point || !chart.contains(event.target)) {
              start = null;
              return;
            }
            start = {x: point.clientX, y: point.clientY, t: Date.now()};
          }

          function onEnd(event) {
            if (!start) return;
            const point = event.changedTouches ? event.changedTouches[0] : event;
            const dx = point.clientX - start.x;
            const dy = point.clientY - start.y;
            const dt = Date.now() - start.t;
            start = null;
            if (dt > 600 || Math.abs(dx) < 56 || Math.abs(dx) < Math.abs(dy) * 1.3) return;
            press(dx < 0 ? "Next" : "Previous");
          }

          doc.addEventListener("touchstart", onStart, {capture: true, passive: true});
          doc.addEventListener("touchend", onEnd, {capture: true, passive: true});
          doc.addEventListener("pointerdown", onStart, true);
          doc.addEventListener("pointerup", onEnd, true);
          doc.addEventListener("keydown", (event) => {
            if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
            const target = event.target;
            const tag = (target && target.tagName) || "";
            if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || (target && target.isContentEditable)) return;
            if (target && target.closest && target.closest('[data-baseweb="select"], [role="listbox"], [role="combobox"]')) return;
            const chart = stockChart();
            if (!chart) return;
            if (target !== doc.body && target !== doc.documentElement && !chart.contains(target)) return;
            event.preventDefault();
            press(event.key === "ArrowRight" ? "Next" : "Previous");
          }, true);
        })();
        </script>
        """,
        height=1,
        width=1,
        alt="chart-swipe",
    )
