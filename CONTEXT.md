# Strategy desk

This app is a bank of book strategies and the backtest report for each one. A strategy is Python in `strategies/`. It is filtered, bought, sold, sized, and charged, then run on the whole Nifty 500. The bank lists every strategy. Opening one shows that strategy's saved report. A switch at the top selects the train or the test window. The charts of the names that traded come next, then that window's performance, the portfolio value, and the trades.

The pages are Bank, Code, and Indicators. Indicators is a study of every entry in `indicators.json`: what it is, how the calculation works, and why it would be used. It does not run a backtest or build a signal. Do not add sample counts, fold counts, or a separate charts page. The indicator search runs when the report is built, on the train window only. The chart switcher belongs inside the report.

## What a strategy file must do

Subclass `Strategy` in `strategies/<key>.py`. Set `key`, `name`, `stages`, `cost_params`, and `param_space`. Implement `signals` and `indicator_frame`.

Stages, in this order:

1. `filter`. A name can be bought only when every condition passes.
2. `buy`. The entry, decided at the close and filled on the next traded open.
3. `sell`. Closes the shares that are still open, on the next traded open.
4. `stop`. Optional. The simulator always freezes a stop from the entry fill. If a stop and a sell fill on the same open, the stop sells first.
5. `size`. Share count is the portfolio value divided by 20, times the size factor, divided by ATR, then the rupee cap, then cash. An order that still does not fit is dropped. The size factor is `risk_per_atr`, and the default is 1.

`cost_params` is the charge list in `engine.book.COST_PARAMS`: slippage, brokerage, STT, exchange, SEBI, stamp, and GST. `param_space` must also include `capital`, `max_position_inr`, `risk_per_atr`, `stop_atr`, `partial_pct`, and `entry_rank`.

Give every indicator setting a default, a minimum, a maximum, and a step. The default is the first sample. The range is what the train window may try. That includes lengths and thresholds on the filter, the buy, and the sell, and the ATR length when the size stage owns it. Capital, the position cap, the size factor, the stop distance, the partial stop percent, entry rank, and the costs stay on their defaults. They are not searched. A fast length has to stay shorter than its slow pair.

`signals(bars, params)` returns one row per bar, on the same index:

| Column | Meaning |
| --- | --- |
| `entry` | Buy is allowed. The filter and the buy are already combined. |
| `exit` | Sell the open shares. |
| `atr` | Wilder ATR used for the stop and the share count. A non-positive value skips the buy. |
| `udr` | Optional rank. `entry_rank` 0 fills the highest value first when several names buy on the same open. |
| `spread_atr` | Optional rank. `entry_rank` 1 uses this. `entry_rank` 2 fills by symbol name. |

`stage_line(key, params)` returns one sentence for that stage at the current settings. The report can use that sentence. If you omit it, the stage `summary` is used, so write the summary as a sentence a reader can check against the code.

The parameters in the report are the locked set from the train window: the chosen indicator settings, plus the file defaults for capital, costs, the stop, and entry rank. The bank does not search them again when the report is opened.

`engine.book.new_strategy_source` is the starting file. `engine.forest.backtest_books` is the backtest. It studies every book strategy in `strategies/`, including a file added later. Each one goes through `engine.forest.study_book`, which chooses the indicator parameters, then `engine.book.run_book` scores that locked set. A file with no stages is not part of the run. Do not send a book strategy through the one-symbol backtest.

## How a report is shown

The universe is every listed Nifty 500 name that has bars, using adjusted prices. Adjusted prices undo splits and bonuses.

The backtest is run outside the portal and saved as one report on the strategy. The portal does not ask for a run and has no backtest button. The bank table shows every saved report: verdict, train and test CAGR, train and test max drawdown, and trade count. Opening a strategy shows that full report.

The split is chronological. The first 70% of sessions is the train window. The last 30% is the test window. Before either window is scored, indicator parameters are chosen on the train window only. That window is split into expanding folds. Each candidate is scored by CAGR / |max drawdown| on those folds. A random forest reads the finished samples and proposes a short extra list. A proposal is kept only when the same folds score it. The test window is not an input to the choice. The winning indicator settings are the strategy's parameters, together with the file's capital, costs, stop, and entry rank. Both windows are then scored once with that locked set. Indicator math for the test window may use earlier bars. Entries start flat at the split. The test score does not change the parameters.

The decision ratio is CAGR / |max drawdown|. Higher is better.

The report opens on the test window and names the train dates and the test dates. A switch at the top selects test or train, and the charts, the statistics, the portfolio value, and the trades follow that choice. The selected window's dates stay with its charts and its performance. The verdict stays visible, because it compares the two windows:

- Holds on test: the test ratio is at least 75% of the train ratio, and the train ratio is positive.
- Decays on test: both are positive and the test ratio is below that bar.
- Does not hold: the train window is positive and the test window is not.
- Weak: neither window pays for its drawdown.
- Test only: the test window is positive and the train window is not. That is not a confirmed edge.

The charts come next. Then the dashboard for the selected window: CAGR, absolute return, investment, closing amount, winning trades, losing trades, average winning-trade return, average losing-trade return, max drawdown, Sortino, Sharpe, and the annualized standard deviation of daily returns.

Under the dashboard, portfolio value through time. Under that curve, the five most recent closed trades in the selected window, and a collapsible list of every closed trade in that window. The locked parameters stay at the bottom. Indicator settings chosen on the train window are marked as the train choice. The rest are marked as the file values. Do not show a search, a forest, fold scores, or sample tables.

## How a chart is shown

Charts are one name at a time, for names that have a closed trade in the selected window. Swipe the chart left for the next name and right for the previous name. Previous and next do the same. Arrow keys do too, when the chart has focus.

`indicator_frame(bars, params)` returns `(frame, kinds)`. `kinds` maps a column name to `overlay` or `oscillator`. Those names are the legend. Overlays are drawn on the candles. Oscillators each get a panel; a column named `RSI.length` is grouped under the panel `RSI`. Use a plain name (`ATR`, `UDR`, `Fast average`) when one line is the whole indicator.

The column index matches the bars. A weekly or monthly value is carried forward from the session that completed it, never backward.

Buys and sells are arrow markers at the fill, not at the signal close. The marks are the selected window's fills. Train buys are green triangles up and train sells are red triangles down. Test buys are blue triangles up and test sells are darker orange triangles down. The marker hover includes the exit reason. The test window is a light blue band on the price panel.

Under the chart, show that name's trades in the selected window: dates, prices, shares, pnl, and reason. The range control collapses candles to week, month, quarter, or year. Markers stay on the fill date.

If `indicator_frame` fails, show the candles and the arrows anyway, and show the error.

## How the indicator study is shown

Indicators reads `indicators.json` through `engine.catalog` and the study notes in `engine.indicator_lessons`. Every catalog id has a lesson. The page can search, filter by the catalog group, and read one indicator or the whole group. One indicator shows what it is, how it works, why you would use it, the inputs, the default parameters, the outputs, and the formula. Previous and next step through the current list. A lesson for an indicator the engine cannot compute from one symbol's bars says what extra input it needs.

## What to leave out

The Indicators page is the study of the catalog. Do not turn it into a signal builder. Do not put a one-symbol laboratory, a rule builder, a saved-run comparison, or a single-parameter sweep on the desk. The indicator search runs when the report is built, on the train window only. One-symbol examples may stay in `strategies/` for tests; they do not appear in the bank because they have no stages.

When you add a strategy, check it with `engine.book.check_book_strategy`. After a backtest, the bank must list it and open its report, including the chart for each name that traded.
