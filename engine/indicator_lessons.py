"""Study notes for every indicator in indicators.json.

Each entry is (what it is, how it works, why you would use it). The catalog
still supplies the name, formula, inputs, and default parameters.
"""

from __future__ import annotations

GROUP_INTROS = {
    "Trend: Moving Averages": (
        "A moving average is a smoothed line of price. It answers where price has been, "
        "on average, over a chosen window. Traders use the slope and the position of price "
        "relative to the line as a trend filter, and they use crossovers of two lengths as a signal."
    ),
    "Trend: Bands and Channels": (
        "A band wraps a moving average with a fixed percentage on each side. "
        "It marks a normal distance from the average so a stretch beyond the band stands out."
    ),
    "Trend: Trend Strength and Direction": (
        "These tools separate direction from strength. Direction says which way price is pushing. "
        "Strength says whether that push is organized enough to treat as a trend, or whether the market is chopping."
    ),
    "Momentum": (
        "Momentum measures how fast price is changing. Oscillators in this group swing around a midpoint "
        "or a fixed range. Extreme readings and turns in the oscillator are used to time entries inside a larger trend, "
        "or to notice when a move is losing speed."
    ),
    "Volatility": (
        "Volatility is the size of price movement, not its direction. These measures size stops, "
        "compare quiet and active periods, and build bands that widen when the market is moving more."
    ),
    "Volume": (
        "Volume is how much was traded. These indicators ask whether price movement was accompanied by participation, "
        "and whether that participation was concentrated near the high or the low of the bar."
    ),
    "Support/Resistance": (
        "Support and resistance are prices where trading has previously stalled or reversed. "
        "The tools here turn prior highs, lows, and ranges into levels a later bar can be compared with."
    ),
    "Price Transforms": (
        "A transform rewrites the bar into another price, a return, or a different kind of chart. "
        "The new series is then easier to smooth, compare, or read as a trend."
    ),
    "Statistical": (
        "These are descriptive statistics of a rolling window: slope, fit, distance from the mean, "
        "and the shape of returns. They quantify a stretch of bars instead of drawing a trading line."
    ),
    "Cycle": (
        "Cycle tools try to separate a repeating swing from the underlying trend. "
        "John Ehlers' filters in this group estimate a dominant cycle length and then smooth or oscillate at that pace."
    ),
    "Market Structure": (
        "Market structure is the sequence of swings, gaps, and session levels. "
        "These markers describe where the day opened, where a prior extreme sits, and whether a swing was broken."
    ),
    "Relative/Breadth": (
        "Relative tools compare one series with another, usually a stock with an index. "
        "Breadth tools summarize how many names are participating in a market move. Both need more than one symbol's bars."
    ),
    "Options": (
        "Option indicators describe implied volatility and the sensitivity of an option's price. "
        "They need an option price, an implied-volatility series, or a full chain."
    ),
    "Futures/Crypto": (
        "These read the futures basis, perpetual-swap funding, or a positioning report. "
        "They describe how crowded a derivatives market is, and they need that extra data."
    ),
    "Candlestick Patterns": (
        "A candlestick pattern is a small shape in one to three bars: body size, shadow length, and the relation to the previous bar. "
        "The shape records a shift between buyers and sellers. Context decides whether that shift is worth acting on."
    ),
    "Chart Patterns": (
        "A chart pattern is a larger shape across many bars, such as a double top or a triangle. "
        "Most of these shapes depend on which swings a person chooses to connect. The N-bar breakout is the mechanical member of the group."
    ),
    "Exhaustion": (
        "Exhaustion counts how long a one-way move has already run. "
        "A long streak or a completed setup count is used as a warning that the push may be late, not as a guarantee that it will reverse."
    ),
    "Mean Reversion": (
        "Mean reversion looks for price that has moved unusually far from a reference and may travel back toward it. "
        "The reference can be the bar's own range, a moving average, or the residual of two related prices."
    ),
    "Risk/Performance": (
        "These scores describe a return series or an equity curve: how much was earned per unit of risk, "
        "how deep the decline from a peak was, and how large a position that risk implies."
    ),
    "Calendar": (
        "Calendar tools attach the clock and the calendar to a return. "
        "They ask whether a weekday, a month, a session window, or a known event lines up with the result."
    ),
    "Sentiment": (
        "Sentiment indicators summarize positioning or a published mood score. "
        "Extreme crowd positioning is often read as a contrary clue, and both entries here need data beyond the price file."
    ),
    "Fundamental": (
        "Fundamental ratios compare a market price with a figure from the accounts, or score the accounts themselves. "
        "They change when filings change, and they need that fundamental data."
    ),
    "Factor": (
        "A factor is a characteristic used to rank a universe of stocks. "
        "Momentum and low volatility can be built from price. Value and quality need fundamentals."
    ),
}

# id -> (what, how, why)
LESSONS: dict[str, tuple[str, str, str]] = {
    "sma": (
        "The simple moving average is the arithmetic mean of the last N prices. It is one line that follows price and dampens day-to-day noise.",
        "On each bar, add the last N values and divide by N. The default window is 20 sessions. Every price in the window has the same weight, and the oldest price leaves the sum as the newest one enters. Along a straight trend the line trails price by about half the window.",
        "Use it as a plain definition of the recent average price. A close above a rising average is a common filter before buying. Bollinger Bands and several other tools are built on this same average.",
    ),
    "ema": (
        "The exponential moving average is a smoothed price that gives more weight to recent bars and never fully drops the older ones.",
        "The smoothing constant is k = 2 / (N + 1). Each bar is k times today's price plus (1 - k) times yesterday's average. With the default of 20, k is about 0.095, so a new price moves the line quickly and then the effect fades. The line starts once N prices exist.",
        "Use it when the recent path should matter more than a price from the far end of the window. Crossovers of a fast and a slow EMA, and the distance between them, are the basis of MACD.",
    ),
    "wma": (
        "The weighted moving average is a mean that gives the newest bar weight N, the one before it weight N - 1, and the oldest bar weight 1.",
        "Multiply each of the last N prices by its weight, add those products, and divide by the sum of the weights, which is N(N + 1) / 2. The default period is 20. A new extreme moves this line faster than a simple average of the same length, and a price older than N bars drops out completely.",
        "Use it when you want a smooth line that still answers the latest bars. The Hull average is built from two of these weighted averages.",
    ),
    "dema": (
        "The double exponential moving average shortens the lag of an ordinary EMA by combining the EMA with an EMA of that EMA.",
        "Compute an EMA of price, then an EMA of that line, both of length N. DEMA is twice the first line minus the second. The subtraction puts back the lag that the second smoothing removed. The default period is 20.",
        "Use it when an EMA of the length you want is too slow for the turn you are trying to see. It still smooths, and it hugs price more closely than the EMA inside it.",
    ),
    "tema": (
        "The triple exponential moving average reduces lag further by mixing three cascaded EMAs.",
        "Let E1 be the EMA of price, E2 the EMA of E1, and E3 the EMA of E2, all of length N. TEMA is 3·E1 - 3·E2 + E3. The default period is 20. The extra stage removes more of the delay that repeated smoothing would add.",
        "Use it for a responsive trend line when a single or double EMA still turns late. Because it tracks tightly, small wiggles show up sooner than they do on a simple average.",
    ),
    "hma": (
        "The Hull moving average is a fast, smooth line built so that lag falls while the line stays even.",
        "Take a weighted average of length N/2, double it, and subtract a weighted average of length N. Smooth that result with a weighted average of length equal to the square root of N. The default period is 20. The long subtraction cancels lag, and the short final average removes the roughness that subtraction creates.",
        "Use it when you want a moving average that turns with the trend and does not stair-step. A change in its slope is a common way to mark the end of a swing.",
    ),
    "smma": (
        "The smoothed moving average, also called Wilder's RMA, is an exponential average whose smoothing constant is 1/N.",
        "Today's value is the previous average times (N - 1), plus today's price, all divided by N. That is an EMA with k = 1/N, which is slower than the usual 2/(N + 1). The default period is 14, the length Wilder used.",
        "Use it when you are reading Wilder's indicators on their own terms. RSI, ATR, and ADX all smooth with this average, so a matching length keeps those studies consistent.",
    ),
    "kama": (
        "Kaufman's adaptive moving average speeds up when price travels in one direction and slows down when price chops.",
        "The efficiency ratio is the net change over N bars divided by the sum of the absolute bar-to-bar changes. A straight move scores near 1 and a noisy path scores near 0. That ratio sets a smoothing constant between a fast EMA and a slow EMA, and the constant is squared so the change in speed is gradual. Defaults are period 10, fast 2, and slow 30.",
        "Use it when one fixed speed is wrong for both trends and ranges. In a trend it can act as a trailing line. In a range it stays flat so small crosses are fewer.",
    ),
    "vidya": (
        "The variable index dynamic average is an EMA whose speed is scaled by the absolute Chande momentum oscillator.",
        "The base constant is 2/(N + 1). It is multiplied by the absolute value of a CMO, which sits between 0 and 1. Strong momentum makes the average follow price. Quiet momentum makes it almost stick. Defaults are period 14 and a CMO length of 9.",
        "Use it when the average should tighten up during a strong move and go quiet when momentum fades. The CMO is the switch that decides which of those two states you are in.",
    ),
    "alma": (
        "The Arnaud Legoux moving average is a Gaussian-weighted mean. An offset slides the peak of the weights toward the recent bars.",
        "Each price in the window gets a weight from a bell curve. The center of the bell sits at offset times (N - 1), and the width of the bell is N divided by sigma. The average is the weighted sum divided by the sum of the weights. Defaults are period 9, offset 0.85, and sigma 6, so the heaviest weight is near the newest bars and the curve is fairly narrow.",
        "Use it when you want a smooth line whose lag you can set directly. A higher offset responds faster. A higher sigma spreads the weight and smooths more.",
    ),
    "zlema": (
        "The zero-lag EMA is an ordinary EMA run on a price that has already been pushed forward to cancel lag.",
        "The lag of an EMA is about (N - 1) / 2 bars. The adjusted price is twice the current price minus the price that many bars ago. An EMA of that adjusted series is the ZLEMA. The default period is 20.",
        "Use it when you like the shape of an EMA but want the line nearer the current price. The adjustment can overshoot on a sharp reversal, which is the cost of removing the lag.",
    ),
    "t3": (
        "Tillson's T3 is a very smooth, low-lag average made by cascading six EMAs and mixing them with a volume factor.",
        "Six EMAs are stacked, each smoothed with the same period. Four coefficients built from the volume factor v combine the last four of those lines. The default period is 5 and v is 0.7. A higher v smooths more. A lower v follows price more closely. The name refers to the mixing weight, not to traded volume.",
        "Use it when you want a trend line that ignores small noise and still turns without a long delay. It suits a slow filter on daily bars.",
    ),
    "trima": (
        "The triangular moving average is a simple average of a simple average. The double pass puts the most weight on the middle of the window.",
        "Both passes use a length of ceil((N + 1) / 2). The default N is 20, so each pass is 11 bars. Weights rise toward the center of the original window and fall toward both ends. The line is smoother than one SMA, and it lags by more.",
        "Use it when smoothness matters more than a quick turn. It is a heavy trend filter, useful for defining a slow background direction.",
    ),
    "vwma": (
        "The volume-weighted moving average is an average of closes in which each close is weighted by that bar's volume.",
        "Over the last N bars, add close times volume, and divide by the sum of volume. The default period is 20. A high-volume bar pulls the line toward its close. A quiet bar barely moves it. If volume is zero across the window the average is undefined.",
        "Use it when you want the average price that was actually traded, rather than an average that treats a thin bar like a busy one. It is a natural companion to a plain SMA: a gap between them shows where volume agreed with price.",
    ),
    "lsma": (
        "The least squares moving average is the endpoint of a straight line fitted through the last N prices.",
        "Fit an ordinary least-squares line of price on the bar index inside the window. The value plotted on the last bar is the intercept plus the slope times (N - 1). The default window is 25. The line can sit ahead of a simple average because it follows the fitted direction.",
        "Use it when the question is where a straight trend would put price today. A close through the line says price has left the regression path of the recent window.",
    ),
    "mcginley": (
        "The McGinley dynamic is a moving average that adjusts its own speed from how far price has pulled away from it.",
        "The update adds (price - previous average) divided by (0.6 · N · (price / previous average) to the fourth). When price runs above the line, the denominator shrinks and the line catches up. When price is near the line, the line moves slowly. The default period is 14.",
        "Use it as a trailing line that tries to avoid the small crosses a fixed average makes in a fast market. It was built to track price more closely when the market accelerates.",
    ),
    "frama": (
        "The fractal adaptive moving average is an EMA whose speed comes from how rough the recent price path is.",
        "The fractal dimension compares the range of two half-windows with the range of the full window. A straight path has a dimension near 1 and a noisy path is higher. That dimension becomes the EMA alpha, limited to the range 0.01 to 1. The default period is 16, and the window is split in half, so an even length fits the construction.",
        "Use it when the average should follow a clean trend and hold still while the path is jagged. The dimension is the same idea as a trend-versus-chop switch.",
    ),
    "jma": (
        "The Jurik-style average is a low-lag, low-noise adaptive line. The original Jurik formula is proprietary, so this is the usual open approximation.",
        "An adaptive band around recent volatility sets how fast the line may move, and a phase input leans the line toward leading or lagging. The default period is 14 and the phase is 0. The result is a smooth curve that still turns with a sustained move. Two implementations of the approximation can differ in the first bars.",
        "Use it when you want one average that stays quiet in noise and catches a trend without a separate fast and slow pair. Treat the level as an approximation of Jurik's line.",
    ),
    "ma_envelope": (
        "A moving-average envelope is a pair of bands set a fixed percentage above and below a moving average.",
        "The middle line is an SMA of the chosen period, default 20. The upper band is that average times (1 + percent/100) and the lower band is the average times (1 - percent/100). The default percent is 2.5. The distance stays a constant fraction of the average, so it does not widen when volatility rises.",
        "Use it when you want a simple stretch band: a close outside the envelope is far from the average by a rule you chose in advance. It is the fixed-percent cousin of Bollinger Bands.",
    ),
    "macd": (
        "MACD is the difference between a fast and a slow exponential average, plus a signal line and a histogram of the gap between them.",
        "Subtract the 26-period EMA from the 12-period EMA. Smooth that difference with a 9-period EMA to make the signal. The histogram is MACD minus the signal. All three use the close. A crossing of the zero line means the fast average has crossed the slow one.",
        "Use it to see trend direction and whether that trend is gaining or losing speed. The histogram shrinking toward zero is an early sign that the two averages are coming back together.",
    ),
    "ppo": (
        "The percentage price oscillator is MACD divided by the slow average, so the reading is a percent and can be compared across prices.",
        "It is 100 times (fast EMA minus slow EMA) divided by the slow EMA, with the same 12, 26, and 9 defaults as MACD. A signal EMA and a histogram are built the same way. A stock at 50 and a stock at 5,000 can share a scale.",
        "Use it when you rank names or set a threshold that should mean the same thing on a low-priced and a high-priced stock. The shape of the signal matches MACD. The units are percent.",
    ),
    "adx": (
        "The average directional index measures how strong a trend is, on a scale that is usually read from 0 to 100. It does not say whether the trend is up or down. The plus and minus directional indicators carry the direction.",
        "An up move is today's high minus yesterday's high, and it counts only when it beats the down move and is positive. The down move is defined the same way from the lows. Each is Wilder-smoothed and divided by ATR to make +DI and -DI. DX is 100 times the absolute gap between the two DIs, divided by their sum. ADX is a Wilder average of DX. The default period is 14.",
        "Use ADX to decide whether a trend rule is even appropriate. A rising ADX means the directional gap is widening. A low or falling ADX means the market is not trending, so a crossover system will be chopped up. Read +DI above -DI as upward pressure.",
    ),
    "adxr": (
        "The average directional movement rating is a slower version of ADX that averages today's ADX with the ADX from N bars ago.",
        "Compute ADX with the default period of 14, then average it with the ADX value one period earlier in the series of ADX readings. The result moves less than ADX itself and confirms that strength has persisted.",
        "Use it when a single ADX spike is too jumpy for the decision. A high ADXR says trend strength has been present at both ends of the window.",
    ),
    "dmi": (
        "The directional movement index is the pair of lines underneath ADX: +DI and -DI.",
        "They are built exactly as in ADX. +DI is smoothed upward movement divided by ATR, and -DI is smoothed downward movement divided by ATR, both scaled to 100. The default period is 14. ADX is then derived from the gap between them, so these two lines are the raw directional story.",
        "Use a cross of +DI above -DI as a shift toward upward pressure, and the opposite cross as a shift toward downward pressure. Check ADX if you also need to know whether the cross happened inside a real trend.",
    ),
    "aroon": (
        "Aroon measures how recently the highest high and the lowest low occurred inside a window of N bars.",
        "Aroon Up is 100 times (N minus the number of bars since the highest high) divided by N. Aroon Down does the same with the lowest low. The default period is 25. A reading of 100 means the extreme happened on the current bar. A reading of 0 means it happened at the far end of the window.",
        "Use it to see which side is making fresh extremes. Up near 100 and Down near 0 is the signature of an advance. The two lines falling together means neither side has made a new extreme lately.",
    ),
    "aroon_osc": (
        "The Aroon oscillator is Aroon Up minus Aroon Down.",
        "It uses the same lookback, default 25. The result runs from -100 to +100. Positive values mean the highest high is more recent than the lowest low.",
        "Use it when you want one number for the Aroon comparison. A cross through zero is the same event as Aroon Up crossing Aroon Down.",
    ),
    "psar": (
        "Parabolic SAR is a trailing stop that accelerates toward price as a trend extends, and flips to the other side when price touches it.",
        "The stop starts on the opposite side of price from the trend. Each bar it moves closer by an acceleration factor times the distance to the extreme price of the trend. The factor starts at 0.02 and rises by 0.02 each time a new extreme prints, up to 0.20. In an uptrend the stop only rises. A trade through the stop reverses the side.",
        "Use it as a trailing exit that tightens by itself the longer a trend runs. The flip is both the exit of one side and the start of the other, so in a range it reverses often.",
    ),
    "supertrend": (
        "SuperTrend is an ATR trailing line that sits below price in an uptrend and above price in a downtrend, and flips when the close crosses it.",
        "The basic bands are the midpoint of high and low, plus and minus a multiple of ATR. The final band only moves in the direction of the trend: the lower band can rise and the upper band can fall. A close through the active band flips the side. Defaults are an ATR length of 10 and a multiple of 3. The second output is the direction after the flip.",
        "Use it as a one-line trend filter and a trailing stop. A wider multiple ignores more noise. A narrower multiple flips sooner and rides less of the trend.",
    ),
    "ichimoku": (
        "Ichimoku is a set of five lines that put trend, momentum, and support or resistance on one chart. Two of the lines are shifted forward to form a cloud.",
        "Tenkan is the midpoint of the highest high and lowest low over 9 bars. Kijun is the same midpoint over 26. Senkou A is the midpoint of Tenkan and Kijun, shifted 26 bars forward. Senkou B is the midpoint of the 52-bar high and low, also shifted forward 26. Chikou is the close shifted 26 bars backward. Price relative to the cloud, and Tenkan relative to Kijun, are the main readings.",
        "Use it when you want a full trend map from one system: the cloud as a zone, the two midpoints as a cross, and the lagged close as a check against price 26 bars ago. The defaults are the daily-chart lengths from the original system.",
    ),
    "vortex": (
        "The vortex indicator looks for the start of a trend by comparing upward and downward travel with true range.",
        "Upward movement is the absolute distance from today's high to yesterday's low. Downward movement is the absolute distance from today's low to yesterday's high. Each is summed over N bars and divided by the sum of true range. The default period is 14. VI+ above VI- says the upward travel is larger.",
        "Use a cross of the two lines as a trend-start marker. The lines are normalized by range, so a wide bar and a narrow bar are put on the same scale.",
    ),
    "trix": (
        "TRIX is the percent rate of change of a triple-smoothed exponential average. The three smoothings strip out the short swings.",
        "Smooth the close three times with an EMA of length N, default 15. TRIX is 100 times the change in that final line divided by the previous value of the line. It oscillates around zero. A signal line is sometimes added. This catalog entry is the rate of change itself.",
        "Use it when you want momentum of the trend and not momentum of the noise. A cross of zero means the triple-smoothed line has turned.",
    ),
    "mass_index": (
        "The mass index looks for a reversal by measuring whether the high-low range is expanding, a pattern called the reversal bulge.",
        "Take an EMA of the high-low range, then an EMA of that EMA, both of length 9 by default. Divide the first by the second, and sum those ratios over 25 bars. The ratio rises when the range is widening faster than its own average. A bulge toward 27 and a fall back through 26.5 is the classical warning.",
        "Use it to notice range expansion that often shows up before a turn. It does not say which way the turn will go. Direction comes from price.",
    ),
    "cci": (
        "The commodity channel index measures how far the typical price has moved from its own average, in units of mean deviation.",
        "Typical price is (high + low + close) / 3. Subtract its SMA, default length 20, and divide by 0.015 times the mean absolute deviation of typical price over the same window. The 0.015 scale was chosen so that most readings fall between -100 and +100.",
        "Use a reading beyond +100 or -100 as a sign that price is extended relative to its recent average. Traders also use a cross back inside those levels as a mean-reversion trigger, and a cross through zero as a momentum shift.",
    ),
    "rsi": (
        "The relative strength index compares the size of up closes with the size of down closes and scales the result from 0 to 100.",
        "Separate the close-to-close changes into gains and losses. Smooth each with Wilder's average over N bars, default 14. RS is the average gain divided by the average loss. RSI is 100 minus 100 / (1 + RS). If there is no loss in the window, RSI is 100. Around 50 the two sides are in balance.",
        "Use it to judge whether recent closes have been dominated by gains or by losses. Readings near 70 and 30 are the usual stretch marks. A turn in RSI while price is still pushing is the classic divergence clue.",
    ),
    "stoch_rsi": (
        "Stochastic RSI applies the stochastic formula to RSI, so it measures where today's RSI sits inside its own recent range.",
        "Compute RSI, default length 14. Over the last 14 RSI values, locate today's RSI between the lowest and the highest. Smooth that result over 3 bars for %K and smooth again over 3 for %D. Because RSI is already bounded, this second pass swings from oversold to overbought faster than RSI itself.",
        "Use it when RSI spends a long time in the middle and you want a more sensitive timing line. The extra sensitivity also means more crosses, so it is usually read inside a trend filter.",
    ),
    "stoch": (
        "The stochastic oscillator locates the close inside the high-low range of the last N bars.",
        "%K is 100 times (close minus the lowest low) divided by (highest high minus lowest low), over the K window, default 14. %D is a 3-period average of %K. In this library %K is the raw location and %D is its smooth. A close at the top of the range scores near 100. A close at the bottom scores near 0.",
        "Use it to see whether buyers are finishing the bar near the highs or sellers near the lows. Crosses of %K and %D inside the 20 and 80 zones are the usual timing marks.",
    ),
    "williams_r": (
        "Williams %R is the stochastic location of the close, flipped so the scale runs from -100 to 0.",
        "It is -100 times (highest high minus close) divided by (highest high minus lowest low) over N bars, default 14. Zero means the close is at the high of the window. -100 means the close is at the low. The -20 and -80 levels match the stochastic 80 and 20 levels.",
        "Use it for the same range-location reading as the stochastic, on the scale Williams published. A rise from below -80 says the close has left the bottom of the range.",
    ),
    "roc": (
        "Rate of change is the percentage change in price over N bars.",
        "It is 100 times (price today minus price N bars ago) divided by the older price. The default lookback is 12. The result is zero when price is unchanged, positive after a rise, and negative after a fall. It is not smoothed.",
        "Use it as a direct momentum reading in percent. A cross through zero means price has round-tripped the lookback. Comparing ROC across names is fair because the unit is a percent.",
    ),
    "mom": (
        "Momentum here is the raw price change over N bars, in price points.",
        "Subtract the price from N bars ago from today's price. The default lookback is 10. A stock that rises from 100 to 110 scores 10. The same percentage move on a stock at 1,000 scores 100, so the number is not comparable across prices.",
        "Use it when you want the point change itself, for example inside a system that already works in price units. For a cross-sectional comparison, prefer the percent rate of change.",
    ),
    "cmo": (
        "The Chande momentum oscillator compares the sum of up changes with the sum of down changes, without Wilder smoothing, and scales the result from -100 to +100.",
        "Over N bars, default 14, add the positive close-to-close changes and add the absolute negative changes. CMO is 100 times (up sum minus down sum) divided by (up sum plus down sum). Equal sums give zero. A window of only gains gives +100.",
        "Use it as a faster cousin of RSI. Because it is a plain sum, a single large bar moves it more. VIDYA uses the absolute value of this oscillator to set its speed.",
    ),
    "tsi": (
        "The true strength index is double-smoothed momentum, divided by double-smoothed absolute momentum, so the scale stays bounded.",
        "Take the one-bar price change. Smooth it with a long EMA, default 25, then a short EMA, default 13. Do the same to the absolute value of the change. TSI is 100 times the first result divided by the second. A signal EMA, default 7, is the trigger line.",
        "Use it when ordinary momentum is too noisy and RSI is too slow. The double smooth keeps the line readable, and a cross of the signal line is the timing mark.",
    ),
    "ultimate_osc": (
        "The ultimate oscillator blends buying pressure over three windows so a turn on one timeframe is checked against the other two.",
        "Buying pressure is the close minus the lower of today's low and yesterday's close. True range for this study is the higher of today's high and yesterday's close, minus that same lower price. Average the ratio of pressure to range over 7, 14, and 28 bars. The oscillator is 100 times (4 times the short average plus 2 times the middle plus the long) divided by 7.",
        "Use it to look for a low that shows up on the short window and is supported by the longer ones. Larry Williams used that agreement to cut the false divergences a single-length oscillator produces.",
    ),
    "awesome_osc": (
        "The awesome oscillator is the gap between a fast and a slow simple average of the bar's midpoint.",
        "The midpoint is (high + low) / 2. Subtract a 34-period SMA of that midpoint from a 5-period SMA. There is no extra signal line in the base reading. Positive values mean the short midpoint average is above the long one.",
        "Use it as a simple momentum histogram of the median price. A cross through zero is a shift in that momentum. The saucer and twin-peaks readings are patterns on this same histogram.",
    ),
    "accel_osc": (
        "The accelerator oscillator is the awesome oscillator minus its own moving average. It shows whether momentum is speeding up or slowing down.",
        "Build the awesome oscillator with the 5 and 34 midpoint averages, then subtract a 5-period SMA of that oscillator. A positive bar means the awesome oscillator is above its own recent average, so momentum is accelerating upward relative to that average.",
        "Use it when the awesome oscillator's level is not enough and you want the change in that momentum. A color change or a cross of zero is the usual trigger, taken in the direction of the awesome oscillator.",
    ),
    "dpo": (
        "The detrended price oscillator removes a moving average so the leftover swing is easier to see.",
        "Shift the window so the average is centered: subtract an SMA of length N from the price of the bar that sits about N/2 + 1 bars back. The default period is 20. The line is not a forecast. It is past price with the trend of that window taken out, which is why the plot is displaced.",
        "Use it to estimate the length of a cycle. Peaks and troughs in the DPO are the swings that a trend line would have hidden. Because of the displacement, the latest bars are incomplete.",
    ),
    "kst": (
        "Know Sure Thing is a weighted sum of four smoothed rates of change, from a short lookback to a long one.",
        "The default rate-of-change lengths are 10, 15, 20, and 30. Each is smoothed with an SMA, defaults 10, 10, 10, and 15. The weights are 1, 2, 3, and 4, so the longest swing dominates. A 9-period SMA of the sum is the signal line.",
        "Use it for a broad momentum confirmation that looks at several horizons at once. A cross of the signal line says the weighted blend has turned. It is slow on purpose.",
    ),
    "coppock": (
        "The Coppock curve is a long-term momentum sum designed to mark major lows.",
        "Add a 14-period rate of change to an 11-period rate of change, then smooth that sum with a 10-period weighted moving average. The original was monthly. On daily bars the same lengths are a much shorter cycle, so the reading is faster than Coppock's original use.",
        "Use an upturn from below zero as the classical long-term buy clue. It was built for bottoms on a market index, and a downturn is a weaker signal in the original method.",
    ),
    "connors_rsi": (
        "Connors RSI averages three short-term ranks: a 3-period RSI of price, a 2-period RSI of the up/down streak, and the percentile rank of the one-bar return.",
        "RSI of the close uses length 3. The streak length rises while closes keep the same direction and resets when they flip, and that streak is passed through an RSI of length 2. The third piece is the percent of the last 100 one-bar returns that sit below today's return. The three are averaged. All three defaults can be changed.",
        "Use it for short-horizon mean reversion. Very low readings mean price, the streak, and the latest return are all pressed to the downside together. Connors-style pullback rules buy that stretch and exit when it snaps back.",
    ),
    "rvi": (
        "The relative vigor index compares the close-open body with the high-low range, on the idea that closes finish near the high during an advance.",
        "A symmetric weighted average is applied to (close - open) and to (high - low). RVI is an SMA of the first, divided by an SMA of the second, default length 10. A signal line is a short weighted mix of recent RVI values. The result is positive when bodies are up relative to the range.",
        "Use a cross of the signal line as a momentum turn. It belongs with the close location inside the bar: a string of strong up bodies lifts it.",
    ),
    "rmi": (
        "The relative momentum index is RSI with the one-bar change replaced by the change over a longer momentum length.",
        "Gains and losses are measured from the close N momentum bars ago, default 5, and then smoothed over the RSI period, default 14. The rest of the RSI formula is unchanged. A longer momentum length skips the smallest wiggles.",
        "Use it when standard RSI flips too often because each bar's change is small and noisy. Lengthening the momentum step asks whether the larger swing is up or down.",
    ),
    "stc": (
        "The Schaff trend cycle runs MACD through a stochastic calculation twice, so a slow trend difference becomes a faster oscillator between 0 and 100.",
        "Build MACD with a fast length of 23 and a slow length of 50. Apply a stochastic of length 10 to that MACD, smooth it, apply the stochastic again, and smooth once more. The default cycle length is 10. The double pass is what makes the line travel from low to high faster than MACD.",
        "Use a rise from the 25 zone or a fall from the 75 zone as a timing mark in the direction of the larger trend. It is built to turn earlier than a MACD cross.",
    ),
    "pmo": (
        "The price momentum oscillator is a rate of change smoothed twice.",
        "Take a one-bar rate of change, scale it by 10, smooth with an EMA of length 35, and smooth that result with an EMA of length 20. A signal EMA of length 10 is the trigger. The long first smooth is what keeps the line from spiking on one bar.",
        "Use a cross of the signal line, or a cross of zero, as a smoothed momentum turn. It is a quieter alternative to a raw rate of change.",
    ),
    "fisher": (
        "The Fisher transform reshapes a bounded price location so extremes become sharp turning points.",
        "Locate the midpoint of the bar inside the high-low range of N bars, default 10, and scale it to sit between -1 and 1. Blend in a fraction of the previous value so the series is stable, then apply one-half times the log of (1 + x) / (1 - x). A one-bar lag of that result is the trigger line.",
        "Use a turn at an extreme, or a cross of the trigger, to mark a possible reversal. The transform is designed so that quiet prices stay near zero and a press against the edge of the range spikes.",
    ),
    "inertia": (
        "Inertia is a linear-regression smooth of the relative volatility index.",
        "The relative volatility index treats standard deviation the way RSI treats price change. Inertia then fits a regression over the default 20 bars and plots the fitted value. The RVI length defaults to 14. Smoothing with a regression keeps the line steadier than the raw RVI.",
        "Use a cross of 50 as the trend-quality switch the author intended: above 50 the volatility is dominated by up days, below 50 by down days. The regression is there so the cross is less twitchy.",
    ),
    "squeeze_mom": (
        "Squeeze momentum flags a volatility compression, when Bollinger Bands sit inside Keltner Channels, and plots a momentum value for the direction of the eventual release.",
        "Bollinger Bands use length 20 and 2 standard deviations. Keltner Channels use length 20 and 1.5 ATR. The squeeze is on while the Bollinger band is inside the Keltner band. Momentum is a linear regression of the close minus the average of the window's midpoint and the close's average. A release is the bar where the squeeze turns off.",
        "Use the on/off flag to find a market that has gone quiet, and use the momentum histogram to judge which way it is leaning as the quiet ends. The release is the event. The histogram is the direction clue.",
    ),
    "elder_ray": (
        "Elder Ray splits a bar into bull power and bear power, each measured from an exponential average.",
        "Bull power is the high minus an EMA of the close, default length 13. Bear power is the low minus that same EMA. Bull power is positive when the high reaches above the average. Bear power is negative when the low reaches below it.",
        "Use it to see whether buyers or sellers are still able to push price away from the average. Elder's buy looks for a rising average, negative but rising bear power, and a bull-power confirmation.",
    ),
    "elder_impulse": (
        "The Elder impulse system colors each bar from the slope of an EMA and the slope of the MACD histogram.",
        "The EMA length is 13. MACD uses 12, 26, and 9. If both the EMA and the histogram are higher than the previous bar, the impulse is green. If both are lower, it is red. Any disagreement is blue. This catalog stores that state as a code rather than a color.",
        "Use green as permission to hold or add in an uptrend, and red as a reason to stand aside or exit. Blue is the in-between state where the trend line and the momentum histogram disagree.",
    ),
    "bop": (
        "Balance of power compares the close-open body with the high-low range of the same bar.",
        "It is (close - open) divided by (high - low), then smoothed with an SMA, default 14. The raw value is +1 when the bar opens at the low and closes at the high, and -1 when it does the opposite. A doji or a zero range is near zero or undefined.",
        "Use a sustained positive reading as a sign that bars are closing in their upper half with upward bodies. A cross of zero is a shift in who has been finishing the bars.",
    ),
    "qstick": (
        "Qstick is a moving average of the candle body, close minus open.",
        "Subtract the open from the close on each bar and take an SMA, default length 8. Positive values mean the recent bodies have been up on average. The shadows are ignored, so this is only about where the bar settled relative to where it opened.",
        "Use a cross of zero as a simple shift from down bodies to up bodies. It is a compact way to quantify a run of candle color.",
    ),
    "chaikin_vol": (
        "Chaikin volatility is the rate of change of a smoothed high-low range.",
        "Take an EMA of (high - low), default length 10, then a rate of change of that EMA over 10 bars, in percent. A rising reading means ranges are widening. A falling reading means they are narrowing. Volume is not in this particular Chaikin formula.",
        "Use a surge as a warning that the market is becoming more active, often around a break from a quiet period. A collapse in the reading marks a market that is going quiet.",
    ),
    "tr": (
        "True range is the width of a bar, including a gap from the previous close.",
        "It is the largest of three numbers: high minus low, the absolute gap from high to the previous close, and the absolute gap from low to the previous close. There is no lookback. One bar, one range.",
        "Use it whenever a high-low range would miss a gap. ATR is a smoothed average of this series, and many stops are a multiple of that average.",
    ),
    "atr": (
        "Average true range is Wilder's smoothed average of true range. It is the standard measure of how far price typically moves, and the usual unit for a stop distance.",
        "Build true range for each bar, then smooth it with the Wilder average over N bars, default 14. The result is in price points. A higher ATR means recent bars, including gaps, have been wider.",
        "Use it to place a stop and to size a position: a fixed rupee risk divided by ATR gives a smaller position when the market is wider. This desk uses it that way. It is also the width unit inside SuperTrend, Keltner Channels, and the chandelier exit.",
    ),
    "natr": (
        "Normalized ATR is average true range as a percentage of the close, so names at different prices can be compared.",
        "Divide ATR by the close and multiply by 100. The default ATR length is 14. A reading of 2 means the smoothed true range is 2 percent of price.",
        "Use it to rank a universe by how violent the bars are, or to set a percent-style threshold that means the same thing on every name. Raw ATR cannot do that comparison.",
    ),
    "bbands": (
        "Bollinger Bands are a simple moving average with a band on each side at a chosen number of standard deviations.",
        "The middle line is an SMA, default 20. The band width is the standard deviation of the same window, default 2, times that deviation. The upper band is the average plus the width and the lower band is the average minus the width. Both the average and the width use the closes in the window. In code this indicator is called bollinger, and the middle output is mid.",
        "Use the bands to see whether price is quiet or stretched relative to its own recent volatility. A squeeze is a narrow width. A close outside a band is a large deviation. The bands adapt, unlike a fixed-percent envelope.",
    ),
    "bb_percent_b": (
        "Bollinger %B says where price sits between the lower and upper band.",
        "It is (price - lower band) divided by (upper band - lower band), using the same 20 and 2 defaults as the bands. A reading of 1 is the upper band, 0 is the lower band, and 0.5 is the middle. A reading above 1 or below 0 means price is outside the bands.",
        "Use it when a rule needs a number instead of a visual touch of the band. Mean-reversion rules often buy a very low %B and trend rules sometimes buy a %B that holds above 1.",
    ),
    "bb_width": (
        "Bollinger bandwidth is the distance between the upper and lower bands, divided by the middle band.",
        "Subtract the lower band from the upper and divide by the SMA. With the default of 2 standard deviations this is about four standard deviations divided by the average, so it is a relative width. Low readings are squeezes. High readings are wide, active markets.",
        "Use a very low width as a setup that volatility has compressed, and a rise from that low as the release. The level of a low width depends on the name, so it is usually compared with its own history.",
    ),
    "keltner": (
        "Keltner Channels are an exponential average with bands a multiple of ATR above and below it.",
        "The middle line is an EMA of the close, default 20. The band is a multiple of ATR, default ATR length 10 and multiple 2. ATR responds to gaps, so the channel widens when ranges or gaps expand. This is the modern ATR version of the channel.",
        "Use it as a trend channel: a close through the upper band is a strength break, and the middle line is a trailing reference. The squeeze indicator treats a Bollinger band inside this channel as compression.",
    ),
    "donchian": (
        "Donchian Channels are the highest high and the lowest low over N bars, with a midpoint between them.",
        "The upper line is the highest high of the window and the lower line is the lowest low. The middle is their average. The default period is 20. In this library the channel is taken from the prior bars only, so today's close can break the channel of the bars behind it. That shift is what makes a breakout test possible.",
        "Use a close through the upper channel as a breakout entry, which is the turtle rule, and the lower channel as the failure level. The width of the channel is itself a volatility reading.",
    ),
    "starc": (
        "STARC bands are Stoller's average-range channels: a short simple average with ATR bands around it.",
        "The middle is an SMA of the close, default length 6. The bands sit a multiple of ATR away, default ATR length 15 and multiple 2. The short average makes the center quick, and the longer ATR keeps the width from jumping on one bar.",
        "Use a tag of the band as a stretch within a short-term trend. Stoller used the bands as a place to fade an extreme or to trail, depending on the larger direction.",
    ),
    "chandelier": (
        "The chandelier exit is an ATR stop hung from the highest high for a long and from the lowest low for a short.",
        "The long exit is the highest high over N bars minus a multiple of ATR. The short exit is the lowest low plus that same distance. Defaults are 22 bars and a multiple of 3. As a new high prints, the long stop rises with it, always three ATR below the recent peak.",
        "Use it as a trailing stop that gives the trade a fixed amount of room from the best price of the trend. A wider multiple stays in longer. The period decides which peak the stop is hung from.",
    ),
    "hist_vol": (
        "Historical volatility is the annualized standard deviation of log returns. It is what traders mean by realized volatility.",
        "Take the natural log of each close divided by the previous close. The standard deviation of those log returns over N bars, default 20, is multiplied by the square root of the annualization factor, default 252 for daily bars. The result is a yearly volatility figure.",
        "Use it to compare current movement with a name's own past, or with an implied volatility if you have one. Position sizing that targets a volatility uses this kind of number.",
    ),
    "stddev": (
        "Standard deviation is the typical distance of price from its average over the window, in price points.",
        "Subtract the mean of the last N prices, default 20, square those gaps, average them, and take the square root. This is the population form, dividing by N. It is the same width unit Bollinger Bands use.",
        "Use it when a rule needs dispersion itself: a break of a number of deviations, or a ranking of names by how widely they swing in points.",
    ),
    "rvi_vol": (
        "The relative volatility index runs the RSI formula on standard deviation, split by whether price closed up or down.",
        "Compute a standard deviation of price, default length 10. On up closes that deviation counts as a gain, and on down closes it counts as a loss. Those two series are smoothed as in RSI, default length 14. The result sits between 0 and 100.",
        "Use a reading above 50 as a sign that the larger ranges are happening on up closes, which is the original trend-confirmation use. Inertia is this series smoothed again.",
    ),
    "ulcer": (
        "The ulcer index measures downside volatility only: how deep and how long price sits below its recent peak.",
        "Over N bars, default 14, find the running peak of the close, express each bar as a percent below that peak, square those percents, average them, and take the square root. Upside movement does not add to the score. A calm rise scores near zero.",
        "Use it when you care about the pain of drawdowns and a two-sided standard deviation would treat a rally as risk. It is a direct way to rank how hard a name has been to hold.",
    ),
    "parkinson": (
        "Parkinson volatility estimates variance from the high and the low, which carry more information about the path than the close alone.",
        "For each bar take the squared log of high over low. Average that over N bars, default 20, divide by 4 times the natural log of 2, and take the square root. The estimator assumes no drift and no opening gaps. A gap-heavy series biases it.",
        "Use it when you want a volatility number that sees the intrabar range. On daily bars with small gaps it is a tighter estimate than close-to-close volatility.",
    ),
    "garman_klass": (
        "Garman-Klass volatility uses the open, high, low, and close, and is more efficient than a close-to-close estimate when the open is available.",
        "Each bar contributes 0.5 times the squared log of high over low, minus (2·ln 2 - 1) times the squared log of close over open. Average those values over N bars, default 20, and take the square root. It still assumes the open continues from the previous close, so gaps are a weakness.",
        "Use it for a range-based volatility that also accounts for where the bar opened and closed. Prefer Yang-Zhang when overnight gaps are a large part of the move.",
    ),
    "rogers_satchell": (
        "Rogers-Satchell volatility uses the open, high, low, and close and stays valid when price has a drift.",
        "Each bar contributes the log of high over close times the log of high over open, plus the log of low over close times the log of low over open. Average over N bars, default 20, and take the square root. Drift does not bias this estimator. Opening gaps still sit outside it.",
        "Use it when the market is trending and a Parkinson or Garman-Klass estimate would be distorted by that drift. It is one of the two pieces inside Yang-Zhang.",
    ),
    "yang_zhang": (
        "Yang-Zhang volatility combines overnight variance, open-to-close variance, and Rogers-Satchell variance so gaps and drift are both handled.",
        "Overnight variance comes from the log gap between today's open and yesterday's close. Open-to-close variance comes from the log of close over open. Rogers-Satchell covers the intrabar path. A weight k mixes the open-to-close piece with Rogers-Satchell, and the overnight piece is added in full. The default window is 20.",
        "Use it as the general daily volatility estimator when bars gap, which cash equities do. It is the one to reach for before the simpler range estimators if the series has overnight jumps.",
    ),
    "obv": (
        "On-balance volume is a running total of volume, added on an up close and subtracted on a down close.",
        "Start at zero. If today's close is above yesterday's, add today's volume. If it is below, subtract today's volume. An unchanged close adds nothing. The level depends on where you started, so the shape and the direction matter more than the number.",
        "Use a disagreement between price and OBV as the clue: price making a new high while OBV fails to is the classic warning that the new high was not confirmed by volume. A rising OBV says volume is landing on up days.",
    ),
    "ad_line": (
        "The accumulation/distribution line is a running total of volume, weighted by where the close sits inside the bar's range.",
        "The money-flow multiplier is ((close - low) - (high - close)) divided by (high - low). It is +1 at the high and -1 at the low. Multiply by volume and add to the previous total. A close in the middle of the range adds almost nothing, however large the volume.",
        "Use it to see whether volume is accumulating on closes near the high or distributing on closes near the low. A divergence from price is read the same way as an OBV divergence, with the extra detail of the close's location.",
    ),
    "cmf": (
        "Chaikin money flow is the accumulation/distribution amount over a window, divided by the volume of that window, so it stays bounded.",
        "Sum the money-flow multiplier times volume over N bars, default 20, and divide by the sum of volume. The result sits between -1 and +1. Positive values mean closes have been in the upper part of the range on the heavier volume.",
        "Use a positive reading as accumulation and a negative reading as distribution over the window you care about. Unlike the cumulative line, a threshold such as zero is meaningful.",
    ),
    "chaikin_osc": (
        "The Chaikin oscillator is a MACD of the accumulation/distribution line.",
        "Subtract a slow EMA of the A/D line from a fast EMA. Defaults are 3 and 10. A cross above zero means the short volume-weighted line has risen above the longer one, so accumulation is improving.",
        "Use it when the cumulative A/D line is too slow to time. The oscillator turns when the flow of volume-weighted closes changes pace.",
    ),
    "mfi": (
        "The money flow index is a volume-weighted RSI. It compares positive money flow with negative money flow.",
        "Typical price is (high + low + close) / 3, and money flow is typical price times volume. A bar whose typical price rose adds to the positive sum, and a bar whose typical price fell adds to the negative sum, over N bars, default 14. MFI is 100 minus 100 / (1 + positive/negative). It sits between 0 and 100.",
        "Use 80 and 20 as the stretch zones, the way you would with RSI, when you want volume to have a vote. A push in price on weak money flow is the divergence people look for.",
    ),
    "vwap": (
        "The volume-weighted average price is the average price traded, weighted by volume, reset on a schedule. On daily bars the natural reset is weak, so the cumulative line is read with care.",
        "Typical price times volume is accumulated and divided by accumulated volume. The usual anchor is the session, which matters on intraday bars. On daily data the running total covers the series unless you reset it. A close above VWAP says the current price is above the volume-weighted average since the anchor.",
        "Use it as the fair-price line institutions actually trade against during a session. On daily bars, prefer an anchored VWAP from a swing or an event if a session reset does not apply.",
    ),
    "avwap": (
        "Anchored VWAP is a volume-weighted average price that starts at a bar you choose, such as a swing low or an earnings day.",
        "From the anchor bar onward, accumulate typical price times volume and divide by accumulated volume. The default anchor is the first bar of the series. Everything before the anchor is blank. The line is the average price paid since that event.",
        "Use it when the relevant average is 'since this happened', not 'over the last N bars'. A return to the anchored line is a common test of whether the move off that event is still intact.",
    ),
    "vwap_bands": (
        "VWAP bands are standard-deviation bands around VWAP, with the deviation weighted by volume.",
        "The center is VWAP. The width is the volume-weighted standard deviation of typical price around that VWAP. The defaults draw bands at 1 and at 2 widths. On a session VWAP the bands widen as the day's range develops.",
        "Use a touch of the outer band as an extension from the volume-weighted fair price. Mean-reversion trades fade that extension. Trend trades treat a ride along the upper band as strength.",
    ),
    "eom": (
        "Ease of movement relates how far the midpoint moved to how much volume it took to move it.",
        "The distance is today's midpoint minus yesterday's midpoint. The box ratio is volume, scaled by the default 100,000,000, divided by the high-low range. Divide distance by that box ratio, then smooth with an SMA, default 14. A large move on light volume makes a large reading.",
        "Use a positive reading as price rising easily, and a negative reading as price falling easily. It highlights moves that did not need much volume, which is a different question from on-balance volume.",
    ),
    "force_index": (
        "The force index multiplies the change in close by volume, then smooths that product.",
        "Today's raw force is (close - previous close) times volume. An EMA, default 13, smooths it. A big up close on big volume is a large positive number. The unsmoothed one-bar version is sometimes used for a very short signal. This entry is the smoothed line.",
        "Use a cross of zero, or a divergence from price, to see whether volume is confirming the direction of the close. Elder used a short force index for timing and a longer one for the larger push.",
    ),
    "nvi": (
        "The negative volume index changes only on bars where volume fell. It is meant to track the quieter days.",
        "Start at a base such as 1,000. When today's volume is below yesterday's, multiply the index by 1 plus the percent change in close. When volume rises or is unchanged, leave the index where it is. Only the quiet days move it.",
        "Use a break of a moving average of NVI as the classical signal that the quiet-day trend has changed. The story attached to it is that quieter days are more informative. The calculation itself is just the volume filter.",
    ),
    "pvi": (
        "The positive volume index changes only on bars where volume rose. It tracks the busier days.",
        "The update matches NVI, with the test flipped: apply the percent change in close only when volume is higher than the day before. Rising-volume days move the line. Falling-volume days do not.",
        "Use it beside NVI. PVI follows the days the crowd showed up. A trend in PVI is the trend of the high-volume sessions.",
    ),
    "pvt": (
        "Price volume trend accumulates volume multiplied by the percent change in close.",
        "Each bar adds volume times (close - previous close) / previous close. An up day of 1 percent on a million shares adds 10,000. Unlike OBV, a large percentage move adds more than a small one, and unlike a simple volume total, direction is signed.",
        "Use the slope, and divergences from price, as a volume confirmation that respects how large the price change was. It sits between OBV, which ignores the size of the move, and an indicator that ignores volume.",
    ),
    "vroc": (
        "Volume rate of change is the percent change in volume over N bars.",
        "It is 100 times (today's volume minus volume N bars ago) divided by that older volume. The default lookback is 14. A spike means today's activity is far above the volume of that earlier bar. One unusual bar in the denominator can distort it.",
        "Use a surge to confirm a breakout or to notice a climax. Pair it with price direction, because the reading itself is only about activity.",
    ),
    "kvo": (
        "The Klinger volume oscillator compares a fast and a slow average of a volume-force quantity that flips with the trend of typical price.",
        "Trend is the sign of the change in typical price. Volume force multiplies volume by a factor built from the daily range and a cumulative range, signed by that trend. Subtract a 55-period EMA of volume force from a 34-period EMA. A 13-period EMA is the signal. The long lengths are intentional.",
        "Use a cross of the signal line as a shift in longer-term money flow. Klinger built it to catch major turns in flow, so the defaults are slow.",
    ),
    "vol_osc": (
        "The volume oscillator is the percent gap between a fast and a slow average of volume.",
        "It is 100 times (fast EMA of volume minus slow EMA of volume) divided by the slow EMA. Defaults are 5 and 10. Positive values mean recent volume is above its slightly longer average.",
        "Use a spike as confirmation that activity has picked up, and a negative reading as a quiet market. It is a volume-only momentum line.",
    ),
    "twiggs": (
        "Twiggs money flow is a Chaikin-style money flow that uses true high and true low, then Wilder smoothing.",
        "True high is the higher of today's high and yesterday's close. True low is the lower of today's low and yesterday's close. The close's position between those two, times volume, is smoothed and divided by smoothed volume. The default length is 21. Gaps are included because of the true range.",
        "Use a positive reading as accumulation and a negative reading as distribution, with overnight gaps counted in the range. It is the gap-aware version of Chaikin money flow.",
    ),
    "vol_profile": (
        "Volume profile bins the recent price range and sums the volume traded in each bin. The point of control is the busiest bin, and the value area is the band of bins around it that holds about 70 percent of the volume.",
        "Over the lookback, default 100 bars, split the high-to-low span into bins, default 24. Each bar's volume is assigned to the bins its range covers. The point of control is the bin with the most volume. The value-area high and low expand outward from it until the share of volume is about 70 percent.",
        "Use the point of control as the price the market traded most, and the value area as the fair zone. A move out of the value area is a break from that accepted range. On daily bars the profile is a swing map.",
    ),
    "mfi_bw": (
        "The market facilitation index is Bill Williams' range per unit of volume.",
        "Divide (high - low) by volume. There is no lookback. A higher number means the bar covered more price per share traded. Williams paired the change in this index with the change in volume to label four states, from a facilitated move to a fade.",
        "Use a rise in the index together with a rise in volume as his 'green' state, where the market is facilitating the move. The raw ratio is also a simple activity-adjusted range.",
    ),
    "oi_change": (
        "Open-interest change is the difference in open interest over N bars. Rising open interest means new contracts are being opened.",
        "Subtract open interest from N bars ago, default 1, from today's open interest. Price up with open interest up is the classical new-buying confirmation. Price up with open interest down is short covering. The series needs an open-interest field the daily equity file does not have.",
        "Use it on futures to judge whether a price move is bringing in new positions or is just positions being closed. The four combinations of price and open interest are the whole reading.",
    ),
    "cvd": (
        "Cumulative volume delta is the running total of aggressive buy volume minus aggressive sell volume.",
        "Each bar adds buy-initiated volume and subtracts sell-initiated volume, usually classified from the tape or from the bid and ask. The line rises when aggressive buying dominates. The equity bars here have only total volume, so the split is unavailable.",
        "Use a divergence between price and delta as an order-flow warning, once you have a feed that separates buyers and sellers. Total volume cannot stand in for that split.",
    ),
    "pivot_classic": (
        "Classic pivot points are support and resistance levels derived from the previous bar's high, low, and close.",
        "The pivot is (high + low + close) / 3 of the prior bar. The first resistance is twice the pivot minus the low, and the first support is twice the pivot minus the high. The second pair adds and subtracts the prior range. A third pair extends one range further. On daily bars these are tomorrow's levels from today's bar.",
        "Use them as a pre-defined map for the next session: the pivot as the balance price, and R1 or S1 as the first places a move often pauses. They are arithmetic, so every name gets a level even on a quiet day.",
    ),
    "pivot_fib": (
        "Fibonacci pivots use the same central pivot and then step out by Fibonacci fractions of the prior range.",
        "The pivot is (high + low + close) / 3. Resistances add 0.382, 0.618, and 1.0 times the prior high-low range. Supports subtract those same fractions. The levels cluster closer to the pivot than the classic R2 and R3.",
        "Use them when you want pivot-style levels spaced by the ratios traders already watch. The first level is nearer than classic R1, so it is a tighter reference.",
    ),
    "pivot_camarilla": (
        "Camarilla pivots are a tight set of intraday levels around the prior close, built for fading and for breakouts.",
        "From the prior close, add and subtract the prior range times 1.1 divided by 12, 6, 4, and 2. That gives four resistances and four supports. The third pair is the usual reversal zone. The fourth pair is the breakout zone. No central pivot is required.",
        "Use the third level as a place mean-reversion traders fade, and a move through the fourth as a breakout. They sit close to price, so they belong to the next session.",
    ),
    "pivot_woodie": (
        "Woodie's pivot weights the close more heavily than the classic formula.",
        "The pivot is (high + low + 2·close) / 4 of the prior bar. First resistance is twice the pivot minus the low, and first support is twice the pivot minus the high. The second pair adds and subtracts the range. Because the close is doubled, a bar that closes at the high lifts the whole set.",
        "Use it when you want the pivot to reflect where the bar settled. The levels are read the same way as classic pivots: balance at the pivot, then the R and S steps.",
    ),
    "pivot_demark": (
        "DeMark pivots change the formula according to whether the prior bar closed above, below, or equal to its open.",
        "If the close is below the open, the raw sum is high + 2·low + close. If the close is above the open, it is 2·high + low + close. If they are equal, it is high + low + 2·close. The pivot is that sum divided by 4. Resistance is the sum divided by 2, minus the low. Support is the sum divided by 2, minus the high. There is one level on each side.",
        "Use the single resistance and support as the next bar's projected extremes under DeMark's relationship between the open and the close. It is a sparser map than the classic five-level set.",
    ),
    "fib_retrace": (
        "A Fibonacci retracement places horizontal levels at standard ratios between a swing high and a swing low.",
        "Find the swing range. For a move up, each level is the high minus the ratio times the range. The stored ratios are 0.236, 0.382, 0.5, 0.618, and 0.786. In this library the swing is the high-low span of a lookback, so the levels update as that window's extremes change.",
        "Use the 0.382 to 0.618 band as the usual pullback zone inside a trend you already believe in. The level is a place to look, and the trend is a separate decision.",
    ),
    "fib_extension": (
        "A Fibonacci extension projects targets beyond a swing, using ratios greater than 1.",
        "From the end of the swing, add the ratio times the swing's range. The stored ratios are 1.0, 1.272, 1.618, 2.0, and 2.618. A 1.618 extension is the common measured target. As with the retracement, this library takes the swing from a lookback window.",
        "Use the projections as price targets once a swing has been defined and price has left it. They are arithmetic targets, so they do not know about nearby traded levels.",
    ),
    "gann_fan": (
        "A Gann fan is a set of angled lines drawn from a pivot at fixed price-per-bar slopes, such as 1x1, 1x2, and 2x1.",
        "Pick a pivot price and a scale that converts price units into bar units. Each angle is a line whose slope is that ratio times the scale. The 1x1 is the balance angle in Gann's reading. The lines are geometry on the chart. They are not one number per bar, so they are not computed as a series here.",
        "Use the fan as a visual map of support and resistance in both price and time, once you have chosen the pivot and the scale. The scale choice changes every line, which is why it stays a study reference.",
    ),
    "swing_hl": (
        "A swing high is a high that is higher than the highs a set number of bars to its left and right. A swing low is the same idea on the lows. With two bars on each side these are Williams fractals.",
        "A bar is a swing high when its high is strictly greater than the highs of the left bars and the right bars. The default is 2 and 2. The right-hand bars have to exist, so the swing is confirmed only after those bars have printed. Until then the candidate can still be exceeded.",
        "Use confirmed swings as the pivots for structure: a break of the last swing high, a stop beyond a swing low, or the anchors of a Fibonacci level. The confirmation delay is the price of the pivot being real.",
    ),
    "zigzag": (
        "ZigZag connects swing points that reverse by at least a chosen percentage, and it drops the smaller noise between them.",
        "Track the current extreme. When price reverses from it by the deviation, default 5 percent, confirm that extreme as a pivot and start a new leg. A later extension of the same leg moves the pivot. That revision is the repaint: the latest leg is provisional until the reverse is confirmed.",
        "Use it to see the larger swings and to measure their size. Do not treat the last unfinished leg as a fixed historical value. Confirmed pivots are the ones that are safe to use in a backtest.",
    ),
    "regression_channel": (
        "A linear regression channel is the least-squares line through the last N closes, with parallel bands a chosen number of standard errors away.",
        "Fit price on the bar index over the window, default 100. The middle line is the fitted value. The bands add and subtract a multiple of the residual standard error, default 2. As the window rolls, the slope and the bands change.",
        "Use the middle line as the trend of the window and a tag of the outer band as an extension away from that trend. A high R-squared, from the companion statistic, says the channel is describing a real line and not a cloud.",
    ),
    "price_channel": (
        "A price channel is the rolling highest high and lowest low, used as breakout levels.",
        "The upper line is the highest high over N bars and the lower line is the lowest low, default 20. This entry is the raw channel of the window. The Donchian entry used for signals shifts the window by one bar so the current close is not inside its own channel.",
        "Use a close beyond the channel as a breakout, and the opposite side as a place the break has failed. It is the simplest range rule in the catalog.",
    ),
    "typical_price": (
        "Typical price is the average of the high, the low, and the close.",
        "Add the three and divide by 3. There is no lookback. One bar produces one price. It pulls the close toward the middle of the range.",
        "Use it as the price input to CCI, money flow, and VWAP, or whenever a single close feels like a poor summary of a wide bar.",
    ),
    "median_price": (
        "Median price is the midpoint of the bar's high and low.",
        "Add the high and the low and divide by 2. The open and the close do not enter. A bar with a long tail still contributes its geometric middle.",
        "Use it when the range midpoint is the price you want to smooth, which is what the awesome oscillator does. It ignores where inside the range the bar closed.",
    ),
    "weighted_close": (
        "The weighted close averages the high, the low, and the close, with the close counted twice.",
        "Add high, low, and twice the close, then divide by 4. The close has half the weight. The high and the low share the rest.",
        "Use it as a bar summary that respects the range and still leans on the settlement. It is a middle step between the close alone and the typical price.",
    ),
    "ohlc4": (
        "The OHLC average is the mean of the open, high, low, and close.",
        "Add the four prices and divide by 4. Each part of the bar has equal weight. There is no lookback.",
        "Use it as a neutral one-number summary of the bar when you have no reason to prefer the close. Heikin-Ashi uses this average as its close.",
    ),
    "heikin_ashi": (
        "Heikin-Ashi candles are a smoothed rewriting of the bars that makes a trend easier to see as a run of same-colored candles.",
        "The Heikin-Ashi close is the average of open, high, low, and close. The Heikin-Ashi open is the average of the previous Heikin-Ashi open and close. The high is the max of the real high and those two, and the low is the min. Each bar depends on the previous rewritten bar, so a trend tends to stay one color until the average turns.",
        "Use a color change as a simple trend-exit cue, and a run of candles with no lower shadow as a strong advance. The prices are synthetic, so orders still belong on the real open, high, low, and close.",
    ),
    "renko": (
        "Renko draws a brick only when price has moved a fixed amount. Time does not create a brick. A quiet stretch draws nothing.",
        "Choose a brick size, often an ATR. A new brick prints when the close has traveled at least that far from the last brick's close. Reversal bricks usually require two brick sizes. The bricks do not line up one-for-one with the daily bars, so they are a different axis.",
        "Use Renko to see price movement with time and small noise removed. Because the brick boundaries are not the session bars, a backtest on this desk stays on the time bars and treats Renko as a way to study the path.",
    ),
    "log_return": (
        "A log return is the natural log of the price ratio over N bars. Log returns add across time.",
        "It is the natural log of today's close divided by the close N bars ago. The default N is 1, a single bar. A 1 percent move is about 0.01 in log terms. The sum of daily log returns is the log return of the whole stretch.",
        "Use it as the input to volatility, correlation, beta, and any statistic that assumes additive returns. For a number you will read as a percent, use the simple percentage return.",
    ),
    "pct_return": (
        "Percentage return is the simple change in price over N bars.",
        "It is today's close divided by the close N bars ago, minus 1. The default N is 1. A close that goes from 100 to 110 is 0.10. These returns compound by multiplying, and they do not add cleanly across many bars.",
        "Use it whenever a rule or a report should speak in percent. It is the ordinary return behind momentum factors and performance ratios.",
    ),
    "gap": (
        "Gap size is the jump from yesterday's close to today's open, in points and in percent.",
        "Subtract the previous close from today's open. The percent form divides that difference by the previous close. A positive gap means the open is above the prior close. The rest of the bar is a separate question.",
        "Use it to study overnight information, to skip entries on an extreme open, or to measure how often a gap fills. The gap-fill tool tracks that last question.",
    ),
    "linreg_slope": (
        "The linear regression slope is the steepness of the straight line that best fits the last N prices.",
        "Fit price on the bar index by least squares. The slope is covariance of time and price divided by the variance of time. The default window is 14. The unit is price points per bar. A positive slope means the fitted line rises.",
        "Use the sign as a trend direction and the size as a trend speed. A strategy filter can require the slope of a chosen window to be positive before a long entry.",
    ),
    "linreg_r2": (
        "R-squared is the fraction of price variation inside the window that a straight line explains. It runs from 0 to 1.",
        "It is the square of the correlation between the bar index and price over N bars, default 14. A value near 1 means price has hugged a straight line. A value near 0 means the line explains little and the window is bent or noisy.",
        "Use a high reading to trust a slope or a regression channel, and a low reading to treat the same slope as noise. Direction comes from the slope. This number is only the quality of the fit.",
    ),
    "linreg_angle": (
        "The regression angle converts the fitted slope into degrees.",
        "Take the regression slope over N bars, default 14, and apply the arctangent, then convert from radians to degrees. The angle depends on the price scale: the same percent trend on a high-priced stock produces a steeper slope in points and a larger angle.",
        "Use it for a reading you want in degrees, and be careful comparing angles across names. For a comparable measure, prefer the slope of log price or a percent rate of change.",
    ),
    "zscore": (
        "The z-score says how many standard deviations the current price is from its rolling mean.",
        "Subtract the SMA from price and divide by the standard deviation, both over N bars, default 20. A reading of 2 means price is two deviations above the average. A reading of 0 means price is on the average. This is the same distance Bollinger %B describes, in deviation units.",
        "Use a large positive or negative reading as a stretch in a mean-reversion rule, or as a ranking of which name is most extended today. The window decides what 'normal' means.",
    ),
    "correlation": (
        "Rolling correlation is the Pearson correlation of two series over the last N bars.",
        "Divide the covariance of the two series by the product of their standard deviations, over the default 30 bars. The result runs from -1 to +1. It needs a second series. One symbol's bars are not enough.",
        "Use it to see whether two names are moving together right now. Pairs trades and hedges start from a correlation that is high and stable, then look at the spread.",
    ),
    "beta": (
        "Rolling beta is the sensitivity of one return series to a benchmark's returns.",
        "Divide the covariance of the asset's returns and the market's returns by the variance of the market's returns, over the default 60 bars. A beta of 1 moves with the benchmark. A beta of 1.5 has tended to move 1.5 times as much. It needs the benchmark series.",
        "Use it to size a hedge, to compare a stock's market exposure with another's, or to remove the market piece before judging idiosyncratic momentum.",
    ),
    "skew": (
        "Rolling skewness measures whether recent returns have a longer tail on the upside or the downside.",
        "Take the standardized returns over the window, default 30, cube them, and average. Positive skew means the larger surprises have been to the upside. Negative skew means the larger surprises have been to the downside. A short window is noisy.",
        "Use it to notice a name whose recent path is built from many small gains and a few sharp drops, which is the uncomfortable shape for a long holder. It describes the window. It does not forecast the next bar.",
    ),
    "kurt": (
        "Rolling kurtosis measures how heavy the tails of recent returns are.",
        "Average the fourth power of standardized returns over the window, default 30. A normal distribution scores 3 on the raw measure. A higher number means more of the variance came from a few large bars. Estimators differ on whether they subtract 3.",
        "Use a high reading as a sign the recent window contains outliers, so a volatility number from that window may be dominated by one or two days. It is a check on the quality of the other statistics.",
    ),
    "hurst": (
        "The Hurst exponent describes whether a series has tended to trend, wander, or revert.",
        "Over the window, default 100, the rescaled-range method looks at how the range of cumulative deviations grows as the sub-window length grows. The exponent is the slope of log(range/scale) against log(length). Above 0.5 the series has persisted. Near 0.5 it has looked random. Below 0.5 it has tended to reverse.",
        "Use it as a regime hint before choosing a trend rule or a mean-reversion rule. The estimate is slow and noisy, so treat a small difference from 0.5 as nothing.",
    ),
    "autocorr": (
        "Autocorrelation is the correlation of returns with their own past values a set number of bars earlier.",
        "Correlate the return series with a copy of itself shifted by the lag, default 1, inside a window, default 50. A positive lag-1 reading means an up bar has tended to follow an up bar. A negative reading means it has tended to follow the opposite.",
        "Use the sign as a hint about short-horizon continuation or reversal in that window. A single lag is a small piece of the picture, so confirm it on a second window before leaning on it.",
    ),
    "percent_rank": (
        "Percentile rank is the share of values in the lookback that sit below the current value.",
        "Count how many of the last N values, default 100, are below today's value, and divide by N. The result is a percent. A reading of 100 means today's value is the highest in the window. A reading of 0 means it is the lowest.",
        "Use it to turn any series into a 0-to-100 rank against its own history. Connors RSI uses this rank on the one-bar return. It is also a simple way to call a breakout: a rank near 100.",
    ),
    "er": (
        "The efficiency ratio is the net price change divided by the sum of the absolute bar-to-bar changes. A perfectly straight move scores 1.",
        "Take the absolute difference between today's close and the close N bars ago, default 10. Divide by the sum of the absolute one-bar changes inside that window. Chop that cancels itself out lowers the ratio. Kaufman uses this ratio to set the speed of KAMA.",
        "Use a high reading as a trend filter and a low reading as a reason to stand aside or to fade. It is a direct, unitless measure of how much of the path was progress.",
    ),
    "fractal_dim": (
        "The fractal dimension index estimates how rough the price path is. A smoother trend scores lower. A random path sits near 1.5.",
        "Compare the length of the path, built from ranges, with the length of the window, default 30. The index is 1 plus a log ratio of those lengths. Values drift between 1 and 2. Lower means the path is closer to a line.",
        "Use a falling index as a sign a trend is becoming cleaner, which is the same information FRAMA turns into a smoothing speed. A high index argues for fading the range.",
    ),
    "choppiness": (
        "The choppiness index scores how sideways a market is, on a scale from 0 to 100. High values are chop. Low values are trend.",
        "Divide the sum of one-bar true ranges over N bars, default 14, by the high-low range of that whole window. Take the base-10 log and divide by the log of N, then multiply by 100. If the bars' ranges add up to far more than the net range, the market went back and forth and the index is high. Readings above 61.8 and below 38.2 are the usual zones.",
        "Use a high reading to turn trend entries off, and a low reading to allow them. It answers the same question as ADX from the range arithmetic.",
    ),
    "vhf": (
        "The vertical horizontal filter compares the net range with the path length, and was built to tell a trend from a congestion.",
        "Subtract the lowest close in the window from the highest close, and divide by the sum of absolute close-to-close changes. The default period is 28. A higher reading means more of the movement was in one direction. It is the efficiency ratio computed on the close range.",
        "Use a rise through a threshold you choose as permission for a trend system, and a fall as a switch toward a range approach. The original use was exactly that switch.",
    ),
    "cycle_ht_dcperiod": (
        "The Hilbert dominant-cycle period is Ehlers' estimate of how many bars the main cycle in price currently lasts.",
        "A Hilbert transform builds an in-phase and a quadrature component of a detrended price. The rate of change of the phase angle between them is converted into a cycle length. The result is smoothed so it does not jump every bar. There is no user period. The series estimates its own.",
        "Use the estimated length as the lookback for a cycle-aware average or oscillator, or as a regime description: a lengthening cycle is a slowing swing. It is an estimate, and it wanders in a trend.",
    ),
    "ht_sine": (
        "The Hilbert sine wave plots a sine of the dominant-cycle phase and a lead sine about 45 degrees ahead of it.",
        "From the same Hilbert phase used for the cycle period, take the sine, and take the sine of the phase plus 45 degrees. The lead line crosses the sine line before a cycle turn in Ehlers' reading. In a strong trend the phase assumption fails and the lines cling to the rails.",
        "Use a cross of the two lines as a cycle turning point when the companion trend-mode flag says the market is cycling. Ignore the cross when that flag says the market is trending.",
    ),
    "ht_trendmode": (
        "The Hilbert trend-mode flag is 1 when Ehlers' measure says price is trending and 0 when it says price is cycling.",
        "The flag looks at the rate of change of the dominant-cycle phase and at whether price is holding on one side of the instantaneous trendline. A fast, orderly phase advance with price away from the trendline is a trend. A phase that fits a cycle is mode 0.",
        "Use it as a switch between two playbooks: trend rules when the flag is 1, and cycle rules such as the sine-wave cross when the flag is 0.",
    ),
    "ht_trendline": (
        "The Hilbert instantaneous trendline is a moving average whose length follows the dominant cycle.",
        "Smooth price with an average on the order of the dominant-cycle period, then apply a weighted smooth. The line is the trend with the cycle damped. Price crossing it is Ehlers' early trend trigger, later refined by the trend-mode flag.",
        "Use the line as a cycle-adaptive trend filter. When the mode flag is trending, trade in the direction of a cross of this line. When the mode is cycling, expect price to oscillate around it.",
    ),
    "mama": (
        "MAMA is Ehlers' MESA adaptive moving average. Its speed follows the rate of change of the Hilbert phase. FAMA is a slower follower of MAMA.",
        "The phase rate sets an alpha between a fast limit, default 0.5, and a slow limit, default 0.05. MAMA is that alpha times price plus the rest times the previous MAMA. FAMA uses half of MAMA's alpha, so it lags behind. A cross of MAMA and FAMA is the signal.",
        "Use the cross as an adaptive trend signal that speeds up when the cycle phase is changing quickly and slows down when it is not. The two limits are the only speeds you set.",
    ),
    "cyber_cycle": (
        "The cyber cycle is a high-pass filter that tries to leave the cyclic swing and remove the trend.",
        "A smoothed price is passed through a two-pole filter whose alpha, default 0.07, sets how much of the slow trend is rejected. The output oscillates around zero. A one-bar lag is the signal line. A smaller alpha keeps a slower cycle.",
        "Use a cross of zero or of the signal line as a cycle turn. It assumes a cycle is present. In a strong trend the oscillator can stay pinned to one side.",
    ),
    "super_smoother": (
        "The Ehlers super smoother is a two-pole Butterworth low-pass filter with less lag than a same-length moving average.",
        "From the cutoff period, default 10, compute three coefficients from an exponential decay and a cosine at the Butterworth angle. Today's value blends the average of the last two prices with the last two filter values. The filter rejects frequencies above the cutoff and damps aliasing.",
        "Use it as a general cleaner of a noisy series, including price or another indicator, when you want the smoothness of a longer average and less delay.",
    ),
    "itrend": (
        "Ehlers' instantaneous trendline is a two-pole recursive filter that tracks the trend with little lag.",
        "Alpha, default 0.07, sets the cutoff. The formula mixes today's price and the two previous prices with the two previous filter values, with coefficients fixed by alpha. A smaller alpha is slower and smoother. This one is the fixed-alpha cousin of the Hilbert trendline.",
        "Use a cross of price and the trendline as a simple trend trigger, or use the line as the trend to subtract when you want a cycle oscillator.",
    ),
    "stoch_cg": (
        "The center-of-gravity oscillator finds the balance point of prices inside the window, like a center of mass.",
        "Weight the newest price by 1, the one before by 2, and so on, out to the period, default 10. Divide the weighted sum by the plain sum of prices and take the negative, so the line faces the usual way. The center shifts before a simple average does, which is why Ehlers presents it as a low-lag oscillator.",
        "Use a turn in the line, or a cross of its one-bar lag, as a cycle turning point. It needs a cycle-like window. A strong trend pushes it to a side and holds it there.",
    ),
    "market_profile_tpo": (
        "Market profile counts how many time blocks traded at each price. The busiest price is the point of control, and the value area holds about 70 percent of those time blocks.",
        "Split the session into blocks, classically 30 minutes, and drop a count into every price bin the block's range touched. Prices with more counts were accepted for longer. This is time at price. Volume profile is the volume-weighted relative. On daily bars the block structure is coarse.",
        "Use the value area as the day's accepted range and the point of control as the fairest price of that time. A later test of the point of control is a standard trade location. The shape of the profile is the study.",
    ),
    "opening_range": (
        "The opening range is the high and the low of the first part of the session.",
        "Take the max high and the min low over the first N minutes, default 30. Those two prices stay fixed for the rest of the session. A later break of the high or the low is the opening-range breakout. On daily bars the first bar is the whole range, so the idea belongs to intraday data.",
        "Use a break of the range as an intraday entry in the direction of the break, and the opposite side as the failure point. The length of the window is the whole design choice.",
    ),
    "prev_day_levels": (
        "Previous-period levels are yesterday's high, low, and close, carried forward onto today's bars. The same idea works for the prior week or month.",
        "Once a period closes, store its high, low, and close and plot them as horizontal references on the next period. They do not move until the next period completes. On daily bars the previous day is the previous row.",
        "Use them as the simplest support and resistance: a break of yesterday's high, a test of yesterday's low, or a reclaim of yesterday's close. Many intraday plans are built only from these three numbers.",
    ),
    "gap_fill": (
        "Gap-fill status records whether price has traded back to the previous close after opening away from it.",
        "The gap is today's open minus yesterday's close. The gap is filled when a later price crosses that previous close: a down gap fills if price rises to the old close, and an up gap fills if price falls to it. The outputs mark the open gap and whether the fill has happened.",
        "Use it to study how often gaps fill and how long they take, or to avoid a target that assumes a gap will stay open. A fill is a description of the path.",
    ),
    "order_block": (
        "An order block, in the usual retail definition, is the last opposing candle before an impulsive move. It is treated as a zone where unfinished business may sit.",
        "Look back a set number of bars, default 10, for a down candle immediately before a push up that exceeds a multiple of ATR, or the mirror image for a down impulse. The candle's high and low become the zone. Definitions differ on what counts as impulsive, so two implementations will mark different candles.",
        "Use the zone as a place to watch for a return during a later pullback. The marking rule is mechanical here. The interpretation that institutions left orders there is a story on top of the candle.",
    ),
    "fvg": (
        "A fair value gap is a three-bar imbalance: the range of the first bar and the range of the third bar do not overlap, leaving a price gap in between.",
        "A bullish gap exists when the current low is above the high from two bars ago. A bearish gap exists when the current high is below the low from two bars ago. The empty interval is the gap. A later bar that trades into that interval is a fill.",
        "Use an unfilled gap as a magnet or as a location for a pullback entry in the direction of the impulse that created it. Overlapping bars leave no gap, so most sessions mark nothing.",
    ),
    "bos_choch": (
        "A break of structure is a close beyond the last swing in the direction of the current trend. A change of character is the first close that breaks a swing against that trend.",
        "Swings are confirmed with a chosen width, default 5 bars on each side. After an uptrend, a close above the last swing high is a bullish break of structure. The first close below the last swing low that follows is a change of character. The labels swap in a downtrend.",
        "Use a break of structure to say the trend is still making progress, and a change of character to say the sequence of swings has flipped. Both depend on the swing length you pick.",
    ),
    "adr": (
        "Average daily range is the simple average of the high-minus-low range over N sessions.",
        "Subtract the low from the high on each bar and take an SMA, default 14. Gaps are not added, which is the difference from ATR. The result is the typical cash-session range in points.",
        "Use it to set an intraday target or to judge whether today's range is already used up. A stock that has already traveled its average range has less room if the range tends to mean-revert.",
    ),
    "rel_volume": (
        "Relative volume is today's volume divided by its recent average.",
        "Divide volume by an SMA of volume, default 20. A reading of 2 means today's volume is twice the recent typical day. A reading below 1 is a quiet bar.",
        "Use it to confirm a breakout or to ignore a signal that printed on a dead bar. It is the simplest activity filter in the catalog.",
    ),
    "rs_ratio": (
        "Relative strength in this entry is the price of the asset divided by the price of a benchmark. A rising line means the asset is outperforming.",
        "Divide the two prices bar by bar. There is no lookback and no rank. The scale is arbitrary and depends on the two price levels, so read the slope. It needs a benchmark series beside the name.",
        "Use a rising ratio as a filter to buy strength against the index, and a falling ratio to avoid a name that is lagging even if its own chart looks fine. This is the raw input to a relative-rotation graph.",
    ),
    "rrg": (
        "A relative rotation graph plots two normalized series: the RS-ratio, which is smoothed relative strength, and RS-momentum, which is the momentum of that ratio.",
        "Smooth the price ratio to the benchmark, turn it into a z-score style reading around 100, and call that RS-ratio. Take a rate of change of RS-ratio, normalize it the same way, and call that RS-momentum. The default smooth is 10. The four quadrants are leading, weakening, lagging, and improving. It needs the benchmark.",
        "Use the quadrant and the direction of travel to see whether outperformance is strengthening or fading. A name moving from improving into leading is the rotation traders try to catch.",
    ),
    "dual_momentum": (
        "Relative momentum rank orders a universe by the return over a lookback. The strongest name ranks first.",
        "Compute the percent change over the lookback, default 126 sessions, for every symbol, and rank them. The rank is cross-sectional: one symbol's bars cannot produce it. Absolute momentum, the sign of the same return, is often required as well so a leader of a falling market is skipped.",
        "Use the top ranks as the candidates in a rotation strategy, and rebalance on a schedule. The lookback decides whether you are ranking a quarter, a half-year, or something else.",
    ),
    "ad_breadth": (
        "The advance/decline line is the running total of advancing issues minus declining issues.",
        "Each session adds the number of stocks that closed up and subtracts the number that closed down. The level is cumulative, so the slope is the reading. It needs the market's advance and decline counts, which a single price series does not contain.",
        "Use a divergence from the index as a breadth warning: an index making highs while the line falls says fewer names are carrying the high. It is a participation check.",
    ),
    "mcclellan_osc": (
        "The McClellan oscillator is the difference between a fast and a slow EMA of net advances.",
        "Net advances are advancers minus decliners. Subtract a 39-period EMA of that series from a 19-period EMA. Positive values mean the short breadth trend is above the longer one. The inputs are market-wide counts.",
        "Use a surge above zero as broadening participation and a drop below zero as narrowing. Extremes are also used as short-term exhaustion readings for the index.",
    ),
    "mcclellan_sum": (
        "The McClellan summation index is the running total of the McClellan oscillator.",
        "Add each bar's oscillator to the previous total. A rising sum means the oscillator has been positive on balance. It is the slow breadth trend, built from the same advance and decline counts.",
        "Use the direction of the sum as the breadth trend behind the index. A falling sum while the index rises is the longer version of a breadth divergence.",
    ),
    "trin": (
        "The Arms index, or TRIN, compares the advance/decline count ratio with the advance/decline volume ratio.",
        "Divide advancers by decliners, divide advancing volume by declining volume, and divide the first ratio by the second. A reading of 1 means volume is split in proportion to the counts. Above 1, down issues are receiving a larger share of the volume. It needs all four market totals.",
        "Use a high reading as heavy selling pressure and a low reading as heavy buying pressure. Arms treated very high and very low prints as short-term exhaustion, so the extreme is often read as a contrary clue.",
    ),
    "pct_above_ma": (
        "This is the percentage of stocks in a universe that are trading above a chosen moving average.",
        "For each member, compare its close with its own SMA, default 50. Count the share that are above and multiply by 100. The series is a property of the universe, so one symbol cannot produce it.",
        "Use a high reading as broad participation and a low reading as a market where most names are already below their trend. Crosses of 50, and extremes near 20 or 80, are the usual study points.",
    ),
    "new_hl": (
        "New highs minus new lows is the count of names at a 52-week high minus the count at a 52-week low.",
        "Over the lookback, default 252 sessions, count how many members made a new high and how many made a new low, and subtract. The result is positive when fresh highs dominate. It needs the whole universe.",
        "Use the sign and the trend as a leadership check. An index high with a shrinking or negative reading says the high is narrow. A surge in new lows marks a broad break.",
    ),
    "put_call": (
        "The put/call ratio divides put volume by call volume. A high number means more puts are trading.",
        "Divide the two volumes for the day, for an index or for the market. There is no lookback in the raw ratio. Smoothing is a separate choice. It needs option-volume data.",
        "Use an extreme high as a sign the crowd is hedging or betting on a decline, which contrary readers treat as a bullish setup, and an extreme low as complacency. The level that counts as extreme comes from the ratio's own history.",
    ),
    "vix_term": (
        "The VIX term-structure ratio divides the 30-day implied-volatility index by the 3-month one.",
        "VIX over VIX3M is near 1 when the curve is flat, below 1 when the usual upward slope is in place, and above 1 when the near term is more expensive than the longer term. That last state is backwardation. Both indexes have to be in the data.",
        "Use a ratio above 1 as a stress flag: the market is paying more for near-term protection than for protection further out. A return below 1 is the market leaving that stress.",
    ),
    "iv_rank": (
        "IV rank locates today's implied volatility between its lowest and highest value over the lookback. IV percentile is the share of days in that lookback when implied volatility was lower than today.",
        "Rank is 100 times (today minus the minimum) divided by (maximum minus minimum), default lookback 252. Percentile is the percent of days in the window below today. One spike a year ago keeps rank elevated until it leaves the window. Percentile is less sensitive to that single spike. Both need an implied-volatility series.",
        "Use a high rank or percentile as a reason option sellers are being paid more than usual for this name, and a low reading as a reason premium is cheap. The two numbers answer slightly different questions, so read the one that matches the decision.",
    ),
    "hv_iv_spread": (
        "The IV-minus-HV spread subtracts realized volatility from implied volatility. A positive spread means options are priced richer than recent movement.",
        "Compute historical volatility from closes over the default 20 bars, and subtract it from the implied volatility of the chosen contract. The unit is volatility points. It needs the implied-volatility input.",
        "Use a persistently positive spread as the variance risk premium: sellers have historically been paid for taking the gap between implied and later realized volatility. A negative spread means the option market is pricing less movement than the recent past delivered.",
    ),
    "greeks": (
        "The option Greeks are the sensitivities of an option's price: delta to the underlying, gamma to delta, theta to time, vega to volatility, and rho to interest rates.",
        "Under Black-Scholes, each Greek is a partial derivative of the model price, given the spot, the strike, the time left, the rate, and the volatility. Delta near 0.5 is an at-the-money call. Gamma is highest there too. Theta is the daily decay. Vega is the price change for a one-point change in volatility. The inputs are the option terms, not a stock bar.",
        "Use delta to translate an option into share exposure, gamma to see how fast that exposure changes, theta to know the carry of holding it, and vega to know the volatility bet. They are model numbers. A different model gives different Greeks.",
    ),
    "max_pain": (
        "Max pain is the strike at which the total payout to holders of the current option chain would be smallest at expiry.",
        "For each candidate strike, add up what every in-the-money call and put in the open chain would be worth, and pick the strike where that sum is smallest. The calculation needs the open interest of the chain. It changes as open interest changes, and it says nothing after expiry.",
        "Use it as a reference some traders expect price to gravitate toward into expiration because hedgers adjust around the largest open interest. Treat it as a hypothesis about positioning, and check it against the actual open interest.",
    ),
    "pcr_oi": (
        "The put/call open-interest ratio divides put open interest by call open interest.",
        "Sum the open interest of the puts in view, sum the calls, and divide. Unlike the volume ratio, this is the stock of positions, not the day's trades. It needs those open-interest totals.",
        "Use a high reading as a market positioned in puts relative to calls. It is a positioning backdrop. It updates slowly compared with the volume put/call ratio.",
    ),
    "basis": (
        "Basis is the futures price minus the spot price. The percent form divides that gap by the spot.",
        "Subtract the two prices. A positive basis is a futures premium, contango in the curve sense when it persists across expiries. A negative basis is a discount. The fair premium for a financed hold is interest minus dividends or storage, so the interesting part is the gap from that fair value. It needs both prices.",
        "Use a premium far from fair value as a positioning or carry clue, and use the basis as the spread you would trade if you were long one and short the other.",
    ),
    "funding_rate": (
        "The perpetual funding rate is the periodic payment between longs and shorts on a perpetual swap, set by the exchange from the premium of the swap to the spot.",
        "When the perpetual trades above the spot, the funding rate is positive and longs pay shorts. When it trades below, shorts pay longs. The exchange publishes the rate. There is nothing to compute from an equity bar.",
        "Use an extreme positive rate as a sign longs are crowded and paying up to stay long, which is a risk for those longs. The same reading in reverse applies to a very negative rate.",
    ),
    "cot_net": (
        "COT net positioning is the long contracts minus the short contracts for one trader group in the CFTC Commitments of Traders report.",
        "Subtract the group's short positions from its long positions for that market. The report is weekly. Groups include commercials, large speculators, and small traders. The series needs that report.",
        "Use an extreme net position against a group's own history as a crowding reading. Commercial extremes and speculative extremes are often read differently, so keep the group identity attached to the number.",
    ),
    "candle_doji": (
        "A doji is a bar whose open and close are almost the same, so the body is tiny compared with the range.",
        "The body, the absolute difference of close and open, is at most the body-percent of the high-low range. The default percent is 0.1, so the body is at most a tenth of the range. The signal is 1 on a doji bar and 0 otherwise.",
        "Use it as a mark of indecision: both sides traded and the bar settled where it opened. After a run, a doji says the push paused. It is a location in a trend, and the next bar is the confirmation.",
    ),
    "candle_hammer": (
        "A hammer or hanging man has a small body at the top of the range and a long lower shadow, at least twice the body.",
        "The lower shadow is at least twice the body, and the upper shadow is small. The same shape is called a hammer after a decline and a hanging man after an advance. The signal marks the shape. The name comes from the trend you supply.",
        "Use a hammer at a low as a sign sellers pushed the bar down and buyers closed it back near the high. Wait for a later bar in the new direction if you want confirmation. A hanging man is the same shape at a high, read as a warning.",
    ),
    "candle_inv_hammer": (
        "An inverted hammer or shooting star has a small body at the bottom of the range and a long upper shadow.",
        "The upper shadow is at least twice the body, and the lower shadow is small. After a decline the shape is an inverted hammer. After an advance it is a shooting star. The calculation marks the shape.",
        "Use a shooting star at a high as a sign buyers pushed up and could not hold it. An inverted hammer at a low is the same shape, read as a possible end to the decline once a later bar confirms.",
    ),
    "candle_engulfing": (
        "An engulfing bar is a candle whose body fully covers the previous candle's body and closes in the opposite direction.",
        "A bullish engulfing follows a down body. The current body is up, its open is at or below the previous close, and its close is at or above the previous open. The bearish pattern is the mirror image. Shadows do not have to engulf.",
        "Use a bullish engulfing after a decline as a shift in who finished the bar, and a bearish engulfing after a rally the same way. The pattern is stronger when the second body is large and the volume is higher.",
    ),
    "candle_harami": (
        "A harami is a small candle body sitting entirely inside the previous candle's body, in the opposite color.",
        "The current open and close both fall inside the previous open and close, and the body direction flips. It is the opposite of an engulfing bar: the second bar is contained. A very small second body is a harami cross.",
        "Use it as a sign the previous push has stalled. It is a weaker reversal shape than an engulfing bar, so traders often want a third bar to confirm.",
    ),
    "candle_piercing": (
        "A piercing line and a dark cloud cover are two-bar reversals that close beyond the midpoint of the previous body.",
        "In a piercing line, a down bar is followed by an up bar that opens below the previous close and closes above the midpoint of the previous body, without fully engulfing it. Dark cloud cover is the bearish mirror: an up bar, then a down bar that closes below the prior midpoint.",
        "Use the piercing line after a decline as a partial recovery of the prior body, and the dark cloud after a rally as a partial give-back. Closing through the midpoint is the requirement that makes it more than a small bounce.",
    ),
    "candle_star": (
        "A morning star and an evening star are three-bar reversals: a long body, a small star, then a long body in the new direction.",
        "The first bar is a large body with the trend. The second is a small body, classically gapped. The third closes beyond the midpoint of the first body in the opposite direction. Morning star is the bullish sequence. Evening star is the bearish one.",
        "Use the completed third bar as the reversal confirmation. The small middle bar is the stall, and the third bar is the side that took control.",
    ),
    "candle_soldiers": (
        "Three white soldiers are three strong up candles in a row. Three black crows are three strong down candles in a row.",
        "Each candle continues the direction, opens inside the previous body, and closes beyond the previous close, with small shadows. Three up candles mark the soldiers. Three down candles mark the crows. A very extended prior move makes the same shape late.",
        "Use soldiers as evidence of a steady advance that began after a base, and crows as evidence of a steady decline. In a trend that is already long in the tooth, the same pattern can be a climax.",
    ),
    "candle_marubozu": (
        "A marubozu is a candle with almost no shadows, so the bar opened at one end of the range and closed at the other.",
        "Both shadows are at most a small fraction of the range. A bullish marubozu opens at the low and closes at the high. A bearish one does the opposite. The signal marks that commitment.",
        "Use it as a sign of one-sided trade for that bar. A marubozu through a level is a stronger break than a bar that spent most of its range on the other side of the level.",
    ),
    "candle_spinning": (
        "A spinning top has a small body and long shadows on both sides.",
        "The body is small relative to the range, and both the upper and the lower shadow are longer than the body. Buyers and sellers each pushed the bar and neither kept it. It differs from a doji because the body can be a bit larger, and from a hammer because both shadows are long.",
        "Use it as indecision inside a larger move. A single spinning top is a pause. A cluster of them is a small range.",
    ),
    "candle_dragonfly": (
        "A dragonfly doji has almost no body and a long lower shadow. A gravestone doji has almost no body and a long upper shadow.",
        "The open and close are within the doji tolerance, default a tenth of the range, and one shadow accounts for nearly the whole range. Dragonfly means the long shadow is the lower one: price sold off and came back. Gravestone means the long shadow is the upper one.",
        "Use a dragonfly at a low as a rejection of lower prices, and a gravestone at a high as a rejection of higher prices. The long shadow is the event. The tiny body is what makes it a doji.",
    ),
    "candle_tweezer": (
        "A tweezer top is two bars with nearly the same high. A tweezer bottom is two bars with nearly the same low.",
        "The highs, or the lows, match within a tolerance, default 0.1 percent of price. The bars can be any color. The match says the market failed at the same price twice in a row.",
        "Use a tweezer at a known level as a double test of that price. Two matching highs after a rise are a small rejection. The tolerance decides how exact the match must be.",
    ),
    "candle_inside_outside": (
        "An inside bar sits entirely within the previous bar's high and low. An outside bar's high and low both exceed the previous bar's.",
        "Inside means today's high is at or below yesterday's high and today's low is at or above yesterday's low. Outside means today's high is above yesterday's high and today's low is below yesterday's low. The two flags are separate outputs. A bar can be neither.",
        "Use an inside bar as a compression and trade the break of its parent bar. Use an outside bar as a wide bar that swallowed the previous range, and read its close to see which side won it.",
    ),
    "candle_three_inside": (
        "Three inside up or down is a harami followed by a third bar that closes beyond the first bar's body.",
        "The first two bars form a harami. In the bullish version the third bar closes above the first bar's open, confirming the small second bar was a turn. The bearish version closes below the first bar's body. Without the third bar it remains only a harami.",
        "Use the third close as the confirmation the harami does not provide by itself. It is a completed reversal sequence.",
    ),
    "candle_kicker": (
        "A kicker is a sharp gap reversal: the second bar opens at or beyond the previous bar's open and runs in the opposite direction.",
        "The two bodies have opposite colors, and the second open gaps through the first open rather than merely through the close. There is no overlap of the bodies. The bullish kicker gaps up after a down bar. The bearish kicker gaps down after an up bar.",
        "Use it as a shock reversal, often driven by overnight news. The gap through the prior open is the part that distinguishes it from an ordinary engulfing bar.",
    ),
    "chart_double_top": (
        "A double top is two peaks at a similar price, separated by a pullback. A double bottom is two troughs. The pattern completes when price breaks the intervening neckline.",
        "Find two swing highs within a tolerance, default about 2 percent, with a trough between them. The neckline is that trough. A later close through the neckline completes the top. The bottom is the mirror image. Choosing the swings is the discretionary part, which is why this shape is a study entry.",
        "Use a completed neckline break as a reversal of the prior swing, and the distance from the peaks to the neckline as a measured objective. An incomplete second peak is only a retest.",
    ),
    "chart_hs": (
        "A head and shoulders is three peaks with the middle one highest, and a neckline through the two troughs. The inverse pattern has three troughs with the middle one lowest.",
        "The left shoulder, head, and right shoulder form a swing sequence, with the shoulders at a similar height and the head beyond them. The pattern completes on a break of the neckline. The inverse completes on a break up through its neckline. Wave labeling of this kind is not unique.",
        "Use a neckline break as the reversal event, and the head-to-neckline distance as a target. The right shoulder failing below the head is the story. Until the neckline breaks, it is still a possible shape.",
    ),
    "chart_triangle": (
        "A triangle is a consolidation between two converging trendlines. Ascending has a flat top and a rising bottom. Descending has a flat bottom and a falling top. Symmetrical has both lines sloping toward each other.",
        "Fit a line through the swing highs and a line through the swing lows. The lines converge. A break is a close beyond one of the lines. Which swings to use, and how many touches count, is a judgment, so the catalog keeps the shape as a study entry. The N-bar breakout is the mechanical relative.",
        "Use the break of a well-touched triangle as a continuation or a reversal depending on the prior trend and the direction of the break. Tightening ranges inside the triangle are the compression.",
    ),
    "chart_flag": (
        "A flag or pennant is a short pause after a sharp move. A flag is a small channel against the impulse. A pennant is a small triangle.",
        "An impulse of several times ATR is followed by a tight, short consolidation that drifts against the impulse or sideways. The pattern completes on a break in the direction of the impulse. The length of the pause is short relative to the pole. Marking the pole and the pause is discretionary.",
        "Use a break from the pause as a continuation entry, with a target often measured by the length of the pole. A pause that runs on too long stops being a flag.",
    ),
    "chart_wedge": (
        "A wedge is a pair of trendlines that both slope the same way and converge. A rising wedge drifts up. A falling wedge drifts down.",
        "Both the highs and the lows make lines in the same direction, and the distance between them narrows. A break against the slope of the wedge is the usual reversal. A rising wedge breaking down is the common bearish reading. Drawing the lines is a judgment.",
        "Use a break against the wedge as a sign the squeezed trend has failed. The converging slopes are what separate it from a channel, where the lines stay parallel.",
    ),
    "chart_cup_handle": (
        "A cup and handle is a rounded base followed by a shallow pullback, read as a bullish continuation.",
        "The cup is a U-shaped decline and recovery that returns to the old high, classically a moderate depth. The handle is a short, shallow drift near that high. The pattern completes on a break above the handle. The roundness and the depth are judged by eye, within rough limits such as a cup that is not a V.",
        "Use a break of the handle high as the entry, with the prior high as the level that had to be reclaimed first. The handle is the last shakeout before the break.",
    ),
    "chart_rectangle": (
        "A rectangle is a range: horizontal resistance tested more than once and horizontal support tested more than once.",
        "Several swing highs cluster at one price and several swing lows cluster at a lower price, with no meaningful slope. A close through either side completes the break. Deciding how many touches and how tight the cluster must be is the judgment.",
        "Use the range as a map while it lasts, and the break as the signal that the balance has resolved. Failed breaks back into the box are common, so the close through the level is the event to define.",
    ),
    "chart_breakout": (
        "An N-bar breakout is a close above the highest high of the previous N bars, or below the lowest low of those bars.",
        "Look back N bars, default 20, not including the current bar. A close above that highest high is an upside break. A close below that lowest low is a downside break. There is no trendline to draw. The same construction is the Donchian breakout.",
        "Use it as the mechanical chart-pattern entry: a new extreme of the window. Pair it with a volatility or trend filter if you want fewer breaks in a choppy name. The period is the entire definition of the setup.",
    ),
    "elliott": (
        "An Elliott wave count labels a trend as a five-wave impulse and a correction as a three-wave move, with rules about which waves may overlap.",
        "In a strict impulse, wave 2 does not pass the start of wave 1, wave 3 is not the shortest of waves 1, 3, and 5, and wave 4 does not overlap wave 1. Real counts differ between analysts because the swings can be grouped in more than one way. A ZigZag can propose swings. It cannot make the labeling unique.",
        "Use the rules as a discipline when you are already counting a market by hand. For a backtest, prefer a rule with one answer per bar, such as a break of a confirmed swing.",
    ),
    "td_sequential": (
        "TD Sequential counts a setup of consecutive closes beyond the close four bars earlier, then a countdown toward a possible exhaustion.",
        "A buy setup is nine consecutive closes below the close four bars earlier. A sell setup is nine closes above that reference. After a completed setup, the countdown looks for thirteen closes beyond the close two bars earlier, with interruptions allowed. Defaults are setup 9 and countdown 13. A completed count says the move is mature in this arithmetic.",
        "Use a completed 9 as an early exhaustion warning and a completed 13 as the fuller one. DeMark's method also has qualifiers and cancellation rules. This entry is the count itself.",
    ),
    "td_setup": (
        "TD setup is the running count of closes that stay beyond the close from four bars earlier, without the later countdown.",
        "Each bar that closes below the close four bars ago adds one to a buy count, and the count resets when the comparison fails. The sell count adds when the close is above the close four bars ago. The lookback default is 4. A print of 9 is the completed setup.",
        "Use the count to see how stretched a one-directional sequence of closes is. It is the part of TD Sequential that many short-term traders watch on its own.",
    ),
    "consec_bars": (
        "This is the length of the current streak of up closes or down closes.",
        "If today's close is above yesterday's, add one to a positive streak, or start at +1 if the previous streak was down. A down close does the same with negative numbers. An unchanged close holds the streak. The value is the signed length.",
        "Use a long positive streak as a run that may be late for a fresh entry, which is the exhaustion reading, or as confirmation that a trend is underway. Connors RSI turns this streak into one of its three components.",
    ),
    "ibs": (
        "Internal bar strength is where the close sits inside the bar's own high-low range.",
        "It is (close - low) divided by (high - low). A close at the high scores 1. A close at the low scores 0. A close in the middle scores 0.5. If the range is zero the value is undefined. There is no lookback.",
        "Use a very low reading as a short-term washout: the bar closed at its low, and the next bars have historically tended to bounce in the research this measure comes from. A very high reading is the mirror image. It is a one-bar mean-reversion input.",
    ),
    "pct_from_ma": (
        "Percent from a moving average is how far the close is from its average, in percent of that average.",
        "Subtract an SMA, default 50, from the close, divide by the SMA, and multiply by 100. A reading of 10 means price is 10 percent above the average. A reading of -10 means it is 10 percent below.",
        "Use a large negative reading as a stretch away from the trend, which mean-reversion rules buy if the longer trend is still up, and a large positive reading as an extension. The percent form compares cleanly across names.",
    ),
    "rsi2": (
        "RSI(2) is the relative strength index with a length of 2. It is extremely sensitive and is used for short pullbacks.",
        "Apply the ordinary RSI formula with N = 2. Two strong down closes push it near 0. Two strong up closes push it near 100. The default in this entry is that length of 2, exposed as a parameter so it can be lengthened slightly.",
        "Use a very low RSI(2) in an uptrend as the Connors-style pullback entry, and exit when it snaps back above a high threshold. It is a timing tool. The trend filter lives elsewhere.",
    ),
    "pairs_spread": (
        "A pairs spread is the residual of one price regressed on another. The z-score of that residual says how far the relationship has stretched.",
        "Over the window, default 60, estimate a hedge ratio of the second series to the first. The spread is price A minus that ratio times price B. The z-score subtracts the spread's mean and divides by its standard deviation. It needs both series.",
        "Use a large z-score as the entry of a mean-reversion pair: short the rich leg and long the cheap leg, and exit when the z-score returns toward zero. The hedge ratio is what makes the two legs comparable.",
    ),
    "cointegration": (
        "The Engle-Granger test asks whether the residual of a pair is mean-reverting enough to call the two series cointegrated.",
        "Regress one series on the other over the window, default 250, and run an Augmented Dickey-Fuller test on the residual. A small p-value says the residual looks stationary. The test needs both series, and the p-value thresholds for a regression residual are stricter than for a raw series.",
        "Use a passing test as permission to treat the pair as a spread that has historically come back together. Re-estimate it. A relationship that passed a year ago may have broken.",
    ),
    "half_life": (
        "The half-life is the number of bars a spread would take to close half of its current deviation, under an Ornstein-Uhlenbeck fit.",
        "Regress the change in the spread on the previous level of the spread. The coefficient lambda is the speed of reversion. The half-life is -ln(2) / lambda when lambda is negative. The default window is 250. A positive lambda means the spread is pushing away, and the half-life is not meaningful.",
        "Use the half-life to set how long a pairs trade is allowed to work. A half-life of a few bars is a fast relationship. A half-life of many months may not be tradable on the capital you have.",
    ),
    "kalman_filter": (
        "A Kalman filter estimates a hidden level, or a changing hedge ratio, by blending a prediction with each new price.",
        "Predict the next state from the previous state. When the new price arrives, move the estimate part of the way toward it. Process noise, default a very small number, lets the hidden level wander. Measurement noise, default 0.001, says how much of each price to ignore. A higher measurement noise makes a smoother estimate.",
        "Use the filtered level as an adaptive average that updates one bar at a time, or use the same machinery to track a hedge ratio that drifts. The two noise settings are the whole personality of the filter.",
    ),
    "sharpe_roll": (
        "The rolling Sharpe ratio is the excess return of the window divided by the volatility of the window, annualized.",
        "Take the mean daily return minus the daily risk-free rate, divide by the standard deviation of daily returns, and multiply by the square root of 252. The default window is 60 and the risk-free input is 0. A higher number means more return per unit of up-and-down movement.",
        "Use it to compare recent paths on a risk-adjusted basis. A high Sharpe from a tiny sample is fragile, so keep the window long enough to contain a few different markets.",
    ),
    "sortino_roll": (
        "The rolling Sortino ratio is like the Sharpe ratio, except the risk in the denominator is only the downside deviation.",
        "Subtract a minimum acceptable return, default 0, from the mean return. Divide by the standard deviation of the returns that fall below that floor, and annualize. Upside volatility does not reduce the score. The default window is 60.",
        "Use it when a strategy's upside spikes should not be treated as risk. Two paths with the same Sharpe can have very different Sortino ratios if one of them only varies on the way up.",
    ),
    "drawdown": (
        "Drawdown is the percent decline from the running peak of the series. Max drawdown is the deepest such decline so far.",
        "Track the cumulative maximum. Drawdown is the current value divided by that peak, minus 1, so it is zero at a new high and negative below it. Max drawdown is the lowest drawdown observed up to that bar. On a price series the peak is a price peak. On an equity curve it is an equity peak.",
        "Use the current drawdown to see how far the path is from its best, and the max to see the worst hold a holder would have sat through. This desk scores strategies with return and this depth together.",
    ),
    "var_hist": (
        "Historical value at risk is a loss threshold taken from the empirical distribution of recent returns.",
        "Over the window, default 250, take the return at the (1 - confidence) quantile. With the default confidence of 0.95 this is the 5th percentile. The sign is flipped so a loss is reported as a positive amount of risk. It is what the worst twentieth of the recent bars exceeded.",
        "Use it as a plain, model-free loss figure for the window. It does not know about a loss worse than the window has already seen. A short window understates a quiet market.",
    ),
    "cvar": (
        "Conditional value at risk, or expected shortfall, is the average loss on the days that were worse than the value-at-risk threshold.",
        "Find the historical VaR at the chosen confidence, default 0.95, over the default 250 bars. Average the returns that sit at or beyond that loss, and report the magnitude. Because it averages the tail, one extreme day moves it more than VaR, which only marks the edge of the tail.",
        "Use it when you want the severity of bad days and not only the threshold they crossed. A risk limit based on expected shortfall reacts to how bad the worst days were.",
    ),
    "kelly": (
        "The Kelly fraction is the bet size that maximizes the growth of wealth given a win rate and a payoff ratio. Practitioners usually bet a half or a quarter of it.",
        "If W is the fraction of trades that win and R is the average win divided by the average loss, the fraction is W - (1 - W) / R. A negative result means the edge is not positive. The inputs are a closed trade list, which an indicator of bars does not have.",
        "Use it as an upper bound on position size once you have a real sample of trades. Full Kelly assumes you know the edge and can survive the path. A fraction of Kelly is the practical use.",
    ),
    "calmar": (
        "The Calmar ratio divides annualized return by the absolute value of the worst drawdown over the window.",
        "Compute the compound annual growth of the series over the default 3 years, and divide by the magnitude of the maximum drawdown in that span. A higher ratio means more growth per unit of the worst decline. A path that never declined has an undefined ratio.",
        "Use it to compare equity curves when the pain you care about is the worst peak-to-trough loss. This desk's decision ratio, CAGR over absolute max drawdown, is the same construction.",
    ),
    "vol_target": (
        "A volatility target scales a position so that its recent volatility matches a chosen annual volatility.",
        "Divide the target, default 15 percent a year, by the realized volatility over the window, default 20 bars. The result is a weight. If realized volatility is 30 percent, the weight is one half, so a full position is cut in half. If realized volatility is below the target, the weight rises above one and needs a cap you choose.",
        "Use it to keep risk more stable across quiet and wild periods. The position gets smaller as the market gets louder, which is the whole point of the target.",
    ),
    "atr_position": (
        "ATR position sizing chooses a share count so that a stop a fixed number of ATR away loses a fixed fraction of equity.",
        "The rupee risk is equity times the risk percent, default 1 percent. The risk per share is the ATR multiple, default 2, times ATR, default length 14. Shares are the rupee risk divided by the risk per share. A wider ATR or a wider stop cuts the share count.",
        "Use it so every trade risks a similar amount of money if the stop is hit. This is the sizing idea behind the desk's risk-per-ATR parameter. The stop still has to be placed for the risk to be real.",
    ),
    "seasonality_dow": (
        "Day-of-week and month seasonality is the average return inside each calendar bucket.",
        "Compute the one-bar return and average it by weekday, by month, or by a turn-of-month window. The output is a table of averages, one per bucket. A single number on the chart can show the average that belongs to today's bucket. Small samples make a pretty pattern that is only noise.",
        "Use it to notice a calendar tilt that is large and repeated, such as a month-end effect you have already decided to study. Require a long history before treating a weekday average as a tendency.",
    ),
    "time_filter": (
        "A session filter is a yes-or-no flag that is on only between two times of day.",
        "Compare the bar's timestamp with a start and an end, default 09:15 to 15:30 in Asia/Kolkata, the cash session. Bars inside the window pass. Bars outside it do not. On daily bars the timestamp is often midnight, so the filter is meaningful on intraday bars.",
        "Use it to restrict signals to the session you actually trade, or to exclude the open and the close if those minutes are too wild for the rule. It is a gate, not a forecast.",
    ),
    "event_flag": (
        "An event-window flag marks the bars within a set distance of a scheduled event, such as earnings, an expiry, or a macro release.",
        "For each bar, check whether any event time falls inside the window, default one bar on either side. The output is 1 inside a window and 0 outside. It needs an event calendar joined to the bars.",
        "Use it to turn a strategy off around announcements, or to study only those windows. The flag is as good as the calendar you feed it.",
    ),
    "fear_greed": (
        "A fear-and-greed composite blends several market readings, such as momentum, breadth, volatility, put/call, and demand for safety, into one mood score.",
        "Each component is normalized against its own history and the results are averaged. The exact components and the scale are chosen by the publisher. A low score is the fearful end and a high score is the greedy end. The inputs are those component series.",
        "Use an extreme low as a contrary setup only after you know which components are in the score and that you are not double-counting a market you already measure. The level matters less than the publisher's own history of extremes.",
    ),
    "short_interest": (
        "Days to cover is the short-interest position divided by average daily volume. It estimates how many sessions of typical volume it would take to buy the short back.",
        "Divide the reported shares sold short by the average volume. A higher number means the short is larger relative to the market that would have to absorb a cover. The report is periodic, often biweekly, so the series steps.",
        "Use a very high days-to-cover as a squeeze risk: if price rises, covering that position takes longer and can feed the rise. It is a positioning fact. A high short interest can also be a fundamental bet that is right.",
    ),
    "pe_ratio": (
        "The price/earnings ratio divides the share price by earnings per share.",
        "Divide price by trailing or by forward earnings per share. A higher number means you pay more for a rupee of earnings. A negative earnings figure makes the ratio meaningless, and a one-time earnings spike makes it look cheap. It needs the earnings input.",
        "Use it to compare the market's price of earnings across names in a similar business, or against the same name's own history. The comparison fails across businesses with different growth and accounting.",
    ),
    "pb_ratio": (
        "The price/book ratio divides the share price by book value per share.",
        "Book value per share is shareholders' equity divided by the share count. Divide price by that book value. Below 1 means the market prices the company under the accounting value of its equity. Asset-light businesses often look permanently expensive on this ratio. It needs the book figure.",
        "Use it for banks, insurers, and other businesses where the balance sheet is the franchise. A low ratio is a value clue only if the book value is real.",
    ),
    "ev_ebitda": (
        "EV/EBITDA divides enterprise value by earnings before interest, tax, depreciation, and amortization.",
        "Enterprise value is the market value of equity plus net debt. Dividing by EBITDA puts capital structure to the side, so a levered and an unlevered peer are more comparable than on a price/earnings ratio. It needs both the enterprise value and the EBITDA.",
        "Use it to compare operating earnings across companies with different debt levels. A low multiple is cheap only relative to peers and to the company's own history, and EBITDA ignores real capital spending.",
    ),
    "roe": (
        "Return on equity is net income divided by shareholders' equity.",
        "Divide the period's net income by the equity, often an average of beginning and ending equity. A higher figure means more profit per rupee of book equity. Leverage raises it, because the same assets on a smaller equity base look more profitable. It needs the accounts.",
        "Use a high and stable return on equity as a quality clue, and check the leverage beside it so you do not praise a ratio that debt created. A falling series says the business is earning less on its capital.",
    ),
    "piotroski": (
        "The Piotroski F-score adds up nine yes-or-no tests of profitability, leverage, liquidity, and operating efficiency. The score runs from 0 to 9.",
        "The tests include positive return on assets, positive operating cash flow, improving cash versus earnings, a falling long-term debt ratio, a higher current ratio, no new shares, a rising gross margin, and a rising asset turnover, plus the sign of return on assets. Each pass adds one. The inputs are the financial statements.",
        "Use a high score, especially on a stock that already looks cheap on book value, as Piotroski's quality filter. A low score on a cheap stock is the value trap he was trying to avoid.",
    ),
    "altman_z": (
        "The Altman Z-score combines five balance-sheet and income ratios into a bankruptcy-risk score.",
        "The original manufacturing form is 1.2 times working capital over assets, plus 1.4 times retained earnings over assets, plus 3.3 times EBIT over assets, plus 0.6 times market equity over liabilities, plus sales over assets. Lower scores were associated with distress. Later variants change the weights for private and non-manufacturing firms.",
        "Use a low score as a distress flag before treating a cheap multiple as a bargain. The zones Altman published belong to the model variant you actually computed.",
    ),
    "momentum_12_1": (
        "Twelve-minus-one momentum is the return over the past year, skipping the most recent month. It is the classic cross-sectional momentum factor.",
        "Divide the close from 21 sessions ago by the close from 252 sessions ago, and subtract 1. The default lookback is 252 and the skip is 21. Skipping the last month avoids the short-term reversal that often shows up in the most recent weeks. Rank the result across the universe to find winners and losers.",
        "Use the leaders of this ranking as the momentum portfolio and the laggards as the short side, rebalanced on a schedule. The skip is part of the factor. Including the last month changes the bet.",
    ),
    "low_vol_factor": (
        "The low-volatility factor is realized volatility used as a rank. Names with lower volatility have historically earned more return per unit of risk than a simple risk model predicts.",
        "Compute the standard deviation of daily returns over the window, default 252. Rank the universe from quiet to loud. The factor portfolio prefers the quiet end. The number on a single name is just its volatility. The factor is the rank.",
        "Use a low reading, in a universe rank, as a candidate for a defensive portfolio. Pair it with a liquidity check so the quiet names are actually tradable.",
    ),
    "value_factor": (
        "A value composite averages the ranks of several valuation ratios, such as earnings yield, book to price, and sales to enterprise value.",
        "Rank each ratio across the universe so cheap is high, then average the ranks. Using yields rather than multiples keeps a negative earnings name from breaking the scale. The inputs are the fundamental fields.",
        "Use a high composite as a broadly cheap stock, cheaper than a single ratio can establish. One ratio can be distorted by a single accounting item. The average is the protection.",
    ),
    "quality_factor": (
        "A quality composite averages the ranks of profitability and stability measures, such as return on equity, gross margin, and low leverage.",
        "Rank each measure across the universe, with low leverage scored so that less debt ranks higher, and average the ranks. The inputs are the accounts. A high score is a profitable, stable, lightly levered business relative to the universe.",
        "Use a high score as a quality filter on top of value or momentum, so a cheap or strong name is also a business that earns well and is not drowning in debt.",
    ),
}
