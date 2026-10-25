# Trading Indicator Catalog

inputs use: O,H,L,C,V (open, high, low, close, volume), OI (open interest), P (any price series). Params are defaults. Outputs are the series the indicator returns.

Total indicators: 230

## Contents

- Trend: Moving Averages (18)
- Trend: Bands and Channels (1)
- Trend: Trend Strength and Direction (13)
- Momentum (29)
- Volatility (19)
- Volume (21)
- Support/Resistance (12)
- Price Transforms (9)
- Statistical (13)
- Cycle (9)
- Market Structure (9)
- Relative/Breadth (11)
- Options (5)
- Futures/Crypto (3)
- Candlestick Patterns (15)
- Chart Patterns (9)
- Exhaustion (3)
- Mean Reversion (7)
- Risk/Performance (9)
- Calendar (3)
- Sentiment (2)
- Fundamental (6)
- Factor (4)


## Trend: Moving Averages

### Simple Moving Average (`sma`)
Arithmetic mean of the last N values. Smooths noise; lags price by about (N-1)/2 bars.

- Inputs: P
- Default params: period=20
- Outputs: sma
- Formula: `SMA_t = (P_t + P_{t-1} + ... + P_{t-N+1}) / N`

### Exponential Moving Average (`ema`)
Moving average that weights recent prices more heavily using an exponentially decaying factor.

- Inputs: P
- Default params: period=20
- Outputs: ema
- Formula: `k = 2/(N+1); EMA_t = k*P_t + (1-k)*EMA_{t-1}`

### Weighted Moving Average (`wma`)
Moving average with linearly decreasing weights, newest bar weighted N and oldest weighted 1.

- Inputs: P
- Default params: period=20
- Outputs: wma
- Formula: `WMA = sum(i*P_{t-N+i}, i=1..N) / (N(N+1)/2)`

### Double Exponential Moving Average (`dema`)
Reduces EMA lag by combining an EMA with an EMA of that EMA.

- Inputs: P
- Default params: period=20
- Outputs: dema
- Formula: `DEMA = 2*EMA(P,N) - EMA(EMA(P,N),N)`

### Triple Exponential Moving Average (`tema`)
Further lag reduction using three cascaded EMAs.

- Inputs: P
- Default params: period=20
- Outputs: tema
- Formula: `TEMA = 3*E1 - 3*E2 + E3, where E1=EMA(P), E2=EMA(E1), E3=EMA(E2)`

### Hull Moving Average (`hma`)
Fast, smooth moving average built from weighted MAs to cut lag and keep smoothness.

- Inputs: P
- Default params: period=20
- Outputs: hma
- Formula: `HMA = WMA(2*WMA(P,N/2) - WMA(P,N), sqrt(N))`

### Smoothed Moving Average (Wilder / RMA) (`smma`)
EMA variant with alpha = 1/N, used inside RSI, ATR and ADX.

- Inputs: P
- Default params: period=14
- Outputs: smma
- Formula: `SMMA_t = (SMMA_{t-1}*(N-1) + P_t) / N`

### Kaufman Adaptive Moving Average (`kama`)
Moving average that adapts its smoothing speed to the efficiency ratio: fast in trends, slow in chop.

- Inputs: P
- Default params: period=10, fast=2, slow=30
- Outputs: kama
- Formula: `ER = |P_t-P_{t-N}| / sum|P_i-P_{i-1}|; SC = (ER*(2/(fast+1)-2/(slow+1)) + 2/(slow+1))^2; KAMA_t = KAMA_{t-1} + SC*(P_t-KAMA_{t-1})`

### Variable Index Dynamic Average (`vidya`)
EMA whose smoothing constant is scaled by the absolute Chande Momentum Oscillator (volatility/momentum).

- Inputs: P
- Default params: period=14, cmo_period=9
- Outputs: vidya
- Formula: `VIDYA_t = a*|CMO|*P_t + (1 - a*|CMO|)*VIDYA_{t-1}, a=2/(N+1)`

### Arnaud Legoux Moving Average (`alma`)
Gaussian-weighted moving average with adjustable offset to trade smoothness against responsiveness.

- Inputs: P
- Default params: period=9, offset=0.85, sigma=6
- Outputs: alma
- Formula: `w_i = exp(-(i-m)^2/(2s^2)), m=offset*(N-1), s=N/sigma; ALMA = sum(w_i*P_i)/sum(w_i)`

### Zero-Lag EMA (`zlema`)
EMA applied to a de-lagged price series.

- Inputs: P
- Default params: period=20
- Outputs: zlema
- Formula: `lag=(N-1)/2; ZLEMA = EMA(2*P_t - P_{t-lag}, N)`

### Tillson T3 (`t3`)
Six-stage cascaded EMA with a volume factor giving a very smooth, low-lag average.

- Inputs: P
- Default params: period=5, v_factor=0.7
- Outputs: t3
- Formula: `T3 = c1*E6 + c2*E5 + c3*E4 + c4*E3, with c from v-factor`

### Triangular Moving Average (`trima`)
Double-smoothed SMA that puts the most weight on the middle of the window.

- Inputs: P
- Default params: period=20
- Outputs: trima
- Formula: `TRIMA = SMA(SMA(P, ceil((N+1)/2)), ceil((N+1)/2))`

### Volume-Weighted Moving Average (`vwma`)
Moving average weighting each close by its bar volume.

- Inputs: C, V
- Default params: period=20
- Outputs: vwma
- Formula: `VWMA = sum(C_i*V_i)/sum(V_i) over N bars`

### Least Squares Moving Average (Linear Regression Value) (`lsma`)
Endpoint of the least-squares regression line fitted over the last N bars.

- Inputs: P
- Default params: period=25
- Outputs: lsma
- Formula: `LSMA = intercept + slope*(N-1) from OLS of P on bar index`

### McGinley Dynamic (`mcginley`)
Self-adjusting moving average that tracks price more closely in fast markets and avoids whipsaw.

- Inputs: P
- Default params: period=14
- Outputs: md
- Formula: `MD_t = MD_{t-1} + (P_t - MD_{t-1}) / (k*N*(P_t/MD_{t-1})^4), k=0.6`

### Fractal Adaptive Moving Average (`frama`)
EMA whose alpha is derived from the fractal dimension of price.

- Inputs: H, L, C
- Default params: period=16
- Outputs: frama
- Formula: `D = (ln(N1+N2)-ln(N3))/ln2; alpha = exp(-4.6*(D-1)), bounded to [0.01,1]`

### Jurik-style Adaptive Moving Average (`jma`)
Low-lag, low-noise adaptive average; open-source approximations are commonly used since the original is proprietary.

- Inputs: P
- Default params: period=14, phase=0
- Outputs: jma
- Formula: `Adaptive volatility-banded smoothing (implementation-specific approximation)`


## Trend: Bands and Channels

### Moving Average Envelope (`ma_envelope`)
Fixed-percentage bands above and below a moving average.

- Inputs: P
- Default params: period=20, percent=2.5
- Outputs: upper, middle, lower
- Formula: `upper = MA*(1+pct/100); lower = MA*(1-pct/100)`


## Trend: Trend Strength and Direction

### MACD (`macd`)
Difference between fast and slow EMAs, with a signal EMA and histogram; measures trend direction and momentum.

- Inputs: C
- Default params: fast=12, slow=26, signal=9
- Outputs: macd, signal, histogram
- Formula: `MACD = EMA(C,fast)-EMA(C,slow); signal = EMA(MACD,sig); hist = MACD-signal`

### Average Directional Index (with +DI/-DI) (`adx`)
Measures trend strength regardless of direction (0-100); +DI/-DI show direction.

- Inputs: H, L, C
- Default params: period=14
- Outputs: adx, plus_di, minus_di
- Formula: `+DM=H_t-H_{t-1} (if > down move and >0); -DM likewise; DI=100*smooth(DM)/ATR; DX=100*|+DI - -DI|/(+DI + -DI); ADX=smooth(DX)`

### Average Directional Movement Rating (`adxr`)
Smoothed ADX comparing the current ADX with ADX N bars ago.

- Inputs: H, L, C
- Default params: period=14
- Outputs: adxr
- Formula: `ADXR = (ADX_t + ADX_{t-N}) / 2`

### Directional Movement Index (`dmi`)
Directional components underlying ADX; crossovers signal trend changes.

- Inputs: H, L, C
- Default params: period=14
- Outputs: plus_di, minus_di
- Formula: `+DI/-DI as in ADX`

### Aroon (`aroon`)
Measures how recently the highest high and lowest low occurred within N bars.

- Inputs: H, L
- Default params: period=25
- Outputs: aroon_up, aroon_down
- Formula: `Up = 100*(N - bars since highest high)/N; Down = 100*(N - bars since lowest low)/N`

### Aroon Oscillator (`aroon_osc`)
Difference between Aroon Up and Aroon Down.

- Inputs: H, L
- Default params: period=25
- Outputs: aroon_osc
- Formula: `Aroon Up - Aroon Down`

### Parabolic SAR (`psar`)
Trailing stop-and-reverse that accelerates toward price as a trend extends.

- Inputs: H, L
- Default params: af_start=0.02, af_step=0.02, af_max=0.2
- Outputs: sar
- Formula: `SAR_{t+1} = SAR_t + AF*(EP - SAR_t); AF increases by step on each new extreme point EP up to max`

### SuperTrend (`supertrend`)
ATR-based trailing line that flips between support and resistance as price closes through it.

- Inputs: H, L, C
- Default params: period=10, multiplier=3
- Outputs: supertrend, direction
- Formula: `basic = (H+L)/2 +/- mult*ATR; final bands ratchet in trend direction; flip when close crosses`

### Ichimoku Cloud (`ichimoku`)
Multi-line system showing trend, momentum and support/resistance via a forward-projected cloud.

- Inputs: H, L, C
- Default params: tenkan=9, kijun=26, senkou_b=52, displacement=26
- Outputs: tenkan, kijun, senkou_a, senkou_b, chikou
- Formula: `Tenkan=(HH9+LL9)/2; Kijun=(HH26+LL26)/2; SenkouA=(Tenkan+Kijun)/2 shifted +26; SenkouB=(HH52+LL52)/2 shifted +26; Chikou=Close shifted -26`

### Vortex Indicator (`vortex`)
Detects trend starts using upward and downward vortex movement relative to true range.

- Inputs: H, L, C
- Default params: period=14
- Outputs: vi_plus, vi_minus
- Formula: `VM+=|H_t-L_{t-1}|; VM-=|L_t-H_{t-1}|; VI+=sum(VM+)/sum(TR); VI-=sum(VM-)/sum(TR)`

### Mass Index (`mass_index`)
Detects reversals by tracking range expansion ('reversal bulge').

- Inputs: H, L
- Default params: ema_period=9, sum_period=25
- Outputs: mass
- Formula: `sum over 25 of EMA9(H-L)/EMA9(EMA9(H-L))`

### Choppiness Index (`choppiness`)
Scale from 0-100 where high values indicate sideways markets and low values trending.

- Inputs: H, L, C
- Default params: period=14
- Outputs: chop
- Formula: `100*log10(sum(ATR1,N)/(HH_N-LL_N))/log10(N)`

### Vertical Horizontal Filter (`vhf`)
Distinguishes trending from congested phases.

- Inputs: C
- Default params: period=28
- Outputs: vhf
- Formula: `(max C - min C)/sum|C_i - C_{i-1}|`


## Momentum

### Percentage Price Oscillator (`ppo`)
MACD expressed as a percentage of the slow EMA, so it is comparable across instruments.

- Inputs: C
- Default params: fast=12, slow=26, signal=9
- Outputs: ppo, signal, histogram
- Formula: `PPO = 100*(EMA_fast-EMA_slow)/EMA_slow`

### TRIX (`trix`)
Rate of change of a triple-smoothed EMA; filters out minor cycles.

- Inputs: C
- Default params: period=15
- Outputs: trix
- Formula: `TRIX = 100*(EMA3_t - EMA3_{t-1})/EMA3_{t-1}`

### Commodity Channel Index (`cci`)
Measures deviation of typical price from its mean in units of mean deviation.

- Inputs: H, L, C
- Default params: period=20
- Outputs: cci
- Formula: `TP=(H+L+C)/3; CCI=(TP-SMA(TP,N))/(0.015*MeanDev(TP,N))`

### Relative Strength Index (`rsi`)
Bounded oscillator (0-100) comparing average gains to average losses.

- Inputs: C
- Default params: period=14
- Outputs: rsi
- Formula: `RS = SMMA(gain,N)/SMMA(loss,N); RSI = 100 - 100/(1+RS)`

### Stochastic RSI (`stoch_rsi`)
Stochastic formula applied to RSI values to increase sensitivity.

- Inputs: C
- Default params: rsi_period=14, stoch_period=14, k=3, d=3
- Outputs: k, d
- Formula: `(RSI - min RSI_N)/(max RSI_N - min RSI_N)`

### Stochastic Oscillator (`stoch`)
Locates close within the recent high-low range.

- Inputs: H, L, C
- Default params: k_period=14, k_smooth=3, d_period=3
- Outputs: k, d
- Formula: `%K = 100*(C-LL_N)/(HH_N-LL_N); %D = SMA(%K,3)`

### Williams %R (`williams_r`)
Inverted stochastic, ranging from -100 to 0.

- Inputs: H, L, C
- Default params: period=14
- Outputs: wr
- Formula: `%R = -100*(HH_N-C)/(HH_N-LL_N)`

### Rate of Change (`roc`)
Percentage change in price over N bars.

- Inputs: P
- Default params: period=12
- Outputs: roc
- Formula: `ROC = 100*(P_t - P_{t-N})/P_{t-N}`

### Momentum (`mom`)
Absolute price change over N bars.

- Inputs: P
- Default params: period=10
- Outputs: mom
- Formula: `MOM = P_t - P_{t-N}`

### Chande Momentum Oscillator (`cmo`)
Momentum oscillator (-100 to +100) using raw up and down sums, unsmoothed.

- Inputs: C
- Default params: period=14
- Outputs: cmo
- Formula: `CMO = 100*(Su - Sd)/(Su + Sd)`

### True Strength Index (`tsi`)
Double-smoothed momentum normalized by double-smoothed absolute momentum.

- Inputs: C
- Default params: long=25, short=13, signal=7
- Outputs: tsi, signal
- Formula: `TSI = 100*EMA(EMA(dP,long),short)/EMA(EMA(|dP|,long),short)`

### Ultimate Oscillator (`ultimate_osc`)
Weighted blend of buying pressure over three timeframes to reduce false divergences.

- Inputs: H, L, C
- Default params: p1=7, p2=14, p3=28
- Outputs: uo
- Formula: `BP=C-min(L,prevC); TR=max(H,prevC)-min(L,prevC); UO=100*(4*A1+2*A2+A3)/7, Ai=sum(BP)/sum(TR)`

### Awesome Oscillator (`awesome_osc`)
Difference between fast and slow SMAs of median price.

- Inputs: H, L
- Default params: fast=5, slow=34
- Outputs: ao
- Formula: `AO = SMA((H+L)/2,5) - SMA((H+L)/2,34)`

### Accelerator Oscillator (`accel_osc`)
Awesome Oscillator minus its SMA; shows acceleration of momentum.

- Inputs: H, L
- Default params: ao_signal=5
- Outputs: ac
- Formula: `AC = AO - SMA(AO,5)`

### Detrended Price Oscillator (`dpo`)
Removes the trend by comparing price to a displaced SMA to isolate cycles.

- Inputs: C
- Default params: period=20
- Outputs: dpo
- Formula: `DPO = C_{t-(N/2+1)} - SMA(C,N)`

### Know Sure Thing (`kst`)
Weighted sum of four smoothed rate-of-change values across timeframes.

- Inputs: C
- Default params: roc=[10, 15, 20, 30], sma=[10, 10, 10, 15], signal=9
- Outputs: kst, signal
- Formula: `KST = 1*SMA(ROC1)+2*SMA(ROC2)+3*SMA(ROC3)+4*SMA(ROC4)`

### Coppock Curve (`coppock`)
Long-term momentum indicator for identifying market bottoms.

- Inputs: C
- Default params: roc1=14, roc2=11, wma=10
- Outputs: coppock
- Formula: `WMA(ROC14 + ROC11, 10)`

### Connors RSI (`connors_rsi`)
Average of short-term RSI, RSI of up/down streak length and percentile rank of ROC.

- Inputs: C
- Default params: rsi=3, streak=2, rank=100
- Outputs: crsi
- Formula: `CRSI = (RSI(C,3)+RSI(streak,2)+PercentRank(ROC1,100))/3`

### Relative Vigor Index (`rvi`)
Compares close-open to high-low range, assuming closes are higher than opens in uptrends.

- Inputs: O, H, L, C
- Default params: period=10
- Outputs: rvi, signal
- Formula: `RVI = SMA(swma(C-O),N)/SMA(swma(H-L),N)`

### Relative Momentum Index (`rmi`)
RSI variant using momentum over k bars instead of 1-bar change.

- Inputs: C
- Default params: period=14, momentum=5
- Outputs: rmi
- Formula: `RSI formula applied to C_t - C_{t-k}`

### Schaff Trend Cycle (`stc`)
MACD passed through a double stochastic to produce a faster cyclical oscillator.

- Inputs: C
- Default params: fast=23, slow=50, cycle=10
- Outputs: stc
- Formula: `Stoch(Stoch(MACD))`

### Price Momentum Oscillator (`pmo`)
Double-smoothed rate of change.

- Inputs: C
- Default params: s1=35, s2=20, signal=10
- Outputs: pmo, signal
- Formula: `PMO = EMA(EMA(10*ROC1, s1), s2)`

### Fisher Transform (`fisher`)
Transforms prices to an approximately Gaussian distribution to flag sharp turning points.

- Inputs: H, L
- Default params: period=10
- Outputs: fisher, trigger
- Formula: `x=0.66*((mid-LL)/(HH-LL)-0.5)+0.67*x_prev; F=0.5*ln((1+x)/(1-x))+0.5*F_prev`

### Inertia (`inertia`)
Linear regression smoothing of the Relative Volatility Index.

- Inputs: C
- Default params: rvi_period=14, reg_period=20
- Outputs: inertia
- Formula: `LinReg(RVI, N)`

### Squeeze Momentum (`squeeze_mom`)
Detects volatility compression (Bollinger inside Keltner) and plots a regression-based momentum histogram.

- Inputs: H, L, C
- Default params: bb=20, kc=20, bb_mult=2, kc_mult=1.5
- Outputs: momentum, squeeze_on
- Formula: `squeeze_on = BB inside KC; mom = LinReg(C - avg(avg(HH,LL),SMA(C)), N)`

### Elder Ray (Bull/Bear Power) (`elder_ray`)
Measures buyer and seller strength relative to an EMA.

- Inputs: H, L, C
- Default params: period=13
- Outputs: bull_power, bear_power
- Formula: `Bull = H - EMA(C,N); Bear = L - EMA(C,N)`

### Elder Impulse System (`elder_impulse`)
Colors bars by combined slope of EMA and MACD histogram (green, red, blue).

- Inputs: C
- Default params: ema=13, macd=[12, 26, 9]
- Outputs: impulse
- Formula: `green if EMA and hist rising; red if both falling; else blue`

### Balance of Power (`bop`)
Gauges buyers vs sellers by close-open relative to range.

- Inputs: O, H, L, C
- Default params: smooth=14
- Outputs: bop
- Formula: `BOP = (C - O)/(H - L)`

### QStick (`qstick`)
Moving average of close-open candle bodies.

- Inputs: O, C
- Default params: period=8
- Outputs: qstick
- Formula: `SMA(C - O, N)`


## Volatility

### Chaikin Volatility (`chaikin_vol`)
Rate of change of the EMA of the high-low range.

- Inputs: H, L
- Default params: ema=10, roc=10
- Outputs: cv
- Formula: `100*(EMA(H-L,N)_t - EMA(H-L,N)_{t-R})/EMA(H-L,N)_{t-R}`

### True Range (`tr`)
Largest of the current range and gaps from the previous close.

- Inputs: H, L, C
- Default params: none
- Outputs: tr
- Formula: `max(H-L, |H-C_prev|, |L-C_prev|)`

### Average True Range (`atr`)
Wilder-smoothed average of true range; the standard volatility and stop-distance measure.

- Inputs: H, L, C
- Default params: period=14
- Outputs: atr
- Formula: `ATR = SMMA(TR, N)`

### Normalized ATR (`natr`)
ATR as a percentage of close, comparable across instruments.

- Inputs: H, L, C
- Default params: period=14
- Outputs: natr
- Formula: `100*ATR/C`

### Bollinger Bands (`bbands`)
SMA with bands at k standard deviations; width reflects volatility.

- Inputs: P
- Default params: period=20, stddev=2
- Outputs: upper, middle, lower
- Formula: `upper = SMA + k*sigma; lower = SMA - k*sigma`

### Bollinger %B (`bb_percent_b`)
Position of price relative to Bollinger Bands.

- Inputs: P
- Default params: period=20, stddev=2
- Outputs: pctb
- Formula: `(P - lower)/(upper - lower)`

### Bollinger Bandwidth (`bb_width`)
Relative distance between the bands; low values signal a squeeze.

- Inputs: P
- Default params: period=20, stddev=2
- Outputs: bbw
- Formula: `(upper - lower)/middle`

### Keltner Channels (`keltner`)
EMA centerline with ATR-based bands.

- Inputs: H, L, C
- Default params: ema=20, atr=10, mult=2
- Outputs: upper, middle, lower
- Formula: `EMA(C) +/- mult*ATR`

### Donchian Channels (`donchian`)
Highest high and lowest low over N bars; basis of turtle breakout systems.

- Inputs: H, L
- Default params: period=20
- Outputs: upper, middle, lower
- Formula: `upper=HH_N; lower=LL_N; middle=(upper+lower)/2`

### STARC Bands (`starc`)
SMA with ATR bands (Stoller Average Range Channels).

- Inputs: H, L, C
- Default params: sma=6, atr=15, mult=2
- Outputs: upper, middle, lower
- Formula: `SMA(C) +/- mult*ATR`

### Chandelier Exit (`chandelier`)
ATR trailing stop hung from the highest high (long) or lowest low (short).

- Inputs: H, L, C
- Default params: period=22, mult=3
- Outputs: long_exit, short_exit
- Formula: `long = HH_N - mult*ATR; short = LL_N + mult*ATR`

### Historical (Realized) Volatility (`hist_vol`)
Annualized standard deviation of log returns.

- Inputs: C
- Default params: period=20, annualization=252
- Outputs: hv
- Formula: `stdev(ln(C_t/C_{t-1}), N)*sqrt(annualization)`

### Standard Deviation (`stddev`)
Dispersion of price around its mean.

- Inputs: P
- Default params: period=20
- Outputs: stddev
- Formula: `sqrt(sum((P-mean)^2)/N)`

### Relative Volatility Index (`rvi_vol`)
RSI applied to standard deviation instead of price change.

- Inputs: C
- Default params: stddev=10, smooth=14
- Outputs: rvi
- Formula: `RSI formula using stdev up/down depending on price direction`

### Ulcer Index (`ulcer`)
Downside-only volatility based on depth and duration of drawdowns.

- Inputs: C
- Default params: period=14
- Outputs: ui
- Formula: `sqrt(mean(((C-max(C,N))/max(C,N)*100)^2))`

### Parkinson Volatility (`parkinson`)
Range-based volatility estimator using high and low prices.

- Inputs: H, L
- Default params: period=20
- Outputs: vol
- Formula: `sqrt(1/(4N ln2) * sum(ln(H/L)^2))`

### Garman-Klass Volatility (`garman_klass`)
OHLC-based volatility estimator more efficient than close-to-close.

- Inputs: O, H, L, C
- Default params: period=20
- Outputs: vol
- Formula: `sqrt(mean(0.5*ln(H/L)^2 - (2ln2-1)*ln(C/O)^2))`

### Rogers-Satchell Volatility (`rogers_satchell`)
Drift-independent OHLC volatility estimator.

- Inputs: O, H, L, C
- Default params: period=20
- Outputs: vol
- Formula: `sqrt(mean(ln(H/C)ln(H/O)+ln(L/C)ln(L/O)))`

### Yang-Zhang Volatility (`yang_zhang`)
Combines overnight, open-close and Rogers-Satchell variance; handles gaps and drift.

- Inputs: O, H, L, C
- Default params: period=20
- Outputs: vol
- Formula: `sigma^2 = sigma_o^2 + k*sigma_c^2 + (1-k)*sigma_RS^2`


## Volume

### On-Balance Volume (`obv`)
Cumulative volume signed by the direction of the close.

- Inputs: C, V
- Default params: none
- Outputs: obv
- Formula: `OBV_t = OBV_{t-1} + V if C>C_prev; - V if C<C_prev; else +0`

### Accumulation/Distribution Line (`ad_line`)
Cumulative money flow volume based on where the close sits in the bar's range.

- Inputs: H, L, C, V
- Default params: none
- Outputs: adl
- Formula: `MFM=((C-L)-(H-C))/(H-L); ADL += MFM*V`

### Chaikin Money Flow (`cmf`)
Sum of money flow volume divided by sum of volume over N bars.

- Inputs: H, L, C, V
- Default params: period=20
- Outputs: cmf
- Formula: `sum(MFM*V,N)/sum(V,N)`

### Chaikin Oscillator (`chaikin_osc`)
MACD applied to the Accumulation/Distribution Line.

- Inputs: H, L, C, V
- Default params: fast=3, slow=10
- Outputs: co
- Formula: `EMA(ADL,3) - EMA(ADL,10)`

### Money Flow Index (`mfi`)
Volume-weighted RSI.

- Inputs: H, L, C, V
- Default params: period=14
- Outputs: mfi
- Formula: `MF=TP*V; MFR=sum(posMF)/sum(negMF); MFI=100-100/(1+MFR)`

### Volume-Weighted Average Price (`vwap`)
Cumulative average price weighted by volume, usually reset each session.

- Inputs: H, L, C, V
- Default params: anchor=session
- Outputs: vwap
- Formula: `sum(TP*V)/sum(V) since anchor`

### Anchored VWAP (`avwap`)
VWAP started from a user-chosen bar (e.g., swing low or earnings).

- Inputs: H, L, C, V
- Default params: anchor_index=0
- Outputs: avwap
- Formula: `sum(TP*V)/sum(V) from anchor bar`

### VWAP Bands (`vwap_bands`)
Volume-weighted standard deviation bands around VWAP.

- Inputs: H, L, C, V
- Default params: mult=[1, 2]
- Outputs: upper1, lower1, upper2, lower2
- Formula: `VWAP +/- k*volume-weighted sigma`

### Ease of Movement (`eom`)
Relates price change to volume to show how easily price moves.

- Inputs: H, L, V
- Default params: period=14, scale=100000000
- Outputs: eom
- Formula: `((H+L)/2 - (Hp+Lp)/2) / ((V/scale)/(H-L)), then SMA`

### Force Index (`force_index`)
Price change multiplied by volume, smoothed with an EMA.

- Inputs: C, V
- Default params: period=13
- Outputs: fi
- Formula: `EMA((C - C_prev)*V, N)`

### Negative Volume Index (`nvi`)
Updates only on days when volume falls; tracks 'smart money' behavior.

- Inputs: C, V
- Default params: none
- Outputs: nvi
- Formula: `NVI_t = NVI_{t-1}*(1 + dC/C_prev) if V<V_prev else unchanged`

### Positive Volume Index (`pvi`)
Updates only on days when volume rises; tracks crowd behavior.

- Inputs: C, V
- Default params: none
- Outputs: pvi
- Formula: `PVI_t = PVI_{t-1}*(1 + dC/C_prev) if V>V_prev else unchanged`

### Price Volume Trend (`pvt`)
Cumulative volume scaled by percentage price change.

- Inputs: C, V
- Default params: none
- Outputs: pvt
- Formula: `PVT += V*(C - C_prev)/C_prev`

### Volume Rate of Change (`vroc`)
Percentage change in volume over N bars.

- Inputs: V
- Default params: period=14
- Outputs: vroc
- Formula: `100*(V - V_{t-N})/V_{t-N}`

### Klinger Volume Oscillator (`kvo`)
Compares volume force to price trend to detect long-term money flow.

- Inputs: H, L, C, V
- Default params: fast=34, slow=55, signal=13
- Outputs: kvo, signal
- Formula: `EMA(VF,34) - EMA(VF,55), VF = V*trend*|2*dm/cm - 1|*100`

### Volume Oscillator (`vol_osc`)
Percentage difference between fast and slow volume moving averages.

- Inputs: V
- Default params: fast=5, slow=10
- Outputs: vo
- Formula: `100*(EMA(V,fast)-EMA(V,slow))/EMA(V,slow)`

### Twiggs Money Flow (`twiggs`)
CMF variant using true high/low and Wilder smoothing.

- Inputs: H, L, C, V
- Default params: period=21
- Outputs: tmf
- Formula: `EMA(AD*V)/EMA(V), AD uses TrueHigh/TrueLow`

### Volume Profile (`vol_profile`)
Histogram of traded volume by price level, with POC (point of control) and value area.

- Inputs: H, L, C, V
- Default params: bins=24, lookback=100
- Outputs: profile, poc, vah, val
- Formula: `bin price range; sum volume per bin; POC=max bin; value area around POC holding ~70% volume`

### Market Facilitation Index (`mfi_bw`)
Price range per unit of volume.

- Inputs: H, L, V
- Default params: none
- Outputs: mfi_bw
- Formula: `(H - L)/V`

### Open Interest Change (`oi_change`)
Change in futures/options open interest; confirms trend participation.

- Inputs: OI
- Default params: period=1
- Outputs: oi_delta
- Formula: `OI_t - OI_{t-N}`

### Cumulative Volume Delta (`cvd`)
Running total of aggressive buy volume minus aggressive sell volume (needs tick or bid/ask-split data).

- Inputs: V_buy, V_sell
- Default params: none
- Outputs: cvd
- Formula: `CVD += V_buy - V_sell`


## Support/Resistance

### Pivot Points (Classic) (`pivot_classic`)
Support and resistance levels from the prior period's high, low and close.

- Inputs: H, L, C
- Default params: none
- Outputs: pp, r1, r2, r3, s1, s2, s3
- Formula: `PP=(H+L+C)/3; R1=2PP-L; S1=2PP-H; R2=PP+(H-L); S2=PP-(H-L)`

### Pivot Points (Fibonacci) (`pivot_fib`)
Pivot levels offset by Fibonacci ratios of the prior range.

- Inputs: H, L, C
- Default params: none
- Outputs: pp, r1, r2, r3, s1, s2, s3
- Formula: `R1=PP+0.382*(H-L); R2=PP+0.618*(H-L); R3=PP+1.0*(H-L); S mirrored`

### Pivot Points (Camarilla) (`pivot_camarilla`)
Tight intraday levels around the prior close.

- Inputs: H, L, C
- Default params: none
- Outputs: r1, r2, r3, r4, s1, s2, s3, s4
- Formula: `R_n = C + (H-L)*1.1/k; S_n = C - (H-L)*1.1/k, k=12,6,4,2`

### Pivot Points (Woodie) (`pivot_woodie`)
Pivot variant that weights the close more heavily.

- Inputs: H, L, C
- Default params: none
- Outputs: pp, r1, r2, s1, s2
- Formula: `PP=(H+L+2C)/4; R1=2PP-L; S1=2PP-H`

### Pivot Points (DeMark) (`pivot_demark`)
Pivot levels conditioned on the relationship between prior open and close.

- Inputs: O, H, L, C
- Default params: none
- Outputs: pp, r1, s1
- Formula: `X = H+2L+C if C<O; 2H+L+C if C>O; H+L+2C if equal; PP=X/4; R1=X/2-L; S1=X/2-H`

### Fibonacci Retracement (`fib_retrace`)
Horizontal levels at Fibonacci ratios between a swing high and swing low.

- Inputs: H, L
- Default params: levels=[0.236, 0.382, 0.5, 0.618, 0.786]
- Outputs: levels
- Formula: `level = High - ratio*(High-Low) for uptrend`

### Fibonacci Extension (`fib_extension`)
Projection targets beyond a swing using Fibonacci ratios.

- Inputs: H, L
- Default params: levels=[1.0, 1.272, 1.618, 2.0, 2.618]
- Outputs: levels
- Formula: `target = swing_end + ratio*(swing_range)`

### Gann Fan (`gann_fan`)
Angled trendlines from a pivot at fixed price/time ratios.

- Inputs: H, L
- Default params: angles=['1x1', '1x2', '2x1']
- Outputs: lines
- Formula: `price = pivot + slope*(bars since pivot), slope=ratio*scale`

### Swing Highs/Lows (Fractals) (`swing_hl`)
Bars whose high/low is the extreme of surrounding left/right bars (Williams Fractals when 2/2).

- Inputs: H, L
- Default params: left=2, right=2
- Outputs: swing_high, swing_low
- Formula: `H_t > max(H over left & right neighbours)`

### ZigZag (`zigzag`)
Connects swing points that exceed a percentage threshold; filters minor moves. Repaints by design.

- Inputs: H, L
- Default params: deviation_pct=5
- Outputs: zigzag
- Formula: `Pivot confirmed when price reverses by >= deviation from the last extreme`

### Linear Regression Channel (`regression_channel`)
OLS line over N bars with parallel bands at k standard errors.

- Inputs: C
- Default params: period=100, mult=2
- Outputs: upper, middle, lower
- Formula: `line = a + b*t; bands = line +/- k*stderr`

### Price Channel / Range Breakout Levels (`price_channel`)
Rolling highest high and lowest low used as breakout triggers.

- Inputs: H, L
- Default params: period=20
- Outputs: upper, lower
- Formula: `HH_N, LL_N`


## Price Transforms

### Typical Price (`typical_price`)
Average of high, low and close.

- Inputs: H, L, C
- Default params: none
- Outputs: tp
- Formula: `(H+L+C)/3`

### Median Price (`median_price`)
Midpoint of the bar's range.

- Inputs: H, L
- Default params: none
- Outputs: mp
- Formula: `(H+L)/2`

### Weighted Close (`weighted_close`)
Average with double weight on the close.

- Inputs: H, L, C
- Default params: none
- Outputs: wc
- Formula: `(H+L+2C)/4`

### OHLC Average (`ohlc4`)
Mean of all four prices.

- Inputs: O, H, L, C
- Default params: none
- Outputs: ohlc4
- Formula: `(O+H+L+C)/4`

### Heikin-Ashi Candles (`heikin_ashi`)
Smoothed candles that average price to make trends easier to read.

- Inputs: O, H, L, C
- Default params: none
- Outputs: ha_open, ha_high, ha_low, ha_close
- Formula: `HA_C=(O+H+L+C)/4; HA_O=(HA_O_prev+HA_C_prev)/2; HA_H=max(H,HA_O,HA_C); HA_L=min(L,HA_O,HA_C)`

### Renko Bricks (`renko`)
Time-independent bricks that form only when price moves by a fixed size.

- Inputs: C
- Default params: brick_size=ATR(14)
- Outputs: bricks
- Formula: `new brick when |C - last_brick_close| >= brick_size`

### Log Return (`log_return`)
Natural log of price ratio; additive across time.

- Inputs: C
- Default params: period=1
- Outputs: logret
- Formula: `ln(C_t/C_{t-N})`

### Percentage Return (`pct_return`)
Simple percentage change.

- Inputs: C
- Default params: period=1
- Outputs: ret
- Formula: `C_t/C_{t-N} - 1`

### Gap Size (`gap`)
Difference between today's open and prior close.

- Inputs: O, C
- Default params: none
- Outputs: gap, gap_pct
- Formula: `O_t - C_{t-1}`


## Statistical

### Linear Regression Slope (`linreg_slope`)
Slope of the OLS line fit over N bars; quantifies trend steepness.

- Inputs: P
- Default params: period=14
- Outputs: slope
- Formula: `b = cov(t,P)/var(t)`

### R-Squared (`linreg_r2`)
Goodness of fit of price to a linear trend (0-1).

- Inputs: P
- Default params: period=14
- Outputs: r2
- Formula: `corr(t,P)^2`

### Linear Regression Angle (`linreg_angle`)
Slope converted to degrees.

- Inputs: P
- Default params: period=14
- Outputs: angle
- Formula: `atan(slope)*180/pi`

### Z-Score (`zscore`)
Distance of price from its rolling mean in standard deviations; common in mean reversion.

- Inputs: P
- Default params: period=20
- Outputs: z
- Formula: `(P - SMA)/stdev`

### Rolling Correlation (`correlation`)
Pearson correlation of two series over N bars.

- Inputs: P, Q
- Default params: period=30
- Outputs: corr
- Formula: `cov(P,Q)/(sigma_P*sigma_Q)`

### Rolling Beta (`beta`)
Sensitivity of an asset's returns to a benchmark.

- Inputs: P, M
- Default params: period=60
- Outputs: beta
- Formula: `cov(r_asset, r_mkt)/var(r_mkt)`

### Rolling Skewness (`skew`)
Asymmetry of the return distribution.

- Inputs: C
- Default params: period=30
- Outputs: skew
- Formula: `E[((r-mu)/sigma)^3]`

### Rolling Kurtosis (`kurt`)
Tail heaviness of the return distribution.

- Inputs: C
- Default params: period=30
- Outputs: kurt
- Formula: `E[((r-mu)/sigma)^4]`

### Hurst Exponent (`hurst`)
Measures persistence (>0.5 trending), randomness (0.5) or mean reversion (<0.5).

- Inputs: C
- Default params: period=100
- Outputs: hurst
- Formula: `slope of log(R/S) vs log(n) (rescaled range)`

### Autocorrelation (`autocorr`)
Correlation of returns with lagged returns.

- Inputs: C
- Default params: lag=1, period=50
- Outputs: acf
- Formula: `corr(r_t, r_{t-lag})`

### Percentile Rank (`percent_rank`)
Share of values in the lookback window below the current value.

- Inputs: P
- Default params: period=100
- Outputs: prank
- Formula: `count(P_i < P_t)/N*100`

### Efficiency Ratio (`er`)
Net price change divided by total path length; 1 means a perfectly straight move.

- Inputs: C
- Default params: period=10
- Outputs: er
- Formula: `|C_t-C_{t-N}|/sum|C_i-C_{i-1}|`

### Fractal Dimension Index (`fractal_dim`)
Estimates market roughness; near 1.5 means random, lower means trending.

- Inputs: H, L
- Default params: period=30
- Outputs: fdi
- Formula: `1 + (ln(sum(dN)) + ln2)/ln(2N)`


## Cycle

### Hilbert Transform Dominant Cycle Period (`cycle_ht_dcperiod`)
Estimates the dominant cycle length using Ehlers' Hilbert Transform.

- Inputs: C
- Default params: none
- Outputs: period
- Formula: `Hilbert transform based phase differencing (see TA-Lib HT_DCPERIOD)`

### Hilbert Sine Wave (`ht_sine`)
Sine and lead-sine of the dominant cycle phase for turning point detection.

- Inputs: C
- Default params: none
- Outputs: sine, leadsine
- Formula: `sin(phase), sin(phase+45deg)`

### Hilbert Trend vs Cycle Mode (`ht_trendmode`)
Binary flag: trending (1) or cycling (0).

- Inputs: C
- Default params: none
- Outputs: mode
- Formula: `Based on dominant cycle phase rate and price vs trendline`

### Hilbert Instantaneous Trendline (`ht_trendline`)
Smoothed average over the dominant cycle period.

- Inputs: C
- Default params: none
- Outputs: trendline
- Formula: `SMA over DCPeriod, then WMA smoothing`

### MESA Adaptive Moving Average (MAMA/FAMA) (`mama`)
Ehlers adaptive MA whose alpha depends on the Hilbert phase rate of change.

- Inputs: C
- Default params: fast_limit=0.5, slow_limit=0.05
- Outputs: mama, fama
- Formula: `alpha = fast_limit/delta_phase, bounded; MAMA = alpha*P+(1-alpha)*MAMA_prev; FAMA = 0.5*alpha*MAMA+(1-0.5*alpha)*FAMA_prev`

### Cyber Cycle (`cyber_cycle`)
Ehlers cycle oscillator extracting the cyclic component of price.

- Inputs: P
- Default params: alpha=0.07
- Outputs: cycle, signal
- Formula: `Cycle_t = (1-a/2)^2*(S_t-2S_{t-1}+S_{t-2}) + 2(1-a)Cycle_{t-1} - (1-a)^2 Cycle_{t-2}`

### Ehlers Super Smoother (`super_smoother`)
Two-pole Butterworth filter with minimal lag and aliasing.

- Inputs: P
- Default params: period=10
- Outputs: ss
- Formula: `a1=exp(-1.414*pi/N); c2=2*a1*cos(1.414*pi/N); c3=-a1^2; c1=1-c2-c3; SS=c1*(P+P_prev)/2 + c2*SS1 + c3*SS2`

### Ehlers Instantaneous Trendline (`itrend`)
Low-lag trendline from a two-pole recursive filter.

- Inputs: P
- Default params: alpha=0.07
- Outputs: itrend
- Formula: `IT = (a-a^2/4)P + 0.5a^2 P1 - (a-0.75a^2)P2 + 2(1-a)IT1 - (1-a)^2 IT2`

### Center of Gravity Oscillator (`stoch_cg`)
Ehlers oscillator locating the 'balance point' of prices over a window.

- Inputs: P
- Default params: period=10
- Outputs: cg
- Formula: `-sum((i+1)*P_i)/sum(P_i)`


## Market Structure

### Market Profile (TPO) (`market_profile_tpo`)
Time-price-opportunity distribution of where price spent time in a session.

- Inputs: H, L
- Default params: tick_size=auto, period_minutes=30
- Outputs: tpo, poc, va
- Formula: `count 30-min letters per price bin; value area = ~70% of TPOs around POC`

### Opening Range (`opening_range`)
High and low of the first N minutes of the session; breakouts are common entries.

- Inputs: H, L
- Default params: minutes=30
- Outputs: orh, orl
- Formula: `max/min of H/L within first N minutes`

### Previous Period High/Low/Close (`prev_day_levels`)
Prior day/week/month key levels used as reference.

- Inputs: H, L, C
- Default params: timeframe=D
- Outputs: pdh, pdl, pdc
- Formula: `H,L,C of previous completed period`

### Gap Fill Status (`gap_fill`)
Tracks whether an opening gap has been traded back to the prior close.

- Inputs: O, H, L, C
- Default params: none
- Outputs: gap_open, filled
- Formula: `filled when price crosses prior close after gap`

### Order Blocks / Supply-Demand Zones (`order_block`)
Last opposing candle before an impulsive move; treated as institutional footprint zones (discretionary; rule definitions vary).

- Inputs: O, H, L, C
- Default params: lookback=10
- Outputs: zones
- Formula: `last down candle before N-bar up-impulse exceeding k*ATR`

### Fair Value Gap (`fvg`)
Three-candle imbalance where bar 1 and bar 3 ranges do not overlap.

- Inputs: H, L
- Default params: none
- Outputs: fvg_top, fvg_bottom
- Formula: `bullish: L_{t} > H_{t-2}; bearish: H_{t} < L_{t-2}`

### Break of Structure / Change of Character (`bos_choch`)
Detects close beyond prior swing high/low (BOS = continuation, CHoCH = reversal).

- Inputs: H, L, C
- Default params: swing=5
- Outputs: bos, choch
- Formula: `close > last swing high -> bullish BOS; first opposite break after trend -> CHoCH`

### Average Daily Range (`adr`)
Mean of daily high-low ranges; used for target and stop sizing.

- Inputs: H, L
- Default params: period=14
- Outputs: adr
- Formula: `SMA(H-L, N) on daily bars`

### Relative Volume (`rel_volume`)
Current volume divided by its recent average.

- Inputs: V
- Default params: period=20
- Outputs: rvol
- Formula: `V_t / SMA(V,N)`


## Relative/Breadth

### Relative Strength (Price Ratio) (`rs_ratio`)
Asset price divided by benchmark price; rising line means outperformance.

- Inputs: P, M
- Default params: none
- Outputs: rs
- Formula: `P_t/M_t`

### Relative Rotation Graph (RS-Ratio / RS-Momentum) (`rrg`)
Normalized relative strength and its momentum; plots rotation through leading, weakening, lagging, improving quadrants.

- Inputs: P, M
- Default params: period=10
- Outputs: rs_ratio, rs_momentum
- Formula: `RS-Ratio = 100 + zscore(smoothed RS); RS-Mom = 100 + zscore(ROC(RS-Ratio))`

### Relative Momentum Rank (`dual_momentum`)
Cross-sectional rank of N-bar returns across a universe; basis for rotation strategies.

- Inputs: C
- Default params: lookback=126
- Outputs: rank
- Formula: `rank(ROC_N) across symbols`

### Advance/Decline Line (`ad_breadth`)
Cumulative net advancing issues.

- Inputs: advancers, decliners
- Default params: none
- Outputs: adl_breadth
- Formula: `sum(advancers - decliners)`

### McClellan Oscillator (`mcclellan_osc`)
Difference between fast and slow EMAs of net advances.

- Inputs: advancers, decliners
- Default params: fast=19, slow=39
- Outputs: mco
- Formula: `EMA19(adv-dec) - EMA39(adv-dec)`

### McClellan Summation Index (`mcclellan_sum`)
Cumulative sum of the McClellan Oscillator.

- Inputs: advancers, decliners
- Default params: none
- Outputs: msi
- Formula: `MSI += MCO`

### Arms Index (TRIN) (`trin`)
Ratio of advance/decline count to advance/decline volume; short-term overbought/oversold.

- Inputs: adv, dec, adv_vol, dec_vol
- Default params: none
- Outputs: trin
- Formula: `(adv/dec)/(adv_vol/dec_vol)`

### Percent of Stocks Above Moving Average (`pct_above_ma`)
Share of index constituents trading above their N-day MA.

- Inputs: universe
- Default params: period=50
- Outputs: pct
- Formula: `count(C>SMA_N)/count(universe)*100`

### New Highs - New Lows (`new_hl`)
Number of 52-week highs minus lows.

- Inputs: universe
- Default params: period=252
- Outputs: nhnl
- Formula: `count(new 252d highs) - count(new 252d lows)`

### Put/Call Ratio (`put_call`)
Options sentiment gauge; extremes are contrarian signals.

- Inputs: put_vol, call_vol
- Default params: none
- Outputs: pcr
- Formula: `put volume / call volume`

### VIX Term Structure Ratio (`vix_term`)
Ratio of near-term to 3-month implied vol; >1 (backwardation) signals stress.

- Inputs: VIX, VIX3M
- Default params: none
- Outputs: ratio
- Formula: `VIX/VIX3M`


## Options

### IV Rank / IV Percentile (`iv_rank`)
Where current implied vol sits within its 1-year range (rank) or distribution (percentile).

- Inputs: IV
- Default params: period=252
- Outputs: ivr, ivp
- Formula: `IVR=(IV-minIV)/(maxIV-minIV)*100; IVP=% of days IV below current`

### IV minus HV Spread (`hv_iv_spread`)
Variance risk premium proxy.

- Inputs: IV, C
- Default params: hv_period=20
- Outputs: spread
- Formula: `IV - HV`

### Option Greeks (Delta, Gamma, Theta, Vega, Rho) (`greeks`)
Sensitivities of option price to underlying, convexity, time, volatility and rates.

- Inputs: S, K, T, r, sigma
- Default params: model=black-scholes
- Outputs: delta, gamma, theta, vega, rho
- Formula: `Partial derivatives of Black-Scholes price`

### Max Pain (`max_pain`)
Strike at which option holders collectively lose the most at expiry.

- Inputs: chain
- Default params: none
- Outputs: strike
- Formula: `argmin over K of sum(payout to all ITM calls and puts at K)`

### Put/Call Open Interest Ratio (`pcr_oi`)
Ratio of open interest in puts to calls, a positioning gauge.

- Inputs: put_oi, call_oi
- Default params: none
- Outputs: pcr_oi
- Formula: `put OI / call OI`


## Futures/Crypto

### Futures Basis / Premium (`basis`)
Difference between futures and spot price; reflects carry and sentiment.

- Inputs: F, S
- Default params: none
- Outputs: basis, basis_pct
- Formula: `F - S`

### Perpetual Funding Rate (`funding_rate`)
Periodic payment between longs and shorts on perpetual swaps; extremes signal crowding.

- Inputs: funding
- Default params: none
- Outputs: funding
- Formula: `exchange-provided`

### COT Net Positioning (`cot_net`)
Net long/short positions of a trader group from the CFTC Commitments of Traders report.

- Inputs: long, short
- Default params: none
- Outputs: net
- Formula: `long - short`


## Candlestick Patterns

### Doji (`candle_doji`)
Open and close nearly equal; indecision.

- Inputs: O, H, L, C
- Default params: body_pct=0.1
- Outputs: signal
- Formula: `|C-O| <= body_pct*(H-L)`

### Hammer / Hanging Man (`candle_hammer`)
Small body at top with long lower shadow (>=2x body); hammer after downtrend, hanging man after uptrend.

- Inputs: O, H, L, C
- Default params: none
- Outputs: signal
- Formula: `lower_shadow >= 2*body and upper_shadow small`

### Inverted Hammer / Shooting Star (`candle_inv_hammer`)
Small body at bottom with long upper shadow.

- Inputs: O, H, L, C
- Default params: none
- Outputs: signal
- Formula: `upper_shadow >= 2*body and lower_shadow small`

### Engulfing (Bullish/Bearish) (`candle_engulfing`)
Second candle's body fully covers the previous candle's body in the opposite direction.

- Inputs: O, H, L, C
- Default params: none
- Outputs: signal
- Formula: `bullish: prev red, curr green, O_c<=C_p, C_c>=O_p`

### Harami (`candle_harami`)
Small candle body contained within previous large body; potential reversal.

- Inputs: O, H, L, C
- Default params: none
- Outputs: signal
- Formula: `body_c inside body_p, opposite colour`

### Piercing Line / Dark Cloud Cover (`candle_piercing`)
Two-candle reversal closing beyond the midpoint of the previous body.

- Inputs: O, H, L, C
- Default params: none
- Outputs: signal
- Formula: `curr close > midpoint of prev body (bull) or < (bear)`

### Morning Star / Evening Star (`candle_star`)
Three-candle reversal: large body, small star body, large opposite body.

- Inputs: O, H, L, C
- Default params: none
- Outputs: signal
- Formula: `c1 large, c2 small gapped, c3 closes beyond c1 midpoint`

### Three White Soldiers / Three Black Crows (`candle_soldiers`)
Three consecutive strong candles in one direction, each opening within the previous body.

- Inputs: O, H, L, C
- Default params: none
- Outputs: signal
- Formula: `3 consecutive up (or down) candles, rising closes, small shadows`

### Marubozu (`candle_marubozu`)
Candle with almost no shadows; strong conviction.

- Inputs: O, H, L, C
- Default params: none
- Outputs: signal
- Formula: `shadows <= small fraction of range`

### Spinning Top (`candle_spinning`)
Small body with long upper and lower shadows; indecision.

- Inputs: O, H, L, C
- Default params: none
- Outputs: signal
- Formula: `body small, both shadows > body`

### Dragonfly / Gravestone Doji (`candle_dragonfly`)
Doji with a long lower (dragonfly) or upper (gravestone) shadow.

- Inputs: O, H, L, C
- Default params: none
- Outputs: signal
- Formula: `|C-O| tiny and one shadow dominates`

### Tweezer Top / Bottom (`candle_tweezer`)
Two consecutive candles with nearly identical highs (top) or lows (bottom).

- Inputs: O, H, L, C
- Default params: tol=0.001
- Outputs: signal
- Formula: `|H_t - H_{t-1}| <= tol*H`

### Inside Bar / Outside Bar (`candle_inside_outside`)
Bar fully within (inside) or fully engulfing (outside) the prior bar's range.

- Inputs: H, L
- Default params: none
- Outputs: inside, outside
- Formula: `inside: H<=Hp & L>=Lp; outside: H>Hp & L<Lp`

### Three Inside Up/Down (`candle_three_inside`)
Harami followed by a confirming third candle.

- Inputs: O, H, L, C
- Default params: none
- Outputs: signal
- Formula: `harami then close beyond first candle's body`

### Kicker (`candle_kicker`)
Gap reversal in which the second candle opens at or beyond the previous open in the opposite direction.

- Inputs: O, H, L, C
- Default params: none
- Outputs: signal
- Formula: `opposite colours with gap beyond previous open`


## Chart Patterns

### Double Top / Bottom (`chart_double_top`)
Two peaks (or troughs) at similar levels separated by a trough; reversal on neckline break.

- Inputs: H, L
- Default params: tolerance=0.02
- Outputs: signal
- Formula: `two swing highs within tolerance; breakout of intervening low`

### Head and Shoulders (and Inverse) (`chart_hs`)
Three peaks with the middle highest; reversal on neckline break.

- Inputs: H, L
- Default params: none
- Outputs: signal
- Formula: `swing sequence LS<H>RS with shoulders similar; neckline break`

### Triangles (Ascending/Descending/Symmetrical) (`chart_triangle`)
Converging trendlines forming consolidation, breaking in either direction.

- Inputs: H, L
- Default params: none
- Outputs: signal
- Formula: `fit lines to swing highs and lows; slopes converge`

### Flags and Pennants (`chart_flag`)
Short consolidation after a sharp move; continuation pattern.

- Inputs: H, L, C
- Default params: none
- Outputs: signal
- Formula: `impulse > k*ATR followed by tight counter-trend channel`

### Rising / Falling Wedge (`chart_wedge`)
Converging trendlines sloping the same direction; typically reversal.

- Inputs: H, L
- Default params: none
- Outputs: signal
- Formula: `both trendlines slope same way, converging`

### Cup and Handle (`chart_cup_handle`)
Rounded base followed by a small pullback; bullish continuation.

- Inputs: H, L, C
- Default params: none
- Outputs: signal
- Formula: `U-shaped base with ~12-33% depth and shallow handle`

### Rectangle / Range (`chart_rectangle`)
Horizontal support and resistance consolidation.

- Inputs: H, L
- Default params: none
- Outputs: signal
- Formula: `multiple touches at flat upper and lower bounds`

### N-Bar Breakout (`chart_breakout`)
Close above the prior N-bar high or below N-bar low.

- Inputs: H, L, C
- Default params: period=20
- Outputs: signal
- Formula: `C_t > max(H_{t-N..t-1})`

### Elliott Wave Count (`elliott`)
Subjective impulse (5) and corrective (3) wave labeling; hard to backtest objectively, usually approximated with ZigZag rules.

- Inputs: H, L
- Default params: none
- Outputs: waves
- Formula: `rule-based: wave 2 not below start, wave 3 not shortest, wave 4 no overlap with wave 1`


## Exhaustion

### TD Sequential (`td_sequential`)
Counts consecutive closes versus the close four bars earlier to flag trend exhaustion.

- Inputs: H, L, C
- Default params: setup=9, countdown=13
- Outputs: setup, countdown
- Formula: `setup: 9 consecutive C_t < C_{t-4} (buy) or > (sell); then countdown to 13`

### TD Setup (Simple) (`td_setup`)
Running count of closes beyond the close N bars prior.

- Inputs: C
- Default params: lookback=4
- Outputs: count
- Formula: `count += 1 if C_t < C_{t-4}`

### Consecutive Up/Down Closes (`consec_bars`)
Length of the current up or down streak.

- Inputs: C
- Default params: none
- Outputs: streak
- Formula: `streak += sign(C_t - C_{t-1}) with reset on flip`


## Mean Reversion

### Internal Bar Strength (`ibs`)
Where the close sits in the bar's range; low values often followed by short-term bounces.

- Inputs: H, L, C
- Default params: none
- Outputs: ibs
- Formula: `(C - L)/(H - L)`

### Percent From Moving Average (`pct_from_ma`)
Distance of price from its MA as a percent.

- Inputs: C
- Default params: period=50
- Outputs: pct
- Formula: `100*(C - SMA)/SMA`

### RSI(2) (`rsi2`)
Very short RSI used in Connors-style pullback strategies.

- Inputs: C
- Default params: period=2
- Outputs: rsi
- Formula: `RSI with N=2`

### Pairs Spread / Hedge Ratio (`pairs_spread`)
Residual of regressing one asset on another; z-score of residual drives stat-arb entries.

- Inputs: P, Q
- Default params: period=60
- Outputs: spread, z
- Formula: `spread = P - beta*Q; z = (spread - mean)/std`

### Engle-Granger Cointegration Test (`cointegration`)
ADF test on regression residuals to check whether a pair is mean-reverting.

- Inputs: P, Q
- Default params: period=250
- Outputs: pvalue
- Formula: `ADF(residual of OLS(P ~ Q))`

### Mean Reversion Half-Life (`half_life`)
Expected bars for a deviation to decay by half under an Ornstein-Uhlenbeck fit.

- Inputs: spread
- Default params: period=250
- Outputs: half_life
- Formula: `-ln(2)/lambda, from d(spread) = lambda*spread_{t-1} + e`

### Kalman Filter (Price / Hedge Ratio) (`kalman_filter`)
Recursive estimator of a hidden level or dynamic hedge ratio.

- Inputs: P
- Default params: q=1e-05, r=0.001
- Outputs: state
- Formula: `predict/update with process noise Q and measurement noise R`


## Risk/Performance

### Rolling Sharpe Ratio (`sharpe_roll`)
Excess return per unit of volatility over a rolling window.

- Inputs: C
- Default params: period=60, rf=0
- Outputs: sharpe
- Formula: `(mean(r)-rf)/stdev(r)*sqrt(252)`

### Rolling Sortino Ratio (`sortino_roll`)
Like Sharpe but only penalizes downside deviation.

- Inputs: C
- Default params: period=60, mar=0
- Outputs: sortino
- Formula: `(mean(r)-mar)/downside_dev*sqrt(252)`

### Drawdown (`drawdown`)
Percentage decline from the running peak.

- Inputs: equity
- Default params: none
- Outputs: dd, max_dd
- Formula: `(equity - cummax(equity))/cummax(equity)`

### Historical Value at Risk (`var_hist`)
Loss threshold not exceeded with the given confidence over a horizon, from the empirical return distribution.

- Inputs: C
- Default params: period=250, confidence=0.95
- Outputs: var
- Formula: `-percentile(r, 1-confidence)`

### Conditional VaR / Expected Shortfall (`cvar`)
Average loss beyond the VaR threshold.

- Inputs: C
- Default params: period=250, confidence=0.95
- Outputs: cvar
- Formula: `-mean(r | r <= -VaR)`

### Kelly Fraction (`kelly`)
Optimal bet fraction from win rate and payoff ratio (usually halved or quartered in practice).

- Inputs: trades
- Default params: none
- Outputs: kelly
- Formula: `f = W - (1-W)/R`

### Calmar Ratio (`calmar`)
Annualized return divided by max drawdown.

- Inputs: equity
- Default params: years=3
- Outputs: calmar
- Formula: `CAGR/|MaxDD|`

### Volatility-Target Position Size (`vol_target`)
Scales position so that portfolio volatility hits a target.

- Inputs: C
- Default params: target_vol=0.15, period=20
- Outputs: weight
- Formula: `weight = target_vol/realized_vol`

### ATR Position Sizing (`atr_position`)
Shares sized so that an ATR-multiple stop risks a fixed fraction of equity.

- Inputs: H, L, C
- Default params: risk_pct=0.01, atr_mult=2, period=14
- Outputs: shares
- Formula: `shares = equity*risk_pct/(atr_mult*ATR)`


## Calendar

### Day-of-Week / Month Seasonality (`seasonality_dow`)
Average return by weekday, month or turn-of-month window.

- Inputs: C
- Default params: none
- Outputs: avg_return_by_bucket
- Formula: `groupby(bucket).mean(return)`

### Session / Time-of-Day Filter (`time_filter`)
Boolean flag restricting signals to given sessions; essential for intraday backtests.

- Inputs: timestamp
- Default params: start=09:15, end=15:30, tz=Asia/Kolkata
- Outputs: in_session
- Formula: `start <= time <= end`

### Event Window Flag (earnings, expiry, macro) (`event_flag`)
Marks bars within a window around scheduled events.

- Inputs: timestamp, events
- Default params: window=1
- Outputs: flag
- Formula: `|t - event_time| <= window`


## Sentiment

### Sentiment / Fear-Greed Composite (`fear_greed`)
Composite of momentum, breadth, volatility, put/call and safe-haven demand (provider-specific).

- Inputs: multiple
- Default params: none
- Outputs: score
- Formula: `average of normalized components`

### Short Interest Ratio (Days to Cover) (`short_interest`)
Shares sold short divided by average daily volume.

- Inputs: short_int, V
- Default params: none
- Outputs: dtc
- Formula: `short_interest / avg_volume`


## Fundamental

### Price/Earnings Ratio (`pe_ratio`)
Price divided by trailing or forward earnings per share.

- Inputs: price, eps
- Default params: none
- Outputs: pe
- Formula: `P/EPS`

### Price/Book Ratio (`pb_ratio`)
Price divided by book value per share.

- Inputs: price, bvps
- Default params: none
- Outputs: pb
- Formula: `P/BVPS`

### EV/EBITDA (`ev_ebitda`)
Enterprise value divided by EBITDA; capital-structure-neutral valuation multiple.

- Inputs: ev, ebitda
- Default params: none
- Outputs: ev_ebitda
- Formula: `EV/EBITDA`

### Return on Equity (`roe`)
Net income divided by shareholders' equity.

- Inputs: net_income, equity
- Default params: none
- Outputs: roe
- Formula: `NI/Equity`

### Piotroski F-Score (`piotroski`)
0-9 score of profitability, leverage and efficiency tests.

- Inputs: financials
- Default params: none
- Outputs: fscore
- Formula: `sum of nine binary tests`

### Altman Z-Score (`altman_z`)
Bankruptcy-risk score from five financial ratios.

- Inputs: financials
- Default params: none
- Outputs: z
- Formula: `1.2*A+1.4*B+3.3*C+0.6*D+1.0*E`


## Factor

### 12-1 Momentum (Factor) (`momentum_12_1`)
Return over the past 12 months excluding the most recent month; classic cross-sectional momentum factor.

- Inputs: C
- Default params: lookback=252, skip=21
- Outputs: mom
- Formula: `C_{t-21}/C_{t-252} - 1`

### Low-Volatility Factor (`low_vol_factor`)
Realized volatility used to rank stocks; low-vol tends to deliver better risk-adjusted returns.

- Inputs: C
- Default params: period=252
- Outputs: vol
- Formula: `stdev of daily returns`

### Value Composite (`value_factor`)
Rank average of valuation ratios (e.g., earnings yield, book/price, sales/EV).

- Inputs: fundamentals
- Default params: none
- Outputs: score
- Formula: `mean(rank(E/P), rank(B/P), rank(S/EV))`

### Quality Composite (`quality_factor`)
Rank average of profitability and stability metrics (ROE, gross margin, low leverage).

- Inputs: fundamentals
- Default params: none
- Outputs: score
- Formula: `mean(rank(ROE), rank(GM), rank(-D/E))`
