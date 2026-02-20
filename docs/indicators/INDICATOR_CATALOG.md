# Indicator Catalog

Complete reference for every indicator in the V2 library.

---

## Trend Indicators

### SMA — Simple Moving Average

**Formula:** `SMA(n) = Σ close[i] / n` for i in [0, n)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `period`  | int  | 20      | Lookback window |

```python
from v2.core.indicators.trend import SMA
sma = SMA(period=50)
# Returns pl.Series of rolling mean
```

### EMA — Exponential Moving Average

**Formula:** `EMA(t) = α × close(t) + (1 − α) × EMA(t−1)`, where `α = 2 / (period + 1)`

| Parameter | Type | Default |
|-----------|------|---------|
| `period`  | int  | 20      |

```python
ema = EMA(period=9)
```

### DEMA — Double Exponential Moving Average

**Formula:** `DEMA = 2 × EMA(n) − EMA(EMA(n))`

| Parameter | Type | Default |
|-----------|------|---------|
| `period`  | int  | 20      |

```python
dema = DEMA(period=21)
```

### TEMA — Triple Exponential Moving Average

**Formula:** `TEMA = 3×EMA − 3×EMA(EMA) + EMA(EMA(EMA))`

| Parameter | Type | Default |
|-----------|------|---------|
| `period`  | int  | 20      |

```python
tema = TEMA(period=21)
```

### WMA — Weighted Moving Average

**Formula:** `WMA = Σ (weight_i × close_i) / Σ weight_i` where `weight_i = n − i`

| Parameter | Type | Default |
|-----------|------|---------|
| `period`  | int  | 20      |

```python
wma = WMA(period=10)
```

### VWAP — Volume-Weighted Average Price

**Formula:** `VWAP = Σ (price × volume) / Σ volume` (reset each session)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `anchor`  | str  | "session" | Reset period: "session", "week", "month" |

```python
vwap = VWAP(anchor="session")
```

### MACD — Moving Average Convergence Divergence

**Formula:**
- `MACD line = EMA(fast) − EMA(slow)`
- `Signal line = EMA(MACD line, signal_period)`
- `Histogram = MACD line − Signal line`

| Parameter | Type | Default |
|-----------|------|---------|
| `fast`    | int  | 12      |
| `slow`    | int  | 26      |
| `signal`  | int  | 9       |

**Outputs:** `macd_line`, `macd_signal`, `macd_histogram`

```python
macd = MACD(fast=12, slow=26, signal=9)
```

---

## Momentum Indicators

### RSI — Relative Strength Index

**Formula:**
1. `avg_gain = rolling_mean(max(Δclose, 0), n)`
2. `avg_loss = rolling_mean(max(-Δclose, 0), n)`
3. `RS = avg_gain / avg_loss`
4. `RSI = 100 − 100 / (1 + RS)`

| Parameter | Type | Default |
|-----------|------|---------|
| `period`  | int  | 14      |

```python
rsi = RSI(period=14)
# Typical levels: <30 oversold, >70 overbought
```

### Stochastic Oscillator

**Formula:**
- `%K = (close − lowest_low(n)) / (highest_high(n) − lowest_low(n)) × 100`
- `%D = SMA(%K, d_period)`

| Parameter  | Type | Default |
|------------|------|---------|
| `k_period` | int  | 14      |
| `d_period` | int  | 3       |
| `smooth`   | int  | 3       |

**Outputs:** `stoch_k`, `stoch_d`

```python
stoch = Stochastic(k_period=14, d_period=3)
```

### ROC — Rate of Change

**Formula:** `ROC = (close − close[n]) / close[n] × 100`

| Parameter | Type | Default |
|-----------|------|---------|
| `period`  | int  | 12      |

```python
roc = ROC(period=12)
```

### MFI — Money Flow Index

**Formula:**
1. `typical_price = (H + L + C) / 3`
2. `money_flow = typical_price × volume`
3. `MFI = 100 − 100 / (1 + positive_flow / negative_flow)`

| Parameter | Type | Default |
|-----------|------|---------|
| `period`  | int  | 14      |

```python
mfi = MFI(period=14)
```

### Williams %R

**Formula:** `%R = (highest_high − close) / (highest_high − lowest_low) × −100`

| Parameter | Type | Default |
|-----------|------|---------|
| `period`  | int  | 14      |

```python
willr = WilliamsR(period=14)
# Range: -100 to 0. < -80 oversold, > -20 overbought
```

### CCI — Commodity Channel Index

**Formula:**
1. `tp = (H + L + C) / 3`
2. `CCI = (tp − SMA(tp, n)) / (0.015 × mean_deviation)`

| Parameter | Type | Default |
|-----------|------|---------|
| `period`  | int  | 20      |

```python
cci = CCI(period=20)
```

### RMI — Relative Momentum Index

**Formula:** Like RSI but uses `close − close[momentum_period]` instead of `close − close[1]`.

| Parameter         | Type | Default |
|-------------------|------|---------|
| `period`          | int  | 14      |
| `momentum_period` | int  | 4       |

```python
rmi = RMI(period=14, momentum_period=4)
```

---

## Volatility Indicators

### ATR — Average True Range

**Formula:**
1. `TR = max(H−L, |H−prev_close|, |L−prev_close|)`
2. `ATR = EMA(TR, period)` (Wilder smoothing)

| Parameter | Type | Default |
|-----------|------|---------|
| `period`  | int  | 14      |

```python
atr = ATR(period=14)
```

### Bollinger Bands

**Formula:**
- `Middle = SMA(close, period)`
- `Upper = Middle + std_dev × σ`
- `Lower = Middle − std_dev × σ`

| Parameter | Type  | Default |
|-----------|-------|---------|
| `period`  | int   | 20      |
| `std_dev` | float | 2.0     |

**Outputs:** `bb_upper`, `bb_middle`, `bb_lower`, `bb_width`, `bb_pct_b`

```python
bb = BollingerBands(period=20, std_dev=2.0)
```

### Keltner Channels

**Formula:**
- `Middle = EMA(close, period)`
- `Upper = Middle + multiplier × ATR(atr_period)`
- `Lower = Middle − multiplier × ATR(atr_period)`

| Parameter    | Type  | Default |
|-------------|-------|---------|
| `period`     | int   | 20      |
| `multiplier` | float | 1.5     |
| `atr_period` | int   | 10      |

**Outputs:** `kc_upper`, `kc_middle`, `kc_lower`

```python
kc = KeltnerChannels(period=20, multiplier=1.5)
```

### Donchian Channels

**Formula:**
- `Upper = highest_high(period)`
- `Lower = lowest_low(period)`
- `Middle = (Upper + Lower) / 2`

| Parameter | Type | Default |
|-----------|------|---------|
| `period`  | int  | 20      |

**Outputs:** `dc_upper`, `dc_middle`, `dc_lower`

```python
dc = DonchianChannels(period=20)
```

---

## Volume Indicators

### OBV — On-Balance Volume

**Formula:** `OBV(t) = OBV(t−1) + sign(Δclose) × volume`

No parameters.

```python
obv = OBV()
```

### VWAP

See Trend section above.

### AD Line — Accumulation/Distribution Line

**Formula:**
1. `CLV = ((C−L) − (H−C)) / (H−L)`
2. `AD = cumsum(CLV × volume)`

No parameters.

```python
ad = ADLine()
```

### CMF — Chaikin Money Flow

**Formula:** `CMF = SMA(AD, period) / SMA(volume, period)`

| Parameter | Type | Default |
|-----------|------|---------|
| `period`  | int  | 20      |

```python
cmf = CMF(period=20)
# Positive → buying pressure; negative → selling pressure
```
