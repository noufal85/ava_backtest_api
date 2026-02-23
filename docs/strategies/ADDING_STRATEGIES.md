# Adding Strategies

This guide covers all three ways to add a trading strategy to the platform.

## Method 1: AI Strategy Builder (Recommended)

Convert plain English, Pine Script, or chart screenshots into a registered strategy.

### From Text
```bash
python3 /path/to/build_strategy.py \
  --name vwap_bounce \
  --category mean_reversion \
  --input "Buy when price drops 2% below VWAP and RSI < 30. Sell when price returns to VWAP."
```

### From Pine Script
```bash
cat > /tmp/strategy.pine << 'PINE'
//@version=5
strategy("My Strategy", overlay=true)
ema8 = ta.ema(close, 8)
ema21 = ta.ema(close, 21)
if ta.crossover(ema8, ema21)
    strategy.entry("Long", strategy.long)
if ta.crossunder(ema8, ema21)
    strategy.close("Long")
PINE

python3 build_strategy.py --name ema_8_21 --category trend --input /tmp/strategy.pine
```

### From Screenshot
```bash
python3 build_strategy.py --name my_pattern --category pattern --input /path/to/chart.png
```

### With Immediate Backtest
```bash
python3 build_strategy.py \
  --name my_strategy --category trend --input "..." \
  --backtest --universe sp500_liquid --start 2023-01-01 --end 2024-06-01
```

### What Happens
1. Input type auto-detected (text / Pine / image)
2. Gemini Flash generates the Python code
3. Validated: syntax, @register, required methods, name match
4. Written to `src/strategies/custom/<name>.py`
5. Container restarted (no rebuild — src is volume-mounted)
6. Strategy verified in registry

## Method 2: Manual Creation

Create a file in `src/strategies/custom/`:

```python
"""My Custom Strategy — brief description."""
import polars as pl
from src.core.strategy.base import BaseStrategy, Signal
from src.core.strategy.registry import register

@register
class MyCustomStrategy(BaseStrategy):
    name = "my_custom_strategy"      # Must be snake_case, unique
    version = "1.0.0"
    description = "What this strategy does"
    category = "trend"               # trend | mean_reversion | momentum | multi_factor | volatility
    tags = ["trend_following", "ema"]

    def __init__(self, fast_period: int = 12, slow_period: int = 26):
        self.fast_period = fast_period
        self.slow_period = slow_period

    def get_warmup_periods(self) -> int:
        """Bars needed before first signal (longest indicator period + buffer)."""
        return self.slow_period + 5

    def get_parameter_schema(self) -> dict:
        """JSON Schema for each __init__ parameter."""
        return {
            "fast_period": {
                "type": "integer",
                "default": 12,
                "minimum": 5,
                "maximum": 50,
                "description": "Fast EMA period"
            },
            "slow_period": {
                "type": "integer",
                "default": 26,
                "minimum": 10,
                "maximum": 100,
                "description": "Slow EMA period"
            },
        }

    def generate_signal(self, window) -> Signal | None:
        """Process one bar. Return Signal or None."""
        hist = window.historical()
        cur = window.current_bar()

        if len(hist) < self.slow_period:
            return None

        combined = pl.concat([hist, cur])
        closes = combined["close"].to_list()

        # Calculate EMAs
        def ema(data, period):
            mult = 2 / (period + 1)
            result = [data[0]]
            for price in data[1:]:
                result.append(price * mult + result[-1] * (1 - mult))
            return result

        fast = ema(closes, self.fast_period)
        slow = ema(closes, self.slow_period)

        # Crossover
        if fast[-2] <= slow[-2] and fast[-1] > slow[-1]:
            return Signal("buy", 1.0, 0.8, {"fast": fast[-1], "slow": slow[-1]})
        if fast[-2] >= slow[-2] and fast[-1] < slow[-1]:
            return Signal("sell", 1.0, 0.8, {"fast": fast[-1], "slow": slow[-1]})

        return None
```

Then restart:
```bash
docker compose restart backtester-v2
```

## Method 3: Batch Migration

Convert strategies from the old `trading-backtester` repo:

```bash
# Clone source repo
git clone https://github.com/noufal85/trading-backtester.git /tmp/trading-backtester

# Dry run
python3 convert_strategies.py --all --dry-run

# Convert top performers
python3 convert_strategies.py --top-performers

# Convert everything
python3 convert_strategies.py --all

# Restart
docker compose restart backtester-v2
```

## Rules

1. **Use `@register`** — every strategy must have this decorator
2. **Use polars only** — no pandas, no ta-lib
3. **Implement indicators from scratch** — simple math with close/high/low/volume lists
4. **No look-ahead** — only `window.historical()` + `window.current_bar()`
5. **All params need defaults** — `__init__(self, period: int = 20)`
6. **`get_parameter_schema()`** — proper JSON Schema for every parameter
7. **`get_warmup_periods()`** — return longest indicator period + small buffer
8. **Category must be one of**: `trend`, `mean_reversion`, `momentum`, `multi_factor`, `volatility`

## Window API Reference

```python
def generate_signal(self, window) -> Signal | None:
    hist = window.historical()     # polars DataFrame (all bars before current)
    cur = window.current_bar()     # polars DataFrame (1 row — current bar)

    # Columns: open, high, low, close, volume (float64), timestamp (date)

    # Combine for indicator calculation
    combined = pl.concat([hist, cur])
    closes = combined["close"].to_list()
    highs = combined["high"].to_list()
    lows = combined["low"].to_list()
    volumes = combined["volume"].to_list()

    # Return Signal or None
    return Signal(
        action="buy",        # "buy" | "sell" | "hold"
        strength=1.0,        # 0.0 – 1.0
        confidence=0.8,      # 0.0 – 1.0
        metadata={"info": "any debug data"}
    )
```

## Common Indicator Implementations

### SMA
```python
def sma(data, period):
    return sum(data[-period:]) / period
```

### EMA
```python
def ema(data, period):
    mult = 2 / (period + 1)
    result = [data[0]]
    for price in data[1:]:
        result.append(price * mult + result[-1] * (1 - mult))
    return result
```

### RSI
```python
def rsi(closes, period=14):
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i-1]
        gains.append(max(0, diff))
        losses.append(max(0, -diff))
    if len(gains) < period:
        return 50.0
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))
```

### ATR
```python
def atr(highs, lows, closes, period=14):
    trs = []
    for i in range(1, len(highs)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        trs.append(tr)
    if len(trs) < period:
        return sum(trs) / max(len(trs), 1)
    return sum(trs[-period:]) / period
```

### Bollinger Bands
```python
def bollinger(closes, period=20, std_dev=2.0):
    mean = sum(closes[-period:]) / period
    variance = sum((c - mean) ** 2 for c in closes[-period:]) / period
    std = variance ** 0.5
    return mean, mean + std_dev * std, mean - std_dev * std  # middle, upper, lower
```

## File Structure

```
src/strategies/
├── __init__.py              # Imports classic + custom
├── classic/                 # 9 built-in (don't edit)
│   ├── sma_crossover.py
│   ├── rsi_mean_reversion.py
│   └── ...
└── custom/                  # Auto-discovered
    ├── __init__.py           # pkgutil auto-import
    ├── connors_rsi.py
    ├── tv_ichimoku_cloud.py
    └── ... (115 strategies)
```

## Lifecycle

| Action | Command | Rebuild? |
|--------|---------|----------|
| Add strategy | Drop `.py` in `custom/` | No — restart only |
| Remove strategy | Delete `.py` | No — restart only |
| Update strategy | Edit `.py` | No — restart only |
| List all | `curl localhost:8201/api/v2/strategies` | — |
| View detail | `curl localhost:8201/api/v2/strategies/{name}` | — |
| Run backtest | `POST /api/v2/backtests` | — |
