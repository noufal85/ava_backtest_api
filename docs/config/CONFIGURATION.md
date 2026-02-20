# Configuration Reference

Complete reference for V2 strategy YAML configuration.

---

## Top-Level Structure

```yaml
meta:         # Strategy identity
runtime:      # Execution settings
components:   # Indicators, signals, sizing, risk, regime
execution:    # Backtest parameters
optimization: # (optional) Parameter search
output:       # (optional) What to save
```

---

## `meta` — Strategy Identity

| Field         | Type     | Required | Description |
|---------------|----------|----------|-------------|
| `strategy`    | string   | ✅       | Unique strategy name (snake_case) |
| `version`     | string   | ✅       | Semantic version, e.g. `"2.1.0"` |
| `description` | string   |          | Human-readable summary |
| `tags`        | string[] |          | Searchable labels |

```yaml
meta:
  strategy: bb_mean_reversion
  version: "2.1.0"
  description: "Bollinger Band mean reversion with RSI filter"
  tags: ["mean_reversion", "bollinger", "intermediate"]
```

---

## `runtime` — Execution Settings

| Field                | Type     | Default        | Description |
|----------------------|----------|----------------|-------------|
| `primary_timeframe`  | string   | `"1d"`         | Main bar timeframe (`1m`,`5m`,`15m`,`1h`,`1d`) |
| `required_timeframes`| string[] | `[]`           | Additional timeframes loaded & aligned |
| `warmup_periods`     | int      | `50`           | Bars before first signal |
| `execution_mode`     | string   | `"event_driven"` | `"vectorized"`, `"event_driven"`, `"hybrid"` |

```yaml
runtime:
  primary_timeframe: "15m"
  required_timeframes: ["1d", "1h"]
  warmup_periods: 100
  execution_mode: "event_driven"
```

---

## `components.indicators` — Indicator Definitions

Each key is a user-chosen name; value specifies type, optional timeframe, and params.

| Field       | Type   | Required | Description |
|-------------|--------|----------|-------------|
| `type`      | string | ✅       | Indicator type from INDICATOR_CATALOG |
| `timeframe` | string |          | Override timeframe (default: primary) |
| `params`    | object | ✅       | Indicator-specific parameters |

```yaml
components:
  indicators:
    trend_ema:
      type: exponential_ma
      timeframe: "1d"
      params: {period: 50}
    bb:
      type: bollinger_bands
      params: {period: 20, std_dev: 2.0}
    rsi:
      type: rsi
      params: {period: 14}
```

---

## `components.signals` — Entry/Exit Conditions

| Field        | Type     | Description |
|--------------|----------|-------------|
| `conditions` | object[] | List of condition checks |
| `logic`      | string   | `"all"` (AND), `"any"` (OR), `"weighted"` |

**Condition types:**

| Condition        | Params                        | Description |
|------------------|-------------------------------|-------------|
| `crossover`      | `fast`, `slow`                | fast crosses above slow |
| `crossunder`     | `fast`, `slow`                | fast crosses below slow |
| `price_above`    | `ref`                         | close > indicator value |
| `price_below`    | `ref`                         | close < indicator value |
| `rsi_below`      | `ref`, `threshold`            | RSI < threshold |
| `rsi_above`      | `ref`, `threshold`            | RSI > threshold |
| `bb_lower_touch` | `indicator`                   | low ≤ BB lower |
| `bb_upper_touch` | `indicator`                   | high ≥ BB upper |
| `trend_bullish`  | `indicator`, `method`         | price above MA |
| `trend_bearish`  | `indicator`, `method`         | price below MA |

```yaml
components:
  signals:
    entry_long:
      conditions:
        - crossover: {fast: fast_ema, slow: slow_ema}
        - rsi_below: {ref: rsi, threshold: 70}
      logic: all
    exit_long:
      conditions:
        - crossunder: {fast: fast_ema, slow: slow_ema}
      logic: any
```

---

## `components.sizing` — Position Sizing

| Field  | Type   | Required | Description |
|--------|--------|----------|-------------|
| `type` | string | ✅       | Sizing algorithm |
| `params` | object | ✅     | Algorithm-specific |

**Sizing types:**

| Type               | Params                                    |
|--------------------|-------------------------------------------|
| `fixed_pct`        | `pct` (fraction of equity)                |
| `fixed_dollar`     | `dollars`                                 |
| `volatility_scaled`| `target_vol`, `lookback`                  |
| `kelly_optimal`    | `lookback_trades`, `max_kelly_pct`        |
| `risk_parity`      | `target_risk`, `lookback`                 |

```yaml
components:
  sizing:
    type: kelly_optimal
    params:
      lookback_trades: 50
      max_kelly_pct: 0.20
      min_position_dollars: 1000
      max_position_pct: 0.05
```

---

## `components.risk` — Risk Management Rules

Array of rules evaluated in order. Each can pass, modify (reduce size), or block a trade.

| Field    | Type   | Required |
|----------|--------|----------|
| `type`   | string | ✅       |
| `params` | object | ✅       |

**Rule types:**

| Type                | Params                                      |
|---------------------|---------------------------------------------|
| `stop_loss`         | `pct`                                       |
| `trailing_stop`     | `pct`, `activation_pct`                     |
| `take_profit`       | `pct`                                       |
| `max_position`      | `pct` (max single position % of equity)     |
| `max_exposure`      | `total_pct`, `per_sector_pct`               |
| `max_drawdown`      | `pct` (halt trading if drawdown exceeds)    |
| `correlation_limit` | `max_correlation`, `lookback_days`          |
| `time_stop`         | `max_hold_days`                             |

```yaml
components:
  risk:
    rules:
      - type: trailing_stop
        params: {pct: 0.02, activation_pct: 0.01}
      - type: max_exposure
        params: {total_pct: 0.80, per_sector_pct: 0.30}
      - type: max_drawdown
        params: {pct: 0.15}
```

---

## `components.regime` — Market Regime Filter

| Field                  | Type     | Default |
|------------------------|----------|---------|
| `enabled`              | bool     | `false` |
| `allowed_regimes`      | string[] | `["bull","neutral"]` |
| `block_regimes`        | string[] | `[]`    |
| `confidence_threshold` | float    | `0.8`   |

```yaml
components:
  regime:
    enabled: true
    allowed_regimes: ["bull", "neutral"]
    confidence_threshold: 0.8
```

---

## `execution` — Backtest Parameters

| Field              | Type   | Default                |
|--------------------|--------|------------------------|
| `initial_capital`  | float  | `100000`               |
| `universe`         | string |                        |
| `symbols`          | list   | (overrides universe)   |
| `start_date`       | date   |                        |
| `end_date`         | date   |                        |
| `fill_model`       | string | `"realistic_volume"`   |
| `commission_model` | string | `"interactive_brokers"` |
| `slippage_model`   | string | `"sqrt_volume"`        |
| `max_workers`      | int    | `8`                    |

```yaml
execution:
  initial_capital: 100000
  universe: sp500_liquid
  start_date: "2020-01-01"
  end_date: "2024-12-31"
  fill_model: realistic_volume
  commission_model: interactive_brokers
  slippage_model: sqrt_volume
```

---

## `optimization` — Parameter Search (optional)

| Field    | Type   | Default         |
|----------|--------|-----------------|
| `enabled`| bool   | `false`         |
| `method` | string | `"bayesian"`    |
| `objective`| string | `"sharpe_ratio"` |
| `trials` | int    | `100`           |
| `cv_folds`| int   | `5`             |

```yaml
optimization:
  enabled: true
  method: bayesian
  objective: sharpe_ratio
  trials: 200
  cv_folds: 5
```

---

## `output` — What to Save (optional)

| Field              | Type     | Default  |
|--------------------|----------|----------|
| `save_trades`      | bool     | `true`   |
| `save_signals`     | bool     | `false`  |
| `save_equity_curve`| bool     | `true`   |
| `save_attribution` | bool     | `true`   |
| `export_format`    | string[] | `["json"]` |

---

## Complete Annotated Example

```yaml
meta:
  strategy: bb_mean_reversion
  version: "2.1.0"
  description: "BB mean reversion with RSI confirmation and regime filtering"
  tags: ["mean_reversion", "bollinger_bands"]

runtime:
  primary_timeframe: "15m"
  required_timeframes: ["1d"]     # Daily for regime + trend
  warmup_periods: 50
  execution_mode: "event_driven"

components:
  indicators:
    daily_ema:
      type: exponential_ma
      timeframe: "1d"
      params: {period: 50}
    bb:
      type: bollinger_bands
      params: {period: 20, std_dev: 2.0}
    rsi:
      type: rsi
      params: {period: 14}

  signals:
    entry_long:
      conditions:
        - trend_bullish: {indicator: daily_ema, method: price_above}
        - bb_lower_touch: {indicator: bb}
        - rsi_below: {ref: rsi, threshold: 35}
      logic: all
    exit_long:
      conditions:
        - price_above: {ref: bb.middle}
        - rsi_above: {ref: rsi, threshold: 70}
      logic: any

  sizing:
    type: volatility_scaled
    params: {target_vol: 0.02, lookback: 20}

  risk:
    rules:
      - type: trailing_stop
        params: {pct: 0.02, activation_pct: 0.01}
      - type: max_exposure
        params: {total_pct: 0.80, per_sector_pct: 0.30}
      - type: max_drawdown
        params: {pct: 0.15}

  regime:
    enabled: true
    allowed_regimes: ["bull", "neutral"]
    confidence_threshold: 0.8

execution:
  initial_capital: 100000
  universe: sp500_liquid
  start_date: "2020-01-01"
  end_date: "2024-12-31"
  fill_model: realistic_volume
  commission_model: interactive_brokers

optimization:
  enabled: false

output:
  save_trades: true
  save_signals: false
  save_equity_curve: true
  export_format: ["json", "csv"]
```
