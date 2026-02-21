# ava_backtest — Multi-Market Architecture
## Exact Changes to trading-backtester_v2 + trading-backtester-ui_v2

**Date:** 2026-02-21  
**Supersedes:** MULTI_MARKET_DESIGN.md (was written before reading the actual repos)

This document describes precise, surgical changes to the existing v2 repos. Every file path, type name, and SQL statement references actual code that already exists.

---

## 1. Market Registry — new file

**File:** `src/core/markets/registry.py` *(new)*

```python
"""
Market Registry — single source of truth for all market metadata.
Adding a new market = add one MarketConfig entry here. Zero other changes.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from zoneinfo import ZoneInfo


class MarketCode(str, Enum):
    US = "US"   # NYSE / NASDAQ / BATS
    IN = "IN"   # NSE / BSE
    # Future: GB, JP, SG, AU — add here only


@dataclass(frozen=True)
class MarketConfig:
    code:                  MarketCode
    name:                  str
    exchange:              str
    currency:              str            # ISO 4217: "USD", "INR"
    currency_symbol:       str            # "$", "₹"
    timezone:              ZoneInfo
    session_open:          str            # "09:30" local time
    session_close:         str            # "16:00" local time
    settlement_days:       int            # T+ lag
    lot_size:              int            # minimum tradeable shares
    tick_size:             float          # minimum price increment
    default_universe:      str            # shown first in UniverseSelector
    data_providers:        list[str]      # preferred order; router tries each
    commission_model:      str            # maps to existing fill_model keys
    flags:                 dict = field(default_factory=dict)


MARKET_REGISTRY: dict[MarketCode, MarketConfig] = {

    MarketCode.US: MarketConfig(
        code=MarketCode.US,
        name="United States",
        exchange="NYSE/NASDAQ",
        currency="USD",
        currency_symbol="$",
        timezone=ZoneInfo("America/New_York"),
        session_open="09:30",
        session_close="16:00",
        settlement_days=2,
        lot_size=1,
        tick_size=0.01,
        default_universe="sp500_liquid",
        data_providers=["fmp", "alpaca"],
        commission_model="interactive_brokers",
    ),

    MarketCode.IN: MarketConfig(
        code=MarketCode.IN,
        name="India",
        exchange="NSE/BSE",
        currency="INR",
        currency_symbol="₹",
        timezone=ZoneInfo("Asia/Kolkata"),
        session_open="09:15",
        session_close="15:30",
        settlement_days=1,    # T+1 since Jan 2023
        lot_size=1,
        tick_size=0.05,
        default_universe="nifty50",
        data_providers=["nsepy", "upstox"],
        commission_model="zerodha_flat",   # ₹20 flat or 0.03% whichever lower
        flags={
            "stt_buy_pct":   0.001,   # 0.1% STT on delivery buy
            "stt_sell_pct":  0.001,   # 0.1% STT on delivery sell
            "gst_pct":       0.18,    # GST on brokerage
            "stamp_duty_pct": 0.00015, # 0.015% on buy value
            "has_fo_segment": True,
        },
    ),
}


def get_market(code: str) -> MarketConfig:
    try:
        return MARKET_REGISTRY[MarketCode(code.upper())]
    except (ValueError, KeyError):
        valid = [m.value for m in MARKET_REGISTRY]
        raise ValueError(f"Unknown market '{code}'. Valid: {valid}")


def list_markets() -> list[dict]:
    """Serialisable list for /api/v2/markets endpoint."""
    return [
        {
            "code":             m.code.value,
            "name":             m.name,
            "exchange":         m.exchange,
            "currency":         m.currency,
            "currency_symbol":  m.currency_symbol,
            "default_universe": m.default_universe,
            "is_default":       m.code == MarketCode.US,
        }
        for m in MARKET_REGISTRY.values()
    ]
```

**File:** `src/core/markets/__init__.py` *(new, empty)*

---

## 2. Provider Abstraction — fits into existing `src/core/data/providers/`

The existing `src/core/data/providers/` directory already exists. Add these files:

**File:** `src/core/data/providers/base.py` *(new)*

```python
"""Abstract DataProvider — all market data sources implement this."""
from abc import ABC, abstractmethod
from datetime import date
import polars as pl
from src.core.markets.registry import MarketCode


class DataProvider(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique key: 'fmp', 'nsepy', 'upstox', 'alpaca'"""
        ...

    @property
    @abstractmethod
    def supported_markets(self) -> list[MarketCode]:
        ...

    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        market: MarketCode,
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> pl.DataFrame:
        """
        Returns DataFrame with columns:
          ts (TIMESTAMPTZ-aware), open, high, low, close, volume, adj_close
        Sorted ascending. No gaps filled.
        """
        ...

    @abstractmethod
    async def search_symbols(self, query: str, market: MarketCode) -> list[dict]:
        """Returns [{symbol, name, exchange, sector}]"""
        ...

    @abstractmethod
    async def get_universe_symbols(self, universe_name: str, market: MarketCode) -> list[str]:
        ...
```

**File:** `src/core/data/providers/fmp.py` *(modify existing — add `market` param)*

Change signature:
```python
# Before (existing)
async def fetch_ohlcv(self, symbol: str, start: date, end: date, timeframe="1d")

# After
async def fetch_ohlcv(self, symbol: str, market: MarketCode, start: date, end: date, timeframe="1d")
# market is always MarketCode.US for FMP; raise ValueError if called for others
```

**File:** `src/core/data/providers/nsepy.py` *(new — India)*

```python
"""NSEpy provider — free NSE/BSE daily data."""
import asyncio
from datetime import date
import polars as pl
from src.core.data.providers.base import DataProvider
from src.core.markets.registry import MarketCode

NIFTY_UNIVERSES = {
    "nifty50":        ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
                       "HINDUNILVR.NS","SBIN.NS","BAJFINANCE.NS","BHARTIARTL.NS","KOTAKBANK.NS",
                       # ... full list of 50
                       ],
    "nifty100":       [...],  # 100 symbols
    "nifty500":       [...],  # 500 symbols
    "nse_large_cap":  [...],
    "nse_mid_cap":    [...],
    "nse_small_cap":  [...],
}

class NSEpyProvider(DataProvider):
    name = "nsepy"
    supported_markets = [MarketCode.IN]

    async def fetch_ohlcv(self, symbol, market, start, end, timeframe="1d"):
        if timeframe != "1d":
            raise ValueError("NSEpy only supports daily ('1d'). Use Upstox for intraday.")
        # Run nsepy sync call in thread pool
        loop = asyncio.get_event_loop()
        df_pd = await loop.run_in_executor(None, self._fetch_sync, symbol, start, end)
        return pl.from_pandas(df_pd).rename({"Date": "ts", "Open": "open",
                                              "High": "high", "Low": "low",
                                              "Close": "close", "Volume": "volume"})

    def _fetch_sync(self, symbol: str, start: date, end: date):
        from nsepy import get_history
        clean_sym = symbol.replace(".NS", "").replace(".BO", "")
        return get_history(symbol=clean_sym, start=start, end=end)

    async def search_symbols(self, query, market):
        # NSE symbol list is static; filter in-memory
        all_syms = NIFTY_UNIVERSES["nifty500"]
        return [{"symbol": s, "name": s, "exchange": "NSE"} 
                for s in all_syms if query.upper() in s]

    async def get_universe_symbols(self, universe_name, market):
        if universe_name not in NIFTY_UNIVERSES:
            raise ValueError(f"Unknown India universe '{universe_name}'")
        return NIFTY_UNIVERSES[universe_name]
```

**File:** `src/core/data/providers/router.py` *(new)*

```python
"""ProviderRouter — selects and fails-over between DataProviders for a market."""
from datetime import date
import polars as pl
from src.core.markets.registry import MarketCode, MARKET_REGISTRY
from src.core.data.providers.base import DataProvider


class ProviderRouter:

    def __init__(self, providers: list[DataProvider]):
        # Index: (market, provider_name) → provider
        self._index: dict[tuple[MarketCode, str], DataProvider] = {
            (market, p.name): p
            for p in providers
            for market in p.supported_markets
        }

    async def fetch_ohlcv(
        self,
        symbol: str,
        market: MarketCode,
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> pl.DataFrame:
        config = MARKET_REGISTRY[market]
        last_err = None
        for provider_name in config.data_providers:
            key = (market, provider_name)
            if key not in self._index:
                continue
            try:
                return await self._index[key].fetch_ohlcv(symbol, market, start, end, timeframe)
            except Exception as e:
                last_err = e
        raise RuntimeError(f"All providers failed for {symbol}@{market}: {last_err}")

    async def search_symbols(self, query: str, market: MarketCode) -> list[dict]:
        config = MARKET_REGISTRY[market]
        provider = self._index[(market, config.data_providers[0])]
        return await provider.search_symbols(query, market)

    async def get_universe_symbols(self, universe_name: str, market: MarketCode) -> list[str]:
        config = MARKET_REGISTRY[market]
        for pname in config.data_providers:
            key = (market, pname)
            if key in self._index:
                return await self._index[key].get_universe_symbols(universe_name, market)
        raise RuntimeError(f"No provider for universe '{universe_name}' in {market}")
```

---

## 3. Schema Changes — `docs/schema/schema.sql`

Add `market_code` to the three tables that need it. Everything else stays exactly as-is.

```sql
-- ── Patch 1: candles_daily — add market_code column ──────────────────────
ALTER TABLE backtester.candles_daily
  ADD COLUMN market_code TEXT NOT NULL DEFAULT 'US';

-- Drop old PK and re-create with market_code included
ALTER TABLE backtester.candles_daily
  DROP CONSTRAINT candles_daily_pkey;

ALTER TABLE backtester.candles_daily
  ADD PRIMARY KEY (market_code, symbol, ts);

-- Update compression segmentby to include market_code
ALTER TABLE backtester.candles_daily SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'market_code,symbol',
    timescaledb.compress_orderby   = 'ts DESC'
);

-- Add partial index per market for fast scans
CREATE INDEX ix_candles_daily_us ON backtester.candles_daily (symbol, ts DESC)
    WHERE market_code = 'US';
CREATE INDEX ix_candles_daily_in ON backtester.candles_daily (symbol, ts DESC)
    WHERE market_code = 'IN';


-- ── Patch 2: candles_intraday — same treatment ────────────────────────────
ALTER TABLE backtester.candles_intraday
  ADD COLUMN market_code TEXT NOT NULL DEFAULT 'US';

ALTER TABLE backtester.candles_intraday
  DROP CONSTRAINT candles_intraday_pkey;

ALTER TABLE backtester.candles_intraday
  ADD PRIMARY KEY (market_code, symbol, timeframe, ts);


-- ── Patch 3: backtests — market_code + currency ───────────────────────────
ALTER TABLE backtester.backtests
  ADD COLUMN market_code TEXT NOT NULL DEFAULT 'US',
  ADD COLUMN currency    TEXT NOT NULL DEFAULT 'USD';

CREATE INDEX ix_backtests_market ON backtester.backtests (market_code, created_at DESC);


-- ── Patch 4: backtest_results — add India-specific cost columns ───────────
ALTER TABLE backtester.backtest_results
  ADD COLUMN stt_total    NUMERIC(15,2) DEFAULT 0,
  ADD COLUMN gst_total    NUMERIC(15,2) DEFAULT 0,
  ADD COLUMN stamp_duty   NUMERIC(15,2) DEFAULT 0;


-- ── Patch 5: trades — STT/GST columns (zero for US, populated for IN) ─────
ALTER TABLE backtester.trades
  ADD COLUMN stt_cost       NUMERIC(10,2) DEFAULT 0,
  ADD COLUMN gst_cost       NUMERIC(10,2) DEFAULT 0,
  ADD COLUMN stamp_duty     NUMERIC(10,2) DEFAULT 0;


-- ── Patch 6: symbols — add market_code ───────────────────────────────────
ALTER TABLE backtester.symbols
  DROP CONSTRAINT symbols_pkey;

ALTER TABLE backtester.symbols
  ADD COLUMN market_code TEXT NOT NULL DEFAULT 'US';

ALTER TABLE backtester.symbols
  ADD PRIMARY KEY (market_code, symbol);


-- ── Patch 7: universes — scope to market ─────────────────────────────────
ALTER TABLE backtester.universes
  ADD COLUMN market_code TEXT NOT NULL DEFAULT 'US';

DROP INDEX IF EXISTS backtester.universes_name_key;
ALTER TABLE backtester.universes
  ADD CONSTRAINT universes_name_market_unique UNIQUE (name, market_code);

CREATE INDEX ix_universes_market ON backtester.universes (market_code);


-- ── Patch 8: seed default universes ──────────────────────────────────────
INSERT INTO backtester.universes (name, description, type, market_code) VALUES
  ('nifty50',       'Nifty 50 index',          'index',  'IN'),
  ('nifty100',      'Nifty 100 index',          'index',  'IN'),
  ('nifty500',      'Nifty 500 index',          'index',  'IN'),
  ('nse_large_cap', 'NSE Large Cap (>₹20,000Cr)','filter', 'IN'),
  ('nse_mid_cap',   'NSE Mid Cap',               'filter', 'IN'),
  ('nse_small_cap', 'NSE Small Cap',             'filter', 'IN')
ON CONFLICT DO NOTHING;
```

---

## 4. OpenAPI Changes — `docs/api/openapi.yaml`

### 4a. Add `/api/v2/markets` endpoint

```yaml
  /api/v2/markets:
    get:
      tags: [Markets]
      summary: List supported markets
      responses:
        "200":
          description: All registered markets
          content:
            application/json:
              schema:
                type: array
                items: { $ref: "#/components/schemas/Market" }
              example:
                - code: "US"
                  name: "United States"
                  currency: "USD"
                  currency_symbol: "$"
                  default_universe: "sp500_liquid"
                  is_default: true
                - code: "IN"
                  name: "India"
                  currency: "INR"
                  currency_symbol: "₹"
                  default_universe: "nifty50"
                  is_default: false
```

### 4b. Add `market` query param to ALL existing endpoints

Add to `GET /api/v2/backtests`:
```yaml
        - name: market
          in: query
          required: true
          schema: { type: string, enum: [US, IN], default: US }
```

Add to `POST /api/v2/backtests` — extend `CreateBacktestRequest`:
```yaml
    CreateBacktestRequest:
      required: [strategy_name, universe, start_date, end_date, initial_capital, market]
      properties:
        market:
          type: string
          enum: [US, IN]
          default: "US"
        # ... all existing fields unchanged
```

Add to `GET /api/v2/strategies`:
```yaml
        - name: market
          in: query
          required: false
          description: "Filter strategies compatible with this market"
          schema: { type: string }
```

Add to `GET /api/v2/universes`:
```yaml
        - name: market
          in: query
          required: true
          schema: { type: string, enum: [US, IN] }
```

Add to `GET /api/v2/data/candles`:
```yaml
        - name: market
          in: query
          required: true
          schema: { type: string, enum: [US, IN] }
```

### 4c. Add Market schema

```yaml
    Market:
      type: object
      properties:
        code:             { type: string }
        name:             { type: string }
        currency:         { type: string }
        currency_symbol:  { type: string }
        default_universe: { type: string }
        is_default:       { type: boolean }
```

### 4d. Extend `BacktestSummary` and `Backtest`

```yaml
    BacktestSummary:
      properties:
        # Add to existing:
        market_code: { type: string }
        currency:    { type: string }

    Backtest:
      properties:
        # Add to existing:
        market_code: { type: string }
        currency:    { type: string }
```

---

## 5. Cost Model — `src/core/execution/costs.py`

Add market-aware costs on top of the existing `CommissionModel` classes:

```python
# Add to existing costs.py — below existing commission models

from src.core.markets.registry import MarketCode, MARKET_REGISTRY

class ZerodhaFlatCommission(CommissionModel):
    """India — ₹20 flat or 0.03% of trade value, whichever is lower."""

    def calculate(self, quantity: int, price: float, side: str) -> float:
        trade_value = quantity * price
        return min(20.0, trade_value * 0.0003)


class MarketAwareCostModel:
    """
    Wraps existing CommissionModel + adds market-specific taxes.
    Used by the engine in place of bare CommissionModel.
    """

    def calculate_total(
        self,
        quantity: int,
        price: float,
        side: str,
        market: MarketCode,
    ) -> dict[str, float]:
        config = MARKET_REGISTRY[market]
        trade_value = quantity * price

        # Base commission (existing models unchanged)
        if config.commission_model == "interactive_brokers":
            commission = InteractiveBrokersCommission().calculate(quantity, price, side)
        elif config.commission_model == "zerodha_flat":
            commission = ZerodhaFlatCommission().calculate(quantity, price, side)
        else:
            commission = 0.0

        stt = gst = stamp_duty = 0.0

        if market == MarketCode.IN:
            flags = config.flags
            # STT — both sides for equity delivery
            stt = trade_value * flags.get("stt_buy_pct" if side in ("buy","cover") else "stt_sell_pct", 0)
            # GST on brokerage
            gst = commission * flags.get("gst_pct", 0)
            # Stamp duty on buy side only
            if side in ("buy", "cover"):
                stamp_duty = trade_value * flags.get("stamp_duty_pct", 0)

        return {
            "commission":  commission,
            "stt":         stt,
            "gst":         gst,
            "stamp_duty":  stamp_duty,
            "total":       commission + stt + gst + stamp_duty,
        }
```

---

## 6. YAML Config — add `market` field

Add to the existing `execution` section in `docs/config/CONFIGURATION.md`:

```yaml
execution:
  market: "US"             # NEW: "US" | "IN" (default: "US")
  initial_capital: 100000
  universe: sp500_liquid   # must match a universe for the selected market
  # ... rest unchanged
```

For India strategy configs:
```yaml
execution:
  market: "IN"
  initial_capital: 1000000   # ₹10 lakh
  universe: nifty50
  start_date: "2020-01-01"
  end_date: "2024-12-31"
  fill_model: realistic_volume
  commission_model: zerodha_flat
```

---

## 7. docker-compose.yml changes

Add India provider API keys. Modify the existing `backtester-v2` service:

```yaml
services:
  backtester-v2:
    environment:
      - DATABASE_URL=postgresql+asyncpg://ava:ava2026@ava-db:5432/ava
      - REDIS_URL=redis://redis:6379
      # Existing
      - FMP_API_KEY=${FMP_API_KEY}
      # New — India providers (optional; falls back to NSEpy if absent)
      - UPSTOX_API_KEY=${UPSTOX_API_KEY}
      - UPSTOX_API_SECRET=${UPSTOX_API_SECRET}
      - ZERODHA_API_KEY=${ZERODHA_API_KEY}
      - ZERODHA_API_SECRET=${ZERODHA_API_SECRET}
      - LOG_LEVEL=INFO
```

---

## 8. UI Changes — exact file diffs

### 8a. `src/types/api.ts` — add Market type + extend existing types

```typescript
// Add new type
export interface Market {
  code: string;
  name: string;
  currency: string;
  currency_symbol: string;
  default_universe: string;
  is_default: boolean;
}

// Extend existing BacktestSummary (add 2 fields)
export interface BacktestSummary {
  // ... all existing fields unchanged ...
  market_code: string;   // ADD
  currency: string;      // ADD
}

// Extend existing Backtest
export interface Backtest {
  // ... all existing fields unchanged ...
  market_code: string;   // ADD
  currency: string;      // ADD
}

// Extend CreateBacktestRequest
export interface CreateBacktestRequest {
  // ... all existing fields unchanged ...
  market: string;        // ADD — required
}
```

### 8b. `src/api/client.ts` — inject market into every call

```typescript
// Add at top of file (after existing imports)
import { getSelectedMarket } from '@/contexts/MarketContext'

// Add new function
export async function listMarkets(): Promise<Market[]> {
  return request('/api/v2/markets')
}

// Modify existing listBacktests — add market param
export async function listBacktests(opts?: {
  status?: string
  strategy?: string
  limit?: number
  offset?: number
  market?: string     // ADD
}): Promise<Paginated<BacktestSummary>> {
  const market = opts?.market ?? getSelectedMarket()
  return request(`/api/v2/backtests${qs({ ...opts, market })}`)
}

// Modify existing createBacktest
export async function createBacktest(body: CreateBacktestRequest): Promise<Backtest> {
  const market = body.market ?? getSelectedMarket()
  return request('/api/v2/backtests', {
    method: 'POST',
    body: JSON.stringify({ ...body, market }),
  })
}

// Modify existing listUniverses
export async function listUniverses(market?: string): Promise<Universe[]> {
  const m = market ?? getSelectedMarket()
  return request(`/api/v2/universes${qs({ market: m })}`)
}

// Modify existing getCandles
export async function getCandles(opts: {
  symbol: string; timeframe?: string; start: string; end: string; market?: string
}): Promise<CandleResponse> {
  const market = opts.market ?? getSelectedMarket()
  return request(`/api/v2/data/candles${qs({ ...opts, market })}`)
}
```

### 8c. New file: `src/contexts/MarketContext.tsx`

```tsx
import { createContext, useContext, useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { listMarkets } from '@/api/client'
import type { Market } from '@/types/api'

const STORAGE_KEY = 'ava_selected_market'

interface MarketContextValue {
  market: Market | null
  setMarket: (code: string) => void
  markets: Market[]
  isLoading: boolean
}

const MarketContext = createContext<MarketContextValue | null>(null)

export function MarketProvider({ children }: { children: React.ReactNode }) {
  const [selectedCode, setSelectedCode] = useState<string>(
    () => localStorage.getItem(STORAGE_KEY) ?? 'US'
  )

  const { data: markets = [], isLoading } = useQuery({
    queryKey: ['markets'],
    queryFn: listMarkets,
    staleTime: Infinity,   // markets list never changes at runtime
  })

  const market = markets.find(m => m.code === selectedCode) ?? markets[0] ?? null

  const setMarket = (code: string) => {
    localStorage.setItem(STORAGE_KEY, code)
    setSelectedCode(code)
  }

  return (
    <MarketContext.Provider value={{ market, setMarket, markets, isLoading }}>
      {children}
    </MarketContext.Provider>
  )
}

export function useMarket() {
  const ctx = useContext(MarketContext)
  if (!ctx) throw new Error('useMarket must be used inside MarketProvider')
  return ctx
}

/** Utility for use outside React components (e.g. in api/client.ts) */
export function getSelectedMarket(): string {
  return localStorage.getItem(STORAGE_KEY) ?? 'US'
}
```

### 8d. `src/App.tsx` — wrap with MarketProvider

```tsx
// Add import
import { MarketProvider } from '@/contexts/MarketContext'

// Wrap inside QueryClientProvider:
function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <MarketProvider>           {/* ADD */}
        <Router>
          {/* ... rest unchanged ... */}
        </Router>
      </MarketProvider>           {/* ADD */}
    </QueryClientProvider>
  )
}
```

### 8e. `src/components/layout/Navbar.tsx` — add market selector

```tsx
import { useMarket } from '@/contexts/MarketContext'

const MARKET_FLAGS: Record<string, string> = {
  US: '🇺🇸',
  IN: '🇮🇳',
  GB: '🇬🇧',
  JP: '🇯🇵',
}

export function Navbar() {
  const { market, markets, setMarket } = useMarket()

  return (
    <nav className="bg-bg-secondary border-b border-border-primary h-[60px] px-6 flex items-center justify-between sticky top-0 z-50">
      <div className="flex items-center gap-3">
        <span className="text-lg font-bold text-accent">📈 Backtester V2</span>
      </div>

      <div className="flex items-center gap-4">
        {/* Market Selector */}
        <div className="relative">
          <select
            value={market?.code ?? 'US'}
            onChange={e => setMarket(e.target.value)}
            className="bg-bg-tertiary border border-border-primary text-text-primary text-sm rounded px-3 py-1.5 pr-8 appearance-none cursor-pointer hover:border-accent focus:outline-none focus:border-accent"
          >
            {markets.map(m => (
              <option key={m.code} value={m.code}>
                {MARKET_FLAGS[m.code] ?? '🌍'} {m.name} ({m.currency})
              </option>
            ))}
          </select>
          <span className="absolute right-2 top-1/2 -translate-y-1/2 text-text-secondary pointer-events-none">▾</span>
        </div>

        <span className="text-text-secondary text-sm">Trading Research Platform</span>
      </div>
    </nav>
  )
}
```

### 8f. `src/utils/formatters.ts` — market-aware currency formatter

Replace the existing hardcoded `formatCurrency`:

```typescript
// Before (existing — hardcoded USD):
export function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value)
}

// After — market-aware:
const MARKET_LOCALE: Record<string, string> = {
  USD: 'en-US',
  INR: 'en-IN',   // uses lakh/crore grouping automatically
}

export function formatCurrency(value: number, currency = 'USD'): string {
  const locale = MARKET_LOCALE[currency] ?? 'en-US'
  return new Intl.NumberFormat(locale, {
    style: 'currency',
    currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value)
}

// All existing callers that pass no currency arg still work (default = 'USD').
// Callers with market context:
//   const { market } = useMarket()
//   formatCurrency(value, market.currency)
```

### 8g. `src/components/layout/Sidebar.tsx` — no changes needed

The sidebar links are market-agnostic — they already navigate to `/strategies`, `/backtests`, etc., and each page reads the market from context. No changes needed.

---

## 9. Extension Playbook — Adding a New Market

**Example: UK / LSE**

1. **`src/core/markets/registry.py`** — add one entry:
```python
MarketCode.GB: MarketConfig(
    code=MarketCode.GB,
    name="United Kingdom",
    exchange="LSE",
    currency="GBP",
    currency_symbol="£",
    timezone=ZoneInfo("Europe/London"),
    session_open="08:00",
    session_close="16:30",
    settlement_days=2,
    lot_size=1,
    tick_size=0.001,
    default_universe="ftse100",
    data_providers=["alpha_vantage", "refinitiv"],
    commission_model="interactive_brokers",
),
```

2. **Implement `src/core/data/providers/alpha_vantage.py`** (if not already done) following `DataProvider` ABC.

3. **Run the SQL patch:**
```sql
INSERT INTO backtester.universes (name, description, type, market_code) VALUES
  ('ftse100', 'FTSE 100 index', 'index', 'GB'),
  ('ftse250', 'FTSE 250 index', 'index', 'GB');
```

4. **Add env var** to `docker-compose.yml`: `ALPHA_VANTAGE_API_KEY=${ALPHA_VANTAGE_API_KEY}`

5. **Done.** The `/api/v2/markets` endpoint immediately returns GB. The market selector adds 🇬🇧. All backtests, trades, candles, and universes are automatically scoped to GB. The `formatCurrency` call uses `£` for GBP. Zero changes to engine, strategies, UI routing, or analytics.

---

## 10. What Does NOT Change

- `src/core/strategy/base.py` — strategies are market-agnostic; zero changes
- `src/core/execution/engine.py` — engine gets `market_code` injected at run time; no structural changes
- `src/core/analytics/` — metrics, attribution, optimization — zero changes
- `src/components/` (all chart/table components) — receive `currency` as a prop where needed
- `src/pages/` — read market from `useMarket()` context; pass `market.currency` to formatters
- WebSocket protocol — unchanged; `market_code` is on the backtest record itself
- TanStack Query cache keys — add `market` to existing query keys: `['backtests', market, filters]`
