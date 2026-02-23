"""Data sync endpoint — pre-fetch OHLCV data to local cache before backtesting."""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone

import structlog
from fastapi import APIRouter, BackgroundTasks, Query

from src.core.data import get_provider, get_cache
from src.core.data.universe.symbols import UNIVERSE_SYMBOLS
from src.core.markets.registry import MarketCode

router = APIRouter(tags=["Data Sync"])
logger = structlog.get_logger()

# Module-level sync state
_sync_state = {
    "status": "idle",
    "total": 0,
    "done": 0,
    "current": "",
    "errors": [],
    "started_at": None,
    "completed_at": None,
}


@router.post("/data/sync")
async def start_sync(
    background_tasks: BackgroundTasks,
    universe: str = Query("sp500_liquid", description="Universe to sync"),
    days: int = Query(365, description="Days of history to sync"),
    max_symbols: int = Query(0, description="Max symbols (0 = all)"),
):
    """Pre-fetch OHLCV data for all symbols in a universe to local Parquet cache."""
    if _sync_state["status"] == "running":
        return {"error": "Sync already running", **_sync_state}

    symbols = UNIVERSE_SYMBOLS.get(universe, UNIVERSE_SYMBOLS["sp500_liquid"])
    if max_symbols > 0:
        symbols = symbols[:max_symbols]

    _sync_state.update({
        "status": "running",
        "total": len(symbols),
        "done": 0,
        "current": "",
        "errors": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
    })

    background_tasks.add_task(_sync_task, symbols, days)
    return {"message": f"Syncing {len(symbols)} symbols ({days} days)", **_sync_state}


@router.get("/data/sync/status")
async def sync_status():
    """Check sync progress."""
    return _sync_state


@router.post("/data/sync/all")
async def sync_all_universes(
    background_tasks: BackgroundTasks,
    days: int = Query(365, description="Days of history to sync"),
):
    """Sync all US universes to local cache."""
    if _sync_state["status"] == "running":
        return {"error": "Sync already running", **_sync_state}

    # Deduplicate symbols across all US universes
    all_symbols = set()
    for name, syms in UNIVERSE_SYMBOLS.items():
        if not any(s.endswith(".NS") for s in syms):  # skip Indian markets
            all_symbols.update(syms)

    symbols = sorted(all_symbols)
    _sync_state.update({
        "status": "running",
        "total": len(symbols),
        "done": 0,
        "current": "",
        "errors": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
    })

    background_tasks.add_task(_sync_task, symbols, days)
    return {"message": f"Syncing {len(symbols)} unique symbols ({days} days)", **_sync_state}


async def _sync_task(symbols: list[str], days: int):
    """Background task: fetch each symbol and store in cache."""
    end = date.today()
    start = end - timedelta(days=days)
    provider = get_provider()
    cache = get_cache()
    market = MarketCode.US
    market_str = "US"

    for i, sym in enumerate(symbols):
        _sync_state["current"] = sym
        try:
            # Check if we need to hit the API (for rate-limit sleep)
            needs_fetch = cache.is_stale(market_str, sym, start, end, "1d", max_age_hours=24)

            # CachedProvider handles cache check + upstream fetch + store
            df = await provider.fetch_ohlcv(sym, market, start, end)
            if df.is_empty():
                logger.warning("sync.empty", symbol=sym)
                _sync_state["errors"].append({"symbol": sym, "error": "empty"})
            else:
                logger.info("sync.ok", symbol=sym, rows=len(df))

            # Respect FMP rate limit only when we actually hit the API
            if needs_fetch:
                await asyncio.sleep(6.5)

        except Exception as e:
            logger.warning("sync.error", symbol=sym, error=str(e))
            _sync_state["errors"].append({"symbol": sym, "error": str(e)})
            await asyncio.sleep(6.5)  # sleep on error too (likely hit API)

        _sync_state["done"] = i + 1

    _sync_state["status"] = "completed"
    _sync_state["completed_at"] = datetime.now(timezone.utc).isoformat()
    logger.info("sync.complete", total=len(symbols), errors=len(_sync_state["errors"]))
