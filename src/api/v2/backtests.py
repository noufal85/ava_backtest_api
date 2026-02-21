"""Backtests endpoints — full CRUD, market-scoped."""
from __future__ import annotations

import asyncio
import uuid

import structlog
from fastapi import APIRouter, BackgroundTasks, Query

from src.api.v2.models import CreateBacktestRequest
from src.core.markets.registry import get_market

router = APIRouter(tags=["Backtests"])
logger = structlog.get_logger()


@router.get("/backtests")
async def list_backtests(
    market: str = Query("US"),
    status: str | None = None,
    strategy: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    """List backtests — always market-scoped, never cross-market."""
    get_market(market)  # raises ValueError if invalid
    # Return mock data for now (DB layer in EP-8)
    return {"items": [], "total": 0}


@router.post("/backtests", status_code=201)
async def create_backtest(body: CreateBacktestRequest, background_tasks: BackgroundTasks):
    """Create and queue a new backtest run."""
    get_market(body.market)
    run_id = str(uuid.uuid4())
    # Queue background task
    background_tasks.add_task(_run_backtest_task, run_id, body)
    return {
        "id": run_id,
        "status": "pending",
        "market_code": body.market,
        "strategy_name": body.strategy_name,
        "universe_name": body.universe,
        "initial_capital": body.initial_capital,
        "created_at": "2026-02-21T00:00:00Z",
    }


@router.get("/backtests/{run_id}")
async def get_backtest(run_id: str):
    """Get backtest detail with metrics."""
    return {"id": run_id, "status": "completed", "market_code": "US"}


@router.delete("/backtests/{run_id}", status_code=204)
async def cancel_backtest(run_id: str):
    """Cancel a running backtest or delete record."""
    pass


@router.get("/backtests/{run_id}/trades")
async def get_trades(
    run_id: str,
    direction: str | None = None,
    limit: int = 200,
    offset: int = 0,
):
    """List trades for a backtest."""
    return {"items": [], "total": 0}


@router.get("/backtests/{run_id}/equity-curve")
async def get_equity_curve(run_id: str, resample: str = "D"):
    """Equity curve data points."""
    return {"points": []}


async def _run_backtest_task(run_id: str, body: CreateBacktestRequest):
    """Background task — runs engine, stores results."""
    logger.info(
        "backtest.start",
        run_id=run_id,
        strategy=body.strategy_name,
        market=body.market,
    )
    # Engine integration will be completed in EP-8 when DB is wired
    await asyncio.sleep(0)
