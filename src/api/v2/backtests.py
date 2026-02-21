"""Backtests endpoints — full CRUD, market-scoped, engine-wired."""
from __future__ import annotations

import asyncio
import uuid
from datetime import date, datetime, timezone
from functools import partial

import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from src.api.v2.models import CreateBacktestRequest
from src.core.analytics.metrics import calculate_all_metrics
from src.core.data.providers.fmp import FMPProvider
from src.core.data.universe.symbols import get_symbols
from src.core.execution.engine import BacktestEngine
from src.core.execution.run_store import create_run, get_run, list_runs, update_run
from src.core.markets.registry import MarketCode, get_market
from src.core.strategy.registry import get_strategy

router = APIRouter(tags=["Backtests"])
logger = structlog.get_logger()


# ─────────────────────────── LIST ────────────────────────────────────────────

@router.get("/backtests")
async def list_backtests(
    market: str = Query("US"),
    status: str | None = None,
    strategy: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    get_market(market)
    runs = list_runs(market=market, status=status, strategy=strategy, limit=limit, offset=offset)
    return {"items": runs, "total": len(runs)}


# ─────────────────────────── CREATE ──────────────────────────────────────────

@router.post("/backtests", status_code=201)
async def create_backtest(body: CreateBacktestRequest, background_tasks: BackgroundTasks):
    get_market(body.market)
    run_id = str(uuid.uuid4())

    run = create_run(run_id, {
        "market_code": body.market,
        "strategy_name": body.strategy_name,
        "strategy_version": body.strategy_version,
        "universe_name": body.universe,
        "initial_capital": body.initial_capital,
        "start_date": body.start_date,
        "end_date": body.end_date,
        "parameters": body.parameters,
    })

    background_tasks.add_task(_run_backtest_task, run_id, body)
    return run


# ─────────────────────────── GET ─────────────────────────────────────────────

@router.get("/backtests/{run_id}")
async def get_backtest(run_id: str):
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Backtest {run_id!r} not found")
    return run


@router.delete("/backtests/{run_id}", status_code=204)
async def cancel_backtest(run_id: str):
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Not found")
    update_run(run_id, status="cancelled", completed_at=datetime.now(timezone.utc).isoformat())


# ─────────────────────────── TRADES ──────────────────────────────────────────

@router.get("/backtests/{run_id}/trades")
async def get_trades(
    run_id: str,
    direction: str | None = None,
    limit: int = 200,
    offset: int = 0,
):
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Not found")
    trades = run.get("trades", [])
    if direction:
        trades = [t for t in trades if t.get("side") == direction]
    total = len(trades)
    return {"items": trades[offset: offset + limit], "total": total}


# ─────────────────────────── EQUITY CURVE ────────────────────────────────────

@router.get("/backtests/{run_id}/equity-curve")
async def get_equity_curve(run_id: str, resample: str = "D"):
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Not found")
    return {"points": run.get("equity_curve", [])}


# ─────────────────────────── ENGINE TASK ─────────────────────────────────────

async def _run_backtest_task(run_id: str, body: CreateBacktestRequest) -> None:
    """Background task — fetches data, runs engine, stores results."""
    log = logger.bind(run_id=run_id, strategy=body.strategy_name, market=body.market)
    log.info("backtest.start")
    update_run(run_id, status="running")

    try:
        # ── Market ────────────────────────────────────────────────────────────
        market_code = MarketCode.US if body.market == "US" else MarketCode.IN

        # ── Strategy ──────────────────────────────────────────────────────────
        StrategyCls = get_strategy(body.strategy_name)
        strategy = StrategyCls(**(body.parameters or {}))

        # ── Universe symbols (cap at 5 for speed) ─────────────────────────────
        symbols = get_symbols(body.universe, limit=5)
        log.info("backtest.symbols", symbols=symbols)

        # ── Date range ────────────────────────────────────────────────────────
        start = date.fromisoformat(body.start_date)
        end   = date.fromisoformat(body.end_date)

        # ── Engine ────────────────────────────────────────────────────────────
        engine = BacktestEngine(
            market=market_code,
            initial_capital=body.initial_capital / max(len(symbols), 1),
        )

        provider = FMPProvider()
        all_trades: list[dict] = []
        equity_by_date: dict[str, float] = {}

        loop = asyncio.get_event_loop()

        for sym in symbols:
            update_run(run_id, progress_msg=f"Fetching {sym}…")
            try:
                df = await provider.fetch_ohlcv(sym, market_code, start, end)
            except Exception as exc:
                log.warning("backtest.data_skip", symbol=sym, error=str(exc))
                continue

            if df.is_empty():
                log.warning("backtest.empty_data", symbol=sym)
                continue

            log.info("backtest.run_symbol", symbol=sym, bars=len(df))
            update_run(run_id, progress_msg=f"Running {sym}…")

            # Engine is sync — run in thread executor
            result = await loop.run_in_executor(
                None, partial(engine.run, df, strategy, sym)
            )

            all_trades.extend(result["trades"])

            # Merge equity curves (sum per date across symbols)
            for point in result["equity_curve"]:
                dt = str(point["date"])
                equity_by_date[dt] = equity_by_date.get(dt, body.initial_capital) + (
                    point["equity"] - body.initial_capital / max(len(symbols), 1)
                )

        # ── Build combined equity curve ────────────────────────────────────────
        equity_curve = [
            {"date": dt, "equity": round(eq, 2)}
            for dt, eq in sorted(equity_by_date.items())
        ]

        # ── Metrics ───────────────────────────────────────────────────────────
        metrics: dict = {}
        if equity_curve and all_trades:
            try:
                metrics = calculate_all_metrics(
                    equity_curve=equity_curve,
                    trades=all_trades,
                    initial_capital=body.initial_capital,
                )
            except Exception as exc:
                log.warning("backtest.metrics_error", error=str(exc))

        update_run(
            run_id,
            status="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            trades=all_trades,
            equity_curve=equity_curve,
            metrics=metrics,
            progress_msg="Done",
            total_trades=len(all_trades),
            final_equity=equity_curve[-1]["equity"] if equity_curve else body.initial_capital,
            total_return_pct=(
                (equity_curve[-1]["equity"] / body.initial_capital - 1) * 100
                if equity_curve else 0.0
            ),
        )
        log.info("backtest.complete", trades=len(all_trades), equity_points=len(equity_curve))

    except Exception as exc:
        log.error("backtest.failed", error=str(exc))
        update_run(
            run_id,
            status="failed",
            error=str(exc),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
