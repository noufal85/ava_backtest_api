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
from src.core.data import get_provider
from src.core.data.universe.symbols import get_symbols
from src.core.execution.engine import BacktestEngine
from src.core.execution.run_store import create_run, get_run, list_runs, update_run
from src.core.markets.registry import MarketCode, get_market
from src.core.strategy.registry import get_strategy

router = APIRouter(tags=["Backtests"])
logger = structlog.get_logger()


# ─── helpers ──────────────────────────────────────────────────────────────────

def _to_trade_records(raw_trades: list[dict]) -> list[dict]:
    """Transform raw buy/sell pairs into single closed-trade records for the UI."""
    records = []
    for t in raw_trades:
        if t.get("side") != "sell":
            continue
        entry_dt = str(t.get("entry_date", "") or "")
        exit_dt  = str(t.get("date", "") or "")
        entry_p  = float(t.get("entry_price", 0) or 0)
        exit_p   = float(t.get("price", 0) or 0)
        qty      = int(t.get("quantity", 0) or 0)
        pnl      = float(t.get("realized_pnl", 0) or 0)
        pnl_pct  = ((exit_p - entry_p) / entry_p * 100) if entry_p else 0.0
        try:
            ed  = datetime.fromisoformat(entry_dt.split("+")[0].split("-04")[0].split("-05")[0].strip())
            xd  = datetime.fromisoformat(exit_dt.split("+")[0].split("-04")[0].split("-05")[0].strip())
            hold = max(0, (xd - ed).days)
        except Exception:
            hold = 0
        records.append({
            "id":           str(uuid.uuid4()),
            "symbol":       t.get("symbol", ""),
            "direction":    "long",
            "entry_date":   entry_dt[:10],
            "entry_price":  round(entry_p, 4),
            "exit_date":    exit_dt[:10],
            "exit_price":   round(exit_p, 4),
            "shares":       qty,
            "pnl":          round(pnl, 2),
            "pnl_pct":      round(pnl_pct, 2),
            "hold_days":    hold,
            "exit_reason":  t.get("reason", "signal"),
        })
    return records


def _clean_date(raw: object) -> str:
    """Strip timezone/time from a date string → YYYY-MM-DD."""
    s = str(raw)
    return s[:10]


# ─── LIST ─────────────────────────────────────────────────────────────────────

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


# ─── CREATE ───────────────────────────────────────────────────────────────────

@router.post("/backtests", status_code=201)
async def create_backtest(body: CreateBacktestRequest, background_tasks: BackgroundTasks):
    get_market(body.market)
    run_id   = str(uuid.uuid4())
    currency = "USD" if body.market == "US" else "INR"

    run = create_run(run_id, {
        "market_code":       body.market,
        "strategy_name":     body.strategy_name,
        "strategy_version":  body.strategy_version,
        "universe_name":     body.universe,
        "initial_capital":   body.initial_capital,
        "start_date":        body.start_date,
        "end_date":          body.end_date,
        "parameters":        body.parameters,
        "param_yaml":        str(body.parameters or {}),
        "currency":          currency,
        "duration_seconds":  None,
        "results":           None,
        "symbols_total":     0,
        "symbols_done":      0,
        "current_symbol":    "",
    })
    background_tasks.add_task(_run_backtest_task, run_id, body)
    return run


# ─── GET ──────────────────────────────────────────────────────────────────────

@router.get("/backtests/{run_id}")
async def get_backtest(run_id: str):
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Backtest {run_id!r} not found")
    return run


@router.delete("/backtests/{run_id}", status_code=204)
async def cancel_backtest(run_id: str):
    if get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="Not found")
    update_run(run_id, status="cancelled", completed_at=datetime.now(timezone.utc).isoformat())


# ─── TRADES ───────────────────────────────────────────────────────────────────

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
        trades = [t for t in trades if t.get("direction") == direction]
    return {"items": trades[offset: offset + limit], "total": len(trades)}


# ─── EQUITY CURVE ─────────────────────────────────────────────────────────────

@router.get("/backtests/{run_id}/equity-curve")
async def get_equity_curve(run_id: str, resample: str = "D"):
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Not found")
    return {"points": run.get("equity_curve", [])}


# ─── ENGINE TASK ──────────────────────────────────────────────────────────────

async def _run_backtest_task(run_id: str, body: CreateBacktestRequest) -> None:
    log = logger.bind(run_id=run_id, strategy=body.strategy_name, market=body.market)
    log.info("backtest.start")
    update_run(run_id, status="running")
    t0 = datetime.now(timezone.utc)

    try:
        market_code  = MarketCode.US if body.market == "US" else MarketCode.IN
        StrategyCls  = get_strategy(body.strategy_name)
        strategy     = StrategyCls(**(body.parameters or {}))
        symbols      = get_symbols(body.universe, limit=5)
        start        = date.fromisoformat(body.start_date)
        end          = date.fromisoformat(body.end_date)

        update_run(run_id, symbols_total=len(symbols))

        engine   = BacktestEngine(
            market=market_code,
            initial_capital=body.initial_capital / max(len(symbols), 1),
        )
        provider = get_provider()

        raw_trades:    list[dict] = []
        equity_by_date: dict[str, float] = {}
        loop = asyncio.get_event_loop()

        for i, sym in enumerate(symbols):
            update_run(run_id, symbols_done=i, current_symbol=sym, progress_msg=f"Fetching {sym}…")
            try:
                df = await provider.fetch_ohlcv(sym, market_code, start, end)
            except Exception as exc:
                log.warning("backtest.data_skip", symbol=sym, error=str(exc))
                continue

            if df.is_empty():
                continue

            update_run(run_id, progress_msg=f"Running {sym}…")
            result = await loop.run_in_executor(None, partial(engine.run, df, strategy, sym))

            raw_trades.extend(result["trades"])

            per_sym_capital = body.initial_capital / max(len(symbols), 1)
            for point in result["equity_curve"]:
                dt = _clean_date(point["date"])
                equity_by_date[dt] = equity_by_date.get(dt, body.initial_capital) + (
                    float(point["equity"]) - per_sym_capital
                )

        # ── Build equity curve (clean dates, sorted) ───────────────────────────
        equity_curve = [
            {"date": dt, "equity": round(eq, 2)}
            for dt, eq in sorted(equity_by_date.items())
        ]

        # ── Transform raw trades → UI-friendly paired records ──────────────────
        trade_records = _to_trade_records(raw_trades)

        # ── Metrics (use raw trades so sell-side realized_pnl is available) ────
        metrics: dict = {}
        if equity_curve:
            try:
                metrics = calculate_all_metrics(
                    equity_curve=equity_curve,
                    trades=raw_trades,
                    initial_capital=body.initial_capital,
                )
            except Exception as exc:
                log.warning("backtest.metrics_error", error=str(exc))

        duration = round((datetime.now(timezone.utc) - t0).total_seconds(), 1)
        final_eq  = equity_curve[-1]["equity"] if equity_curve else body.initial_capital
        ret_pct   = (final_eq / body.initial_capital - 1) * 100

        update_run(
            run_id,
            status="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            duration_seconds=duration,
            trades=trade_records,
            equity_curve=equity_curve,
            metrics=metrics,
            results=metrics,          # UI accesses bt.results
            symbols_done=len(symbols),
            progress_msg="Done",
            total_trades=len(trade_records),
            final_equity=round(final_eq, 2),
            total_return_pct=round(ret_pct, 2),
            sharpe_ratio=metrics.get("sharpe_ratio"),
        )
        log.info("backtest.complete", trades=len(trade_records), equity_points=len(equity_curve))

    except Exception as exc:
        log.error("backtest.failed", error=str(exc))
        update_run(
            run_id,
            status="failed",
            error=str(exc),
            completed_at=datetime.now(timezone.utc).isoformat(),
            duration_seconds=round((datetime.now(timezone.utc) - t0).total_seconds(), 1),
        )
