"""WebSocket — streams live backtest progress from run_store."""
import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.core.execution.run_store import get_run

router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws/backtests/{run_id}")
async def backtest_ws(websocket: WebSocket, run_id: str):
    """Stream backtest status until completed or failed."""
    await websocket.accept()
    try:
        while True:
            run = get_run(run_id)

            if run is None:
                await websocket.send_text(json.dumps({"type": "error", "message": "run not found"}))
                break

            status = run["status"]

            if status == "running" or status == "pending":
                total  = run.get("symbols_total", 1) or 1
                done   = run.get("symbols_done", 0) or 0
                pct    = round(done / total * 100, 1)
                payload = {
                    "type":           "progress",
                    "pct":            pct,
                    "current_symbol": run.get("current_symbol", ""),
                    "symbols_done":   done,
                    "symbols_total":  total,
                }

            elif status == "completed":
                m = run.get("metrics") or {}
                payload = {
                    "type":             "completed",
                    "backtest_id":      run_id,
                    "sharpe":           m.get("sharpe_ratio", 0),
                    "total_return_pct": m.get("total_return_pct", run.get("total_return_pct", 0)),
                }

            elif status == "failed":
                payload = {"type": "error", "message": run.get("error", "Backtest failed")}

            else:
                payload = {"type": "progress", "pct": 0, "current_symbol": "", "symbols_done": 0, "symbols_total": 0}

            await websocket.send_text(json.dumps(payload))

            if status in ("completed", "failed", "cancelled"):
                break

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        pass
