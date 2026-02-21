"""WebSocket — streams live backtest progress from run_store."""
import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.core.execution.run_store import get_run

router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws/backtest/{run_id}")
async def backtest_ws(websocket: WebSocket, run_id: str):
    """Stream backtest status until completed or failed."""
    await websocket.accept()
    try:
        while True:
            run = get_run(run_id)
            if run is None:
                await websocket.send_text(json.dumps({"error": "run not found", "run_id": run_id}))
                break

            payload = {
                "run_id": run_id,
                "status": run["status"],
                "trade_count": len(run.get("trades", [])),
                "progress_msg": run.get("progress_msg", ""),
                "error": run.get("error"),
                "total_return_pct": run.get("total_return_pct"),
                "final_equity": run.get("final_equity"),
            }
            await websocket.send_text(json.dumps(payload))

            if run["status"] in ("completed", "failed", "cancelled"):
                break

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        pass
