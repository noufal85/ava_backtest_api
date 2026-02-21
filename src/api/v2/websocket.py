"""WebSocket endpoint — real-time backtest progress."""
from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import structlog

router = APIRouter(tags=["WebSocket"])
logger = structlog.get_logger()


class ConnectionManager:
    def __init__(self):
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, backtest_id: str, ws: WebSocket):
        await ws.accept()
        self._connections.setdefault(backtest_id, []).append(ws)

    def disconnect(self, backtest_id: str, ws: WebSocket):
        if backtest_id in self._connections:
            self._connections[backtest_id].remove(ws)

    async def broadcast(self, backtest_id: str, message: dict):
        for ws in self._connections.get(backtest_id, []):
            try:
                await ws.send_json(message)
            except Exception:
                pass


ws_manager = ConnectionManager()


@router.websocket("/ws/backtests/{backtest_id}")
async def backtest_ws(websocket: WebSocket, backtest_id: str):
    """Real-time backtest progress. Server pushes JSON frames."""
    await ws_manager.connect(backtest_id, websocket)
    try:
        while True:
            # Keep alive — client can send pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(backtest_id, websocket)
