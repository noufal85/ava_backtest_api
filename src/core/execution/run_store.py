"""In-memory run store — thread-safe dict backing for backtest runs."""
import threading
from datetime import datetime, timezone

_lock = threading.Lock()
_runs: dict[str, dict] = {}


def create_run(run_id: str, metadata: dict) -> dict:
    run = {
        "id": run_id,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "error": None,
        "metrics": None,
        "equity_curve": [],
        "trades": [],
        **metadata,
    }
    with _lock:
        _runs[run_id] = run
    return run


def update_run(run_id: str, **kwargs) -> None:
    with _lock:
        if run_id in _runs:
            _runs[run_id].update(kwargs)


def get_run(run_id: str) -> dict | None:
    with _lock:
        return dict(_runs[run_id]) if run_id in _runs else None


def list_runs(
    market: str | None = None,
    status: str | None = None,
    strategy: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    with _lock:
        runs = [dict(r) for r in _runs.values()]
    if market:
        runs = [r for r in runs if r.get("market_code") == market]
    if status:
        runs = [r for r in runs if r.get("status") == status]
    if strategy:
        runs = [r for r in runs if r.get("strategy_name") == strategy]
    runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return runs[offset : offset + limit]
