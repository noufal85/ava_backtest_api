"""Strategies endpoints — matches UI StrategySummary and StrategyDetail types."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Query

from src.core.strategy.registry import get_strategy, list_strategies
import src.strategies  # noqa: F401 — trigger self-registration of all 8 strategies

router = APIRouter(tags=["Strategies"])


@router.get("/strategies")
async def list_all_strategies(market: str = Query("US")):
    """All registered strategies in UI-compatible StrategySummary shape."""
    raw = list_strategies()
    return [
        {
            "name":           s["name"],
            "latest_version": s["version"],
            "description":    s["description"],
            "tags":           s["tags"],
            "category":       s.get("category", ""),
            "is_active":      True,
        }
        for s in raw
    ]


@router.get("/strategies/{name}")
async def get_strategy_detail(name: str, version: str = "latest"):
    """Strategy detail with parameter schema — matches UI StrategyDetail type."""
    try:
        cls = get_strategy(name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Strategy {name!r} not found")

    try:
        instance = cls()
    except Exception:
        instance = cls.__new__(cls)  # fallback if constructor needs args

    schema   = instance.get_parameter_schema() if hasattr(instance, "get_parameter_schema") else {}
    defaults = {k: v.get("default") for k, v in schema.items() if isinstance(v, dict) and "default" in v}
    warmup   = instance.get_warmup_periods() if hasattr(instance, "get_warmup_periods") else 0

    return {
        "name":                cls.name,
        "version":             cls.version,
        "description":         cls.description,
        "tags":                cls.tags,
        "category":            getattr(cls, "category", ""),
        "default_config":      defaults,
        "parameter_schema":    schema,
        "required_timeframes": ["1d"],
        "warmup_periods":      warmup,
        "changelog":           f"{cls.version} — initial release",
    }


@router.post("/strategies/{name}/backtest", status_code=201)
async def run_strategy_backtest(name: str, body: dict):
    """Shortcut — delegates to /backtests endpoint."""
    return {"id": str(uuid.uuid4()), "status": "pending", "strategy_name": name}
