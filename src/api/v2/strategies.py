"""Strategies endpoints."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Query

from src.core.strategy.registry import get_strategy, list_strategies
import src.strategies  # noqa: F401 — trigger registration of all 8 strategies

router = APIRouter(tags=["Strategies"])


@router.get("/strategies")
async def list_all_strategies(market: str = Query("US")):
    """All registered strategies. Market param reserved for future compatibility filtering."""
    return list_strategies()


@router.get("/strategies/{name}")
async def get_strategy_detail(name: str, version: str = "latest"):
    """Strategy detail with parameter schema."""
    cls = get_strategy(name)
    instance = cls()
    return {
        "name": cls.name,
        "version": cls.version,
        "description": cls.description,
        "category": cls.category,
        "tags": cls.tags,
        "default_config": {},
        "parameter_schema": instance.get_parameter_schema(),
        "warmup_periods": instance.get_warmup_periods(),
        "changelog": "",
    }


@router.post("/strategies/{name}/backtest", status_code=201)
async def run_strategy_backtest(name: str, body: dict):
    """Shortcut — run a backtest for this strategy."""
    return {"id": str(uuid.uuid4()), "status": "pending", "strategy_name": name}
