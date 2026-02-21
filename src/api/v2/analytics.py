"""Analytics endpoints — correlation and portfolio metrics."""
from __future__ import annotations

from fastapi import APIRouter, Query

router = APIRouter(tags=["Analytics"])


@router.get("/analytics/correlation")
async def get_correlation(
    backtest_ids: list[str] = Query(...),
    market: str = Query("US"),
):
    """Cross-strategy return correlation matrix."""
    # Placeholder — full implementation in EP-8 when DB is wired
    return {"labels": [], "matrix": [], "market": market}


@router.get("/analytics/portfolio")
async def get_portfolio_metrics(
    backtest_ids: list[str] = Query(...),
    weights: list[float] | None = Query(None),
    market: str = Query("US"),
):
    """Combined portfolio metrics for multiple backtests."""
    return {
        "combined_return_pct": 0,
        "combined_sharpe": 0,
        "combined_max_drawdown_pct": 0,
        "diversification_ratio": 0,
        "weights": {},
        "equity_curve": [],
        "market": market,
    }
