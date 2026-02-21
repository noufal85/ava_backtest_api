"""Markets endpoint."""
from fastapi import APIRouter

from src.core.markets.registry import list_markets

router = APIRouter(tags=["Markets"])


@router.get("/markets")
async def get_markets():
    """List all supported markets."""
    return list_markets()
