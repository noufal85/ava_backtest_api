"""FastAPI application — ava_backtest_api v2."""
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import src.strategies  # noqa: F401 — registers all strategies
from src.api.v2 import analytics, backtests, data, markets, strategies, sync, universes, websocket
from src.api.v2.errors import value_error_handler

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup", version="2.0.0", market_support=["US", "IN"])
    yield
    logger.info("shutdown")


app = FastAPI(
    title="AvaAI Backtester API",
    version="2.0.0",
    description="Professional multi-market quantitative trading research platform",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(markets.router, prefix="/api/v2")
app.include_router(backtests.router, prefix="/api/v2")
app.include_router(strategies.router, prefix="/api/v2")
app.include_router(universes.router, prefix="/api/v2")
app.include_router(data.router, prefix="/api/v2")
app.include_router(sync.router, prefix="/api/v2")
app.include_router(analytics.router, prefix="/api/v2")
app.include_router(websocket.router, prefix="/api/v2")

app.add_exception_handler(ValueError, value_error_handler)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}
