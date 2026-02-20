# V2 API & UI Design

## Overview

The V2 API and UI represent a complete reimagining of the user experience, moving from V1's simple endpoints to a sophisticated real-time trading research platform. The API provides RESTful endpoints with GraphQL flexibility, WebSocket real-time updates, and the UI delivers a professional dark-themed trading dashboard with advanced analytics.

## API Architecture

### Core API Design Principles

1. **RESTful with GraphQL-style flexibility** - Clean resource endpoints with query customization
2. **Real-time WebSocket integration** - Live backtest progress, portfolio updates, alerts
3. **Comprehensive error handling** - Detailed error context and suggested fixes
4. **Rate limiting and caching** - Production-ready performance and abuse prevention
5. **Versioning strategy** - Backward compatibility with clear migration paths
6. **Authentication & authorization** - Secure access control for multi-user deployments

### FastAPI Application Structure

```python
from fastapi import FastAPI, Depends, HTTPException, WebSocket, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi_limiter import FastAPILimiter
from fastapi_cache import FastAPICache
from contextlib import asynccontextmanager
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from pydantic import BaseModel, Field
import uuid

# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    # Startup
    await FastAPICache.init("redis://localhost:6379")
    await FastAPILimiter.init("redis://localhost:6379")
    
    # Initialize background services
    app.state.backtest_queue = asyncio.Queue()
    app.state.websocket_manager = WebSocketManager()
    app.state.notification_service = NotificationService()
    
    # Start background workers
    asyncio.create_task(backtest_worker(app.state.backtest_queue))
    asyncio.create_task(data_sync_worker())
    
    yield
    
    # Shutdown
    await FastAPICache.clear()
    await FastAPILimiter.close()

# Application setup
app = FastAPI(
    title="Trading Backtester V2",
    description="Professional quantitative trading research platform",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://backtester.yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Custom middleware for request logging and metrics
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(
        "API Request",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=process_time,
        user_agent=request.headers.get("user-agent")
    )
    
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

### Request/Response Models

```python
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime, date
from decimal import Decimal

# Base response model
class APIResponse(BaseModel):
    """Standard API response wrapper."""
    success: bool = True
    message: Optional[str] = None
    data: Optional[Any] = None
    errors: Optional[List[str]] = None
    meta: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

# Strategy configuration
class StrategyConfig(BaseModel):
    """Strategy configuration for backtests."""
    name: str = Field(..., description="Strategy name")
    version: str = Field(default="latest", description="Strategy version")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")
    
    # Execution settings
    initial_capital: Decimal = Field(default=100000, description="Starting capital")
    position_sizing: Dict[str, Any] = Field(default_factory=dict, description="Position sizing config")
    risk_management: Dict[str, Any] = Field(default_factory=dict, description="Risk management rules")
    
    # Data settings
    universe: str = Field(default="sp500_liquid", description="Symbol universe")
    custom_symbols: Optional[List[str]] = Field(default=None, description="Custom symbol list")
    start_date: date = Field(..., description="Backtest start date")
    end_date: date = Field(..., description="Backtest end date") 
    timeframe: str = Field(default="1d", description="Primary timeframe")
    
    # Execution mode
    execution_mode: Literal["vectorized", "event_driven", "hybrid"] = Field(
        default="hybrid", description="Execution engine mode"
    )
    max_workers: int = Field(default=8, description="Parallel processing workers")

# Backtest request/response
class BacktestRequest(BaseModel):
    """Request to start a new backtest."""
    strategy: StrategyConfig
    save_results: bool = Field(default=True, description="Save results to database")
    notify_completion: bool = Field(default=True, description="Send notification when done")
    tags: Optional[List[str]] = Field(default=None, description="Tags for organization")

class BacktestResponse(BaseModel):
    """Response from backtest submission."""
    run_id: uuid.UUID
    status: Literal["queued", "running", "completed", "failed"]
    estimated_duration: Optional[int] = Field(None, description="Estimated duration in seconds")
    websocket_url: str = Field(..., description="WebSocket URL for live updates")

# Results models
class TradeResult(BaseModel):
    """Individual trade result."""
    id: uuid.UUID
    symbol: str
    direction: Literal["long", "short"]
    entry_date: date
    exit_date: Optional[date]
    entry_price: Decimal
    exit_price: Optional[Decimal]
    quantity: int
    pnl: Optional[Decimal]
    pnl_pct: Optional[Decimal]
    commission: Decimal
    slippage: Decimal
    hold_days: Optional[int]
    entry_signal: Dict[str, Any]
    exit_reason: Optional[str]
    
    # Attribution
    regime_at_entry: Optional[str]
    volatility_at_entry: Optional[Decimal]
    portfolio_heat: Optional[Decimal]

class PerformanceMetrics(BaseModel):
    """Comprehensive performance metrics."""
    # Returns
    total_return: Decimal
    annual_return: Decimal
    monthly_returns: List[Decimal]
    
    # Risk metrics  
    volatility: Decimal
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    calmar_ratio: Decimal
    max_drawdown: Decimal
    current_drawdown: Decimal
    
    # Trade statistics
    total_trades: int
    win_rate: Decimal
    profit_factor: Decimal
    avg_win: Decimal
    avg_loss: Decimal
    avg_hold_days: Decimal
    
    # Advanced metrics
    var_95: Optional[Decimal] = Field(None, description="95% Value at Risk")
    expected_shortfall: Optional[Decimal] = Field(None, description="Expected Shortfall")
    omega_ratio: Optional[Decimal] = Field(None, description="Omega ratio")
    
class BacktestResults(BaseModel):
    """Complete backtest results."""
    run_id: uuid.UUID
    strategy_name: str
    strategy_version: str
    status: Literal["completed", "failed", "partial"]
    
    # Execution info
    start_time: datetime
    end_time: datetime
    duration_seconds: Decimal
    symbols_processed: int
    
    # Results
    metrics: PerformanceMetrics
    trades: List[TradeResult]
    equity_curve: List[Dict[str, Any]]  # [{date, equity, drawdown}]
    
    # Error info
    error_message: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

# Query models
class BacktestQuery(BaseModel):
    """Query parameters for backtest results."""
    strategy_names: Optional[List[str]] = None
    date_range: Optional[tuple[date, date]] = None
    min_sharpe: Optional[Decimal] = None
    max_drawdown: Optional[Decimal] = None
    tags: Optional[List[str]] = None
    limit: int = Field(default=50, le=1000)
    offset: int = Field(default=0, ge=0)
    sort_by: str = Field(default="created_at")
    sort_order: Literal["asc", "desc"] = Field(default="desc")
```

### Core API Endpoints

```python
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, WebSocket
from fastapi_cache.decorator import cache
from fastapi_limiter.depends import RateLimiter
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import asyncio

# Initialize routers
strategies_router = APIRouter(prefix="/api/v2/strategies", tags=["strategies"])
backtests_router = APIRouter(prefix="/api/v2/backtests", tags=["backtests"])
data_router = APIRouter(prefix="/api/v2/data", tags=["data"])
optimization_router = APIRouter(prefix="/api/v2/optimization", tags=["optimization"])
websocket_router = APIRouter(prefix="/api/v2/ws", tags=["websocket"])

# Strategy endpoints
@strategies_router.get("/", response_model=APIResponse)
@cache(expire=300)  # Cache for 5 minutes
async def list_strategies(
    category: Optional[str] = None,
    search: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
) -> APIResponse:
    """List available strategies with metadata."""
    try:
        query_params = {}
        if category:
            query_params["category"] = category
        if search:
            query_params["search"] = search
            
        strategies = await StrategyService.list_strategies(db, **query_params)
        
        return APIResponse(
            data={
                "strategies": strategies,
                "total": len(strategies),
                "categories": await StrategyService.get_categories(db)
            }
        )
    except Exception as e:
        logger.error(f"Failed to list strategies: {e}")
        raise HTTPException(status_code=500, detail="Failed to load strategies")

@strategies_router.get("/{strategy_name}", response_model=APIResponse)
@cache(expire=300)
async def get_strategy(
    strategy_name: str,
    version: str = "latest",
    db: AsyncSession = Depends(get_db)
) -> APIResponse:
    """Get strategy details and configuration schema."""
    strategy = await StrategyService.get_strategy(db, strategy_name, version)
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    return APIResponse(data={
        "strategy": strategy,
        "parameter_schema": strategy.get_parameter_schema(),
        "example_config": strategy.get_example_config(),
        "performance_history": await StrategyService.get_performance_history(
            db, strategy_name, limit=10
        )
    })

@strategies_router.post("/{strategy_name}/validate", response_model=APIResponse)
async def validate_strategy_config(
    strategy_name: str,
    config: StrategyConfig,
    db: AsyncSession = Depends(get_db)
) -> APIResponse:
    """Validate strategy configuration before running backtest."""
    validation_result = await StrategyService.validate_config(db, strategy_name, config)
    
    return APIResponse(
        success=validation_result.is_valid,
        data=validation_result.model_dump(),
        errors=validation_result.errors
    )

# Backtest endpoints
@backtests_router.post("/", response_model=BacktestResponse)
@Depends(RateLimiter(times=10, seconds=60))  # 10 backtests per minute
async def create_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> BacktestResponse:
    """Submit a new backtest for execution."""
    
    # Validate strategy configuration
    validation = await StrategyService.validate_config(
        db, request.strategy.name, request.strategy
    )
    if not validation.is_valid:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Invalid strategy configuration", 
                "errors": validation.errors
            }
        )
    
    # Create run record
    run_id = uuid.uuid4()
    run_record = BacktestRun(
        id=run_id,
        user_id=current_user.id,
        strategy_name=request.strategy.name,
        strategy_version=request.strategy.version,
        config=request.strategy.model_dump(),
        status="queued",
        created_at=datetime.now()
    )
    
    db.add(run_record)
    await db.commit()
    
    # Queue for background execution
    await app.state.backtest_queue.put({
        "run_id": run_id,
        "config": request.strategy,
        "user_id": current_user.id,
        "save_results": request.save_results,
        "notify_completion": request.notify_completion
    })
    
    # Estimate duration based on historical data
    estimated_duration = await BacktestService.estimate_duration(request.strategy)
    
    return BacktestResponse(
        run_id=run_id,
        status="queued",
        estimated_duration=estimated_duration,
        websocket_url=f"/api/v2/ws/backtest/{run_id}"
    )

@backtests_router.get("/", response_model=APIResponse)
async def list_backtests(
    query: BacktestQuery = Depends(),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> APIResponse:
    """List user's backtest runs with filtering."""
    
    results = await BacktestService.query_runs(
        db, user_id=current_user.id, query=query
    )
    
    return APIResponse(
        data={
            "runs": results.runs,
            "total": results.total,
            "has_more": results.has_more,
            "aggregations": {
                "strategies": results.strategy_breakdown,
                "performance_distribution": results.performance_distribution
            }
        },
        meta={
            "query": query.model_dump(),
            "execution_time": results.query_time
        }
    )

@backtests_router.get("/{run_id}", response_model=APIResponse)
async def get_backtest_results(
    run_id: uuid.UUID,
    include_trades: bool = Query(default=False, description="Include individual trades"),
    include_signals: bool = Query(default=False, description="Include trading signals"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> APIResponse:
    """Get detailed backtest results."""
    
    result = await BacktestService.get_results(
        db, run_id, user_id=current_user.id,
        include_trades=include_trades,
        include_signals=include_signals
    )
    
    if not result:
        raise HTTPException(status_code=404, detail="Backtest run not found")
    
    return APIResponse(data=result)

@backtests_router.delete("/{run_id}", response_model=APIResponse)
async def delete_backtest(
    run_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> APIResponse:
    """Delete a backtest run and its results."""
    
    deleted = await BacktestService.delete_run(db, run_id, user_id=current_user.id)
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Backtest run not found")
    
    return APIResponse(message="Backtest deleted successfully")

# Comparison endpoint
@backtests_router.post("/compare", response_model=APIResponse)
async def compare_backtests(
    run_ids: List[uuid.UUID],
    metrics: Optional[List[str]] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> APIResponse:
    """Compare multiple backtest results."""
    
    if len(run_ids) > 10:  # Reasonable limit
        raise HTTPException(status_code=422, detail="Too many runs to compare (max 10)")
    
    comparison = await BacktestService.compare_runs(
        db, run_ids, user_id=current_user.id, metrics=metrics
    )
    
    return APIResponse(data=comparison)
```

### WebSocket Real-Time Updates

```python
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json
import asyncio

class WebSocketManager:
    """Manage WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.backtest_subscribers: Dict[uuid.UUID, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, channel: str):
        """Accept new WebSocket connection."""
        await websocket.accept()
        
        if channel not in self.active_connections:
            self.active_connections[channel] = set()
        
        self.active_connections[channel].add(websocket)
        logger.info(f"WebSocket connected to channel: {channel}")
    
    def disconnect(self, websocket: WebSocket, channel: str):
        """Remove WebSocket connection."""
        if channel in self.active_connections:
            self.active_connections[channel].discard(websocket)
        
        # Clean up backtest subscriptions
        for run_id, subscribers in self.backtest_subscribers.items():
            subscribers.discard(websocket)
    
    async def broadcast_to_channel(self, channel: str, message: dict):
        """Broadcast message to all subscribers of a channel."""
        if channel not in self.active_connections:
            return
        
        message_json = json.dumps(message, default=str)
        dead_connections = set()
        
        for websocket in self.active_connections[channel]:
            try:
                await websocket.send_text(message_json)
            except Exception as e:
                logger.warning(f"Failed to send WebSocket message: {e}")
                dead_connections.add(websocket)
        
        # Clean up dead connections
        self.active_connections[channel] -= dead_connections
    
    async def send_backtest_update(self, run_id: uuid.UUID, update: dict):
        """Send update to subscribers of a specific backtest."""
        if run_id not in self.backtest_subscribers:
            return
        
        message = {
            "type": "backtest_update",
            "run_id": str(run_id),
            "timestamp": datetime.now().isoformat(),
            **update
        }
        
        await self._send_to_subscribers(run_id, message)
    
    async def _send_to_subscribers(self, run_id: uuid.UUID, message: dict):
        """Send message to backtest subscribers."""
        if run_id not in self.backtest_subscribers:
            return
        
        message_json = json.dumps(message, default=str)
        dead_connections = set()
        
        for websocket in self.backtest_subscribers[run_id]:
            try:
                await websocket.send_text(message_json)
            except Exception:
                dead_connections.add(websocket)
        
        # Clean up
        self.backtest_subscribers[run_id] -= dead_connections

# WebSocket endpoints
@websocket_router.websocket("/backtest/{run_id}")
async def backtest_websocket(websocket: WebSocket, run_id: uuid.UUID):
    """WebSocket endpoint for backtest progress updates."""
    await app.state.websocket_manager.connect(websocket, f"backtest_{run_id}")
    
    # Subscribe to backtest updates
    if run_id not in app.state.websocket_manager.backtest_subscribers:
        app.state.websocket_manager.backtest_subscribers[run_id] = set()
    app.state.websocket_manager.backtest_subscribers[run_id].add(websocket)
    
    try:
        # Send initial status
        status = await BacktestService.get_status(run_id)
        await websocket.send_json({
            "type": "status",
            "run_id": str(run_id),
            "status": status.status,
            "progress": status.progress,
            "estimated_remaining": status.estimated_remaining
        })
        
        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            # Handle client messages (ping, pause, cancel requests)
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            elif message.get("type") == "cancel":
                await BacktestService.cancel_run(run_id)
            
    except WebSocketDisconnect:
        app.state.websocket_manager.disconnect(websocket, f"backtest_{run_id}")
        if run_id in app.state.websocket_manager.backtest_subscribers:
            app.state.websocket_manager.backtest_subscribers[run_id].discard(websocket)

@websocket_router.websocket("/live_feed")
async def live_feed_websocket(websocket: WebSocket):
    """WebSocket endpoint for live market data and alerts."""
    await app.state.websocket_manager.connect(websocket, "live_feed")
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "subscribe":
                # Subscribe to specific symbols or strategies
                await handle_subscription(websocket, message)
            elif message.get("type") == "unsubscribe":
                await handle_unsubscription(websocket, message)
                
    except WebSocketDisconnect:
        app.state.websocket_manager.disconnect(websocket, "live_feed")
```

### Background Task Processing

```python
async def backtest_worker(queue: asyncio.Queue):
    """Background worker to process backtest queue."""
    logger.info("Backtest worker started")
    
    while True:
        try:
            # Get next job from queue
            job = await queue.get()
            run_id = job["run_id"]
            config = job["config"]
            user_id = job["user_id"]
            
            logger.info(f"Processing backtest job: {run_id}")
            
            # Update status to running
            await BacktestService.update_status(run_id, "running", progress=0)
            await app.state.websocket_manager.send_backtest_update(run_id, {
                "status": "running",
                "progress": 0,
                "message": "Initializing backtest..."
            })
            
            # Execute backtest with progress callbacks
            def progress_callback(progress: int, message: str):
                asyncio.create_task(
                    app.state.websocket_manager.send_backtest_update(run_id, {
                        "status": "running", 
                        "progress": progress,
                        "message": message
                    })
                )
            
            # Run the actual backtest
            execution_engine = HybridExecutionEngine()
            result = await execution_engine.run_backtest_with_progress(
                config, progress_callback
            )
            
            # Save results
            if job["save_results"]:
                await BacktestService.save_results(run_id, result)
            
            # Update final status
            await BacktestService.update_status(run_id, "completed", progress=100)
            await app.state.websocket_manager.send_backtest_update(run_id, {
                "status": "completed",
                "progress": 100,
                "message": "Backtest completed successfully",
                "results_summary": {
                    "total_return": result.metrics.total_return,
                    "sharpe_ratio": result.metrics.sharpe_ratio,
                    "max_drawdown": result.metrics.max_drawdown,
                    "total_trades": result.metrics.total_trades
                }
            })
            
            # Send notification if requested
            if job["notify_completion"]:
                await app.state.notification_service.send_completion_notification(
                    user_id, run_id, result
                )
            
            queue.task_done()
            
        except Exception as e:
            logger.error(f"Backtest worker error: {e}")
            
            # Update status to failed
            await BacktestService.update_status(
                run_id, "failed", error_message=str(e)
            )
            await app.state.websocket_manager.send_backtest_update(run_id, {
                "status": "failed",
                "error": str(e)
            })
            
            queue.task_done()

async def data_sync_worker():
    """Background worker for data synchronization."""
    logger.info("Data sync worker started")
    
    while True:
        try:
            # Check for stale data
            stale_symbols = await DataService.get_stale_symbols()
            
            if stale_symbols:
                logger.info(f"Syncing {len(stale_symbols)} stale symbols")
                
                # Update data
                sync_results = await DataService.bulk_sync(stale_symbols)
                
                # Broadcast updates to live feed subscribers
                await app.state.websocket_manager.broadcast_to_channel("live_feed", {
                    "type": "data_sync",
                    "symbols_updated": len(sync_results.successful),
                    "symbols_failed": len(sync_results.failed),
                    "sync_time": datetime.now().isoformat()
                })
            
            # Sleep for 15 minutes
            await asyncio.sleep(900)
            
        except Exception as e:
            logger.error(f"Data sync worker error: {e}")
            await asyncio.sleep(300)  # Back off on error
```

### Advanced Query Capabilities

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from sqlalchemy import text, and_, or_

class BacktestQueryBuilder:
    """Advanced query builder for backtest results."""
    
    def __init__(self):
        self.filters = []
        self.aggregations = []
        self.sorts = []
    
    def filter_by_strategy(self, strategy_names: List[str]):
        """Filter by strategy names."""
        self.filters.append(("strategy_name", "in", strategy_names))
        return self
    
    def filter_by_performance(self, 
                            min_sharpe: Optional[float] = None,
                            max_drawdown: Optional[float] = None,
                            min_return: Optional[float] = None):
        """Filter by performance metrics."""
        if min_sharpe:
            self.filters.append(("sharpe_ratio", ">=", min_sharpe))
        if max_drawdown:
            self.filters.append(("max_drawdown", "<=", max_drawdown))
        if min_return:
            self.filters.append(("total_return", ">=", min_return))
        return self
    
    def filter_by_date_range(self, start_date: date, end_date: date):
        """Filter by backtest date range."""
        self.filters.append(("start_date", ">=", start_date))
        self.filters.append(("end_date", "<=", end_date))
        return self
    
    def aggregate_by_strategy(self):
        """Add strategy aggregation."""
        self.aggregations.append({
            "field": "strategy_name",
            "metrics": ["count", "avg_sharpe", "avg_return", "avg_drawdown"]
        })
        return self
    
    def sort_by(self, field: str, order: str = "desc"):
        """Add sorting."""
        self.sorts.append((field, order))
        return self
    
    async def execute(self, db: AsyncSession, user_id: uuid.UUID) -> Dict[str, Any]:
        """Execute the query."""
        
        # Build base query
        query = """
        SELECT r.*, m.total_return, m.sharpe_ratio, m.max_drawdown, m.volatility
        FROM backtester.strategy_runs_v2 r
        LEFT JOIN backtester.performance_metrics m ON r.id = m.run_id
        WHERE r.user_id = :user_id
        """
        params = {"user_id": str(user_id)}
        
        # Apply filters
        for field, operator, value in self.filters:
            if operator == "in":
                placeholders = ",".join([f":filter_{field}_{i}" for i in range(len(value))])
                query += f" AND r.{field} IN ({placeholders})"
                for i, v in enumerate(value):
                    params[f"filter_{field}_{i}"] = v
            else:
                query += f" AND m.{field} {operator} :filter_{field}"
                params[f"filter_{field}"] = value
        
        # Apply sorting
        if self.sorts:
            order_clauses = []
            for field, order in self.sorts:
                order_clauses.append(f"m.{field} {order.upper()}")
            query += " ORDER BY " + ", ".join(order_clauses)
        
        # Execute query
        result = await db.execute(text(query), params)
        rows = result.fetchall()
        
        # Build response
        runs = []
        for row in rows:
            runs.append({
                "id": row.id,
                "strategy_name": row.strategy_name,
                "strategy_version": row.strategy_version,
                "created_at": row.created_at,
                "status": row.status,
                "metrics": {
                    "total_return": row.total_return,
                    "sharpe_ratio": row.sharpe_ratio,
                    "max_drawdown": row.max_drawdown,
                    "volatility": row.volatility
                }
            })
        
        # Execute aggregations
        aggregation_results = {}
        for agg in self.aggregations:
            if agg["field"] == "strategy_name":
                agg_query = """
                SELECT r.strategy_name,
                       COUNT(*) as run_count,
                       AVG(m.sharpe_ratio) as avg_sharpe,
                       AVG(m.total_return) as avg_return,
                       AVG(m.max_drawdown) as avg_drawdown
                FROM backtester.strategy_runs_v2 r
                LEFT JOIN backtester.performance_metrics m ON r.id = m.run_id
                WHERE r.user_id = :user_id
                GROUP BY r.strategy_name
                ORDER BY avg_sharpe DESC
                """
                
                agg_result = await db.execute(text(agg_query), {"user_id": str(user_id)})
                aggregation_results["by_strategy"] = [
                    {
                        "strategy_name": row.strategy_name,
                        "run_count": row.run_count,
                        "avg_sharpe": row.avg_sharpe,
                        "avg_return": row.avg_return,
                        "avg_drawdown": row.avg_drawdown
                    }
                    for row in agg_result.fetchall()
                ]
        
        return {
            "runs": runs,
            "total": len(runs),
            "aggregations": aggregation_results
        }

# Usage in endpoint
@backtests_router.get("/advanced_query", response_model=APIResponse)
async def advanced_backtest_query(
    strategy_names: Optional[str] = Query(None, description="Comma-separated strategy names"),
    min_sharpe: Optional[float] = Query(None, description="Minimum Sharpe ratio"),
    max_drawdown: Optional[float] = Query(None, description="Maximum drawdown"),
    start_date: Optional[date] = Query(None, description="Start date filter"),
    end_date: Optional[date] = Query(None, description="End date filter"),
    aggregate_by_strategy: bool = Query(False, description="Include strategy aggregations"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> APIResponse:
    """Advanced backtest query with filtering and aggregation."""
    
    builder = BacktestQueryBuilder()
    
    if strategy_names:
        names = [name.strip() for name in strategy_names.split(",")]
        builder.filter_by_strategy(names)
    
    if any([min_sharpe, max_drawdown]):
        builder.filter_by_performance(min_sharpe=min_sharpe, max_drawdown=max_drawdown)
    
    if start_date and end_date:
        builder.filter_by_date_range(start_date, end_date)
    
    if aggregate_by_strategy:
        builder.aggregate_by_strategy()
    
    builder.sort_by(sort_by, sort_order)
    
    results = await builder.execute(db, current_user.id)
    
    return APIResponse(data=results)
```

## Frontend UI Design

### React Architecture

```typescript
// Core app structure
interface AppState {
  user: User | null;
  theme: 'dark' | 'light';
  notifications: Notification[];
  activeBacktests: Map<string, BacktestStatus>;
  websocketConnections: Map<string, WebSocket>;
}

interface User {
  id: string;
  username: string;
  email: string;
  preferences: UserPreferences;
}

interface UserPreferences {
  defaultUniverse: string;
  defaultTimeframe: string;
  notifications: {
    email: boolean;
    push: boolean;
    websocket: boolean;
  };
  dashboard: {
    layout: DashboardLayout;
    defaultMetrics: string[];
  };
}

// Main App component
import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';

// Components
import { Navbar } from './components/layout/Navbar';
import { Sidebar } from './components/layout/Sidebar';
import { Dashboard } from './pages/Dashboard';
import { StrategyList } from './pages/StrategyList';
import { BacktestResults } from './pages/BacktestResults';
import { StrategyBuilder } from './pages/StrategyBuilder';
import { LiveTrading } from './pages/LiveTrading';

// Context
import { AuthProvider } from './contexts/AuthContext';
import { ThemeProvider } from './contexts/ThemeProvider';
import { WebSocketProvider } from './contexts/WebSocketProvider';

// Styles
import './styles/globals.css';
import './styles/dark-theme.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <ThemeProvider>
          <WebSocketProvider>
            <Router>
              <div className="app dark-theme">
                <Navbar />
                <div className="app-container">
                  <Sidebar />
                  <main className="main-content">
                    <Routes>
                      <Route path="/" element={<Dashboard />} />
                      <Route path="/strategies" element={<StrategyList />} />
                      <Route path="/strategies/:name" element={<StrategyBuilder />} />
                      <Route path="/backtests" element={<BacktestResults />} />
                      <Route path="/backtests/:runId" element={<BacktestDetails />} />
                      <Route path="/live" element={<LiveTrading />} />
                      <Route path="/data" element={<DataManagement />} />
                      <Route path="/settings" element={<Settings />} />
                    </Routes>
                  </main>
                </div>
                <Toaster 
                  position="top-right"
                  toastOptions={{
                    className: 'toast-notification',
                    duration: 4000
                  }}
                />
              </div>
            </Router>
          </WebSocketProvider>
        </ThemeProvider>
      </AuthProvider>
    </QueryClientProvider>
  );
}

export default App;
```

### Dashboard Design

```typescript
import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Grid } from '@mui/material';

// Components
import { MetricsCard } from '../components/dashboard/MetricsCard';
import { PerformanceChart } from '../components/charts/PerformanceChart';
import { RecentBacktests } from '../components/dashboard/RecentBacktests';
import { StrategyLeaderboard } from '../components/dashboard/StrategyLeaderboard';
import { MarketOverview } from '../components/dashboard/MarketOverview';
import { ActiveBacktests } from '../components/dashboard/ActiveBacktests';
import { QuickActions } from '../components/dashboard/QuickActions';

// Hooks
import { useWebSocket } from '../hooks/useWebSocket';
import { useAuth } from '../contexts/AuthContext';

// API
import { getDashboardData } from '../api/backtests';

interface DashboardData {
  portfolio_summary: PortfolioSummary;
  recent_backtests: BacktestRun[];
  strategy_leaderboard: StrategyPerformance[];
  market_overview: MarketData;
  active_backtests: ActiveBacktest[];
}

export function Dashboard() {
  const { user } = useAuth();
  const [selectedTimeframe, setSelectedTimeframe] = useState('1M');
  
  // Fetch dashboard data
  const { data, isLoading, error } = useQuery({
    queryKey: ['dashboard', user?.id, selectedTimeframe],
    queryFn: () => getDashboardData({ timeframe: selectedTimeframe }),
    refetchInterval: 30000, // Refresh every 30 seconds
  });
  
  // WebSocket for real-time updates
  const { lastMessage } = useWebSocket('/api/v2/ws/dashboard', {
    onMessage: (message) => {
      // Handle real-time updates
      const update = JSON.parse(message.data);
      if (update.type === 'backtest_completed') {
        // Refetch dashboard data
        queryClient.invalidateQueries(['dashboard']);
        toast.success(`Backtest completed: ${update.strategy_name}`);
      }
    }
  });
  
  if (isLoading) return <DashboardSkeleton />;
  if (error) return <ErrorMessage error={error} />;
  
  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h1>Trading Dashboard</h1>
        <div className="dashboard-controls">
          <TimeframePicker 
            value={selectedTimeframe}
            onChange={setSelectedTimeframe}
          />
          <QuickActions />
        </div>
      </div>
      
      <Grid container spacing={3}>
        {/* Portfolio Overview */}
        <Grid item xs={12}>
          <div className="metrics-row">
            <MetricsCard
              title="Total Return"
              value={data.portfolio_summary.total_return}
              format="percentage"
              trend={data.portfolio_summary.return_trend}
            />
            <MetricsCard
              title="Sharpe Ratio"
              value={data.portfolio_summary.sharpe_ratio}
              format="decimal"
              benchmark={1.0}
            />
            <MetricsCard
              title="Max Drawdown"
              value={data.portfolio_summary.max_drawdown}
              format="percentage"
              inverted={true}
            />
            <MetricsCard
              title="Active Strategies"
              value={data.portfolio_summary.active_strategies}
              format="integer"
            />
          </div>
        </Grid>
        
        {/* Performance Chart */}
        <Grid item xs={12} lg={8}>
          <PerformanceChart
            data={data.portfolio_summary.equity_curve}
            benchmark={data.market_overview.benchmark_data}
            timeframe={selectedTimeframe}
          />
        </Grid>
        
        {/* Active Backtests */}
        <Grid item xs={12} lg={4}>
          <ActiveBacktests backtests={data.active_backtests} />
        </Grid>
        
        {/* Strategy Leaderboard */}
        <Grid item xs={12} lg={6}>
          <StrategyLeaderboard strategies={data.strategy_leaderboard} />
        </Grid>
        
        {/* Recent Backtests */}
        <Grid item xs={12} lg={6}>
          <RecentBacktests backtests={data.recent_backtests} />
        </Grid>
        
        {/* Market Overview */}
        <Grid item xs={12}>
          <MarketOverview data={data.market_overview} />
        </Grid>
      </Grid>
    </div>
  );
}

// Performance Chart Component
interface PerformanceChartProps {
  data: EquityPoint[];
  benchmark?: EquityPoint[];
  timeframe: string;
}

export function PerformanceChart({ data, benchmark, timeframe }: PerformanceChartProps) {
  const chartData = useMemo(() => {
    return data.map(point => ({
      date: new Date(point.date),
      equity: point.equity,
      drawdown: point.drawdown,
      benchmark: benchmark?.find(b => b.date === point.date)?.equity
    }));
  }, [data, benchmark]);
  
  const chartConfig = {
    responsive: true,
    interaction: {
      mode: 'index' as const,
      intersect: false,
    },
    scales: {
      x: {
        type: 'time' as const,
        time: {
          displayFormats: {
            day: 'MMM dd',
            week: 'MMM dd',
            month: 'MMM yyyy'
          }
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)',
        },
        ticks: {
          color: 'rgba(255, 255, 255, 0.7)',
        }
      },
      y: {
        beginAtZero: false,
        grid: {
          color: 'rgba(255, 255, 255, 0.1)',
        },
        ticks: {
          color: 'rgba(255, 255, 255, 0.7)',
          callback: (value: any) => `$${(value / 1000).toFixed(0)}K`
        }
      }
    },
    plugins: {
      legend: {
        display: true,
        position: 'top' as const,
        labels: {
          color: 'rgba(255, 255, 255, 0.9)',
        }
      },
      tooltip: {
        backgroundColor: 'rgba(20, 20, 20, 0.95)',
        titleColor: 'white',
        bodyColor: 'white',
        borderColor: 'rgba(255, 255, 255, 0.2)',
        borderWidth: 1,
        callbacks: {
          label: (context: any) => {
            const label = context.dataset.label || '';
            const value = context.parsed.y;
            return `${label}: $${value.toLocaleString()}`;
          }
        }
      }
    }
  };
  
  return (
    <div className="performance-chart-container">
      <div className="chart-header">
        <h3>Portfolio Performance</h3>
        <div className="chart-controls">
          <button 
            className="chart-toggle active" 
            onClick={() => setShowDrawdown(!showDrawdown)}
          >
            {showDrawdown ? 'Hide' : 'Show'} Drawdown
          </button>
        </div>
      </div>
      <div className="chart-wrapper">
        <Line data={chartData} options={chartConfig} />
        {showDrawdown && (
          <div className="drawdown-overlay">
            <DrawdownChart data={chartData} />
          </div>
        )}
      </div>
    </div>
  );
}
```

### Strategy Builder Interface

```typescript
import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useMutation, useQuery } from '@tanstack/react-query';
import { Form, Formik } from 'formik';
import * as Yup from 'yup';

// Components
import { ParameterInput } from '../components/strategy/ParameterInput';
import { StrategyPreview } from '../components/strategy/StrategyPreview';
import { BacktestConfiguration } from '../components/strategy/BacktestConfiguration';
import { ValidationResults } from '../components/strategy/ValidationResults';

// API
import { getStrategy, validateStrategyConfig, createBacktest } from '../api/strategies';

interface StrategyBuilderProps {
  strategyName?: string;
}

export function StrategyBuilder() {
  const { name: strategyName } = useParams<{ name: string }>();
  const navigate = useNavigate();
  
  const [activeTab, setActiveTab] = useState<'parameters' | 'backtest' | 'preview'>('parameters');
  const [validationResults, setValidationResults] = useState<ValidationResult | null>(null);
  
  // Fetch strategy details
  const { data: strategy, isLoading } = useQuery({
    queryKey: ['strategy', strategyName],
    queryFn: () => getStrategy(strategyName!),
    enabled: !!strategyName,
  });
  
  // Validation mutation
  const validateMutation = useMutation({
    mutationFn: validateStrategyConfig,
    onSuccess: (results) => {
      setValidationResults(results);
    }
  });
  
  // Backtest submission mutation
  const backtestMutation = useMutation({
    mutationFn: createBacktest,
    onSuccess: (response) => {
      navigate(`/backtests/${response.run_id}`);
      toast.success('Backtest started successfully!');
    },
    onError: (error) => {
      toast.error(`Failed to start backtest: ${error.message}`);
    }
  });
  
  if (isLoading) return <StrategySkeleton />;
  if (!strategy) return <NotFound />;
  
  const initialValues = {
    // Strategy parameters
    ...strategy.default_parameters,
    
    // Backtest configuration
    start_date: '2020-01-01',
    end_date: '2024-12-31',
    initial_capital: 100000,
    universe: 'sp500_liquid',
    custom_symbols: [],
    execution_mode: 'hybrid',
    
    // Position sizing
    position_sizing: {
      method: 'fixed_dollar',
      amount: 5000
    },
    
    // Risk management
    risk_management: {
      max_position_pct: 0.05,
      max_total_exposure: 0.8,
      stop_loss_pct: 0.02
    }
  };
  
  const validationSchema = Yup.object().shape({
    // Dynamic validation based on strategy parameter schema
    ...generateValidationSchema(strategy.parameter_schema),
    
    start_date: Yup.date().required('Start date is required'),
    end_date: Yup.date()
      .min(Yup.ref('start_date'), 'End date must be after start date')
      .required('End date is required'),
    initial_capital: Yup.number()
      .min(1000, 'Minimum capital is $1,000')
      .required('Initial capital is required'),
  });
  
  return (
    <div className="strategy-builder">
      <div className="strategy-header">
        <div className="strategy-info">
          <h1>{strategy.name}</h1>
          <p className="strategy-description">{strategy.description}</p>
          <div className="strategy-tags">
            {strategy.tags.map(tag => (
              <span key={tag} className="tag">{tag}</span>
            ))}
          </div>
        </div>
        
        <div className="strategy-stats">
          <div className="stat">
            <label>Version</label>
            <span>{strategy.version}</span>
          </div>
          <div className="stat">
            <label>Category</label>
            <span>{strategy.category}</span>
          </div>
          <div className="stat">
            <label>Avg Performance</label>
            <span className="positive">+23.4%</span>
          </div>
        </div>
      </div>
      
      <div className="strategy-tabs">
        <button 
          className={`tab ${activeTab === 'parameters' ? 'active' : ''}`}
          onClick={() => setActiveTab('parameters')}
        >
          Parameters
        </button>
        <button 
          className={`tab ${activeTab === 'backtest' ? 'active' : ''}`}
          onClick={() => setActiveTab('backtest')}
        >
          Backtest Setup
        </button>
        <button 
          className={`tab ${activeTab === 'preview' ? 'active' : ''}`}
          onClick={() => setActiveTab('preview')}
        >
          Preview & Run
        </button>
      </div>
      
      <Formik
        initialValues={initialValues}
        validationSchema={validationSchema}
        onSubmit={async (values, { setSubmitting }) => {
          // Validate configuration
          await validateMutation.mutateAsync({
            strategy_name: strategyName!,
            config: values
          });
          
          if (validationResults?.is_valid) {
            // Submit backtest
            await backtestMutation.mutateAsync({
              strategy: {
                name: strategyName!,
                version: strategy.version,
                parameters: values,
                ...values
              }
            });
          }
          
          setSubmitting(false);
        }}
      >
        {({ values, errors, touched, setFieldValue, isSubmitting, isValid }) => (
          <Form className="strategy-form">
            {activeTab === 'parameters' && (
              <div className="parameters-tab">
                <div className="parameters-grid">
                  {Object.entries(strategy.parameter_schema).map(([paramName, schema]) => (
                    <ParameterInput
                      key={paramName}
                      name={paramName}
                      schema={schema}
                      value={values[paramName]}
                      onChange={(value) => setFieldValue(paramName, value)}
                      error={touched[paramName] && errors[paramName]}
                    />
                  ))}
                </div>
                
                <div className="parameter-presets">
                  <h3>Parameter Presets</h3>
                  <div className="presets-grid">
                    {strategy.presets.map(preset => (
                      <PresetCard
                        key={preset.name}
                        preset={preset}
                        onApply={(params) => {
                          Object.entries(params).forEach(([key, value]) => {
                            setFieldValue(key, value);
                          });
                        }}
                      />
                    ))}
                  </div>
                </div>
              </div>
            )}
            
            {activeTab === 'backtest' && (
              <BacktestConfiguration
                values={values}
                setFieldValue={setFieldValue}
                errors={errors}
                touched={touched}
              />
            )}
            
            {activeTab === 'preview' && (
              <div className="preview-tab">
                <div className="preview-grid">
                  <div className="config-preview">
                    <StrategyPreview
                      strategy={strategy}
                      configuration={values}
                    />
                    
                    {validationResults && (
                      <ValidationResults results={validationResults} />
                    )}
                  </div>
                  
                  <div className="actions-panel">
                    <div className="estimated-duration">
                      <h4>Estimated Duration</h4>
                      <p>~3.5 minutes</p>
                      <small>Based on similar backtests</small>
                    </div>
                    
                    <div className="resource-usage">
                      <h4>Resource Usage</h4>
                      <div className="usage-bars">
                        <div className="usage-bar">
                          <label>CPU</label>
                          <div className="bar"><div className="fill" style={{width: '65%'}}></div></div>
                          <span>65%</span>
                        </div>
                        <div className="usage-bar">
                          <label>Memory</label>
                          <div className="bar"><div className="fill" style={{width: '40%'}}></div></div>
                          <span>~2.1GB</span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="action-buttons">
                      <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={() => validateMutation.mutate({
                          strategy_name: strategyName!,
                          config: values
                        })}
                        disabled={validateMutation.isLoading}
                      >
                        {validateMutation.isLoading ? 'Validating...' : 'Validate'}
                      </button>
                      
                      <button
                        type="submit"
                        className="btn btn-primary"
                        disabled={!isValid || isSubmitting || backtestMutation.isLoading}
                      >
                        {isSubmitting ? 'Starting...' : 'Run Backtest'}
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </Form>
        )}
      </Formik>
    </div>
  );
}
```

### Dark Theme Design System

```css
/* globals.css - Dark theme trading dashboard */

:root {
  /* Color Palette */
  --bg-primary: #0d1117;
  --bg-secondary: #161b22;
  --bg-tertiary: #21262d;
  --bg-overlay: rgba(13, 17, 23, 0.95);
  
  --border-primary: #30363d;
  --border-secondary: #21262d;
  --border-accent: #58a6ff;
  
  --text-primary: #f0f6fc;
  --text-secondary: #8b949e;
  --text-muted: #6e7681;
  
  /* Trading Colors */
  --color-bull: #2ea043;  /* Green for profits/bullish */
  --color-bear: #da3633;  /* Red for losses/bearish */
  --color-neutral: #656d76;
  
  --color-bull-bg: rgba(46, 160, 67, 0.15);
  --color-bear-bg: rgba(218, 54, 51, 0.15);
  
  /* Accent Colors */
  --color-primary: #58a6ff;
  --color-secondary: #8957e5;
  --color-warning: #d29922;
  --color-success: var(--color-bull);
  --color-error: var(--color-bear);
  
  /* Gradients */
  --gradient-primary: linear-gradient(135deg, #58a6ff 0%, #8957e5 100%);
  --gradient-bull: linear-gradient(135deg, #2ea043 0%, #56d364 100%);
  --gradient-bear: linear-gradient(135deg, #da3633 0%, #ff7b72 100%);
  
  /* Typography */
  --font-mono: 'JetBrains Mono', 'SF Mono', monospace;
  --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  
  /* Spacing */
  --space-xs: 0.25rem;
  --space-sm: 0.5rem;
  --space-md: 1rem;
  --space-lg: 1.5rem;
  --space-xl: 2rem;
  --space-2xl: 3rem;
  
  /* Shadows */
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.25);
  --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.3);
  --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.4);
  --shadow-xl: 0 20px 25px rgba(0, 0, 0, 0.5);
  
  /* Animations */
  --transition-fast: 150ms ease-in-out;
  --transition-medium: 300ms ease-in-out;
  --transition-slow: 500ms ease-in-out;
}

/* Base Styles */
* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: var(--font-sans);
  background: var(--bg-primary);
  color: var(--text-primary);
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* Layout Components */
.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.navbar {
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border-primary);
  padding: 0 var(--space-lg);
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky;
  top: 0;
  z-index: 100;
}

.app-container {
  display: flex;
  flex: 1;
}

.sidebar {
  width: 240px;
  background: var(--bg-secondary);
  border-right: 1px solid var(--border-primary);
  padding: var(--space-lg);
  overflow-y: auto;
}

.main-content {
  flex: 1;
  padding: var(--space-lg);
  background: var(--bg-primary);
  overflow-y: auto;
}

/* Card Components */
.card {
  background: var(--bg-secondary);
  border: 1px solid var(--border-primary);
  border-radius: 8px;
  padding: var(--space-lg);
  box-shadow: var(--shadow-sm);
  transition: box-shadow var(--transition-medium);
}

.card:hover {
  box-shadow: var(--shadow-md);
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: var(--space-md);
  padding-bottom: var(--space-sm);
  border-bottom: 1px solid var(--border-secondary);
}

.card-title {
  margin: 0;
  font-size: 1.125rem;
  font-weight: 600;
  color: var(--text-primary);
}

/* Metrics Components */
.metrics-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: var(--space-lg);
  margin-bottom: var(--space-xl);
}

.metrics-card {
  background: var(--bg-secondary);
  border: 1px solid var(--border-primary);
  border-radius: 8px;
  padding: var(--space-lg);
  text-align: center;
  position: relative;
  overflow: hidden;
}

.metrics-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: var(--gradient-primary);
}

.metrics-card.positive::before {
  background: var(--gradient-bull);
}

.metrics-card.negative::before {
  background: var(--gradient-bear);
}

.metrics-title {
  font-size: 0.875rem;
  color: var(--text-secondary);
  margin-bottom: var(--space-sm);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.metrics-value {
  font-size: 2rem;
  font-weight: 700;
  font-family: var(--font-mono);
  margin: 0;
}

.metrics-value.positive {
  color: var(--color-bull);
}

.metrics-value.negative {
  color: var(--color-bear);
}

.metrics-trend {
  font-size: 0.75rem;
  color: var(--text-muted);
  margin-top: var(--space-xs);
}

/* Charts */
.chart-container {
  background: var(--bg-secondary);
  border: 1px solid var(--border-primary);
  border-radius: 8px;
  padding: var(--space-lg);
}

.chart-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: var(--space-lg);
}

.chart-controls {
  display: flex;
  gap: var(--space-sm);
}

.chart-toggle {
  background: var(--bg-tertiary);
  border: 1px solid var(--border-primary);
  color: var(--text-secondary);
  padding: var(--space-sm) var(--space-md);
  border-radius: 4px;
  font-size: 0.875rem;
  cursor: pointer;
  transition: all var(--transition-fast);
}

.chart-toggle:hover {
  border-color: var(--border-accent);
  color: var(--text-primary);
}

.chart-toggle.active {
  background: var(--color-primary);
  border-color: var(--color-primary);
  color: white;
}

/* Tables */
.data-table {
  width: 100%;
  background: var(--bg-secondary);
  border: 1px solid var(--border-primary);
  border-radius: 8px;
  overflow: hidden;
}

.data-table th {
  background: var(--bg-tertiary);
  padding: var(--space-md);
  text-align: left;
  font-weight: 600;
  color: var(--text-primary);
  border-bottom: 1px solid var(--border-primary);
}

.data-table td {
  padding: var(--space-md);
  border-bottom: 1px solid var(--border-secondary);
}

.data-table tr:hover {
  background: rgba(88, 166, 255, 0.05);
}

/* Forms */
.form-group {
  margin-bottom: var(--space-lg);
}

.form-label {
  display: block;
  font-weight: 500;
  color: var(--text-primary);
  margin-bottom: var(--space-sm);
}

.form-input {
  width: 100%;
  background: var(--bg-primary);
  border: 1px solid var(--border-primary);
  border-radius: 4px;
  padding: var(--space-md);
  color: var(--text-primary);
  font-family: var(--font-mono);
  transition: border-color var(--transition-fast);
}

.form-input:focus {
  outline: none;
  border-color: var(--color-primary);
  box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.2);
}

.form-error {
  color: var(--color-error);
  font-size: 0.875rem;
  margin-top: var(--space-xs);
}

/* Buttons */
.btn {
  display: inline-flex;
  align-items: center;
  gap: var(--space-sm);
  padding: var(--space-md) var(--space-lg);
  border: 1px solid transparent;
  border-radius: 6px;
  font-weight: 500;
  text-decoration: none;
  cursor: pointer;
  transition: all var(--transition-fast);
  font-family: inherit;
}

.btn-primary {
  background: var(--color-primary);
  color: white;
  border-color: var(--color-primary);
}

.btn-primary:hover {
  background: #4493e6;
  border-color: #4493e6;
}

.btn-secondary {
  background: var(--bg-tertiary);
  color: var(--text-primary);
  border-color: var(--border-primary);
}

.btn-secondary:hover {
  background: var(--bg-primary);
  border-color: var(--border-accent);
}

.btn-success {
  background: var(--color-bull);
  color: white;
  border-color: var(--color-bull);
}

.btn-danger {
  background: var(--color-bear);
  color: white;
  border-color: var(--color-bear);
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

/* Status Indicators */
.status-indicator {
  display: inline-flex;
  align-items: center;
  gap: var(--space-xs);
  font-size: 0.875rem;
  font-weight: 500;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.status-running .status-dot {
  background: var(--color-warning);
  animation: pulse 2s infinite;
}

.status-completed .status-dot {
  background: var(--color-success);
}

.status-failed .status-dot {
  background: var(--color-error);
}

.status-queued .status-dot {
  background: var(--text-muted);
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

/* Progress Bars */
.progress-bar {
  width: 100%;
  height: 8px;
  background: var(--bg-primary);
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: var(--gradient-primary);
  transition: width var(--transition-medium);
  position: relative;
}

.progress-fill::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(
    90deg,
    transparent,
    rgba(255, 255, 255, 0.2),
    transparent
  );
  animation: shimmer 2s infinite;
}

@keyframes shimmer {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}

/* Responsive Design */
@media (max-width: 768px) {
  .app-container {
    flex-direction: column;
  }
  
  .sidebar {
    width: 100%;
    height: auto;
    border-right: none;
    border-bottom: 1px solid var(--border-primary);
  }
  
  .metrics-row {
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: var(--space-md);
  }
  
  .main-content {
    padding: var(--space-md);
  }
}

/* Custom Scrollbars */
::-webkit-scrollbar {
  width: 8px;
}

::-webkit-scrollbar-track {
  background: var(--bg-primary);
}

::-webkit-scrollbar-thumb {
  background: var(--border-primary);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: var(--text-muted);
}

/* Tooltips */
.tooltip {
  position: absolute;
  background: var(--bg-overlay);
  color: var(--text-primary);
  padding: var(--space-sm) var(--space-md);
  border-radius: 4px;
  font-size: 0.875rem;
  z-index: 1000;
  border: 1px solid var(--border-primary);
  box-shadow: var(--shadow-lg);
  backdrop-filter: blur(10px);
}

/* Loading States */
.skeleton {
  background: linear-gradient(90deg, var(--bg-secondary) 25%, var(--bg-tertiary) 50%, var(--bg-secondary) 75%);
  background-size: 200% 100%;
  animation: loading 1.5s infinite;
  border-radius: 4px;
}

@keyframes loading {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}

/* Notifications */
.toast-notification {
  background: var(--bg-overlay) !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border-primary) !important;
  backdrop-filter: blur(10px);
  box-shadow: var(--shadow-lg) !important;
}
```

## Error Handling & User Experience

### Comprehensive Error System

```typescript
// Error handling utilities
export class APIError extends Error {
  constructor(
    public status: number,
    public message: string,
    public code?: string,
    public details?: any
  ) {
    super(message);
    this.name = 'APIError';
  }
}

export class ValidationError extends APIError {
  constructor(public errors: Record<string, string[]>) {
    super(422, 'Validation failed', 'VALIDATION_ERROR', errors);
  }
}

// Error boundary component
export class ErrorBoundary extends React.Component<
  { children: React.ReactNode; fallback?: React.ComponentType<{ error: Error }> },
  { hasError: boolean; error: Error | null }
> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  
  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }
  
  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    // Log error to monitoring service
    logger.error('React Error Boundary caught error:', error, errorInfo);
  }
  
  render() {
    if (this.state.hasError) {
      const FallbackComponent = this.props.fallback || DefaultErrorFallback;
      return <FallbackComponent error={this.state.error!} />;
    }
    
    return this.props.children;
  }
}

function DefaultErrorFallback({ error }: { error: Error }) {
  return (
    <div className="error-boundary">
      <div className="error-content">
        <h2>Something went wrong</h2>
        <p>We're sorry, but something unexpected happened.</p>
        <details className="error-details">
          <summary>Technical Details</summary>
          <pre>{error.message}</pre>
          <pre>{error.stack}</pre>
        </details>
        <div className="error-actions">
          <button onClick={() => window.location.reload()} className="btn btn-primary">
            Reload Page
          </button>
          <button onClick={() => window.history.back()} className="btn btn-secondary">
            Go Back
          </button>
        </div>
      </div>
    </div>
  );
}
```

## Performance Optimizations

### Caching Strategy

- **API Response Caching**: React Query with smart cache invalidation
- **Chart Data Memoization**: Expensive chart calculations cached
- **Virtual Scrolling**: Handle large trade lists efficiently
- **Image Optimization**: Lazy loading for charts and visualizations
- **Bundle Splitting**: Code splitting by route and component

### Real-Time Efficiency

- **WebSocket Connection Pooling**: Reuse connections across components
- **Selective Updates**: Only re-render components with changed data
- **Background Sync**: Update data without blocking UI
- **Progressive Loading**: Load critical data first, details on demand

## Benefits Over V1

### API Improvements
- **10x more endpoints** with comprehensive coverage
- **Real-time capabilities** via WebSocket integration
- **Advanced querying** with filtering and aggregation
- **Better error handling** with actionable error messages
- **Rate limiting** and abuse prevention
- **Comprehensive documentation** with interactive examples

### UI/UX Improvements
- **Professional dark theme** designed for trading
- **Real-time updates** with live backtest progress
- **Advanced charting** with interactive visualizations
- **Responsive design** works on all devices
- **Keyboard shortcuts** for power users
- **Accessibility compliant** with screen reader support

### Developer Experience
- **Type-safe APIs** with comprehensive TypeScript support
- **Component library** for consistent UI
- **Hot module replacement** for instant development feedback
- **Comprehensive testing** with Jest and React Testing Library
- **Storybook integration** for component development
- **Performance monitoring** with React DevTools integration

This V2 API and UI transform the backtesting experience from a simple tool into a professional quantitative research platform that rivals commercial offerings.