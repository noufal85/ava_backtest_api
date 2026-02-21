"""
BacktestEngine — bar-loop execution.
Signal fires on bar N close → execute at bar N+1 open.
"""
import math
from typing import Callable

import polars as pl
import structlog

from src.core.markets.registry import MarketCode
from src.core.execution.portfolio import Portfolio, Fill
from src.core.execution.costs import MarketAwareCostModel
from src.core.execution.fills import FillSimulator
from src.core.execution.data_window import DataWindow

logger = structlog.get_logger()


class BacktestEngine:
    def __init__(
        self,
        market: MarketCode,
        initial_capital: float,
        position_size_pct: float = 0.95,
        slippage_pct: float = 0.05,
        stop_loss_pct: float = 0.0,
        progress_callback: Callable[[float, str], None] | None = None,
    ):
        self.market = market
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.cost_model = MarketAwareCostModel()
        self.fill_sim = FillSimulator(slippage_pct=slippage_pct)
        self.stop_loss_pct = stop_loss_pct
        self.progress_callback = progress_callback

    def run(self, df: pl.DataFrame, strategy, symbol: str) -> dict:
        """
        Run a backtest for a single symbol.
        Returns dict with: trades, equity_curve, final_equity, metrics_raw
        """
        portfolio = Portfolio(self.initial_capital)
        trades: list[dict] = []
        pending_signal: str | None = None
        entry_price: float | None = None
        entry_date: str | None = None
        stop_price: float | None = None
        warmup = strategy.get_warmup_periods()

        total_bars = len(df)
        progress_step = max(1, math.ceil(total_bars * 0.05))

        for i in range(warmup, total_bars):
            window = DataWindow(df, i, symbol)
            bar = window.current_bar().row(0, named=True)
            bar_open = bar["open"]
            bar_close = bar["close"]
            bar_low = bar["low"]
            bar_date = str(bar.get("ts", ""))

            prices = {symbol: bar_close}

            # ── STEP 1: Execute pending signal from previous bar at today's open ──
            if pending_signal == "buy" and symbol not in portfolio.positions:
                allocatable = portfolio.cash * self.position_size_pct
                fill_price = self.fill_sim.simulate_fill(bar_open, "buy", 1)
                quantity = int(allocatable / fill_price) if fill_price > 0 else 0
                if quantity > 0:
                    costs = self.cost_model.calculate_total(quantity, fill_price, "buy", self.market)
                    fill = Fill(
                        symbol=symbol,
                        side="buy",
                        quantity=quantity,
                        price=fill_price,
                        commission=costs["commission"],
                        stt=costs["stt"],
                        gst=costs["gst"],
                        stamp_duty=costs["stamp_duty"],
                        date=bar_date,
                        reason="signal",
                    )
                    portfolio.apply_fill(fill)
                    entry_price = fill_price
                    entry_date = bar_date
                    if self.stop_loss_pct > 0:
                        stop_price = fill_price * (1 - self.stop_loss_pct / 100)
                    trades.append({
                        "symbol": symbol,
                        "side": "buy",
                        "date": bar_date,
                        "price": fill_price,
                        "quantity": quantity,
                        "commission": costs["commission"],
                        "stt": costs["stt"],
                        "gst": costs["gst"],
                        "stamp_duty": costs["stamp_duty"],
                    })
                pending_signal = None

            elif pending_signal in ("sell", "exit") and symbol in portfolio.positions:
                pos = portfolio.positions[symbol]
                fill_price = self.fill_sim.simulate_fill(bar_open, "sell", pos.quantity)
                costs = self.cost_model.calculate_total(
                    pos.quantity, fill_price, "sell", self.market
                )
                fill = Fill(
                    symbol=symbol,
                    side="sell",
                    quantity=pos.quantity,
                    price=fill_price,
                    commission=costs["commission"],
                    stt=costs["stt"],
                    gst=costs["gst"],
                    stamp_duty=costs["stamp_duty"],
                    date=bar_date,
                    reason="signal",
                )
                realized = portfolio.apply_fill(fill)
                trades.append({
                    "symbol": symbol,
                    "side": "sell",
                    "date": bar_date,
                    "price": fill_price,
                    "quantity": fill.quantity,
                    "commission": costs["commission"],
                    "stt": costs["stt"],
                    "gst": costs["gst"],
                    "stamp_duty": costs["stamp_duty"],
                    "realized_pnl": realized,
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                })
                entry_price = None
                entry_date = None
                stop_price = None
                pending_signal = None

            # ── STEP 2: Check stop-loss on current bar ──
            if (
                self.stop_loss_pct > 0
                and stop_price is not None
                and symbol in portfolio.positions
                and bar_low <= stop_price
            ):
                pos = portfolio.positions[symbol]
                fill_price = stop_price  # fill at stop price
                costs = self.cost_model.calculate_total(
                    pos.quantity, fill_price, "sell", self.market
                )
                fill = Fill(
                    symbol=symbol,
                    side="sell",
                    quantity=pos.quantity,
                    price=fill_price,
                    commission=costs["commission"],
                    stt=costs["stt"],
                    gst=costs["gst"],
                    stamp_duty=costs["stamp_duty"],
                    date=bar_date,
                    reason="stop_loss",
                )
                realized = portfolio.apply_fill(fill)
                trades.append({
                    "symbol": symbol,
                    "side": "sell",
                    "date": bar_date,
                    "price": fill_price,
                    "quantity": fill.quantity,
                    "commission": costs["commission"],
                    "stt": costs["stt"],
                    "gst": costs["gst"],
                    "stamp_duty": costs["stamp_duty"],
                    "realized_pnl": realized,
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "reason": "stop_loss",
                })
                entry_price = None
                entry_date = None
                stop_price = None
                pending_signal = None

            # ── STEP 3: Mark-to-market ──
            portfolio.update_market_values(prices, bar_date)

            # ── STEP 4: Generate new signal from strategy ──
            signal = strategy.generate(window)
            if signal and signal.action in ("buy", "sell", "exit"):
                pending_signal = signal.action

            # ── STEP 5: Progress callback ──
            if self.progress_callback and (i - warmup) % progress_step == 0:
                pct = (i - warmup + 1) / (total_bars - warmup) * 100
                self.progress_callback(pct, symbol)

        # Final progress
        if self.progress_callback:
            self.progress_callback(100.0, symbol)

        return {
            "trades": trades,
            "equity_curve": portfolio.equity_history,
            "final_equity": portfolio.equity,
            "total_return_pct": (portfolio.equity / self.initial_capital - 1) * 100,
            "total_trades": len(trades),
            "fills": portfolio.fills,
            "portfolio": portfolio,
        }
