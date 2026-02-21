"""Unit tests for the EP-3 backtest execution layer."""
from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl
import pytest

from src.core.execution.data_window import DataWindow, TemporalViolationError
from src.core.execution.portfolio import Portfolio, Position, Fill
from src.core.execution.costs import (
    InteractiveBrokersCommission,
    ZerodhaFlatCommission,
    MarketAwareCostModel,
)
from src.core.execution.fills import FillSimulator
from src.core.execution.pipeline import BacktestPipeline, Middleware, EngineState
from src.core.execution.engine import BacktestEngine
from src.core.markets.registry import MarketCode


# ── Helpers ─────────────────────────────────────────────────────────────


def _make_ohlcv(n: int = 200, base_price: float = 100.0) -> pl.DataFrame:
    """Create a synthetic OHLCV DataFrame with n rows."""
    dates = [str(date(2020, 1, 1) + timedelta(days=i)) for i in range(n)]
    return pl.DataFrame({
        "ts": dates,
        "open": [base_price + i * 0.5 for i in range(n)],
        "high": [base_price + i * 0.5 + 2.0 for i in range(n)],
        "low": [base_price + i * 0.5 - 1.0 for i in range(n)],
        "close": [base_price + i * 0.5 + 1.0 for i in range(n)],
        "volume": [1_000_000] * n,
    })


@dataclass
class Signal:
    action: str  # "buy" | "sell" | "exit" | "hold"


class MockStrategy:
    """A simple strategy that buys on bar 20 and sells on bar 40 (post-warmup)."""

    def __init__(self, buy_idx: int = 20, sell_idx: int = 40, warmup: int = 10):
        self.buy_idx = buy_idx
        self.sell_idx = sell_idx
        self._warmup = warmup

    def get_warmup_periods(self) -> int:
        return self._warmup

    def generate(self, window) -> Signal | None:
        idx = window.current_idx
        if idx == self.buy_idx:
            return Signal(action="buy")
        elif idx == self.sell_idx:
            return Signal(action="sell")
        return Signal(action="hold")


class AlwaysBuyStrategy:
    """Buys on first bar after warmup, never sells."""

    def get_warmup_periods(self) -> int:
        return 5

    def generate(self, window) -> Signal | None:
        if window.current_idx == 5:
            return Signal(action="buy")
        return Signal(action="hold")


# ── DataWindow tests ────────────────────────────────────────────────────


class TestDataWindow:

    def test_current_bar_returns_single_row(self):
        df = _make_ohlcv(100)
        w = DataWindow(df, 50, "AAPL")
        bar = w.current_bar()
        assert bar.shape[0] == 1
        assert bar.row(0, named=True)["ts"] == df.row(50, named=True)["ts"]

    def test_historical_excludes_current_bar(self):
        df = _make_ohlcv(100)
        w = DataWindow(df, 50, "AAPL")
        hist = w.historical()
        assert len(hist) == 50  # bars 0..49
        # Last historical bar should be bar 49, not bar 50
        last_ts = hist.row(-1, named=True)["ts"]
        assert last_ts == df.row(49, named=True)["ts"]

    def test_historical_never_includes_future(self):
        df = _make_ohlcv(100)
        w = DataWindow(df, 10, "AAPL")
        hist = w.historical()
        assert len(hist) == 10  # bars 0..9
        # Verify none of the future timestamps are present
        future_ts = set(df.slice(11)["ts"].to_list())
        hist_ts = set(hist["ts"].to_list())
        assert hist_ts.isdisjoint(future_ts)

    def test_historical_with_n_returns_last_n(self):
        df = _make_ohlcv(100)
        w = DataWindow(df, 50, "AAPL")
        hist = w.historical(n=5)
        assert len(hist) == 5
        # Should be bars 45..49
        assert hist.row(0, named=True)["ts"] == df.row(45, named=True)["ts"]
        assert hist.row(-1, named=True)["ts"] == df.row(49, named=True)["ts"]

    def test_indicators_includes_current_bar(self):
        df = _make_ohlcv(100)
        w = DataWindow(df, 50, "AAPL")
        ind = w.indicators()
        assert len(ind) == 51  # bars 0..50

    def test_symbol_property(self):
        df = _make_ohlcv(10)
        w = DataWindow(df, 5, "TSLA")
        assert w.symbol == "TSLA"

    def test_len(self):
        df = _make_ohlcv(100)
        w = DataWindow(df, 30, "AAPL")
        assert len(w) == 31  # bars 0..30

    def test_current_idx_at_start(self):
        df = _make_ohlcv(100)
        w = DataWindow(df, 0, "AAPL")
        assert w.current_bar().shape[0] == 1
        assert len(w.historical()) == 0

    def test_current_idx_at_end(self):
        df = _make_ohlcv(100)
        w = DataWindow(df, 99, "AAPL")
        assert len(w.historical()) == 99
        assert len(w) == 100


# ── Portfolio tests ─────────────────────────────────────────────────────


class TestPortfolio:

    def test_initial_state(self):
        p = Portfolio(100_000.0)
        assert p.cash == 100_000.0
        assert p.equity == 100_000.0
        assert p.initial_capital == 100_000.0
        assert len(p.positions) == 0
        assert len(p.fills) == 0

    def test_buy_reduces_cash(self):
        p = Portfolio(100_000.0)
        fill = Fill(symbol="AAPL", side="buy", quantity=100, price=150.0, commission=1.0)
        p.apply_fill(fill)
        # Cash = 100000 - (100*150 + 1.0) = 84999.0
        assert p.cash == pytest.approx(84_999.0)
        assert "AAPL" in p.positions
        assert p.positions["AAPL"].quantity == 100
        assert p.positions["AAPL"].avg_cost == 150.0

    def test_sell_adds_cash_and_computes_realized_pnl(self):
        p = Portfolio(100_000.0)
        p.apply_fill(Fill(symbol="AAPL", side="buy", quantity=100, price=150.0, commission=1.0))
        realized = p.apply_fill(
            Fill(symbol="AAPL", side="sell", quantity=100, price=160.0, commission=1.0)
        )
        # realized = (100*160) - (100*150) - 1.0 = 999.0
        assert realized == pytest.approx(999.0)
        assert "AAPL" not in p.positions

    def test_sell_with_india_costs(self):
        p = Portfolio(1_000_000.0)
        p.apply_fill(Fill(
            symbol="RELIANCE.NS", side="buy", quantity=50, price=2500.0,
            commission=20.0, stt=125.0, gst=3.6, stamp_duty=18.75,
        ))
        realized = p.apply_fill(Fill(
            symbol="RELIANCE.NS", side="sell", quantity=50, price=2600.0,
            commission=20.0, stt=130.0, gst=3.6, stamp_duty=0.0,
        ))
        # Buy cost: 50*2500 + 20 + 125 + 3.6 + 18.75 = 125167.35
        # Sell: 50*2600 = 130000, costs = 153.6
        # realized = 130000 - 125000 - 153.6 = 4846.4
        assert realized == pytest.approx(4846.4)

    def test_add_to_existing_position(self):
        p = Portfolio(200_000.0)
        p.apply_fill(Fill(symbol="AAPL", side="buy", quantity=100, price=150.0, commission=1.0))
        p.apply_fill(Fill(symbol="AAPL", side="buy", quantity=100, price=160.0, commission=1.0))
        assert p.positions["AAPL"].quantity == 200
        assert p.positions["AAPL"].avg_cost == pytest.approx(155.0)

    def test_partial_sell(self):
        p = Portfolio(100_000.0)
        p.apply_fill(Fill(symbol="AAPL", side="buy", quantity=100, price=150.0, commission=1.0))
        p.apply_fill(Fill(symbol="AAPL", side="sell", quantity=50, price=160.0, commission=1.0))
        assert p.positions["AAPL"].quantity == 50

    def test_mark_to_market(self):
        p = Portfolio(100_000.0)
        p.apply_fill(Fill(symbol="AAPL", side="buy", quantity=100, price=150.0, commission=1.0))
        p.update_market_values({"AAPL": 155.0}, "2024-01-02")
        assert p.positions["AAPL"].market_value == pytest.approx(15500.0)
        assert p.positions["AAPL"].unrealized_pnl == pytest.approx(500.0)
        # equity = cash + positions_value
        expected_cash = 100_000.0 - (100 * 150.0 + 1.0)  # 84999.0
        assert p.equity == pytest.approx(expected_cash + 15500.0)

    def test_equity_history_tracks_snapshots(self):
        p = Portfolio(100_000.0)
        p.update_market_values({}, "2024-01-01")
        p.update_market_values({}, "2024-01-02")
        assert len(p.equity_history) == 2
        assert p.equity_history[0]["date"] == "2024-01-01"
        assert p.equity_history[1]["date"] == "2024-01-02"

    def test_gross_exposure(self):
        p = Portfolio(100_000.0)
        p.apply_fill(Fill(symbol="AAPL", side="buy", quantity=100, price=150.0, commission=0.0))
        p.update_market_values({"AAPL": 160.0}, "2024-01-01")
        assert p.gross_exposure == pytest.approx(16_000.0)

    def test_exposure_pct(self):
        p = Portfolio(100_000.0)
        p.apply_fill(Fill(symbol="AAPL", side="buy", quantity=100, price=150.0, commission=0.0))
        p.update_market_values({"AAPL": 150.0}, "2024-01-01")
        # exposure = 15000 / 100000 * 100 = 15%
        assert p.exposure_pct == pytest.approx(15.0)

    def test_fill_total_cost(self):
        fill = Fill(
            symbol="X", side="buy", quantity=1, price=100.0,
            commission=20.0, stt=10.0, gst=3.6, stamp_duty=1.5,
        )
        assert fill.total_cost == pytest.approx(35.1)


# ── Cost model tests ────────────────────────────────────────────────────


class TestInteractiveBrokersCommission:

    def test_minimum_commission(self):
        c = InteractiveBrokersCommission()
        # 10 shares * $0.005 = $0.05, minimum is $1.00
        assert c.calculate(10, 100.0, "buy") == 1.0

    def test_above_minimum(self):
        c = InteractiveBrokersCommission()
        # 1000 shares * $0.005 = $5.00
        assert c.calculate(1000, 100.0, "buy") == 5.0


class TestZerodhaFlatCommission:

    def test_flat_20_cap(self):
        c = ZerodhaFlatCommission()
        # Large trade: 100 * 2500 = 250000 * 0.0003 = 75 → capped at 20
        assert c.calculate(100, 2500.0, "buy") == 20.0

    def test_percentage_lower_than_flat(self):
        c = ZerodhaFlatCommission()
        # Small trade: 10 * 100 = 1000 * 0.0003 = 0.30
        assert c.calculate(10, 100.0, "buy") == pytest.approx(0.30)


class TestMarketAwareCostModel:

    def test_us_trade_no_stt(self):
        model = MarketAwareCostModel()
        costs = model.calculate_total(100, 150.0, "buy", MarketCode.US)
        assert costs["stt"] == 0.0
        assert costs["gst"] == 0.0
        assert costs["stamp_duty"] == 0.0
        assert costs["commission"] > 0.0
        assert costs["total"] == costs["commission"]

    def test_india_buy_has_stt_gst_stamp_duty(self):
        model = MarketAwareCostModel()
        costs = model.calculate_total(50, 2500.0, "buy", MarketCode.IN)
        trade_value = 50 * 2500.0  # 125000

        # STT: 0.1% of trade value = 125
        assert costs["stt"] == pytest.approx(trade_value * 0.001)
        # Commission: min(20, 125000 * 0.0003) = min(20, 37.5) = 20
        assert costs["commission"] == pytest.approx(20.0)
        # GST: 18% of commission = 3.6
        assert costs["gst"] == pytest.approx(20.0 * 0.18)
        # Stamp duty: 0.015% of trade value = 18.75
        assert costs["stamp_duty"] == pytest.approx(trade_value * 0.00015)
        # Total
        expected_total = 20.0 + 125.0 + 3.6 + 18.75
        assert costs["total"] == pytest.approx(expected_total)

    def test_india_sell_no_stamp_duty(self):
        model = MarketAwareCostModel()
        costs = model.calculate_total(50, 2500.0, "sell", MarketCode.IN)
        # Stamp duty is only on buy side
        assert costs["stamp_duty"] == 0.0
        # STT still applies
        assert costs["stt"] > 0.0


# ── Fill simulator tests ────────────────────────────────────────────────


class TestFillSimulator:

    def test_buy_slippage_increases_price(self):
        sim = FillSimulator(slippage_pct=0.05)
        fill_price = sim.simulate_fill(100.0, "buy", 100)
        assert fill_price > 100.0

    def test_sell_slippage_decreases_price(self):
        sim = FillSimulator(slippage_pct=0.05)
        fill_price = sim.simulate_fill(100.0, "sell", 100)
        assert fill_price < 100.0

    def test_zero_slippage(self):
        sim = FillSimulator(slippage_pct=0.0)
        assert sim.simulate_fill(100.0, "buy", 100) == pytest.approx(100.0)
        assert sim.simulate_fill(100.0, "sell", 100) == pytest.approx(100.0)

    def test_large_order_extra_impact(self):
        sim = FillSimulator(slippage_pct=0.05)
        # Normal order: 100 shares vs 1M avg volume (0.01%)
        normal_price = sim.simulate_fill(100.0, "buy", 100, avg_volume=1_000_000)
        # Large order: 50,000 shares vs 1M avg volume (5%)
        large_price = sim.simulate_fill(100.0, "buy", 50_000, avg_volume=1_000_000)
        assert large_price > normal_price

    def test_cover_applies_buy_side_slippage(self):
        sim = FillSimulator(slippage_pct=0.05)
        fill_price = sim.simulate_fill(100.0, "cover", 100)
        assert fill_price > 100.0

    def test_short_applies_sell_side_slippage(self):
        sim = FillSimulator(slippage_pct=0.05)
        fill_price = sim.simulate_fill(100.0, "short", 100)
        assert fill_price < 100.0


# ── Pipeline tests ──────────────────────────────────────────────────────


class TestBacktestPipeline:

    def test_middlewares_run_in_order(self):
        call_order = []

        class MW1(Middleware):
            def process(self, state):
                call_order.append("mw1")
                state.extra["mw1"] = True
                return state

        class MW2(Middleware):
            def process(self, state):
                call_order.append("mw2")
                state.extra["mw2"] = True
                return state

        df = _make_ohlcv(10)
        window = DataWindow(df, 5, "AAPL")
        portfolio = Portfolio(100_000.0)
        state = EngineState(window=window, portfolio=portfolio)

        pipeline = BacktestPipeline([MW1(), MW2()])
        result = pipeline.run(state)

        assert call_order == ["mw1", "mw2"]
        assert result.extra["mw1"] is True
        assert result.extra["mw2"] is True

    def test_empty_pipeline(self):
        df = _make_ohlcv(10)
        window = DataWindow(df, 5, "AAPL")
        portfolio = Portfolio(100_000.0)
        state = EngineState(window=window, portfolio=portfolio)

        pipeline = BacktestPipeline([])
        result = pipeline.run(state)
        assert result is state


# ── BacktestEngine tests ────────────────────────────────────────────────


class TestBacktestEngine:

    def test_basic_buy_sell_roundtrip(self):
        """Buy at bar 20, sell at bar 40 — verify fills happen at next bar's open."""
        df = _make_ohlcv(100)
        strategy = MockStrategy(buy_idx=20, sell_idx=40, warmup=10)
        engine = BacktestEngine(
            market=MarketCode.US,
            initial_capital=100_000.0,
            slippage_pct=0.0,  # no slippage for deterministic test
        )
        result = engine.run(df, strategy, "AAPL")

        assert result["total_trades"] == 2
        trades = result["trades"]

        # Buy trade: signal at bar 20, fill at bar 21 open
        buy_trade = trades[0]
        assert buy_trade["side"] == "buy"
        bar_21_open = df.row(21, named=True)["open"]
        assert buy_trade["price"] == pytest.approx(bar_21_open)

        # Sell trade: signal at bar 40, fill at bar 41 open
        sell_trade = trades[1]
        assert sell_trade["side"] == "sell"
        bar_41_open = df.row(41, named=True)["open"]
        assert sell_trade["price"] == pytest.approx(bar_41_open)

    def test_no_fill_at_signal_bar(self):
        """Signal fires on bar N close. Fill must NOT happen on bar N."""
        df = _make_ohlcv(100)
        strategy = MockStrategy(buy_idx=20, sell_idx=40, warmup=10)
        engine = BacktestEngine(
            market=MarketCode.US,
            initial_capital=100_000.0,
            slippage_pct=0.0,
        )
        result = engine.run(df, strategy, "AAPL")
        trades = result["trades"]

        buy_trade = trades[0]
        signal_bar_ts = df.row(20, named=True)["ts"]
        # Buy should happen at bar 21, not bar 20
        assert buy_trade["date"] != signal_bar_ts

    def test_equity_curve_has_entries(self):
        df = _make_ohlcv(50)
        strategy = MockStrategy(buy_idx=15, sell_idx=30, warmup=10)
        engine = BacktestEngine(
            market=MarketCode.US,
            initial_capital=100_000.0,
        )
        result = engine.run(df, strategy, "AAPL")
        # One equity entry per bar from warmup to end
        assert len(result["equity_curve"]) == 50 - 10

    def test_final_equity_after_profitable_trade(self):
        """In an uptrending market, buy low and sell high should be profitable."""
        df = _make_ohlcv(100, base_price=100.0)
        strategy = MockStrategy(buy_idx=15, sell_idx=50, warmup=10)
        engine = BacktestEngine(
            market=MarketCode.US,
            initial_capital=100_000.0,
            slippage_pct=0.0,
        )
        result = engine.run(df, strategy, "AAPL")
        # Market trends up, so trade should be profitable
        assert result["final_equity"] > 100_000.0
        assert result["total_return_pct"] > 0.0

    def test_no_trade_when_no_signal(self):
        """A strategy that never signals should produce no trades."""

        class NoSignalStrategy:
            def get_warmup_periods(self):
                return 5

            def generate(self, window):
                return Signal(action="hold")

        df = _make_ohlcv(50)
        engine = BacktestEngine(
            market=MarketCode.US,
            initial_capital=100_000.0,
        )
        result = engine.run(df, NoSignalStrategy(), "AAPL")
        assert result["total_trades"] == 0
        assert result["final_equity"] == pytest.approx(100_000.0)

    def test_stop_loss_triggers(self):
        """Stop-loss should sell when bar low breaches stop price."""
        # Create data where price drops sharply
        n = 50
        dates = [str(date(2020, 1, 1) + timedelta(days=i)) for i in range(n)]
        prices_up = [100.0 + i for i in range(20)]
        prices_down = [120.0 - i * 3 for i in range(30)]
        all_prices = prices_up + prices_down

        df = pl.DataFrame({
            "ts": dates,
            "open": all_prices,
            "high": [p + 2.0 for p in all_prices],
            "low": [p - 2.0 for p in all_prices],
            "close": [p + 1.0 for p in all_prices],
            "volume": [1_000_000] * n,
        })

        # Buy at bar 12, 10% stop loss
        strategy = MockStrategy(buy_idx=12, sell_idx=999, warmup=10)
        engine = BacktestEngine(
            market=MarketCode.US,
            initial_capital=100_000.0,
            stop_loss_pct=10.0,
            slippage_pct=0.0,
        )
        result = engine.run(df, strategy, "TEST")
        trades = result["trades"]

        # Should have a buy and a stop-loss sell
        assert len(trades) >= 2
        sell_trade = trades[-1]
        assert sell_trade.get("reason", sell_trade["side"]) in ("stop_loss", "sell")

    def test_india_market_costs_in_trades(self):
        """Indian market trades should include STT, GST, stamp duty."""
        df = _make_ohlcv(100)
        strategy = MockStrategy(buy_idx=20, sell_idx=40, warmup=10)
        engine = BacktestEngine(
            market=MarketCode.IN,
            initial_capital=1_000_000.0,
            slippage_pct=0.0,
        )
        result = engine.run(df, strategy, "RELIANCE.NS")
        trades = result["trades"]

        buy_trade = trades[0]
        assert buy_trade["stt"] > 0.0
        assert buy_trade["gst"] > 0.0
        assert buy_trade["stamp_duty"] > 0.0

    def test_progress_callback_called(self):
        df = _make_ohlcv(100)
        strategy = MockStrategy(buy_idx=20, sell_idx=40, warmup=10)
        progress_calls = []

        def callback(pct, sym):
            progress_calls.append((pct, sym))

        engine = BacktestEngine(
            market=MarketCode.US,
            initial_capital=100_000.0,
            progress_callback=callback,
        )
        engine.run(df, strategy, "AAPL")

        assert len(progress_calls) > 0
        # Last call should be 100%
        assert progress_calls[-1][0] == pytest.approx(100.0)
        assert progress_calls[-1][1] == "AAPL"

    def test_warmup_bars_skipped(self):
        """Engine should not process bars before warmup period."""
        df = _make_ohlcv(50)

        class EarlyBuyStrategy:
            def get_warmup_periods(self):
                return 20

            def generate(self, window):
                # Try to buy immediately
                return Signal(action="buy")

        engine = BacktestEngine(
            market=MarketCode.US,
            initial_capital=100_000.0,
            slippage_pct=0.0,
        )
        result = engine.run(df, EarlyBuyStrategy(), "AAPL")
        # Buy signal fires at bar 20, fill at bar 21
        if result["trades"]:
            buy_trade = result["trades"][0]
            buy_date = buy_trade["date"]
            # Bar 21's ts
            assert buy_date == df.row(21, named=True)["ts"]

    def test_cannot_buy_twice(self):
        """Engine should not buy if already holding a position."""

        class AlwaysBuy:
            def get_warmup_periods(self):
                return 5

            def generate(self, window):
                return Signal(action="buy")

        df = _make_ohlcv(50)
        engine = BacktestEngine(
            market=MarketCode.US,
            initial_capital=100_000.0,
            slippage_pct=0.0,
        )
        result = engine.run(df, AlwaysBuy(), "AAPL")
        # Should only have 1 buy trade despite constant buy signals
        buy_trades = [t for t in result["trades"] if t["side"] == "buy"]
        assert len(buy_trades) == 1

    def test_pnl_matches_manual_calculation(self):
        """Verify P&L matches hand-calculated values for a simple trade."""
        df = _make_ohlcv(100, base_price=100.0)
        strategy = MockStrategy(buy_idx=20, sell_idx=40, warmup=10)
        engine = BacktestEngine(
            market=MarketCode.US,
            initial_capital=100_000.0,
            slippage_pct=0.0,
        )
        result = engine.run(df, strategy, "AAPL")
        trades = result["trades"]

        buy_price = trades[0]["price"]
        buy_qty = trades[0]["quantity"]
        sell_price = trades[1]["price"]
        sell_qty = trades[1]["quantity"]

        # Both trades have IBKR commission
        buy_commission = max(1.0, buy_qty * 0.005)
        sell_commission = max(1.0, sell_qty * 0.005)

        expected_pnl = (sell_price - buy_price) * sell_qty - buy_commission - sell_commission
        actual_pnl = trades[1]["realized_pnl"]

        # Small tolerance for floating point
        assert actual_pnl == pytest.approx(expected_pnl, rel=0.01)
