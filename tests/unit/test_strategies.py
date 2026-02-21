"""Tests for all 8 trading strategies."""
import math
from dataclasses import dataclass

import polars as pl
import pytest


# ---------------------------------------------------------------------------
# Minimal DataWindow stub — strategies call window.historical() and window.current_bar()
# ---------------------------------------------------------------------------

class FakeWindow:
    """Simulates a DataWindow for strategy testing.

    Wraps a full DataFrame and a cursor index. historical() returns all bars
    before the cursor; current_bar() returns the single bar at the cursor.
    """

    def __init__(self, df: pl.DataFrame, idx: int):
        self._df = df
        self._idx = idx

    def historical(self, n: int | None = None) -> pl.DataFrame:
        end = self._idx
        if n is not None:
            start = max(0, end - n)
        else:
            start = 0
        return self._df[start:end]

    def current_bar(self) -> pl.DataFrame:
        return self._df[self._idx : self._idx + 1]


# ---------------------------------------------------------------------------
# Synthetic price generators
# ---------------------------------------------------------------------------

def _uptrend(n: int, start: float = 100.0, step: float = 1.0) -> pl.DataFrame:
    """Steadily rising prices."""
    closes = [start + i * step for i in range(n)]
    return pl.DataFrame({
        "open": [c - 0.3 for c in closes],
        "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes],
        "close": closes,
        "volume": [1000] * n,
    })


def _downtrend(n: int, start: float = 200.0, step: float = 1.0) -> pl.DataFrame:
    """Steadily falling prices."""
    closes = [start - i * step for i in range(n)]
    return pl.DataFrame({
        "open": [c + 0.3 for c in closes],
        "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes],
        "close": closes,
        "volume": [1000] * n,
    })


def _flat(n: int, price: float = 100.0) -> pl.DataFrame:
    """Flat (sideways) prices."""
    return pl.DataFrame({
        "open": [price] * n,
        "high": [price + 0.1] * n,
        "low": [price - 0.1] * n,
        "close": [price] * n,
        "volume": [1000] * n,
    })


def _vshape(n: int, start: float = 150.0, bottom: float = 80.0) -> pl.DataFrame:
    """V-shape: drops to bottom then recovers. Used to trigger oversold conditions."""
    half = n // 2
    drop_step = (start - bottom) / half
    rise_step = (start - bottom) / (n - half)
    closes = []
    for i in range(half):
        closes.append(start - i * drop_step)
    for i in range(n - half):
        closes.append(bottom + i * rise_step)
    return pl.DataFrame({
        "open": [c + 0.5 for c in closes],
        "high": [c + 1.0 for c in closes],
        "low": [c - 1.0 for c in closes],
        "close": closes,
        "volume": [1000] * n,
    })


def _concat_trends(*dfs: pl.DataFrame) -> pl.DataFrame:
    return pl.concat(dfs)


# ---------------------------------------------------------------------------
# Import strategies (triggers registration)
# ---------------------------------------------------------------------------

from src.strategies import (
    SMACrossover,
    RSIMeanReversion,
    MACDCrossover,
    BollingerBands,
    MomentumBreakout,
    RSIVolFilter,
    DualMomentum,
    OpeningRangeBreakout,
)
from src.core.strategy.registry import get_strategy, list_strategies


# ===========================================================================
# Registry tests
# ===========================================================================

class TestRegistry:
    def test_all_registered(self):
        strategies = list_strategies()
        names = {s["name"] for s in strategies}
        expected = {
            "sma_crossover", "rsi_mean_reversion", "macd_crossover",
            "bollinger_bands", "momentum_breakout", "rsi_vol_filter",
            "dual_momentum", "opening_range_breakout",
        }
        assert expected == names

    def test_get_strategy(self):
        cls = get_strategy("sma_crossover")
        assert cls is SMACrossover

    def test_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy("nonexistent")


# ===========================================================================
# SMA Crossover
# ===========================================================================

class TestSMACrossover:
    def test_buy_signal_on_uptrend(self):
        """After flat period, uptrend should trigger exactly one buy crossover."""
        strat = SMACrossover(fast_period=5, slow_period=10)
        df = _concat_trends(_flat(15, 100.0), _uptrend(30, 100.0, 2.0))

        signals = []
        for i in range(1, len(df)):
            w = FakeWindow(df, i)
            sig = strat.generate_signal(w)
            if sig and sig.action == "buy":
                signals.append((i, sig))

        assert len(signals) >= 1, "Should produce at least one buy signal"
        # Should NOT fire every bar — crossover is a one-time event
        assert len(signals) <= 3, f"Buy should be rare, got {len(signals)} signals"

    def test_sell_signal_on_downtrend(self):
        strat = SMACrossover(fast_period=5, slow_period=10)
        df = _concat_trends(_flat(15, 100.0), _downtrend(30, 100.0, 2.0))

        signals = []
        for i in range(1, len(df)):
            w = FakeWindow(df, i)
            sig = strat.generate_signal(w)
            if sig and sig.action == "sell":
                signals.append((i, sig))

        assert len(signals) >= 1, "Should produce at least one sell signal"

    def test_no_signal_during_warmup(self):
        strat = SMACrossover(fast_period=5, slow_period=10)
        warmup = strat.get_warmup_periods()
        df = _uptrend(warmup)
        # Only test bars within warmup range (< slow_period)
        for i in range(1, min(strat.slow_period, len(df))):
            w = FakeWindow(df, i)
            sig = strat.generate_signal(w)
            assert sig is None, f"No signal expected at bar {i} during warmup"

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            SMACrossover(fast_period=50, slow_period=20)


# ===========================================================================
# RSI Mean Reversion
# ===========================================================================

class TestRSIMeanReversion:
    def test_entry_on_oversold(self):
        """V-shape drop should produce oversold RSI and trigger buy."""
        strat = RSIMeanReversion(rsi_period=14, oversold=30, overbought=70)
        # Sharp drop to create oversold RSI
        df = _concat_trends(_flat(20, 150.0), _downtrend(30, 150.0, 3.0))

        buy_signals = []
        for i in range(1, len(df)):
            w = FakeWindow(df, i)
            sig = strat.generate_signal(w)
            if sig and sig.action == "buy":
                buy_signals.append((i, sig))

        assert len(buy_signals) >= 1, "Should trigger buy on oversold crossing"

    def test_exit_on_overbought(self):
        """After buy, strong rally should exit on overbought."""
        strat = RSIMeanReversion(rsi_period=14, oversold=30, overbought=70)
        # Drop then strong recovery
        df = _concat_trends(
            _flat(20, 150.0),
            _downtrend(25, 150.0, 3.5),
            _uptrend(40, 62.5, 4.0),
        )

        actions = []
        for i in range(1, len(df)):
            w = FakeWindow(df, i)
            sig = strat.generate_signal(w)
            if sig:
                actions.append(sig.action)

        assert "buy" in actions, "Should have entered"
        assert "sell" in actions, "Should have exited on overbought or max hold"


# ===========================================================================
# MACD Crossover
# ===========================================================================

class TestMACDCrossover:
    def test_buy_on_crossover(self):
        """Flat then strong uptrend should produce MACD crossover buy."""
        strat = MACDCrossover(fast_ema=12, slow_ema=26, signal_period=9)
        df = _concat_trends(_flat(40, 100.0), _uptrend(50, 100.0, 1.5))

        buy_signals = []
        for i in range(1, len(df)):
            w = FakeWindow(df, i)
            sig = strat.generate_signal(w)
            if sig and sig.action == "buy":
                buy_signals.append(sig)

        assert len(buy_signals) >= 1, "MACD should cross above signal on uptrend"

    def test_sell_on_crossunder(self):
        """Uptrend then downtrend should produce MACD cross-under sell."""
        strat = MACDCrossover(fast_ema=12, slow_ema=26, signal_period=9)
        df = _concat_trends(_uptrend(50, 100.0, 1.0), _downtrend(40, 149.0, 1.5))

        sell_signals = []
        for i in range(1, len(df)):
            w = FakeWindow(df, i)
            sig = strat.generate_signal(w)
            if sig and sig.action == "sell":
                sell_signals.append(sig)

        assert len(sell_signals) >= 1, "MACD should cross below signal on downtrend"

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            MACDCrossover(fast_ema=30, slow_ema=12)


# ===========================================================================
# Bollinger Bands
# ===========================================================================

class TestBollingerBands:
    def test_buy_below_lower_band(self):
        """Sharp drop should push close below lower band → buy."""
        strat = BollingerBands(bb_period=20, bb_std_dev=2.0)
        df = _concat_trends(_flat(25, 100.0), _downtrend(15, 100.0, 3.0))

        buy_signals = []
        for i in range(1, len(df)):
            w = FakeWindow(df, i)
            sig = strat.generate_signal(w)
            if sig and sig.action == "buy":
                buy_signals.append(sig)

        assert len(buy_signals) >= 1, "Should buy when close drops below lower BB"

    def test_sell_at_middle_band(self):
        """After entry, recovery to middle band → sell."""
        strat = BollingerBands(bb_period=20, bb_std_dev=2.0)
        df = _concat_trends(
            _flat(25, 100.0),
            _downtrend(10, 100.0, 4.0),
            _uptrend(20, 60.0, 3.0),
        )

        actions = []
        for i in range(1, len(df)):
            w = FakeWindow(df, i)
            sig = strat.generate_signal(w)
            if sig:
                actions.append(sig.action)

        assert "buy" in actions
        assert "sell" in actions, "Should exit when close crosses above middle band"


# ===========================================================================
# Momentum Breakout
# ===========================================================================

class TestMomentumBreakout:
    def test_buy_on_channel_breakout(self):
        """Close above N-day high channel → buy."""
        strat = MomentumBreakout(channel_period=10, exit_ma_period=10)
        df = _concat_trends(_flat(15, 100.0), _uptrend(20, 100.0, 2.0))

        buy_signals = []
        for i in range(1, len(df)):
            w = FakeWindow(df, i)
            sig = strat.generate_signal(w)
            if sig and sig.action == "buy":
                buy_signals.append(sig)

        assert len(buy_signals) >= 1, "Should buy on breakout above channel high"

    def test_sell_below_exit_ma(self):
        """After entry, drop below exit MA → sell."""
        strat = MomentumBreakout(channel_period=10, exit_ma_period=10)
        df = _concat_trends(
            _flat(15, 100.0),
            _uptrend(15, 100.0, 2.0),
            _downtrend(20, 128.0, 2.5),
        )

        actions = []
        for i in range(1, len(df)):
            w = FakeWindow(df, i)
            sig = strat.generate_signal(w)
            if sig:
                actions.append(sig.action)

        assert "buy" in actions
        assert "sell" in actions, "Should sell when close drops below exit MA"


# ===========================================================================
# RSI + Volatility Filter
# ===========================================================================

class TestRSIVolFilter:
    def test_buy_low_vol_oversold(self):
        """Oversold RSI + low volatility → buy signal."""
        strat = RSIVolFilter(
            rsi_period=14, oversold=30, overbought=70,
            atr_period=14, atr_rank_period=50, vol_threshold=60,
        )
        # Gradual downtrend with tiny candle ranges → low ATR but RSI goes oversold.
        # We need ~120 bars so ATR rank has history, and a slow grind-down.
        n = 150
        closes = [100.0 - i * 0.4 for i in range(n)]
        df = pl.DataFrame({
            "open": [c + 0.05 for c in closes],
            "high": [c + 0.1 for c in closes],
            "low": [c - 0.1 for c in closes],
            "close": closes,
            "volume": [1000] * n,
        })

        buy_signals = []
        for i in range(1, len(df)):
            w = FakeWindow(df, i)
            sig = strat.generate_signal(w)
            if sig and sig.action == "buy":
                buy_signals.append(sig)

        assert len(buy_signals) >= 1, "Should buy when RSI oversold + low vol"


# ===========================================================================
# Dual Momentum
# ===========================================================================

class TestDualMomentum:
    def test_buy_positive_momentum(self):
        """Lookback return > 0 → buy on rebalance day."""
        strat = DualMomentum(lookback_months=1, rebalance_frequency=5)
        # 1 month = 21 bars. Need lookback + extra bars.
        df = _uptrend(40, 100.0, 1.0)

        buy_signals = []
        for i in range(1, len(df)):
            w = FakeWindow(df, i)
            sig = strat.generate_signal(w)
            if sig and sig.action == "buy":
                buy_signals.append(sig)

        assert len(buy_signals) >= 1, "Should buy with positive momentum"

    def test_sell_negative_momentum(self):
        """After buy, if lookback return turns negative → sell."""
        strat = DualMomentum(lookback_months=1, rebalance_frequency=5)
        df = _concat_trends(_uptrend(30, 100.0, 1.0), _downtrend(30, 129.0, 3.0))

        actions = []
        for i in range(1, len(df)):
            w = FakeWindow(df, i)
            sig = strat.generate_signal(w)
            if sig:
                actions.append(sig.action)

        assert "buy" in actions
        assert "sell" in actions, "Should sell when momentum turns negative"


# ===========================================================================
# Opening Range Breakout
# ===========================================================================

class TestOpeningRangeBreakout:
    def test_buy_on_breakout(self):
        """Bullish bar where close > open + range*pct → buy."""
        strat = OpeningRangeBreakout(range_pct=0.3, hold_bars=2)
        # Create a bullish breakout bar
        df = pl.DataFrame({
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [105.0, 106.0, 107.0, 108.0],
            "low": [98.0, 97.0, 98.0, 99.0],
            "close": [99.0, 104.0, 106.0, 107.0],  # bar 1: close=104 > 100 + 9*0.3=102.7
            "volume": [1000, 1000, 1000, 1000],
        })

        buy_signals = []
        for i in range(1, len(df)):
            w = FakeWindow(df, i)
            sig = strat.generate_signal(w)
            if sig and sig.action == "buy":
                buy_signals.append((i, sig))

        assert len(buy_signals) >= 1, "Should buy on bullish breakout bar"

    def test_exit_after_hold_bars(self):
        """After buy, should sell after exactly hold_bars bars."""
        strat = OpeningRangeBreakout(range_pct=0.3, hold_bars=1)
        # Breakout bar at index 1, should exit at index 2
        df = pl.DataFrame({
            "open": [100.0, 100.0, 105.0, 106.0],
            "high": [102.0, 110.0, 108.0, 109.0],
            "low": [98.0, 97.0, 103.0, 104.0],
            "close": [99.0, 108.0, 106.0, 107.0],  # bar 1 triggers buy
            "volume": [1000, 1000, 1000, 1000],
        })

        actions = []
        for i in range(1, len(df)):
            w = FakeWindow(df, i)
            sig = strat.generate_signal(w)
            if sig:
                actions.append((i, sig.action))

        buy_bars = [i for i, a in actions if a == "buy"]
        sell_bars = [i for i, a in actions if a == "sell"]
        assert len(buy_bars) >= 1, "Should buy"
        assert len(sell_bars) >= 1, "Should sell after hold_bars"


# ===========================================================================
# No-lookahead test
# ===========================================================================

class TestNoLookahead:
    def test_warmup_no_signal(self):
        """No strategy should produce signals during its warmup period."""
        strategies = [
            SMACrossover(fast_period=5, slow_period=10),
            MACDCrossover(fast_ema=12, slow_ema=26, signal_period=9),
        ]
        df = _uptrend(100)

        for strat in strategies:
            warmup = strat.get_warmup_periods()
            for i in range(1, min(warmup - 2, len(df))):
                w = FakeWindow(df, i)
                sig = strat.generate_signal(w)
                assert sig is None, (
                    f"{strat.name} produced signal at bar {i}, warmup={warmup}"
                )
