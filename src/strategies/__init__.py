"""Import all strategies so they self-register via @register decorator."""
from src.strategies.classic import sma_crossover, rsi_mean_reversion, macd_crossover, bollinger_bands, momentum_breakout  # noqa: F401
from src.strategies.modern import rsi_vol_filter, dual_momentum, orb  # noqa: F401
from src.strategies import custom  # noqa: F401  ← auto-discovers everything in custom/
