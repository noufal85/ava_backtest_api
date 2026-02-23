"""tv_linear_regression_channel — converted from trading-backtester."""
import polars as pl
import numpy as np
import math


class LinearRegression:
    """Minimal linear regression (no sklearn needed)."""
    def fit(self, X, y):
        X = np.array(X).reshape(-1, 1) if len(np.array(X).shape) == 1 else np.array(X)
        y = np.array(y)
        X_b = np.c_[np.ones(X.shape[0]), X]
        theta = np.linalg.lstsq(X_b, y, rcond=None)[0]
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        return self
    def predict(self, X):
        X = np.array(X).reshape(-1, 1) if len(np.array(X).shape) == 1 else np.array(X)
        return np.c_[np.ones(X.shape[0]), X] @ np.r_[self.intercept_, self.coef_]
from src.core.strategy.base import BaseStrategy, Signal
from src.core.strategy.registry import register

@register
class TvLinearRegressionChannel(BaseStrategy):
    name: str = "tv_linear_regression_channel"
    version: str = "1.0.0"
    description: str = "Linear Regression Channel: Trend following with regression channel breakouts"
    category: str = "multi_factor"
    tags: list[str] = []

    def __init__(self, length: int = 50, multiplier: float = 2.0):
        self.length = length
        self.multiplier = multiplier

    def get_warmup_periods(self) -> int:
        return self.length + 5  # Add a small buffer

    def _linear_regression(self, closes: list[float], length: int) -> list[float]:
        """Calculate linear regression line."""
        result = [None] * len(closes)
        
        for i in range(length - 1, len(closes)):
            # Get the window of data
            y_values = closes[i - length + 1:i + 1]
            x_values = np.arange(len(y_values)).reshape(-1, 1)
            
            # Fit linear regression
            lr = LinearRegression()
            lr.fit(x_values, y_values)
            
            # Predict the current value (last point in window)
            result[i] = lr.predict([[len(y_values) - 1]])[0]
        
        return result

    def _rolling_std(self, series: list[float], length: int) -> list[float]:
        """Calculate rolling standard deviation."""
        result = [None] * len(series)
        for i in range(length - 1, len(series)):
            window = series[i - length + 1: i + 1]
            result[i] = np.std(window)
        return result

    def generate_signal(self, window) -> Signal | None:
        historical_df = window.historical()
        current_bar = window.current_bar()
        df = pl.concat([historical_df, current_bar])

        length = self.length
        multiplier = self.multiplier

        closes = df["close"].to_list()

        if len(closes) < length:
            return None

        lin_reg = self._linear_regression(closes, length)
        std_dev = self._rolling_std(closes, length)
        
        lr_upper = [lin_reg[i] + std_dev[i] * multiplier if lin_reg[i] is not None and std_dev[i] is not None else None for i in range(len(closes))]
        lr_lower = [lin_reg[i] - std_dev[i] * multiplier if lin_reg[i] is not None and std_dev[i] is not None else None for i in range(len(closes))]

        current_close = closes[-1]
        prev_close = closes[-2] if len(closes) > 1 else None
        prev_lr_upper = lr_upper[-2] if len(lr_upper) > 1 and lr_upper[-2] is not None else None
        prev_lr_lower = lr_lower[-2] if len(lr_lower) > 1 and lr_lower[-2] is not None else None

        upper_breakout = False
        lower_breakdown = False

        if prev_close is not None and prev_lr_upper is not None:
            upper_breakout = (current_close > prev_lr_upper) and (prev_close <= prev_lr_upper)
        if prev_close is not None and prev_lr_lower is not None:
            lower_breakdown = (current_close < prev_lr_lower) and (prev_close >= prev_lr_lower)

        if upper_breakout:
            return Signal(action="buy", strength=1.0, confidence=0.8, metadata={"level": "upper"})
        elif lower_breakdown:
            return Signal(action="sell", strength=1.0, confidence=0.8, metadata={"level": "lower"})
        else:
            return None

    def get_parameter_schema(self) -> dict:
        return {
            "length": {"type": "integer", "default": 50, "title": "Length"},
            "multiplier": {"type": "number", "default": 2.0, "title": "Multiplier"},
        }