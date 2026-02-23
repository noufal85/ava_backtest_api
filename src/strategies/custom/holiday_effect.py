from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, timedelta
import polars as pl
import numpy as np
from typing import List

@dataclass
class Signal:
    action: str          # "buy" | "sell" | "hold"
    strength: float      # 0.0 – 1.0
    confidence: float    # 0.0 – 1.0
    metadata: dict = field(default_factory=dict)

class BaseStrategy(ABC):
    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    category: str = ""   # trend | mean_reversion | momentum | multi_factor | volatility
    tags: list[str] = []

    @abstractmethod
    def get_warmup_periods(self) -> int: ...

    @abstractmethod
    def generate_signal(self, window) -> Signal | None:
        # window.current_bar()   → polars DataFrame row (current bar)
        # window.historical()    → polars DataFrame of all historical bars
        # Columns: open, high, low, close, volume (float64), timestamp (date)
        # Combine: pl.concat([window.historical(), window.current_bar()])
        # NEVER look ahead
        ...

    def get_parameter_schema(self) -> dict:
        return {}

from src.core.strategy.registry import register

def us_market_holidays(year: int) -> List[date]:
    """Crude approximation of US market holidays."""
    # New Year's Day, Martin Luther King Day, President's Day, Good Friday, Memorial Day,
    # Independence Day, Labor Day, Thanksgiving, Christmas
    holidays = [
        date(year, 1, 1),  # New Year's Day
        date(year, 7, 4),  # Independence Day
        date(year, 12, 25), # Christmas
    ]

    # MLK Day (3rd Monday in January)
    holidays.append(date(year, 1, 15) + timedelta(days=(7 - date(year, 1, 15).weekday()) % 7 + 14))

    # President's Day (3rd Monday in February)
    holidays.append(date(year, 2, 15) + timedelta(days=(7 - date(year, 2, 15).weekday()) % 7 + 14))

    # Memorial Day (Last Monday in May)
    holidays.append(date(year, 5, 31) - timedelta(days=date(year, 5, 31).weekday() % 7))

    # Labor Day (1st Monday in September)
    holidays.append(date(year, 9, 1) + timedelta(days=(7 - date(year, 9, 1).weekday()) % 7))

    # Thanksgiving (4th Thursday in November)
    holidays.append(date(year, 11, 1) + timedelta(days=(3 - date(year, 11, 1).weekday()) % 7 + 21))

    return holidays

@register
class HolidayEffect(BaseStrategy):
    name = "holiday_effect"
    version = "1.0.0"
    description = "Calendar: buy before market holidays, sell after"
    category = "multi_factor"
    tags = ["calendar", "holiday"]

    def __init__(self, days_before: int = 2):
        self.days_before = days_before

    def get_parameter_schema(self) -> dict:
        return {
            "days_before": {"type": "integer", "default": 2, "description": "Trading days before holiday to enter"},
        }

    def get_warmup_periods(self) -> int:
        return 30  # Lookback for holiday calculation

    def generate_signal(self, window) -> Signal | None:
        current_bar = window.current_bar()
        historical_data = window.historical()
        df = pl.concat([historical_data, current_bar])

        current_date = current_bar["ts"][0].date()

        trading_dates_sorted = df["ts"].dt.date().sort().to_list()
        trading_dates_set = set(trading_dates_sorted)

        years = {d.year for d in trading_dates_sorted}
        all_holidays = set()
        for year in years:
            all_holidays.update(us_market_holidays(year))
        all_holidays.update(us_market_holidays(max(years) + 1))

        holiday_dates = set()
        for h in all_holidays:
            if isinstance(h, date):
                holiday_dates.add(h)
            else:
                holiday_dates.add(h.to_date())

        date_to_idx = {d: i for i, d in enumerate(trading_dates_sorted)}

        idx = date_to_idx.get(current_date, -1)
        if idx < 0:
            return None

        found = False
        next_holiday = None
        for future_idx in range(idx + 1, min(idx + 30, len(trading_dates_sorted))):
            future_date = trading_dates_sorted[future_idx]
            if future_idx > idx:
                trading_days_between = future_idx - idx
                check_date = future_date
                day_after = check_date + timedelta(days=1)
                while day_after not in trading_dates_set and day_after <= check_date + timedelta(days=10):
                    if day_after in holiday_dates:
                        if trading_days_between == self.days_before:
                            found = True
                            next_holiday = day_after
                            break
                        day_after += timedelta(days=1)
                if found:
                    break

        if found:
            return Signal(action="buy", strength=1.0, confidence=1.0, metadata={"next_holiday": str(next_holiday)})
        else:
            return None