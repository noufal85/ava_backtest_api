"""Walk-forward analysis — rolling IS/OOS windows."""
from dataclasses import dataclass
from typing import Any


@dataclass
class WindowResult:
    window_num: int
    is_start: str
    is_end: str
    oos_start: str
    oos_end: str
    is_sharpe: float
    oos_sharpe: float
    degradation_pct: float
    best_params: dict


class WalkForwardAnalyzer:
    def __init__(self, is_months: int = 18, oos_months: int = 6):
        self.is_months = is_months
        self.oos_months = oos_months

    def generate_windows(self, start_date: str, end_date: str) -> list[dict]:
        """Generate rolling IS/OOS date windows."""
        from datetime import datetime, timedelta
        from dateutil.relativedelta import relativedelta

        windows = []
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)

        window_start = start
        window_num = 0

        while True:
            is_start = window_start
            is_end = is_start + relativedelta(months=self.is_months)
            oos_start = is_end
            oos_end = oos_start + relativedelta(months=self.oos_months)

            if oos_end > end:
                break

            windows.append({
                "window_num": window_num,
                "is_start": is_start.date().isoformat(),
                "is_end": is_end.date().isoformat(),
                "oos_start": oos_start.date().isoformat(),
                "oos_end": oos_end.date().isoformat(),
            })
            window_start += relativedelta(months=self.oos_months)
            window_num += 1

        return windows

    def calculate_degradation(self, is_sharpe: float, oos_sharpe: float) -> float:
        """Degradation score: how much worse OOS is vs IS (0=no degradation, 1=total breakdown)."""
        if is_sharpe <= 0:
            return 1.0
        return max(0.0, (is_sharpe - oos_sharpe) / is_sharpe)
