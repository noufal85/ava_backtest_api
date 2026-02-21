import pytest
from src.core.analytics.metrics import calculate_all_metrics


def make_curve(equities: list[float]) -> list[dict]:
    return [{"date": f"2020-01-{i+1:02d}", "equity": e, "positions_value": 0}
            for i, e in enumerate(equities)]


def test_total_return():
    curve = make_curve([100000, 110000])
    m = calculate_all_metrics(curve, [], 100000)
    assert abs(m["total_return_pct"] - 10.0) < 0.01


def test_zero_drawdown_flat():
    curve = make_curve([100000] * 10)
    m = calculate_all_metrics(curve, [], 100000)
    assert m["max_drawdown_pct"] == 0.0


def test_drawdown_calculated():
    curve = make_curve([100000, 110000, 90000, 95000])
    m = calculate_all_metrics(curve, [], 100000)
    # Peak at 110000, trough at 90000 → DD = (110000-90000)/110000 * 100 ≈ 18.18%
    assert m["max_drawdown_pct"] > 18.0


def test_win_rate():
    trades = [{"pnl": 500}, {"pnl": -200}, {"pnl": 300}]
    m = calculate_all_metrics(make_curve([100000, 100600]), trades, 100000)
    assert abs(m["win_rate_pct"] - 66.67) < 0.1


def test_monthly_returns():
    curve = [{"date": f"2020-01-{i+1:02d}", "equity": 100000 + i*100, "positions_value": 0} for i in range(31)]
    m = calculate_all_metrics(curve, [], 100000)
    assert "2020-01" in m["monthly_returns"]
