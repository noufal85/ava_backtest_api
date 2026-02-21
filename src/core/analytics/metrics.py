"""
Metrics calculator — 20+ quantitative performance metrics.
Input: equity_curve (list of {date, equity}), trades (list of dicts), initial_capital, benchmark_returns (optional)
"""
import math
from typing import Any


def calculate_all_metrics(
    equity_curve: list[dict],
    trades: list[dict],
    initial_capital: float,
    benchmark_returns: list[float] | None = None,
    risk_free_rate: float = 0.04,  # 4% annual
) -> dict[str, Any]:
    if not equity_curve:
        return {}

    equities = [p["equity"] for p in equity_curve]
    dates = [p["date"] for p in equity_curve]
    returns = [(equities[i] - equities[i-1]) / equities[i-1] for i in range(1, len(equities))]
    n_years = len(equities) / 252 if len(equities) > 1 else 1

    # Primary metrics
    total_return_pct = (equities[-1] - initial_capital) / initial_capital * 100
    cagr_pct = ((equities[-1] / initial_capital) ** (1 / max(n_years, 0.01)) - 1) * 100

    # Volatility
    if len(returns) > 1:
        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        daily_vol = math.sqrt(variance)
        annual_vol_pct = daily_vol * math.sqrt(252) * 100
    else:
        daily_vol = annual_vol_pct = 0.0

    # Sharpe ratio
    daily_rf = risk_free_rate / 252
    excess_returns = [r - daily_rf for r in returns]
    if daily_vol > 0:
        sharpe = (sum(excess_returns) / len(excess_returns)) / daily_vol * math.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino ratio (downside deviation)
    neg_returns = [r for r in returns if r < daily_rf]
    if neg_returns:
        downside_var = sum((r - daily_rf) ** 2 for r in neg_returns) / len(returns)
        downside_dev = math.sqrt(downside_var) * math.sqrt(252)
        sortino = (cagr_pct / 100 - risk_free_rate) / downside_dev if downside_dev > 0 else 0.0
    else:
        sortino = 0.0

    # Max drawdown
    peak = initial_capital
    max_dd_pct = 0.0
    max_dd_days = 0
    dd_start = 0
    for i, eq in enumerate(equities):
        if eq >= peak:
            peak = eq
            dd_start = i
        else:
            dd = (peak - eq) / peak * 100
            days_in_dd = i - dd_start
            if dd > max_dd_pct:
                max_dd_pct = dd
                max_dd_days = days_in_dd

    # Calmar ratio
    calmar = (cagr_pct / max_dd_pct) if max_dd_pct > 0 else 0.0

    # Trade statistics
    pnls = [t.get("pnl", 0) or 0 for t in trades]
    winning = [p for p in pnls if p > 0]
    losing = [p for p in pnls if p <= 0]
    total_trades = len(pnls)
    win_rate_pct = (len(winning) / total_trades * 100) if total_trades > 0 else 0.0
    avg_winner = sum(winning) / len(winning) if winning else 0.0
    avg_loser = sum(losing) / len(losing) if losing else 0.0
    profit_factor = (sum(winning) / abs(sum(losing))) if losing and sum(losing) != 0 else 0.0
    avg_trade_pnl = sum(pnls) / total_trades if total_trades > 0 else 0.0
    best_trade = max(pnls) if pnls else 0.0
    worst_trade = min(pnls) if pnls else 0.0

    # Hold days
    hold_days_list = [t.get("hold_days", 0) or 0 for t in trades]
    avg_hold_days = sum(hold_days_list) / len(hold_days_list) if hold_days_list else 0.0

    # Exposure %
    position_days = sum(1 for e in equity_curve if e.get("positions_value", 0) > 0)
    exposure_pct = (position_days / len(equity_curve) * 100) if equity_curve else 0.0

    # Monthly returns
    monthly_returns = {}
    if dates:
        from collections import defaultdict
        monthly_equities = defaultdict(list)
        for p in equity_curve:
            d = str(p["date"])[:7]  # YYYY-MM
            monthly_equities[d].append(p["equity"])
        prev_end = initial_capital
        for month in sorted(monthly_equities):
            end_eq = monthly_equities[month][-1]
            monthly_returns[month] = (end_eq - prev_end) / prev_end * 100
            prev_end = end_eq

    # Benchmark comparison
    alpha = beta = information_ratio = None
    if benchmark_returns and len(benchmark_returns) == len(returns):
        b_mean = sum(benchmark_returns) / len(benchmark_returns)
        port_mean = sum(returns) / len(returns)
        cov = sum((returns[i] - port_mean) * (benchmark_returns[i] - b_mean)
                  for i in range(len(returns))) / len(returns)
        b_var = sum((b - b_mean) ** 2 for b in benchmark_returns) / len(benchmark_returns)
        beta = cov / b_var if b_var > 0 else None
        if beta:
            alpha = (port_mean - (daily_rf + beta * (b_mean - daily_rf))) * 252 * 100
        tracking_error = math.sqrt(sum((returns[i] - benchmark_returns[i]) ** 2
                                       for i in range(len(returns))) / len(returns)) * math.sqrt(252)
        information_ratio = ((port_mean - b_mean) * 252) / tracking_error if tracking_error > 0 else None

    return {
        "total_return_pct": round(total_return_pct, 4),
        "cagr_pct": round(cagr_pct, 4),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio": round(calmar, 4),
        "max_drawdown_pct": round(max_dd_pct, 4),
        "max_drawdown_days": max_dd_days,
        "annual_volatility_pct": round(annual_vol_pct, 4),
        "win_rate_pct": round(win_rate_pct, 2),
        "profit_factor": round(profit_factor, 4),
        "avg_trade_pnl": round(avg_trade_pnl, 2),
        "avg_winner": round(avg_winner, 2),
        "avg_loser": round(avg_loser, 2),
        "best_trade_pnl": round(best_trade, 2),
        "worst_trade_pnl": round(worst_trade, 2),
        "total_trades": total_trades,
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "avg_hold_days": round(avg_hold_days, 1),
        "exposure_pct": round(exposure_pct, 2),
        "final_equity": round(equities[-1], 2),
        "monthly_returns": monthly_returns,
        "alpha": round(alpha, 4) if alpha is not None else None,
        "beta": round(beta, 4) if beta is not None else None,
        "information_ratio": round(information_ratio, 4) if information_ratio is not None else None,
    }
