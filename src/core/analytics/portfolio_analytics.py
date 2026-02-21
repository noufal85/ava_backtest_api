"""Cross-strategy portfolio analytics."""
import math


def calculate_correlation_matrix(equity_curves: list[list[float]]) -> list[list[float]]:
    """Pearson correlation matrix of daily returns for multiple strategies."""
    n = len(equity_curves)
    returns_list = [
        [(eq[i] - eq[i-1]) / eq[i-1] for i in range(1, len(eq))]
        for eq in equity_curves
    ]
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 1.0
            else:
                r1, r2 = returns_list[i], returns_list[j]
                min_len = min(len(r1), len(r2))
                r1, r2 = r1[:min_len], r2[:min_len]
                if min_len < 2:
                    continue
                m1 = sum(r1) / min_len
                m2 = sum(r2) / min_len
                cov = sum((r1[k]-m1)*(r2[k]-m2) for k in range(min_len)) / min_len
                std1 = math.sqrt(sum((r-m1)**2 for r in r1) / min_len)
                std2 = math.sqrt(sum((r-m2)**2 for r in r2) / min_len)
                matrix[i][j] = cov / (std1 * std2) if std1 * std2 > 0 else 0.0
    return matrix


def combine_equity_curves(equity_curves: list[list[dict]], weights: list[float] | None = None) -> list[dict]:
    """Weighted combination of equity curves."""
    if not equity_curves:
        return []
    n = len(equity_curves)
    if weights is None:
        weights = [1.0 / n] * n
    min_len = min(len(ec) for ec in equity_curves)
    combined = []
    for i in range(min_len):
        weighted_equity = sum(equity_curves[j][i]["equity"] * weights[j] for j in range(n))
        combined.append({"date": equity_curves[0][i]["date"], "equity": weighted_equity})
    return combined
