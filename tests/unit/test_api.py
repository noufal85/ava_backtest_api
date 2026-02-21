"""API endpoint tests — FastAPI TestClient."""
from fastapi.testclient import TestClient

from src.api.v2.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["version"] == "2.0.0"


def test_get_markets():
    r = client.get("/api/v2/markets")
    assert r.status_code == 200
    codes = [m["code"] for m in r.json()]
    assert "US" in codes and "IN" in codes


def test_list_strategies_us():
    r = client.get("/api/v2/strategies?market=US")
    assert r.status_code == 200
    assert len(r.json()) == 8


def test_get_strategy_detail():
    r = client.get("/api/v2/strategies/sma_crossover")
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "sma_crossover"
    assert "fast_period" in data["parameter_schema"]


def test_get_strategy_not_found():
    r = client.get("/api/v2/strategies/nonexistent_strategy")
    assert r.status_code == 422


def test_list_universes_us():
    r = client.get("/api/v2/universes?market=US")
    assert r.status_code == 200
    names = [u["name"] for u in r.json()]
    assert "sp500" in names


def test_list_universes_india():
    r = client.get("/api/v2/universes?market=IN")
    assert r.status_code == 200
    names = [u["name"] for u in r.json()]
    assert "nifty50" in names and "nifty500" in names


def test_invalid_market():
    r = client.get("/api/v2/universes?market=XX")
    assert r.status_code == 422


def test_create_backtest():
    r = client.post(
        "/api/v2/backtests",
        json={
            "strategy_name": "sma_crossover",
            "market": "US",
            "universe": "sp500_liquid",
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "initial_capital": 100000,
        },
    )
    assert r.status_code == 201
    data = r.json()
    assert "id" in data
    assert data["status"] == "pending"
    assert data["market_code"] == "US"


def test_create_backtest_invalid_market():
    r = client.post(
        "/api/v2/backtests",
        json={
            "strategy_name": "sma_crossover",
            "market": "XX",
            "universe": "sp500_liquid",
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "initial_capital": 100000,
        },
    )
    assert r.status_code == 422


def test_get_backtest():
    r = client.get("/api/v2/backtests/some-run-id")
    assert r.status_code == 200
    assert r.json()["id"] == "some-run-id"


def test_list_backtests():
    r = client.get("/api/v2/backtests?market=US")
    assert r.status_code == 200
    assert r.json()["total"] == 0


def test_get_trades():
    r = client.get("/api/v2/backtests/some-id/trades")
    assert r.status_code == 200
    assert r.json()["total"] == 0


def test_get_equity_curve():
    r = client.get("/api/v2/backtests/some-id/equity-curve")
    assert r.status_code == 200
    assert "points" in r.json()


def test_cancel_backtest():
    r = client.delete("/api/v2/backtests/some-id")
    assert r.status_code == 204


def test_get_candles():
    r = client.get("/api/v2/data/candles?symbol=AAPL&market=US&start=2020-01-01&end=2024-12-31")
    assert r.status_code == 200
    assert r.json()["symbol"] == "AAPL"


def test_search_symbols():
    r = client.get("/api/v2/symbols/search?q=AAPL&market=US")
    assert r.status_code == 200
    assert r.json()["market"] == "US"


def test_analytics_correlation():
    r = client.get("/api/v2/analytics/correlation?backtest_ids=id1&backtest_ids=id2")
    assert r.status_code == 200
    assert "matrix" in r.json()


def test_analytics_portfolio():
    r = client.get("/api/v2/analytics/portfolio?backtest_ids=id1&backtest_ids=id2")
    assert r.status_code == 200
    assert "combined_sharpe" in r.json()


def test_strategy_shortcut_backtest():
    r = client.post(
        "/api/v2/strategies/sma_crossover/backtest",
        json={
            "universe": "sp500_liquid",
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "initial_capital": 100000,
        },
    )
    assert r.status_code == 201
    assert r.json()["strategy_name"] == "sma_crossover"
