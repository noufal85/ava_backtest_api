import pytest
from src.core.markets.registry import get_market, list_markets, MarketCode

def test_get_us_market():
    m = get_market("US")
    assert m.currency == "USD"
    assert m.currency_symbol == "$"
    assert m.default_universe == "sp500_liquid"

def test_get_in_market():
    m = get_market("IN")
    assert m.currency == "INR"
    assert m.currency_symbol == "₹"
    assert m.settlement_days == 1
    assert m.flags["stt_buy_pct"] == 0.001

def test_invalid_market():
    with pytest.raises(ValueError):
        get_market("XX")

def test_list_markets():
    markets = list_markets()
    assert len(markets) == 2
    defaults = [m for m in markets if m["is_default"]]
    assert len(defaults) == 1
    assert defaults[0]["code"] == "US"
