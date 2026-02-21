"""Hardcoded symbol lists per universe — used when no DB is available."""

UNIVERSE_SYMBOLS: dict[str, list[str]] = {
    "sp500": [
        "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM","V","XOM",
        "UNH","JNJ","WMT","MA","PG","HD","CVX","MRK","LLY","ABBV",
    ],
    "sp500_liquid": [
        "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM","V","XOM",
        "UNH","JNJ","WMT","MA","PG","HD","CVX","MRK","LLY","ABBV",
    ],
    "nasdaq100": [
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AVGO","COST","NFLX",
        "ADBE","AMD","QCOM","INTC","INTU","CSCO","CMCSA","PEP","AMAT","MU",
    ],
    "djia": [
        "AAPL","MSFT","JPM","V","JNJ","WMT","PG","UNH","HD","MCD",
        "GS","CAT","AXP","BA","IBM","HON","MMM","DIS","NKE","CRM",
    ],
    "russell2000": [
        "IWM","ACIW","ACHC","AEIS","AGEN","AGYS","AIMC","ALGT","AMED","AMPH",
    ],
    "nifty50": [
        "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
        "HINDUNILVR.NS","BAJFINANCE.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS",
    ],
    "nifty_midcap": [
        "MPHASIS.NS","COFORGE.NS","PERSISTENT.NS","LTIM.NS","TATAELXSI.NS",
    ],
    "nifty_smallcap": [
        "TATAELXSI.NS","MPHASIS.NS","COFORGE.NS","PERSISTENT.NS","LTIM.NS",
    ],
}


def get_symbols(universe_name: str, limit: int = 5) -> list[str]:
    """Return up to `limit` symbols for a given universe."""
    syms = UNIVERSE_SYMBOLS.get(universe_name, UNIVERSE_SYMBOLS["sp500_liquid"])
    return syms[:limit]
