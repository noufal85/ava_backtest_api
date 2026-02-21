"""Portfolio — tracks cash, positions, equity history."""
from dataclasses import dataclass, field


@dataclass
class Position:
    symbol: str
    quantity: int
    avg_cost: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class Fill:
    symbol: str
    side: str           # "buy" | "sell" | "short" | "cover"
    quantity: int
    price: float
    commission: float
    stt: float = 0.0
    gst: float = 0.0
    stamp_duty: float = 0.0
    date: str = ""
    reason: str = "signal"

    @property
    def total_cost(self) -> float:
        return self.commission + self.stt + self.gst + self.stamp_duty


class Portfolio:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, Position] = {}
        self.equity_history: list[dict] = []
        self.fills: list[Fill] = []

    def apply_fill(self, fill: Fill) -> float:
        """Apply a fill, update position, return realized P&L."""
        realized = 0.0
        symbol = fill.symbol
        total_costs = fill.total_cost

        if fill.side == "buy":
            trade_value = fill.quantity * fill.price
            self.cash -= (trade_value + total_costs)
            if symbol in self.positions:
                pos = self.positions[symbol]
                total_qty = pos.quantity + fill.quantity
                pos.avg_cost = (pos.avg_cost * pos.quantity + fill.price * fill.quantity) / total_qty
                pos.quantity = total_qty
            else:
                self.positions[symbol] = Position(
                    symbol=symbol, quantity=fill.quantity, avg_cost=fill.price
                )

        elif fill.side in ("sell", "exit"):
            if symbol in self.positions:
                pos = self.positions[symbol]
                sell_value = fill.quantity * fill.price
                cost_basis = fill.quantity * pos.avg_cost
                realized = sell_value - cost_basis - total_costs
                pos.realized_pnl += realized
                pos.quantity -= fill.quantity
                self.cash += sell_value - total_costs
                if pos.quantity <= 0:
                    del self.positions[symbol]

        self.fills.append(fill)
        return realized

    def update_market_values(self, prices: dict[str, float], timestamp: str) -> None:
        """Mark-to-market all positions and record equity snapshot."""
        total_position_value = 0.0
        for symbol, pos in self.positions.items():
            price = prices.get(symbol, pos.avg_cost)
            pos.market_value = pos.quantity * price
            pos.unrealized_pnl = pos.market_value - (pos.quantity * pos.avg_cost)
            total_position_value += pos.market_value
        equity = self.cash + total_position_value
        self.equity_history.append({
            "date": timestamp,
            "equity": equity,
            "cash": self.cash,
            "positions_value": total_position_value,
        })

    @property
    def equity(self) -> float:
        if self.equity_history:
            return self.equity_history[-1]["equity"]
        return self.cash

    @property
    def gross_exposure(self) -> float:
        return sum(abs(p.market_value) for p in self.positions.values())

    @property
    def exposure_pct(self) -> float:
        return (self.gross_exposure / self.equity * 100) if self.equity > 0 else 0.0
