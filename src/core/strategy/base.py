"""BaseStrategy — all strategies implement this interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Signal:
    action: str  # "buy" | "sell" | "exit" | "hold" | "short" | "cover"
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    metadata: dict = field(default_factory=dict)


class BaseStrategy(ABC):
    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    category: str = ""  # "trend" | "mean_reversion" | "momentum" | "multi_factor"
    tags: list[str] = []

    @abstractmethod
    def get_warmup_periods(self) -> int:
        """Minimum bars needed before first signal."""
        ...

    @abstractmethod
    def generate_signal(self, window) -> Signal | None:
        """
        Generate a signal given the current DataWindow.
        window.current_bar() — bar N (current)
        window.historical(n) — last n bars BEFORE current
        NEVER access bar N+1 or beyond.
        """
        ...

    def generate(self, window) -> "Signal | None":
        """Alias for generate_signal — called by the engine."""
        return self.generate_signal(window)

    def get_parameter_schema(self) -> dict:
        """Returns JSON Schema describing accepted parameters."""
        return {}
