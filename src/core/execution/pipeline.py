"""Middleware pipeline — composable, testable engine stages."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from src.core.execution.portfolio import Portfolio
from src.core.execution.data_window import DataWindow


@dataclass
class EngineState:
    window: DataWindow
    portfolio: Portfolio
    pending_signal: str | None = None  # "buy" | "sell" | None
    pending_metadata: dict = field(default_factory=dict)
    indicator_values: dict = field(default_factory=dict)
    current_bar_prices: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


class Middleware(ABC):
    @abstractmethod
    def process(self, state: EngineState) -> EngineState:
        ...


class BacktestPipeline:
    def __init__(self, middlewares: list[Middleware]):
        self.middlewares = middlewares

    def run(self, state: EngineState) -> EngineState:
        for mw in self.middlewares:
            state = mw.process(state)
        return state
