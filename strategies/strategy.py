from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd

class BaseStrategy(ABC):
    """
    Abstract base class for a trading strategy.
    """
    @abstractmethod
    def generate_orders(
        self, 
        dt: datetime, 
        data: pd.DataFrame, 
        portfolio_state: dict
    ) -> list:
        """
        Generates a list of orders based on market data and portfolio state.
        
        Args:
            dt: The current datetime of the backtest.
            data: A dictionary or DataFrame of market data for signal generation.
            portfolio_state: A dictionary containing the current state of the portfolio 
                             (e.g., equity, cash, positions).

        Returns:
            A list of Order objects.
        """
        raise NotImplementedError
