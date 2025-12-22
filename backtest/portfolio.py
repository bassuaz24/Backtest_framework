from dataclasses import dataclass
from datetime import datetime
import pandas as pd

@dataclass
class Fill:
    """
    Represents an executed trade (a fill).
    """
    dt: datetime
    symbol: str
    shares: float  # Positive for buy, negative for sell
    price: float
    fee: float = 0.0

    @property
    def notional(self) -> float:
        return self.shares * self.price

    @property
    def cash_change(self) -> float:
        return -self.notional - self.fee

class Portfolio:
    """
    Manages the state of a portfolio: cash, positions, and performance.
    """
    def __init__(self, initial_cash: float = 100000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: dict[str, float] = {}
        self.fees_paid = 0.0
        
        self.fills_log: list[Fill] = []
        self.daily_history: list[dict] = []
        
        self._last_equity = initial_cash

    def apply_fill(self, fill: Fill):
        """
        Updates the portfolio state based on a new fill.
        """
        # --- Validation ---
        if fill.price <= 0:
            raise ValueError("Fill price must be positive.")
        if fill.fee < 0:
            raise ValueError("Fee must be non-negative.")

        # --- Update position ---
        self.positions[fill.symbol] = self.positions.get(fill.symbol, 0) + fill.shares
        # Remove position if shares are zero (or very close to it)
        if abs(self.positions[fill.symbol]) < 1e-9:
            del self.positions[fill.symbol]

        # --- Update cash and fees ---
        self.cash += fill.cash_change
        self.fees_paid += fill.fee

        self.fills_log.append(fill)

    def mark_to_market(self, prices: dict[str, float]) -> float:
        """
        Calculates and returns the current equity of the portfolio.
        """
        # --- Strict validation for missing prices ---
        for symbol in self.positions:
            if symbol not in prices:
                raise ValueError(f"Missing price for position '{symbol}' during mark-to-market.")

        positions_value = sum(self.positions[symbol] * prices[symbol] for symbol in self.positions)
        
        equity = self.cash + positions_value
        
        # --- Sanity check ---
        if not abs(equity - (self.cash + positions_value)) < 1e-9:
             raise RuntimeError("Portfolio invariant violated: equity != cash + positions_value")

        return equity

    def get_weights(self, prices: dict[str, float]) -> dict[str, float]:
        """
        Calculates the weight of each asset in the portfolio.
        """
        equity = self.mark_to_market(prices)
        if abs(equity) < 1e-9:
            return {symbol: 0.0 for symbol in self.positions}

        weights = {
            symbol: (self.positions[symbol] * prices[symbol]) / equity
            for symbol in self.positions
        }
        weights['cash'] = self.cash / equity
        return weights

    def take_snapshot(self, dt: datetime, prices: dict[str, float]):
        """
        Records a daily snapshot of the portfolio's state and performance.
        """
        equity = self.mark_to_market(prices)
        positions_value = equity - self.cash
        
        gross_exposure = sum(abs(self.positions[symbol] * prices[symbol]) for symbol in self.positions)
        
        snapshot = {
            'datetime': dt,
            'equity': equity,
            'cash': self.cash,
            'positions_value': positions_value,
            'gross_exposure': gross_exposure,
            'net_exposure': positions_value, # For long-only, net exposure = positions value
            'num_positions': len(self.positions),
            'daily_return': (equity / self._last_equity) - 1 if self._last_equity != 0 else 0.0,
        }
        self.daily_history.append(snapshot)
        self._last_equity = equity

    @property
    def history_df(self) -> pd.DataFrame:
        """Returns the daily snapshot history as a pandas DataFrame."""
        return pd.DataFrame(self.daily_history).set_index('datetime')

    @property
    def fills_df(self) -> pd.DataFrame:
        """Returns the fill log as a pandas DataFrame."""
        return pd.DataFrame([f.__dict__ for f in self.fills_log]).set_index('dt')
