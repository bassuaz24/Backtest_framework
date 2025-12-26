from datetime import datetime
import pandas as pd
import numpy as np

from strategies.strategy import BaseStrategy
from core.sizer import target_weights_to_quantities

class MeanReversionStrategy(BaseStrategy):
    """
    A basic mean reversion strategy.
    
    Generates a buy signal when a security's recent return drops below a
    Z-score threshold relative to its recent volatility.
    """
    def __init__(
        self,
        universe: list[str],
        lookback_short: int = 5,
        lookback_vol: int = 20,
        entry_z: float = 1.5,
        exit_z: float = 0.5
    ):
        self.universe = universe
        self.lookback_short = lookback_short
        self.lookback_vol = lookback_vol
        self.entry_z = -abs(entry_z)  # Ensure it's negative
        self.exit_z = -abs(exit_z)    # Ensure it's negative

        self.required_lookback = self.lookback_vol + 1

    def generate_orders(
        self, 
        dt: datetime, 
        data: pd.DataFrame, 
        portfolio_state: dict
    ) -> dict[str, float]:
        """
        Generates a dictionary of target quantities based on the mean reversion signal.
        """
        target_weights = {}
        
        for symbol in self.universe:
            if symbol not in data.columns:
                target_weights[symbol] = 0.0
                continue

            hist = data[symbol]
            
            # --- Guardrails ---
            if len(hist) < self.required_lookback:
                target_weights[symbol] = 0.0
                continue # Not enough data

            # --- Signal Calculation ---
            daily_returns = hist.pct_change()
            volatility = daily_returns.rolling(window=self.lookback_vol).std().iloc[-1]
            
            if volatility < 1e-6: # Volatility floor
                target_weights[symbol] = 0.0
                continue

            short_horizon_return = (hist.iloc[-1] / hist.iloc[-self.lookback_short]) - 1
            z_score = short_horizon_return / volatility

            # --- Position Sizing ---
            current_weight = portfolio_state['weights'].get(symbol, 0.0)
            
            if z_score < self.entry_z:
                # Signal to buy/hold
                target_weights[symbol] = 1.0 / len(self.universe) # Equal weight
            elif z_score > self.exit_z and current_weight > 0:
                # Signal to exit
                target_weights[symbol] = 0.0
            else:
                # Maintain current position (no new signal)
                target_weights[symbol] = current_weight
        
        # --- Sizing ---
        quantities = target_weights_to_quantities(
            target_weights=target_weights,
            current_positions=portfolio_state['positions'],
            equity=portfolio_state['equity'],
            close_prices=data.iloc[-1] # Last row of history is current close
        )
        
        return quantities
