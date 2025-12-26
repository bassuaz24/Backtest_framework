from datetime import timedelta, datetime
import pandas as pd
from core.data import DataHandler
from core.execution import ExecutionHandler, Order
from core.portfolio import Portfolio
from strategies.strategy import BaseStrategy

class BacktestEngine:
    """
    Orchestrates the backtest by integrating data, strategy, execution, and portfolio components.
    """
    def __init__(
        self,
        start_date: str,
        end_date: str,
        data_handler: DataHandler,
        execution_handler: ExecutionHandler,
        portfolio: Portfolio,
        strategy: BaseStrategy,
        rebalance_schedule: str = 'M' # 'M' for month-end, 'W' for week-end, etc.
    ):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.data_handler = data_handler
        self.execution_handler = execution_handler
        self.portfolio = portfolio
        self.strategy = strategy
        self.rebalance_schedule = rebalance_schedule
        
        self.trading_days = self._get_trading_calendar()
        self.scheduled_orders: dict[datetime, list] = {}

    def _get_trading_calendar(self) -> pd.DatetimeIndex:
        """
        Returns a calendar of trading days based on the available data for a proxy symbol.
        """
        if not self.strategy.universe:
            print("Warning: No universe specified for strategy. Falling back to business days.")
            return pd.bdate_range(self.start_date, self.end_date, tz="UTC")

        proxy_symbol = self.strategy.universe[0]
        print(f"Building trading calendar from proxy symbol: {proxy_symbol}")

        all_dates_df = self.data_handler.get_bars(
            proxy_symbol,
            self.start_date.strftime('%Y-%m-%d'),
            self.end_date.strftime('%Y-%m-%d')
        )
        
        if all_dates_df.empty:
             print("Warning: Could not build calendar from data. Falling back to business days.")
             return pd.bdate_range(self.start_date, self.end_date, tz="UTC")
        
        trading_days = pd.to_datetime(all_dates_df['date']).unique()
        return pd.DatetimeIndex(trading_days).sort_values()

    def _is_rebalance_day(self, current_date: datetime) -> bool:
        """Checks if the current date is a rebalance day based on the schedule."""
        # This is a simple implementation. A more robust one would use pandas offsets.
        if self.rebalance_schedule == 'D': # Daily
            return True
        if self.rebalance_schedule == 'W': # End of Week
            return current_date.weekday() == 4 # Friday
        if self.rebalance_schedule == 'M': # End of Month
            return (current_date + timedelta(days=1)).month != current_date.month
        return False

    def run_backtest(self):
        """
        Runs the main backtesting loop.
        """
        print("Starting backtest...")
        
        for i, t_date in enumerate(self.trading_days):
            print(f"Processing {t_date.date()}...")
            
            # --- 1. Start-of-day: Execute scheduled orders ---
            orders_to_execute = self.scheduled_orders.pop(t_date, [])
            if orders_to_execute:
                fills, rejected = self.execution_handler.simulate_execution(
                    orders=orders_to_execute,
                    data_handler=self.data_handler
                )
                for fill in fills:
                    self.portfolio.apply_fill(fill)
                # TODO: Log rejected orders

            # --- Get all symbols and close prices for end-of-day processes ---
            all_symbols = self.strategy.universe + list(self.portfolio.positions.keys())
            all_symbols = sorted(list(set(all_symbols)))
            
            # This is inefficient, but simple. A better way is a single call for all prices.
            close_prices = {}
            for symbol in all_symbols:
                price = self.data_handler.get_price(symbol, t_date.strftime('%Y-%m-%d'), field='adj_close')
                if price:
                    close_prices[symbol] = price

            # --- 2. End-of-day: Generate new orders on rebalance days ---
            if self._is_rebalance_day(t_date):
                current_equity = self.portfolio.mark_to_market(close_prices)
                
                portfolio_state = {
                    'equity': current_equity,
                    'cash': self.portfolio.cash,
                    'positions': self.portfolio.positions,
                    'weights': self.portfolio.get_weights(close_prices)
                }

                # Get historical data for the strategy
                hist_data = self.data_handler.get_history(
                    symbols=self.strategy.universe, 
                    end_date=t_date, 
                    lookback_days=self.strategy.required_lookback,
                    field='adj_close'
                )

                if not hist_data.empty:
                    target_quantities = self.strategy.generate_orders(t_date, hist_data, portfolio_state)
                    
                    # Create and schedule orders for the next trading day
                    if i + 1 < len(self.trading_days):
                        next_day = self.trading_days[i+1]
                        new_orders = []
                        for symbol, quantity in target_quantities.items():
                            new_orders.append(
                                Order(
                                    symbol=symbol,
                                    shares=quantity,
                                    generated_dt=t_date,
                                    execute_dt=next_day
                                )
                            )
                        self.scheduled_orders.setdefault(next_day, []).extend(new_orders)

            # --- 3. End-of-day: Snapshot portfolio ---
            self.portfolio.take_snapshot(t_date, close_prices)

        print("Backtest complete.")
        return self.portfolio.history_df
