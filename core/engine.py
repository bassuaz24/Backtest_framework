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
        self.start_date = pd.to_datetime(start_date).tz_localize('UTC')
        self.end_date = pd.to_datetime(end_date).tz_localize('UTC')
        self.data_handler = data_handler
        self.execution_handler = execution_handler
        self.portfolio = portfolio
        self.strategy = strategy
        self.rebalance_schedule = rebalance_schedule
        
        self.all_data = self._preload_data()
        self.trading_days = self._get_trading_calendar()
        self.scheduled_orders: dict[datetime, list] = {}

    def _preload_data(self):
        """
        Pre-loads all necessary historical data for the backtest period.
        """
        print("Pre-loading all historical data...")
        # Add a buffer for lookbacks
        start_date_with_lookback = self.start_date - pd.tseries.offsets.BDay(self.strategy.required_lookback + 5)
        
        all_symbols = self.strategy.universe
        
        bars = self.data_handler.get_bars(
            symbols=all_symbols,
            start=start_date_with_lookback.strftime('%Y-%m-%d'),
            end=self.end_date.strftime('%Y-%m-%d')
        )
        
        if bars.empty:
            print("Warning: No data loaded for the specified universe and date range.")
            return {}

        # Pivot data for easier access
        adj_close_prices = bars.pivot(index='date', columns='symbol', values='adj_close')
        open_prices = bars.pivot(index='date', columns='symbol', values='open')
        
        return {
            'adj_close': adj_close_prices,
            'open': open_prices
        }

    def _get_trading_calendar(self) -> pd.DatetimeIndex:
        """
        Returns a calendar of trading days based on the pre-loaded data.
        """
        if not self.all_data or self.all_data['adj_close'].empty:
            print("Warning: No data pre-loaded. Falling back to business days.")
            return pd.bdate_range(self.start_date, self.end_date, tz="UTC")

        trading_days = self.all_data['adj_close'].index
        return trading_days[(trading_days >= self.start_date) & (trading_days <= self.end_date)]

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
        if not self.all_data or self.all_data['adj_close'].empty:
            print("Cannot run backtest, no data pre-loaded.")
            return pd.DataFrame()
            
        print("Starting backtest...")
        
        adj_close_prices = self.all_data['adj_close']
        open_prices = self.all_data['open']
        
        for i, t_date in enumerate(self.trading_days):
            print(f"Processing {t_date.date()}...")
            
            # --- 1. Start-of-day: Execute scheduled orders ---
            orders_to_execute = self.scheduled_orders.pop(t_date, [])
            if orders_to_execute:
                open_prices_today = open_prices.loc[t_date] if t_date in open_prices.index else pd.Series()
                fills, rejected = self.execution_handler.simulate_execution(
                    orders=orders_to_execute,
                    open_prices_for_day=open_prices_today
                )
                for fill in fills:
                    self.portfolio.apply_fill(fill)
                # TODO: Log rejected orders

            # --- Get all symbols and close prices for end-of-day processes ---
            close_prices_today = adj_close_prices.loc[t_date] if t_date in adj_close_prices.index else pd.Series()
            
            # --- 2. End-of-day: Generate new orders on rebalance days ---
            if self._is_rebalance_day(t_date):
                current_equity = self.portfolio.mark_to_market(close_prices_today.to_dict())
                
                portfolio_state = {
                    'equity': current_equity,
                    'cash': self.portfolio.cash,
                    'positions': self.portfolio.positions,
                    'weights': self.portfolio.get_weights(close_prices_today.to_dict())
                }

                # Get historical data for the strategy
                hist_data_end_idx = adj_close_prices.index.get_loc(t_date)
                hist_data_start_idx = max(0, hist_data_end_idx - self.strategy.required_lookback)
                hist_data = adj_close_prices.iloc[hist_data_start_idx:hist_data_end_idx+1]
                
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
            self.portfolio.take_snapshot(t_date, close_prices_today.to_dict())

        print("Backtest complete.")
        return self.portfolio.history_df
