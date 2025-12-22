import unittest
from datetime import datetime
from backtest.portfolio import Portfolio, Fill

class TestPortfolio(unittest.TestCase):

    def test_buy_and_mark_to_market(self):
        """Test 1: Buy a position and mark it to market."""
        portfolio = Portfolio(initial_cash=1000.0)
        
        # Define the fill
        fill = Fill(dt=datetime.now(), symbol='AAPL', shares=10, price=10.0, fee=1.0)
        
        # Apply the fill
        portfolio.apply_fill(fill)
        
        # --- Assertions ---
        # Check cash: 1000 - (10 * 10) - 1 = 899
        self.assertAlmostEqual(portfolio.cash, 899.0)
        
        # Check positions
        self.assertEqual(portfolio.positions.get('AAPL'), 10)
        
        # Check fees
        self.assertAlmostEqual(portfolio.fees_paid, 1.0)
        
        # Mark to market at the buy price
        equity = portfolio.mark_to_market(prices={'AAPL': 10.0})
        # Equity should be cash + position_value = 899 + (10 * 10) = 999
        self.assertAlmostEqual(equity, 999.0)
        
        # Mark to market at a different price
        equity_new_price = portfolio.mark_to_market(prices={'AAPL': 12.0})
        # Equity should be cash + position_value = 899 + (10 * 12) = 1019
        self.assertAlmostEqual(equity_new_price, 1019.0)

    def test_sell_partial(self):
        """Test 2: Sell a partial position."""
        portfolio = Portfolio(initial_cash=1000.0)
        
        # Initial buy
        buy_fill = Fill(dt=datetime.now(), symbol='AAPL', shares=10, price=10.0, fee=1.0)
        portfolio.apply_fill(buy_fill)
        # Cash is now 899

        # Partial sell
        sell_fill = Fill(dt=datetime.now(), symbol='AAPL', shares=-5, price=12.0, fee=1.0)
        portfolio.apply_fill(sell_fill)

        # --- Assertions ---
        # Check cash: 899 + (-5 * 12 * -1) - 1 = 899 + 60 - 1 = 958
        self.assertAlmostEqual(portfolio.cash, 958.0)
        
        # Check position: 10 - 5 = 5
        self.assertEqual(portfolio.positions.get('AAPL'), 5)
        
        # Check fees: 1 (buy) + 1 (sell) = 2
        self.assertAlmostEqual(portfolio.fees_paid, 2.0)
        
        # Mark to market
        equity = portfolio.mark_to_market(prices={'AAPL': 12.0})
        # Equity = cash + position_value = 958 + (5 * 12) = 1018
        self.assertAlmostEqual(equity, 1018.0)

    def test_round_trip(self):
        """Test 3: A full round trip (buy then sell all)."""
        portfolio = Portfolio(initial_cash=1000.0)
        
        # 1. Buy
        buy_fill = Fill(dt=datetime.now(), symbol='AAPL', shares=10, price=10.0, fee=1.0)
        portfolio.apply_fill(buy_fill)
        # Cash: 1000 - 100 - 1 = 899

        # 2. Sell all
        sell_fill = Fill(dt=datetime.now(), symbol='AAPL', shares=-10, price=12.0, fee=1.0)
        portfolio.apply_fill(sell_fill)

        # --- Assertions ---
        # Final cash = initial_cash + PnL - total_fees
        # PnL = (12.0 - 10.0) * 10 = 20
        # Total fees = 1.0 (buy) + 1.0 (sell) = 2.0
        # Final cash = 1000 + 20 - 2 = 1018
        self.assertAlmostEqual(portfolio.cash, 1018.0)
        
        # Position should be gone
        self.assertNotIn('AAPL', portfolio.positions)
        
        # Total fees
        self.assertAlmostEqual(portfolio.fees_paid, 2.0)
        
        # Equity should equal final cash since there are no positions
        equity = portfolio.mark_to_market(prices={})
        self.assertAlmostEqual(equity, 1018.0)

if __name__ == '__main__':
    unittest.main()
