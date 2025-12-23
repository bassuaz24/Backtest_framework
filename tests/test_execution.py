import unittest
from unittest.mock import MagicMock
from datetime import datetime, date
from backtest.execution import Order, ExecutionHandler, NextOpenFillModel, BasicCostModel
from backtest.portfolio import Fill

class TestExecution(unittest.TestCase):

    def setUp(self):
        """Set up mock objects for each test."""
        self.mock_data_handler = MagicMock()
        self.fill_model = NextOpenFillModel()

    def test_price_selection(self):
        """Test 1: Correct fill price selection."""
        # --- Arrange ---
        # Mock the data handler to return a specific price
        self.mock_data_handler.get_price.return_value = 100.0
        
        cost_model = BasicCostModel() # No costs for this test
        execution_handler = ExecutionHandler(self.fill_model, cost_model)
        
        order = Order(
            symbol='AAPL',
            shares=10,
            generated_dt=datetime(2025, 12, 21),
            execute_dt=datetime(2025, 12, 22),
        )

        # --- Act ---
        fills, rejected = execution_handler.simulate_execution([order], self.mock_data_handler)

        # --- Assert ---
        # Ensure get_price was called correctly
        self.mock_data_handler.get_price.assert_called_once_with(
            'AAPL', '2025-12-22', field='open'
        )
        
        # Check the returned fill
        self.assertEqual(len(fills), 1)
        self.assertEqual(len(rejected), 0)
        self.assertEqual(fills[0].price, 100.0)
        self.assertEqual(fills[0].shares, 10)

    def test_costs(self):
        """Test 2: Correct fee calculation."""
        # --- Arrange ---
        self.mock_data_handler.get_price.return_value = 100.0
        
        # 10 bps for slippage, 1 bp for commission
        cost_model = BasicCostModel(commission_bps=1, slippage_bps=10)
        execution_handler = ExecutionHandler(self.fill_model, cost_model)
        
        order = Order(
            symbol='AAPL',
            shares=10,
            generated_dt=datetime(2025, 12, 21),
            execute_dt=datetime(2025, 12, 22),
        )
        
        # --- Act ---
        fills, rejected = execution_handler.simulate_execution([order], self.mock_data_handler)
        
        # --- Assert ---
        # notional = 10 * 100 = 1000
        # commission = 0.0001 * 1000 = 0.1
        # slippage = 0.001 * 1000 = 1.0
        # total_fee = 1.1
        self.assertEqual(len(fills), 1)
        self.assertAlmostEqual(fills[0].fee, 1.1)
        
    def test_missing_price_rejection(self):
        """Test 3: Order rejection when price is not available."""
        # --- Arrange ---
        # Mock the data handler to simulate a missing price
        self.mock_data_handler.get_price.return_value = None
        
        cost_model = BasicCostModel()
        execution_handler = ExecutionHandler(self.fill_model, cost_model)
        
        order = Order(
            symbol='AAPL',
            shares=10,
            generated_dt=datetime(2025, 12, 21),
            execute_dt=datetime(2025, 12, 22),
        )

        # --- Act ---
        fills, rejected = execution_handler.simulate_execution([order], self.mock_data_handler)
        
        # --- Assert ---
        self.assertEqual(len(fills), 0)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]['order'], order)
        self.assertIn('Missing or invalid price', rejected[0]['reason'])

if __name__ == '__main__':
    unittest.main()
