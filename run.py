import os
from dotenv import load_dotenv
import pandas as pd
import pathlib

from backtest.data import DataHandler
from backtest.portfolio import Portfolio
from backtest.execution import ExecutionHandler, NextOpenFillModel, BasicCostModel
from backtest.engine import BacktestEngine
from backtest.strategies.mean_reversion import MeanReversionStrategy

def main():
    """
    Main function to set up and run a backtest.
    """
    load_dotenv(dotenv_path='alpaca.env')

    # --- Configuration ---
    START_DATE = "2022-01-01"
    END_DATE = "2024-12-31"
    INITIAL_CASH = 100000.0
    UNIVERSE = ['NVDA', 'ORCL'] # Start with a single ETF
    
    # --- Component Setup ---
    data_handler = DataHandler(data_dir='data')
    
    # Pre-load data if not already present
    # In a real application, you might have a separate ingestion step.
    # Here, we'll just ensure the data we need is available.
    # A check can be added here to see if data exists before downloading.
    print("Ensuring data is available for backtest universe...")
    data_handler.run_ingestion(symbols=UNIVERSE, start="2019-01-01", end=END_DATE) # fetch extra for lookbacks

    portfolio = Portfolio(initial_cash=INITIAL_CASH)
    
    fill_model = NextOpenFillModel()
    cost_model = BasicCostModel(commission_bps=1.0, slippage_bps=5.0) # 1 bp commission, 5 bps slippage
    execution_handler = ExecutionHandler(fill_model, cost_model)

    strategy = MeanReversionStrategy(
        universe=UNIVERSE,
        lookback_short=5,
        lookback_vol=20,
        entry_z=1.5,
        exit_z=0.5
    )

    engine = BacktestEngine(
        start_date=START_DATE,
        end_date=END_DATE,
        data_handler=data_handler,
        execution_handler=execution_handler,
        portfolio=portfolio,
        strategy=strategy,
        rebalance_schedule='D' # Rebalance daily for this strategy
    )
    
    # --- Run Backtest ---
    results = engine.run_backtest()
    
    # --- Analysis / Reporting ---
    print("\n--- Generating Report ---")

    # Create a unique directory for this backtest run
    tickers_str = "_".join(UNIVERSE)
    report_dir_name = f"{START_DATE}_to_{END_DATE}_{tickers_str}"
    report_dir = pathlib.Path("reports") / report_dir_name
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch benchmark data (SPY)
    print("Fetching benchmark data (SPY)...")
    data_handler.run_ingestion(symbols=['SPY'], start="2019-01-01", end=END_DATE)
    spy_data = data_handler.get_history(symbols=['SPY'], end_date=END_DATE, lookback_days=1304, field="adj_close")
    # Have gemini check lookback_days

    from backtest.reporting import BacktestReport
    report = BacktestReport(
        daily_snapshots=results,
        fills=portfolio.fills_df,
        benchmark=spy_data,
        report_dir=report_dir
    )
    
    report.display_summary()
    report.plot_equity()
    report.plot_drawdown()
    
    # Save results to a file
    results.to_csv(report_dir / 'backtest_results.csv')
    portfolio.fills_df.to_csv(report_dir / 'fills_log.csv')
    print(f"\nSaved results to '{report_dir}'")


if __name__ == "__main__":
    main()
