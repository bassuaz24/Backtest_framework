import pandas as pd
import pathlib

from core.data import DataHandler
from core.portfolio import Portfolio
from core.execution import ExecutionHandler, NextOpenFillModel, BasicCostModel
from core.engine import BacktestEngine
from strategies.registry import STRATEGY_REGISTRY
from reporting.compute_report import BacktestReport
from core.backtest_result import BacktestResult
from dotenv import load_dotenv

def run_backtest(config: dict) -> BacktestResult:
    """
    Runs a backtest with the given configuration.

    Args:
        config: A dictionary containing the backtest configuration.

    Returns:
        A BacktestResult object containing the results.
    """
    load_dotenv(dotenv_path='alpaca.env')
    
    # --- Component Setup ---
    data_handler = DataHandler(data_dir='data')
    
    print("Ensuring data is available for backtest universe and benchmark...")
    symbols_to_ingest = config['universe'] + [config.get('benchmark', 'SPY')]
    data_handler.run_ingestion(symbols=list(set(symbols_to_ingest)), start="2019-01-01", end=config['end_date'])

    portfolio = Portfolio(initial_cash=config['initial_cash'])
    
    fill_model = NextOpenFillModel()
    cost_model = BasicCostModel(commission_bps=1.0, slippage_bps=5.0)
    execution_handler = ExecutionHandler(fill_model, cost_model)

    strategy_info = STRATEGY_REGISTRY[config['strategy']]
    strategy_class = strategy_info['class']
    strategy_params = {k: v for k, v in config.items() if k in strategy_info['params']}
    strategy = strategy_class(universe=config['universe'], **strategy_params)

    engine = BacktestEngine(
        start_date=config['start_date'],
        end_date=config['end_date'],
        data_handler=data_handler,
        execution_handler=execution_handler,
        portfolio=portfolio,
        strategy=strategy,
        rebalance_schedule='D'
    )
    
    # --- Run Backtest ---
    daily_snapshots = engine.run_backtest()
    
    # --- Analysis / Reporting ---
    print("\n--- Generating Report ---")

    report_dir = pathlib.Path("reports") # Reports are temporary for the UI
    
    # Fetch benchmark data
    benchmark_symbol = config.get('benchmark', 'SPY')
    print(f"Fetching benchmark data ({benchmark_symbol})...")
    # Calculate lookback days needed
    start_date = pd.to_datetime(config['start_date'])
    end_date = pd.to_datetime(config['end_date'])
    lookback_days = (end_date - start_date).days + 252 # Add a year for rolling metrics
    
    benchmark_data = data_handler.get_history(symbols=[benchmark_symbol], end_date=config['end_date'], lookback_days=lookback_days, field="adj_close")

    report = BacktestReport(
        daily_snapshots=daily_snapshots,
        fills=portfolio.fills_df,
        benchmark=benchmark_data,
        report_dir=report_dir
    )
    
    metrics = report.metrics
    
    return BacktestResult(
        daily_snapshots=daily_snapshots,
        fills=portfolio.fills_df,
        metrics=metrics,
        benchmark=benchmark_data
    )
