from dataclasses import dataclass
import pandas as pd

@dataclass
class BacktestResult:
    """
    Contains all the results of a backtest run.
    """
    daily_snapshots: pd.DataFrame
    fills: pd.DataFrame
    metrics: dict
    benchmark: pd.DataFrame = None
