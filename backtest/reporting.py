
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pathlib
reports_dir = pathlib.Path("reports")
reports_dir.mkdir(exist_ok=True)

from tabulate import tabulate

class BacktestReport:
    def __init__(self, daily_snapshots: pd.DataFrame, fills: pd.DataFrame, benchmark: pd.DataFrame = None):
        self.daily_snapshots = daily_snapshots
        self.fills = fills
        self.benchmark = benchmark
        self.returns = self._calculate_returns()
        self.metrics = self._calculate_metrics()

    def _calculate_returns(self):
        return self.daily_snapshots['equity'].pct_change().fillna(0)

    def _calculate_metrics(self):
        total_return = (self.daily_snapshots['equity'].iloc[-1] / self.daily_snapshots['equity'].iloc[0]) - 1
        
        days = (self.daily_snapshots.index[-1] - self.daily_snapshots.index[0]).days
        cagr = (1 + total_return) ** (365.25 / days) - 1 if days > 0 else 0

        annualized_volatility = self.returns.std() * np.sqrt(252)
        
        sharpe_ratio = (cagr / annualized_volatility) if annualized_volatility != 0 else 0

        downside_returns = self.returns[self.returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (cagr / downside_volatility) if downside_volatility != 0 else 0

        cumulative_returns = (1 + self.returns).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1
        max_drawdown = drawdown.min()
        
        calmar_ratio = (cagr / abs(max_drawdown)) if max_drawdown != 0 else 0

        metrics = {
            "Total Return": f"{total_return:.2%}",
            "CAGR": f"{cagr:.2%}",
            "Annualized Volatility": f"{annualized_volatility:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Sortino Ratio": f"{sortino_ratio:.2f}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Calmar Ratio": f"{calmar_ratio:.2f}"
        }
        return metrics

    def display_summary(self):
        reports_dir = pathlib.Path("reports")
        reports_dir.mkdir(exist_ok=True)

        summary_text = tabulate(self.metrics.items(), headers=["Metric", "Value"], tablefmt="grid")
        print("\n--- Backtest Summary ---")
        print(summary_text)

        summary_path = reports_dir / "summary.txt"
        with summary_path.open("w") as f:
            f.write(summary_text)

    def plot_equity(self):
        sns.set(style='darkgrid')
        plt.figure(figsize=(12, 6))
        
        # Plot portfolio equity
        self.daily_snapshots['equity'].plot(label='Portfolio', color='blue')
        
        # Plot benchmark if available
        if self.benchmark is not None:
            benchmark_series = self.benchmark.iloc[:, 0]
            start_equity = self.daily_snapshots['equity'].iloc[0]
            rebased_benchmark = (benchmark_series /benchmark_series.iloc[0]) * start_equity
            rebased_benchmark.plot(label='Benchmark', color='grey', linestyle='--')

        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.legend()
        plt.savefig(reports_dir / "equity_curve.png", bbox_inches="tight")
        plt.close()


    def plot_drawdown(self):
        sns.set(style='darkgrid')
        plt.figure(figsize=(12, 6))

        cumulative_returns = (1 + self.returns).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1
        
        drawdown.plot(color='red')
        plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        
        plt.title('Drawdown Curve (Underwater Plot)')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.legend()
        plt.savefig(reports_dir / "drawdown_curve.png", bbox_inches="tight")
        plt.close()

if __name__ == '__main__':
    # Example usage:
    # Load your backtest results here
    # daily_snapshots = pd.read_csv('daily_snapshots.csv', index_col='date', parse_dates=True)
    # fills = pd.read_csv('fills.csv', index_col='dt', parse_dates=True)
    # report = BacktestReport(daily_snapshots, fills)
    # report.display_summary()
    # report.plot_equity()
    # report.plot_drawdown()
    pass
