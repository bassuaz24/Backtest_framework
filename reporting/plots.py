import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_equity_curve(daily_snapshots: pd.DataFrame, benchmark: pd.DataFrame = None):
    """Generates an equity curve plot using Plotly."""
    fig = go.Figure()

    # Add portfolio equity
    fig.add_trace(go.Scatter(
        x=daily_snapshots.index,
        y=daily_snapshots['equity'],
        mode='lines',
        name='Portfolio',
        line=dict(color='blue')
    ))

    # Add benchmark if available and not empty
    if benchmark is not None and not benchmark.empty:
        benchmark_series = benchmark.iloc[:, 0]
        start_equity = daily_snapshots['equity'].iloc[0]
        rebased_benchmark = (benchmark_series / benchmark_series.iloc[0]) * start_equity
        fig.add_trace(go.Scatter(
            x=rebased_benchmark.index,
            y=rebased_benchmark,
            mode='lines',
            name='Benchmark',
            line=dict(color='grey', dash='dash')
        ))

    fig.update_layout(
        title='Equity Curve',
        xaxis_title='Date',
        yaxis_title='Equity',
        legend_title='Legend'
    )
    return fig

def plot_drawdown_curve(daily_snapshots: pd.DataFrame):
    """Generates a drawdown curve plot using Plotly."""
    returns = daily_snapshots['equity'].pct_change().fillna(0)
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown,
        mode='lines',
        name='Drawdown',
        fill='tozeroy',
        line=dict(color='red')
    ))

    fig.update_layout(
        title='Drawdown Curve (Underwater Plot)',
        xaxis_title='Date',
        yaxis_title='Drawdown',
    )
    return fig

def plot_rolling_sharpe_vol(daily_snapshots: pd.DataFrame, window: int = 252):
    """Generates rolling Sharpe and Volatility plots."""
    returns = daily_snapshots['equity'].pct_change().fillna(0)
    
    rolling_vol = returns.rolling(window=window).std() * (252**0.5)
    rolling_sharpe = returns.rolling(window=window).mean() * 252 / rolling_vol

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Rolling Volatility', 'Rolling Sharpe Ratio'))

    fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol, name='Volatility'), row=1, col=1)
    fig.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe, name='Sharpe Ratio'), row=2, col=1)

    fig.update_layout(title_text=f"{window}-Day Rolling Metrics")
    return fig


def plot_exposure_turnover(daily_snapshots: pd.DataFrame):
    """Generates exposure and turnover plots."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Exposure', 'Turnover'))

    fig.add_trace(go.Scatter(x=daily_snapshots.index, y=daily_snapshots['gross_exposure'], name='Gross Exposure'), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily_snapshots.index, y=daily_snapshots['net_exposure'], name='Net Exposure'), row=1, col=1)

    # Turnover is not calculated in the daily snapshots yet. I will add it later.
    # For now, I'll leave the turnover plot empty.
    
    fig.update_layout(title_text="Exposure and Turnover")
    return fig

def plot_benchmark_comparison(daily_snapshots: pd.DataFrame, benchmark: pd.DataFrame):
    """Generates benchmark comparison plots."""
    
    if benchmark is None or benchmark.empty:
        fig = go.Figure()
        fig.update_layout(title_text="Benchmark Comparison (Benchmark data not available)")
        return fig

    returns = daily_snapshots['equity'].pct_change().fillna(0)
    benchmark_returns = benchmark.iloc[:,0].pct_change().fillna(0)

    # Rebased equity
    strat_equity_rebased = (1 + returns).cumprod()
    bench_equity_rebased = (1 + benchmark_returns).cumprod()

    # Excess return
    excess_return = returns - benchmark_returns
    excess_return_curve = (1 + excess_return).cumprod()


    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Rebased Equity Curves', 'Excess Return vs Benchmark'))

    fig.add_trace(go.Scatter(x=strat_equity_rebased.index, y=strat_equity_rebased, name='Strategy'), row=1, col=1)
    fig.add_trace(go.Scatter(x=bench_equity_rebased.index, y=bench_equity_rebased, name='Benchmark'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=excess_return_curve.index, y=excess_return_curve, name='Excess Return'), row=2, col=1)

    fig.update_layout(title_text="Benchmark Comparison")
    return fig
