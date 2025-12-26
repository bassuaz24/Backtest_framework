import streamlit as st
import pandas as pd
import pathlib
from datetime import datetime, date

from core.runner import run_backtest
from strategies.registry import STRATEGY_REGISTRY
import reporting.plots as plots

from dotenv import load_dotenv
load_dotenv(dotenv_path='alpaca.env')

def get_available_symbols():
    """Gets available symbols from the data directory."""
    data_dir = pathlib.Path("data/clean/bars")
    if not data_dir.exists():
        return []
    return sorted([p.name.split('=')[1] for p in data_dir.iterdir() if p.is_dir()])

def get_sp500_symbols():
    """Gets S&P 500 symbols from the csv file."""
    if not pathlib.Path("sp500_tickers.csv").exists():
        return []
    df = pd.read_csv("sp500_tickers.csv")
    return sorted(df['Symbol'].tolist())

def main():
    """
    Streamlit UI for the backtesting framework.
    """
    st.set_page_config(layout="wide")
    st.title("Backtesting Framework")

    sp500_symbols = get_sp500_symbols()
    if not sp500_symbols:
        st.error("S&P 500 ticker list not found. Please make sure 'sp500_tickers.csv' is in the root directory.")
        st.stop()
    
    # --- State Management ---
    if "ran" not in st.session_state:
        st.session_state["ran"] = False
    if "result" not in st.session_state:
        st.session_state["result"] = None
    if "last_config" not in st.session_state:
        st.session_state["last_config"] = {}

    # --- Sidebar ---
    with st.sidebar.form("controls"):
        st.header("Configuration")
        
        # --- General Settings ---
        today = date.today()
        start_date = st.date_input("Start Date", date(2022, 1, 1), max_value=today)
        end_date = st.date_input("End Date", date(2024, 12, 31), max_value=today)
        initial_cash = st.number_input("Initial Cash", value=100000.0, step=10000.0, format="%.2f")
        
        st.header("Universe")
        selected_symbols_multiselect = st.multiselect("Select S&P 500 Symbols", options=sp500_symbols, default=[])
        select_all = st.checkbox("Use all S&P 500 symbols")

        benchmarks = ["SPY", "QQQ", "DIA", "IWM", "AGG"]
        benchmark = st.selectbox("Benchmark", options=benchmarks, index=0)

        # --- Strategy Selection ---
        st.header("Strategy")
        strategy_name = st.selectbox("Select Strategy", options=list(STRATEGY_REGISTRY.keys()), format_func=lambda k: STRATEGY_REGISTRY[k]['display_name'])
        
        strategy_params = {}
        param_schema = STRATEGY_REGISTRY[strategy_name]['params']
        for param, schema in param_schema.items():
            if schema['type'] == 'int':
                strategy_params[param] = st.slider(schema.get('help', param), min_value=schema['min'], max_value=schema['max'], value=schema['default'], step=schema['step'])
            elif schema['type'] == 'float':
                strategy_params[param] = st.slider(schema.get('help', param), min_value=schema['min'], max_value=schema['max'], value=schema['default'], step=schema['step'])

        submitted = st.form_submit_button("Run Backtest")

    # --- Backtest Execution ---
    if submitted:

        if select_all:
            final_selected_symbols = sp500_symbols
        else:
            final_selected_symbols = selected_symbols_multiselect

        # --- Validation ---
        if start_date > today:
            st.error("Start date cannot be in the future.")
            st.stop()
        if end_date > today:
            st.error("End date cannot be in the future.")
            st.stop()
        if start_date > end_date:
            st.error("End date cannot be before start date.")
            st.stop()
        if not final_selected_symbols:
            st.error("Please select at least one symbol for the universe or check 'Use all S&P 500 symbols'.")
            st.stop()
        
        config = {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "initial_cash": initial_cash,
            "universe": final_selected_symbols,
            "benchmark": benchmark,
            "strategy": strategy_name,
            **strategy_params
        }
        
        with st.spinner("Running backtest... (this may take up to 20 minutes for large universes)"):
            try:
                result = run_backtest(config)
                st.session_state["result"] = result
                st.session_state["ran"] = True
                st.session_state["last_config"] = config
                st.success("Backtest complete!")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state["ran"] = False

    # --- Results Display ---
    if st.session_state["ran"]:
        result = st.session_state["result"]
        config = st.session_state["last_config"]

        st.header("Results")

        # --- Summary Metrics ---
        st.subheader("Summary Metrics")
        st.dataframe(pd.DataFrame.from_dict(result.metrics, orient='index', columns=['Value']))

        # --- Run Metadata ---
        with st.expander("Run Metadata"):
            st.json(config)

        # --- Tabs ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Equity & Drawdown", "Rolling Sharpe/Vol", "Trades", "Exposure & Turnover", "Benchmark"])

        with tab1:
            st.plotly_chart(plots.plot_equity_curve(result.daily_snapshots, result.benchmark), use_container_width=True)
            st.plotly_chart(plots.plot_drawdown_curve(result.daily_snapshots), use_container_width=True)
        
        with tab2:
            rolling_window = st.slider("Rolling Window", 30, 252, 63)
            st.plotly_chart(plots.plot_rolling_sharpe_vol(result.daily_snapshots, window=rolling_window), use_container_width=True)

        with tab3:
            st.subheader("Trade Blotter")
            st.dataframe(result.fills)
            
            csv = result.fills.to_csv().encode('utf-8')
            st.download_button(
                label="Download Trades",
                data=csv,
                file_name="trades.csv",
                mime="text/csv",
            )

        with tab4:
            st.plotly_chart(plots.plot_exposure_turnover(result.daily_snapshots), use_container_width=True)

        with tab5:
            st.plotly_chart(plots.plot_benchmark_comparison(result.daily_snapshots, result.benchmark), use_container_width=True)


if __name__ == "__main__":
    main()