import os
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from functools import lru_cache
from pathlib import Path

class DataHandler:
    """
    Handles data ingestion, cleaning, and access for the backtesting framework.
    """

    def __init__(self, data_dir: str = 'data'):
        self.alpaca_client = StockHistoricalDataClient(
            api_key=os.getenv("ALPACA_API_KEY"),
            secret_key=os.getenv("ALPACA_SECRET_KEY")
        )
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.clean_dir = self.data_dir / 'clean' / 'bars'
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.clean_dir.mkdir(parents=True, exist_ok=True)

        # Step 0: Canonical Bar Schema (as a reference)
        self.canonical_schema = {
            'open': 'float64',
            'high': 'float64',
            'low': 'float64',
            'close': 'float64',
            'volume': 'int64',
            'adj_close': 'float64'
        }

    def run_ingestion(self, symbols: list[str], start: str, end: str):
        """
        Runs the entire data ingestion pipeline, but only for symbols that
        don't have up-to-date data locally.
        """
        required_start = pd.to_datetime(start).tz_localize('UTC')
        required_end = pd.to_datetime(end).tz_localize('UTC')

        symbols_to_ingest = []
        for symbol in symbols:
            try:
                # Efficiently read only the date column for the specific symbol from the partitioned dataset
                date_df = pd.read_parquet(
                    self.clean_dir,
                    engine='pyarrow',
                    filters=[('symbol', '==', symbol)],
                    columns=['date']
                )

                if date_df.empty:
                    # No data for this symbol, so we need to ingest it.
                    symbols_to_ingest.append(symbol)
                    continue
                
                min_date = date_df['date'].min()
                max_date = date_df['date'].max()

                if required_start < min_date or required_end > max_date:
                    symbols_to_ingest.append(symbol)
            except Exception as e:
                # This might happen if the directory is corrupted or on certain filesystem errors.
                print(f"Could not read existing data for {symbol}, will re-ingest. Error: {e}")
                symbols_to_ingest.append(symbol)
        
        if not symbols_to_ingest:
            print("All required data is already present locally.")
            return

        print(f"Starting data ingestion for {len(symbols_to_ingest)} symbols...")
        
        # Add a buffer to the date range to ensure we get data around holidays.
        buffered_start = (pd.to_datetime(start) - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
        buffered_end = (pd.to_datetime(end) + pd.Timedelta(days=7)).strftime('%Y-%m-%d')
        
        raw_df = self._download_raw_data(symbols_to_ingest, buffered_start, buffered_end)
        
        if raw_df.empty:
            print("No new raw data was downloaded.")
            return
            
        print(f"Downloaded {len(raw_df)} raw bars.")
        
        normalized_df = self._normalize_data(raw_df)
        print("Normalized data.")

        self._validate_data(normalized_df)
        print("Validated data.")

        adjusted_df = self._adjust_for_corporate_actions(normalized_df)
        print("Adjusted for corporate actions.")

        self._save_clean_data(adjusted_df)
        print(f"Saved clean data to {self.clean_dir}")
        print("Data ingestion complete.")

    def _download_raw_data(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        """
        Step 1: Download raw data from Alpaca.
        """
        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=pd.to_datetime(start).tz_localize('UTC'),
            end=pd.to_datetime(end).tz_localize('UTC'),
            adjustment='raw'
        )
        bars = self.alpaca_client.get_stock_bars(request_params)
        df = bars.df
        df.reset_index(inplace=True)
        
        # Store raw data
        for symbol in symbols:
            symbol_df = df[df['symbol'] == symbol]
            if not symbol_df.empty:
                output_path = self.raw_dir / f"{symbol}_{start}_to_{end}.parquet"
                symbol_df.to_parquet(output_path)
                
        return df

    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2: Normalize raw data into the canonical schema.
        """
        df.rename(columns={'timestamp': 'date'}, inplace=True)
        
        # Ensure timezone is normalized (Alpaca returns UTC)
        df['date'] = pd.to_datetime(df['date']).dt.tz_convert('UTC').dt.normalize()
        
        # Sort and remove duplicates
        df.sort_values(by=['symbol', 'date'], inplace=True)
        df.drop_duplicates(subset=['symbol', 'date'], keep='first', inplace=True)
        
        return df

    def _validate_data(self, df: pd.DataFrame):
        """
        Step 3: Perform basic quality checks.
        """
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
             raise ValueError("Date column is not of datetime type.")
        
        for symbol, group in df.groupby('symbol'):
            if not group['date'].is_monotonic_increasing:
                raise ValueError(f"Dates for symbol {symbol} are not monotonic increasing.")
            if (group['high'] < group[['open', 'close']].max(axis=1)).any():
                raise ValueError(f"High is not the highest price for symbol {symbol}.")
            if (group['low'] > group[['open', 'close']].min(axis=1)).any():
                raise ValueError(f"Low is not the lowest price for symbol {symbol}.")
            if (group['volume'] < 0).any():
                raise ValueError(f"Volume is negative for symbol {symbol}.")
    
    def _adjust_for_corporate_actions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 4: Handle corporate actions by getting adjusted data.
        This function now performs a left merge to keep all raw data and fills
        missing adjusted values with raw values to prevent data loss.
        """
        # df contains raw data, including unadjusted ohlc and volume
        unadjusted_df = df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']].copy()

        symbols = df['symbol'].unique().tolist()
        start = df['date'].min().strftime('%Y-%m-%d')
        end = df['date'].max().strftime('%Y-%m-%d')

        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=pd.to_datetime(start).tz_localize('UTC'),
            end=pd.to_datetime(end).tz_localize('UTC'),
            adjustment='all' # gets fully adjusted data
        )
        adj_bars = self.alpaca_client.get_stock_bars(request_params)
        adj_df = adj_bars.df.reset_index()
        adj_df.rename(columns={
            'timestamp': 'date',
            # 'open', 'high', 'low' from this request are the adjusted values
            'close': 'adj_close' # This is adjusted close
        }, inplace=True)
        adj_df['date'] = pd.to_datetime(adj_df['date']).dt.tz_convert('UTC').dt.normalize()
        
        # We only need the adjusted ohl and adj_close from this
        adj_df = adj_df[['date', 'symbol', 'open', 'high', 'low', 'adj_close']]

        # Perform a left merge to keep all data from the raw source.
        # Pandas automatically adds _x and _y suffixes to overlapping column names.
        final_df = pd.merge(unadjusted_df, adj_df, on=['date', 'symbol'], how='left')

        # Coalesce adjusted and unadjusted data. Prefer adjusted (_y) but fallback to raw (_x).
        # The 'close' column from unadjusted_df remains the unadjusted close.
        final_df['open'] = final_df['open_y'].fillna(final_df['open_x'])
        final_df['high'] = final_df['high_y'].fillna(final_df['high_x'])
        final_df['low'] = final_df['low_y'].fillna(final_df['low_x'])

        # For adj_close, we fallback to the unadjusted 'close'
        final_df['adj_close'] = final_df['adj_close'].fillna(final_df['close'])
        
        # Ensure final dataframe contains all original rows.
        if len(final_df) != len(unadjusted_df):
             raise ValueError("Data loss occurred unexpectedly during left merge.")

        # Reorder to match canonical schema, selecting the newly created coalesced columns.
        final_df = final_df[['date', 'symbol', 'open', 'high', 'low', 'close', 'adj_close', 'volume']]
        
        return final_df

    def _save_clean_data(self, df: pd.DataFrame):
        """
        Step 5: Store data in a layout optimized for backtests (partitioned by symbol).
        We partition by symbol to avoid pyarrow's max partition limit when using a 
        long date range.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(df)
        
        pq.write_to_dataset(
            table,
            root_path=self.clean_dir,
            partition_cols=['symbol'],
            existing_data_behavior='delete_matching'
        )

    # Step 6 & 7: Data API with caching
    def get_bars(self, symbols: list[str] | str, start: str, end: str) -> pd.DataFrame:
        """
        Public method to load bar data. Converts list to tuple for caching.
        """
        if isinstance(symbols, list):
            symbols = tuple(symbols)
        return self._get_bars_cached(symbols, start, end)

    def get_history(self, symbols: list[str] | str, end_date: str, lookback_days: int, field: str = 'adj_close') -> pd.DataFrame:
        """
        Public method to get historical data. Converts list to tuple for caching.
        """
        if isinstance(symbols, list):
            symbols = tuple(symbols)
        return self._get_history_cached(symbols, end_date, lookback_days, field)

    @lru_cache(maxsize=1024)
    def get_price(self, symbol: str, date: str, field: str = 'adj_close') -> float | None:
        """
        Gets a single price point for a symbol and date.
        """
        bars = self.get_bars(symbol, date, date)
        if not bars.empty:
            return bars.iloc[0][field]
        return None

    @lru_cache(maxsize=128)
    def _get_bars_cached(self, symbols: tuple[str] | str, start: str, end:str) -> pd.DataFrame:
        """
        Cached implementation for loading bar data. Assumes symbols are hashable.
        """
        if isinstance(symbols, str):
            symbols = (symbols,) # Convert single string to tuple
        
        def to_utc(ts):
            ts = pd.to_datetime(ts)
            if ts.tzinfo is None:
                return ts.tz_localize('UTC')
            return ts.tz_convert('UTC')

        filters = [
            ('symbol', 'in', symbols),
            ('date', '>=', to_utc(start)),
            ('date', '<=', to_utc(end))
        ]
        
        df = pd.read_parquet(self.clean_dir, engine='pyarrow', filters=filters)
        return df.copy()

    @lru_cache(maxsize=128)
    def _get_history_cached(self, symbols: tuple[str] | str, end_date: str, lookback_days: int, field: str = 'adj_close') -> pd.DataFrame:
        """
        Cached implementation for getting historical data. Assumes symbols are hashable.
        """
        start_date = (pd.to_datetime(end_date) - pd.tseries.offsets.BDay(lookback_days)).strftime('%Y-%m-%d')
        bars = self.get_bars(symbols, start_date, end_date)
        
        if bars.empty:
            return pd.DataFrame()
        
        history = bars.pivot(index='date', columns='symbol', values=field)
        return history
