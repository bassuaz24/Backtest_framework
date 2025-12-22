import argparse
import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data import DataHandler

def main():
    """
    Main function to run the data ingestion script.
    """
    # Load environment variables from alpaca.env if it exists
    dotenv_path = 'alpaca.env'
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
    else:
        # Fallback to default .env or environment variables
        load_dotenv()

    parser = argparse.ArgumentParser(description="Data ingestion script for the backtesting framework.")
    parser.add_argument('--symbols', nargs='+', required=True, help='List of stock symbols to ingest.')
    parser.add_argument('--start', required=True, help='Start date in YYYY-MM-DD format.')
    parser.add_argument('--end', required=True, help='End date in YYYY-MM-DD format.')
    parser.add_argument('--data-dir', default='data', help='Directory to store data.')

    args = parser.parse_args()

    # Check for Alpaca API keys
    if "ALPACA_API_KEY" not in os.environ or "ALPACA_SECRET_KEY" not in os.environ:
        print("Error: Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.")
        print("You can create a .env file in the root directory with these keys.")
        return

    data_handler = DataHandler(data_dir=args.data_dir)
    data_handler.run_ingestion(
        symbols=args.symbols,
        start=args.start,
        end=args.end
    )

if __name__ == "__main__":
    main()
