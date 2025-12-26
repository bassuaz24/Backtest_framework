import pandas as pd

def target_weights_to_quantities(
    target_weights: dict[str, float],
    current_positions: dict[str, float],
    equity: float,
    close_prices: pd.Series
) -> dict[str, float]:
    """
    Converts a dictionary of target weights into a dictionary of target quantities.

    Args:
        target_weights: A dictionary mapping symbols to their target weights.
        current_positions: A dictionary of current holdings (symbol -> shares).
        equity: The current total equity of the portfolio.
        close_prices: A Series of the most recent close prices.

    Returns:
        A dictionary of {symbol: quantity} for orders to be placed.
    """
    quantities = {}
    
    all_symbols = set(target_weights.keys()) | set(current_positions.keys())

    for symbol in all_symbols:
        target_w = target_weights.get(symbol, 0.0)
        current_shares = current_positions.get(symbol, 0.0)
        
        price = close_prices.get(symbol)
        if price is None or pd.isna(price) or price <= 0:
            continue

        target_shares = (target_w * equity) / price
        order_qty = target_shares - current_shares
        
        # Simple rounding and dust filter
        order_qty = round(order_qty)
        if abs(order_qty) < 1:
            continue
            
        quantities[symbol] = order_qty
        
    return quantities
