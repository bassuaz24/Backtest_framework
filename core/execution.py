from dataclasses import dataclass
from datetime import datetime
from core.portfolio import Fill
from core.data import DataHandler

@dataclass
class Order:
    """
    Represents a trade intent.
    """
    symbol: str
    shares: float  # Positive for buy, negative for sell
    generated_dt: datetime
    execute_dt: datetime
    style: str = "MKT" # Market order. Other options: LMT, STP, etc.

class BaseFillModel:
    """Abstract base class for fill models."""
    def get_fill_price(self, order: Order, data_handler: DataHandler) -> float | None:
        raise NotImplementedError

class NextOpenFillModel(BaseFillModel):
    """Fills orders at the next day's open price."""
    def get_fill_price(self, order: Order, data_handler: DataHandler) -> float | None:
        price = data_handler.get_price(order.symbol, order.execute_dt.strftime('%Y-%m-%d'), field='open')
        if price is None or price <= 0:
            return None
        return price

class BaseCostModel:
    """Abstract base class for cost models."""
    def calculate_fee(self, shares: float, price: float) -> float:
        raise NotImplementedError

class BasicCostModel(BaseCostModel):
    """
    A basic cost model with fixed basis points for commission and slippage.
    """
    def __init__(self, commission_bps: float = 0.0, slippage_bps: float = 0.0):
        self.commission_bps = commission_bps / 10000.0  # Convert bps to decimal
        self.slippage_bps = slippage_bps / 10000.0

    def calculate_fee(self, shares: float, price: float) -> float:
        notional = abs(shares * price)
        commission = self.commission_bps * notional
        slippage_cost = self.slippage_bps * notional
        return commission + slippage_cost

class ExecutionHandler:
    """
    Simulates the execution of orders, turning them into fills.
    """
    def __init__(self, fill_model: BaseFillModel, cost_model: BaseCostModel):
        self.fill_model = fill_model
        self.cost_model = cost_model

    def simulate_execution(
        self,
        orders: list[Order],
        data_handler: DataHandler
    ) -> tuple[list[Fill], list[dict]]:
        """
        Processes a list of orders for a given execution datetime.
        Returns a list of fills and a log of rejected orders.
        """
        fills = []
        rejected_log = []

        for order in orders:
            fill_price = self.fill_model.get_fill_price(order, data_handler)

            if fill_price is None:
                rejected_log.append({
                    'order': order,
                    'reason': f"Missing or invalid price for {order.symbol} on {order.execute_dt.date()}"
                })
                continue

            fee = self.cost_model.calculate_fee(order.shares, fill_price)
            
            fill = Fill(
                dt=order.execute_dt,
                symbol=order.symbol,
                shares=order.shares,
                price=fill_price,
                fee=fee,
            )
            fills.append(fill)

        return fills, rejected_log
