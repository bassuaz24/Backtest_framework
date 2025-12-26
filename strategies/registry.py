from strategies.mean_reversion import MeanReversionStrategy

STRATEGY_REGISTRY = {
    "mean_reversion": {
        "class": MeanReversionStrategy,
        "display_name": "Mean Reversion",
        "params": {
            "lookback_short": {
                "type": "int",
                "min": 5,
                "max": 30,
                "step": 1,
                "default": 5,
                "help": "Short-term lookback period for return calculation."
            },
            "lookback_vol": {
                "type": "int",
                "min": 10,
                "max": 60,
                "step": 1,
                "default": 20,
                "help": "Lookback period for volatility calculation."
            },
            "entry_z": {
                "type": "float",
                "min": 1.0,
                "max": 3.0,
                "step": 0.1,
                "default": 1.5,
                "help": "The Z-score threshold to trigger a buy signal."
            },
            "exit_z": {
                "type": "float",
                "min": 0.0,
                "max": 2.0,
                "step": 0.1,
                "default": 0.5,
                "help": "The Z-score threshold to trigger a sell signal."
            }
        }
    }
}
