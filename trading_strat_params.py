"""
Trading Strategy Parameters

This file contains all configuration parameters for the trading bot.
Modify these parameters carefully as they directly impact trading behavior and risk management.
"""

# Trading pairs configuration
# Note: Not all pairs may be available. The bot will automatically skip unavailable pairs.
# Recommended: Start with major pairs (BTC, ETH) which have higher liquidity
# Consider: Trading volume, spread, and volatility when selecting pairs
TRADING_PAIRS = [
    ('XXBTZEUR', 'BTC/EUR'),   # Bitcoin - Highest liquidity
    ('XETHZEUR', 'ETH/EUR'),   # Ethereum - High liquidity
    ('XXRPZEUR', 'XRP/EUR'),   # Ripple - Medium-high liquidity
    ('ADAEUR', 'ADA/EUR'),     # Cardano - Medium liquidity
    ('SOLEUR', 'SOL/EUR'),     # Solana - Medium liquidity
    ('ATOMEUR', 'ATOM/EUR'),   # Cosmos - Medium liquidity
    ('DOTEUR', 'DOT/EUR'),     # Polkadot - Medium liquidity
    ('MATICEUR', 'MATIC/EUR'), # Polygon - Medium liquidity
    ('LINKEUR', 'LINK/EUR'),   # Chainlink - Medium liquidity
    ('UNIEUR', 'UNI/EUR'),     # Uniswap - Lower liquidity
    ('AAVEEUR', 'AAVE/EUR'),   # Aave - Lower liquidity
    ('AVAXEUR', 'AVAX/EUR')    # Avalanche - Lower liquidity
]

# Model configuration
# These parameters control the machine learning model's behavior
# Adjust carefully as they affect prediction accuracy and training time
MODEL_CONFIG = {
    'validation_split': 0.2,    # 20% of data used for validation
    'n_trials': 100,           # Increased number of trials for better optimization
    'random_state': 42,        # Seed for reproducibility
    'early_stopping': 50,      # Stops training if no improvement
    'interval_minutes': 15,    # Data granularity
    'prediction_window': 4,    # Predict price movement over next 1 hour
    'fast_training': True,    # Disable fast training for better accuracy
    
    # Default parameters for fast training (used when fast_training is True)
    'fast_params': {
        'n_estimators': 500,
        'learning_rate': 0.01,
        'max_depth': 5,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1
    },
    
    # Technical indicator parameters
    'periods': {
        'rsi': 14,
        'price_channel': 20,
        'sma': [5, 10, 20, 50],   # Added longer SMA
        'ema': [5, 10, 20, 50],   # Added longer EMA
        'volume_ma': [5, 20],     # Added longer volume MA
        'volatility_ma': [5, 20], # Added longer volatility MA
        'price_change': [1, 2, 5, 10] # Added longer price change period
    },
    
    # Model hyperparameter search space
    'param_space': {
        'n_estimators': (300, 1000),   # Increased number of trees
        'learning_rate': (0.001, 0.1),  # Wider learning rate range
        'max_depth': (3, 8),           # Increased max depth range
        'min_child_samples': (30, 100), # Increased min samples
        'subsample': (0.6, 0.9),       # Wider subsample range
        'colsample_bytree': (0.6, 0.9), # Wider column sample range
        'reg_alpha': (0.01, 1.0),      # Increased regularization range
        'reg_lambda': (0.01, 1.0)      # Increased regularization range
    }
}

# Trading configuration
# These parameters control the trading behavior and risk management
# Adjust based on your risk tolerance and capital
TRADING_CONFIG = {
    # Position parameters
    'position': {
        'size': 100.0,         # Base position size in EUR
        'take_profit': 1.0,    # Lower take profit for more frequent wins
        'min_confidence': 0.6  # Lower confidence threshold for more trades
    },
    
    # Technical parameters
    'technical': {
        'sma_fast': 5,        # Faster moving average for quicker signals
        'sma_slow': 10,       # Slower moving average for trend confirmation
    },
    
    # Risk management parameters
    'risk_management': {
        'total_balance': 1000.0,   # Total trading balance in EUR
        'max_positions': 3,        # Allow more simultaneous positions
        'max_position_size': 100.0, # Maximum single position size in EUR
        'trailing_stop_distance': 0.5,  # Tighter trailing stop
        'initial_stop_loss': 0.8,      # Tighter initial stop loss
        'max_hold_bars': 16,           # About 4 hours at 15-minute intervals
        'daily_loss_limit': -2.0,      # Conservative daily loss limit
        'fee_rate': 0.0026            # Kraken maker fee rate
    }
}

# Recommended adjustments based on market conditions:
# 
# Bull Market:
# - Decrease min_confidence (easier entry)
# - Increase take_profit
# - Increase max_positions
# - Consider larger position sizes
#
# Bear Market:
# - Increase min_confidence (stricter entry)
# - Decrease take_profit
# - Decrease max_positions
# - Consider smaller position sizes
#
# High Volatility:
# - Increase min_confidence
# - Decrease position sizes
# - Decrease max_positions
# - Consider shorter prediction_window
#
# Low Volatility:
# - Decrease min_confidence
# - Consider longer prediction_window
# - Can increase position sizes
#
# Risk Management Tips:
# 1. Start with small position sizes
# 2. Monitor drawdown closely
# 3. Consider reducing max_positions for new strategies
# 4. Account for fees and slippage in profit targets
# 5. Test changes in simulation before live trading 