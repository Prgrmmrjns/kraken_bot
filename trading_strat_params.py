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
    'n_trials': 25,            # Reduced number of trials for faster optimization
    'random_state': 42,        # Seed for reproducibility
    'early_stopping': 20,      # Reduced early stopping rounds
    'interval_minutes': 15,    # Data granularity
    'prediction_window': 4,    # Predict price movement over next 1 hour
    'fast_training': True,     # Enable fast training mode
    
    # Default parameters for fast training
    'fast_params': {
        'n_estimators': 200,       # Reduced number of trees
        'learning_rate': 0.05,     # Increased learning rate for faster convergence
        'max_depth': 4,            # Reduced tree depth
        'min_child_samples': 20,   # Reduced minimum samples
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1
    },
    
    # Technical indicator parameters (simplified)
    'periods': {
        'rsi': 14,
        'price_channel': 20,
        'sma': [5, 20],        # Reduced number of SMAs
        'ema': [5, 20],        # Reduced number of EMAs
        'volume_ma': [5],      # Single volume MA
        'volatility_ma': [5],  # Single volatility MA
        'price_change': [1, 5] # Reduced price change periods
    },
    
    # Reduced parameter search space
    'param_space': {
        'n_estimators': (100, 300),    # Reduced range
        'learning_rate': (0.01, 0.1),  # Simplified range
        'max_depth': (3, 5),           # Reduced depth range
        'min_child_samples': (20, 50), # Reduced samples range
        'subsample': (0.7, 0.9),       # Simplified range
        'colsample_bytree': (0.7, 0.9),# Simplified range
        'reg_alpha': (0.01, 0.5),      # Reduced range
        'reg_lambda': (0.01, 0.5)      # Reduced range
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
        'total_balance': 100.0,   # Total trading balance in EUR
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