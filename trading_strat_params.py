"""
Trading Strategy Parameters

This file contains all configuration parameters for the trading bot.
Modify these parameters carefully as they directly impact trading behavior and risk management.
"""

# Trading pairs configuration
# Focus on high liquidity pairs for better execution
TRADING_PAIRS = [
    ('XXBTZEUR', 'BTC/EUR'),   # Bitcoin
    ('XETHZEUR', 'ETH/EUR'),   # Ethereum
    ('XXRPZEUR', 'XRP/EUR'),   # Ripple
    ('SOLEUR', 'SOL/EUR'),     # Solana
    ('MATICEUR', 'MATIC/EUR'), # Polygon
    ('TURBOEUR', 'TURBO/EUR'),  # Turbos
    ('KASEUR', 'KAS/EUR'),     # Kaspa
    ('DOTEUR', 'DOT/EUR'),     # Polkadot
    ('AVAXEUR', 'AVAX/EUR'),   # Avalanche
    ('SHIBEUR', 'SHIB/EUR'),   # Shiba Inu
    ('PEPEEUR', 'PEPE/EUR'),   # Pepe
    ('GALAEUR', 'GALA/EUR'),   # Galxe
    ('KEYEUR', 'KEY/EUR'),   # SelfKey
    ('SUIEUR', 'SUI/EUR'),   # Sui
    ('LINKEUR', 'LINK/EUR'),   # Chainlink
    ('ICPEUR', 'ICP/EUR'),   # Internet Computer Protocol
    ('RENDEREUR', 'RENDER/EUR'),   # Reneder
    ('TAOEUR', 'TAO/EUR'),   # Bittensor
    ('ARBEUR', 'ARB/EUR'),   # Arbitrum
    ('ENAEUR', 'ENA/EUR'),   # Ethena
    ('DAIEUR', 'DAI/EUR'),   # Dai
    ('ALGOEUR', 'ALGO/EUR'),   # Algorand
    ('TIAEUR', 'TIA/EUR'),   # Celestia
    ('POLSEUR', 'POLS/EUR'),   # Polkastarter
    ('BNCEUR', 'BNC/EUR'),   # Bifrost
    ('SAFEEUR', 'SAFE/EUR'),   # Safe
    ('L3EUR', 'L3/EUR'),   # L3
    ('MNGOEUR', 'MNGO/EUR'),   # Mango
    ('SCEUR', 'SC/EUR'),   # Siacoin
]

# Model configuration
MODEL_CONFIG = {
    'validation_split': 0.2,
    'n_trials': 10,            # Reduced trials for faster optimization
    'random_state': 42,
    'interval_minutes': 15,
    'prediction_window': 4,    # 1-hour prediction
    'fast_training': True,
    
    # Simplified fast training parameters
    'fast_params': {
        'n_estimators': 100,   # Reduced trees
        'learning_rate': 0.1,  # Faster learning
        'max_depth': 3,        # Simpler trees
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    },
    
    # Simplified technical indicators
    'periods': {
        'rsi': 14,
        'sma': [5, 20],
        'ema': [5, 20],
        'volume_ma': [5],
        'price_change': [1, 5]
    }
}

# Trading configuration
TRADING_CONFIG = {
    # Position parameters
    'position': {
        'size': 100.0,
        'take_profit': 1.0,
        'min_confidence': 0.6
    },
    
    # Risk management parameters
    'risk_management': {
        'total_balance': 100.0,
        'max_positions': 2,        # Reduced concurrent positions
        'max_position_size': 100.0,
        'trailing_stop': 0.5,
        'max_hold_time': 4,        # 1 hour (4 x 15min intervals)
        'fee_rate': 0.0026
    },
    
    # Trading behavior
    'behavior': {
        'confirm_order': False,    # When False, execute trades automatically
        'update_interval': MODEL_CONFIG['interval_minutes'] * 60,  # Convert minutes to seconds
        'max_retries': 3,          # Number of retries for failed API calls
        'retry_delay': 5,          # Seconds to wait between retries
        'only_safe_pairs': True    # Only trade pairs with positive backtest returns
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
# 5. Test changes in simulation before live trading 