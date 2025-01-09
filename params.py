"""
Kraken Trading Bot Configuration Parameters

This file contains all user-configurable parameters for the trading bot.
Modify these parameters carefully as they directly impact trading behavior and risk management.
"""

# Trading pairs configuration
TRADING_PAIRS = [
    ('XXBTZEUR', 'BTC/EUR'),   # Bitcoin
    ('XETHZEUR', 'ETH/EUR'),   # Ethereum
    ('XXRPZEUR', 'XRP/EUR'),   # Ripple
    ('SOLEUR', 'SOL/EUR'),     # Solana
    ('MATICEUR', 'MATIC/EUR'), # Polygon
    ('TURBOEUR', 'TURBO/EUR'), # Turbos
    ('KASEUR', 'KAS/EUR'),     # Kaspa
    ('DOTEUR', 'DOT/EUR'),     # Polkadot
    ('AVAXEUR', 'AVAX/EUR'),   # Avalanche
    ('SHIBEUR', 'SHIB/EUR'),   # Shiba Inu
    ('PEPEEUR', 'PEPE/EUR'),   # Pepe
    ('GALAEUR', 'GALA/EUR'),   # Galxe
    ('KEYEUR', 'KEY/EUR'),     # SelfKey
    ('SUIEUR', 'SUI/EUR'),     # Sui
    ('LINKEUR', 'LINK/EUR'),   # Chainlink
    ('ICPEUR', 'ICP/EUR'),     # Internet Computer Protocol
    ('RENDEREUR', 'RENDER/EUR'), # Render
    ('TAOEUR', 'TAO/EUR'),     # Bittensor
    ('ARBEUR', 'ARB/EUR'),     # Arbitrum
    ('ENAEUR', 'ENA/EUR'),     # Ethena
    ('DAIEUR', 'DAI/EUR'),     # Dai
    ('ALGOEUR', 'ALGO/EUR'),   # Algorand
    ('TIAEUR', 'TIA/EUR'),     # Celestia
    ('SAFEEUR', 'SAFE/EUR'),   # Safe
    ('L3EUR', 'L3/EUR'),       # L3
    ('MNGOEUR', 'MNGO/EUR'),   # Mango
    ('SCEUR', 'SC/EUR')        # Siacoin
]

# Model configuration
MODEL_CONFIG = {
    'validation_split': 0.1,
    'n_trials': 20,             # Number of optimization trials
    'random_state': 42,
    'interval_minutes': 15,     # Data interval in minutes
    'prediction_window': 12,    # 3-hour prediction window
    'fast_training': True,      # Use fast training mode
    'max_lookback_days': 30,    # Days of historical data to use
    
    # Parameter space for model optimization
    'param_space': {
        'n_estimators': [150, 250],
        'learning_rate': [0.01, 0.05],
        'max_depth': [4, 6],
        'min_child_samples': [30, 50],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'reg_alpha': [0.1, 0.3],
        'reg_lambda': [0.1, 0.3]
    },
    
    # Fast training parameters
    'fast_params': {
        'n_estimators': 150,    
        'learning_rate': 0.03,
        'max_depth': 5,         
        'min_child_samples': 30,
        'subsample': 0.85,
    },
    
    # Technical indicator periods
    'periods': {
        'rsi': 14,
        'sma': [20, 50],
        'ema': [10, 30],
        'volume_ma': [20, 50],
        'price_change': [5, 20]
    }
}

# Trading configuration
TRADING_CONFIG = {
    # Position parameters
    'position': {
        'size': 100.0,          # Base position size in EUR
        'take_profit': 3.0,     # Take profit percentage
        'min_confidence': 1.5   # Minimum prediction confidence
    },
    
    # Risk management
    'risk_management': {
        'total_balance': 100.0, # Starting balance in EUR
        'max_positions': 2,     # Maximum concurrent positions
        'max_position_size': 100.0,
        'trailing_stop': 1.0,   # Trailing stop percentage
        'max_hold_time': 12,    # Maximum hold time in hours
        'fee_rate': 0.009    # Trading fee rate
    },
    
    # Trading behavior
    'behavior': {
        'test_run': True,      # Test mode (no real trades)
        'confirm_order': False, # Require order confirmation
        'update_interval': 15,  # Minutes between updates
        'max_retries': 3,      # API retry attempts
        'retry_delay': 5,      # Seconds between retries
        'only_safe_pairs': True, # Only trade proven pairs
        'min_volume_usd': 1000000,  # Minimum 24h volume
        'max_spread': 0.5,     # Maximum spread percentage
        'min_trades': 5,       # Minimum trades for validation
        'duration_hours': 72   # Bot runtime in hours
    }
}

# API configuration (load from environment variables)
API_CONFIG = {
    'url': "https://api.kraken.com",
    'version': '0',
    'rate_limit': 0.2  # Seconds between API calls
} 