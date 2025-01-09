import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import optuna
import joblib
import json
from datetime import datetime
from params import MODEL_CONFIG, TRADING_CONFIG, TRADING_PAIRS
import os
import ta

# Set Optuna's logging level to WARNING
optuna.logging.set_verbosity(optuna.logging.WARNING)

def create_features_for_pair(df, pair_name):
    """Create features for a single pair using ta package."""
    if df is None or df.empty:
        return None
        
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    try:
        # Add technical indicators
        data = add_technical_indicators(data)
        
        # Target: Maximum price increase in next X periods
        target_periods = MODEL_CONFIG['prediction_window']
        data['target'] = data['close'].rolling(target_periods).max().shift(-target_periods)
        data['target'] = (data['target'] - data['close']) / data['close'] * 100
        
        # Drop unnecessary columns and NaN values
        data = data.dropna()
        
        return data
        
    except Exception as e:
        print(f"Error creating features for {pair_name}: {str(e)}")
        return None

def create_features(pair_data):
    """Create features for all pairs using ta package."""
    pair_features = {}
    
    for pair, df in pair_data.items():
        if df is None or df.empty:
            continue
            
        # Create a copy to avoid modifying original data
        data = df.copy()
        
        # Add technical indicators
        data = add_technical_indicators(data)
        
        # Target: Maximum price increase in next X periods
        target_periods = MODEL_CONFIG['prediction_window']
        data['target'] = data['close'].rolling(target_periods).max().shift(-target_periods)
        data['target'] = (data['target'] - data['close']) / data['close'] * 100
        
        # Drop rows with NaN values
        data = data.dropna()
        
        # Store features
        pair_features[pair] = data
    
    return pair_features

def add_technical_indicators(data):
    """Add technical indicators to dataframe."""
    # Volume indicators
    data['volume_ema'] = ta.volume.volume_weighted_average_price(data['high'], data['low'], data['close'], data['volume'])
    data['volume_fi'] = ta.volume.force_index(data['close'], data['volume'])
    data['volume_em'] = ta.volume.ease_of_movement(data['high'], data['low'], data['volume'])
    data['volume_vwap'] = ta.volume.volume_weighted_average_price(data['high'], data['low'], data['close'], data['volume'])
    
    # Trend indicators
    data['trend_sma_fast'] = ta.trend.sma_indicator(data['close'], window=10)
    data['trend_sma_slow'] = ta.trend.sma_indicator(data['close'], window=30)
    data['trend_ema_fast'] = ta.trend.ema_indicator(data['close'], window=10)
    data['trend_ema_slow'] = ta.trend.ema_indicator(data['close'], window=30)
    data['trend_adx'] = ta.trend.adx(data['high'], data['low'], data['close'])
    data['trend_macd'] = ta.trend.macd_diff(data['close'])
    data['trend_vortex_pos'] = ta.trend.vortex_indicator_pos(data['high'], data['low'], data['close'])
    data['trend_vortex_neg'] = ta.trend.vortex_indicator_neg(data['high'], data['low'], data['close'])
    
    # Momentum indicators
    data['momentum_rsi'] = ta.momentum.rsi(data['close'])
    data['momentum_stoch'] = ta.momentum.stoch(data['high'], data['low'], data['close'])
    data['momentum_stoch_signal'] = ta.momentum.stoch_signal(data['high'], data['low'], data['close'])
    data['momentum_tsi'] = ta.momentum.tsi(data['close'])
    data['momentum_uo'] = ta.momentum.ultimate_oscillator(data['high'], data['low'], data['close'])
    data['momentum_stoch_rsi'] = ta.momentum.stochrsi(data['close'])
    
    # Volatility indicators
    data['volatility_bbm'] = ta.volatility.bollinger_mavg(data['close'])
    data['volatility_bbh'] = ta.volatility.bollinger_hband(data['close'])
    data['volatility_bbl'] = ta.volatility.bollinger_lband(data['close'])
    data['volatility_bbw'] = ta.volatility.bollinger_wband(data['close'])
    data['volatility_atr'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'])
    data['volatility_ui'] = ta.volatility.ulcer_index(data['close'])
    
    # Price-based features
    data['price_change'] = data['close'].pct_change()
    data['price_range'] = (data['high'] - data['low']) / data['close']
    data['price_cumret'] = (1 + data['price_change']).rolling(window=12).apply(np.prod) - 1
    
    return data

def train_model(X_train, X_val, y_train, y_val, quick_mode=False):
    """Train a LightGBM regression model."""
    
    # Use fast training if specified in MODEL_CONFIG or if quick_mode is True
    if MODEL_CONFIG.get('fast_training', False) or quick_mode:
        params = {
            'objective': 'regression',
            'random_state': MODEL_CONFIG['random_state'],
            'verbosity': -1,
            **MODEL_CONFIG['fast_params']
        }
        final_model = LGBMRegressor(**params)
        final_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse'
        )

        return final_model

    print("Running hyperparameter optimization...")
    def objective(trial):
        params = {
            'objective': 'regression',
            'n_estimators': trial.suggest_int('n_estimators', *MODEL_CONFIG['param_space']['n_estimators']),
            'learning_rate': trial.suggest_float('learning_rate', *MODEL_CONFIG['param_space']['learning_rate'], log=True),
            'max_depth': trial.suggest_int('max_depth', *MODEL_CONFIG['param_space']['max_depth']),
            'min_child_samples': trial.suggest_int('min_child_samples', *MODEL_CONFIG['param_space']['min_child_samples']),
            'subsample': trial.suggest_float('subsample', *MODEL_CONFIG['param_space']['subsample']),
            'colsample_bytree': trial.suggest_float('colsample_bytree', *MODEL_CONFIG['param_space']['colsample_bytree']),
            'reg_alpha': trial.suggest_float('reg_alpha', *MODEL_CONFIG['param_space']['reg_alpha']),
            'reg_lambda': trial.suggest_float('reg_lambda', *MODEL_CONFIG['param_space']['reg_lambda']),
            'random_state': MODEL_CONFIG['random_state'],
            'verbosity': -1
        }
        
        model = LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse'
        )
        
        val_pred = model.predict(X_val)
        return mean_squared_error(y_val, val_pred)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=MODEL_CONFIG['n_trials'], show_progress_bar=False)

    print("\nBest hyperparameters found:")
    for key, value in study.best_params.items():
        print(f"- {key}: {value}")

    # Train final model with best parameters
    best_params = study.best_params
    best_params.update({
        'objective': 'regression',
        'random_state': MODEL_CONFIG['random_state'],
        'verbosity': -1
    })
    
    final_model = LGBMRegressor(**best_params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse'
    )
    
    # Calculate final metrics
    train_pred = final_model.predict(X_train)
    val_pred = final_model.predict(X_val)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    
    return final_model

def optimize_trading_strategy(df, model, initial_balance=1000.0, n_trials=10):
    """Optimize trading strategy parameters using Optuna."""
    def objective(trial):
        # Define parameter search space with simplified bounds
        params = {
            'buy_threshold': trial.suggest_float('buy_threshold', 0.5, 2.0),
            'take_profit': trial.suggest_float('take_profit', 0.5, 2.0),
            'max_hold_time': trial.suggest_int('max_hold_time', 4, 8),  # 1-2 hours
            'trailing_stop': trial.suggest_float('trailing_stop', 0.3, 1.0),
            'target_shift': trial.suggest_int('target_shift', 2, 8),    # 30min to 2 hours
            'target_window': trial.suggest_int('target_window', 2, 8)   # Prediction window size
        }
        
        # Use only recent data for faster optimization
        recent_data = df.tail(1000)  # Last 1000 periods
        features_df = create_features_for_pair(
            recent_data, 
            '', 
            target_shift=params['target_shift'],
            target_window=params['target_window']
        )
        
        # Get features and target
        features = features_df.drop(columns=['timestamp', 'target'])
        if 'predicted_price' in features.columns:
            features = features.drop(columns=['predicted_price'])
        
        # Train a quick model
        X = features
        y = features_df['target']
        model = train_model(X, X, y, y, quick_mode=True)
        
        # Get predictions
        recent_data = recent_data[:len(features)]  # Align lengths
        recent_data['predicted_price'] = model.predict(features)
        
        # Run simulation with these parameters
        results = simulate_trading(
            df=recent_data,
            predicted_price="predicted_price",
            buy_threshold=params['buy_threshold'],
            take_profit=params['take_profit'],
            max_hold_time=params['max_hold_time'],
            trailing_stop=params['trailing_stop'],
            fee_rate=TRADING_CONFIG['risk_management']['fee_rate'],
            initial_balance=initial_balance
        )
        
        # Calculate score based on multiple factors
        sell_trades = [t for t in results['trades'] if t['type'] == 'sell']
        if not sell_trades:
            return float('-inf')  # Penalize no trades
        
        n_trades = len(sell_trades)
        if n_trades < 3:  # Require at least 3 trades
            return float('-inf')
        
        returns = [t['return'] for t in sell_trades]
        mean_return = np.mean(returns)
        win_rate = len([r for r in returns if r > 0]) / len(returns)
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6)  # Add small epsilon to avoid division by zero
        
        # Score combines multiple factors:
        # - Mean return (weighted by 1)
        # - Win rate (weighted by 50 to scale it similarly to returns)
        # - Sharpe ratio (weighted by 2 to emphasize risk-adjusted returns)
        # - Number of trades bonus (log scale to not overemphasize)
        score = (mean_return + 
                win_rate * 50 + 
                sharpe_ratio * 2 + 
                np.log1p(n_trades))
        
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params
    best_params['optimization_score'] = study.best_value
    
    return best_params

def simulate_trading(
    df, 
    predicted_price="predicted_price",
    buy_threshold=1.5,
    take_profit=1.0,
    max_hold_time=4,  # 1 hour
    fee_rate=0.0026,
    initial_balance=100.0,
    trailing_stop=0.5
):
    """Run a simplified trading simulation."""
    results = {
        'trades': [],
        'metrics': {},
    }
    
    data = df.copy()
    
    # Ensure datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        if 'timestamp' in data.columns:
            data.index = pd.to_datetime(data['timestamp'])
            data = data.drop(columns=['timestamp'])
        else:
            data.index = pd.to_datetime(data.index)
    
    # Calculate predicted returns
    data["predicted_return"] = (data[predicted_price] - data["close"]) / data["close"] * 100
    data["signal"] = data["predicted_return"] > buy_threshold

    balance = initial_balance
    position = 0.0
    entry_index = None
    entry_price = 0.0
    trades = []
    trailing_stop_price = None
    
    for i in range(len(data)):
        current_price = data["close"].iloc[i]
        current_time = data.index[i]
        
        # Check for entry
        if position == 0 and data["signal"].iloc[i]:
            # Simple position sizing (90% of balance)
            position = (balance * 0.9) / current_price
            entry_price = current_price
            entry_index = i
            trailing_stop_price = entry_price * (1 - trailing_stop/100)
            
            # Record trade
            cost = position * current_price
            fees = cost * fee_rate
            balance -= (cost + fees)
            
            trades.append({
                'type': 'buy',
                'timestamp': current_time,
                'price': current_price,
                'size': position,
                'fees': fees,
                'balance': balance
            })

        # Check for exit
        elif position > 0:
            # Update trailing stop
            if current_price > entry_price:
                new_stop = current_price * (1 - trailing_stop/100)
                trailing_stop_price = max(trailing_stop_price, new_stop)
            
            # Exit conditions
            exit_conditions = [
                current_price >= entry_price * (1 + take_profit/100),  # Take profit
                current_price <= trailing_stop_price,                   # Trailing stop
                (i - entry_index) >= max_hold_time                     # Max hold time
            ]
            
            if any(exit_conditions):
                # Calculate returns
                exit_value = position * current_price
                fees = exit_value * fee_rate
                balance += (exit_value - fees)
                
                trades.append({
                    'type': 'sell',
                    'timestamp': current_time,
                    'price': current_price,
                    'size': position,
                    'fees': fees,
                    'balance': balance,
                    'return': ((current_price - entry_price) / entry_price * 100),
                    'hours_held': (i - entry_index) / 4
                })
                
                position = 0.0
                entry_price = 0.0
                trailing_stop_price = None
    
    # Calculate metrics
    sell_trades = [t for t in trades if t['type'] == 'sell']
    results['trades'] = trades
    results['metrics'] = {
        'final_balance': balance,
        'total_return': ((balance - initial_balance) / initial_balance * 100),
        'n_trades': len(sell_trades),
        'win_rate': len([t for t in sell_trades if t['return'] > 0]) / len(sell_trades) if sell_trades else 0,
        'avg_return': np.mean([t['return'] for t in sell_trades]) if sell_trades else 0,
        'total_fees': sum(t['fees'] for t in trades)
    }
    
    return results

def prepare_features(features_df, expected_features=None):
    """Prepare features to match model's expectations."""
    try:
        # Create a copy to avoid modifying original
        df = features_df.copy()
        
        # Drop non-feature columns if they exist
        drop_cols = ['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count', 'target']
        for col in drop_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # If no expected features provided, return all numeric features
        if expected_features is None:
            return df.select_dtypes(include=[np.number])
        
        # Check if all expected features are present
        missing_features = set(expected_features) - set(df.columns)
        if missing_features:
            print(f"Missing features: {missing_features}")
            return None
        
        # Return only the expected features in the correct order
        return df[expected_features]
        
    except Exception as e:
        print(f"Error preparing features: {str(e)}")
        return None

def get_model_features(model_file):
    """Get the list of features used by the model."""
    try:
        model = joblib.load(model_file)
        # For LightGBM models, feature names are stored in feature_name_
        if hasattr(model, 'feature_name_'):
            return model.feature_name_
        # For scikit-learn models, try feature_names_in_
        elif hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)
        return None
    except Exception as e:
        print(f"Error getting model features: {str(e)}")
        return None

def evaluate_trading_opportunities(trading_pairs, models, pair_params):
    """Evaluate trading opportunities for all pairs."""
    opportunities = []
    best_opportunity = None
    best_prediction = float('-inf')
    
    for pair, display_name in trading_pairs:
        try:
            # Get current market data
            df = download_recent_ohlc(pair=pair)
            if df is None or len(df) < 50:
                continue
            
            # Use model
            model = models.get(pair)
            if model is None:
                continue
            
            # Create and prepare features
            features_df = create_features_for_pair(df, display_name)
            if features_df is None:
                continue
            
            model_features = get_model_features(f'models/model_{pair}.joblib')
            features = prepare_features(features_df, model_features)
            if features is None:
                continue
            
            # Make prediction
            prediction = model.predict(features)[-1]
            current_price = float(df['close'].iloc[-1])
            
            # Track best opportunity
            if prediction > best_prediction:
                best_prediction = prediction
                best_opportunity = {
                    'pair': pair,
                    'display_name': display_name,
                    'current_price': current_price,
                    'predicted_return': prediction,
                    'params': pair_params[pair]
                }
            
            # Check trading conditions
            params = pair_params[pair]
            if prediction > params['buy_threshold']:
                opportunities.append({
                    'pair': pair,
                    'display_name': display_name,
                    'current_price': current_price,
                    'predicted_return': prediction,
                    'params': params
                })
        
        except Exception as e:
            print(f"Error processing {display_name}: {str(e)}")
            continue
    
    return opportunities, best_opportunity

def calculate_position_size(available_balance, risk_percentage):
    """Calculate position size based on available balance and risk percentage."""
    if available_balance < 10.0:
        return 0.0
    
    # Ensure minimum position size is €10 if balance allows
    position_size = max(10.0, available_balance * (risk_percentage / 100))
    
    # Don't risk more than the available balance
    return min(position_size, available_balance)

def get_safe_pairs():
    """Get list of pairs that had positive returns in backtesting."""
    try:
        with open('backtesting_results/backtest_summary.json', 'r') as f:
            backtest_data = json.load(f)
            safe_pairs = []
            for pair_name, data in backtest_data.items():
                if data['total_return'] > 0:
                    # Convert back to display name format (e.g., 'BTC_EUR' -> 'BTC/EUR')
                    display_name = pair_name.replace('_', '/')
                    safe_pairs.append(display_name)
            return safe_pairs
    except Exception as e:
        print(f"Error loading safe pairs: {str(e)}")
        return []

def load_trading_params(pair):
    """Load trading parameters from file."""
    try:
        filename = f"backtesting_results/{pair.replace('/', '_')}_trades.txt"
        
        with open(filename, 'r') as f:
            data = json.load(f)
            
            # Check if we should only use safe pairs
            if TRADING_CONFIG['behavior']['only_safe_pairs']:
                safe_pairs = get_safe_pairs()
                if pair not in safe_pairs:
                    print(f"⚠️ Skipping {pair} - not in safe pairs list")
                    return None
            
            return data['strategy_params']
        
    except Exception as e:
        return {
            'buy_threshold': 1.5,
            'take_profit': 2.0,
            'max_hold_time': 4,  # 1 hour
            'trailing_stop': 0.5,
            'optimization_score': 0.0
        }

def initialize_trading(trading_pairs):
    """Initialize trading parameters and models."""
    pair_params = {}
    models = {}
    
    for pair, display_name in trading_pairs:
        pair_params[pair] = load_trading_params(display_name)
        model_file = f'models/model_{pair}.joblib'
        if os.path.exists(model_file):
            models[pair] = joblib.load(model_file)
    
    return pair_params, models

def handle_trading_opportunity(opportunity, position_size, auto_confirm=None):
    """Handle a trading opportunity."""
    if opportunity and position_size >= 10.0:
        # Use the config setting if auto_confirm is not explicitly provided
        should_confirm = TRADING_CONFIG['behavior']['confirm_order'] if auto_confirm is None else auto_confirm
        
        # In test mode, we don't need user confirmation
        if TRADING_CONFIG['behavior']['test_run']:
            should_confirm = False
        
        if execute_order(opportunity['pair'], position_size, "BUY", skip_confirm=not should_confirm):
            action = "Simulated buy" if TRADING_CONFIG['behavior']['test_run'] else "Buy"
            print(f"✅ {action} order executed for {opportunity['display_name']}")
            return True, position_size
    return False, 0.0