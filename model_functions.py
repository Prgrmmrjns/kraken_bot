import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import optuna
import joblib
import json
from datetime import datetime
from trading_strat_params import MODEL_CONFIG, TRADING_CONFIG, TRADING_PAIRS
import os

# Set Optuna's logging level to WARNING
optuna.logging.set_verbosity(optuna.logging.WARNING)

def create_features_for_pair(df, pair_name, target_shift=4, target_window=4):
    """Create essential technical indicators as features."""
    df = df.copy()
    
    # Convert timestamp to datetime if not already
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Price changes (1 and 5 periods)
    df['price_change_1'] = df['close'].pct_change(1, fill_method=None) * 100
    df['price_change_5'] = df['close'].pct_change(5, fill_method=None) * 100
    
    # Volume features
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=5).mean()
    
    # Moving averages (5 and 20 periods)
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_dist_5'] = (df['close'] - df['sma_5']) / df['sma_5'] * 100
    df['sma_dist_20'] = (df['close'] - df['sma_20']) / df['sma_20'] * 100
    
    # Volatility
    df['volatility'] = df['close'].pct_change(fill_method=None).rolling(window=5).std() * 100
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Target with dynamic shift and window
    df['target'] = df['close'].shift(-target_shift).pct_change(target_window, fill_method=None) * 100
    
    # Clean up NaN values
    df = df.dropna()
    
    return df

def create_features(data_dict, prediction_horizon=8):
    """Create features for each pair separately."""
    pair_features = {}
    
    for pair, df in data_dict.items():
        pair_data = create_features_for_pair(df, pair, prediction_horizon)
        pair_features[pair] = pair_data
    
    return pair_features

def train_model(X_train, X_val, y_train, y_val, quick_mode=False):
    """Train a LightGBM regression model."""
    
    # Use fast training if specified in MODEL_CONFIG or if quick_mode is True
    if MODEL_CONFIG.get('fast_training', False) or quick_mode:
        params = {
            'objective': 'regression',
            'random_state': MODEL_CONFIG['random_state'],
            'verbosity': -1,
            **MODEL_CONFIG['fast_params']  # Use default fast parameters
        }
        final_model = LGBMRegressor(**params)
        final_model.fit(X_train, y_train)
        
        # Calculate metrics
        val_pred = final_model.predict(X_val)
        mse = mean_squared_error(y_val, val_pred)
        rmse = np.sqrt(mse)
        
        metrics = {
            'val_mse': float(mse),
            'val_rmse': float(rmse),
            'feature_importance': {
                str(col): float(imp) 
                for col, imp in zip(X_train.columns, final_model.feature_importances_)
            }
        }

        model_info = {
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'parameters': {k: float(v) if isinstance(v, np.float64) else v for k, v in params.items()},
            'metrics': metrics,
            'data_info': {
                'train_samples': int(len(X_train)),
                'val_samples': int(len(X_val)),
                'features': list(X_train.columns)
            }
        }

        with open('model_info.txt', 'w') as f:
            json.dump(model_info, f, indent=4)

        return final_model

    # If not fast training, do hyperparameter optimization
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
        model.fit(X_train, y_train)
        
        # Use mean squared error as metric
        val_pred = model.predict(X_val)
        return mean_squared_error(y_val, val_pred)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=MODEL_CONFIG['n_trials'], show_progress_bar=False)

    # Train final model with best parameters
    best_params = study.best_params
    best_params.update({
        'objective': 'regression',
        'random_state': MODEL_CONFIG['random_state'],
        'verbosity': -1
    })
    
    final_model = LGBMRegressor(**best_params)
    final_model.fit(X_train, y_train)
    
    # Save model info
    val_pred = final_model.predict(X_val)
    mse = mean_squared_error(y_val, val_pred)
    rmse = np.sqrt(mse)
    
    metrics = {
        'val_mse': float(mse),
        'val_rmse': float(rmse),
        'feature_importance': {
            str(col): float(imp) 
            for col, imp in zip(X_train.columns, final_model.feature_importances_)
        }
    }

    model_info = {
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'parameters': {k: float(v) if isinstance(v, np.float64) else v for k, v in best_params.items()},
        'metrics': metrics,
        'data_info': {
            'train_samples': int(len(X_train)),
            'val_samples': int(len(X_val)),
            'features': list(X_train.columns)
        }
    }

    with open('model_info.txt', 'w') as f:
        json.dump(model_info, f, indent=4)

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

def get_model_features(model_file):
    """Get the list of features used by the model."""
    try:
        model = joblib.load(model_file)
        if hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)
        return None
    except Exception as e:
        print(f"Error getting model features: {str(e)}")
        return None

def prepare_features(features_df, expected_features):
    """Prepare features to match model's expectations."""
    if expected_features is None:
        return features_df
    
    try:
        # Drop target column if present
        if 'target' in features_df.columns:
            features_df = features_df.drop(columns=['target'])
        
        # Check if all expected features are present
        missing_features = set(expected_features) - set(features_df.columns)
        if missing_features:
            print(f"Missing features: {missing_features}")
            return None
        
        # Select and order features to match model's expectations
        return features_df[expected_features]
        
    except Exception as e:
        print(f"Error preparing features: {str(e)}")
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
        print(f"⚠️ Using default parameters for {pair}")
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
        should_confirm = False if auto_confirm is None else not auto_confirm
        should_confirm = should_confirm and TRADING_CONFIG['behavior']['confirm_order']
        
        if execute_order(opportunity['pair'], position_size, "BUY", skip_confirm=not should_confirm):
            print(f"Buy order executed for {opportunity['display_name']}")
            return True, position_size
    return False, 0.0