import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import mean_squared_error, log_loss, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import optuna
import joblib
import json
from datetime import datetime
from trading_strat_params import MODEL_CONFIG, TRADING_CONFIG, TRADING_PAIRS

# Set Optuna's logging level to WARNING
optuna.logging.set_verbosity(optuna.logging.WARNING)

def create_features_for_pair(df, pair_name, prediction_horizon=8):
    """
    Create technical indicators as features for a single pair.
    prediction_horizon: number of 15-min periods to look ahead (4=1h, 8=2h, 16=4h)
    """
    df = df.copy()
    
    # Convert timestamp to datetime if not already
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Basic price changes
    for period in [1, 2, 3, 4, 8]:  # Looking at recent price changes
        df[f'price_change_{period}'] = df['close'].pct_change(period, fill_method=None) * 100
    
    # Volume features
    df['volume_change'] = df['volume'].pct_change(fill_method=None)
    df['volume_ma5'] = df['volume'].rolling(window=5).mean()
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma5']
    df['volume_ratio_20'] = df['volume'] / df['volume_ma20']
    
    # Moving averages
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        df[f'sma_dist_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}'] * 100
        
        # Exponential moving averages
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'ema_dist_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}'] * 100
    
    # Volatility
    df['high_low_range'] = (df['high'] - df['low']) / df['close'] * 100
    df['volatility'] = df['close'].pct_change(fill_method=None).rolling(window=5).std() * 100
    df['volatility_20'] = df['close'].pct_change(fill_method=None).rolling(window=20).std() * 100
    
    # VWAP and related
    df['vwap_dist'] = (df['close'] - df['vwap']) / df['vwap'] * 100
    df['vwap_change'] = df['vwap'].pct_change(fill_method=None) * 100
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Create target based on prediction horizon
    if prediction_horizon == 'flex_max':
        # Look at next 16 periods (4 hours) and find maximum return
        future_returns = pd.DataFrame()
        for i in range(1, 17):  # Up to 4 hours ahead
            future_returns[f'return_{i}'] = df['close'].shift(-i).pct_change(i, fill_method=None) * 100
        df['target'] = future_returns.max(axis=1)
    else:
        # Fixed horizon prediction - calculate percentage return
        df['target'] = df['close'].shift(-prediction_horizon).pct_change(prediction_horizon, fill_method=None) * 100
    
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

def optimize_trading_strategy(df, model, initial_balance=1000.0, n_trials=50):
    """
    Optimize trading strategy parameters using Optuna.
    Returns the best parameters found.
    """
    def objective(trial):
        # Define parameter search space with wider bounds
        params = {
            'prediction_horizon': trial.suggest_categorical('prediction_horizon', [4, 8, 16, 'flex_max']),  # 1h, 2h, 4h, or flex max
            'buy_threshold': trial.suggest_float('buy_threshold', 0.2, 3.0),
            'take_profit_threshold': trial.suggest_float('take_profit_threshold', 0.5, 5.0),
            'max_hold_hours': trial.suggest_int('max_hold_hours', 1, 24),
            'trailing_stop_distance': trial.suggest_float('trailing_stop_distance', 0.2, 2.0),
            'min_rsi': trial.suggest_int('min_rsi', 20, 40),
            'max_rsi': trial.suggest_int('max_rsi', 60, 80),
            'min_volume_ratio': trial.suggest_float('min_volume_ratio', 0.5, 2.0),
            'max_volatility': trial.suggest_float('max_volatility', 1.0, 5.0),
            'profit_lock_pct': trial.suggest_float('profit_lock_pct', 0.3, 1.0)
        }
        
        # Prepare data with the selected prediction horizon
        data = df.copy()
        features_df = create_features_for_pair(data, '', params['prediction_horizon'])
        
        # Get features and target
        features = features_df.drop(columns=['timestamp', 'target'])
        if 'predicted_price' in features.columns:
            features = features.drop(columns=['predicted_price'])
            
        # Train a new model with the selected prediction horizon
        X = features
        y = features_df['target']  # Target is already set correctly in create_features_for_pair
        
        # Train model
        model = train_model(X, X, y, y, quick_mode=True)  # Use quick mode for optimization
        
        # Get predictions
        data = data[:len(features)]  # Align lengths
        data['predicted_price'] = model.predict(features)
        
        # Run simulation with these parameters
        results = simulate_trading(
            df=data,
            predicted_price="predicted_price",
            buy_threshold=params['buy_threshold'],
            take_profit_threshold=params['take_profit_threshold'],
            max_hold_hours=params['max_hold_hours'],
            trailing_stop_distance=params['trailing_stop_distance'],
            min_rsi=params['min_rsi'],
            max_rsi=params['max_rsi'],
            min_volume_ratio=params['min_volume_ratio'],
            max_volatility=params['max_volatility'],
            profit_lock_pct=params['profit_lock_pct'],
            fee_rate=TRADING_CONFIG['risk_management']['fee_rate'],
            slippage_rate=0.0005,
            initial_balance=initial_balance
        )
        
        if not results['trades']:
            return -100.0  # Penalize no trades
        
        returns = pd.Series([t['return_percent'] for t in results['trades'] if t['type'] == 'sell'])
        if len(returns) < 5:  # Require at least 5 trades
            return -50.0 - (5 - len(returns)) * 10  # Penalize based on how far from minimum trades
            
        # Calculate risk-adjusted return metrics
        mean_return = returns.mean()
        max_drawdown = abs(returns.min())
        win_rate = (returns > 0).mean()
        
        # Penalize strategies with poor risk management
        if max_drawdown > 3.0:  # Max drawdown > 3%
            return -10.0 - max_drawdown  # Penalize based on drawdown size
        if win_rate < 0.6:  # Win rate < 60%
            return -20.0 - (0.6 - win_rate) * 100  # Penalize based on how far from target win rate
            
        # Calculate Sharpe-like ratio with minimum volatility floor
        volatility = max(returns.std(), 0.001)  # Minimum volatility to avoid division by zero
        sharpe = mean_return / volatility
        
        # Combine multiple metrics for final score
        score = (
            sharpe * 2.0 +                # Reward risk-adjusted returns
            win_rate * 3.0 +             # Reward high win rate
            mean_return * 2.0 -          # Reward high returns
            max_drawdown * 1.0 +         # Penalize drawdowns
            len(returns) * 0.1           # Small reward for more trades
        )
        
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params
    best_params['optimization_score'] = study.best_value
    
    return best_params

def simulate_trading(
    df, 
    predicted_price="predicted_price",
    buy_threshold=3.0,
    take_profit_threshold=2.0,
    max_hold_hours=72,
    fee_rate=0.0009,
    slippage_rate=0.0005, 
    initial_balance=100.0,
    trailing_stop_distance=1.5,
    min_rsi=30,
    max_rsi=70,
    min_volume_ratio=0.8,
    max_volatility=3.0,
    profit_lock_pct=0.5
):
    """
    Run a trading simulation with improved risk management.
    """
    results = {
        'trades': [],
        'metrics': {},
        'positions': []
    }
    
    data = df.copy()
    
    # Ensure index is datetime
    if not isinstance(data.index, pd.DatetimeIndex):
        if 'timestamp' in data.columns:
            data.index = pd.to_datetime(data['timestamp'])
            data = data.drop(columns=['timestamp'])
        else:
            data.index = pd.to_datetime(data.index)
    
    # Calculate the % difference between predicted future price and current close
    data["price_increase"] = (data[predicted_price] - data["close"]) / data["close"] * 100
    data["signal"] = data["price_increase"] > buy_threshold

    balance = initial_balance
    position = 0.0
    entry_index = None
    entry_price = 0.0
    trades = []
    trailing_stop = None
    max_position_return = float('-inf')
    
    # Risk management: maximum position size as % of balance
    max_position_pct = 0.95  # Never use more than 95% of balance for a trade
    min_trade_amount = 10.0  # Minimum trade size in EUR
    
    for i in range(len(data)):
        current_price = data["close"].iloc[i]
        current_time = data.index[i]
        
        # Check if we are not in a position
        if position == 0 and data["signal"].iloc[i]:
            # Entry conditions
            rsi = data['rsi'].iloc[i]
            volume_ratio = data['volume_ratio'].iloc[i]
            volatility = data['volatility'].iloc[i]
            
            # Skip if conditions not met
            if not (min_rsi <= rsi <= max_rsi and 
                   volume_ratio >= min_volume_ratio and 
                   volatility <= max_volatility):
                continue
                
            # Trend confirmation
            if not all(data[f'ema_dist_{period}'].iloc[i] > 0 for period in [5, 10]):
                continue
                
            # Check recent price action
            if i > 0:
                price_change = (current_price - data["close"].iloc[i-1]) / data["close"].iloc[i-1] * 100
                if price_change < -0.5:
                    continue
            
            # Calculate position size (account for fees and slippage)
            total_cost_factor = (1 + fee_rate) * (1 + slippage_rate)
            max_position_size = (balance * max_position_pct) / (current_price * total_cost_factor)
            
            # Skip if minimum trade amount not met
            if max_position_size * current_price < min_trade_amount:
                continue
            
            # Slippage on buy
            buy_price = current_price * (1 + slippage_rate)
            position = max_position_size
            entry_cost = position * buy_price
            fees_paid = entry_cost * fee_rate
            balance = balance - entry_cost - fees_paid
            
            entry_price = buy_price
            entry_index = i
            trailing_stop = -trailing_stop_distance
            max_position_return = float('-inf')
            trades.append({
                'type': 'buy',
                'timestamp': current_time,
                'price': buy_price,
                'size': position,
                'cost': entry_cost,
                'fees_paid': fees_paid,
                'balance_after': balance,
                'predicted_price': data[predicted_price].iloc[i]
            })

        # If we're in a position, check exit conditions
        elif position > 0:
            current_return = (current_price - entry_price) / entry_price * 100
            max_position_return = max(max_position_return, current_return)
            
            # Update trailing stop if price moved higher
            if current_return > trailing_stop + trailing_stop_distance:
                trailing_stop = current_return - trailing_stop_distance
            
            # Lock in profits if we've reached a good gain
            if current_return >= profit_lock_pct and trailing_stop < 0:
                trailing_stop = profit_lock_pct * 0.5  # Lock in 50% of the profit
            
            # Exit conditions
            exit_conditions = [
                current_return >= take_profit_threshold,  # Take profit
                current_return <= trailing_stop,          # Trailing stop hit
                (i - entry_index) >= max_hold_hours,     # Max hold time reached
                data["price_increase"].iloc[i] <= -1.0,  # Strong negative prediction
                current_return <= -2.0,                  # Hard stop loss
                max_position_return - current_return >= 2.0  # Maximum drawdown from peak
            ]
            
            if any(exit_conditions):
                # Slippage on sell
                sell_price = current_price * (1 - slippage_rate)
                exit_value = position * sell_price
                fees_paid = exit_value * fee_rate
                balance = balance + exit_value - fees_paid
                
                # Calculate actual return including fees
                entry_trade = trades[-1]  # Get the corresponding entry trade
                total_fees = entry_trade['fees_paid'] + fees_paid
                net_pnl = exit_value - entry_trade['cost'] - total_fees
                actual_return = (net_pnl / entry_trade['cost']) * 100
                
                trades.append({
                    'type': 'sell',
                    'timestamp': current_time,
                    'price': sell_price,
                    'size': position,
                    'value': exit_value,
                    'fees_paid': fees_paid,
                    'total_fees': total_fees,
                    'net_pnl': net_pnl,
                    'balance_after': balance,
                    'hours_held': i - entry_index,
                    'return_percent': actual_return  # Include fees in return calculation
                })
                position = 0
    
    # Final liquidation if still in position
    if position > 0:
        sell_price = data["close"].iloc[-1] * (1 - slippage_rate)
        exit_value = position * sell_price
        fees_paid = exit_value * fee_rate
        balance = balance + exit_value - fees_paid
        
        # Calculate actual return including fees
        entry_trade = trades[-1]
        total_fees = entry_trade['fees_paid'] + fees_paid
        net_pnl = exit_value - entry_trade['cost'] - total_fees
        actual_return = (net_pnl / entry_trade['cost']) * 100
        
        trades.append({
            'type': 'sell',
            'timestamp': data.index[-1],
            'price': sell_price,
            'size': position,
            'value': exit_value,
            'fees_paid': fees_paid,
            'total_fees': total_fees,
            'net_pnl': net_pnl,
            'balance_after': balance,
            'hours_held': len(data) - entry_index,
            'return_percent': actual_return
        })
    
    # Calculate metrics
    sell_trades = [t for t in trades if t['type'] == 'sell']
    results['trades'] = trades
    results['metrics'] = {
        'final_balance': balance,
        'total_return_pct': (balance - initial_balance) / initial_balance * 100,
        'n_trades': len(sell_trades),
        'win_rate': len([t for t in sell_trades if t['return_percent'] > 0]) / len(sell_trades) if sell_trades else 0,
        'avg_return_per_trade': np.mean([t['return_percent'] for t in sell_trades]) if sell_trades else 0,
        'total_fees_paid': sum(t['fees_paid'] for t in trades),
        'avg_hold_time': np.mean([t['hours_held'] for t in sell_trades]) if sell_trades else 0
    }
    
    return results