"""
Model Training Module

This module handles all model training and feature engineering functionality.
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import optuna
import joblib
import os
import ta
from params import MODEL_CONFIG
from kraken_api import parallel_download_ohlc
import lightgbm

def create_features(df):
    """Create technical indicators and features for model training."""
    if df is None or df.empty or len(df) < 50:  # Ensure enough data for features
        return None
        
    try:
        # Create a copy and ensure index is datetime
        data = df.copy()
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Convert to numpy arrays for faster computation
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values
        
        # Preallocate numpy arrays for faster computation
        n = len(data)
        sma_fast = np.zeros(n)
        sma_slow = np.zeros(n)
        ema_fast = np.zeros(n)
        ema_slow = np.zeros(n)
        
        # Volume indicators (vectorized)
        data['volume_fi'] = ta.volume.force_index(data['close'], data['volume'])
        
        # Calculate volume ratio with proper padding and array sizes
        window_size = 20
        volume_ma = np.zeros(n)
        volume_ratio = np.ones(n)  # Initialize with ones
        
        # Calculate moving average first
        for i in range(window_size-1, n):
            window_vol = volume[i-window_size+1:i+1]
            if np.any(window_vol > 0):  # Only calculate if we have some volume
                volume_ma[i] = np.mean(window_vol)
        
        # Calculate ratio where moving average is significant
        significant_volume = volume_ma > 1e-8
        volume_ratio[significant_volume] = volume[significant_volume] / volume_ma[significant_volume]
        volume_ratio[~significant_volume] = 1.0  # Set to neutral value where volume is insignificant
        
        # Clip extreme values
        volume_ratio = np.clip(volume_ratio, 0, 10)  # Limit to reasonable range
        data['volume_ratio'] = volume_ratio
        
        # Trend indicators (vectorized)
        fast_window = MODEL_CONFIG['periods']['sma'][0]
        slow_window = MODEL_CONFIG['periods']['sma'][1]
        
        # Calculate SMAs using numpy's cumsum for O(n) computation
        cumsum = np.cumsum(np.insert(close, 0, 0))
        sma_fast[fast_window-1:] = (cumsum[fast_window:] - cumsum[:-fast_window]) / fast_window
        sma_slow[slow_window-1:] = (cumsum[slow_window:] - cumsum[:-slow_window]) / slow_window
        
        data['trend_sma_fast'] = sma_fast
        data['trend_sma_slow'] = sma_slow
        
        # EMA calculation using numpy (vectorized)
        ema_alpha_fast = 2.0 / (MODEL_CONFIG['periods']['ema'][0] + 1)
        ema_alpha_slow = 2.0 / (MODEL_CONFIG['periods']['ema'][1] + 1)
        
        # Initialize EMAs with SMA values for better stability
        ema_fast[:fast_window] = sma_fast[fast_window-1]
        ema_slow[:slow_window] = sma_slow[slow_window-1]
        
        # Calculate EMAs
        for i in range(fast_window, n):
            ema_fast[i] = close[i] * ema_alpha_fast + ema_fast[i-1] * (1 - ema_alpha_fast)
        for i in range(slow_window, n):
            ema_slow[i] = close[i] * ema_alpha_slow + ema_slow[i-1] * (1 - ema_alpha_slow)
        
        data['trend_ema_fast'] = ema_fast
        data['trend_ema_slow'] = ema_slow
        
        # Momentum indicators (vectorized)
        data['momentum_rsi'] = ta.momentum.rsi(data['close'], window=MODEL_CONFIG['periods']['rsi'])
        
        # Volatility indicators (vectorized)
        data['volatility_atr'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'])
        
        # Price-based features (vectorized)
        returns = np.diff(close)
        returns = np.append([0], returns) / close  # Add 0 for first element and divide by price
        data['price_change'] = returns
        data['price_range'] = (high - low) / close
        
        # Rolling returns using numpy (faster than pandas rolling)
        window = 12
        roll_returns = np.zeros(n)
        for i in range(n - window + 1):
            roll_returns[i+window-1] = np.prod(1 + returns[i:i+window]) - 1
        data['price_cumret'] = roll_returns
        
        # Target calculation (vectorized)
        target_periods = MODEL_CONFIG['prediction_window']
        max_prices = np.zeros(n)
        for i in range(n - target_periods + 1):
            future_prices = close[i:i+target_periods]
            current_price = future_prices[0]
            max_price = np.max(future_prices)
            max_prices[i] = ((max_price - current_price) / current_price) * 100  # Direct percentage change
            
        # Fill remaining positions with 0 as we can't calculate future returns
        max_prices[-(target_periods-1):] = 0
        data['target'] = max_prices
        
        # Drop rows with NaN values and first few rows that lack proper indicators
        warmup_period = max(slow_window, target_periods, window, window_size)
        data = data.iloc[warmup_period:].copy()
        data = data.dropna()
        
        return data
        
    except Exception as e:
        print(f"Error creating features: {str(e)}")
        return None

def train_model(X_train, X_val, y_train, y_val, quick_mode=False):
    """Train a LightGBM regression model with hyperparameter optimization."""
    
    # Store feature names
    feature_names = X_train.columns.tolist()
    
    def calculate_metrics(y_true, y_pred):
        """Calculate Mean Absolute Error for percentage predictions"""
        mae = np.mean(np.abs(y_true - y_pred))  # Direct difference of percentages
        r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
        return mae, r2
    
    # Use fast training if specified in MODEL_CONFIG or if quick_mode is True
    if MODEL_CONFIG.get('fast_training', False) or quick_mode:
        params = {
            'objective': 'regression',
            'random_state': MODEL_CONFIG['random_state'],
            'verbosity': -1,
            'n_estimators': 1000,
            'learning_rate': 0.01,
            'max_depth': 3,
            'num_leaves': 8,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'bagging_freq': 1
        }
        final_model = LGBMRegressor(**params)
        final_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric='mae',
            feature_name=feature_names,
            callbacks=[
                lightgbm.early_stopping(stopping_rounds=50, verbose=False),
                lightgbm.log_evaluation(period=0)
            ]
        )
        
        # Calculate and print performance metrics
        train_pred = final_model.predict(X_train)
        val_pred = final_model.predict(X_val)
        train_mae, train_r2 = calculate_metrics(y_train, train_pred)
        val_mae, val_r2 = calculate_metrics(y_val, val_pred)
        
        print("\n📊 Model Performance (Fast Mode):")
        print(f"Training - MAE: {train_mae:.3f}% | R²: {train_r2:.3f}")
        print(f"Validation - MAE: {val_mae:.3f}% | R²: {val_r2:.3f}")
        
        # Store metrics in the model
        final_model.feature_names_ = feature_names
        final_model.train_mae_ = train_mae
        final_model.val_mae_ = val_mae
        final_model.train_r2_ = train_r2
        final_model.val_r2_ = val_r2
        return final_model

    print("🔍 Running hyperparameter optimization...")
    def objective(trial):
        params = {
            'objective': 'regression',
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.01, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 4),
            'num_leaves': trial.suggest_int('num_leaves', 4, 16),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 0.8),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.8),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
            'random_state': MODEL_CONFIG['random_state'],
            'verbosity': -1,
            'bagging_freq': 1
        }
        
        model = LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric='mae',
            feature_name=feature_names,
            callbacks=[
                lightgbm.early_stopping(stopping_rounds=50, verbose=False),
                lightgbm.log_evaluation(period=0)
            ]
        )
        
        val_pred = model.predict(X_val)
        val_mae, _ = calculate_metrics(y_val, val_pred)
        return val_mae

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=MODEL_CONFIG['n_trials'], show_progress_bar=False)

    print("\n✨ Best hyperparameters found:")
    for key, value in study.best_params.items():
        print(f"- {key}: {value}")

    # Train final model with best parameters
    best_params = study.best_params
    best_params.update({
        'objective': 'regression',
        'random_state': MODEL_CONFIG['random_state'],
        'verbosity': -1,
        'bagging_freq': 1
    })
    
    final_model = LGBMRegressor(**best_params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_metric='mae',
        feature_name=feature_names,
        callbacks=[
            lightgbm.early_stopping(stopping_rounds=50, verbose=False),
            lightgbm.log_evaluation(period=0)
        ]
    )
    
    # Calculate and print performance metrics
    train_pred = final_model.predict(X_train)
    val_pred = final_model.predict(X_val)
    train_mae, train_r2 = calculate_metrics(y_train, train_pred)
    val_mae, val_r2 = calculate_metrics(y_val, val_pred)
    
    print("\n📊 Final Model Performance:")
    print(f"Training - MAE: {train_mae:.3f}% | R²: {train_r2:.3f}")
    print(f"Validation - MAE: {val_mae:.3f}% | R²: {val_r2:.3f}")
    
    # Store metrics in the model
    final_model.feature_names_ = feature_names
    final_model.train_mae_ = train_mae
    final_model.val_mae_ = val_mae
    final_model.train_r2_ = train_r2
    final_model.val_r2_ = val_r2
    return final_model

def train_and_save_models(trading_pairs):
    """Train and save models for all trading pairs."""
    print("\n🚀 Downloading historical data...")
    pair_data = parallel_download_ohlc(trading_pairs, interval=MODEL_CONFIG['interval_minutes'])
    
    if not pair_data:
        raise Exception("❌ Failed to download data for any trading pair")
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    
    print("\n🧠 Training models for each pair...")
    for pair, display_name in trading_pairs:
        try:
            if pair not in pair_data or pair_data[pair] is None:
                print(f"⚠️ Skipping {display_name} - no data available")
                continue
                
            print(f"\n🔄 Processing {display_name}...")
            
            # Create features
            df = create_features(pair_data[pair])
            if df is None or df.empty:
                print(f"⚠️ Skipping {display_name} - feature creation failed")
                continue
            
            # Prepare training data
            features = df.drop(columns=['timestamp', 'target', 'close', 'volume', 'open', 'high', 'low'])
            target = df['target']
            
            # Split data
            split_idx = int(len(df) * (1 - MODEL_CONFIG['validation_split']))
            X_train, X_val = features[:split_idx], features[split_idx:]
            y_train, y_val = target[:split_idx], target[split_idx:]
            
            # Train model
            model = train_model(X_train, X_val, y_train, y_val)
            
            # Make predictions
            predictions = model.predict(features)
            df['predicted_return'] = predictions
            
            # Save model and predictions
            model_path = f'models/model_{pair}.joblib'
            joblib.dump(model, model_path)
            print(f"✅ Model saved to {model_path}")
            
        except Exception as e:
            print(f"❌ Error processing {display_name}: {str(e)}")
            continue
    
    print("\n✨ Model training completed!") 