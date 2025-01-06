import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import optuna
import joblib
import logging

# Constants
HOURS_IN_12 = 48  # Number of 15-min periods in 12 hours
RSI_PERIOD = 14
PRICE_CHANNEL_PERIOD = 20
SMA_PERIODS = [5, 10, 20]
EMA_PERIODS = [5, 10, 20]
VOLUME_MA_PERIODS = [5, 10]
VOLATILITY_MA_PERIODS = [5, 10]
PRICE_CHANGE_PERIODS = [1, 2, 5]

# Default trading parameters
DEFAULT_BUY_THRESHOLD = 3.0
DEFAULT_TAKE_PROFIT = 2.0
DEFAULT_MAX_HOLD_HOURS = 72
DEFAULT_FEE_RATE = 0.0009
DEFAULT_INITIAL_BALANCE = 100.0

# Default model parameters
DEFAULT_TEST_SPLIT = 0.2
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_N_TRIALS = 50
DEFAULT_RANDOM_STATE = 42
DEFAULT_PARAM_SPACE = {
    'n_estimators': (50, 500),
    'learning_rate': (0.001, 0.1),
    'max_depth': (3, 10),
    'min_child_samples': (10, 50),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'reg_alpha': (0.0, 1.0),
    'reg_lambda': (0.0, 1.0)
}

# Set Optuna's logging level to WARNING to reduce verbosity
logging.getLogger("optuna").setLevel(logging.WARNING)

def create_features(df):
    """Create technical indicators as features."""
    df = df.copy()
    
    # Convert timestamp to datetime if not already
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Datetime features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    
    # Price changes
    for period in PRICE_CHANGE_PERIODS:
        df[f'price_change_{period}'] = df['close'].pct_change(period)
    
    # Moving averages
    for period in SMA_PERIODS:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
    
    # Exponential moving averages
    for period in EMA_PERIODS:
        df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
    
    # Volume features
    df['volume_change'] = df['volume'].pct_change()
    for period in VOLUME_MA_PERIODS:
        df[f'volume_ma{period}'] = df['volume'].rolling(window=period).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma5']
    
    # Price volatility
    df['volatility'] = df['high'] - df['low']
    for period in VOLATILITY_MA_PERIODS:
        df[f'volatility_ma{period}'] = df['volatility'].rolling(window=period).mean()
    
    # VWAP distance and related
    df['vwap_dist'] = (df['close'] - df['vwap']) / df['vwap']
    df['vwap_change'] = df['vwap'].pct_change()
    
    # Price channels
    df['upper_channel'] = df['high'].rolling(window=PRICE_CHANNEL_PERIOD).max()
    df['lower_channel'] = df['low'].rolling(window=PRICE_CHANNEL_PERIOD).min()
    df['channel_position'] = (df['close'] - df['lower_channel']) / (df['upper_channel'] - df['lower_channel'])
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Target: maximum price in next 12 hours
    df['target'] = df['high'].rolling(window=HOURS_IN_12, min_periods=1).max().shift(-HOURS_IN_12)
    
    return df

def train_model(
    X_train, X_val, y_train, y_val,
    n_trials=DEFAULT_N_TRIALS,
    random_state=DEFAULT_RANDOM_STATE,
    param_space=DEFAULT_PARAM_SPACE,
    X_test=None,
    y_test=None
):
    """
    Train a LightGBM model on the given dataframe using Optuna for hyperparameter optimization.
    
    Args:
        df: pandas DataFrame containing the raw data
        test_split: fraction of data to use for testing (default 0.2)
        val_split: fraction of remaining data to use for validation (default 0.2) 
        n_trials: number of Optuna optimization trials (default 50)
        random_state: random seed (default 42)
        param_space: dictionary defining the hyperparameter search space
        
    Returns:
        trained LightGBM model
    """

    # Define objective function for optimization
    def objective(params):
        model = LGBMRegressor(
            n_estimators=int(params['n_estimators']),
            learning_rate=params['learning_rate'],
            max_depth=int(params['max_depth']),
            min_child_samples=int(params['min_child_samples']),
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            random_state=random_state,
            verbosity=-1
        )
        
        if X_test is not None:
            # For production model, use combined training data
            X_train_full = pd.concat([X_train, X_val])
            y_train_full = pd.concat([y_train, y_val])
            model.fit(X_train_full, y_train_full)
            y_pred = model.predict(X_test)
            return mean_squared_error(y_test, y_pred)
        else:
            # For testing, use validation set
            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)
            return mean_squared_error(y_val, y_val_pred)

    # Run optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective({
        'n_estimators': trial.suggest_int('n_estimators', *param_space['n_estimators']),
        'learning_rate': trial.suggest_float('learning_rate', *param_space['learning_rate'], log=True),
        'max_depth': trial.suggest_int('max_depth', *param_space['max_depth']),
        'min_child_samples': trial.suggest_int('min_child_samples', *param_space['min_child_samples']),
        'subsample': trial.suggest_float('subsample', *param_space['subsample']),
        'colsample_bytree': trial.suggest_float('colsample_bytree', *param_space['colsample_bytree']),
        'reg_alpha': trial.suggest_float('reg_alpha', *param_space['reg_alpha']),
        'reg_lambda': trial.suggest_float('reg_lambda', *param_space['reg_lambda'])
    }), n_trials=n_trials, show_progress_bar=False)

    # Train final model with best parameters
    final_model = LGBMRegressor(
        **study.best_params,
        random_state=random_state,
        verbosity=-1
    )

    if X_test is not None:
        # For production model, use combined training data
        X_train_full = pd.concat([X_train, X_val])
        y_train_full = pd.concat([y_train, y_val])
        final_model.fit(X_train_full, y_train_full)
        # Save the production model
        joblib.dump(final_model, 'model.joblib')
    else:
        # For testing, use original training data
        final_model.fit(X_train, y_train)
    
    return final_model

def simulate_trading(
    df, 
    predicted_price_col="predicted_price",
    buy_threshold=DEFAULT_BUY_THRESHOLD,
    take_profit_threshold=DEFAULT_TAKE_PROFIT,
    max_hold_hours=DEFAULT_MAX_HOLD_HOURS,
    fee_rate=DEFAULT_FEE_RATE,
    initial_balance=DEFAULT_INITIAL_BALANCE
):
    """
    Run a simple trading simulation on the given df. The df should have:
      df["close"]
      df[predicted_price_col] -> your model's predicted future price
    Returns final_balance after simulating trades.
    """

    # Copy to avoid modifying original
    data = df.copy()
    
    # Calculate the % difference between predicted future price and current close
    data["price_increase"] = (data[predicted_price_col] - data["close"]) / data["close"] * 100
    data["signal"] = data["price_increase"] > buy_threshold

    balance = initial_balance
    position = 0.0
    entry_index = None
    entry_price = 0.0

    # We'll track trades to for clarity (optional)
    trades = []

    for i in range(len(data)):
        current_price = data["close"].iloc[i]

        # Check if we are *not* in a position
        if position == 0 and data["signal"].iloc[i]:
            # Buy
            position = balance * (1 - fee_rate) / current_price
            balance = 0
            entry_price = current_price
            entry_index = i
            trades.append({
                'type': 'buy',
                'timestamp': data.index[i],
                'price': current_price,
                'predicted_price': data[predicted_price_col].iloc[i]
            })

        # If we're in a position, check exit conditions
        elif position > 0:
            current_return_percent = (current_price - entry_price) / entry_price * 100
            # Condition 1: take profit threshold
            # Condition 2: hold time limit
            if (current_return_percent >= take_profit_threshold) or ((i - entry_index) >= max_hold_hours):
                # Sell
                balance = position * current_price * (1 - fee_rate)
                position = 0
                trades.append({
                    'type': 'sell',
                    'timestamp': data.index[i],
                    'price': current_price,
                    'hours_held': i - entry_index
                })

    # Final liquidation if still in position
    if position > 0:
        balance = position * data["close"].iloc[-1] * (1 - fee_rate)
        trades.append({
            'type': 'sell',
            'timestamp': data.index[-1],
            'price': data["close"].iloc[-1],
            'hours_held': len(data) - entry_index
        })

    return balance, trades


def optimize_trading_strategy(val_df, val_predictions):
    """
    Uses Optuna to find the best trading parameters on the validation set.
    val_df:       DataFrame slice for validation (must contain close prices).
    val_predictions: numpy array or list of predicted prices for next 6h, 
                     aligned with val_df rows.
    Returns: best_params, best_value (final balance).
    """

    # Add predicted_price to val_df
    df_opt = val_df.copy()
    df_opt["predicted_price"] = val_predictions

    def objective(trial):
        # Define the hyper-parameters to optimize
        buy_threshold = trial.suggest_float("buy_threshold", 1.0, 5.0)
        take_profit_threshold = trial.suggest_float("take_profit_threshold", 1.0, 10.0)
        max_hold_hours = trial.suggest_int("max_hold_hours", 12, 72)
        
        # Simulate trading
        final_balance, _ = simulate_trading(
            df_opt,
            predicted_price_col="predicted_price",
            buy_threshold=buy_threshold,
            take_profit_threshold=take_profit_threshold,
            max_hold_hours=max_hold_hours,
            initial_balance=100.0
        )
        # We want to maximize final_balance, so return it directly.
        return final_balance

    # Create a study that maximizes the objective (final balance)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    
    return study.best_params, study.best_value