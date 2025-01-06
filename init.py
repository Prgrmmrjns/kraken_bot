import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from model_functions import create_features, train_model, simulate_trading, optimize_trading_strategy
from kraken_functions import download_full_ohlc_data
from sklearn.metrics import mean_squared_error
import os

# Configuration parameters
PAIR = os.getenv('TRADING_PAIR', 'XXRPZEUR')  # Get pair from environment, default to XRP/EUR
INTERVAL = 60  # minutes
TRAIN_SPLIT = 0.7  # 70% for training
TEST_SPLIT = 0.8   # 80% total for train+val, rest for test
INITIAL_BALANCE = 100.0  # EUR
MIN_CONFIDENCE = 0.02  # 2% minimum return threshold
POSITION_SIZE = 25  # EUR position size

print(f"Training model for pair: {PAIR}")

# Load and prepare data
df = download_full_ohlc_data(pair=PAIR, interval=INTERVAL)
df = create_features(df)
df = df.dropna(subset=['target'])
X = df.drop(columns=['timestamp', 'target'])
y = df['target']

# Split data
test_split_idx = int(len(df) * TEST_SPLIT)
val_split_idx = int(len(df) * TRAIN_SPLIT)

X_train = X[:val_split_idx]
X_val = X[val_split_idx:test_split_idx]
X_test = X[test_split_idx:]

y_train = y[:val_split_idx]
y_val = y[val_split_idx:test_split_idx]
y_test = y[test_split_idx:]

# Train model and make predictions
final_model = train_model(X_train, X_val, y_train, y_val)
y_train_pred = final_model.predict(X_train)
y_val_pred = final_model.predict(X_val)
y_test_pred = final_model.predict(X_test)

# Calculate MSE for each set
train_mse = mean_squared_error(y_train, y_train_pred)
val_mse = mean_squared_error(y_val, y_val_pred)
test_mse = mean_squared_error(y_test.dropna(), y_test_pred[y_test.notna()])

print(f"\nModel Performance for {PAIR}:")
print(f"Training MSE: {train_mse:.6f}")
print(f"Validation MSE: {val_mse:.6f}")
print(f"Test MSE: {test_mse:.6f}")

# Optimize trading strategy
val_df = df.iloc[val_split_idx:test_split_idx].copy()
best_params, best_balance = optimize_trading_strategy(val_df, y_val_pred)

print(f"\nOptimized Trading Parameters:")
print(f"Best Parameters: {best_params}")
print(f"Best Validation Balance: €{best_balance:.2f}")

# Apply optimized strategy to test set
test_df = df[test_split_idx:].copy()
test_df['predicted_price'] = y_test_pred
final_balance, trades = simulate_trading(
    test_df,
    predicted_price_col="predicted_price",
    **best_params,
    initial_balance=INITIAL_BALANCE
)

# Print trading results
total_return = ((final_balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
test_results = f"""
Test Set Results for {PAIR}:
Initial Balance: {INITIAL_BALANCE:.2f} EUR
Final Balance: {final_balance:.2f} EUR
Total Return: {total_return:.2f}%

Trade History:
"""

# Print trade details
for i in range(0, len(trades), 2):
    buy = trades[i]
    sell = trades[i+1] if i+1 < len(trades) else None
    
    if sell:
        profit_pct = (sell['price'] - buy['price']) / buy['price'] * 100
        
        # Convert timestamps to pandas datetime objects
        buy_time = pd.to_datetime(buy['timestamp'])
        sell_time = pd.to_datetime(sell['timestamp'])
        
        test_results += f"\nTrade {i//2 + 1}:\n"
        test_results += f"Buy:  {buy_time.strftime('%m-%d %H:%M')} @ {buy['price']:.5f} EUR\n"
        test_results += f"Sell: {sell_time.strftime('%m-%d %H:%M')} @ {sell['price']:.5f} EUR ({profit_pct:.2f}%)"

print(test_results)

# Visualize predictions
plt.figure(figsize=(15, 6))

train_dates = df['timestamp'][:val_split_idx]
val_dates = df['timestamp'][val_split_idx:test_split_idx]
test_dates = df['timestamp'][test_split_idx:]

plt.plot(df['timestamp'], df['target'], 'b-', alpha=0.5, label='Actual')
plt.plot(train_dates, y_train_pred, 'r--', label='Train Predictions')
plt.plot(val_dates, y_val_pred, 'g--', label='Validation Predictions')
plt.plot(test_dates, y_test_pred, 'y--', label='Test Predictions')

plt.title(f'Model Predictions vs Actual Values for {PAIR}')
plt.xlabel('Date')
plt.ylabel('Price (EUR)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('initial_model_predictions.png')
plt.close()

final_model = train_model(X_train, X_val, y_train, y_val, X_test=X_test, y_test=y_test)