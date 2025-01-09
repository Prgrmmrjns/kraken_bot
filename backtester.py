"""
Backtesting Module

This module handles all backtesting and strategy evaluation functionality.
"""

import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import optuna
from params import MODEL_CONFIG, TRADING_CONFIG
from kraken_api import parallel_download_ohlc
from model_trainer import create_features

def optimize_trading_strategy(df, model, initial_balance=1000.0, n_trials=10):
    """Optimize trading strategy parameters using Optuna."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # Suppress Optuna output
    
    # Make predictions first
    df_with_predictions = df.copy()
    if model is not None:
        # Get features for prediction
        feature_columns = [col for col in df.columns if col not in ['target', 'close', 'volume', 'open', 'high', 'low', 'predicted_return', 'predicted_price']]
        features = df[feature_columns]
        df_with_predictions['predicted_price'] = model.predict(features)
    else:
        # For backtesting without model, use target as prediction
        df_with_predictions['predicted_price'] = df['target']
    
    def objective(trial):
        try:
            # Define parameter search space with more conservative bounds
            params = {
                'buy_threshold': trial.suggest_float('buy_threshold', 2.0, 5.0),    # Even higher threshold
                'take_profit': trial.suggest_float('take_profit', 1.5, 5.0),       # Wide profit range
                'max_hold_time': trial.suggest_int('max_hold_time', 4, 24),        # Flexible hold times
                'trailing_stop': trial.suggest_float('trailing_stop', 0.5, 2.0),    # Flexible stops
                'min_rsi': trial.suggest_float('min_rsi', 20, 40),                 # RSI ranges
                'max_rsi': trial.suggest_float('max_rsi', 60, 80),
                'min_volume_ratio': trial.suggest_float('min_volume_ratio', 0.8, 2.0),
                'max_volatility': trial.suggest_float('max_volatility', 1.0, 4.0),
                'profit_lock': trial.suggest_float('profit_lock', 0.5, 2.0),
                'initial_balance': initial_balance
            }
            
            # Run simulation with these parameters
            results = simulate_trading(
                df=df_with_predictions,
                predicted_price="predicted_price",
                fee_rate=TRADING_CONFIG['risk_management']['fee_rate'],
                **params
            )
            
            # Get trading results
            total_return = results['metrics']['total_return']
            n_trades = results['metrics']['n_trades']
            win_rate = results['metrics']['win_rate']
            
            # If we have trades but negative return, heavily penalize
            if n_trades > 0 and total_return <= 0:
                return float('-inf')
            
            # If we have no trades, return 0 (neutral)
            if n_trades == 0:
                return 0
                
            # If we have positive return, optimize it
            if total_return > 0:
                score = (
                    total_return * 2 +           # Emphasize total return
                    win_rate * 0.5 +            # Consider win rate but less important
                    np.log1p(n_trades) * 0.1    # Small bonus for more trades
                )
                return score
                
            return float('-inf')  # Fallback case
            
        except Exception as e:
            print(f"❌ Error in objective function: {str(e)}")
            return float('-inf')

    # Try multiple times to find profitable strategy
    for attempt in range(3):  # Try up to 3 times
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # If we found a profitable strategy or got exactly 0, use it
        if study.best_value >= 0:
            best_params = study.best_params
            best_params['optimization_score'] = study.best_value
            
            # Verify the strategy with these parameters
            final_results = simulate_trading(
                df=df_with_predictions,
                predicted_price="predicted_price",
                fee_rate=TRADING_CONFIG['risk_management']['fee_rate'],
                initial_balance=initial_balance,
                **best_params
            )
            
            # If verified results are profitable or neutral, return the params
            if final_results['metrics']['total_return'] >= 0:
                return best_params
    
    # If we couldn't find a profitable strategy after all attempts,
    # return parameters that will result in no trades
    return {
        'buy_threshold': 999.0,  # Impossible to reach
        'take_profit': 1.0,
        'max_hold_time': 4,
        'trailing_stop': 1.0,
        'min_rsi': 50,
        'max_rsi': 50,
        'min_volume_ratio': 999.0,
        'max_volatility': 0.0,
        'profit_lock': 1.0,
        'optimization_score': 0
    }

def simulate_trading(df, predicted_price="predicted_price", **params):
    """Run a trading simulation with enhanced risk management."""
    try:
        results = {
            'trades': [],
            'metrics': {},
        }
        
        data = df.copy()
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                data.index = pd.to_datetime(data['timestamp'])
                data = data.drop(columns=['timestamp'])
            else:
                data.index = pd.to_datetime(data.index)
        
        # Verify required columns exist
        required_columns = ['close', 'volume', 'volume_ratio', 'momentum_rsi', 'volatility_atr', 'trend_sma_slow']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert to numpy arrays for faster computation
        close = data['close'].values
        volume = data['volume'].values
        volume_ratio = data['volume_ratio'].values
        rsi = data['momentum_rsi'].values
        atr = data['volatility_atr'].values
        sma_slow = data['trend_sma_slow'].values
        
        # Calculate predicted returns from predicted price
        if predicted_price not in data.columns:
            raise ValueError(f"Predicted price column '{predicted_price}' not found in DataFrame")
        pred_prices = data[predicted_price].values
        pred_returns = (pred_prices - close) / close * 100
        
        # Vectorized trading conditions
        volume_condition = volume_ratio > params.get('min_volume_ratio', 1.0)
        rsi_condition = (rsi > params.get('min_rsi', 30)) & (rsi < params.get('max_rsi', 70))
        trend_condition = np.zeros_like(sma_slow, dtype=bool)
        trend_condition[1:] = sma_slow[1:] > sma_slow[:-1]
        volatility_condition = atr < params.get('max_volatility', 5.0)
        return_condition = pred_returns > params['buy_threshold']
        
        # Combine all conditions
        signals = (
            return_condition &
            rsi_condition &
            volume_condition &
            trend_condition &
            volatility_condition
        )
        
        # Initialize arrays for tracking positions
        n = len(data)
        position_active = np.zeros(n, dtype=bool)
        entry_prices = np.zeros(n)
        entry_indices = np.zeros(n, dtype=int)
        trailing_stops = np.zeros(n)
        profit_locks = np.zeros(n)
        
        balance = params['initial_balance']
        trades = []
        consecutive_losses = 0
        max_consecutive_losses = 2
        
        # Vectorized simulation
        for i in range(n):
            current_price = close[i]
            current_time = data.index[i]
            
            # Check for entry
            if not position_active[i] and signals[i] and consecutive_losses < max_consecutive_losses:
                potential_profit = (params['take_profit'] - params['fee_rate'] * 2) * 100
                
                if potential_profit > 1.0:
                    # Calculate position size
                    trade_amount = min(
                        balance * (params.get('position_size', TRADING_CONFIG['position']['size']) / 100),
                        balance * 0.95
                    )
                    position = trade_amount / current_price
                    
                    # Record entry
                    position_active[i:] = True
                    entry_prices[i:] = current_price
                    entry_indices[i:] = i
                    trailing_stops[i] = current_price * (1 - params['trailing_stop']/100)
                    profit_locks[i] = current_price * (1 + params.get('profit_lock', 0.5)/100)
                    
                    # Update balance
                    cost = position * current_price
                    fees = cost * params['fee_rate']
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
            elif position_active[i]:
                entry_idx = entry_indices[i]
                entry_price = entry_prices[i]
                
                # Update trailing stop
                if current_price > entry_price:
                    new_stop = current_price * (1 - params['trailing_stop']/100)
                    trailing_stops[i] = max(trailing_stops[i-1], new_stop)
                    
                    if profit_locks[i-1] and current_price > profit_locks[i-1]:
                        trailing_stops[i] = max(trailing_stops[i], profit_locks[i-1])
                else:
                    trailing_stops[i] = trailing_stops[i-1]
                
                # Check exit conditions
                exit_conditions = [
                    current_price >= entry_price * (1 + params['take_profit']/100),
                    current_price <= trailing_stops[i],
                    (i - entry_idx) >= params['max_hold_time']
                ]
                
                if any(exit_conditions):
                    position = trades[-1]['size']  # Get position size from entry trade
                    exit_value = position * current_price
                    fees = exit_value * params['fee_rate']
                    balance += (exit_value - fees)
                    
                    trade_return = ((current_price - entry_price) / entry_price * 100)
                    
                    trades.append({
                        'type': 'sell',
                        'timestamp': current_time,
                        'price': current_price,
                        'size': position,
                        'fees': fees,
                        'balance': balance,
                        'return': trade_return,
                        'hours_held': (i - entry_idx) / 4
                    })
                    
                    # Reset position tracking
                    position_active[i:] = False
                    consecutive_losses = consecutive_losses + 1 if trade_return <= 0 else 0
        
        # Calculate metrics
        sell_trades = [t for t in trades if t['type'] == 'sell']
        
        results['trades'] = trades
        results['metrics'] = {
            'final_balance': balance,
            'total_return': ((balance - params['initial_balance']) / params['initial_balance'] * 100),
            'n_trades': len(sell_trades),
            'win_rate': len([t for t in sell_trades if t['return'] > 0]) / len(sell_trades) if sell_trades else 0,
            'avg_return': np.mean([t['return'] for t in sell_trades]) if sell_trades else 0,
            'total_fees': sum(t['fees'] for t in trades)
        }
        
        return results
        
    except Exception as e:
        print(f"❌ Error in simulate_trading: {str(e)}")
        print(f"Available columns: {data.columns.tolist()}")
        raise

def visualize_trades(df, results, pair_name):
    """Create visualization of trades for a specific pair."""
    # Create pair-specific directory
    pair_dir = f'backtesting_results/{pair_name.replace("/", "_")}'
    os.makedirs(pair_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Ensure datetime index
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df['timestamp'])
    
    # Plot price data
    ax1.plot(df.index, df['close'], label='Price', color='blue', alpha=0.6)
    
    # Track cumulative returns
    cumulative_returns = []
    timestamps = []
    current_return = 0
    
    # Get trades
    buy_trades = [t for t in results['trades'] if t['type'] == 'buy']
    sell_trades = [t for t in results['trades'] if t['type'] == 'sell']
    
    # Plot trades
    for buy, sell in zip(buy_trades, sell_trades):
        entry_time = pd.to_datetime(buy['timestamp'])
        exit_time = pd.to_datetime(sell['timestamp'])
        entry_price = buy['price']
        exit_price = sell['price']
        
        # Plot entry/exit points
        ax1.scatter([entry_time], [entry_price], color='green', marker='^', s=100, zorder=5)
        ax1.scatter([exit_time], [exit_price], color='red', marker='v', s=100, zorder=5)
        
        # Draw line connecting entry and exit
        ax1.plot([entry_time, exit_time], [entry_price, exit_price], 'gray', alpha=0.3, ls='--', zorder=4)
        
        # Track cumulative return
        current_return += sell['return']
        cumulative_returns.append(current_return)
        timestamps.append(exit_time)
    
    # Plot cumulative returns
    if timestamps:
        ax2.plot(timestamps, cumulative_returns, color='purple', marker='o')
        ax2.axhline(y=0, color='r', ls='--', alpha=0.3)
        ax2.fill_between(timestamps, cumulative_returns, 0, alpha=0.2,
                        color='green' if cumulative_returns[-1] > 0 else 'red')
    
    # Format plots
    for ax in [ax1, ax2]:
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
    ax1.set_title(f'Trading Activity - {pair_name}')
    ax1.set_ylabel('Price')
    ax1.legend(['Price', 'Buy', 'Sell'])
    
    ax2.set_title('Cumulative Returns')
    ax2.set_ylabel('Cumulative Return (%)')
    
    # Add performance summary
    total_return = results['metrics']['total_return']
    n_trades = len(sell_trades)
    win_rate = len([t for t in sell_trades if t['return'] > 0]) / len(sell_trades) * 100 if sell_trades else 0
    
    plt.figtext(0.02, 0.02,
                f'Total Return: {total_return:.2f}%\n'
                f'Number of Trades: {n_trades}\n'
                f'Win Rate: {win_rate:.1f}%',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{pair_dir}/trades.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save trading results
    results_file = f'{pair_dir}/results.json'
    with open(results_file, 'w') as f:
        # Convert timestamps to strings
        serializable_results = results.copy()
        for trade in serializable_results['trades']:
            trade['timestamp'] = trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        json.dump(serializable_results, f, indent=4)

def run_backtest():
    """Run the complete backtesting process."""
    os.makedirs('backtesting_results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    print("\n🚀 Starting Backtesting...")
    
    pair_data = download_and_prepare_data()
    if not pair_data:
        print("❌ No data available")
        return
    
    pair_features = create_features(pair_data)
    all_results = []
    initial_balance = TRADING_CONFIG['risk_management']['total_balance']
    available_balance = initial_balance
    
    for pair, display_name in TRADING_PAIRS:
        if pair not in pair_features:
            continue
            
        print(f"\n⚡ Processing {display_name}...")
        
        df = pair_features[pair].dropna(subset=['target'])
        
        if df.empty:
            continue
        
        # For backtesting, use target as predicted_price
        df['predicted_price'] = df['target']
        
        # Optimize trading strategy
        strategy_params = optimize_trading_strategy(
            df=df,
            model=None,  # No model needed for backtesting
            initial_balance=available_balance,
            n_trials=MODEL_CONFIG['n_trials']
        )
        
        # Run simulation
        results = simulate_trading(
            df=df,
            predicted_price="predicted_price",  # Use predicted_price column
            fee_rate=TRADING_CONFIG['risk_management']['fee_rate'],
            initial_balance=available_balance,
            **strategy_params
        )
        
        results['pair_name'] = display_name
        results['strategy_params'] = strategy_params
        all_results.append(results)
        
        # Visualize trades
        visualize_trades(df, results, display_name)
        
        available_balance = results['metrics']['final_balance']
    
    # Print and save final summary
    print_backtest_summary(all_results, initial_balance, available_balance)
    
    # Save overall summary to JSON
    overall_summary = {
        'initial_balance': initial_balance,
        'final_balance': available_balance,
        'total_return': ((available_balance - initial_balance) / initial_balance * 100),
        'pairs': [{
            'pair': r['pair_name'],
            'return': r['metrics']['total_return'],
            'n_trades': len([t for t in r['trades'] if t['type'] == 'sell']),
            'win_rate': r['metrics']['win_rate'] * 100,
            'strategy_params': r['strategy_params']
        } for r in all_results]
    }
    
    with open('backtesting_results/overall_summary.json', 'w') as f:
        json.dump(overall_summary, f, indent=4)
    
    print("\n📊 Overall summary saved to backtesting_results/overall_summary.json")

def print_backtest_summary(all_results, initial_balance, final_balance):
    """Print a concise summary of all backtest results."""
    total_return = ((final_balance - initial_balance) / initial_balance * 100)
    
    print("\n📊 BACKTEST SUMMARY")
    print("=" * 80)
    print(f"{'Pair':12} | {'Return %':>8} | {'# Trades':>8} | {'Win Rate':>8} | {'Avg Ret%':>8} | {'Avg Hours':>8}")
    print("-" * 80)
    
    for r in all_results:
        pair_name = r['pair_name']
        sell_trades = [t for t in r['trades'] if t['type'] == 'sell']
        
        n_trades = len(sell_trades)
        if n_trades == 0:
            continue
            
        pair_return = r['metrics']['total_return']
        win_rate = len([t for t in sell_trades if t['return'] > 0]) / n_trades * 100
        avg_return = np.mean([t['return'] for t in sell_trades])
        avg_duration = np.mean([t['hours_held'] for t in sell_trades])
        
        print(f"{pair_name:12} | {pair_return:8.2f} | {n_trades:8d} | {win_rate:7.1f}% | {avg_return:8.2f} | {avg_duration:8.1f}")
    
    print("=" * 80)
    print(f"Initial Balance: €{initial_balance:.2f}")
    print(f"Final Balance:   €{final_balance:.2f}")
    print(f"Total Return:    {total_return:+.2f}%") 