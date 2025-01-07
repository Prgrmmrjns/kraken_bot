import pandas as pd
import numpy as np
from datetime import datetime
import json
from model_functions import create_features, train_model, optimize_trading_strategy, simulate_trading, create_features_for_pair
from kraken_functions import download_full_ohlc_data
from trading_strat_params import TRADING_PAIRS, MODEL_CONFIG, TRADING_CONFIG
import matplotlib.pyplot as plt
import os
import joblib
import optuna

def download_and_prepare_data():
    """Download and prepare data for all trading pairs."""
    pair_data = {}
    skipped_pairs = []
    
    for pair, display_name in TRADING_PAIRS:
        try:
            df = download_full_ohlc_data(pair=pair, interval=MODEL_CONFIG['interval_minutes'])
            if df is not None and not df.empty:
                pair_data[pair] = df
            else:
                skipped_pairs.append((pair, display_name))
        except Exception as e:
            print(f"Error processing {display_name} ({pair}): {str(e)}")
            skipped_pairs.append((pair, display_name))
    
    if skipped_pairs:
        print("\nSkipped pairs:")
        for pair, display_name in skipped_pairs:
            print(f"- {display_name} ({pair})")
    
    return pair_data

def visualize_trades(df, results, pair_name):
    """Create visualization of trades for a specific pair."""
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Ensure datetime index
    df = df.copy()
    if isinstance(df.index[0], (int, float)):
        df.index = pd.to_datetime(df.index, unit='s')
    
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
        
        # Plot entry/exit points with correct timestamps
        ax1.scatter(entry_time, entry_price, color='green', marker='^', s=100, zorder=5)
        ax1.scatter(exit_time, exit_price, color='red', marker='v', s=100, zorder=5)
        
        # Draw line connecting entry and exit
        ax1.plot([entry_time, exit_time], [entry_price, exit_price], 'gray', alpha=0.3, ls='--', zorder=4)
        
        # Annotate return percentage at exit point
        y_offset = (df['close'].max() - df['close'].min()) * 0.02  # Dynamic offset based on price range
        y_pos = exit_price + y_offset if exit_price > entry_price else exit_price - y_offset
        ax1.annotate(f"{sell['return_percent']:.1f}%", 
                    xy=(exit_time, exit_price),
                    xytext=(0, 10 if exit_price > entry_price else -20),
                    textcoords='offset points',
                    ha='center',
                    fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                    zorder=6)
        
        # Track cumulative return
        current_return += sell['return_percent']
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
    total_return = results['metrics']['total_return_pct']
    n_trades = len(sell_trades)
    win_rate = len([t for t in sell_trades if t['return_percent'] > 0]) / len(sell_trades) * 100 if sell_trades else 0
    
    plt.figtext(0.02, 0.02,
                f'Total Return: {total_return:.2f}%\n'
                f'Number of Trades: {n_trades}\n'
                f'Win Rate: {win_rate:.1f}%',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'backtesting_results/{pair_name.replace("/", "_")}_trades.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_backtest_summary(all_results, initial_balance, final_balance):
    """Print a summary of all backtest results."""
    print("\n" + "="*50)
    print("BACKTEST SUMMARY")
    print("="*50)
    
    # Aggregate metrics
    total_trades = sum(len([t for t in r['trades'] if t['type'] == 'sell']) for r in all_results)
    winning_trades = sum(len([t for t in r['trades'] if t['type'] == 'sell' and t['return_percent'] > 0]) for r in all_results)
    total_pnl = final_balance - initial_balance
    
    # Calculate overall metrics
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    total_return = ((final_balance - initial_balance) / initial_balance * 100)
    
    # Only consider sell trades for returns and durations
    sell_trades = [t for r in all_results for t in r['trades'] if t['type'] == 'sell']
    avg_trade_return = np.mean([t['return_percent'] for t in sell_trades]) if sell_trades else 0
    avg_trade_duration = np.mean([t['hours_held'] for t in sell_trades]) if sell_trades else 0
    
    # Find best and worst trades (only among sell trades)
    best_trade = max(sell_trades, key=lambda x: x['return_percent']) if sell_trades else None
    worst_trade = min(sell_trades, key=lambda x: x['return_percent']) if sell_trades else None
    
    # Print summary
    print(f"\nInitial Balance: €{initial_balance:.2f}")
    print(f"Final Balance: €{final_balance:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Total P&L: €{total_pnl:.2f}")
    print(f"\nTotal Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Trade Return: {avg_trade_return:.2f}%")
    print(f"Average Trade Duration: {avg_trade_duration:.1f} hours")
    
    if best_trade:
        print(f"\nBest Trade:")
        print(f"Return: {best_trade['return_percent']:.2f}%")
        print(f"Entry Price: €{best_trade['price']:.2f}")
        print(f"Duration: {best_trade['hours_held']:.1f} hours")
    
    if worst_trade:
        print(f"\nWorst Trade:")
        print(f"Return: {worst_trade['return_percent']:.2f}%")
        print(f"Entry Price: €{worst_trade['price']:.2f}")
        print(f"Duration: {worst_trade['hours_held']:.1f} hours")
    
    print("\nPair-specific Results:")
    print("-" * 40)
    for r in all_results:
        pair_name = r['pair_name']
        n_trades = len([t for t in r['trades'] if t['type'] == 'sell'])
        pair_return = r['metrics']['total_return_pct']
        pair_trades = [t for t in r['trades'] if t['type'] == 'sell']
        pair_win_rate = (len([t for t in pair_trades if t['return_percent'] > 0]) / len(pair_trades) * 100) if pair_trades else 0
        print(f"{pair_name:12} | Trades: {n_trades:3d} | Return: {pair_return:6.2f}% | Win Rate: {pair_win_rate:5.2f}%")

def run_backtest():
    """Run the complete backtesting process."""
    # Create results directory
    os.makedirs('backtesting_results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Download and prepare data
    print("\n🚀 Starting Backtesting Process...")
    print("📥 Downloading historical data...")
    pair_data = download_and_prepare_data()
    if not pair_data:
        print("❌ No data available for backtesting")
        return
    
    # Create features and prepare for training
    print("\n🔧 Preparing features...")
    pair_features = create_features(pair_data)
    
    # Train models and run backtests for each pair
    all_results = []
    initial_balance = TRADING_CONFIG['risk_management']['total_balance']
    available_balance = initial_balance
    
    for pair, display_name in TRADING_PAIRS:
        if pair not in pair_features:
            print(f"\n⚠️ Skipping {display_name} - no data available")
            continue
            
        print(f"\n🔄 Processing {display_name}...")
        df = pair_features[pair]
        df = df.dropna(subset=['target'])
        
        if df.empty:
            print(f"❌ No valid data for {display_name}")
            continue
        
        # Prepare features
        X = df.drop(columns=['timestamp', 'target'])
        y = df['target']
        
        # Train model on all data
        print(f"🧠 Training model for {display_name}...")
        model = train_model(X, X, y, y)  # Train on all data
        
        # Save model
        model_file = f'models/model_{pair}.joblib'
        joblib.dump(model, model_file)
        print(f"💾 Saved model to {model_file}")
        
        # Optimize trading strategy
        print(f"⚙️ Optimizing trading strategy for {display_name}...")
        strategy_params = optimize_trading_strategy(
            df,
            model,
            initial_balance=available_balance,
            n_trials=25 if MODEL_CONFIG['fast_training'] else 50
        )
        
        print(f"\n✨ Best strategy parameters for {display_name}:")
        print(f"🎯 Buy threshold: {strategy_params['buy_threshold']:.2f}%")
        print(f"💰 Take profit: {strategy_params['take_profit_threshold']:.2f}%")
        print(f"⏱️ Max hold time: {strategy_params['max_hold_hours']} hours")
        print(f"🛡️ Trailing stop: {strategy_params['trailing_stop_distance']:.2f}%")
        print(f"📉 Min RSI: {strategy_params['min_rsi']}")
        print(f"📈 Max RSI: {strategy_params['max_rsi']}")
        print(f"📊 Min Volume Ratio: {strategy_params['min_volume_ratio']:.2f}")
        print(f"📊 Max Volatility: {strategy_params['max_volatility']:.2f}%")
        print(f"🔒 Profit Lock: {strategy_params['profit_lock_pct']:.2f}%")
        print(f"📈 Strategy Score: {strategy_params['optimization_score']:.2f}")
        
        # Calculate predicted prices
        df_sim = df.copy()
        features = df_sim.drop(columns=['timestamp', 'target'])
        if 'predicted_price' in features.columns:
            features = features.drop(columns=['predicted_price'])
        df_sim['predicted_price'] = model.predict(features)
        
        # Run simulation with optimized parameters
        results = simulate_trading(
            df=df_sim,
            predicted_price="predicted_price",
            buy_threshold=strategy_params['buy_threshold'],
            take_profit_threshold=strategy_params['take_profit_threshold'],
            max_hold_hours=strategy_params['max_hold_hours'],
            trailing_stop_distance=strategy_params['trailing_stop_distance'],
            min_rsi=strategy_params['min_rsi'],
            max_rsi=strategy_params['max_rsi'],
            min_volume_ratio=strategy_params['min_volume_ratio'],
            max_volatility=strategy_params['max_volatility'],
            profit_lock_pct=strategy_params['profit_lock_pct'],
            fee_rate=TRADING_CONFIG['risk_management']['fee_rate'],
            initial_balance=available_balance
        )
        
        results['pair_name'] = display_name
        results['strategy_params'] = strategy_params
        all_results.append(results)
        
        # Create visualization
        visualize_trades(df_sim, results, display_name)
        
        # Save parameters and trade details to file
        param_file = f'backtesting_results/{display_name.replace("/", "_")}_trades.txt'
        with open(param_file, 'w') as f:
            # Write header
            f.write(f"Trade Details for {display_name}\n")
            f.write("="*50 + "\n\n")
            
            # Write strategy parameters (exact format required for parsing)
            f.write("Strategy Parameters:\n")
            for key, value in strategy_params.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.6f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            
            # Write trade details
            f.write("\nIndividual Trades:\n")
            f.write("-"*50 + "\n")
            
            buy_trades = [t for t in results['trades'] if t['type'] == 'buy']
            sell_trades = [t for t in results['trades'] if t['type'] == 'sell']
            
            for i, (buy_trade, sell_trade) in enumerate(zip(buy_trades, sell_trades), 1):
                entry_time = pd.to_datetime(buy_trade['timestamp'])
                exit_time = pd.to_datetime(sell_trade['timestamp'])
                
                f.write(f"\nTrade #{i}:\n")
                f.write(f"Entry Time: {entry_time}\n")
                f.write(f"Entry Price: €{buy_trade['price']:.4f}\n")
                f.write(f"Position Size: {buy_trade['size']:.4f}\n")
                f.write(f"Entry Cost: €{buy_trade['cost']:.4f}\n")
                f.write(f"Entry Fees: €{buy_trade['fees_paid']:.4f}\n")
                f.write(f"Exit Time: {exit_time}\n")
                f.write(f"Exit Price: €{sell_trade['price']:.4f}\n")
                f.write(f"Exit Value: €{sell_trade['value']:.4f}\n")
                f.write(f"Exit Fees: €{sell_trade['fees_paid']:.4f}\n")
                f.write(f"Total Fees: €{sell_trade['total_fees']:.4f}\n")
                f.write(f"Net P&L: €{sell_trade['net_pnl']:.4f}\n")
                f.write(f"Return: {sell_trade['return_percent']:.2f}%\n")
                f.write(f"Duration: {sell_trade['hours_held']:.1f} hours\n")
                f.write(f"Balance After: €{sell_trade['balance_after']:.4f}\n")
            
            # Write summary metrics
            f.write("\nSummary Metrics:\n")
            f.write("-"*50 + "\n")
            for metric, value in results['metrics'].items():
                if isinstance(value, float):
                    if metric.endswith('_pct'):
                        f.write(f"{metric}: {value:.2f}%\n")
                    elif metric.startswith('total_fees'):
                        f.write(f"{metric}: €{value:.4f}\n")
                    else:
                        f.write(f"{metric}: {value:.4f}\n")
                else:
                    f.write(f"{metric}: {value}\n")
        
        # Update available balance
        available_balance = results['metrics']['final_balance']
    
    # Print overall summary
    print_backtest_summary(all_results, initial_balance, available_balance)
    print("\n📊 Detailed results and visualizations saved in backtesting_results directory")

def save_trading_params(pair, params, score):
    """Save trading parameters to a file."""
    filename = f"backtesting_results/{pair.replace('/', '_')}_params.txt"
    with open(filename, 'w') as f:
        f.write(f"buy_threshold={params['buy_threshold']}\n")
        f.write(f"take_profit={params['take_profit']}\n")
        f.write(f"max_hold_time={params['max_hold_time']}\n")
        f.write(f"trailing_stop={params['trailing_stop']}\n")
        f.write(f"min_rsi={params['min_rsi']}\n")
        f.write(f"max_rsi={params['max_rsi']}\n")
        f.write(f"min_volume_ratio={params['min_volume_ratio']}\n")
        f.write(f"max_volatility={params['max_volatility']}\n")
        f.write(f"profit_lock={params['profit_lock']}\n")
        f.write(f"score={score}\n")

def optimize_trading_strategy(df, model, initial_balance=1000.0, n_trials=25):
    """
    Optimize trading strategy parameters using Optuna.
    Returns the best parameters found.
    """
    def objective(trial):
        # Define parameter search space with simplified bounds
        params = {
            'prediction_horizon': trial.suggest_categorical('prediction_horizon', [4, 8]),  # Only 1h or 2h
            'buy_threshold': trial.suggest_float('buy_threshold', 0.5, 2.0),
            'take_profit_threshold': trial.suggest_float('take_profit_threshold', 0.5, 3.0),
            'max_hold_hours': trial.suggest_int('max_hold_hours', 4, 12),
            'trailing_stop_distance': trial.suggest_float('trailing_stop_distance', 0.3, 1.5),
            'min_rsi': trial.suggest_int('min_rsi', 25, 35),
            'max_rsi': trial.suggest_int('max_rsi', 65, 75),
            'min_volume_ratio': trial.suggest_float('min_volume_ratio', 0.8, 1.5),
            'max_volatility': trial.suggest_float('max_volatility', 1.0, 3.0),
            'profit_lock_pct': trial.suggest_float('profit_lock_pct', 0.3, 0.8)
        }
        
        # Use only recent data for faster optimization
        recent_data = df.tail(1000)  # Last 1000 periods
        features_df = create_features_for_pair(recent_data, '', params['prediction_horizon'])
        
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
            take_profit_threshold=params['take_profit_threshold'],
            max_hold_hours=params['max_hold_hours'],
            trailing_stop_distance=params['trailing_stop_distance'],
            min_rsi=params['min_rsi'],
            max_rsi=params['max_rsi'],
            min_volume_ratio=params['min_volume_ratio'],
            max_volatility=params['max_volatility'],
            profit_lock_pct=params['profit_lock_pct'],
            fee_rate=TRADING_CONFIG['risk_management']['fee_rate'],
            initial_balance=initial_balance
        )
        
        if not results['trades']:
            return -100.0  # Penalize no trades
        
        returns = pd.Series([t['return_percent'] for t in results['trades'] if t['type'] == 'sell'])
        if len(returns) < 3:  # Require at least 3 trades
            return -50.0 - (3 - len(returns)) * 10
            
        # Simplified scoring
        mean_return = returns.mean()
        win_rate = (returns > 0).mean()
        
        # Basic score based on returns and win rate
        score = mean_return * win_rate * 100
        
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params
    best_params['optimization_score'] = study.best_value
    
    return best_params

if __name__ == "__main__":
    run_backtest() 