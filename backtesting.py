import pandas as pd
import numpy as np
import json
from model_functions import (
    create_features,
    train_model,
    optimize_trading_strategy,
    simulate_trading,
)
from kraken_functions import download_full_ohlc_data
from trading_strat_params import TRADING_PAIRS, MODEL_CONFIG, TRADING_CONFIG
import matplotlib.pyplot as plt
import os
import joblib

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
    plt.savefig(f'backtesting_results/{pair_name.replace("/", "_")}_trades.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_backtest_summary(all_results, initial_balance, final_balance):
    """Print a summary of all backtest results."""
    print("\n" + "="*50)
    print("BACKTEST SUMMARY")
    print("="*50)
    
    # Aggregate metrics
    total_trades = sum(len([t for t in r['trades'] if t['type'] == 'sell']) for r in all_results)
    winning_trades = sum(len([t for t in r['trades'] if t['type'] == 'sell' and t['return'] > 0]) for r in all_results)
    total_pnl = final_balance - initial_balance
    
    # Calculate overall metrics
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    total_return = ((final_balance - initial_balance) / initial_balance * 100)
    
    # Only consider sell trades for returns and durations
    sell_trades = [t for r in all_results for t in r['trades'] if t['type'] == 'sell']
    avg_trade_return = np.mean([t['return'] for t in sell_trades]) if sell_trades else 0
    avg_trade_duration = np.mean([t['hours_held'] for t in sell_trades]) if sell_trades else 0
    
    # Find best and worst trades (only among sell trades)
    best_trade = max(sell_trades, key=lambda x: x['return']) if sell_trades else None
    worst_trade = min(sell_trades, key=lambda x: x['return']) if sell_trades else None
    
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
        print(f"Return: {best_trade['return']:.2f}%")
        print(f"Entry Price: €{best_trade['price']:.2f}")
        print(f"Duration: {best_trade['hours_held']:.1f} hours")
    
    if worst_trade:
        print(f"\nWorst Trade:")
        print(f"Return: {worst_trade['return']:.2f}%")
        print(f"Entry Price: €{worst_trade['price']:.2f}")
        print(f"Duration: {worst_trade['hours_held']:.1f} hours")
    
    print("\nPair-specific Results:")
    print("-" * 40)
    for r in all_results:
        pair_name = r['pair_name']
        n_trades = len([t for t in r['trades'] if t['type'] == 'sell'])
        pair_return = r['metrics']['total_return']
        pair_trades = [t for t in r['trades'] if t['type'] == 'sell']
        pair_win_rate = (len([t for t in pair_trades if t['return'] > 0]) / len(pair_trades) * 100) if pair_trades else 0
        print(f"{pair_name:12} | Trades: {n_trades:3d} | Return: {pair_return:6.2f}% | Win Rate: {pair_win_rate:5.2f}%")

def run_backtest():
    """Run the complete backtesting process."""
    os.makedirs('backtesting_results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    print("\n🚀 Starting Backtesting Process...")
    pair_data = download_and_prepare_data()
    if not pair_data:
        print("❌ No data available for backtesting")
        return
    
    pair_features = create_features(pair_data)
    all_results = []
    initial_balance = TRADING_CONFIG['risk_management']['total_balance']
    available_balance = initial_balance
    
    for pair, display_name in TRADING_PAIRS:
        if pair not in pair_features:
            continue
            
        print(f"\n🔄 Processing {display_name}")
        df = pair_features[pair].dropna(subset=['target'])
        
        if df.empty:
            continue
        
        # Train model
        X = df.drop(columns=['timestamp', 'target'])
        y = df['target']
        model = train_model(X, X, y, y)
        joblib.dump(model, f'models/model_{pair}.joblib')
        
        # Add predictions to dataframe
        df['predicted_price'] = model.predict(X)
        
        # Optimize trading strategy on full dataset
        strategy_params = optimize_trading_strategy(
            df,
            model,
            initial_balance=available_balance,
            n_trials=10
        )
        
        # Run simulation
        results = simulate_trading(
            df=df,
            predicted_price="predicted_price",
            buy_threshold=strategy_params['buy_threshold'],
            take_profit=strategy_params['take_profit'],
            max_hold_time=strategy_params['max_hold_time'],
            trailing_stop=strategy_params['trailing_stop'],
            fee_rate=TRADING_CONFIG['risk_management']['fee_rate'],
            initial_balance=available_balance
        )
        
        results['pair_name'] = display_name
        results['strategy_params'] = strategy_params
        all_results.append(results)
        
        # Get trade statistics
        sell_trades = [t for t in results['trades'] if t['type'] == 'sell']
        if sell_trades:
            best_trade = max(sell_trades, key=lambda x: x['return'])
            worst_trade = min(sell_trades, key=lambda x: x['return'])
            win_rate = len([t for t in sell_trades if t['return'] > 0]) / len(sell_trades) * 100
            
            print(f"\n📊 Results for {display_name}:")
            print(f"Strategy Score: {strategy_params['optimization_score']:.2f}")
            print(f"Parameters: Buy > {strategy_params['buy_threshold']:.2f}%, TP {strategy_params['take_profit']:.2f}%, " +
                  f"Stop {strategy_params['trailing_stop']:.2f}%")
            print(f"Target: Shift {strategy_params['target_shift']} periods, Window {strategy_params['target_window']} periods")
            print(f"Total Return: {results['metrics']['total_return']:.2f}%")
            print(f"Number of Trades: {len(sell_trades)}")
            print(f"Win Rate: {win_rate:.1f}%")
            print(f"Best Trade: +{best_trade['return']:.2f}% ({best_trade['hours_held']:.1f}h)")
            print(f"Worst Trade: {worst_trade['return']:.2f}% ({worst_trade['hours_held']:.1f}h)")
        else:
            print(f"\n⚠️ No trades executed for {display_name}")
        
        # Create visualization and save results
        visualize_trades(df, results, display_name)
        
        # Save detailed results to file
        param_file = f'backtesting_results/{display_name.replace("/", "_")}_trades.txt'
        with open(param_file, 'w') as f:
            json.dump({
                'strategy_params': strategy_params,
                'trades': results['trades'],
                'metrics': results['metrics']
            }, f, indent=4, default=str)
        
        available_balance = results['metrics']['final_balance']
    
    # Print final summary
    print("\n" + "="*50)
    print("BACKTEST SUMMARY")
    print("="*50)
    print(f"Initial Balance: €{initial_balance:.2f}")
    print(f"Final Balance: €{available_balance:.2f}")
    print(f"Total Return: {((available_balance - initial_balance) / initial_balance * 100):.2f}%")
    
    # Print pair comparison
    print("\nPair Performance Comparison:")
    print("-" * 60)
    print(f"{'Pair':12} | {'Return':>8} | {'Trades':>6} | {'Win Rate':>8} | {'Best':>8} | {'Worst':>8}")
    print("-" * 60)
    
    # Prepare backtest summary
    backtest_summary = {}
    
    for result in all_results:
        sell_trades = [t for t in result['trades'] if t['type'] == 'sell']
        if sell_trades:
            win_rate = len([t for t in sell_trades if t['return'] > 0]) / len(sell_trades) * 100
            best_return = max([t['return'] for t in sell_trades])
            worst_return = min([t['return'] for t in sell_trades])
            print(f"{result['pair_name']:12} | {result['metrics']['total_return']:8.2f}% | {len(sell_trades):6d} | " +
                  f"{win_rate:7.1f}% | {best_return:7.2f}% | {worst_return:7.2f}%")
            
            # Store results in summary
            pair_name = result['pair_name'].replace('/', '_')
            backtest_summary[pair_name] = {
                'total_return': result['metrics']['total_return'],
                'n_trades': len(sell_trades),
                'win_rate': win_rate,
                'best_return': best_return,
                'worst_return': worst_return
            }
        else:
            print(f"{result['pair_name']:12} | {'No trades':>8} | {0:6d} | {'---':>8} | {'---':>8} | {'---':>8}")
            
            # Store no-trade results
            pair_name = result['pair_name'].replace('/', '_')
            backtest_summary[pair_name] = {
                'total_return': 0,
                'n_trades': 0,
                'win_rate': 0,
                'best_return': 0,
                'worst_return': 0
            }
    
    # Save backtest summary
    with open('backtesting_results/backtest_summary.json', 'w') as f:
        json.dump(backtest_summary, f, indent=4)
    
    print("\nDetailed results saved in backtesting_results directory")

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

if __name__ == "__main__":
    run_backtest() 