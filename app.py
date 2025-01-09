"""
Kraken Trading Bot - Main Application

This is the main entry point for the Kraken trading bot.
"""

import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import joblib
import warnings
import pandas as pd
import json
import numpy as np

# Suppress all warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from params import TRADING_PAIRS, TRADING_CONFIG, MODEL_CONFIG
from model_trainer import create_features, train_model, train_and_save_models
from backtester import run_backtest, optimize_trading_strategy, simulate_trading, visualize_trades
from kraken_api import (
    get_ticker_info,
    execute_order,
    get_account_balance,
    get_open_positions,
    parallel_download_ohlc
)

def prepare_features(features_df, model_features):
    """Prepare features to match model's expected features."""
    if model_features is None:
        return features_df
    
    # Check if we have all required features
    missing_features = set(model_features) - set(features_df.columns)
    if missing_features:
        return None
        
    # Return only the features the model expects, in the correct order
    return features_df[model_features]

def should_run_backtest(last_backtest_time):
    """Check if we should run backtest based on time elapsed."""
    if last_backtest_time is None:
        return True
    hours_elapsed = (datetime.now() - last_backtest_time).total_seconds() / 3600
    return hours_elapsed >= 6

def evaluate_trading_opportunities(trading_pairs, models):
    """Evaluate trading opportunities for all pairs."""
    opportunities = []
    best_opportunity = None
    best_prediction = float('-inf')
    
    print("\n🔍 Evaluating trading opportunities...")
    pair_data = parallel_download_ohlc(trading_pairs, interval=MODEL_CONFIG['interval_minutes'])
    
    for pair, display_name in trading_pairs:
        try:
            if pair not in pair_data or pair_data[pair] is None:
                continue
                
            # Create features
            df = create_features(pair_data[pair])
            if df is None or df.empty:
                continue
            
            # Get model predictions
            model = models.get(pair)
            if model is None:
                continue
                
            features = prepare_features(df, model.feature_names_)
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
                    'predicted_return': prediction
                }
            
            # Check trading conditions
            if prediction > TRADING_CONFIG['position']['min_confidence']:
                opportunities.append({
                    'pair': pair,
                    'display_name': display_name,
                    'current_price': current_price,
                    'predicted_return': prediction
                })
        
        except Exception as e:
            print(f"❌ Error evaluating {display_name}: {str(e)}")
            continue
    
    return opportunities, best_opportunity

def print_trading_status(opportunities, start_time=None):
    """Print current trading status."""
    print("\n" + "="*50)
    print("🤖 TRADING STATUS")
    print("="*50)
    
    if not opportunities:
        print("😴 No trading opportunities found")
        return
    
    print("\n📊 Trading Opportunities:")
    for opp in opportunities:
        predicted_return = opp.get('predicted_return', 0)
        signal_strength = "🟢" if predicted_return > 2 else "🟡" if predicted_return > 1 else "🔴"
        print(f"{signal_strength} {opp['display_name']}: {predicted_return:+.2f}% predicted return")
    
    if start_time:
        elapsed_hours = (datetime.now() - start_time).total_seconds() / 3600
        remaining_hours = max(0, TRADING_CONFIG['behavior']['duration_hours'] - elapsed_hours)
        print(f"\n⏱️ Remaining Time: {remaining_hours:.1f} hours")

def handle_trading_opportunity(opportunity, position_size):
    """Handle a trading opportunity."""
    try:
        if opportunity and position_size >= 10.0:
            if execute_order(opportunity['pair'], position_size, "BUY"):
                action = "Simulated buy" if TRADING_CONFIG['behavior']['test_run'] else "Buy"
                print(f"✅ {action} order executed for {opportunity['display_name']}")
                print(f"💶 Amount: €{position_size:.2f}")
                return True, position_size
        return False, 0.0
    except Exception as e:
        print(f"❌ Error executing trade: {str(e)}")
        return False, 0.0

def process_trading_pair(pair, display_name, pair_data):
    """Process a single trading pair: train model and run backtest."""
    try:
        print(f"\n🔄 Processing {display_name}...")
        
        # Create features
        df = create_features(pair_data)
        if df is None or df.empty:
            print(f"⚠️ Skipping {display_name} - feature creation failed")
            return None
        
        # Prepare training data
        feature_columns = [col for col in df.columns if col not in ['target', 'close', 'volume', 'open', 'high', 'low']]
        features = df[feature_columns]
        target = df['target']
        
        # Split data
        split_idx = int(len(df) * (1 - MODEL_CONFIG['validation_split']))
        X_train, X_val = features[:split_idx], features[split_idx:]
        y_train, y_val = target[:split_idx], target[split_idx:]
        
        # Train model
        model = train_model(X_train, X_val, y_train, y_val)
        
        # Addpredicted prices to DataFrame
        df['predicted_price'] = model.predict(features)
        
        # Save model
        os.makedirs('models', exist_ok=True)
        model_path = f'models/model_{pair}.joblib'
        joblib.dump(model, model_path)
        
        # Run backtest
        strategy_params = optimize_trading_strategy(
            df,
            model,
            initial_balance=TRADING_CONFIG['risk_management']['total_balance'],
            n_trials=MODEL_CONFIG['n_trials']
        )
        
        # Run simulation with optimized parameters
        results = simulate_trading(
            df=df,
            **strategy_params,
            fee_rate=TRADING_CONFIG['risk_management']['fee_rate'],
            initial_balance=TRADING_CONFIG['risk_management']['total_balance']
        )
        
        # Store results
        results['pair_name'] = display_name
        results['strategy_params'] = strategy_params
        
        # Visualize trades
        visualize_trades(df, results, display_name)
        
        return results
        
    except Exception as e:
        print(f"❌ Error processing {display_name}: {str(e)}")
        return None

def print_backtest_summary(all_results, initial_balance, final_balance):
    """Print a concise summary of all backtest results."""
    total_return = ((final_balance - initial_balance) / initial_balance * 100)
    
    print("\n" + "="*120)
    print("📊 COMPREHENSIVE BACKTEST SUMMARY")
    print("="*120)
    
    # Header
    print(f"{'Pair':12} | {'Model Performance':40} | {'Trading Performance':50}")
    print(f"{'':12} | {'Train MAE':>8} {'Val MAE':>8} {'Train R²':>8} {'Val R²':>8} | {'Return %':>8} {'# Trades':>8} {'Win Rate':>8} {'Avg Ret%':>8} {'Avg Hours':>8}")
    print("-"*120)
    
    total_trades = 0
    weighted_win_rate = 0
    
    for r in all_results:
        pair_name = r['pair_name']
        sell_trades = [t for t in r['trades'] if t['type'] == 'sell']
        
        n_trades = len(sell_trades)
        if n_trades == 0:
            continue
            
        # Get model performance metrics
        model = joblib.load(f'models/model_{pair_name.replace("/", "_")}.joblib')
        train_mae = getattr(model, 'train_mae_', 0)
        val_mae = getattr(model, 'val_mae_', 0)
        train_r2 = getattr(model, 'train_r2_', 0)
        val_r2 = getattr(model, 'val_r2_', 0)
        
        # Calculate trading metrics
        pair_return = r['metrics']['total_return']
        win_rate = len([t for t in sell_trades if t['return'] > 0]) / n_trades * 100
        avg_return = np.mean([t['return'] for t in sell_trades])
        avg_duration = np.mean([t['hours_held'] for t in sell_trades])
        
        # Update totals
        total_trades += n_trades
        weighted_win_rate += win_rate * n_trades
        
        print(f"{pair_name:12} | {train_mae:8.3f} {val_mae:8.3f} {train_r2:8.3f} {val_r2:8.3f} | {pair_return:8.2f} {n_trades:8d} {win_rate:7.1f}% {avg_return:8.2f} {avg_duration:8.1f}")
    
    print("="*120)
    
    # Print overall statistics
    if total_trades > 0:
        overall_win_rate = weighted_win_rate / total_trades
        print(f"\n🎯 Overall Statistics:")
        print(f"Initial Balance: €{initial_balance:.2f}")
        print(f"Final Balance:   €{final_balance:.2f}")
        print(f"Total Return:    {total_return:+.2f}%")
        print(f"Total Trades:    {total_trades}")
        print(f"Average Win Rate: {overall_win_rate:.1f}%")
        print(f"\n💰 Annualized Return: {(((1 + total_return/100)**(365/30)) - 1) * 100:.1f}%")
    else:
        print("\n⚠️ No trades were executed during the backtest period")

def main():
    """Main trading bot function."""
    print("\n🚀 Starting Kraken Trading Bot...")
    if TRADING_CONFIG['behavior']['test_run']:
        print("🔬 Running in TEST MODE - No real trades will be executed!")
    
    # Load environment variables
    load_dotenv()
    if not all([os.getenv("KRAKEN_API_KEY"), os.getenv("KRAKEN_API_SECRET")]):
        print("❌ Error: API credentials not found. Check your .env file.")
        return

    # Download historical data
    print("\n📥 Downloading historical data...")
    pair_data = parallel_download_ohlc(TRADING_PAIRS, interval=MODEL_CONFIG['interval_minutes'])
    
    if not pair_data:
        raise Exception("❌ Failed to download data for any trading pair")

    # Initialize trading
    available_balance = TRADING_CONFIG['risk_management']['total_balance']
    last_backtest_time = None
    start_time = datetime.now()
    duration_hours = TRADING_CONFIG['behavior']['duration_hours']
    print(f"⏱️ Bot will run for {duration_hours} hours (until {(start_time + timedelta(hours=duration_hours)).strftime('%Y-%m-%d %H:%M:%S')})")

    # Load or train models and run backtests
    models = {}
    backtest_results = []
    
    for pair, display_name in TRADING_PAIRS:
        if pair not in pair_data or pair_data[pair] is None:
            print(f"\n⚠️ Skipping {display_name} - no data available")
            continue
            
        results = process_trading_pair(pair, display_name, pair_data[pair])
        if results:
            backtest_results.append(results)
            models[pair] = joblib.load(f'models/model_{pair}.joblib')

    # Save overall backtest summary
    if backtest_results:
        summary = {
            'initial_balance': TRADING_CONFIG['risk_management']['total_balance'],
            'final_balance': sum(r['metrics']['final_balance'] for r in backtest_results) / len(backtest_results),
            'total_return': ((sum(r['metrics']['final_balance'] for r in backtest_results) / len(backtest_results)) 
                           - TRADING_CONFIG['risk_management']['total_balance']) / TRADING_CONFIG['risk_management']['total_balance'] * 100,
            'pairs': [{
                'pair': r['pair_name'],
                'return': r['metrics']['total_return'],
                'n_trades': r['metrics']['n_trades'],
                'win_rate': r['metrics']['win_rate'] * 100,
                'strategy_params': r['strategy_params']
            } for r in backtest_results]
        }
        
        os.makedirs('backtesting_results', exist_ok=True)
        with open('backtesting_results/overall_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)

    while True:
        try:
            # Check if we've exceeded the duration
            elapsed_hours = (datetime.now() - start_time).total_seconds() / 3600
            if elapsed_hours >= duration_hours:
                print(f"\n⌛ Duration of {duration_hours} hours reached. Stopping bot...")
                break

            # Evaluate trading opportunities
            opportunities, best_opportunity = evaluate_trading_opportunities(TRADING_PAIRS, models)
            
            # Check positions and print status
            positions = get_open_positions()
            print_trading_status(opportunities, start_time)
            print(f"\n💰 Available Balance: €{available_balance:.2f}")
            print(f"📈 Active Positions: {len(positions)}")
            
            # Handle trading opportunities
            if opportunities and len(positions) < TRADING_CONFIG['risk_management']['max_positions']:
                # Find the opportunity with highest predicted return
                best_opp = max(opportunities, key=lambda x: x['predicted_return'])
                
                print("\n🎯 Best Trading Opportunity Found!")
                print(f"Trading Pair: {best_opp['display_name']}")
                print(f"Current Price: €{best_opp['current_price']:.5f}")
                print(f"Predicted Return: {best_opp['predicted_return']:+.2f}%")
                
                # Calculate position size based on risk management settings
                max_position_size = TRADING_CONFIG['risk_management']['max_position_size']
                position_pct = TRADING_CONFIG['position']['size']
                
                position_size = min(
                    available_balance * (position_pct / 100),  # Percentage of balance
                    max_position_size,  # Maximum position size
                    available_balance * 0.95  # No more than 95% of balance
                )
                
                if position_size < 10.0:
                    print(f"\n⚠️ Available balance too low for trading (minimum €10.00)")
                    print(f"Current balance: €{available_balance:.2f}")
                else:
                    print(f"Position Size: €{position_size:.2f}")
                    success, used_amount = handle_trading_opportunity(best_opp, position_size)
                    if success:
                        available_balance -= used_amount
            else:
                if len(positions) >= TRADING_CONFIG['risk_management']['max_positions']:
                    print("\n⚠️ Maximum number of positions reached")
                else:
                    print("\n😴 No trading opportunities found.")
                if best_opportunity:
                    print("\n👀 Most promising position (below threshold):")
                    print(f"Trading Pair: {best_opportunity['display_name']}")
                    print(f"Current Price: €{best_opportunity['current_price']:.5f}")
                    print(f"Predicted Return: {best_opportunity['predicted_return']:+.2f}%")
            
            print(f"\n💤 Sleeping for {MODEL_CONFIG['interval_minutes']} minutes...")
            time.sleep(MODEL_CONFIG['interval_minutes'] * 60)
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("🔄 Retrying in 15 minutes...")
            time.sleep(900)

if __name__ == "__main__":
    main()