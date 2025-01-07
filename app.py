import os
import time
from datetime import datetime
from dotenv import load_dotenv
import joblib
import warnings
import json

# Suppress all warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from kraken_functions import (
    execute_order,
    check_and_manage_positions,
    download_recent_ohlc,
)
from trading_strat_params import TRADING_PAIRS, TRADING_CONFIG, MODEL_CONFIG
from model_functions import (
    create_features_for_pair,
    get_model_features,
    load_trading_params
)
from backtesting import run_backtest

def prepare_features(features_df, model_features):
    """Prepare features to match model's expected features."""
    if model_features is None:
        return features_df
        
    # Get common features
    common_features = list(set(features_df.columns) & set(model_features))
    
    # Check if we have all required features
    missing_features = set(model_features) - set(features_df.columns)
    if missing_features:
        print(f"Missing features: {missing_features}")
        return None
        
    # Return only the features the model expects, in the correct order
    return features_df[model_features]

def should_run_backtest(last_backtest_time):
    """Check if we should run backtest based on time elapsed."""
    if last_backtest_time is None:
        return True
    hours_elapsed = (datetime.now() - last_backtest_time).total_seconds() / 3600
    return hours_elapsed >= 6

def main():
    """Main trading bot function."""
    print("\n🚀 Starting Kraken Trading Bot...")
    
    # Load environment variables
    load_dotenv()
    if not all([os.getenv("KRAKEN_API_KEY"), os.getenv("KRAKEN_API_SECRET")]):
        print("❌ Error: API credentials not found. Check your .env file.")
        return

    # Initialize balance from trading config
    available_balance = TRADING_CONFIG['risk_management']['total_balance']
    last_backtest_time = None

    while True:
        try:
            # Check if we need to run backtest
            if should_run_backtest(last_backtest_time):
                print("\n🔄 Running scheduled backtest...")
                run_backtest()
                last_backtest_time = datetime.now()
                print(f"✅ Backtest completed at {last_backtest_time.strftime('%Y-%m-%d %H:%M:%S')}")

            # Load trading parameters for each pair
            pair_params = {}
            active_pairs = []
            for pair, display_name in TRADING_PAIRS:
                params = load_trading_params(display_name)
                if params is not None:  # Only add pairs that have valid parameters
                    pair_params[pair] = params
                    active_pairs.append((pair, display_name))
            
            if not active_pairs:
                print("\n⚠️ No active trading pairs found. Waiting for next backtest...")
                time.sleep(300)  # Wait 5 minutes before checking again
                continue

            # Check positions and print status
            positions = check_and_manage_positions()
            print(f"\n{'='*50}")
            print(f"TRADING STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*50}")
            print(f"\n💰 Available Balance: €{available_balance:.2f}")
            print(f"Active Positions: {len(positions)}")
            
            # Evaluate trading opportunities
            opportunities = []
            best_opportunity = None
            best_prediction = float('-inf')
            
            for pair, display_name in active_pairs:  # Only loop through active pairs
                try:
                    # Get current market data
                    df = download_recent_ohlc(pair=pair)
                    if df is None or len(df) < 50:
                        continue
                        
                    # Load model and get expected features
                    model_file = f'models/model_{pair}.joblib'
                    if not os.path.exists(model_file):
                        print(f"⚠️ Model not found for {display_name}")
                        continue
                    
                    model = joblib.load(model_file)
                    model_features = get_model_features(model_file)
                    
                    # Create and prepare features
                    features_df = create_features_for_pair(df, display_name)
                    if features_df is None:
                        continue
                    
                    # Prepare features to match model's expectations
                    features = prepare_features(features_df, model_features)
                    if features is None:
                        print(f"❌ Feature mismatch for {display_name}")
                        continue
                        
                    # Make prediction
                    prediction = model.predict(features)[-1]
                    current_price = float(df['close'].iloc[-1])
                    
                    # Track best opportunity even if below threshold
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
                    print(f"❌ Error processing {display_name}: {str(e)}")
                    continue
            
            # Handle trading opportunities
            if opportunities:
                # Find the opportunity with highest predicted return
                best_opp = max(opportunities, key=lambda x: x['predicted_return'])
                
                print("\n🎯 Best Trading Opportunity Found!")
                print(f"Trading Pair: {best_opp['display_name']}")
                print(f"Current Price: €{best_opp['current_price']:.5f}")
                print(f"Predicted Return: {best_opp['predicted_return']:+.2f}%")
                print(f"Take Profit: {best_opp['params']['take_profit']}%")
                print(f"Stop Loss: {best_opp['params']['trailing_stop']}%")
                
                # Calculate position size based on risk management
                position_size = min(
                    available_balance,
                    max(10.0, available_balance * (float(os.getenv('RISK_PERCENTAGE', 10)) / 100))  # Minimum €10, default 10% risk
                )
                
                if position_size < 10.0:  # Minimum trade amount
                    print(f"\n⚠️ Available balance too low for trading (minimum €10.00)")
                    print(f"Current balance: €{available_balance:.2f}")
                else:
                    print(f"Position Size: €{position_size:.2f}")
                    
                    # Execute trade based on confirmation settings
                    should_confirm = TRADING_CONFIG['behavior']['confirm_order']
                    if should_confirm:
                        confirm = input(f"\nExecute trade for {best_opp['display_name']} at €{best_opp['current_price']:.2f}? (y/n): ").lower()
                        should_execute = confirm == 'y'
                    else:
                        should_execute = True
                        
                    if should_execute:
                        try:
                            if execute_order(best_opp['pair'], position_size, "BUY", skip_confirm=True):
                                print(f"✅ Buy order executed for {best_opp['display_name']}")
                                available_balance -= position_size
                            else:
                                print(f"❌ Failed to execute buy order for {best_opp['display_name']}")
                        except Exception as e:
                            print(f"❌ Error executing trade: {str(e)}")
                    else:
                        print("Trade skipped by user.")
            else:
                print("\n😴 No trading opportunities found.")
                if best_opportunity:
                    print("\n👀 Most promising position (below threshold):")
                    print(f"Trading Pair: {best_opportunity['display_name']}")
                    print(f"Current Price: €{best_opportunity['current_price']:.5f}")
                    print(f"Predicted Return: {best_opportunity['predicted_return']:+.2f}%")
                    print(f"Buy Threshold: {best_opportunity['params']['buy_threshold']}%")
            
            print(f"\n💤 Sleeping for {MODEL_CONFIG['interval_minutes']} minutes...")
            time.sleep(MODEL_CONFIG['interval_minutes'] * 60)
            
        except KeyboardInterrupt:
            print("\n👋 Trading bot stopped by user.")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print(f"🔄 Retrying in {MODEL_CONFIG['interval_minutes']} minutes...")
            time.sleep(MODEL_CONFIG['interval_minutes'] * 60)

if __name__ == "__main__":
    main()