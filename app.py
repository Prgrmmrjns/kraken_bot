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
from model_functions import create_features_for_pair

def load_trading_params(pair):
    """Load trading parameters from file."""
    try:
        filename = f"backtesting_results/{pair.replace('/', '_')}_trades.txt"
        params = {}
        
        with open(filename, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.strip() == "Strategy Parameters:":
                    param_lines = lines[i+1:i+11]
                    for param_line in param_lines:
                        if ':' in param_line:
                            key, value = param_line.strip().split(': ')
                            key_mapping = {
                                'buy_threshold': 'buy_threshold',
                                'take_profit_threshold': 'take_profit',
                                'max_hold_hours': 'max_hold_time',
                                'trailing_stop_distance': 'trailing_stop',
                                'min_rsi': 'min_rsi',
                                'max_rsi': 'max_rsi',
                                'min_volume_ratio': 'min_volume_ratio',
                                'max_volatility': 'max_volatility',
                                'profit_lock_pct': 'profit_lock',
                                'optimization_score': 'score'
                            }
                            for old_key, new_key in key_mapping.items():
                                if old_key in key:
                                    params[new_key] = float(value)
                                    break
                    break
        
        if params:
            return params
        
        print(f"❌ No parameters found in {filename}. Make sure to run backtesting.py first. Using default parameters for now.")
        raise ValueError("Invalid file format")
        
    except Exception as e:
        print(f"⚠️ Using default parameters for {pair}")
        return {
            'buy_threshold': 1.5,
            'take_profit': 2.0,
            'max_hold_time': 24,
            'trailing_stop': 0.5,
            'min_rsi': 30,
            'max_rsi': 70,
            'min_volume_ratio': 1.0,
            'max_volatility': 3.0,
            'profit_lock': 0.5,
            'score': 0.0
        }

def get_model_features(model_file):
    """Get feature names from the trained model."""
    model = joblib.load(model_file)
    try:
        return model.feature_names_
    except AttributeError:
        # If feature names not stored in model, load from model info
        try:
            with open('model_info.txt', 'r') as f:
                model_info = json.loads(f.read())
                return model_info['data_info']['features']
        except:
            return None

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

def is_prediction_realistic(prediction, current_price):
    """Check if prediction is within realistic bounds."""
    # Maximum realistic daily return (e.g., 10%)
    MAX_REALISTIC_RETURN = 10.0
    
    return abs(prediction) <= MAX_REALISTIC_RETURN

def main():
    """Main trading bot function."""
    print("\n🚀 Starting Kraken Trading Bot...")
    
    # Load environment variables
    load_dotenv()
    if not all([os.getenv("KRAKEN_API_KEY"), os.getenv("KRAKEN_API_SECRET")]):
        print("❌ Error: API credentials not found. Check your .env file.")
        return

    # Load trading parameters for each pair
    pair_params = {}
    for pair, display_name in TRADING_PAIRS:
        params = load_trading_params(display_name)
        pair_params[pair] = params
    
    while True:
        try:
            # Check positions and print status
            positions = check_and_manage_positions()
            print(f"\n{'='*50}")
            print(f"TRADING STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*50}")
            print(f"\nActive Positions: {len(positions)}")
            
            # Evaluate trading opportunities
            opportunities = []
            for pair, display_name in TRADING_PAIRS:
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
                    
                    # Check trading conditions
                    params = pair_params[pair]
                    if prediction > params['buy_threshold']:
                        opportunities.append({
                            'pair': pair,  # Kraken's internal symbol (e.g., 'XXBTZEUR')
                            'display_name': display_name,  # Human-readable name (e.g., 'BTC/EUR')
                            'current_price': current_price,
                            'predicted_return': prediction,
                            'params': params
                        })
                        
                        print(f"\n✨ Opportunity found for {display_name}:")
                        print(f"💰 Current Price: €{current_price:.5f}")
                        print(f"📈 Predicted Return: {prediction:+.2f}%")
                
                except Exception as e:
                    print(f"❌ Error processing {display_name}: {str(e)}")
                    continue
            
            # Handle trading opportunities
            if opportunities:
                # Find the opportunity with highest predicted return
                best_opp = max(opportunities, key=lambda x: x['predicted_return'])
                
                print("\n🎯 Best Trading Opportunity Found!")
                print(f"\n{'='*30}")
                print(f"Trading Pair: {best_opp['display_name']}")
                print(f"Current Price: €{best_opp['current_price']:.5f}")
                print(f"Predicted Return: {best_opp['predicted_return']:+.2f}%")
                print(f"Take Profit: {best_opp['params']['take_profit']}%")
                print(f"Stop Loss: {best_opp['params']['trailing_stop']}%")
                
                # Calculate position size based on risk management
                available_balance = float(os.getenv('MAX_POSITION_SIZE', 1000))
                position_size = min(
                    available_balance,
                    available_balance * (float(os.getenv('RISK_PERCENTAGE', 1)) / 100)
                )
                
                print(f"Suggested Position Size: €{position_size:.2f}")
                
                # Ask for user confirmation ONCE
                confirm = input(f"\nExecute trade for {best_opp['display_name']} at €{best_opp['current_price']:.2f}? (y/n): ").lower()
                if confirm == 'y':
                    try:
                        # Execute the trade using Kraken's internal symbol
                        if execute_order(best_opp['pair'], position_size, "BUY", skip_confirm=True):
                            print(f"✅ Buy order executed for {best_opp['display_name']}")
                        else:
                            print(f"❌ Failed to execute buy order for {best_opp['display_name']}")
                    except Exception as e:
                        print(f"❌ Error executing trade: {str(e)}")
                else:
                    print("Trade skipped by user.")
            else:
                print("\n😴 No trading opportunities found.")
            
            print("\n💤 Sleeping for 5 minutes...")
            time.sleep(300)  # Sleep for 5 minutes
            
        except KeyboardInterrupt:
            print("\n👋 Trading bot stopped by user.")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("🔄 Retrying in 5 minutes...")
            time.sleep(300)

if __name__ == "__main__":
    main()