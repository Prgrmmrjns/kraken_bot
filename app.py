import os
import time
from datetime import datetime
import joblib
from dotenv import load_dotenv
from model_functions import create_features, train_model, simulate_trading
from kraken_functions import place_market_order, download_full_ohlc_data

load_dotenv()

API_KEY = os.getenv("KRAKEN_API_KEY")
API_SECRET = os.getenv("KRAKEN_API_SECRET")
# Trading pairs
AVAILABLE_PAIRS = {
    'XRP/EUR': 'XXRPZEUR',
    'BTC/EUR': 'XXBTZEUR',
    'ETH/EUR': 'XETHZEUR',
    'DOT/EUR': 'DOTZEUR',
    'ADA/EUR': 'ADAEUR'
}

def check_api_credentials():
    """Check if API credentials are properly set"""
    api_key = os.getenv("KRAKEN_API_KEY")
    api_secret = os.getenv("KRAKEN_API_SECRET")
    
    if not api_key or not api_secret:
        print("\n❌ ERROR: API credentials not found!")
        print("Please make sure you have set the following environment variables:")
        print("- KRAKEN_API_KEY")
        print("- KRAKEN_API_SECRET")
        print("\nYou can set them by:")
        print("1. Adding them to your .env file:")
        print("   KRAKEN_API_KEY=your_api_key")
        print("   KRAKEN_API_SECRET=your_api_secret")
        print("\n2. Or setting them in your terminal:")
        print("   export KRAKEN_API_KEY=your_api_key")
        print("   export KRAKEN_API_SECRET=your_api_secret")
        return False
    return True

def initialize_model(pair):
    """Initialize or update the model for the selected pair"""
    print(f"\nInitializing model for {pair}...")
    os.environ['TRADING_PAIR'] = pair
    os.system('python init.py')
    return joblib.load('model.joblib')

def evaluate_position(model, current_data):
    """Evaluate whether to open/close a position"""
    df = create_features(current_data)
    df = df.dropna(subset=['target'])
    X = df.drop(columns=['timestamp', 'target'])
    
    predicted_price = model.predict(X)[-1]
    current_price = float(current_data['close'].iloc[-1])
    expected_return = (predicted_price - current_price) / current_price * 100
    
    return current_price, predicted_price, expected_return

def execute_order(order_type, pair, volume, price):
    """Execute a buy/sell order with user confirmation"""
    print(f"\n{'='*50}")
    print(f"EXECUTING {order_type} ORDER")
    print(f"Pair: {pair}")
    print(f"Volume: {volume:.6f}")
    print(f"Current Price: €{price:.5f}")
    print(f"Total Value: €{volume * price:.2f}")
    print(f"{'='*50}")
    
    if not check_api_credentials():
        return False
    
    confirm = input("\nConfirm order (yes/no): ").lower()
    if confirm == 'yes':
        print("\nPlacing order...")
        try:
            response = place_market_order(pair, volume, order_type.lower())
            
            if response and 'error' not in response:
                print(f"\n✅ {order_type} order executed successfully!")
                print("Order details:", response)
                return True
            else:
                error_msg = response.get('error', ['Unknown error'])[0] if response else 'Failed to connect to exchange'
                print(f"\n❌ Failed to execute {order_type} order: {error_msg}")
                return False
        except Exception as e:
            print(f"\n❌ Error executing order: {str(e)}")
            return False
    else:
        print("\nOrder cancelled by user")
        return False

def main():
    # Check API credentials at startup
    if not check_api_credentials():
        print("\nCannot proceed without valid API credentials.")
        return
    
    # Select trading pair
    print("\nAvailable trading pairs:")
    for i, pair in enumerate(AVAILABLE_PAIRS.keys(), 1):
        print(f"{i}. {pair}")
    
    while True:
        try:
            choice = int(input("\nSelect trading pair (1-5): ")) - 1
            display_pair = list(AVAILABLE_PAIRS.keys())[choice]
            selected_pair = AVAILABLE_PAIRS[display_pair]
            break
        except (ValueError, IndexError):
            print("Invalid choice. Please select a number between 1 and 5.")
    
    # Get trading parameters
    min_confidence = 2 # Minimum predicted increase in price to buy
    position_size = 10 # Amount of EUR for each trade
    take_profit = 5 # When to trigger take profit
    
    print(f"\nInitializing bot for {display_pair}...")
    model = initialize_model(selected_pair)
    
    position_open = False
    entry_price = None
    
    print("\nBot is running. Press Ctrl+C to stop.")
    
    try:
        while True:
            # Get current market data
            print(f"\nFetching data for {display_pair}...")
            current_data = download_full_ohlc_data(selected_pair, interval=60)
            
            # Evaluate position
            current_price, predicted_price, expected_return = evaluate_position(model, current_data)
            
            # Print market status
            print(f"\n{'='*50}")
            print(f"MARKET STATUS - {display_pair}")
            print(f"{'='*50}")
            print(f"Current Price: €{current_price:.5f}")
            print(f"Predicted Price: €{predicted_price:.5f}")
            print(f"Expected Return: {expected_return:+.2f}%")
            
            if position_open:
                current_return = (current_price - entry_price) / entry_price * 100
                print(f"Current Position Return: {current_return:+.2f}%")
                print(f"Take Profit Target: {take_profit:.2f}%")
            
            # Trading logic
            if not position_open:
                if expected_return > min_confidence:
                    print("\n💰 BUY SIGNAL DETECTED")
                    print(f"Expected return ({expected_return:.2f}%) exceeds minimum confidence ({min_confidence:.2f}%)")
                    
                    if execute_order("BUY", selected_pair, position_size/current_price, current_price):
                        position_open = True
                        entry_price = current_price
                else:
                    print("\nNo buy signal at this time")
            else:
                current_return = (current_price - entry_price) / entry_price * 100
                take_profit_triggered = current_return >= take_profit
                prediction_sell_signal = expected_return < 0
                
                if take_profit_triggered or prediction_sell_signal:
                    print("\n📈 SELL SIGNAL DETECTED")
                    reason = "Take profit target reached" if take_profit_triggered else "Negative price movement predicted"
                    print(f"Reason: {reason}")
                    print(f"Current return: {current_return:.2f}%")
                    
                    if execute_order("SELL", selected_pair, position_size/entry_price, current_price):
                        position_open = False
                        entry_price = None
                else:
                    print("\nHolding position...")
            
            # Wait for next evaluation
            print("\nWaiting 60 seconds for next evaluation...")
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\n\nBot stopped by user")
        if position_open:
            print(f"\nWARNING: Position still open at €{current_price:.5f} ({current_return:+.2f}%)")

if __name__ == "__main__":
    main() 