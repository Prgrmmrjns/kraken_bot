import os
import time
import json
import pandas as pd
import numpy as np
import requests
import urllib.request
import urllib.parse
import hmac
import hashlib
import base64
from datetime import datetime
from model_functions import create_features, train_model
from trading_strat_params import TRADING_CONFIG, MODEL_CONFIG
import joblib

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Update trading config with API keys
TRADING_CONFIG['api'] = {
    'key': os.getenv("KRAKEN_API_KEY"),
    'secret': os.getenv("KRAKEN_API_SECRET")
}

def download_recent_ohlc(pair, interval=15, since=None):
    """Download recent OHLC data for a trading pair."""
    url = 'https://api.kraken.com/0/public/OHLC'
    
    # Calculate since time if not provided (last 24 hours)
    if since is None:
        since = int(time.time() - 24 * 60 * 60)  # Last 24 hours
    
    params = {
        'pair': pair,
        'interval': interval,
        'since': since
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'error' in data and data['error']:
            print(f"API Error for {pair}: {data['error']}")
            return None
            
        if 'result' not in data or pair not in data['result']:
            print(f"No data available for {pair}")
            return None
            
        df = pd.DataFrame(
            data['result'][pair],
            columns=['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count']
        )
        
        # Convert types
        df = df.astype({
            'timestamp': 'int64',
            'open': 'float64',
            'high': 'float64',
            'low': 'float64',
            'close': 'float64',
            'vwap': 'float64',
            'volume': 'float64',
            'count': 'int64'
        })
        
        return df
        
    except Exception as e:
        print(f"Error downloading OHLC data for {pair}: {str(e)}")
        return None

def get_ticker(pair):
    """Get current ticker information for a trading pair."""
    try:
        url = f"https://api.kraken.com/0/public/Ticker?pair={pair}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if 'error' in data and data['error']:
            print(f"Error getting ticker for {pair}: {data['error']}")
            return None
            
        return data['result'][pair]
    except Exception as e:
        print(f"Error getting ticker: {str(e)}")
        return None

def execute_order(pair, euro_amount, order_type, skip_confirm=False):
    """Execute a buy/sell order with optional user confirmation"""
    print(f"\n{'='*50}")
    print(f"EXECUTING {order_type} ORDER")
    print(f"Pair: {pair}")
    print(f"Amount: €{euro_amount:.2f}")
    
    if not skip_confirm:
        confirm = input("\nConfirm trade (y/n): ").lower()
        if confirm != 'y':
            print("\nTrade cancelled")
            return False

    try:
        response = place_market_order(pair, euro_amount, order_type.lower())
        if response and 'result' in response and 'txid' in response['result']:
            txid = response['result']['txid'][0]
            print(f"\n✅ {order_type} order executed successfully!")
            print(f"Transaction ID: {txid}")
            
            if order_type.lower() == "buy":
                if current_price := get_current_price(pair):
                    volume = euro_amount / current_price
                    open_positions.append(Position(pair, current_price, volume, datetime.now()))
                    print(f"Position added at price: €{current_price:.5f}")
            return True
            
        error_msg = response.get('error', ['Unknown error'])[0] if response else 'Failed to connect'
        print(f"\n❌ Failed to execute {order_type} order: {error_msg}")
        return False
        
    except Exception as e:
        print(f"\n❌ Error executing order: {str(e)}")
        return False

class Position:
    def __init__(self, pair, entry_price, size, timestamp):
        self.pair = pair
        self.entry_price = entry_price
        self.size = size
        self.timestamp = timestamp
        self.current_return = 0.0

    def update_return(self, current_price):
        self.current_return = (current_price - self.entry_price) / self.entry_price * 100
        return self.current_return

# Keep track of all positions
open_positions = []

def get_current_price(pair):
    """Fetch current market price for the specified trading pair."""
    try:
        response = requests.get(f"https://api.kraken.com/0/public/Ticker?pair={pair}")
        response.raise_for_status()
        data = response.json()
        
        if 'error' in data and data['error']:
            print(f"Error getting price for {pair}: {data['error']}")
            return None
            
        # Get the result for this pair
        pair_data = data['result'].get(pair)
        if not pair_data:
            print(f"No price data found for {pair}")
            return None
            
        # Get the last trade price
        last_trade_price = pair_data.get('c')
        if not last_trade_price or not last_trade_price[0]:
            print(f"No last trade price found for {pair}")
            return None
            
        return float(last_trade_price[0])
        
    except Exception as e:
        print(f"Error fetching current price: {str(e)}")
        return None

def place_market_order(pair, euro_amount, order_type):
    """
    Place a market order on Kraken spending the specified amount in EUR to buy/sell the base currency.
    """
    # Get the current price of the base currency in EUR
    price = get_current_price(pair)
    if price is None:
        print("Cannot place order without price information.")
        return None

    # Calculate the volume of base currency to buy/sell
    volume = euro_amount / price
    volume = round(volume, 8)  # Kraken typically allows up to 8 decimal places

    url_path = "/0/private/AddOrder"
    url = "https://api.kraken.com" + url_path

    nonce = str(int(time.time() * 1000))

    data = {
        "nonce": nonce,
        "ordertype": "market",
        "pair": pair,
        "volume": f"{volume:.8f}",
        "type": order_type,
    }
    
    message = data["nonce"] + urllib.parse.urlencode(data)
    sha256_hash = hashlib.sha256(message.encode("utf-8")).digest()
    secret_bytes = base64.b64decode(TRADING_CONFIG['api']['secret'])
    hmac_digest = hmac.new(secret_bytes, url_path.encode("utf-8") + sha256_hash, hashlib.sha512).digest()
    signature = base64.b64encode(hmac_digest)

    headers = {
        "API-Key": TRADING_CONFIG['api']['key'],
        "API-Sign": signature.decode("utf-8")
    }
    
    post_bytes = urllib.parse.urlencode(data).encode("utf-8")

    req = urllib.request.Request(url, data=post_bytes, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as response:
            response_data = json.loads(response.read().decode())
            
        # Check if order was successful (Kraken returns txid for successful orders)
        if 'result' in response_data and 'txid' in response_data['result']:
            return response_data
        else:
            print(f"Unexpected API response: {response_data}")
            return None
            
    except Exception as e:
        print(f"Error placing order: {str(e)}")
        return None

def get_open_positions():
    """Get currently open positions from Kraken."""
    try:
        # Get private data about positions
        url_path = "/0/private/OpenPositions"
        nonce = str(int(time.time() * 1000))
        
        data = {
            "nonce": nonce,
        }
        
        message = data["nonce"] + urllib.parse.urlencode(data)
        sha256_hash = hashlib.sha256(message.encode()).digest()
        hmac_digest = hmac.new(
            base64.b64decode(TRADING_CONFIG['api']['secret']),
            url_path.encode() + sha256_hash,
            hashlib.sha512
        ).digest()
        
        headers = {
            "API-Key": TRADING_CONFIG['api']['key'],
            "API-Sign": base64.b64encode(hmac_digest).decode()
        }
        
        response = requests.post(
            "https://api.kraken.com" + url_path,
            data=data,
            headers=headers
        )
        response.raise_for_status()
        result = response.json()
        
        if 'error' in result and result['error']:
            print(f"API Error: {result['error']}")
            return []
            
        positions = []
        for pos_id, pos_data in result.get('result', {}).items():
            positions.append({
                'pair': pos_data['pair'],
                'type': pos_data['type'],  # 'buy' or 'sell'
                'volume': float(pos_data['vol']),
                'price': float(pos_data['cost']),
                'pnl': float(pos_data.get('net', 0))  # Current P&L
            })
        
        return positions
        
    except Exception as e:
        print(f"Error getting open positions: {str(e)}")
        return []

def download_full_ohlc_data(pair, interval=15):
    """Download historical OHLC data for given trading pair and interval."""
    url = 'https://api.kraken.com/0/public/OHLC'
    params = {'pair': pair, 'interval': interval}
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'error' in data and data['error']:
            print(f"API Error for {pair}: {data['error']}")
            return None
            
        if 'result' not in data or pair not in data['result']:
            print(f"No data available for {pair}")
            return None
            
        df = pd.DataFrame(
            data['result'][pair],
            columns=['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count']
        )
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.astype({col: float for col in df.columns if col != 'timestamp'})
        
        return df
        
    except Exception as e:
        print(f"Error downloading OHLC data: {str(e)}")
        return None

def check_and_manage_positions():
    """Check and manage open positions."""
    closed_positions = []  # Initialize empty list
    
    # Get open positions
    try:
        positions = get_open_positions()
        if not positions:
            return closed_positions  # Return empty list if no open positions
            
        for position in positions:
            current_price = get_current_price(position['pair'])
            entry_price = float(position['price'])
            current_return = (current_price - entry_price) / entry_price * 100
            
            # Check take profit condition
            if current_return >= TRADING_CONFIG['position']['take_profit']:
                if execute_order(position['pair'], position['volume'], "SELL"):
                    closed_positions.append({
                        'pair': position['pair'],
                        'pnl': position['volume'] * current_price * (current_return / 100)
                    })
        
        return closed_positions
        
    except Exception as e:
        print(f"Error managing positions: {str(e)}")
        return closed_positions  # Return empty list on error

def print_trading_status(opportunities):
    """Print current trading status and position information"""
    print(f"\n{'='*50}")
    print("MARKET STATUS")
    print(f"{'='*50}")
    
    for opp in opportunities:
        print(f"\n{opp['display_name']}:")
        print(f"Current Price: €{opp['current_price']:.5f}")
        print(f"Predicted Return: {opp['predicted_return']:+.2f}%")
    
    print(f"\nOpen Positions: {len(open_positions)}")
    for i, pos in enumerate(open_positions, 1):
        if current_price := get_current_price(pos.pair):
            print(f"\nPosition {i}:")
            print(f"Pair: {pos.pair}")
            print(f"Entry Price: €{pos.entry_price:.5f}")
            print(f"Current Price: €{current_price:.5f}")
            print(f"Size: €{pos.size:.2f}")
            print(f"Current Return: {pos.update_return(current_price):+.2f}%")
            print(f"Age: {(datetime.now() - pos.timestamp).seconds // 60} minutes")

def train_and_save_model(trading_pairs):
    """Train and save separate models for each trading pair"""
    print("\nDownloading data for all pairs...")
    pair_data = {}
    skipped_pairs = []
    models = {}
    
    for pair, display_name in trading_pairs:
        try:
            df = download_full_ohlc_data(pair=pair, interval=MODEL_CONFIG['interval_minutes'])
            if df is not None and not df.empty:
                pair_data[pair] = df
            else:
                skipped_pairs.append((pair, display_name))
        except Exception as e:
            print(f"Error processing {display_name} ({pair}): {str(e)}")
            skipped_pairs.append((pair, display_name))
    
    if not pair_data:
        raise Exception("Failed to download data for any trading pair")
    
    if skipped_pairs:
        print("\nSkipped pairs:")
        for pair, display_name in skipped_pairs:
            print(f"- {display_name} ({pair})")
    
    print("\nCreating features and training models for each pair...")
    pair_features = create_features(pair_data)
    
    for pair, df in pair_features.items():
        print(f"\nTraining model for {pair}...")
        df = df.dropna(subset=['target'])
        
        X = df.drop(columns=['timestamp', 'target'])
        y = df['target']
        
        # Split data into train and validation
        val_split_idx = int(len(df) * 0.8)
        X_train, X_val = X[:val_split_idx], X[val_split_idx:]
        y_train, y_val = y[:val_split_idx], y[val_split_idx:]
        
        # Train model for this pair
        models[pair] = train_model(X_train, X_val, y_train, y_val)
        
        # Save pair-specific model
        joblib.dump(models[pair], f'model_{pair}.joblib')
    
    return models