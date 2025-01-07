import os
import time
import json
import base64
import hmac
import hashlib
import urllib.parse
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv('KRAKEN_API_KEY')
API_SECRET = os.getenv('KRAKEN_API_SECRET')
API_URL = "https://api.kraken.com"

def get_kraken_signature(urlpath, data, secret):
    """Generate Kraken API signature."""
    postdata = urllib.parse.urlencode(data)
    encoded = (str(data['nonce']) + postdata).encode()
    message = urlpath.encode() + hashlib.sha256(encoded).digest()
    mac = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
    sigdigest = base64.b64encode(mac.digest())
    return sigdigest.decode()

def kraken_request(uri_path, data, api_key, api_sec):
    """Send signed Kraken API request."""
    headers = {}
    headers['API-Key'] = api_key
    headers['API-Sign'] = get_kraken_signature(uri_path, data, api_sec)
    req = requests.post((API_URL + uri_path), headers=headers, data=data)
    return req

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
    """Execute a trade order."""
    try:
        # Get current price to calculate volume
        df = download_recent_ohlc(pair=pair)
        if df is None or df.empty:
            print(f"Could not get current price for {pair}")
            return False
            
        current_price = float(df['close'].iloc[-1])
        volume = euro_amount / current_price  # Convert euros to crypto volume
        
        # Prepare the order data
        data = {
            "nonce": str(int(1000*time.time())),
            "ordertype": "market",
            "type": order_type.lower(),
            "volume": f"{volume:.8f}",  # Use 8 decimal places for precision
            "pair": pair,
            "trading_agreement": "agree"
        }
        
        # Get user confirmation if needed
        if not skip_confirm:
            confirm = input(f"Confirm {order_type} order for {volume:.8f} {pair} (€{euro_amount:.2f})? (y/n): ")
            if confirm.lower() != 'y':
                print("Order cancelled by user")
                return False
        
        # Send the order request
        response = kraken_request('/0/private/AddOrder', data, API_KEY, API_SECRET)
        
        # Check for errors
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return False
            
        result = response.json()
        if result.get('error'):
            print(f"API Error: {result['error']}")
            return False
            
        # Order successful
        order_id = result['result']['txid'][0]
        print(f"Order executed successfully. ID: {order_id}")
        return True
        
    except Exception as e:
        print(f"Error executing order: {str(e)}")
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

def get_account_balance():
    """Get the current account balance."""
    try:
        # For now, return the initial balance from config
        return TRADING_CONFIG['risk_management']['total_balance']
    except Exception as e:
        print(f"Error getting account balance: {str(e)}")
        return 0.0

def get_open_positions():
    """Get current open positions."""
    try:
        # For now, return an empty list
        return []
    except Exception as e:
        print(f"Error getting open positions: {str(e)}")
        return []

def get_ticker_info(pair):
    """Get current ticker information for a pair."""
    try:
        df = download_recent_ohlc(pair=pair, interval=1)
        if df is not None and not df.empty:
            return {
                'price': float(df['close'].iloc[-1]),
                'volume': float(df['volume'].iloc[-1]),
                'timestamp': df.index[-1]
            }
        return None
    except Exception as e:
        print(f"Error getting ticker info: {str(e)}")
        return None