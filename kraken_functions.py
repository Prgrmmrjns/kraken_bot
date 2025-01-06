import pandas as pd
import urllib.request
import urllib.parse
import json
import time
import hashlib
import hmac
import base64
import os
import requests

API_KEY = os.getenv("KRAKEN_API_KEY")
API_SECRET = os.getenv("KRAKEN_API_SECRET")

def download_full_ohlc_data(pair, interval):
    """
    Downloads historical OHLC data for given trading pair and interval.
    Uses pagination to get as much historical data as possible.
    
    Args:
        pair (str): Trading pair symbol e.g. 'XXRPZEUR'
        interval (int): Time interval in minutes
        
    Returns:
        pd.DataFrame: DataFrame with OHLC data
    """
    all_data = []
    last_timestamp = None
    
    url = 'https://api.kraken.com/0/public/OHLC'
    params = {
        'pair': pair,
        'interval': interval
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"API request failed with status {response.status_code}")
    
    data = response.json()
        
    ohlc_data = data['result'][pair]
        
    df = pd.DataFrame(ohlc_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 
        'vwap', 'volume', 'count'
    ])
    
    # Convert timestamp to datetime for comparison
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Update last timestamp for next iteration
    last_timestamp = df['timestamp'].min()
    most_recent_timestamp = df['timestamp'].max()
    
    # Only append if we got new data
    if not all_data or df['timestamp'].max() < all_data[-1]['timestamp'].min():
        all_data.append(df)
        print(f"Downloaded data starting from {last_timestamp} up to {most_recent_timestamp}")
    
    # Combine all data and process
    full_df = pd.concat(all_data, ignore_index=True)
    
    # Convert strings to numeric
    for col in ['open', 'high', 'low', 'close', 'vwap', 'volume']:
        full_df[col] = pd.to_numeric(full_df[col])
    
    return full_df
def get_current_price(pair):
    """
    Fetch the current market price for the specified trading pair.

    Args:
        pair (str): Trading pair symbol (e.g., "XRPEUR")

    Returns:
        float: Current price of the base currency in quote currency
    """
    url = f"https://api.kraken.com/0/public/Ticker?pair={pair}"
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
        result = data["result"]
        pair_key = list(result.keys())[0]
        price = float(result[pair_key]["c"][0])  # 'c' is the last trade closed price
        return price
    except Exception as e:
        print(f"Error fetching current price: {e}")
        return None

def place_market_order(pair, euro_amount, order_type):
    """
    Place a market order on Kraken spending the specified amount in EUR to buy/sell the base currency.

    Args:
        pair (str): Trading pair symbol (e.g., "XRPEUR")
        euro_amount (float): Amount in EUR to spend
        order_type (str): Type of order - "buy" or "sell"

    Returns:
        dict or None: Response from Kraken API or None if an error occurred
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
    secret_bytes = base64.b64decode(API_SECRET)
    hmac_digest = hmac.new(secret_bytes, url_path.encode("utf-8") + sha256_hash, hashlib.sha512).digest()
    signature = base64.b64encode(hmac_digest)

    headers = {
        "API-Key": API_KEY,
        "API-Sign": signature.decode("utf-8")
    }
    post_bytes = urllib.parse.urlencode(data).encode("utf-8")

    req = urllib.request.Request(url, data=post_bytes, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as response:
            response_data = json.loads(response.read().decode())
        return response_data
    except Exception as e:
        print(f"Error placing order: {e}")
        return None