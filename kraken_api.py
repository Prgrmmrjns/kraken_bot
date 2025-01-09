"""
Kraken API Module

This module handles all interactions with the Kraken cryptocurrency exchange API.
"""

import os
import time
import json
import base64
import hmac
import hashlib
import urllib.parse
import requests
import pandas as pd
import concurrent.futures
from dotenv import load_dotenv
from params import API_CONFIG, TRADING_CONFIG

# Load environment variables
load_dotenv()
API_KEY = os.getenv('KRAKEN_API_KEY')
API_SECRET = os.getenv('KRAKEN_API_SECRET')

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
    req = requests.post((API_CONFIG['url'] + uri_path), headers=headers, data=data)
    return req

def parallel_download_ohlc(pairs, interval=15, since=None):
    """
    Download OHLC data in parallel using ThreadPoolExecutor.
    Faster, this will be, than sequential downloads, hrrrm.
    """
    def fetch_data(pair, since_param=None):
        # Create data directory if it doesn't exist
        os.makedirs('historical_data', exist_ok=True)
        data_file = f'historical_data/{pair}_ohlc.parquet'
        
        # Load existing data if available
        existing_data = None
        if os.path.exists(data_file):
            try:
                existing_data = pd.read_parquet(data_file)
                existing_data.index = pd.to_datetime(existing_data.index)
            except Exception as e:
                print(f"Error loading existing data for {pair}: {str(e)}")
        
        # Calculate the since parameter
        if since_param is None:
            if existing_data is not None:
                # Get data since the last entry, with some overlap for safety
                last_timestamp = existing_data.index.max()
                since_param = int(last_timestamp.timestamp()) - (24 * 60 * 60)  # 1 day overlap
            else:
                # Default to 30 days if no existing data
                since_param = int(time.time() - 30 * 24 * 60 * 60)
        
        url = f"{API_CONFIG['url']}/{API_CONFIG['version']}/public/OHLC"
        params = {
            'pair': pair,
            'interval': interval,
            'since': since_param
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'error' in data and data['error']:
                print(f"API Error for {pair}: {data['error']}")
                return (pair, existing_data)  # Return existing data if available
                
            if 'result' not in data or pair not in data['result']:
                print(f"No data available for {pair}")
                return (pair, existing_data)  # Return existing data if available
                
            # Create DataFrame from new data
            ohlc_data = data['result'][pair]
            if not ohlc_data:  # Check if data is empty
                print(f"No new data available for {pair}")
                return (pair, existing_data)
                
            # Convert data to correct types first
            new_data = {
                'open': [float(x[1]) for x in ohlc_data],
                'high': [float(x[2]) for x in ohlc_data],
                'low': [float(x[3]) for x in ohlc_data],
                'close': [float(x[4]) for x in ohlc_data],
                'vwap': [float(x[5]) for x in ohlc_data],
                'volume': [float(x[6]) for x in ohlc_data],
                'count': [int(x[7]) for x in ohlc_data]
            }
            
            # Create DataFrame with timestamp index
            timestamps = [pd.to_datetime(int(x[0]), unit='s') for x in ohlc_data]
            new_df = pd.DataFrame(new_data, index=timestamps)
            
            # Merge with existing data if available
            if existing_data is not None:
                # Combine and remove duplicates, keeping the newer data
                combined_df = pd.concat([existing_data, new_df])
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                combined_df.sort_index(inplace=True)
                
                # Save merged data
                combined_df.to_parquet(data_file)
                return (pair, combined_df)
            else:
                # Save new data
                new_df.to_parquet(data_file)
                return (pair, new_df)
                
        except Exception as e:
            print(f"Error downloading OHLC data for {pair}: {str(e)}")
            return (pair, existing_data)  # Return existing data if available

    results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(fetch_data, p[0], since): p for p in pairs}
        for future in concurrent.futures.as_completed(futures):
            pair = futures[future][0]
            display_name = futures[future][1]
            pair_symbol, df_result = future.result()
            results[pair_symbol] = df_result
            
            if df_result is not None:
                print(f"✅ Data updated for {display_name} - {len(df_result)} entries")
            else:
                print(f"⚠️ No data available for {display_name}")

    return results

def get_ticker_info(pair):
    """Get current ticker information for a pair."""
    try:
        url = f"{API_CONFIG['url']}/{API_CONFIG['version']}/public/Ticker?pair={pair}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if 'error' in data and data['error']:
            print(f"Error getting ticker for {pair}: {data['error']}")
            return None
            
        return data['result'][pair]
    except Exception as e:
        print(f"Error getting ticker info: {str(e)}")
        return None

def execute_order(pair, euro_amount, order_type, skip_confirm=False):
    """Execute a trade order."""
    try:
        # Check if we're in test mode
        if TRADING_CONFIG['behavior']['test_run']:
            print(f"TEST RUN: Would {order_type} {euro_amount:.2f}€ of {pair}")
            return True
            
        # Get current price to calculate volume
        ticker = get_ticker_info(pair)
        if ticker is None:
            print(f"Could not get current price for {pair}")
            return False
            
        current_price = float(ticker['c'][0])  # Current price
        volume = euro_amount / current_price  # Convert euros to crypto volume
        
        # Prepare the order data
        data = {
            "nonce": str(int(1000*time.time())),
            "ordertype": "market",
            "type": order_type.lower(),
            "volume": f"{volume:.8f}",
            "pair": pair,
            "trading_agreement": "agree"
        }
        
        # Get user confirmation if needed
        if not skip_confirm and TRADING_CONFIG['behavior']['confirm_order']:
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
        print(f"✅ Order executed successfully. ID: {order_id}")
        return True
        
    except Exception as e:
        print(f"Error executing order: {str(e)}")
        return False

def get_account_balance():
    """Get the current account balance."""
    try:
        # If in test mode, return the configured balance
        if TRADING_CONFIG['behavior']['test_run']:
            return TRADING_CONFIG['risk_management']['total_balance']
            
        # Get private data about account balance
        url_path = "/0/private/Balance"
        nonce = str(int(time.time() * 1000))
        
        data = {
            "nonce": nonce,
        }
        
        response = kraken_request(url_path, data, API_KEY, API_SECRET)
        result = response.json()
        
        if 'error' in result and result['error']:
            print(f"API Error: {result['error']}")
            return 0.0
            
        # Sum up all EUR balances
        total_balance = 0.0
        for currency, amount in result.get('result', {}).items():
            if currency.endswith('EUR'):
                total_balance += float(amount)
        
        return total_balance
        
    except Exception as e:
        print(f"Error getting account balance: {str(e)}")
        return 0.0

def get_open_positions():
    """Get currently open positions."""
    try:
        # If in test mode, return empty list
        if TRADING_CONFIG['behavior']['test_run']:
            return []
            
        url_path = "/0/private/OpenPositions"
        nonce = str(int(time.time() * 1000))
        
        data = {
            "nonce": nonce,
        }
        
        response = kraken_request(url_path, data, API_KEY, API_SECRET)
        result = response.json()
        
        if 'error' in result and result['error']:
            print(f"API Error: {result['error']}")
            return []
            
        positions = []
        for pos_id, pos_data in result.get('result', {}).items():
            positions.append({
                'pair': pos_data['pair'],
                'type': pos_data['type'],
                'volume': float(pos_data['vol']),
                'price': float(pos_data['cost']),
                'pnl': float(pos_data.get('net', 0))
            })
        
        return positions
        
    except Exception as e:
        print(f"Error getting open positions: {str(e)}")
        return [] 