import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import subprocess
import joblib
from model_functions import create_features, simulate_trading
from kraken_functions import place_market_order, download_full_ohlc_data

# Example trading pairs
AVAILABLE_PAIRS = {
    'XRP/EUR': 'XXRPZEUR',
    'BTC/EUR': 'XXBTZEUR',
    'ETH/EUR': 'XETHZEUR',
    'DOT/EUR': 'DOTZEUR',
    'ADA/EUR': 'ADAEUR'
}

# Constants
INTERVAL = 60  # minutes
INITIAL_BALANCE = 100.0  # EUR

def run_init_script(pair):
    """Run init.py to retrain the model with latest data"""
    try:
        # Set environment variable for the pair
        import os
        os.environ['TRADING_PAIR'] = pair
        subprocess.run(['python', 'init.py'], check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def load_model():
    """Load the trained model"""
    try:
        return joblib.load('model.joblib')
    except:
        st.error("Model not found. Please ensure init.py has been run successfully.")
        return None

def evaluate_position(model, current_data):
    """Evaluate whether to open/close a position"""
    if model is None:
        return None, None
    
    # Create features for prediction
    df = create_features(current_data)
    df = df.dropna(subset=['target'])
    X = df.drop(columns=['timestamp', 'target'])
    
    # Make prediction
    predicted_price = model.predict(X)[-1]
    current_price = df['close'].iloc[-1]
    
    # Calculate expected return
    expected_return = (predicted_price - current_price) / current_price * 100
    
    return predicted_price, expected_return

def handle_order(order_type, pair, volume, current_price):
    """Handle order placement and update session state"""
    placeholder = st.empty()
    with placeholder.container():
        with st.spinner(f"Placing {order_type} order..."):
            response = place_market_order(pair, volume, order_type.lower())
            
            if response and 'error' not in response:
                if order_type == "BUY":
                    st.session_state.position_open = True
                    st.session_state.entry_price = current_price
                    # Record trade
                    st.session_state.trades.append({
                        'type': 'BUY',
                        'timestamp': datetime.now(),
                        'price': current_price,
                        'size': st.session_state.parameters['position_size']
                    })
                    st.success(f"✅ {order_type} order executed successfully!")
                else:  # SELL
                    st.session_state.position_open = False
                    current_return = (current_price - st.session_state.entry_price) / st.session_state.entry_price * 100
                    # Record trade
                    st.session_state.trades.append({
                        'type': 'SELL',
                        'timestamp': datetime.now(),
                        'price': current_price,
                        'profit': current_return
                    })
                    st.session_state.entry_price = None
                    st.success(f"✅ {order_type} order executed successfully!")
                
                # Show order details
                st.json(response)
                time.sleep(3)  # Give user time to see the success message
            else:
                error_msg = response.get('error', ['Unknown error'])[0] if response else 'Failed to connect to exchange'
                st.error(f"❌ Failed to execute {order_type} order: {error_msg}")
                time.sleep(3)  # Give user time to see the error
            
            placeholder.empty()  # Clear the messages after delay

def initialize_session_state():
    """Initialize or reset session state variables"""
    if 'trades' not in st.session_state:
        st.session_state.trades = []
    if 'position_open' not in st.session_state:
        st.session_state.position_open = False
    if 'last_eval_hour' not in st.session_state:
        st.session_state.last_eval_hour = None
    if 'entry_price' not in st.session_state:
        st.session_state.entry_price = None
    if 'last_model_update' not in st.session_state:
        st.session_state.last_model_update = None
    if 'selected_pair' not in st.session_state:
        st.session_state.selected_pair = None
    if 'parameters' not in st.session_state:
        st.session_state.parameters = {
            'min_confidence': 2.0,
            'position_size': 25.0,
            'take_profit': 2.0
        }
    if 'order_pending' not in st.session_state:
        st.session_state.order_pending = False

def main():
    st.title("Cryptocurrency Trading Bot")
    
    # Add explanation about the bot
    with st.expander("How it works", expanded=False):
        st.markdown("""
        This trading bot uses machine learning to predict cryptocurrency price movements and suggest trading opportunities.
        It analyzes historical price data and various technical indicators to make predictions about future price movements.
        
        ### How it works:
        1. The model predicts the maximum price in the next 12 hours
        2. If the predicted return exceeds your minimum confidence threshold, it suggests a buy
        3. For open positions, it monitors for profitable selling opportunities
        
        ### Important Notes:
        - The confidence threshold represents the minimum expected return for a trade
        - Higher confidence thresholds mean fewer but potentially more reliable trades
        - Lower thresholds mean more frequent trading but potentially higher risk
        - All trades require manual confirmation for safety
        """)
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar configuration
    st.sidebar.header("Trading Parameters")
    
    # Currency pair selection
    selected_display_pair = st.sidebar.selectbox(
        "Select Trading Pair",
        list(AVAILABLE_PAIRS.keys()),
        help="Choose the cryptocurrency pair you want to trade"
    )
    
    # Convert display pair to API pair format
    selected_pair = AVAILABLE_PAIRS[selected_display_pair]
    
    # Parameter inputs
    new_min_confidence = st.sidebar.slider(
        "Minimum Confidence (%)", 
        1.0, 10.0, st.session_state.parameters['min_confidence'],
        help="Minimum expected return required for a buy signal. Higher values mean fewer but potentially more reliable trades."
    )
    
    new_position_size = st.sidebar.number_input(
        "Position Size (EUR)", 
        10.0, 1000.0, st.session_state.parameters['position_size'],
        help="Amount in EUR to invest per trade"
    )
    
    new_take_profit = st.sidebar.slider(
        "Take Profit (%)", 
        1.0, 20.0, st.session_state.parameters['take_profit'],
        help="Percentage gain at which to suggest taking profits"
    )
    
    # Check if parameters changed
    params_changed = (
        new_min_confidence != st.session_state.parameters['min_confidence'] or
        new_position_size != st.session_state.parameters['position_size'] or
        new_take_profit != st.session_state.parameters['take_profit']
    )
    
    # Apply Parameters button
    if params_changed:
        st.sidebar.warning("Parameters have changed. Click 'Apply Parameters' to update.")
        if st.sidebar.button("Apply Parameters"):
            with st.spinner("Updating model with new parameters..."):
                if run_init_script(selected_pair):
                    st.session_state.parameters = {
                        'min_confidence': new_min_confidence,
                        'position_size': new_position_size,
                        'take_profit': new_take_profit
                    }
                    st.session_state.last_model_update = datetime.now().date()
                    st.success("Model updated successfully with new parameters!")
                else:
                    st.error("Failed to update model")
    
    # Only show the start button if pair changed or not started
    if selected_pair != st.session_state.selected_pair:
        if st.sidebar.button("Start Trading Bot"):
            with st.spinner(f"Initializing model for {selected_display_pair}..."):
                if run_init_script(selected_pair):
                    st.session_state.selected_pair = selected_pair
                    st.session_state.last_model_update = datetime.now().date()
                    st.success("Model initialized successfully!")
                else:
                    st.error("Failed to initialize model")
    
    # Display trade history in sidebar
    if st.session_state.trades:
        st.sidebar.markdown("---")
        st.sidebar.header("Trade History")
        for trade in st.session_state.trades:
            st.sidebar.markdown(f"""
            **{trade['type']}**: {trade['timestamp'].strftime('%Y-%m-%d %H:%M')}
            Price: €{trade['price']:.5f}
            {'Profit: ' + f"{trade['profit']:.2f}%" if 'profit' in trade else ''}
            """)
    
    # Only proceed if model is initialized
    if st.session_state.selected_pair is not None:
        # Check if we need to run init.py (daily)
        current_date = datetime.now().date()
        if (st.session_state.last_model_update is None or 
            st.session_state.last_model_update != current_date):
            with st.spinner("Updating model with latest data..."):
                if run_init_script(selected_pair):
                    st.session_state.last_model_update = current_date
                    st.success("Model updated successfully!")
                else:
                    st.error("Failed to update model")
        
        # Load the model
        model = load_model()
        
        # Get current hour
        current_hour = datetime.now().hour
        
        # Download latest data
        with st.spinner("Downloading latest market data..."):
            current_data = download_full_ohlc_data(selected_pair, INTERVAL)
            current_price = float(current_data['close'].iloc[-1])
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        # Display current price and 24h change
        price_24h_ago = float(current_data['close'].iloc[-25])  # 24 periods ago (1 hour intervals)
        price_change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
        
        col1.metric(
            f"Current {selected_display_pair} Price", 
            f"€{current_price:.5f}",
            f"{price_change_24h:+.2f}%"
        )
        
        # Display 24h volume
        volume_24h = current_data['volume'].tail(24).sum()
        col2.metric("24h Volume", f"€{volume_24h:,.2f}")
        
        # Display 24h high/low
        high_24h = current_data['high'].tail(24).max()
        low_24h = current_data['low'].tail(24).min()
        col3.metric("24h High/Low", f"€{high_24h:.5f} / €{low_24h:.5f}")
        
        # Evaluate position if it's a new hour or initial evaluation
        if (st.session_state.last_eval_hour != current_hour or 
            st.session_state.last_eval_hour is None):
            
            predicted_price, expected_return = evaluate_position(model, current_data)
            st.session_state.last_eval_hour = current_hour
            
            if predicted_price is not None:
                # Create prediction metrics
                st.markdown("### Price Prediction")
                pred_col1, pred_col2 = st.columns(2)
                
                pred_col1.metric(
                    "Predicted Maximum Price (12h)", 
                    f"€{predicted_price:.5f}",
                    f"{expected_return:+.2f}%"
                )
                
                pred_col2.metric(
                    "Confidence Score",
                    f"{abs(expected_return):.2f}%",
                    help="Higher confidence scores indicate stronger trading signals"
                )
                
                # Trading logic
                if not st.session_state.position_open:
                    if expected_return > st.session_state.parameters['min_confidence']:
                        st.success("💰 Buy Signal Detected!")
                        st.markdown(f"""
                        **Signal Analysis:**
                        - Expected return exceeds minimum confidence threshold ({st.session_state.parameters['min_confidence']:.1f}%)
                        - Predicted price movement: +{expected_return:.2f}%
                        - Position size: €{st.session_state.parameters['position_size']:.2f}
                        - Estimated profit (before fees): €{st.session_state.parameters['position_size'] * expected_return / 100:.2f}
                        """)
                        
                        if st.button("Confirm Buy"):
                            handle_order(
                                "BUY",
                                selected_pair,
                                st.session_state.parameters['position_size']/current_price,
                                current_price
                            )
                    else:
                        st.info("No buy signal at this time")
                        st.markdown(f"""
                        **Current Analysis:**
                        - Expected return ({expected_return:.2f}%) below minimum confidence threshold ({st.session_state.parameters['min_confidence']:.1f}%)
                        - Monitoring market for better opportunities
                        """)
                else:
                    # Check if we should sell
                    current_return = (current_price - st.session_state.entry_price) / st.session_state.entry_price * 100
                    
                    # Check both take profit and prediction-based sell signals
                    take_profit_triggered = current_return >= st.session_state.parameters['take_profit']
                    prediction_sell_signal = expected_return < 0
                    
                    if take_profit_triggered or prediction_sell_signal:
                        st.warning("📈 Sell Signal Detected!")
                        reason = "Take profit target reached" if take_profit_triggered else "Negative price movement predicted"
                        st.markdown(f"""
                        **Signal Analysis:**
                        - Reason: {reason}
                        - Current return: {current_return:.2f}%
                        - Predicted price movement: {expected_return:.2f}%
                        - Current profit (before fees): €{st.session_state.parameters['position_size'] * current_return / 100:.2f}
                        """)
                        
                        if st.button("Confirm Sell"):
                            handle_order(
                                "SELL",
                                selected_pair,
                                st.session_state.parameters['position_size']/st.session_state.entry_price,
                                current_price
                            )
                    else:
                        st.info("Holding position...")
                        st.markdown(f"""
                        **Position Analysis:**
                        - Current return: {current_return:.2f}%
                        - Take profit target: {st.session_state.parameters['take_profit']:.2f}%
                        - Predicted price movement: {expected_return:.2f}%
                        - Holding for better exit opportunity
                        """)
        
        # Display position status
        if st.session_state.position_open:
            st.sidebar.markdown("---")
            st.sidebar.header("Current Position")
            current_return = (current_price - st.session_state.entry_price) / st.session_state.entry_price * 100
            
            st.sidebar.metric("Entry Price", f"€{st.session_state.entry_price:.5f}")
            st.sidebar.metric("Current Return", f"{current_return:.2f}%")
            st.sidebar.metric("Position Value", 
                            f"€{st.session_state.parameters['position_size'] * (1 + current_return/100):.2f}")
        
        # Auto-refresh every minute if no order is pending
        if not st.session_state.order_pending:
            time.sleep(60)
            st.rerun()
    else:
        st.warning("Please select a trading pair and click 'Start Trading Bot' to begin.")

if __name__ == "__main__":
    main() 