# XRP/EUR Trading Bot

A Streamlit-based trading bot that monitors XRP/EUR prices and suggests trades based on machine learning predictions.

## Features

- Real-time price monitoring
- Interactive candlestick chart
- ML-based price prediction
- Automated trading with manual confirmation
- Configurable trading parameters
- Secure API integration with Kraken

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your Kraken API keys:
   - Create a Kraken account if you don't have one
   - Generate API keys with trading permissions
   - Add your API keys to `.env`:
```
KRAKEN_API_KEY=your_api_key
KRAKEN_API_SECRET=your_api_secret
```

3. Download historical data and train the model:
```bash
python data_downloader.py  # Download historical data
python train_model.py      # Train and save the model
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage

1. The app will display the current XRP/EUR price and a candlestick chart of the last 24 hours.

2. Adjust trading parameters in the sidebar:
   - Minimum confidence: Required confidence level for trade signals
   - Position size: Amount in EUR to invest per trade

3. The model will continuously predict the maximum price in the next 12 hours.

4. When a profitable trade is detected:
   - A BUY signal will appear
   - Review the predicted return
   - Click "Execute Trade" to confirm

5. The app will automatically refresh every minute to update prices and predictions.

## Files

- `app.py`: Streamlit web interface
- `train_model.py`: Model training script
- `data_downloader.py`: Historical data downloader
- `main.py`: Trading functions and API integration
- `model.joblib`: Trained model file
- `.env`: API configuration

## Security

- Never share your API keys
- Start with small position sizes
- Monitor your trades regularly
- Use stop-loss orders for risk management

## Disclaimer

This bot is for educational purposes only. Cryptocurrency trading involves significant risk. Always do your own research and never trade more than you can afford to lose.

## Future Development

The following features are planned for future releases:

1. **Multi-Currency Support**
   - Simultaneous trading of multiple cryptocurrency pairs
   - Portfolio optimization and risk management across currencies
   - Correlation analysis between different crypto assets

2. **LLM-Based Sentiment Analysis**
   - Integration of large language models for crypto news analysis
   - Real-time sentiment scoring of market news
   - Social media trend analysis and impact assessment

3. **Advanced Trading Strategies**
   - Implementation of both bullish and bearish trading strategies
   - Short-selling capabilities
   - Market regime detection and strategy adaptation
   - Analysis at different intervals (10 minutes, 1 hour, 4 hours, 1 day)

4. **Enhanced User Interface**
   - Detailed performance analytics and statistics
   - Advanced charting and technical analysis tools
   - Real-time market insights and trade recommendations
   - Portfolio performance tracking

5. **Advanced Feature Engineering**
   - Integration of on-chain metrics
   - Advanced technical indicators
   - Market microstructure features
   - Cross-market signals and correlations 

6. **Improved and more realistic Backtesting**
   - Historical backtesting of different trading strategies
   - Performance metrics and comparison with other strategies
   - Imporved parameter tuning and optimization
