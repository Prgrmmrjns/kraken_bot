# Kraken Trading Bot 🚀

A command-line cryptocurrency trading bot using machine learning to predict price movements and execute trades on the Kraken exchange.

## Features 🌟

- ML-based price prediction using LightGBM
- Real-time market monitoring
- Automated trading with risk management
- Backtesting with visualizations
- Simple command-line interface

## Prerequisites ⚙️

- Python 3.10+
- Kraken account with API access

## Installation 🛠️

1. Clone the repository:
```bash
git clone https://github.com/yourusername/kraken_bot.git
cd kraken_bot
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your Kraken API credentials
```

## Usage 🎮

1. Run backtesting to optimize strategy:
```bash
python backtesting.py
```

2. Start the trading bot:
```bash
python app.py
```

The bot will:
- Monitor market prices
- Identify trading opportunities
- Execute trades based on ML predictions
- Display real-time status in the terminal

To stop the bot, press Ctrl+C.

## Configuration ⚙️

1. Environment Variables (.env):
```
KRAKEN_API_KEY=your_api_key
KRAKEN_API_SECRET=your_api_secret
```

2. Trading Parameters (trading_strat_params.py):
- Trading pairs
- Risk management settings
- Model parameters

## Project Structure 📁

```
kraken_bot/
├── app.py              # Main trading bot
├── backtesting.py      # Backtesting engine
├── model_functions.py  # ML model functions
├── kraken_functions.py # Exchange API functions
├── models/            # Trained models
└── backtesting_results/ # Backtesting results
```

## Future Directions and To Dos 🚀

### 1. Enhanced Market Analysis
- Integrate LLM-based crypto news sentiment analysis
- Add social media trend analysis (Twitter, Reddit)
- Implement on-chain metrics analysis
- Include market correlation analysis

### 2. Model Improvements
- Reduce overfitting with better cross-validation
- Implement ensemble methods with multiple models
- Add deep learning models (LSTM, Transformers)
- Improve feature engineering with more technical indicators
- Add feature importance analysis and selection

### 3. Trading Strategies
- Implement short selling for bear markets
- Add grid trading capabilities
- Develop market-specific strategies (bull/bear/sideways)
- Implement dynamic position sizing
- Add portfolio rebalancing
- Improve take-profit and stop-loss strategies

### 4. User Interface
- Develop web interface for monitoring
- Add real-time charts and analytics
- Implement mobile notifications
- Create trade performance dashboard
- Add strategy customization interface

### 5. Risk Management
- Implement better drawdown protection
- Add portfolio-level risk management
- Develop dynamic risk adjustment based on market conditions
- Improve position exit strategies
- Add volatility-based position sizing

### 6. Infrastructure
- Add automated model retraining
- Implement high-availability setup
- Add comprehensive logging and monitoring
- Improve error handling and recovery
- Add automated backups

### 7. Fix and improve visualizations
- Fix buy and sell signals
- Add more visualizations of backtesting results and live trading
- Add visualizations of model performance and predictions

## Disclaimer ⚠️

Cryptocurrency trading involves significant risks:
- Start with small amounts
- Use proper risk management
- Monitor the bot regularly
- Never trade more than you can afford to lose
- Use at your own risk. The author is not responsible for any losses.

## License 📄

MIT License

