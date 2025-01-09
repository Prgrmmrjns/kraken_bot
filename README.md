# Kraken Trading Bot 🚀

A sophisticated cryptocurrency trading bot for the Kraken exchange, featuring multiple ML models, automated trading, and comprehensive backtesting.

## Features 🚀

- Multiple trading models:
  - LightGBM (gradient boosting)
  - MACD (technical analysis)
  - Deep Neural Network
  - Custom model support
- Automatic model selection and optimization
- Real-time trading with risk management
- Comprehensive backtesting
- Historical data storage and management
- Performance tracking and reporting

## Configuration ⚙️

### Model Selection

The bot supports multiple models and automatic model selection. Configure in `trading_strat_params.py`:

```python
MODEL_CONFIG = {
    'model_type': 'auto',  # Choose from: 'auto', 'lightgbm', 'macd', 'dnn'
    'fast_training': True,  # Set to False for hyperparameter optimization
    'optimization_trials': 100,  # Number of trials for optimization
}
```

When `model_type` is set to 'auto', the bot will:
1. Test all available models
2. Compare their performance (RMSE and R²)
3. Select the best performing model
4. Optionally optimize hyperparameters if `fast_training` is False

### Custom Models

You can create your own model by:
1. Inheriting from the `BaseModel` class in `model.py`
2. Implementing required methods: `train()`, `predict()`, `optimize()`
3. Adding your model to the `get_model()` function

Example:
```python
class MyCustomModel(BaseModel):
    def __init__(self, params=None):
        super().__init__('MyModel')
        self.params = params or {}
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        # Implement training logic
        pass
        
    def predict(self, X):
        # Implement prediction logic
        return predictions
        
    def optimize(self, trial, X_train, y_train, X_val, y_val):
        # Implement optimization logic
        return rmse
```

## Usage 🛠️

1. Set up your environment:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure your trading parameters in `trading_strat_params.py`

3. Run the bot:
   ```bash
   python app.py
   ```

4. For backtesting:
   ```bash
   python backtesting.py
   ```

## Data Storage 📊

The bot now stores historical OHLC data in parquet files:
- Data is automatically saved in the `data` directory
- Historical data is merged with new data on each run
- Efficient storage and fast loading with parquet format

## Performance Tracking 📈

The bot tracks and saves:
- Trading history
- Model performance metrics
- Balance and position updates
- Risk metrics

Results are saved in:
- `trading_summary_YYYYMMDD_HHMMSS.txt` for trading sessions
- Model performance comparisons during backtesting

## Safety Features 🛡️

- Automatic position management
- Risk limits and stop-losses
- Unsafe pair tracking
- API error handling
- Comprehensive logging

## Requirements 📋

- Python 3.8+
- Dependencies in `requirements.txt`
- Kraken API credentials
- Sufficient balance for trading

## License 📄

MIT License - See LICENSE file for details

