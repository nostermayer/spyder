# S&P 500 Machine Learning Trading Strategy

A comprehensive machine learning trading strategy for the S&P 500 index that combines technical indicators with macroeconomic data to generate buy/sell signals.

## 🎯 Overview

This project implements a complete end-to-end trading strategy that:
- Fetches real-time S&P 500 price data using `yfinance`
- Incorporates Federal Reserve economic data via the FRED API
- Engineers technical indicators and features
- Trains a Random Forest classifier to predict future price movements
- Generates trading signals with configurable probability thresholds
- Backtests the strategy with comprehensive performance metrics

## 📊 Features

### Data Sources
- **S&P 500 Price Data**: Historical OHLCV data from Yahoo Finance
- **Macroeconomic Indicators**:
  - Federal Funds Rate (FEDFUNDS)
  - Unemployment Rate (UNRATE)
  - Consumer Price Index (CPIAUCSL)
  - 10-Year Treasury Yield (DGS10)

### Technical Indicators
- Simple Moving Averages (20-day, 50-day)
- Relative Strength Index (RSI)
- Bollinger Bands (width and position)
- Price momentum indicators (1-day, 5-day, 20-day)
- Volume analysis

### Machine Learning
- **Algorithm**: Random Forest Classifier
- **Target**: Binary classification (1=Buy, 0=Sell/Hold)
- **Prediction Window**: 5-day future price movement
- **Features**: 13 technical and macroeconomic indicators
- **Validation**: Time-based split (80% train, 20% test)

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd spyder
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pandas numpy scikit-learn yfinance fredapi matplotlib jupyter python-dotenv
```

### 4. Set Up Environment Variables
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your FRED API key
nano .env
```

Add your FRED API key to the `.env` file:
```
FRED_API_KEY=your_actual_fred_api_key_here
```

### 5. Get a FRED API Key
1. Visit [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Create a free account
3. Request an API key
4. Add it to your `.env` file

## 💻 Usage

### Basic Usage
```bash
source venv/bin/activate
python sp500_trading_strategy.py
```

### Configuration
Edit the configuration variables in `sp500_trading_strategy.py`:
```python
START_DATE = "2010-01-01"           # Data start date
END_DATE = "2024-01-01"             # Data end date
LOOKBACK_DAYS = 5                   # Future prediction window
PROBABILITY_THRESHOLD = 0.6         # Buy signal threshold
```

### Output
The script will output:
- Data fetching progress
- Technical indicator creation
- Model training results
- Feature importance rankings
- Trading signal generation
- Backtesting performance metrics

## 📈 Performance Results

### Latest Backtest Results (2010-2024)
- **Strategy Return**: 10.2%
- **Buy & Hold Return**: 21.4%
- **Strategy Sharpe Ratio**: 0.613
- **Buy & Hold Sharpe Ratio**: 0.485
- **Maximum Drawdown**: -6.8%
- **Number of Trades**: 73
- **Win Rate**: 50.1% (test accuracy)

### Feature Importance
Top contributing features to trading decisions:
1. 50-day Simple Moving Average (11.7%)
2. Bollinger Band Width (10.9%)
3. 20-day Simple Moving Average (10.0%)
4. Volume Ratio (8.1%)
5. Bollinger Band Position (7.8%)

## 🔧 Project Structure

```
spyder/
├── sp500_trading_strategy.py    # Main trading strategy script
├── .env                         # Environment variables (not tracked)
├── .env.example                 # Environment template
├── .gitignore                   # Git ignore rules
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── venv/                        # Virtual environment
└── instructions.md              # Original project requirements
```

## 🛠 Key Functions

### Data Acquisition
- `fetch_sp500_data()`: Downloads S&P 500 historical data
- `fetch_macro_data()`: Retrieves FRED economic indicators

### Feature Engineering
- `create_technical_indicators()`: Calculates technical analysis indicators
- `combine_data()`: Merges price and economic data
- `create_target_variable()`: Generates binary prediction targets

### Machine Learning
- `prepare_features_and_target()`: Prepares ML-ready datasets
- `train_model()`: Trains Random Forest classifier
- `generate_signals()`: Converts probabilities to trading signals

### Backtesting
- `backtest_strategy()`: Evaluates strategy performance

## ⚙ Configuration Options

### Model Parameters
```python
model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Maximum tree depth
    min_samples_split=10,  # Minimum samples to split
    min_samples_leaf=5,    # Minimum samples per leaf
    random_state=42        # Reproducibility
)
```

### Signal Generation
- **Probability Threshold**: Adjust `PROBABILITY_THRESHOLD` (0.0-1.0)
- **Prediction Window**: Modify `LOOKBACK_DAYS` for different time horizons
- **Data Range**: Change `START_DATE` and `END_DATE` for different periods

## 📋 Requirements

### Python Version
- Python 3.8+

### Dependencies
- pandas >= 1.3.0
- numpy >= 1.16.5
- scikit-learn >= 1.0.0
- yfinance >= 0.2.0
- fredapi >= 0.5.0
- python-dotenv >= 1.0.0

## 🔒 Security

- Environment variables are used for API keys
- `.env` file is excluded from version control
- API keys are never stored in code

## 🚨 Disclaimer

**This project is for educational and research purposes only.**

- Past performance does not guarantee future results
- This is not financial advice
- Trading involves substantial risk of loss
- Always consult with a qualified financial advisor
- Test thoroughly before using real money

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📞 Support

For questions or issues:
1. Check the existing issues
2. Create a new issue with detailed information
3. Include error messages and environment details

## 🔗 Useful Links

- [FRED API Documentation](https://fred.stlouisfed.org/docs/api/)
- [Yahoo Finance API](https://pypi.org/project/yfinance/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

---

**Happy Trading! 📈**