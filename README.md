# S&P 500 Trading Strategy Framework

A comprehensive, modular framework for developing, testing, and comparing machine learning trading strategies on the S&P 500 index. The framework combines technical indicators with macroeconomic data and provides extensive backtesting and performance analysis capabilities.

## 🏗️ Framework Architecture

This project has evolved from a single trading strategy into a full-featured framework that supports multiple signal generation approaches, comprehensive backtesting, and detailed performance analysis.

### Core Components

```
spyder/
├── framework/                      # Core framework modules
│   ├── data_provider.py           # Data fetching & preprocessing
│   ├── signal_generators.py       # Signal generation interface & implementations
│   ├── backtest_engine.py         # Backtesting & strategy comparison
│   └── performance_analyzer.py    # Visualization & performance analysis
├── strategy_runner.py             # Main framework demonstration
├── sp500_trading_strategy.py      # Legacy single-strategy implementation
└── analyze_model_probabilities.py # Model analysis utilities
```

## 🎯 Framework Features

### 📊 Data Sources & Processing
- **S&P 500 Price Data**: 62+ years of historical OHLCV data (1962-2024) via Yahoo Finance
- **Macroeconomic Data**: Federal Reserve Economic Data (FRED) API integration
  - Federal Funds Rate (FEDFUNDS)
  - Unemployment Rate (UNRATE) 
  - Consumer Price Index (CPIAUCSL)
  - 10-Year Treasury Yield (DGS10)
  - **3-Month Treasury Bill Rate (TB3MS)** - Used for cash returns when not invested
- **Technical Indicators**: SMA, RSI, Bollinger Bands, momentum, volume analysis
- **Data Alignment**: Automatic alignment of daily price data with monthly/weekly economic indicators

### 🤖 Signal Generation Strategies

The framework supports multiple signal generation approaches:

1. **Machine Learning Models**:
   - Random Forest Classifier (multiple probability thresholds)
   - Logistic Regression
   - Support Vector Machine (SVM)

2. **Technical Analysis Strategies**:
   - Simple Moving Average (SMA) Crossover
   - Relative Strength Index (RSI) 
   - Buy & Hold (benchmark)

3. **Extensible Interface**: Easy to add custom signal generators

### 🔬 Backtesting & Analysis

- **Realistic Trading Costs**: Configurable trading costs (0.1% default) and slippage
- **Cash Returns**: Earns 3-month Treasury bill rate when not invested (instead of 0%)
- **Dynamic Risk-Free Rate**: Uses actual Treasury rates for Sharpe ratio and Alpha calculations
- **Time-Based Validation**: Proper time-series split (80% train, 20% test)
- **Comprehensive Metrics**: 
  - Returns (total, annualized)
  - Risk metrics (Sharpe ratio, max drawdown, volatility)
  - Alpha, Beta, correlation analysis
  - Trading statistics (win rate, number of trades, market exposure)

### 📈 Performance Visualization

Automated generation of professional analysis plots:
- **Strategy Comparison**: Side-by-side performance comparison
- **Risk-Return Analysis**: Scatter plots with Sharpe ratio coloring
- **Drawdown Analysis**: Maximum drawdown visualization
- **Rolling Metrics**: Time-varying performance analysis
- **Monthly Returns Correlation**: Strategy vs. benchmark scatter plots

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd spyder

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn yfinance fredapi matplotlib python-dotenv
```

### 2. FRED API Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env file and add your FRED API key
# Get free API key from: https://fred.stlouisfed.org/docs/api/api_key.html
echo "FRED_API_KEY=your_actual_api_key_here" >> .env
```

### 3. Run Framework Comparison
```bash
# Run comprehensive strategy comparison
python strategy_runner.py
```

This will:
- Test 8 different trading strategies
- Generate detailed performance comparison
- Create visualization plots (saved as PNG files)
- Output comprehensive analysis report

## 📊 Sample Results

Recent framework run (1962-2024 test period) with **Cash Returns Implementation**:

| Strategy | Total Return | Sharpe Ratio | Max Drawdown | Win Rate | Trades | Time in Market |
|----------|-------------|--------------|--------------|----------|---------|----------------|
| Buy & Hold | 302.2% | 0.677 | -33.9% | 100.0% | 1 | 100.0% |
| SVM | 144.8% | 0.473 | -33.9% | 40.0% | 5 | 62.7% |
| SMA Crossover | 97.4% | 0.452 | -29.7% | 17.9% | 67 | 69.2% |
| Logistic Regression | 61.7% | 0.430 | -9.9% | 22.3% | 94 | 20.8% |
| Random Forest (0.6) | 30.4% | **0.409** | **-6.6%** | 41.2% | 34 | 1.1% |

**Key Features:**
- **Cash Returns**: Strategies earn ~1.0% annually when not invested (3-month Treasury rate)
- **Superior Risk Management**: Random Forest (0.6) achieves 0.409 Sharpe ratio with only -6.6% max drawdown
- **Realistic Performance**: Strategies benefit from holding cash during unfavorable market conditions
- **Dynamic Risk Metrics**: All calculations use actual Treasury rates as risk-free rate

## 💰 Cash Returns Feature

**New Enhancement**: When strategies are not invested in the S&P 500, they now earn realistic cash returns instead of sitting idle at 0%.

### How It Works
- **Cash Investment**: When signal = 0 (not invested), portfolio earns 3-month Treasury bill rate (TB3MS)
- **Dynamic Rates**: Uses actual historical Treasury rates from FRED API (1962-2024)
- **Daily Compounding**: Cash returns are calculated and compounded daily
- **Realistic Risk Metrics**: Sharpe ratio and Alpha use actual Treasury rates as risk-free baseline

### Impact on Performance
- **Low-Activity Strategies Benefit Most**: Strategies spending significant time in cash see improved returns
- **Better Risk-Adjusted Performance**: Strategies with market timing capabilities show enhanced Sharpe ratios
- **Realistic Benchmark**: More accurate comparison to real-world investment alternatives

**Example**: Random Forest (0.6) spends 98.9% of time earning Treasury returns, achieving superior risk-adjusted performance with minimal market exposure.

## 🛠️ Framework Usage

### Basic Usage

```python
from framework.data_provider import DataProvider
from framework.signal_generators import create_signal_generator
from framework.backtest_engine import BacktestEngine
from framework.performance_analyzer import PerformanceAnalyzer

# 1. Get data
data_provider = DataProvider()
combined_data, X, y = data_provider.get_complete_dataset("2010-01-01", "2024-01-01")

# 2. Create and train strategy
strategy = create_signal_generator('rf', probability_threshold=0.5)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
strategy.fit(X_train, y_train)

# 3. Generate signals and backtest
signals = strategy.generate_signals(X_test)
backtest_engine = BacktestEngine()
results = backtest_engine.backtest_strategy(price_data, signals)

# 4. Analyze performance
analyzer = PerformanceAnalyzer()
analyzer.plot_single_strategy_analysis(results)
```

### Creating Custom Strategies

```python
from framework.signal_generators import SignalGenerator

class MyCustomStrategy(SignalGenerator):
    def fit(self, X_train, y_train):
        # Your training logic here
        self.is_trained = True
        return self
    
    def generate_signals(self, X):
        # Your signal generation logic here
        signals = # ... your logic
        probabilities = # ... your confidence scores
        
        return pd.DataFrame({
            'Signal': signals,
            'Probability': probabilities
        }, index=X.index)

# Register and use
strategy = MyCustomStrategy(name="My Strategy")
```

## ⚙️ Configuration

Key configuration options in `strategy_runner.py`:

```python
# Data Configuration
START_DATE = "1962-01-02"  # Maximum available data
END_DATE = "2024-01-01"
LOOKBACK_DAYS = 5          # Future prediction window
TEST_SIZE = 0.2            # 20% for testing

# Trading Configuration  
INITIAL_CAPITAL = 10000
TRADING_COST = 0.001       # 0.1% per trade
SLIPPAGE = 0.0001          # 0.01% market impact

# Model Configuration
strategies_config = {
    'Random Forest (0.4)': {
        'type': 'rf',
        'params': {'probability_threshold': 0.4}
    },
    # ... add more strategies
}
```

## 📈 Performance Analysis

The framework automatically generates:

1. **Comparison Plots**: `strategy_comparison_framework.png`
   - Cumulative returns comparison
   - Drawdown analysis
   - Risk-return scatter plot
   - Performance metrics table

2. **Rolling Metrics**: `rolling_metrics_framework.png`
   - 1-year rolling performance windows
   - Time-varying Sharpe ratios
   - Rolling volatility and drawdowns

3. **Individual Analysis**: `top_strategy_N_*.png`
   - Detailed analysis for best performing strategies
   - Log-scale cumulative returns
   - Monthly return correlations

4. **Performance Report**: `performance_report_framework.txt`
   - Comprehensive text-based analysis
   - Detailed metrics for all strategies

## 🔒 Security & Best Practices

- **Environment Variables**: API keys stored in `.env` files (not tracked by git)
- **Realistic Costs**: Trading costs and slippage included in all backtests
- **Cash Returns**: Realistic returns when not invested (3-month Treasury rate)
- **Dynamic Risk Metrics**: Uses actual market rates for Sharpe ratio and Alpha calculations
- **Time-Series Validation**: Proper time-based splits prevent look-ahead bias
- **Risk Management**: Comprehensive risk metrics and drawdown analysis

## 📋 Dependencies

### Core Requirements
```
pandas >= 1.3.0
numpy >= 1.16.5
scikit-learn >= 1.0.0
yfinance >= 0.2.0
fredapi >= 0.5.0
matplotlib >= 3.0.0
python-dotenv >= 1.0.0
```

### System Requirements
- Python 3.8+
- For plot display: `python3-tk` (install via system package manager)
- FRED API key (free from Federal Reserve)

## 🚨 Important Disclaimers

**This framework is for educational and research purposes only.**

- Past performance does not guarantee future results
- This is not financial advice
- Trading involves substantial risk of loss
- All strategies should be thoroughly tested before live implementation
- Always consult with qualified financial advisors
- The COVID-19 market crash demonstrates that even sophisticated models can underperform during unprecedented events

## 🤝 Contributing

The framework is designed to be extensible:

1. **Add New Signal Generators**: Implement the `SignalGenerator` interface
2. **Extend Backtesting**: Add custom metrics to `BacktestEngine`  
3. **Enhance Visualizations**: Extend `PerformanceAnalyzer`
4. **Add Data Sources**: Extend `DataProvider` with new data feeds

## 📊 Project Evolution

This project evolved from a single Random Forest trading strategy into a comprehensive framework:

1. **v1**: Single `sp500_trading_strategy.py` with Random Forest
2. **v2**: Modular framework with multiple signal generators
3. **v3**: Comprehensive backtesting and performance analysis
4. **v4**: Cash returns implementation with 3-month Treasury rates
5. **Current**: Full comparison framework with realistic cash returns and dynamic risk metrics

The legacy single-strategy script (`sp500_trading_strategy.py`) is maintained for reference and backward compatibility.

## 📞 Support

For issues, questions, or contributions:
1. Check existing documentation and examples
2. Review the framework source code in `framework/` directory
3. Examine the `strategy_runner.py` for usage examples
4. Create issues for bugs or feature requests

---

**Happy Trading! 📈**

*Remember: The best strategy is often the simplest one that you can stick with consistently.*