#!/usr/bin/env python3
"""
S&P 500 Machine Learning Trading Strategy

This script implements a complete machine learning trading strategy for the S&P 500 index,
incorporating both technical indicators and macroeconomic data for signal generation.

Required FRED API Key:
- Visit https://fred.stlouisfed.org/docs/api/api_key.html
- Create a free account and request an API key
- Add your API key to the .env file: FRED_API_KEY=your_actual_key_here

Author: Auto-generated trading strategy
"""

import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

# Configuration
FRED_API_KEY = os.getenv('FRED_API_KEY')
START_DATE = "2010-01-01"
END_DATE = "2024-01-01"
LOOKBACK_DAYS = 5  # Days to look ahead for target variable
PROBABILITY_THRESHOLD = 0.6  # Threshold for buy signal generation

def fetch_sp500_data(start_date, end_date):
    """
    Fetch S&P 500 historical price data using yfinance.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
    
    Returns:
        pd.DataFrame: DataFrame with S&P 500 price data
    """
    print("Fetching S&P 500 data...")
    
    # Fetch S&P 500 data using the ^GSPC ticker
    sp500 = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
    
    # Clean column names (remove multi-level indexing if present)
    sp500.columns = [col[0] if isinstance(col, tuple) else col for col in sp500.columns]
    
    # Ensure we have the required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in sp500.columns:
            raise ValueError(f"Missing required column: {col}")
    
    print(f"Successfully fetched {len(sp500)} days of S&P 500 data")
    return sp500

def fetch_macro_data(fred_api_key, start_date, end_date):
    """
    Fetch macroeconomic data from FRED (Federal Reserve Economic Data).
    
    Args:
        fred_api_key (str): FRED API key
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
    
    Returns:
        pd.DataFrame: DataFrame with macroeconomic indicators
    """
    print("Fetching macroeconomic data...")
    
    if not fred_api_key:
        print("WARNING: FRED API key not found!")
        print("Please add your API key to the .env file: FRED_API_KEY=your_actual_key_here")
        print("Visit https://fred.stlouisfed.org/docs/api/api_key.html to get one")
        # Return dummy data for demonstration
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        return pd.DataFrame({
            'FEDFUNDS': np.random.uniform(0, 5, len(date_range)),
            'UNRATE': np.random.uniform(3, 10, len(date_range)),
            'CPIAUCSL': np.random.uniform(200, 300, len(date_range)),
            'DGS10': np.random.uniform(1, 5, len(date_range))
        }, index=date_range)
    
    try:
        fred = Fred(api_key=fred_api_key)
        
        # Fetch macroeconomic indicators
        indicators = {
            'FEDFUNDS': 'Federal Funds Rate',
            'UNRATE': 'Unemployment Rate', 
            'CPIAUCSL': 'Consumer Price Index',
            'DGS10': '10-Year Treasury Yield'
        }
        
        macro_data = pd.DataFrame()
        
        for indicator, description in indicators.items():
            print(f"  Fetching {description}...")
            data = fred.get_series(indicator, start=start_date, end=end_date)
            macro_data[indicator] = data
        
        # Forward fill missing values (interpolate monthly/weekly data to daily)
        macro_data = macro_data.fillna(method='ffill')
        
        print(f"Successfully fetched macroeconomic data with {len(macro_data)} observations")
        return macro_data
    
    except Exception as e:
        print(f"Error fetching FRED data: {e}")
        print("Using dummy data for demonstration...")
        # Return dummy data if FRED fetch fails
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        return pd.DataFrame({
            'FEDFUNDS': np.random.uniform(0, 5, len(date_range)),
            'UNRATE': np.random.uniform(3, 10, len(date_range)),
            'CPIAUCSL': np.random.uniform(200, 300, len(date_range)),
            'DGS10': np.random.uniform(1, 5, len(date_range))
        }, index=date_range)

def create_technical_indicators(df):
    """
    Create technical indicators from price data.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
    
    Returns:
        pd.DataFrame: DataFrame with additional technical indicator columns
    """
    print("Creating technical indicators...")
    
    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (2 * bb_std)
    df['BB_Lower'] = df['BB_Middle'] - (2 * bb_std)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
    
    # Price momentum indicators
    df['Price_Change_1D'] = df['Close'].pct_change(1)
    df['Price_Change_5D'] = df['Close'].pct_change(5)
    df['Price_Change_20D'] = df['Close'].pct_change(20)
    
    # Volume indicators
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
    
    print("Technical indicators created successfully")
    return df

def combine_data(sp500_data, macro_data):
    """
    Combine S&P 500 price data with macroeconomic data, aligning by date.
    
    Args:
        sp500_data (pd.DataFrame): S&P 500 price data
        macro_data (pd.DataFrame): Macroeconomic data
    
    Returns:
        pd.DataFrame: Combined dataset aligned by date
    """
    print("Combining S&P 500 and macroeconomic data...")
    
    # Ensure both DataFrames have datetime index
    if not isinstance(sp500_data.index, pd.DatetimeIndex):
        sp500_data.index = pd.to_datetime(sp500_data.index)
    if not isinstance(macro_data.index, pd.DatetimeIndex):
        macro_data.index = pd.to_datetime(macro_data.index)
    
    # Reindex macro data to match S&P 500 trading days and forward fill
    macro_data_daily = macro_data.reindex(sp500_data.index, method='ffill')
    
    # Combine the datasets
    combined_data = pd.concat([sp500_data, macro_data_daily], axis=1)
    
    # Forward fill any remaining missing values
    combined_data = combined_data.fillna(method='ffill')
    
    print(f"Combined dataset created with {len(combined_data)} observations")
    return combined_data

def create_target_variable(df, lookback_days=5):
    """
    Create binary target variable based on future price movement.
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        lookback_days (int): Number of days to look ahead for target calculation
    
    Returns:
        pd.DataFrame: DataFrame with target variable added
    """
    print(f"Creating target variable with {lookback_days}-day lookback...")
    
    # Calculate future price change (avoiding look-ahead bias)
    df['Future_Close'] = df['Close'].shift(-lookback_days)
    df['Future_Return'] = (df['Future_Close'] - df['Close']) / df['Close']
    
    # Create binary target: 1 for positive future return, 0 for negative/zero
    df['Target'] = (df['Future_Return'] > 0).astype(int)
    
    # Remove rows where we don't have future data
    df = df.dropna(subset=['Future_Return'])
    
    # Drop the helper columns to prevent data leakage
    df = df.drop(['Future_Close', 'Future_Return'], axis=1)
    
    print(f"Target variable created. Positive signals: {df['Target'].sum()}/{len(df)} ({df['Target'].mean():.1%})")
    return df

def prepare_features_and_target(df):
    """
    Prepare feature matrix X and target vector y for machine learning.
    
    Args:
        df (pd.DataFrame): Combined dataset with all features and target
    
    Returns:
        tuple: (X, y) where X is features DataFrame and y is target Series
    """
    print("Preparing features and target variables...")
    
    # Define feature columns (exclude OHLCV and target)
    feature_columns = [
        # Technical indicators
        'SMA_20', 'SMA_50', 'RSI', 'BB_Width', 'BB_Position',
        'Price_Change_1D', 'Price_Change_5D', 'Price_Change_20D',
        'Volume_Ratio',
        # Macroeconomic indicators
        'FEDFUNDS', 'UNRATE', 'CPIAUCSL', 'DGS10'
    ]
    
    # Select features that exist in the dataframe
    available_features = [col for col in feature_columns if col in df.columns]
    
    if len(available_features) != len(feature_columns):
        missing_features = set(feature_columns) - set(available_features)
        print(f"Warning: Missing features: {missing_features}")
    
    # Create feature matrix X and target vector y
    X = df[available_features].copy()
    y = df['Target'].copy()
    
    # Remove rows with missing values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    print(f"Feature matrix prepared: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features used: {list(X.columns)}")
    
    return X, y

def train_model(X, y, test_size=0.2):
    """
    Train machine learning model with time-based split.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        test_size (float): Proportion of data to use for testing
    
    Returns:
        tuple: (trained_model, X_test, y_test) for evaluation
    """
    print("Training machine learning model...")
    
    # Time-based split (not random) - take last 20% for testing
    split_index = int(len(X) * (1 - test_size))
    
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model performance
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"Training accuracy: {train_accuracy:.3f}")
    print(f"Test accuracy: {test_accuracy:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 most important features:")
    print(feature_importance.head())
    
    return model, X_test, y_test

def generate_signals(model, X, threshold=0.6):
    """
    Generate buy/sell signals based on model predictions and threshold.
    
    Args:
        model: Trained machine learning model
        X (pd.DataFrame): Feature matrix
        threshold (float): Probability threshold for buy signal
    
    Returns:
        pd.DataFrame: DataFrame with signals and probabilities
    """
    print(f"Generating trading signals with threshold {threshold}...")
    
    # Get prediction probabilities
    probabilities = model.predict_proba(X)[:, 1]  # Probability of class 1 (buy)
    
    # Generate binary signals based on threshold
    signals = (probabilities >= threshold).astype(int)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Date': X.index,
        'Buy_Probability': probabilities,
        'Signal': signals
    })
    
    results.set_index('Date', inplace=True)
    
    buy_signals = signals.sum()
    total_signals = len(signals)
    
    print(f"Generated {buy_signals} buy signals out of {total_signals} total ({buy_signals/total_signals:.1%})")
    
    return results

def backtest_strategy(price_data, signals, initial_capital=10000):
    """
    Perform basic backtesting of the trading strategy.
    
    Args:
        price_data (pd.DataFrame): Price data with Close column
        signals (pd.DataFrame): Trading signals
        initial_capital (float): Starting capital amount
    
    Returns:
        dict: Backtesting results and metrics
    """
    print("Running backtest...")
    
    # Align price data and signals
    common_dates = price_data.index.intersection(signals.index)
    prices = price_data.loc[common_dates, 'Close']
    sigs = signals.loc[common_dates, 'Signal']
    
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Strategy returns: only take returns when signal is 1 (buy), otherwise 0
    strategy_returns = returns * sigs.shift(1)  # Shift signals to avoid look-ahead bias
    strategy_returns = strategy_returns.dropna()
    
    # Calculate cumulative returns
    buy_and_hold_cumulative = (1 + returns).cumprod()
    strategy_cumulative = (1 + strategy_returns).cumprod()
    
    # Calculate final values
    final_buy_and_hold = initial_capital * buy_and_hold_cumulative.iloc[-1]
    final_strategy = initial_capital * strategy_cumulative.iloc[-1]
    
    # Calculate metrics
    total_return_bh = (final_buy_and_hold - initial_capital) / initial_capital
    total_return_strategy = (final_strategy - initial_capital) / initial_capital
    
    # Calculate Sharpe ratios (assuming 252 trading days per year)
    sharpe_bh = np.sqrt(252) * returns.mean() / returns.std()
    sharpe_strategy = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
    
    # Calculate maximum drawdown for strategy
    cumulative_max = strategy_cumulative.expanding().max()
    drawdown = (strategy_cumulative - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    
    results = {
        'initial_capital': initial_capital,
        'final_value_buy_and_hold': final_buy_and_hold,
        'final_value_strategy': final_strategy,
        'total_return_buy_and_hold': total_return_bh,
        'total_return_strategy': total_return_strategy,
        'sharpe_ratio_buy_and_hold': sharpe_bh,
        'sharpe_ratio_strategy': sharpe_strategy,
        'max_drawdown': max_drawdown,
        'number_of_trades': sigs.sum(),
        'strategy_cumulative_returns': strategy_cumulative,
        'buy_and_hold_cumulative_returns': buy_and_hold_cumulative
    }
    
    print(f"\nBacktest Results:")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Buy & Hold Final Value: ${final_buy_and_hold:,.2f} ({total_return_bh:.1%} return)")
    print(f"Strategy Final Value: ${final_strategy:,.2f} ({total_return_strategy:.1%} return)")
    print(f"Strategy vs Buy & Hold: {((final_strategy / final_buy_and_hold) - 1):.1%}")
    print(f"Strategy Sharpe Ratio: {sharpe_strategy:.3f}")
    print(f"Buy & Hold Sharpe Ratio: {sharpe_bh:.3f}")
    print(f"Maximum Drawdown: {max_drawdown:.1%}")
    print(f"Number of Trades: {sigs.sum()}")
    
    return results

def main():
    """
    Main function to run the complete trading strategy pipeline.
    """
    print("=== S&P 500 Machine Learning Trading Strategy ===\n")
    
    try:
        # Step 1: Fetch S&P 500 data
        sp500_data = fetch_sp500_data(START_DATE, END_DATE)
        
        # Step 2: Fetch macroeconomic data
        macro_data = fetch_macro_data(FRED_API_KEY, START_DATE, END_DATE)
        
        # Step 3: Create technical indicators
        sp500_data = create_technical_indicators(sp500_data)
        
        # Step 4: Combine all data
        combined_data = combine_data(sp500_data, macro_data)
        
        # Step 5: Create target variable
        combined_data = create_target_variable(combined_data, LOOKBACK_DAYS)
        
        # Step 6: Prepare features and target
        X, y = prepare_features_and_target(combined_data)
        
        # Step 7: Train model
        model, X_test, y_test = train_model(X, y)
        
        # Step 8: Generate signals on test set
        signals = generate_signals(model, X_test, PROBABILITY_THRESHOLD)
        
        # Step 9: Backtest strategy
        test_price_data = combined_data.loc[X_test.index]
        backtest_results = backtest_strategy(test_price_data, signals)
        
        print("\n=== Strategy Complete ===")
        
        return {
            'model': model,
            'signals': signals,
            'backtest_results': backtest_results,
            'combined_data': combined_data
        }
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        return None

if __name__ == "__main__":
    results = main()