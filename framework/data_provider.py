#!/usr/bin/env python3
"""
Data Provider Module

Handles data fetching, preprocessing, and feature engineering for trading strategies.
Provides a clean interface for getting market data with technical indicators.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

class DataProvider:
    """
    Centralized data provider for market and economic data.
    """
    
    def __init__(self, fred_api_key=None):
        """
        Initialize data provider.
        
        Args:
            fred_api_key (str): FRED API key. If None, loads from environment.
        """
        self.fred_api_key = fred_api_key or os.getenv('FRED_API_KEY')
        self._fred_client = None
        
    @property
    def fred_client(self):
        """Lazy initialization of FRED client."""
        if self._fred_client is None and self.fred_api_key:
            self._fred_client = Fred(api_key=self.fred_api_key)
        return self._fred_client
    
    def fetch_sp500_data(self, start_date, end_date):
        """
        Fetch S&P 500 historical price data.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
        
        Returns:
            pd.DataFrame: DataFrame with S&P 500 OHLCV data
        """
        print("Fetching S&P 500 data...")
        
        sp500 = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
        
        # Clean column names
        sp500.columns = [col[0] if isinstance(col, tuple) else col for col in sp500.columns]
        
        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in sp500.columns:
                raise ValueError(f"Missing required column: {col}")
        
        print(f"Successfully fetched {len(sp500)} days of S&P 500 data")
        return sp500
    
    def fetch_macro_data(self, start_date, end_date):
        """
        Fetch macroeconomic data from FRED.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
        
        Returns:
            pd.DataFrame: DataFrame with macroeconomic indicators
        """
        print("Fetching macroeconomic data...")
        
        if not self.fred_api_key:
            print("WARNING: FRED API key not found!")
            print("Please add your API key to the .env file: FRED_API_KEY=your_actual_key_here")
            return self._generate_dummy_macro_data(start_date, end_date)
        
        try:
            indicators = {
                'FEDFUNDS': 'Federal Funds Rate',
                'UNRATE': 'Unemployment Rate', 
                'CPIAUCSL': 'Consumer Price Index',
                'DGS10': '10-Year Treasury Yield',
                'TB3MS': '3-Month Treasury Bill Rate'
            }
            
            macro_data = pd.DataFrame()
            
            for indicator, description in indicators.items():
                print(f"  Fetching {description}...")
                data = self.fred_client.get_series(indicator, start=start_date, end=end_date)
                macro_data[indicator] = data
            
            # Forward fill missing values
            macro_data = macro_data.fillna(method='ffill')
            
            print(f"Successfully fetched macroeconomic data with {len(macro_data)} observations")
            return macro_data
        
        except Exception as e:
            print(f"Error fetching FRED data: {e}")
            print("Using dummy data for demonstration...")
            return self._generate_dummy_macro_data(start_date, end_date)
    
    def _generate_dummy_macro_data(self, start_date, end_date):
        """Generate dummy macroeconomic data for testing."""
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        return pd.DataFrame({
            'FEDFUNDS': np.random.uniform(0, 5, len(date_range)),
            'UNRATE': np.random.uniform(3, 10, len(date_range)),
            'CPIAUCSL': np.random.uniform(200, 300, len(date_range)),
            'DGS10': np.random.uniform(1, 5, len(date_range)),
            'TB3MS': np.random.uniform(0, 4, len(date_range))
        }, index=date_range)
    
    def create_technical_indicators(self, df):
        """
        Create technical indicators from price data.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
        
        Returns:
            pd.DataFrame: DataFrame with additional technical indicator columns
        """
        print("Creating technical indicators...")
        
        df = df.copy()
        
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
    
    def combine_data(self, price_data, macro_data):
        """
        Combine price and macroeconomic data, aligning by date.
        
        Args:
            price_data (pd.DataFrame): Price data with technical indicators
            macro_data (pd.DataFrame): Macroeconomic data
        
        Returns:
            pd.DataFrame: Combined dataset aligned by date
        """
        print("Combining price and macroeconomic data...")
        
        # Ensure both DataFrames have datetime index
        if not isinstance(price_data.index, pd.DatetimeIndex):
            price_data.index = pd.to_datetime(price_data.index)
        if not isinstance(macro_data.index, pd.DatetimeIndex):
            macro_data.index = pd.to_datetime(macro_data.index)
        
        # Reindex macro data to match price data trading days and forward fill
        macro_data_daily = macro_data.reindex(price_data.index, method='ffill')
        
        # Combine the datasets
        combined_data = pd.concat([price_data, macro_data_daily], axis=1)
        
        # Forward fill any remaining missing values
        combined_data = combined_data.fillna(method='ffill')
        
        print(f"Combined dataset created with {len(combined_data)} observations")
        return combined_data
    
    def create_target_variable(self, df, lookback_days=5):
        """
        Create binary target variable based on future price movement.
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            lookback_days (int): Number of days to look ahead for target calculation
        
        Returns:
            pd.DataFrame: DataFrame with target variable added
        """
        print(f"Creating target variable with {lookback_days}-day lookback...")
        
        df = df.copy()
        
        # Calculate future price change (avoiding look-ahead bias)
        df['Future_Close'] = df['Close'].shift(-lookback_days)
        df['Future_Return'] = (df['Future_Close'] - df['Close']) / df['Close']
        
        # Create binary target: 1 for positive future return, 0 for negative/zero
        df['Target'] = (df['Future_Return'] > 0).astype(int)
        
        # Remove rows where we don't have future data
        df = df.dropna(subset=['Future_Return'])
        
        # Drop the helper columns to prevent data leakage
        df = df.drop(['Future_Close', 'Future_Return'], axis=1)
        
        positive_signals = df['Target'].sum()
        total_signals = len(df)
        print(f"Target variable created. Positive signals: {positive_signals}/{total_signals} ({positive_signals/total_signals:.1%})")
        
        return df
    
    def get_feature_columns(self):
        """
        Get list of feature columns for machine learning.
        
        Returns:
            list: List of feature column names
        """
        return [
            # Technical indicators
            'SMA_20', 'SMA_50', 'RSI', 'BB_Width', 'BB_Position',
            'Price_Change_1D', 'Price_Change_5D', 'Price_Change_20D',
            'Volume_Ratio',
            # Macroeconomic indicators
            'FEDFUNDS', 'UNRATE', 'CPIAUCSL', 'DGS10', 'TB3MS'
        ]
    
    def prepare_ml_data(self, df):
        """
        Prepare feature matrix and target vector for machine learning.
        
        Args:
            df (pd.DataFrame): Combined dataset with features and target
        
        Returns:
            tuple: (X, y) where X is features DataFrame and y is target Series
        """
        print("Preparing features and target variables...")
        
        # Get feature columns that exist in the dataframe
        feature_columns = self.get_feature_columns()
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
    
    def get_complete_dataset(self, start_date, end_date, lookback_days=5):
        """
        Get complete dataset with all features and target variable.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            lookback_days (int): Days to look ahead for target calculation
        
        Returns:
            tuple: (combined_data, X, y) - raw data, features, and targets
        """
        # Fetch raw data
        sp500_data = self.fetch_sp500_data(start_date, end_date)
        macro_data = self.fetch_macro_data(start_date, end_date)
        
        # Create features
        sp500_data = self.create_technical_indicators(sp500_data)
        combined_data = self.combine_data(sp500_data, macro_data)
        combined_data = self.create_target_variable(combined_data, lookback_days)
        
        # Prepare ML data
        X, y = self.prepare_ml_data(combined_data)
        
        return combined_data, X, y