#!/usr/bin/env python3
"""
Signal Generator Framework

Abstract base class and concrete implementations for generating trading signals.
Each signal generator takes features as input and outputs buy/sell signals.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

class SignalGenerator(ABC):
    """
    Abstract base class for signal generators.
    """
    
    def __init__(self, name=None):
        """
        Initialize signal generator.
        
        Args:
            name (str): Name of the signal generator
        """
        self.name = name or self.__class__.__name__
        self.is_trained = False
    
    @abstractmethod
    def fit(self, X_train, y_train):
        """
        Train the signal generator.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training targets
        """
        pass
    
    @abstractmethod
    def generate_signals(self, X):
        """
        Generate trading signals.
        
        Args:
            X (pd.DataFrame): Features for signal generation
            
        Returns:
            pd.DataFrame: DataFrame with columns ['Signal', 'Probability'] where
                         Signal is 1 for buy, 0 for sell/hold
                         Probability is confidence score (0-1)
        """
        pass
    
    def get_feature_importance(self):
        """
        Get feature importance if available.
        
        Returns:
            pd.DataFrame or None: DataFrame with feature importance scores
        """
        return None


class RandomForestSignalGenerator(SignalGenerator):
    """
    Random Forest based signal generator.
    """
    
    def __init__(self, name="Random Forest", probability_threshold=0.5, **rf_params):
        """
        Initialize Random Forest signal generator.
        
        Args:
            name (str): Name of the signal generator
            probability_threshold (float): Threshold for buy signal (0-1)
            **rf_params: Additional parameters for RandomForestClassifier
        """
        super().__init__(name)
        self.probability_threshold = probability_threshold
        
        # Default RF parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42
        }
        default_params.update(rf_params)
        
        self.model = RandomForestClassifier(**default_params)
        self.feature_names = None
    
    def fit(self, X_train, y_train):
        """Train the Random Forest model."""
        print(f"Training {self.name} model...")
        
        self.feature_names = X_train.columns.tolist()
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Print training performance
        train_accuracy = self.model.score(X_train, y_train)
        print(f"{self.name} training accuracy: {train_accuracy:.3f}")
        
        return self
    
    def generate_signals(self, X):
        """Generate signals using Random Forest predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before generating signals")
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(X)[:, 1]  # Probability of class 1 (buy)
        
        # Generate binary signals based on threshold
        signals = (probabilities >= self.probability_threshold).astype(int)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Signal': signals,
            'Probability': probabilities
        }, index=X.index)
        
        return results
    
    def get_feature_importance(self):
        """Get feature importance from Random Forest."""
        if not self.is_trained:
            return None
            
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


class LogisticRegressionSignalGenerator(SignalGenerator):
    """
    Logistic Regression based signal generator.
    """
    
    def __init__(self, name="Logistic Regression", probability_threshold=0.5, **lr_params):
        """
        Initialize Logistic Regression signal generator.
        
        Args:
            name (str): Name of the signal generator  
            probability_threshold (float): Threshold for buy signal (0-1)
            **lr_params: Additional parameters for LogisticRegression
        """
        super().__init__(name)
        self.probability_threshold = probability_threshold
        
        # Default LR parameters
        default_params = {
            'random_state': 42,
            'max_iter': 1000
        }
        default_params.update(lr_params)
        
        self.model = LogisticRegression(**default_params)
        self.feature_names = None
    
    def fit(self, X_train, y_train):
        """Train the Logistic Regression model."""
        print(f"Training {self.name} model...")
        
        self.feature_names = X_train.columns.tolist()
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Print training performance
        train_accuracy = self.model.score(X_train, y_train)
        print(f"{self.name} training accuracy: {train_accuracy:.3f}")
        
        return self
    
    def generate_signals(self, X):
        """Generate signals using Logistic Regression predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before generating signals")
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(X)[:, 1]  # Probability of class 1 (buy)
        
        # Generate binary signals based on threshold
        signals = (probabilities >= self.probability_threshold).astype(int)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Signal': signals,
            'Probability': probabilities
        }, index=X.index)
        
        return results
    
    def get_feature_importance(self):
        """Get feature coefficients from Logistic Regression."""
        if not self.is_trained:
            return None
            
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_[0],
            'abs_coefficient': np.abs(self.model.coef_[0])
        }).sort_values('abs_coefficient', ascending=False)
        
        return importance_df


class SVMSignalGenerator(SignalGenerator):
    """
    Support Vector Machine based signal generator.
    """
    
    def __init__(self, name="SVM", probability_threshold=0.5, **svm_params):
        """
        Initialize SVM signal generator.
        
        Args:
            name (str): Name of the signal generator
            probability_threshold (float): Threshold for buy signal (0-1)
            **svm_params: Additional parameters for SVC
        """
        super().__init__(name)
        self.probability_threshold = probability_threshold
        
        # Default SVM parameters
        default_params = {
            'probability': True,  # Enable probability estimates
            'random_state': 42,
            'C': 1.0,
            'kernel': 'rbf'
        }
        default_params.update(svm_params)
        
        self.model = SVC(**default_params)
        self.feature_names = None
    
    def fit(self, X_train, y_train):
        """Train the SVM model."""
        print(f"Training {self.name} model...")
        
        self.feature_names = X_train.columns.tolist()
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Print training performance
        train_accuracy = self.model.score(X_train, y_train)
        print(f"{self.name} training accuracy: {train_accuracy:.3f}")
        
        return self
    
    def generate_signals(self, X):
        """Generate signals using SVM predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before generating signals")
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(X)[:, 1]  # Probability of class 1 (buy)
        
        # Generate binary signals based on threshold
        signals = (probabilities >= self.probability_threshold).astype(int)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Signal': signals,
            'Probability': probabilities
        }, index=X.index)
        
        return results


class SimpleMovingAverageSignalGenerator(SignalGenerator):
    """
    Simple Moving Average crossover signal generator.
    """
    
    def __init__(self, name="SMA Crossover", short_window=20, long_window=50):
        """
        Initialize SMA signal generator.
        
        Args:
            name (str): Name of the signal generator
            short_window (int): Short moving average window
            long_window (int): Long moving average window
        """
        super().__init__(name)
        self.short_window = short_window
        self.long_window = long_window
    
    def fit(self, X_train, y_train):
        """SMA doesn't need training, but we store this for consistency."""
        print(f"Initializing {self.name} (no training required)")
        self.is_trained = True
        return self
    
    def generate_signals(self, X):
        """Generate signals using SMA crossover."""
        if not self.is_trained:
            raise ValueError("Generator must be initialized before generating signals")
        
        # Assume we have SMA columns available
        if f'SMA_{self.short_window}' not in X.columns or f'SMA_{self.long_window}' not in X.columns:
            raise ValueError(f"Required SMA columns not found. Need SMA_{self.short_window} and SMA_{self.long_window}")
        
        short_sma = X[f'SMA_{self.short_window}']
        long_sma = X[f'SMA_{self.long_window}']
        
        # Generate signals: 1 when short SMA > long SMA
        signals = (short_sma > long_sma).astype(int)
        
        # For probability, use the relative difference between SMAs
        sma_ratio = short_sma / long_sma
        probabilities = np.clip((sma_ratio - 1) * 10 + 0.5, 0, 1)  # Normalize to 0-1 range
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Signal': signals,
            'Probability': probabilities
        }, index=X.index)
        
        return results


class RSISignalGenerator(SignalGenerator):
    """
    RSI-based signal generator.
    """
    
    def __init__(self, name="RSI", oversold_threshold=30, overbought_threshold=70):
        """
        Initialize RSI signal generator.
        
        Args:
            name (str): Name of the signal generator
            oversold_threshold (float): RSI level considered oversold (buy signal)
            overbought_threshold (float): RSI level considered overbought (sell signal)
        """
        super().__init__(name)
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
    
    def fit(self, X_train, y_train):
        """RSI doesn't need training."""
        print(f"Initializing {self.name} (no training required)")
        self.is_trained = True
        return self
    
    def generate_signals(self, X):
        """Generate signals using RSI levels."""
        if not self.is_trained:
            raise ValueError("Generator must be initialized before generating signals")
        
        if 'RSI' not in X.columns:
            raise ValueError("RSI column not found in features")
        
        rsi = X['RSI']
        
        # Generate signals: 1 when RSI < oversold_threshold
        signals = (rsi < self.oversold_threshold).astype(int)
        
        # For probability, use distance from thresholds
        probabilities = np.where(
            rsi < self.oversold_threshold,
            (self.oversold_threshold - rsi) / self.oversold_threshold,  # Higher prob when more oversold
            0.0  # No buy signal otherwise
        )
        probabilities = np.clip(probabilities, 0, 1)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Signal': signals,
            'Probability': probabilities
        }, index=X.index)
        
        return results


class BuyAndHoldSignalGenerator(SignalGenerator):
    """
    Buy and hold signal generator (always buy signal).
    """
    
    def __init__(self, name="Buy & Hold"):
        """Initialize buy and hold signal generator."""
        super().__init__(name)
    
    def fit(self, X_train, y_train):
        """Buy and hold doesn't need training."""
        print(f"Initializing {self.name} (no training required)")
        self.is_trained = True
        return self
    
    def generate_signals(self, X):
        """Generate constant buy signals."""
        if not self.is_trained:
            raise ValueError("Generator must be initialized before generating signals")
        
        # Always generate buy signal with 100% probability
        signals = pd.Series(1, index=X.index)
        probabilities = pd.Series(1.0, index=X.index)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Signal': signals,
            'Probability': probabilities
        }, index=X.index)
        
        return results


# Factory function for creating signal generators
def create_signal_generator(generator_type, **kwargs):
    """
    Factory function to create signal generators.
    
    Args:
        generator_type (str): Type of generator ('rf', 'lr', 'svm', 'sma', 'rsi', 'buy_hold')
        **kwargs: Additional parameters for the generator
    
    Returns:
        SignalGenerator: Configured signal generator
    """
    generators = {
        'rf': RandomForestSignalGenerator,
        'random_forest': RandomForestSignalGenerator,
        'lr': LogisticRegressionSignalGenerator,
        'logistic': LogisticRegressionSignalGenerator,
        'svm': SVMSignalGenerator,
        'sma': SimpleMovingAverageSignalGenerator,
        'moving_average': SimpleMovingAverageSignalGenerator,
        'rsi': RSISignalGenerator,
        'buy_hold': BuyAndHoldSignalGenerator,
        'buy_and_hold': BuyAndHoldSignalGenerator
    }
    
    if generator_type.lower() not in generators:
        available = ', '.join(generators.keys())
        raise ValueError(f"Unknown generator type '{generator_type}'. Available: {available}")
    
    generator_class = generators[generator_type.lower()]
    return generator_class(**kwargs)