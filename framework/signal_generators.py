#!/usr/bin/env python3
"""
Signal Generator Framework

Abstract base class and concrete implementations for generating trading signals.
Each signal generator takes features as input and outputs buy/sell signals.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
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


class KNeighborsSignalGenerator(SignalGenerator):
    """
    K-Nearest Neighbors based signal generator.
    """
    
    def __init__(self, name="K-Neighbors", probability_threshold=0.5, **knn_params):
        """
        Initialize K-Neighbors signal generator.
        
        Args:
            name (str): Name of the signal generator
            probability_threshold (float): Threshold for buy signal (0-1)
            **knn_params: Additional parameters for KNeighborsClassifier
        """
        super().__init__(name)
        self.probability_threshold = probability_threshold
        
        # Default KNN parameters
        default_params = {
            'n_neighbors': 5,
            'weights': 'distance',  # Weight neighbors by distance
            'algorithm': 'auto',
            'metric': 'euclidean'
        }
        default_params.update(knn_params)
        
        self.model = KNeighborsClassifier(**default_params)
        self.feature_names = None
    
    def fit(self, X_train, y_train):
        """Train the K-Neighbors model."""
        print(f"Training {self.name} model...")
        
        self.feature_names = X_train.columns.tolist()
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Print training performance
        train_accuracy = self.model.score(X_train, y_train)
        print(f"{self.name} training accuracy: {train_accuracy:.3f}")
        
        return self
    
    def generate_signals(self, X):
        """Generate signals using K-Neighbors predictions."""
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


class XGBoostSignalGenerator(SignalGenerator):
    """
    XGBoost based signal generator.
    """
    
    def __init__(self, name="XGBoost", probability_threshold=0.5, **xgb_params):
        """
        Initialize XGBoost signal generator.
        
        Args:
            name (str): Name of the signal generator
            probability_threshold (float): Threshold for buy signal (0-1)
            **xgb_params: Additional parameters for XGBClassifier
        """
        super().__init__(name)
        self.probability_threshold = probability_threshold
        
        # Default XGBoost parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'eval_metric': 'logloss',
            'verbosity': 0  # Suppress XGBoost output
        }
        default_params.update(xgb_params)
        
        self.model = xgb.XGBClassifier(**default_params)
        self.feature_names = None
    
    def fit(self, X_train, y_train):
        """Train the XGBoost model."""
        print(f"Training {self.name} model...")
        
        self.feature_names = X_train.columns.tolist()
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Print training performance
        train_accuracy = self.model.score(X_train, y_train)
        print(f"{self.name} training accuracy: {train_accuracy:.3f}")
        
        return self
    
    def generate_signals(self, X):
        """Generate signals using XGBoost predictions."""
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
        """Get feature importance from XGBoost."""
        if not self.is_trained:
            return None
            
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


class LightGBMSignalGenerator(SignalGenerator):
    """
    LightGBM based signal generator.
    """
    
    def __init__(self, name="LightGBM", probability_threshold=0.5, **lgb_params):
        """
        Initialize LightGBM signal generator.
        
        Args:
            name (str): Name of the signal generator
            probability_threshold (float): Threshold for buy signal (0-1)
            **lgb_params: Additional parameters for LGBMClassifier
        """
        super().__init__(name)
        self.probability_threshold = probability_threshold
        
        # Default LightGBM parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbosity': -1  # Suppress LightGBM output
        }
        default_params.update(lgb_params)
        
        self.model = lgb.LGBMClassifier(**default_params)
        self.feature_names = None
    
    def fit(self, X_train, y_train):
        """Train the LightGBM model."""
        print(f"Training {self.name} model...")
        
        self.feature_names = X_train.columns.tolist()
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Print training performance
        train_accuracy = self.model.score(X_train, y_train)
        print(f"{self.name} training accuracy: {train_accuracy:.3f}")
        
        return self
    
    def generate_signals(self, X):
        """Generate signals using LightGBM predictions."""
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
        """Get feature importance from LightGBM."""
        if not self.is_trained:
            return None
            
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


class ExtraTreesSignalGenerator(SignalGenerator):
    """
    Extra Trees (Extremely Randomized Trees) based signal generator.
    """
    
    def __init__(self, name="Extra Trees", probability_threshold=0.5, **et_params):
        """
        Initialize Extra Trees signal generator.
        
        Args:
            name (str): Name of the signal generator
            probability_threshold (float): Threshold for buy signal (0-1)
            **et_params: Additional parameters for ExtraTreesClassifier
        """
        super().__init__(name)
        self.probability_threshold = probability_threshold
        
        # Default Extra Trees parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42
        }
        default_params.update(et_params)
        
        self.model = ExtraTreesClassifier(**default_params)
        self.feature_names = None
    
    def fit(self, X_train, y_train):
        """Train the Extra Trees model."""
        print(f"Training {self.name} model...")
        
        self.feature_names = X_train.columns.tolist()
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Print training performance
        train_accuracy = self.model.score(X_train, y_train)
        print(f"{self.name} training accuracy: {train_accuracy:.3f}")
        
        return self
    
    def generate_signals(self, X):
        """Generate signals using Extra Trees predictions."""
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
        """Get feature importance from Extra Trees."""
        if not self.is_trained:
            return None
            
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


class GradientBoostingSignalGenerator(SignalGenerator):
    """
    Gradient Boosting based signal generator (scikit-learn version).
    """
    
    def __init__(self, name="Gradient Boosting", probability_threshold=0.5, **gb_params):
        """
        Initialize Gradient Boosting signal generator.
        
        Args:
            name (str): Name of the signal generator
            probability_threshold (float): Threshold for buy signal (0-1)
            **gb_params: Additional parameters for GradientBoostingClassifier
        """
        super().__init__(name)
        self.probability_threshold = probability_threshold
        
        # Default Gradient Boosting parameters
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42
        }
        default_params.update(gb_params)
        
        self.model = GradientBoostingClassifier(**default_params)
        self.feature_names = None
    
    def fit(self, X_train, y_train):
        """Train the Gradient Boosting model."""
        print(f"Training {self.name} model...")
        
        self.feature_names = X_train.columns.tolist()
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Print training performance
        train_accuracy = self.model.score(X_train, y_train)
        print(f"{self.name} training accuracy: {train_accuracy:.3f}")
        
        return self
    
    def generate_signals(self, X):
        """Generate signals using Gradient Boosting predictions."""
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
        """Get feature importance from Gradient Boosting."""
        if not self.is_trained:
            return None
            
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


class MLPSignalGenerator(SignalGenerator):
    """
    Multi-Layer Perceptron (Neural Network) based signal generator.
    """
    
    def __init__(self, name="Neural Network", probability_threshold=0.5, **mlp_params):
        """
        Initialize MLP signal generator.
        
        Args:
            name (str): Name of the signal generator
            probability_threshold (float): Threshold for buy signal (0-1)
            **mlp_params: Additional parameters for MLPClassifier
        """
        super().__init__(name)
        self.probability_threshold = probability_threshold
        
        # Default MLP parameters
        default_params = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001,
            'learning_rate': 'constant',
            'max_iter': 500,
            'random_state': 42,
            'early_stopping': True,
            'validation_fraction': 0.1
        }
        default_params.update(mlp_params)
        
        self.model = MLPClassifier(**default_params)
        self.feature_names = None
    
    def fit(self, X_train, y_train):
        """Train the MLP model."""
        print(f"Training {self.name} model...")
        
        self.feature_names = X_train.columns.tolist()
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Print training performance
        train_accuracy = self.model.score(X_train, y_train)
        print(f"{self.name} training accuracy: {train_accuracy:.3f}")
        
        return self
    
    def generate_signals(self, X):
        """Generate signals using MLP predictions."""
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
        'knn': KNeighborsSignalGenerator,
        'kneighbors': KNeighborsSignalGenerator,
        'k_neighbors': KNeighborsSignalGenerator,
        'xgb': XGBoostSignalGenerator,
        'xgboost': XGBoostSignalGenerator,
        'lgb': LightGBMSignalGenerator,
        'lightgbm': LightGBMSignalGenerator,
        'et': ExtraTreesSignalGenerator,
        'extra_trees': ExtraTreesSignalGenerator,
        'gb': GradientBoostingSignalGenerator,
        'gradient_boosting': GradientBoostingSignalGenerator,
        'mlp': MLPSignalGenerator,
        'neural_network': MLPSignalGenerator,
        'nn': MLPSignalGenerator,
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