#!/usr/bin/env python3
"""
Analyze ML model probability outputs to understand buy signal generation.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from sklearn.ensemble import RandomForestClassifier
import os
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')  # Use file output
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Import functions from the main script
import sys
sys.path.append('.')

def analyze_model_probabilities():
    """Quick analysis of model probability outputs"""
    print("=== Model Probability Analysis ===\n")
    
    # Simplified version - just run the core model training
    from sp500_trading_strategy import (
        fetch_sp500_data, fetch_macro_data, create_technical_indicators, 
        combine_data, create_target_variable, prepare_features_and_target, 
        train_model
    )
    
    # Configuration
    FRED_API_KEY = os.getenv('FRED_API_KEY')
    START_DATE = "1962-01-02"
    END_DATE = "2024-01-01"
    LOOKBACK_DAYS = 5
    
    try:
        print("1. Fetching data (abbreviated output)...")
        sp500_data = fetch_sp500_data(START_DATE, END_DATE)
        macro_data = fetch_macro_data(FRED_API_KEY, START_DATE, END_DATE)
        sp500_data = create_technical_indicators(sp500_data)
        combined_data = combine_data(sp500_data, macro_data)
        combined_data = create_target_variable(combined_data, LOOKBACK_DAYS)
        X, y = prepare_features_and_target(combined_data)
        
        print("\n2. Training model...")
        model, X_test, y_test = train_model(X, y)
        
        print("\n3. Analyzing probability outputs...")
        
        # Get prediction probabilities for test set
        probabilities = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (buy)
        
        # Probability distribution analysis
        print("\n=== PROBABILITY DISTRIBUTION ANALYSIS ===")
        print(f"Total test samples: {len(probabilities)}")
        print(f"Min probability: {probabilities.min():.3f}")
        print(f"Max probability: {probabilities.max():.3f}")
        print(f"Mean probability: {probabilities.mean():.3f}")
        print(f"Median probability: {np.median(probabilities):.3f}")
        print(f"Std deviation: {probabilities.std():.3f}")
        
        # Percentile analysis
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        print(f"\nPercentiles:")
        for p in percentiles:
            value = np.percentile(probabilities, p)
            print(f"  {p}th percentile: {value:.3f}")
        
        # Threshold analysis
        print(f"\n=== THRESHOLD IMPACT ANALYSIS ===")
        thresholds = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9]
        for threshold in thresholds:
            signals = (probabilities >= threshold).sum()
            percentage = signals / len(probabilities) * 100
            print(f"Threshold {threshold}: {signals:4d} signals ({percentage:5.1f}%)")
        
        # Time series analysis of probabilities
        print(f"\n=== TEMPORAL PROBABILITY ANALYSIS ===")
        prob_series = pd.Series(probabilities, index=X_test.index)
        
        # Annual averages
        annual_probs = prob_series.resample('Y').mean()
        print(f"Annual average probabilities:")
        for year, avg_prob in annual_probs.items():
            signals_that_year = (prob_series[prob_series.index.year == year.year] >= 0.6).sum()
            total_that_year = len(prob_series[prob_series.index.year == year.year])
            print(f"  {year.year}: {avg_prob:.3f} avg prob, {signals_that_year}/{total_that_year} signals at 0.6 threshold")
        
        # Check if model is well-calibrated
        print(f"\n=== MODEL CALIBRATION CHECK ===")
        # For a well-calibrated model, the fraction of actual positives 
        # should match the predicted probabilities
        
        # Bin probabilities and check actual outcomes
        bins = np.linspace(0, 1, 11)  # 10 bins
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        print(f"Calibration analysis (predicted vs actual positive rate):")
        for i in range(len(bins)-1):
            mask = (probabilities >= bins[i]) & (probabilities < bins[i+1])
            if mask.sum() > 0:
                actual_positive_rate = y_test[mask].mean()
                predicted_prob = bin_centers[i]
                count = mask.sum()
                print(f"  Prob {bins[i]:.1f}-{bins[i+1]:.1f}: "
                      f"Predicted {predicted_prob:.2f}, Actual {actual_positive_rate:.2f}, "
                      f"Count: {count}")
        
        # Create probability distribution histogram
        print(f"\n4. Creating probability distribution plot...")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ML Model Probability Analysis', fontsize=16, fontweight='bold')
        
        # 1. Probability histogram
        ax1.hist(probabilities, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(0.6, color='red', linestyle='--', label='Current Threshold (0.6)')
        ax1.axvline(probabilities.mean(), color='green', linestyle='--', label=f'Mean ({probabilities.mean():.3f})')
        ax1.set_xlabel('Predicted Probability')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Predicted Probabilities')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Time series of probabilities
        ax2.plot(prob_series.index, probabilities, alpha=0.6, linewidth=0.5)
        ax2.axhline(0.6, color='red', linestyle='--', label='Current Threshold (0.6)')
        ax2.axhline(probabilities.mean(), color='green', linestyle='--', label='Mean')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Predicted Probability')
        ax2.set_title('Probabilities Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Threshold vs Signal Count
        signal_counts = [((probabilities >= t).sum()) for t in thresholds]
        signal_pcts = [(count/len(probabilities)*100) for count in signal_counts]
        
        ax3.plot(thresholds, signal_pcts, marker='o', linewidth=2, markersize=6)
        ax3.axvline(0.6, color='red', linestyle='--', alpha=0.7, label='Current (0.6)')
        ax3.set_xlabel('Probability Threshold')
        ax3.set_ylabel('Percentage of Buy Signals (%)')
        ax3.set_title('Threshold Impact on Signal Generation')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Annual signal counts
        years = annual_probs.index.year
        annual_signal_counts = []
        for year in years:
            year_probs = prob_series[prob_series.index.year == year]
            signals = (year_probs >= 0.6).sum()
            annual_signal_counts.append(signals)
        
        ax4.bar(years, annual_signal_counts, alpha=0.7, color='orange')
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Number of Buy Signals')
        ax4.set_title('Annual Buy Signals (0.6 threshold)')
        ax4.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        filename = 'model_probability_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Analysis plot saved as '{filename}'")
        plt.close()
        
        return probabilities, y_test
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        return None, None

if __name__ == "__main__":
    probabilities, y_test = analyze_model_probabilities()