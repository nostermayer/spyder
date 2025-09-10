#!/usr/bin/env python3
"""
Trading Strategy Framework Runner

Main script for running and comparing multiple trading strategies.
Demonstrates how to use the framework with different signal generators.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add framework to path
sys.path.append('framework')

from data_provider import DataProvider
from signal_generators import create_signal_generator
from backtest_engine import BacktestEngine, StrategyComparison
from performance_analyzer import PerformanceAnalyzer

def run_strategy_comparison():
    """
    Run comprehensive strategy comparison using the framework.
    """
    print("="*80)
    print("TRADING STRATEGY FRAMEWORK - COMPREHENSIVE COMPARISON")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration
    START_DATE = "1962-01-02"
    END_DATE = "2024-01-01"
    LOOKBACK_DAYS = 5
    TEST_SIZE = 0.2
    
    INITIAL_CAPITAL = 10000
    TRADING_COST = 0.001  # 0.1%
    SLIPPAGE = 0.0001     # 0.01%
    
    print(f"Configuration:")
    print(f"  Data Period: {START_DATE} to {END_DATE}")
    print(f"  Lookback Days: {LOOKBACK_DAYS}")
    print(f"  Test Size: {TEST_SIZE:.0%}")
    print(f"  Initial Capital: ${INITIAL_CAPITAL:,}")
    print(f"  Trading Cost: {TRADING_COST:.3%}")
    print(f"  Slippage: {SLIPPAGE:.4%}")
    print()
    
    # Step 1: Get data
    print("STEP 1: Data Preparation")
    print("-" * 40)
    
    data_provider = DataProvider()
    combined_data, X, y = data_provider.get_complete_dataset(
        START_DATE, END_DATE, LOOKBACK_DAYS
    )
    
    # Time-based split
    split_index = int(len(X) * (1 - TEST_SIZE))
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print()
    
    # Step 2: Define strategies to test
    print("STEP 2: Strategy Definition")
    print("-" * 40)
    
    strategies_config = {
        'Random Forest (0.4)': {
            'type': 'rf',
            'params': {'probability_threshold': 0.4, 'n_estimators': 100}
        },
        'Random Forest (0.5)': {
            'type': 'rf',
            'params': {'probability_threshold': 0.5, 'n_estimators': 100}
        },
        'Random Forest (0.6)': {
            'type': 'rf', 
            'params': {'probability_threshold': 0.6, 'n_estimators': 100}
        },
        'Logistic Regression': {
            'type': 'lr',
            'params': {'probability_threshold': 0.5}
        },
        'SVM': {
            'type': 'svm',
            'params': {'probability_threshold': 0.5, 'C': 1.0}
        },
        'K-Neighbors (5)': {
            'type': 'knn',
            'params': {'probability_threshold': 0.5, 'n_neighbors': 5, 'weights': 'distance'}
        },
        'K-Neighbors (10)': {
            'type': 'knn',
            'params': {'probability_threshold': 0.5, 'n_neighbors': 10, 'weights': 'distance'}
        },
        'K-Neighbors Uniform': {
            'type': 'knn',
            'params': {'probability_threshold': 0.5, 'n_neighbors': 7, 'weights': 'uniform'}
        },
        'XGBoost': {
            'type': 'xgb',
            'params': {'probability_threshold': 0.5, 'n_estimators': 100, 'max_depth': 6}
        },
        'XGBoost Conservative': {
            'type': 'xgb',
            'params': {'probability_threshold': 0.6, 'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.05}
        },
        'LightGBM': {
            'type': 'lgb',
            'params': {'probability_threshold': 0.5, 'n_estimators': 100, 'max_depth': 6}
        },
        'LightGBM Aggressive': {
            'type': 'lgb',
            'params': {'probability_threshold': 0.4, 'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.1}
        },
        'Extra Trees': {
            'type': 'et',
            'params': {'probability_threshold': 0.5, 'n_estimators': 100, 'max_depth': 10}
        },
        'Extra Trees Deep': {
            'type': 'et',
            'params': {'probability_threshold': 0.45, 'n_estimators': 200, 'max_depth': 15}
        },
        'Gradient Boosting': {
            'type': 'gb',
            'params': {'probability_threshold': 0.5, 'n_estimators': 100, 'learning_rate': 0.1}
        },
        'Gradient Boosting Slow': {
            'type': 'gb',
            'params': {'probability_threshold': 0.55, 'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 4}
        },
        'Neural Network': {
            'type': 'mlp',
            'params': {'probability_threshold': 0.5, 'hidden_layer_sizes': (100, 50), 'max_iter': 800, 'alpha': 0.001}
        },
        'Neural Network Deep': {
            'type': 'mlp',
            'params': {'probability_threshold': 0.45, 'hidden_layer_sizes': (200, 100, 50), 'max_iter': 1000, 'alpha': 0.0005}
        },
        'Neural Network Wide': {
            'type': 'mlp',
            'params': {'probability_threshold': 0.48, 'hidden_layer_sizes': (300, 150), 'max_iter': 800, 'alpha': 0.001}
        },
        'Neural Network Narrow Deep': {
            'type': 'mlp',
            'params': {'probability_threshold': 0.52, 'hidden_layer_sizes': (80, 60, 40, 20), 'max_iter': 1200, 'alpha': 0.0008}
        },
        'Neural Network Conservative': {
            'type': 'mlp',
            'params': {'probability_threshold': 0.6, 'hidden_layer_sizes': (150, 100, 50), 'max_iter': 1000, 'alpha': 0.002}
        },
        'Neural Network Aggressive': {
            'type': 'mlp',
            'params': {'probability_threshold': 0.4, 'hidden_layer_sizes': (250, 150, 75), 'max_iter': 1200, 'alpha': 0.0003}
        },
        'Neural Network Tanh': {
            'type': 'mlp',
            'params': {'probability_threshold': 0.5, 'hidden_layer_sizes': (150, 100), 'activation': 'tanh', 'max_iter': 800, 'alpha': 0.001}
        },
        'Neural Network Logistic': {
            'type': 'mlp',
            'params': {'probability_threshold': 0.5, 'hidden_layer_sizes': (120, 80), 'activation': 'logistic', 'max_iter': 600, 'alpha': 0.001}
        },
        'Neural Network High Reg': {
            'type': 'mlp',
            'params': {'probability_threshold': 0.5, 'hidden_layer_sizes': (200, 100), 'max_iter': 1000, 'alpha': 0.01}
        },
        'Neural Network Low Reg': {
            'type': 'mlp',
            'params': {'probability_threshold': 0.48, 'hidden_layer_sizes': (180, 120, 60), 'max_iter': 1200, 'alpha': 0.0001}
        },
        'SMA Crossover': {
            'type': 'sma',
            'params': {'short_window': 20, 'long_window': 50}
        },
        'RSI Strategy': {
            'type': 'rsi',
            'params': {'oversold_threshold': 30}
        },
        'Buy & Hold': {
            'type': 'buy_hold',
            'params': {}
        }
    }
    
    print(f"Strategies to test: {len(strategies_config)}")
    for name in strategies_config.keys():
        print(f"  • {name}")
    print()
    
    # Step 3: Train strategies and generate signals
    print("STEP 3: Strategy Training & Signal Generation")
    print("-" * 40)
    
    strategy_signals = {}
    
    for strategy_name, config in strategies_config.items():
        print(f"Processing {strategy_name}...")
        
        try:
            # Create signal generator
            generator = create_signal_generator(
                config['type'], 
                name=strategy_name,
                **config['params']
            )
            
            # Train (if needed)
            generator.fit(X_train, y_train)
            
            # Generate signals on test set
            signals = generator.generate_signals(X_test)
            strategy_signals[strategy_name] = signals
            
            # Print signal statistics
            signal_count = signals['Signal'].sum()
            signal_rate = signal_count / len(signals)
            avg_probability = signals['Probability'].mean()
            
            print(f"  Signals generated: {signal_count}/{len(signals)} ({signal_rate:.1%})")
            print(f"  Average probability: {avg_probability:.3f}")
            
            # Print feature importance if available
            importance = generator.get_feature_importance()
            if importance is not None:
                print(f"  Top 3 features: {', '.join(importance.head(3)['feature'].tolist())}")
            
            print()
            
        except Exception as e:
            print(f"  ERROR: {e}")
            print()
            continue
    
    # Step 4: Backtesting
    print("STEP 4: Backtesting")
    print("-" * 40)
    
    # Create backtest engine
    backtest_engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        trading_cost_pct=TRADING_COST,
        slippage_pct=SLIPPAGE
    )
    
    # Get test period price data
    test_price_data = combined_data.loc[X_test.index]
    
    # Run backtests
    comparison = StrategyComparison(backtest_engine)
    backtest_results = comparison.run_comparison(test_price_data, strategy_signals)
    
    # Step 5: Performance Analysis
    print("STEP 5: Performance Analysis & Visualization")
    print("-" * 40)
    
    analyzer = PerformanceAnalyzer(interactive_plots=False)
    
    # Generate comparison plots
    comparison_plot = analyzer.plot_strategy_comparison(
        comparison.results, 
        'strategy_comparison_framework.png'
    )
    
    # Generate rolling metrics plot
    rolling_plot = analyzer.create_rolling_metrics_plot(
        comparison.results,
        window_days=252,  # 1 year rolling window
        save_path='rolling_metrics_framework.png'
    )
    
    # Generate individual strategy plots for top performers
    print("\nGenerating individual strategy analysis for top performers...")
    
    # Get top 3 strategies by Sharpe ratio
    sorted_results = sorted(
        comparison.results.items(), 
        key=lambda x: x[1]['sharpe_ratio_strategy'], 
        reverse=True
    )
    
    for i, (strategy_name, result) in enumerate(sorted_results[:3]):
        filename = f'top_strategy_{i+1}_{strategy_name.lower().replace(" ", "_")}.png'
        analyzer.plot_single_strategy_analysis(result, filename)
        print(f"  Generated analysis for {strategy_name}")
    
    # Generate performance report
    report_file = analyzer.generate_performance_report(
        comparison.results,
        'performance_report_framework.txt'
    )
    
    # Step 6: Summary
    print("\nSTEP 6: Summary & Recommendations")
    print("-" * 40)
    
    # Print detailed comparison table
    print("\nDETAILED STRATEGY COMPARISON:")
    comparison_table = comparison.get_comparison_table()
    print(comparison_table.to_string())
    
    # Print additional insights
    print(f"\nSTRATEGY INSIGHTS:")
    print("-" * 20)
    
    # Sort strategies by different metrics
    strategies_by_sharpe = sorted(comparison.results.items(), 
                                 key=lambda x: x[1]['sharpe_ratio_strategy'], reverse=True)
    strategies_by_return = sorted(comparison.results.items(), 
                                 key=lambda x: x[1]['total_return_strategy'], reverse=True)
    strategies_by_drawdown = sorted(comparison.results.items(), 
                                   key=lambda x: x[1]['max_drawdown_strategy'], reverse=True)  # Best = least negative
    
    print(f"Best Sharpe Ratio: {strategies_by_sharpe[0][0]} ({strategies_by_sharpe[0][1]['sharpe_ratio_strategy']:.3f})")
    print(f"Highest Return: {strategies_by_return[0][0]} ({strategies_by_return[0][1]['total_return_strategy']:.1%})")
    print(f"Lowest Drawdown: {strategies_by_drawdown[0][0]} ({strategies_by_drawdown[0][1]['max_drawdown_strategy']:.1%})")
    
    # Trading activity analysis
    print(f"\nTRADING ACTIVITY ANALYSIS:")
    print("-" * 25)
    for name, result in comparison.results.items():
        print(f"{name:20}: {result['total_trades']:3d} trades, {result['signal_rate']:5.1%} in market, {result['win_rate']:5.1%} win rate")
    
    comparison.print_summary()
    
    print(f"\nBest Strategy Analysis:")
    best_strategy = comparison.get_best_strategy('sharpe_ratio_strategy')
    best_result = comparison.results[best_strategy]
    
    print(f"  Best by Sharpe Ratio: {best_strategy}")
    print(f"  Sharpe Ratio: {best_result['sharpe_ratio_strategy']:.3f}")
    print(f"  Total Return: {best_result['total_return_strategy']:.1%}")
    print(f"  Max Drawdown: {best_result['max_drawdown_strategy']:.1%}")
    print(f"  Win Rate: {best_result['win_rate']:.1%}")
    
    print(f"\nFiles Generated:")
    print(f"  • {comparison_plot}")
    print(f"  • {rolling_plot}")
    print(f"  • {report_file}")
    print(f"  • Individual strategy plots for top 3 performers")
    
    print(f"\nFramework demonstration completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return comparison.results

def run_single_strategy_example():
    """
    Example of running a single strategy using the framework.
    """
    print("\nSINGLE STRATEGY EXAMPLE")
    print("-" * 40)
    
    # Configuration
    START_DATE = "2010-01-01"
    END_DATE = "2024-01-01"
    
    # Get data
    data_provider = DataProvider()
    combined_data, X, y = data_provider.get_complete_dataset(START_DATE, END_DATE)
    
    # Split data
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    # Create and train strategy
    strategy = create_signal_generator('rf', 
                                     name='Example RF Strategy',
                                     probability_threshold=0.45,
                                     n_estimators=200)
    
    strategy.fit(X_train, y_train)
    signals = strategy.generate_signals(X_test)
    
    # Backtest
    backtest_engine = BacktestEngine()
    test_data = combined_data.loc[X_test.index]
    result = backtest_engine.backtest_strategy(test_data, signals, "Example Strategy")
    
    # Analyze
    analyzer = PerformanceAnalyzer()
    analyzer.plot_single_strategy_analysis(result, 'example_strategy.png')
    
    print("Single strategy example completed!")
    return result

def custom_strategy_example():
    """
    Example of creating a custom strategy by combining multiple signals.
    """
    print("\nCUSTOM STRATEGY EXAMPLE")
    print("-" * 40)
    
    # This is a placeholder for showing how to create custom strategies
    # You could combine multiple signal generators, add filters, etc.
    print("Custom strategy example would go here...")
    print("Ideas:")
    print("  • Combine ML model with technical indicators")
    print("  • Add volatility filters")
    print("  • Implement regime detection")
    print("  • Add position sizing rules")
    

if __name__ == "__main__":
    try:
        # Run comprehensive comparison
        results = run_strategy_comparison()
        
        # Run single strategy example
        # single_result = run_single_strategy_example()
        
        # Show custom strategy ideas
        # custom_strategy_example()
        
    except KeyboardInterrupt:
        print("\nStrategy comparison interrupted by user")
    except Exception as e:
        print(f"\nError running strategy comparison: {e}")
        import traceback
        traceback.print_exc()