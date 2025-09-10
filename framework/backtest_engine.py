#!/usr/bin/env python3
"""
Backtesting Engine

Handles backtesting of trading strategies with realistic transaction costs,
slippage, and performance metrics calculation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class BacktestEngine:
    """
    Backtesting engine for trading strategies.
    """
    
    def __init__(self, 
                 initial_capital: float = 10000,
                 trading_cost_pct: float = 0.001,
                 slippage_pct: float = 0.0001):
        """
        Initialize backtesting engine.
        
        Args:
            initial_capital (float): Starting capital amount
            trading_cost_pct (float): Trading cost as percentage of trade value
            slippage_pct (float): Slippage as percentage (market impact)
        """
        self.initial_capital = initial_capital
        self.trading_cost_pct = trading_cost_pct
        self.slippage_pct = slippage_pct
    
    def backtest_strategy(self, 
                         price_data: pd.DataFrame, 
                         signals: pd.DataFrame,
                         strategy_name: str = "Strategy") -> Dict[str, Any]:
        """
        Perform backtesting of a trading strategy.
        
        Args:
            price_data (pd.DataFrame): Price data with Close column
            signals (pd.DataFrame): Trading signals with Signal and Probability columns
            strategy_name (str): Name of the strategy for reporting
        
        Returns:
            dict: Comprehensive backtesting results
        """
        print(f"Running backtest for {strategy_name}...")
        
        # Align price data and signals
        common_dates = price_data.index.intersection(signals.index)
        if len(common_dates) == 0:
            raise ValueError("No common dates between price data and signals")
        
        if len(common_dates) < 30:
            raise ValueError(f"Insufficient data for backtesting: {len(common_dates)} observations (minimum 30 required)")
        
        prices = price_data.loc[common_dates, 'Close'].copy()
        sigs = signals.loc[common_dates, 'Signal'].copy()
        probs = signals.loc[common_dates, 'Probability'].copy()
        
        # Validate data integrity
        if prices.isnull().any():
            print(f"Warning: {prices.isnull().sum()} missing price values detected, forward-filling")
            prices = prices.ffill()
        
        if sigs.isnull().any():
            print(f"Warning: {sigs.isnull().sum()} missing signal values detected, filling with 0")
            sigs = sigs.fillna(0)
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Align signals with returns (shift signals to avoid look-ahead bias)
        sigs_aligned = sigs.shift(1).dropna()
        probs_aligned = probs.shift(1).dropna()
        
        # Ensure all series have the same index
        common_index = returns.index.intersection(sigs_aligned.index)
        returns = returns.loc[common_index]
        sigs_aligned = sigs_aligned.loc[common_index]
        probs_aligned = probs_aligned.loc[common_index]
        
        # Identify trading events (signal changes)
        signal_changes = sigs_aligned.diff() != 0
        signal_changes.iloc[0] = sigs_aligned.iloc[0] == 1  # First trade if starting with buy
        
        # Calculate cash returns (3-month Treasury rate when not invested)
        cash_returns = self._calculate_cash_returns(price_data, common_index)
        
        # Calculate strategy returns (equity returns when invested, cash returns when not)
        strategy_returns = returns * sigs_aligned + cash_returns * (1 - sigs_aligned)
        
        # Apply trading costs when signals change (both buying and selling)
        trading_costs = signal_changes * self.trading_cost_pct
        
        # Apply slippage costs
        slippage_costs = signal_changes * self.slippage_pct
        
        # Total transaction costs
        total_transaction_costs = trading_costs + slippage_costs
        
        # Strategy returns after costs
        strategy_returns_net = strategy_returns - total_transaction_costs
        
        # Calculate cumulative returns
        buy_and_hold_cumulative = (1 + returns).cumprod()
        strategy_cumulative = (1 + strategy_returns_net).cumprod()
        
        # Calculate final values
        final_buy_and_hold = self.initial_capital * buy_and_hold_cumulative.iloc[-1]
        final_strategy = self.initial_capital * strategy_cumulative.iloc[-1]
        
        # Calculate basic metrics
        total_return_bh = (final_buy_and_hold - self.initial_capital) / self.initial_capital
        total_return_strategy = (final_strategy - self.initial_capital) / self.initial_capital
        
        # Calculate Sharpe ratios using 3-month Treasury as risk-free rate
        risk_free_daily = cash_returns.mean() if len(cash_returns) > 0 else 0.02/252
        sharpe_bh = self._calculate_sharpe_ratio(returns, risk_free_daily * 252)
        sharpe_strategy = self._calculate_sharpe_ratio(strategy_returns_net, risk_free_daily * 252)
        
        # Calculate maximum drawdown for strategy
        strategy_drawdown = self._calculate_drawdown(strategy_cumulative)
        benchmark_drawdown = self._calculate_drawdown(buy_and_hold_cumulative)
        
        # Calculate additional risk metrics
        volatility_strategy = strategy_returns_net.std() * np.sqrt(252)
        volatility_benchmark = returns.std() * np.sqrt(252)
        
        # Calculate win rate and other trading metrics
        trading_returns = strategy_returns_net[signal_changes]  # Returns only on trading days
        winning_trades = (trading_returns > 0).sum()
        total_trades = signal_changes.sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate average holding period
        positions = sigs_aligned.copy()
        position_changes = positions.diff().abs()
        if position_changes.sum() > 0:
            avg_holding_period = len(positions) / position_changes.sum()
        else:
            avg_holding_period = len(positions)
        
        # Calculate total costs
        total_trading_costs = signal_changes.sum() * self.trading_cost_pct
        total_slippage_costs = signal_changes.sum() * self.slippage_pct
        total_costs = total_trading_costs + total_slippage_costs
        
        # Calculate monthly statistics
        monthly_returns_strategy = self._calculate_monthly_returns(strategy_returns_net)
        monthly_returns_benchmark = self._calculate_monthly_returns(returns)
        
        # Calculate correlation with error handling
        try:
            if len(monthly_returns_strategy) >= 12 and len(monthly_returns_benchmark) >= 12:
                correlation = np.corrcoef(monthly_returns_strategy, monthly_returns_benchmark)[0, 1]
                if np.isnan(correlation) or np.isinf(correlation):
                    correlation = 0.0
            else:
                print("Warning: Insufficient monthly data for correlation calculation")
                correlation = 0.0
        except Exception as e:
            print(f"Warning: Correlation calculation failed: {e}")
            correlation = 0.0
        
        # Beta calculation (strategy vs benchmark) with error handling
        try:
            if len(strategy_returns_net) >= 30 and len(returns) >= 30:
                covariance = np.cov(strategy_returns_net, returns)[0, 1]
                benchmark_variance = np.var(returns)
                if benchmark_variance > 1e-10:  # Avoid division by very small numbers
                    beta = covariance / benchmark_variance
                    # Clip extreme beta values
                    beta = np.clip(beta, -5.0, 5.0)
                else:
                    print("Warning: Near-zero benchmark variance, setting beta to 1.0")
                    beta = 1.0
            else:
                print("Warning: Insufficient data for beta calculation")
                beta = 1.0
        except Exception as e:
            print(f"Warning: Beta calculation failed: {e}")
            beta = 1.0
        
        # Alpha calculation (annualized)
        risk_free_rate = risk_free_daily * 252  # Use actual Treasury rate
        # Calculate actual time period in years for proper annualization
        years = len(returns) / 252.0
        strategy_annual_return = (1 + total_return_strategy) ** (1/years) - 1 if years > 0 else 0
        benchmark_annual_return = (1 + total_return_bh) ** (1/years) - 1 if years > 0 else 0
        alpha = strategy_annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))
        
        # Compile results
        results = {
            # Basic metrics
            'strategy_name': strategy_name,
            'initial_capital': self.initial_capital,
            'final_value_strategy': final_strategy,
            'final_value_benchmark': final_buy_and_hold,
            'total_return_strategy': total_return_strategy,
            'total_return_benchmark': total_return_bh,
            
            # Annualized metrics
            'annualized_return_strategy': strategy_annual_return,
            'annualized_return_benchmark': benchmark_annual_return,
            'annualized_volatility_strategy': volatility_strategy,
            'annualized_volatility_benchmark': volatility_benchmark,
            
            # Risk metrics
            'sharpe_ratio_strategy': sharpe_strategy,
            'sharpe_ratio_benchmark': sharpe_bh,
            'max_drawdown_strategy': strategy_drawdown,
            'max_drawdown_benchmark': benchmark_drawdown,
            'beta': beta,
            'alpha': alpha,
            'correlation': correlation,
            
            # Trading metrics
            'total_trades': int(total_trades),
            'winning_trades': int(winning_trades),
            'win_rate': win_rate,
            'avg_holding_period_days': avg_holding_period,
            
            # Cost metrics
            'total_trading_costs_pct': total_trading_costs,
            'total_slippage_costs_pct': total_slippage_costs,
            'total_costs_pct': total_costs,
            'trading_cost_per_trade_pct': self.trading_cost_pct,
            'slippage_per_trade_pct': self.slippage_pct,
            
            # Time series data for plotting
            'dates': common_index,
            'strategy_cumulative_returns': strategy_cumulative,
            'benchmark_cumulative_returns': buy_and_hold_cumulative,
            'strategy_returns': strategy_returns_net,
            'benchmark_returns': returns,
            'signals': sigs_aligned,
            'probabilities': probs_aligned,
            'monthly_returns_strategy': monthly_returns_strategy,
            'monthly_returns_benchmark': monthly_returns_benchmark,
            
            # Trading periods breakdown
            'trading_periods': len(common_index),
            'signal_rate': sigs_aligned.mean(),  # Percentage of time in market
            'risk_free_rate': risk_free_rate,  # Annualized risk-free rate used
            'avg_cash_return_daily': cash_returns.mean(),  # Average daily cash return
        }
        
        # Print summary
        self._print_backtest_summary(results)
        
        return results
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio with robust error handling."""
        if len(returns) < 30:  # Minimum 30 observations
            print(f"Warning: Insufficient data for Sharpe ratio ({len(returns)} observations)")
            return 0.0
        
        if returns.std() == 0 or np.isnan(returns.std()):
            print("Warning: Zero or NaN volatility detected, returning 0 Sharpe ratio")
            return 0.0
        
        # Remove outliers (clip extreme returns)
        returns_clean = returns.clip(lower=returns.quantile(0.01), 
                                   upper=returns.quantile(0.99))
        
        if returns_clean.std() == 0:
            return 0.0
        
        excess_returns = returns_clean.mean() - risk_free_rate / 252  # Daily risk-free rate
        sharpe = np.sqrt(252) * excess_returns / returns_clean.std()
        
        # Handle extreme values
        return np.clip(sharpe, -10.0, 10.0) if not np.isnan(sharpe) else 0.0
    
    def _calculate_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - cumulative_max) / cumulative_max
        return drawdown.min()
    
    def _calculate_monthly_returns(self, daily_returns: pd.Series) -> pd.Series:
        """Calculate monthly returns from daily returns."""
        return (1 + daily_returns).resample('M').prod() - 1
    
    def _calculate_cash_returns(self, price_data: pd.DataFrame, common_index: pd.DatetimeIndex) -> pd.Series:
        """
        Calculate cash returns using 3-month Treasury bill rate.
        
        Args:
            price_data (pd.DataFrame): Price data containing TB3MS column
            common_index (pd.DatetimeIndex): Index to align cash returns with
        
        Returns:
            pd.Series: Daily cash returns aligned with common_index
        """
        # Check if TB3MS column exists in price_data
        if 'TB3MS' in price_data.columns:
            # Get Treasury bill rates aligned with our index
            tb3ms_data = price_data.loc[common_index, 'TB3MS'].copy()
            
            # Handle missing values with forward fill, then backward fill
            tb3ms_data = tb3ms_data.ffill().bfill()
            
            # Validate Treasury rate values (should be between -5% and 20%)
            tb3ms_data = tb3ms_data.clip(lower=-5.0, upper=20.0)
            
            # Convert annual percentage to daily returns using calendar days
            # Treasury bills are quoted on an annualized basis using 365 days
            daily_cash_returns = (tb3ms_data / 100) / 365
            
        else:
            # Fallback to constant 2% annual rate if TB3MS not available
            print("Warning: TB3MS data not available, using 2% constant cash rate")
            daily_cash_returns = pd.Series(0.02 / 365, index=common_index)
        
        return daily_cash_returns

    def _print_backtest_summary(self, results: Dict[str, Any]):
        """Print backtest summary."""
        print(f"\n=== {results['strategy_name']} Backtest Results ===")
        print(f"Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"Final Value: ${results['final_value_strategy']:,.2f}")
        print(f"Benchmark Final Value: ${results['final_value_benchmark']:,.2f}")
        print(f"Total Return: {results['total_return_strategy']:.1%}")
        print(f"Benchmark Return: {results['total_return_benchmark']:.1%}")
        print(f"Annualized Return: {results['annualized_return_strategy']:.1%}")
        print(f"Annualized Volatility: {results['annualized_volatility_strategy']:.1%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio_strategy']:.3f}")
        print(f"Max Drawdown: {results['max_drawdown_strategy']:.1%}")
        print(f"Alpha: {results['alpha']:.1%}")
        print(f"Beta: {results['beta']:.2f}")
        print(f"Correlation: {results['correlation']:.3f}")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.1%}")
        print(f"Total Costs: {results['total_costs_pct']:.2%}")
        print(f"Signal Rate: {results['signal_rate']:.1%} (% time in market)")
        print(f"Risk-Free Rate: {results['risk_free_rate']:.1%} (cash return when not invested)")


class StrategyComparison:
    """
    Class for comparing multiple strategies.
    """
    
    def __init__(self, backtest_engine: BacktestEngine):
        """
        Initialize strategy comparison.
        
        Args:
            backtest_engine (BacktestEngine): Configured backtest engine
        """
        self.backtest_engine = backtest_engine
        self.results = {}
    
    def add_strategy_result(self, strategy_name: str, result: Dict[str, Any]):
        """
        Add a strategy result to the comparison.
        
        Args:
            strategy_name (str): Name of the strategy
            result (dict): Backtest result dictionary
        """
        self.results[strategy_name] = result
    
    def run_comparison(self, price_data: pd.DataFrame, strategies: Dict[str, Any]) -> pd.DataFrame:
        """
        Run backtest comparison for multiple strategies.
        
        Args:
            price_data (pd.DataFrame): Price data
            strategies (dict): Dictionary of strategy_name -> signals DataFrame
        
        Returns:
            pd.DataFrame: Comparison results table
        """
        print("Running strategy comparison...")
        
        for strategy_name, signals in strategies.items():
            result = self.backtest_engine.backtest_strategy(
                price_data, signals, strategy_name
            )
            self.add_strategy_result(strategy_name, result)
        
        return self.get_comparison_table()
    
    def get_comparison_table(self) -> pd.DataFrame:
        """
        Get comparison table of all strategies.
        
        Returns:
            pd.DataFrame: Comparison table with key metrics
        """
        if not self.results:
            return pd.DataFrame()
        
        comparison_data = []
        
        for strategy_name, result in self.results.items():
            comparison_data.append({
                'Strategy': strategy_name,
                'Total Return': f"{result['total_return_strategy']:.1%}",
                'Annualized Return': f"{result['annualized_return_strategy']:.1%}",
                'Volatility': f"{result['annualized_volatility_strategy']:.1%}",
                'Sharpe Ratio': f"{result['sharpe_ratio_strategy']:.3f}",
                'Max Drawdown': f"{result['max_drawdown_strategy']:.1%}",
                'Alpha': f"{result['alpha']:.1%}",
                'Beta': f"{result['beta']:.2f}",
                'Win Rate': f"{result['win_rate']:.1%}",
                'Total Trades': result['total_trades'],
                'Total Costs': f"{result['total_costs_pct']:.2%}",
                'Signal Rate': f"{result['signal_rate']:.1%}",
                'Final Value': f"${result['final_value_strategy']:,.0f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df.set_index('Strategy')
    
    def get_best_strategy(self, metric: str = 'sharpe_ratio_strategy') -> str:
        """
        Get the best performing strategy based on a specific metric.
        
        Args:
            metric (str): Metric to use for comparison
        
        Returns:
            str: Name of the best performing strategy
        """
        if not self.results:
            return None
        
        best_strategy = max(self.results.items(), key=lambda x: x[1].get(metric, 0))
        return best_strategy[0]
    
    def print_summary(self):
        """Print comparison summary."""
        if not self.results:
            print("No strategies to compare")
            return
        
        print("\n" + "="*80)
        print("STRATEGY COMPARISON SUMMARY")
        print("="*80)
        
        comparison_table = self.get_comparison_table()
        print(comparison_table.to_string())
        
        print(f"\nBest Strategy by Sharpe Ratio: {self.get_best_strategy('sharpe_ratio_strategy')}")
        print(f"Best Strategy by Total Return: {self.get_best_strategy('total_return_strategy')}")
        print(f"Best Strategy by Alpha: {self.get_best_strategy('alpha')}")
        
        print("="*80)