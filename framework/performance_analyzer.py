#!/usr/bin/env python3
"""
Performance Analysis and Visualization Module

Handles plotting and analysis of trading strategy performance,
including comparison plots, risk analysis, and statistical reporting.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend by default
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class PerformanceAnalyzer:
    """
    Performance analysis and visualization for trading strategies.
    """
    
    def __init__(self, interactive_plots: bool = False):
        """
        Initialize performance analyzer.
        
        Args:
            interactive_plots (bool): Whether to use interactive plotting backend
        """
        self.interactive_plots = interactive_plots
        if interactive_plots:
            try:
                matplotlib.use('TkAgg')
            except ImportError:
                print("Interactive backend not available, using file output")
                matplotlib.use('Agg')
    
    def plot_single_strategy_analysis(self, 
                                    result: Dict[str, Any], 
                                    save_path: Optional[str] = None) -> str:
        """
        Create comprehensive analysis plots for a single strategy.
        
        Args:
            result (dict): Backtest result dictionary
            save_path (str): Path to save the plot (optional)
        
        Returns:
            str: Path to saved plot or status message
        """
        strategy_name = result.get('strategy_name', 'Strategy')
        print(f"Creating performance analysis for {strategy_name}...")
        
        # Set up the plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{strategy_name} Performance Analysis', fontsize=16, fontweight='bold')
        
        # Get data
        dates = result['dates']
        strategy_cum = result['strategy_cumulative_returns']
        benchmark_cum = result['benchmark_cumulative_returns']
        
        # 1. Cumulative Returns (Log2 Scale)
        ax1.plot(dates, strategy_cum, label=strategy_name, color='blue', linewidth=2)
        ax1.plot(dates, benchmark_cum, label='S&P 500 (Buy & Hold)', color='red', linewidth=2)
        ax1.set_yscale('log', base=2)
        ax1.set_ylabel('Cumulative Return (Log₂ Scale)', fontweight='bold')
        ax1.set_title('Cumulative Returns Comparison', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Format y-axis to show readable values
        def log2_formatter(x, pos):
            return f'{x:.1f}x'
        ax1.yaxis.set_major_formatter(FuncFormatter(log2_formatter))
        
        # Format x-axis dates
        ax1.xaxis.set_major_locator(mdates.YearLocator(2))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. Drawdown Comparison
        strategy_max = strategy_cum.expanding().max()
        benchmark_max = benchmark_cum.expanding().max()
        
        strategy_drawdown = (strategy_cum - strategy_max) / strategy_max * 100
        benchmark_drawdown = (benchmark_cum - benchmark_max) / benchmark_max * 100
        
        ax2.fill_between(dates, strategy_drawdown, 0, label=strategy_name, color='blue', alpha=0.7)
        ax2.fill_between(dates, benchmark_drawdown, 0, label='S&P 500', color='red', alpha=0.7)
        ax2.set_ylabel('Drawdown (%)', fontweight='bold')
        ax2.set_title('Drawdown Comparison', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax2.xaxis.set_major_locator(mdates.YearLocator(2))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Monthly Returns Scatter Plot
        monthly_strategy = result['monthly_returns_strategy'] * 100
        monthly_benchmark = result['monthly_returns_benchmark'] * 100
        
        # Align monthly returns
        common_months = monthly_strategy.index.intersection(monthly_benchmark.index)
        monthly_strategy_aligned = monthly_strategy.loc[common_months]
        monthly_benchmark_aligned = monthly_benchmark.loc[common_months]
        
        ax3.scatter(monthly_benchmark_aligned, monthly_strategy_aligned, 
                   alpha=0.6, color='green', s=20)
        
        # Add diagonal line (perfect correlation)
        if len(monthly_strategy_aligned) > 0 and len(monthly_benchmark_aligned) > 0:
            min_val = min(monthly_benchmark_aligned.min(), monthly_strategy_aligned.min())
            max_val = max(monthly_benchmark_aligned.max(), monthly_strategy_aligned.max())
            ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Correlation')
        
        ax3.set_xlabel('S&P 500 Monthly Return (%)', fontweight='bold')
        ax3.set_ylabel('Strategy Monthly Return (%)', fontweight='bold')
        ax3.set_title('Monthly Returns Correlation', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add correlation info
        correlation = result.get('correlation', 0)
        ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax3.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. Performance Summary Statistics
        ax4.axis('off')
        
        # Calculate additional metrics for display
        years = len(strategy_cum) / 252
        
        summary_text = f"""
        PERFORMANCE SUMMARY
        ════════════════════════════════
        
        Period: {years:.1f} years
        
        RETURNS:
        • Total Return: {result['total_return_strategy']:.1%}
        • Annualized Return: {result['annualized_return_strategy']:.1%}
        • S&P 500 Total Return: {result['total_return_benchmark']:.1%}
        • S&P 500 Annualized: {result['annualized_return_benchmark']:.1%}
        
        RISK METRICS:
        • Sharpe Ratio: {result['sharpe_ratio_strategy']:.3f}
        • Max Drawdown: {result['max_drawdown_strategy']:.1%}
        • Volatility: {result['annualized_volatility_strategy']:.1%}
        • Alpha: {result['alpha']:.1%}
        • Beta: {result['beta']:.2f}
        
        TRADING:
        • Total Trades: {result['total_trades']}
        • Win Rate: {result['win_rate']:.1%}
        • Avg Holding: {result['avg_holding_period_days']:.1f} days
        • Total Costs: {result['total_costs_pct']:.2%}
        • Signal Rate: {result['signal_rate']:.1%}
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save or show plot
        if save_path is None:
            save_path = f'{strategy_name.lower().replace(" ", "_")}_analysis.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Single strategy analysis saved as '{save_path}'")
        
        if self.interactive_plots:
            plt.show()
        else:
            plt.close()
        
        return save_path
    
    def plot_strategy_comparison(self, 
                               results: Dict[str, Dict[str, Any]], 
                               save_path: Optional[str] = None,
                               top_n: int = 10) -> str:
        """
        Create comparison plots for multiple strategies.
        
        Args:
            results (dict): Dictionary of strategy_name -> result dict
            save_path (str): Path to save the plot (optional)
            top_n (int): Number of top strategies to show in line plots (default: 10)
        
        Returns:
            str: Path to saved plot
        """
        print("Creating strategy comparison plots...")
        
        if len(results) < 2:
            raise ValueError("Need at least 2 strategies to compare")
        
        # Filter to top N strategies by Sharpe ratio for line plots
        sorted_results = sorted(results.items(), 
                               key=lambda x: x[1]['sharpe_ratio_strategy'], 
                               reverse=True)
        top_results = dict(sorted_results[:top_n])
        
        print(f"Showing top {len(top_results)} strategies in line plots (by Sharpe ratio)")
        
        # Set up the plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Strategy Comparison Analysis (Top {len(top_results)} by Sharpe Ratio)', 
                     fontsize=16, fontweight='bold')
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        
        # 1. Cumulative Returns Comparison (Top N only)
        for i, (strategy_name, result) in enumerate(top_results.items()):
            dates = result['dates']
            strategy_cum = result['strategy_cumulative_returns']
            color = colors[i % len(colors)]
            
            ax1.plot(dates, strategy_cum, label=strategy_name, 
                    color=color, linewidth=2)
        
        ax1.set_ylabel('Cumulative Return', fontweight='bold')
        ax1.set_title('Cumulative Returns Comparison', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_locator(mdates.YearLocator(2))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. Drawdown Comparison (Top N only)
        for i, (strategy_name, result) in enumerate(top_results.items()):
            dates = result['dates']
            strategy_cum = result['strategy_cumulative_returns']
            strategy_max = strategy_cum.expanding().max()
            strategy_drawdown = (strategy_cum - strategy_max) / strategy_max * 100
            color = colors[i % len(colors)]
            
            ax2.plot(dates, strategy_drawdown, label=strategy_name, 
                    color=color, linewidth=2)
        
        ax2.set_ylabel('Drawdown (%)', fontweight='bold')
        ax2.set_title('Drawdown Comparison', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_locator(mdates.YearLocator(2))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Risk-Return Scatter Plot
        returns = []
        risks = []
        sharpes = []
        names = []
        
        for strategy_name, result in results.items():
            returns.append(result['annualized_return_strategy'])
            risks.append(result['annualized_volatility_strategy'])
            sharpes.append(result['sharpe_ratio_strategy'])
            names.append(strategy_name)
        
        scatter = ax3.scatter(risks, returns, c=sharpes, s=100, cmap='RdYlGn', 
                             alpha=0.8, edgecolors='black')
        
        # Add strategy labels
        for i, name in enumerate(names):
            ax3.annotate(name, (risks[i], returns[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax3.set_xlabel('Annualized Volatility', fontweight='bold')
        ax3.set_ylabel('Annualized Return', fontweight='bold')
        ax3.set_title('Risk-Return Profile (colored by Sharpe Ratio)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add colorbar for Sharpe ratio
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Sharpe Ratio', fontweight='bold')
        
        # 4. Performance Metrics Comparison Table
        ax4.axis('off')
        
        # Create comparison table data sorted by annualized return
        table_data = []
        headers = ['Strategy', 'Return', 'Volatility', 'Sharpe', 'Max DD', 'Alpha', 'Trades']
        
        # Sort strategies by annualized return (descending)
        sorted_results = sorted(results.items(), 
                               key=lambda x: x[1]['annualized_return_strategy'], 
                               reverse=True)
        
        for strategy_name, result in sorted_results:
            table_data.append([
                strategy_name,  # Show full strategy name
                f"{result['annualized_return_strategy']:.1%}",
                f"{result['annualized_volatility_strategy']:.1%}",
                f"{result['sharpe_ratio_strategy']:.2f}",
                f"{result['max_drawdown_strategy']:.1%}",
                f"{result['alpha']:.1%}",
                f"{result['total_trades']}"
            ])
        
        # Create table with custom column widths
        table = ax4.table(cellText=table_data, colLabels=headers,
                         cellLoc='left', loc='center', 
                         bbox=[0.05, 0.1, 0.9, 0.8])
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.5)
        
        # Set custom column widths - make Strategy column wider
        cellDict = table.get_celld()
        for i in range(len(table_data) + 1):  # +1 for header
            # Strategy column (wider)
            cellDict[(i, 0)].set_width(0.35)
            # Other columns (narrower)
            for j in range(1, len(headers)):
                cellDict[(i, j)].set_width(0.11)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax4.set_title('Performance Metrics Comparison', fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = 'strategy_comparison.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Strategy comparison saved as '{save_path}'")
        
        if self.interactive_plots:
            plt.show()
        else:
            plt.close()
        
        return save_path
    
    def create_rolling_metrics_plot(self, 
                                  results: Dict[str, Dict[str, Any]], 
                                  window_days: int = 252,
                                  save_path: Optional[str] = None,
                                  top_n: int = 10) -> str:
        """
        Create rolling performance metrics plot.
        
        Args:
            results (dict): Dictionary of strategy results
            window_days (int): Rolling window size in days
            save_path (str): Path to save the plot
            top_n (int): Number of top strategies to show (default: 10)
        
        Returns:
            str: Path to saved plot
        """
        print(f"Creating rolling metrics plot (window: {window_days} days)...")
        
        # Filter to top N strategies by Sharpe ratio
        sorted_results = sorted(results.items(), 
                               key=lambda x: x[1]['sharpe_ratio_strategy'], 
                               reverse=True)
        top_results = dict(sorted_results[:top_n])
        
        print(f"Showing top {len(top_results)} strategies in rolling metrics plot")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Rolling Performance Metrics ({window_days}-Day Window, Top {len(top_results)} Strategies)', 
                     fontsize=16, fontweight='bold')
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        
        for i, (strategy_name, result) in enumerate(top_results.items()):
            dates = result['dates']
            returns = result['strategy_returns']
            color = colors[i % len(colors)]
            
            # Rolling returns
            rolling_returns = returns.rolling(window_days).mean() * 252
            ax1.plot(dates, rolling_returns, label=strategy_name, color=color, linewidth=2)
            
            # Rolling volatility
            rolling_vol = returns.rolling(window_days).std() * np.sqrt(252)
            ax2.plot(dates, rolling_vol, label=strategy_name, color=color, linewidth=2)
            
            # Rolling Sharpe ratio
            rolling_sharpe = rolling_returns / rolling_vol
            ax3.plot(dates, rolling_sharpe, label=strategy_name, color=color, linewidth=2)
            
            # Rolling drawdown
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.rolling(window_days, min_periods=1).max()
            rolling_dd = (cumulative - rolling_max) / rolling_max * 100
            ax4.plot(dates, rolling_dd, label=strategy_name, color=color, linewidth=2)
        
        # Format all subplots
        for ax, title, ylabel in zip([ax1, ax2, ax3, ax4], 
                                   ['Rolling Returns', 'Rolling Volatility', 
                                    'Rolling Sharpe Ratio', 'Rolling Drawdown'],
                                   ['Annualized Return', 'Annualized Volatility', 
                                    'Sharpe Ratio', 'Drawdown (%)']):
            ax.set_title(title, fontweight='bold')
            ax.set_ylabel(ylabel, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f'rolling_metrics_{window_days}d.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Rolling metrics plot saved as '{save_path}'")
        
        if self.interactive_plots:
            plt.show()
        else:
            plt.close()
        
        return save_path
    
    def generate_performance_report(self, 
                                  results: Dict[str, Dict[str, Any]], 
                                  save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive performance report.
        
        Args:
            results (dict): Dictionary of strategy results
            save_path (str): Path to save the report
        
        Returns:
            str: Path to saved report
        """
        if save_path is None:
            save_path = 'performance_report.txt'
        
        print(f"Generating performance report...")
        
        with open(save_path, 'w') as f:
            f.write("TRADING STRATEGY PERFORMANCE REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Summary table
            f.write("STRATEGY SUMMARY\n")
            f.write("-" * 40 + "\n")
            
            headers = ["Strategy", "Return", "Volatility", "Sharpe", "Max DD", "Alpha", "Beta", "Trades"]
            f.write(f"{'Strategy':<15} {'Return':<8} {'Vol':<8} {'Sharpe':<8} {'MaxDD':<8} {'Alpha':<8} {'Beta':<6} {'Trades':<8}\n")
            f.write("-" * 80 + "\n")
            
            for strategy_name, result in results.items():
                f.write(f"{strategy_name:<15} "
                       f"{result['annualized_return_strategy']:>7.1%} "
                       f"{result['annualized_volatility_strategy']:>7.1%} "
                       f"{result['sharpe_ratio_strategy']:>7.2f} "
                       f"{result['max_drawdown_strategy']:>7.1%} "
                       f"{result['alpha']:>7.1%} "
                       f"{result['beta']:>5.2f} "
                       f"{result['total_trades']:>7d}\n")
            
            f.write("\n" + "=" * 60 + "\n\n")
            
            # Detailed analysis for each strategy
            for strategy_name, result in results.items():
                f.write(f"DETAILED ANALYSIS: {strategy_name.upper()}\n")
                f.write("-" * 40 + "\n")
                
                f.write(f"Return Metrics:\n")
                f.write(f"  Total Return: {result['total_return_strategy']:.1%}\n")
                f.write(f"  Annualized Return: {result['annualized_return_strategy']:.1%}\n")
                f.write(f"  Benchmark Return: {result['total_return_benchmark']:.1%}\n")
                f.write(f"  Excess Return: {result['total_return_strategy'] - result['total_return_benchmark']:.1%}\n")
                
                f.write(f"\nRisk Metrics:\n")
                f.write(f"  Volatility: {result['annualized_volatility_strategy']:.1%}\n")
                f.write(f"  Sharpe Ratio: {result['sharpe_ratio_strategy']:.3f}\n")
                f.write(f"  Max Drawdown: {result['max_drawdown_strategy']:.1%}\n")
                f.write(f"  Alpha: {result['alpha']:.1%}\n")
                f.write(f"  Beta: {result['beta']:.2f}\n")
                
                f.write(f"\nTrading Metrics:\n")
                f.write(f"  Total Trades: {result['total_trades']}\n")
                f.write(f"  Win Rate: {result['win_rate']:.1%}\n")
                f.write(f"  Avg Holding Period: {result['avg_holding_period_days']:.1f} days\n")
                f.write(f"  Signal Rate: {result['signal_rate']:.1%}\n")
                f.write(f"  Total Costs: {result['total_costs_pct']:.2%}\n")
                
                f.write(f"\nPortfolio Metrics:\n")
                f.write(f"  Final Value: ${result['final_value_strategy']:,.0f}\n")
                f.write(f"  Correlation with Benchmark: {result['correlation']:.3f}\n")
                
                f.write("\n" + "=" * 40 + "\n\n")
        
        print(f"Performance report saved as '{save_path}'")
        return save_path