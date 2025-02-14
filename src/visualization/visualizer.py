import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path

class StockVisualizer:
    """Advanced visualization tools for stock prediction analysis."""
    
    def __init__(self, save_dir: str = "visualizations"):
        """Initialize the visualizer with a save directory."""
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set basic style settings that work across versions
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 8)
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    def plot_prediction_vs_actual(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        dates: pd.DatetimeIndex,
        title: str = "Stock Price Prediction vs Actual",
        save_as: Optional[str] = None
    ):
        """Create a professional visualization of predictions vs actual values."""
        plt.figure(figsize=(15, 8))
        
        # Plot actual and predicted values
        plt.plot(dates, actual, label='Actual', linewidth=2)
        plt.plot(dates, predicted, label='Predicted', linewidth=2, linestyle='--')
        
        # Add confidence interval
        std = np.std(actual - predicted)
        plt.fill_between(
            dates,
            predicted - 2*std,
            predicted + 2*std,
            alpha=0.2,
            label='95% Confidence Interval'
        )
        
        # Customize the plot
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Stock Price', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Tight layout to prevent label cutoff
        plt.tight_layout()
        
        if save_as:
            plt.savefig(self.save_dir / save_as, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_technical_indicators(
        self,
        df: pd.DataFrame,
        save_as: Optional[str] = None
    ):
        """Create a multi-panel plot of technical indicators."""
        fig, axes = plt.subplots(4, 1, figsize=(15, 20))
        fig.suptitle('Technical Analysis Dashboard', fontsize=16, y=0.92)
        
        # Price and Moving Averages
        axes[0].plot(df.index, df['Close'], label='Close Price', linewidth=2)
        axes[0].plot(df.index, df['SMA_20'], label='20-day SMA', linestyle='--')
        axes[0].plot(df.index, df['SMA_50'], label='50-day SMA', linestyle='--')
        axes[0].set_title('Price and Moving Averages')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Volume
        axes[1].bar(df.index, df['Volume'], alpha=0.7)
        axes[1].set_title('Trading Volume')
        axes[1].grid(True, alpha=0.3)
        
        # MACD
        axes[2].plot(df.index, df['MACD'], label='MACD')
        axes[2].plot(df.index, df['MACD_Signal'], label='Signal Line')
        axes[2].bar(df.index, df['MACD'] - df['MACD_Signal'], alpha=0.3, label='MACD Histogram')
        axes[2].set_title('MACD')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # RSI
        axes[3].plot(df.index, df['RSI'])
        axes[3].axhline(y=70, color='r', linestyle='--', alpha=0.5)
        axes[3].axhline(y=30, color='g', linestyle='--', alpha=0.5)
        axes[3].set_title('RSI')
        axes[3].grid(True, alpha=0.3)
        
        # Customize all subplots
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_as:
            plt.savefig(self.save_dir / save_as, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_model_metrics(
        self,
        history: Dict,
        save_as: Optional[str] = None
    ):
        """Plot training history and metrics."""
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        
        # Loss plot
        axes[0].plot(history['loss'], label='Training Loss')
        axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss Over Time')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Metrics plot
        axes[1].plot(history['mae'], label='MAE')
        axes[1].plot(history['val_mae'], label='Validation MAE')
        axes[1].plot(history['mape'], label='MAPE')
        axes[1].plot(history['val_mape'], label='Validation MAPE')
        axes[1].set_title('Model Metrics Over Time')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Metric Value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_as:
            plt.savefig(self.save_dir / save_as, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def create_performance_dashboard(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        metrics: Dict,
        save_as: Optional[str] = None
    ):
        """Create a comprehensive performance dashboard."""
        fig = plt.figure(figsize=(15, 10))
        
        # Create grid for subplots
        gs = fig.add_gridspec(2, 2)
        
        # Prediction vs Actual
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(actual, label='Actual', linewidth=2)
        ax1.plot(predicted, label='Predicted', linewidth=2, linestyle='--')
        ax1.set_title('Prediction vs Actual')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.scatter(actual, predicted, alpha=0.5)
        ax2.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', linewidth=2)
        ax2.set_title('Prediction Scatter Plot')
        ax2.set_xlabel('Actual')
        ax2.set_ylabel('Predicted')
        ax2.grid(True, alpha=0.3)
        
        # Metrics table
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')
        metrics_table = ax3.table(
            cellText=[[f"{v:.4f}"] for v in metrics.values()],
            rowLabels=list(metrics.keys()),
            colLabels=["Value"],
            cellLoc='center',
            loc='center'
        )
        metrics_table.auto_set_font_size(False)
        metrics_table.set_fontsize(9)
        metrics_table.scale(1.2, 1.5)
        
        plt.tight_layout()
        
        if save_as:
            plt.savefig(self.save_dir / save_as, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show() 