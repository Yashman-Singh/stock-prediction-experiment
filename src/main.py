import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from sklearn.preprocessing import MinMaxScaler
import logging
from tqdm import tqdm
import time
import tensorflow as tf

from data.data_collector import StockDataCollector
from models.lstm_attention import StockPredictionModel
from visualization.visualizer import StockVisualizer

# Configure logging with timestamp
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def prepare_data(df: pd.DataFrame, sequence_length: int = 60):
    """Prepare and scale the data for training."""
    logger.info("Starting data preparation...")
    
    # Calculate percentage changes for price-based features
    price_features = ['Close', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26']
    for col in price_features:
        df[f'{col}_pct_change'] = df[col].pct_change()
    
    # Select features
    features = [
        'Close_pct_change',  # Daily returns
        'Volume',
        'SMA_20_pct_change',
        'SMA_50_pct_change',
        'EMA_12_pct_change',
        'EMA_26_pct_change',
        'MACD',
        'RSI'
    ]
    
    # Target will be next day's return
    target_col = 'Close'
    df['target_return'] = df[target_col].pct_change().shift(-1)
    
    data = df[features].copy()
    target = df['target_return'].copy()
    
    # Handle missing values
    data = data.fillna(0)  # Replace NaN with 0 for returns
    target = target.fillna(0)
    
    # Scale the data
    logger.info("Scaling data...")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences with progress bar
    logger.info("Creating sequences...")
    X, y = [], []
    for i in tqdm(range(len(scaled_data) - sequence_length), 
                 desc="Creating sequences", 
                 unit="sequence"):
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(target.iloc[i + sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split into train, validation, and test sets
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    logger.info(f"Data preparation complete. Training set size: {len(X_train)}")
    return (X_train, y_train, X_val, y_val, X_test, y_test), scaler, df

class ProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nEpoch {epoch + 1}/{self.params['epochs']}")
        self.epoch_progress = tqdm(total=self.params['steps'], 
                                 desc="Training",
                                 unit="batch",
                                 leave=False)  # Don't leave progress bars

    def on_batch_end(self, batch, logs=None):
        self.epoch_progress.update(1)
        
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_progress.close()
        # Format metrics nicely
        metrics_str = " - ".join([
            f"{k}: {v:.4f}" if not k.startswith('val_') else f"val_{k[4:]}: {v:.4f}"
            for k, v in logs.items()
        ])
        print(f"Epoch {epoch + 1}: {metrics_str}")

def main():
    print("\n" + "="*50)
    print("Stock Market Prediction System")
    print("="*50 + "\n")
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    
    # Initialize components
    collector = StockDataCollector()
    visualizer = StockVisualizer()
    
    # Fetch data with timezone-aware dates
    symbol = "AAPL"  # Example with Apple stock
    end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(days=1000)  # Get about 3 years of data
    
    logger.info(f"Fetching data for {symbol}...")
    df = collector.fetch_stock_data(symbol, start_date, end_date)
    
    if df is None or df.empty:
        logger.error("Failed to fetch data")
        return
    
    logger.info(f"Successfully fetched {len(df)} days of data")
    
    # Plot technical indicators
    logger.info("Generating technical analysis plots...")
    visualizer.plot_technical_indicators(df, save_as="technical_indicators.png")
    
    # Prepare data
    (X_train, y_train, X_val, y_val, X_test, y_test), scaler, df = prepare_data(df)
    
    # Initialize and train model
    logger.info("Initializing model...")
    model = StockPredictionModel(
        sequence_length=60,
        n_features=8,  # Number of features we're using
        lstm_units=128,
        dropout_rate=0.2
    )
    
    logger.info("Starting model training...")
    progress_callback = ProgressCallback()
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=32,
        callbacks=[progress_callback]
    )
    
    # Plot training metrics
    logger.info("Generating training metrics visualization...")
    visualizer.plot_model_metrics(history.history, save_as="training_metrics.png")
    
    # Make predictions
    logger.info("Making predictions...")
    test_predictions = model.predict(X_test)
    
    # Calculate metrics
    metrics = model.evaluate(X_test, y_test)
    logger.info("\nModel Metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Get the dates for the test period
    test_dates = df.index[-(len(y_test)):]
    
    # Convert returns to cumulative returns
    actual_cumret = (1 + y_test).cumprod()
    pred_cumret = (1 + test_predictions.flatten()).cumprod()
    
    # Calculate actual prices from cumulative returns
    start_price = df['Close'].iloc[-(len(y_test) + 1)]
    actual_prices = start_price * actual_cumret
    predicted_prices = start_price * pred_cumret
    
    logger.info("Generating final visualizations...")
    visualizer.plot_prediction_vs_actual(
        actual_prices,
        predicted_prices,
        test_dates,
        title=f"{symbol} Stock Price Prediction",
        save_as="predictions.png"
    )
    
    # Create additional performance metrics
    metrics['directional_accuracy'] = np.mean(np.sign(y_test) == np.sign(test_predictions.flatten()))
    metrics['cumulative_return_actual'] = actual_cumret[-1] - 1
    metrics['cumulative_return_pred'] = pred_cumret[-1] - 1
    
    visualizer.create_performance_dashboard(
        actual_prices,
        predicted_prices,
        metrics,
        save_as="performance_dashboard.png"
    )
    
    print("\n" + "="*50)
    logger.info("Process completed successfully!")
    logger.info("Check the 'visualizations' directory for results.")
    print("="*50 + "\n")

if __name__ == "__main__":
    main() 