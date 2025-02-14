import yfinance as yf
import pandas as pd
import logging
from typing import List, Optional, Dict
import time
from datetime import datetime, timedelta
import pytz
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataCollector:
    """A  data collection system for fetching stock market data."""
    
    def __init__(self, cache_dir: str = "data/raw"):
        """Initialize the data collector with caching capability."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_stock_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch stock data with  error handling and rate limiting.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date for data collection
            end_date: End date for data collection
            interval: Data interval ('1d', '1h', '15m', etc.)
            
        Returns:
            DataFrame containing stock data or None if fetch fails
        """
        try:
            # Default to last 2 years if no dates specified
            if not end_date:
                end_date = datetime.now(pytz.UTC)
            if not start_date:
                start_date = end_date - timedelta(days=730)
                
            # Ensure dates are timezone-aware
            if not start_date.tzinfo:
                start_date = pytz.UTC.localize(start_date)
            if not end_date.tzinfo:
                end_date = pytz.UTC.localize(end_date)
            
            cache_file = self.cache_dir / f"{symbol}_{interval}.parquet"
            
            # Check cache first
            if cache_file.exists():
                df = pd.read_parquet(cache_file)
                if not self._needs_update(df, end_date):
                    logger.info(f"Using cached data for {symbol}")
                    return df
            
            # Fetch data with retry mechanism
            for attempt in range(3):
                try:
                    stock = yf.Ticker(symbol)
                    df = stock.history(
                        start=start_date,
                        end=end_date,
                        interval=interval
                    )
                    
                    if df.empty:
                        logger.warning(f"No data available for {symbol}")
                        return None
                    
                    # Add technical indicators
                    df = self._add_technical_indicators(df)
                    
                    # Cache the data
                    df.to_parquet(cache_file)
                    
                    logger.info(f"Successfully fetched data for {symbol}")
                    return df
                
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}")
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def fetch_multiple_stocks(
        self,
        symbols: List[str],
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple stocks with rate limiting."""
        results = {}
        for symbol in symbols:
            df = self.fetch_stock_data(symbol, **kwargs)
            if df is not None:
                results[symbol] = df
            time.sleep(1)  # Rate limiting
        return results
    
    def _needs_update(self, df: pd.DataFrame, current_date: datetime) -> bool:
        """Check if cached data needs updating."""
        if df.empty:
            return True
        last_date = pd.to_datetime(df.index[-1])
        return (current_date - last_date).days > 1
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators to the dataset."""
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df

if __name__ == "__main__":
    # Example usage
    collector = StockDataCollector()
    data = collector.fetch_stock_data("AAPL")
    if data is not None:
        print(data.head())
        print(f"Fetched {len(data)} rows of data") 