import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
import joblib
from typing import Any, Dict, List, Optional
import hashlib
import logging
from pathlib import Path

def setup_logger(name: str, log_file: str = None, level: str = "INFO") -> logging.Logger:
    """
    Setup logger with file and console handlers
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger

def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate percentage returns from price series
    """
    return prices.pct_change()

def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate log returns from price series
    """
    return np.log(prices / prices.shift(1))

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return (returns.mean() - risk_free_rate) / returns.std()

def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sortino ratio
    """
    negative_returns = returns[returns < 0]
    if len(negative_returns) == 0 or negative_returns.std() == 0:
        return 0.0
    downside_std = negative_returns.std()
    return (returns.mean() - risk_free_rate) / downside_std

def calculate_calmar_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Calmar ratio
    """
    max_dd = abs(calculate_max_drawdown(returns))
    if max_dd == 0:
        return 0.0
    return (returns.mean() - risk_free_rate) / max_dd

def create_features_lags(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
    """
    Create lag features for specified columns
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            for lag in lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df

def create_features_rolling(df: pd.DataFrame, columns: List[str], windows: List[int]) -> pd.DataFrame:
    """
    Create rolling window features
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            for window in windows:
                df[f"{col}_rolling_mean_{window}"] = df[col].rolling(window).mean()
                df[f"{col}_rolling_std_{window}"] = df[col].rolling(window).std()
                df[f"{col}_rolling_min_{window}"] = df[col].rolling(window).min()
                df[f"{col}_rolling_max_{window}"] = df[col].rolling(window).max()
    return df

def remove_outliers_iqr(df: pd.DataFrame, columns: List[str], factor: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers using IQR method
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

def save_json(data: Dict, filepath: str):
    """
    Save dictionary to JSON file
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_json(filepath: str) -> Dict:
    """
    Load dictionary from JSON file
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def save_pickle(obj: Any, filepath: str):
    """
    Save object to pickle file
    """
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filepath: str) -> Any:
    """
    Load object from pickle file
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def hash_string(text: str) -> str:
    """
    Create MD5 hash of a string
    """
    return hashlib.md5(text.encode()).hexdigest()

def format_currency(value: float) -> str:
    """
    Format number as currency
    """
    if value >= 1_000_000_000:
        return f"${value/1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value/1_000:.2f}K"
    else:
        return f"${value:.2f}"

def format_percentage(value: float) -> str:
    """
    Format number as percentage
    """
    return f"{value:.2f}%"

def get_market_hours() -> Dict[str, tuple]:
    """
    Get trading hours for different markets
    """
    return {
        "asian": (0, 8),     # 00:00 - 08:00 UTC
        "european": (8, 16), # 08:00 - 16:00 UTC
        "us": (16, 24),      # 16:00 - 00:00 UTC
    }

def is_market_open() -> bool:
    """
    Check if markets are open (crypto is always open)
    """
    return True  # Crypto markets are 24/7

def calculate_correlation_matrix(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Calculate correlation matrix for specified columns
    """
    return df[columns].corr()

def calculate_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate rolling volatility
    """
    returns = df['close'].pct_change()
    volatility = returns.rolling(window=window).std() * np.sqrt(365*24)
    return volatility

def create_backtest_signals(predictions: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Create trading signals from predictions
    """
    signals = np.zeros_like(predictions)
    signals[predictions > threshold] = 1  # Buy
    signals[predictions < (1 - threshold)] = -1  # Sell
    return signals

def calculate_position_size(account_size: float, risk_per_trade: float, 
                          entry_price: float, stop_loss: float) -> float:
    """
    Calculate position size based on risk management
    """
    risk_amount = account_size * (risk_per_trade / 100)
    risk_per_unit = abs(entry_price - stop_loss)
    
    if risk_per_unit == 0:
        return 0
    
    position_size = risk_amount / risk_per_unit
    return position_size

class Cache:
    """Simple caching utility"""
    
    def __init__(self, ttl: int = 300):
        self.cache = {}
        self.ttl = ttl  # Time to live in seconds
    
    def set(self, key: str, value: Any):
        """Set cache value with timestamp"""
        self.cache[key] = {
            'value': value,
            'timestamp': datetime.now()
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get cache value if not expired"""
        if key not in self.cache:
            return None
        
        cache_item = self.cache[key]
        age = (datetime.now() - cache_item['timestamp']).total_seconds()
        
        if age > self.ttl:
            del self.cache[key]
            return None
        
        return cache_item['value']
    
    def clear(self):
        """Clear all cache"""
        self.cache = {} 
