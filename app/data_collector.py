import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import requests
import json
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    print("Warning: python-binance not installed")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not installed")

import redis
from app.config import config

class DataCollector:
    """Collect real-time and historical market data"""
    
    def __init__(self):
        self.config = config
        self.binance_client = None
        self.redis_client = None
        
        # Initialize clients
        self._setup_binance()
        self._setup_redis()
        
    def _setup_binance(self):
        """Initialize Binance API client"""
        if BINANCE_AVAILABLE and self.config.BINANCE_API_KEY and self.config.BINANCE_API_SECRET:
            try:
                self.binance_client = Client(
                    api_key=self.config.BINANCE_API_KEY,
                    api_secret=self.config.BINANCE_API_SECRET
                )
                print("✅ Binance client initialized")
            except Exception as e:
                print(f"❌ Binance setup failed: {e}")
                self.binance_client = None
        else:
            print("⚠️  Binance not configured")
    
    def _setup_redis(self):
        """Initialize Redis client"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.REDIS_HOST,
                port=self.config.REDIS_PORT,
                password=self.config.REDIS_PASSWORD or None,
                decode_responses=True,
                socket_connect_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            print("✅ Redis client initialized")
        except Exception as e:
            print(f"❌ Redis connection failed: {e}")
            self.redis_client = None
    
    def get_realtime_price(self, symbol: str) -> float:
        """
        Get current price of a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
        
        Returns:
            Current price as float
        """
        # Check Redis cache first
        cache_key = f"price:{symbol}"
        if self.redis_client:
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    return float(cached)
            except:
                pass
        
        price = 0.0
        
        # Try Binance first
        if self.binance_client:
            try:
                ticker = self.binance_client.get_symbol_ticker(symbol=symbol)
                price = float(ticker['price'])
            except Exception as e:
                print(f"Binance error for {symbol}: {e}")
                price = 0.0
        
        # Fallback to yfinance
        if price == 0 and YFINANCE_AVAILABLE:
            try:
                ticker_symbol = symbol.replace('USDT', '-USD')
                ticker = yf.Ticker(ticker_symbol)
                data = ticker.history(period='1d', interval='1m')
                if not data.empty:
                    price = float(data['Close'].iloc[-1])
            except Exception as e:
                print(f"YFinance error for {symbol}: {e}")
        
        # Cache in Redis
        if price > 0 and self.redis_client:
            try:
                self.redis_client.setex(cache_key, 60, str(price))  # 1 minute cache
            except:
                pass
        
        return price
    
    def get_historical_data(self, 
                           symbol: str, 
                           interval: str = '1h',
                           days: int = 30) -> pd.DataFrame:
        """
        Fetch historical data
        
        Args:
            symbol: Trading symbol
            interval: Time interval ('1m', '5m', '1h', '1d', etc.)
            days: Number of days of data to fetch
        
        Returns:
            DataFrame with historical data
        """
        cache_key = f"history:{symbol}:{interval}:{days}"
        
        # Check cache
        if self.redis_client:
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    df = pd.read_json(cached, orient='split')
                    return df
            except:
                pass
        
        df = pd.DataFrame()
        
        # Map interval to Binance format
        interval_map = {
            '1m': Client.KLINE_INTERVAL_1MINUTE,
            '5m': Client.KLINE_INTERVAL_5MINUTE,
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '4h': Client.KLINE_INTERVAL_4HOUR,
            '1d': Client.KLINE_INTERVAL_1DAY,
            '1w': Client.KLINE_INTERVAL_1WEEK
        }
        
        # Try Binance
        if self.binance_client and interval in interval_map:
            try:
                # Convert days to appropriate lookback
                if interval.endswith('m'):
                    limit = days * 24 * 60 // int(interval[:-1])
                elif interval.endswith('h'):
                    limit = days * 24 // int(interval[:-1])
                else:
                    limit = days
                
                limit = min(limit, 1000)  # Binance limit
                
                klines = self.binance_client.get_historical_klines(
                    symbol=symbol,
                    interval=interval_map[interval],
                    start_str=f"{days} day ago UTC",
                    limit=limit
                )
                
                if klines:
                    df = pd.DataFrame(klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base', 'taker_buy_quote', 'ignore'
                    ])
                    
                    # Convert columns
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                    df[numeric_cols] = df[numeric_cols].astype(float)
                    
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    
            except Exception as e:
                print(f"Binance historical error: {e}")
        
        # Fallback to yfinance
        if df.empty and YFINANCE_AVAILABLE:
            try:
                ticker_symbol = symbol.replace('USDT', '-USD')
                ticker = yf.Ticker(ticker_symbol)
                
                # Map interval to yfinance
                yf_interval = '1m' if interval == '1m' else \
                             '5m' if interval == '5m' else \
                             '15m' if interval == '15m' else \
                             '1h' if interval in ['1h', '4h'] else \
                             '1d' if interval == '1d' else '1d'
                
                period = f"{days}d"
                
                df = ticker.history(period=period, interval=yf_interval)
                df.reset_index(inplace=True)
                
                if 'Datetime' in df.columns:
                    df.rename(columns={'Datetime': 'timestamp'}, inplace=True)
                elif 'Date' in df.columns:
                    df.rename(columns={'Date': 'timestamp'}, inplace=True)
                
                # Ensure required columns
                if 'Close' in df.columns and 'close' not in df.columns:
                    df.rename(columns={'Close': 'close'}, inplace=True)
                if 'Open' in df.columns and 'open' not in df.columns:
                    df.rename(columns={'Open': 'open'}, inplace=True)
                if 'High' in df.columns and 'high' not in df.columns:
                    df.rename(columns={'High': 'high'}, inplace=True)
                if 'Low' in df.columns and 'low' not in df.columns:
                    df.rename(columns={'Low': 'low'}, inplace=True)
                if 'Volume' in df.columns and 'volume' not in df.columns:
                    df.rename(columns={'Volume': 'volume'}, inplace=True)
                
            except Exception as e:
                print(f"YFinance historical error: {e}")
        
        # Cache results
        if not df.empty and self.redis_client:
            try:
                json_data = df.to_json(orient='split', date_format='iso')
                # Cache for 5 minutes for intraday, 1 hour for daily
                cache_time = 300 if interval.endswith(('m', 'h')) else 3600
                self.redis_client.setex(cache_key, cache_time, json_data)
            except:
                pass
        
        return df
    
    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current prices for multiple symbols
        
        Args:
            symbols: List of trading symbols
        
        Returns:
            Dictionary of symbol -> price
        """
        prices = {}
        for symbol in symbols:
            prices[symbol] = self.get_realtime_price(symbol)
            time.sleep(0.1)  # Rate limiting
        return prices
    
    def get_market_summary(self, symbol: str) -> Dict:
        """
        Get comprehensive market summary
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Dictionary with market summary
        """
        summary = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'price': 0,
            '24h_change': 0,
            '24h_high': 0,
            '24h_low': 0,
            '24h_volume': 0
        }
        
        # Get 24h data
        df = self.get_historical_data(symbol, '1h', 2)  # 2 days for 24h calc
        
        if len(df) >= 24:
            summary['price'] = float(df['close'].iloc[-1])
            summary['24h_high'] = float(df['high'].tail(24).max())
            summary['24h_low'] = float(df['low'].tail(24).min())
            summary['24h_volume'] = float(df['volume'].tail(24).sum())
            
            price_24h_ago = float(df['close'].iloc[-24])
            summary['24h_change'] = ((summary['price'] - price_24h_ago) / price_24h_ago * 100)
        
        return summary
    
    def stream_realtime_data(self, 
                            symbols: List[str],
                            callback=None,
                            interval: int = 5):
        """
        Stream real-time data (simulated with polling)
        
        Args:
            symbols: List of symbols to stream
            callback: Function to call with new data
            interval: Polling interval in seconds
        """
        import threading
        
        def poller():
            while True:
                try:
                    for symbol in symbols:
                        price = self.get_realtime_price(symbol)
                        data = {
                            'symbol': symbol,
                            'price': price,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        if callback:
                            callback(data)
                        
                        time.sleep(0.1)  # Small delay between symbols
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    print(f"Streaming error: {e}")
                    time.sleep(interval)
        
        # Start in background thread
        thread = threading.Thread(target=poller, daemon=True)
        thread.start()
        
        return thread
    
    def save_to_csv(self, df: pd.DataFrame, filename: str):
        """
        Save DataFrame to CSV
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        path = self.config.DATA_DIR / filename
        df.to_csv(path, index=False)
        print(f"✅ Data saved to {path}")
    
    def load_from_csv(self, filename: str) -> pd.DataFrame:
        """
        Load DataFrame from CSV
        
        Args:
            filename: Input filename
        
        Returns:
            Loaded DataFrame
        """
        path = self.config.DATA_DIR / filename
        if path.exists():
            return pd.read_csv(path, parse_dates=['timestamp'])
        return pd.DataFrame() 
