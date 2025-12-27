import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration manager for the application"""
    
    # Base paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # Data paths
    DATA_DIR = BASE_DIR / "data"
    LOG_DIR = BASE_DIR / "logs"
    MODEL_DIR = BASE_DIR / "ml_models"
    
    # Create directories if they don't exist
    for directory in [DATA_DIR, LOG_DIR, MODEL_DIR]:
        directory.mkdir(exist_ok=True)
    
    # API Configuration
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
    
    # Database Configuration
    POSTGRES_DB = os.getenv("POSTGRES_DB", "stockdb")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "admin")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
    
    # Redis Configuration
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
    
    # Sentiment API Keys
    NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
    CRYPTOPANIC_AUTH_TOKEN = os.getenv("CRYPTOPANIC_AUTH_TOKEN", "")
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = LOG_DIR / "app.log"
    
    # Model Settings
    MODEL_UPDATE_FREQUENCY = int(os.getenv("MODEL_UPDATE_FREQUENCY", 3600))  # seconds
    PREDICTION_HORIZON = int(os.getenv("PREDICTION_HORIZON", 1))  # hours
    
    # Trading Settings
    SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT"]
    TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
    
    # Feature Engineering
    TECHNICAL_INDICATORS = {
        'trend': ['macd', 'ema', 'sma', 'wma'],
        'momentum': ['rsi', 'stoch', 'williams', 'roc'],
        'volatility': ['bb', 'atr', 'kc'],
        'volume': ['obv', 'ad', 'cmf']
    }
    
    @classmethod
    def get_database_url(cls) -> str:
        """Get database connection URL"""
        return f"postgresql://{cls.POSTGRES_USER}:{cls.POSTGRES_PASSWORD}@" \
               f"{cls.POSTGRES_HOST}:{cls.POSTGRES_PORT}/{cls.POSTGRES_DB}"
    
    @classmethod
    def get_redis_url(cls) -> str:
        """Get Redis connection URL"""
        if cls.REDIS_PASSWORD:
            return f"redis://:{cls.REDIS_PASSWORD}@{cls.REDIS_HOST}:{cls.REDIS_PORT}/0"
        return f"redis://{cls.REDIS_HOST}:{cls.REDIS_PORT}/0"
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration"""
        errors = []
        
        if not cls.BINANCE_API_KEY or not cls.BINANCE_API_SECRET:
            errors.append("Binance API keys not configured")
        
        if not cls.NEWSAPI_KEY and not cls.CRYPTOPANIC_AUTH_TOKEN:
            errors.append("No sentiment API keys configured")
        
        if errors:
            print("Configuration errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True

# Global config instance
config = Config() 
