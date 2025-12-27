"""
Stock Sentiment Tracker - Main application package
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from app.config import Config
from app.data_collector import DataCollector
from app.sentiment_analyzer import SentimentAnalyzer
from app.model_trainer import ModelTrainer

__all__ = [
    "Config",
    "DataCollector",
    "SentimentAnalyzer",
    "ModelTrainer"
] 
