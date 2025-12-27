"""
Simplified Sentiment Analyzer - Only VADER Support
Removes dependencies on torch, transformers, textblob
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import requests
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from app.config import config

# Download NLTK data
try:
    nltk.data.find('vader_lexicon')
except:
    nltk.download('vader_lexicon', quiet=True)

class SentimentAnalyzer:
    """
    Simplified sentiment analyzer using only VADER
    No torch, transformers, or textblob dependencies
    """
    
    def __init__(self, model_type: str = "vader"):
        """
        Initialize sentiment analyzer
        
        Args:
            model_type: Only "vader" supported in simplified version
        """
        # Force VADER only (simplified version)
        self.model_type = "vader"
        self.sia = SentimentIntensityAnalyzer()
        print("✅ SentimentAnalyzer initialized with VADER (simplified version)")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text
        
        Args:
            text: Raw text
        
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def analyze_vader(self, text: str) -> Dict:
        """
        Analyze sentiment using VADER
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with sentiment scores
        """
        scores = self.sia.polarity_scores(text)
        
        # Determine sentiment
        compound = scores['compound']
        if compound >= 0.05:
            sentiment = "positive"
        elif compound <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "text": text,
            "sentiment": sentiment,
            "compound": compound,
            "positive": scores['pos'],
            "negative": scores['neg'],
            "neutral": scores['neu'],
            "confidence": abs(compound),
            "model": "vader"
        }
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze text using VADER
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with sentiment scores
        """
        text = self.clean_text(text)
        return self.analyze_vader(text)
    
    def analyze_batch(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze multiple texts
        
        Args:
            texts: List of texts to analyze
        
        Returns:
            DataFrame with sentiment analysis
        """
        results = []
        
        for text in texts:
            if text and isinstance(text, str):
                result = self.analyze(text)
                results.append(result)
        
        return pd.DataFrame(results)
    
    def get_crypto_news(self, limit: int = 10) -> List[str]:
        """
        Fetch cryptocurrency news or return sample data
        
        Args:
            limit: Maximum number of news items
        
        Returns:
            List of news headlines
        """
        headlines = []
        
        # Try APIs if keys available
        if hasattr(config, 'CRYPTOPANIC_AUTH_TOKEN') and config.CRYPTOPANIC_AUTH_TOKEN:
            try:
                url = "https://cryptopanic.com/api/v1/posts/"
                params = {
                    "auth_token": config.CRYPTOPANIC_AUTH_TOKEN,
                    "public": "true",
                    "kind": "news"
                }
                
                response = requests.get(url, params=params, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    for post in data.get('results', [])[:limit]:
                        if 'title' in post:
                            headlines.append(post['title'])
                
                if len(headlines) >= limit:
                    return headlines[:limit]
                    
            except:
                pass
        
        if hasattr(config, 'NEWSAPI_KEY') and config.NEWSAPI_KEY:
            try:
                url = "https://newsapi.org/v2/everything"
                params = {
                    "q": "cryptocurrency OR bitcoin OR ethereum",
                    "apiKey": config.NEWSAPI_KEY,
                    "pageSize": limit,
                    "language": "en"
                }
                
                response = requests.get(url, params=params, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    for article in data.get('articles', [])[:limit]:
                        if 'title' in article:
                            headlines.append(article['title'])
                
                if len(headlines) >= limit:
                    return headlines[:limit]
                    
            except:
                pass
        
        # Sample data if no API or API failed
        if not headlines:
            headlines = [
                "Bitcoin reaches new all-time high amid institutional adoption",
                "Ethereum upgrade successful, reduces gas fees significantly",
                "Regulatory concerns weigh on cryptocurrency markets",
                "Major financial institution announces crypto trading services",
                "Market volatility expected ahead of Federal Reserve meeting",
                "Cryptocurrency adoption continues to grow globally",
                "New blockchain project raises millions in funding",
                "Analysts predict bullish trend for altcoins",
                "Central bank digital currency developments accelerate",
                "Crypto market shows signs of recovery after correction"
            ]
        
        return headlines[:limit]
    
    def get_news_sentiment(self, limit: int = 10) -> Tuple[pd.DataFrame, Dict]:
        """
        Fetch news and analyze sentiment
        
        Args:
            limit: Number of news items
        
        Returns:
            Tuple of (DataFrame with sentiment, aggregate statistics)
        """
        # Get news headlines
        headlines = self.get_crypto_news(limit)
        
        # Analyze sentiment
        sentiment_df = self.analyze_batch(headlines)
        
        if sentiment_df.empty:
            # Create default aggregate
            aggregate = {
                "avg_sentiment_score": 0,
                "positive_ratio": 0,
                "negative_ratio": 0,
                "neutral_ratio": 1,
                "total_news": 0
            }
            return sentiment_df, aggregate
        
        # Calculate aggregate statistics
        if 'compound' in sentiment_df.columns:
            avg_score = sentiment_df['compound'].mean()
        else:
            # Estimate from sentiment column
            sentiment_map = {"positive": 1, "negative": -1, "neutral": 0}
            sentiment_df['score'] = sentiment_df['sentiment'].map(sentiment_map)
            avg_score = sentiment_df['score'].mean()
        
        # Calculate sentiment ratios
        total = len(sentiment_df)
        positive_ratio = (sentiment_df['sentiment'] == 'positive').sum() / total
        negative_ratio = (sentiment_df['sentiment'] == 'negative').sum() / total
        neutral_ratio = (sentiment_df['sentiment'] == 'neutral').sum() / total
        
        aggregate = {
            "avg_sentiment_score": float(avg_score),
            "positive_ratio": float(positive_ratio),
            "negative_ratio": float(negative_ratio),
            "neutral_ratio": float(neutral_ratio),
            "total_news": int(total)
        }
        
        return sentiment_df, aggregate
    
    def get_sample_sentiment(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Get sample sentiment data for demo
        
        Returns:
            Tuple of (DataFrame with sentiment, aggregate statistics)
        """
        sample_texts = [
            "Bitcoin price surges to new all-time high",
            "Market correction expected after rapid growth",
            "Ethereum network upgrade reduces transaction fees",
            "Regulatory uncertainty causes market volatility",
            "Institutional investors continue crypto adoption"
        ]
        
        df = self.analyze_batch(sample_texts)
        
        aggregate = {
            "avg_sentiment_score": 0.15,
            "positive_ratio": 0.4,
            "negative_ratio": 0.2,
            "neutral_ratio": 0.4,
            "total_news": len(sample_texts)
        }
        
        return df, aggregate