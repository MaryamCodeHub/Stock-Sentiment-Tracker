 #!/usr/bin/env python3
"""
Complete pipeline runner - data collection, sentiment analysis, prediction
Run as: python scripts/run_pipeline.py --schedule
"""

import argparse
import schedule
import time
from datetime import datetime
import threading
import pandas as pd
from app.data_collector import DataCollector
from app.sentiment_analyzer import SentimentAnalyzer
from app.model_trainer import ModelTrainer
from app.config import config
import warnings
warnings.filterwarnings('ignore')

class PipelineRunner:
    """Run complete data and prediction pipeline"""
    
    def __init__(self):
        self.data_collector = DataCollector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.models = {}
        
    def collect_data_pipeline(self):
        """Run data collection pipeline"""
        print(f"[{datetime.now()}] 📊 Starting data collection...")
        
        symbols = config.SYMBOLS
        
        for symbol in symbols:
            try:
                # Get real-time price
                price = self.data_collector.get_realtime_price(symbol)
                
                # Get historical data
                df = self.data_collector.get_historical_data(symbol, '1h', 1)
                
                # Log results
                print(f"  {symbol}: ${price:,.2f} ({len(df)} records)")
                
                # Save to file (optional)
                if len(df) > 0:
                    filename = f"data/{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                    df.to_csv(filename, index=False)
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"  ❌ Error collecting {symbol}: {e}")
        
        print(f"[{datetime.now()}] ✅ Data collection completed")
    
    def sentiment_pipeline(self):
        """Run sentiment analysis pipeline"""
        print(f"[{datetime.now()}] 😊 Starting sentiment analysis...")
        
        try:
            # Get news sentiment
            news_df, aggregate = self.sentiment_analyzer.get_news_sentiment()
            
            # Print results
            print(f"  Average sentiment: {aggregate['avg_sentiment_score']:.3f}")
            print(f"  Positive ratio: {aggregate['positive_ratio']:.1%}")
            print(f"  Negative ratio: {aggregate['negative_ratio']:.1%}")
            print(f"  Total news: {aggregate['total_news']}")
            
            # Save to file
            sentiment_file = f"logs/sentiment_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            news_df.to_csv(sentiment_file, index=False)
            
            print(f"[{datetime.now()}] ✅ Sentiment analysis completed")
            
        except Exception as e:
            print(f"  ❌ Sentiment analysis failed: {e}")
    
    def prediction_pipeline(self):
        """Run prediction pipeline"""
        print(f"[{datetime.now()}] 🤖 Starting prediction pipeline...")
        
        symbols = config.SYMBOLS[:2]  # Limit to first 2 symbols for speed
        
        for symbol in symbols:
            try:
                # Load or train model
                model = self._get_model(symbol)
                
                if model:
                    # Get recent data
                    df = self.data_collector.get_historical_data(symbol, '1h', 7)
                    
                    if len(df) > 100:
                        # Prepare features
                        from app.model_trainer import FeatureEngineer
                        engineer = FeatureEngineer()
                        df_features = engineer.add_technical_indicators(df)
                        
                        # Select features
                        feature_cols = [col for col in df_features.columns 
                                       if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                        X = df_features[feature_cols].iloc[-1:].values
                        
                        # Make prediction
                        if hasattr(model, 'predict_proba'):
                            prediction = model.predict_proba(X)[0, 1]
                            direction = "UP" if prediction > 0.5 else "DOWN"
                            confidence = max(prediction, 1 - prediction) * 100
                        else:
                            prediction = model.predict(X)[0]
                            direction = "UP" if prediction > 0 else "DOWN"
                            confidence = 50.0
                        
                        print(f"  {symbol}: {direction} ({confidence:.1f}% confidence)")
                    
                time.sleep(1)
                
            except Exception as e:
                print(f"  ❌ Prediction failed for {symbol}: {e}")
        
        print(f"[{datetime.now()}] ✅ Prediction pipeline completed")
    
    def _get_model(self, symbol):
        """Get or train model for symbol"""
        if symbol not in self.models:
            try:
                # Try to load existing model
                import joblib
                import os
                
                model_files = [f for f in os.listdir('ml_models') 
                             if f.startswith(f'xgboost_model') and f.endswith('.joblib')]
                
                if model_files:
                    latest_model = sorted(model_files)[-1]
                    model_data = joblib.load(f'ml_models/{latest_model}')
                    self.models[symbol] = model_data['model']
                    print(f"  Loaded model for {symbol}")
                else:
                    print(f"  No model found for {symbol}")
                    return None
                    
            except Exception as e:
                print(f"  Model loading failed for {symbol}: {e}")
                return None
        
        return self.models.get(symbol)
    
    def run_complete_pipeline(self):
        """Run all pipelines"""
        print(f"\n{'='*60}")
        print(f"STARTING COMPLETE PIPELINE - {datetime.now()}")
        print('='*60)
        
        self.collect_data_pipeline()
        self.sentiment_pipeline()
        self.prediction_pipeline()
        
        print(f"\n{'='*60}")
        print(f"PIPELINE COMPLETED - {datetime.now()}")
        print('='*60)

def main():
    parser = argparse.ArgumentParser(description="Run complete data pipeline")
    parser.add_argument("--schedule", action="store_true", help="Run as scheduled job")
    parser.add_argument("--interval", type=int, default=5, help="Schedule interval in minutes")
    
    args = parser.parse_args()
    
    runner = PipelineRunner()
    
    if args.schedule:
        print(f"🚀 Starting scheduled pipeline every {args.interval} minutes")
        
        # Run immediately
        runner.run_complete_pipeline()
        
        # Schedule regular runs
        schedule.every(args.interval).minutes.do(runner.run_complete_pipeline)
        
        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(1)
    
    else:
        # One-time run
        runner.run_complete_pipeline()

if __name__ == "__main__":
    main()
