 #!/usr/bin/env python3
"""
Model training script
Run as: python scripts/train_model.py --symbol BTCUSDT --model xgboost
"""

import argparse
import sys
import os
from datetime import datetime

# Add app directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.model_trainer import ModelTrainer
from app.data_collector import DataCollector
from app.config import config
import warnings
warnings.filterwarnings('ignore')

def train_single_model(symbol: str, model_type: str = "xgboost", days: int = 90):
    """
    Train a single model for a symbol
    """
    print(f"🚀 Training {model_type} model for {symbol}")
    print(f"📊 Using {days} days of data")
    
    # Initialize trainer
    trainer = ModelTrainer(model_type=model_type)
    
    try:
        # Run complete training pipeline
        results = trainer.train_complete_pipeline(
            symbol=symbol,
            horizon=config.PREDICTION_HORIZON,
            days=days
        )
        
        # Print summary
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Symbol: {symbol}")
        print(f"Model: {model_type}")
        print(f"Model saved: {results['model_path']}")
        print(f"Training samples: {results['training_samples']}")
        print(f"Test samples: {results['test_samples']}")
        print(f"Features: {results['num_features']}")
        
        print("\n📊 Performance Metrics:")
        for metric, value in results['metrics'].items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def train_all_models():
    """
    Train models for all symbols
    """
    symbols = config.SYMBOLS
    model_types = ["xgboost", "random_forest", "logistic"]
    
    all_results = {}
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"TRAINING MODELS FOR {symbol}")
        print('='*60)
        
        symbol_results = {}
        
        for model_type in model_types:
            try:
                results = train_single_model(symbol, model_type, days=90)
                if results:
                    symbol_results[model_type] = results
                time.sleep(1)  # Small delay
            except Exception as e:
                print(f"❌ Error training {model_type} for {symbol}: {e}")
        
        all_results[symbol] = symbol_results
    
    # Print final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*60)
    
    for symbol, models in all_results.items():
        print(f"\n{symbol}:")
        for model_type, results in models.items():
            if 'metrics' in results:
                accuracy = results['metrics'].get('accuracy', results['metrics'].get('r2', 0))
                print(f"  {model_type}: {accuracy:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Train ML models for stock prediction")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol to train for")
    parser.add_argument("--model", type=str, default="xgboost", 
                       choices=["xgboost", "random_forest", "logistic"], 
                       help="Model type")
    parser.add_argument("--days", type=int, default=90, help="Days of historical data")
    parser.add_argument("--all", action="store_true", help="Train for all symbols")
    
    args = parser.parse_args()
    
    if args.all:
        train_all_models()
    else:
        train_single_model(args.symbol, args.model, args.days)

if __name__ == "__main__":
    main()
