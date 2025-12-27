#!/usr/bin/env python3
"""
Data collection script for real-time market data
Run as: python scripts/collect_data.py --symbol BTCUSDT --days 30
"""

import argparse
import schedule
import time
from datetime import datetime
import pandas as pd
from app.data_collector import DataCollector
from app.config import config
import warnings
warnings.filterwarnings('ignore')

def collect_data_for_symbol(symbol: str, days: int = 7):
    """
    Collect data for a single symbol
    """
    print(f"📊 Collecting data for {symbol}...")
    
    collector = DataCollector()
    
    # Collect real-time price
    price = collector.get_realtime_price(symbol)
    print(f"  Current price: ${price:,.2f}")
    
    # Collect historical data (different timeframes)
    timeframes = ['1h', '4h', '1d']
    
    for timeframe in timeframes:
        print(f"  Fetching {timeframe} data...")
        df = collector.get_historical_data(symbol, timeframe, days)
        
        if not df.empty:
            # Save to CSV
            filename = f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.csv"
            collector.save_to_csv(df, filename)
            print(f"    Saved {len(df)} records to {filename}")
        else:
            print(f"    No data for {timeframe}")
    
    print(f"✅ Data collection completed for {symbol}")

def collect_all_symbols():
    """
    Collect data for all configured symbols
    """
    symbols = config.SYMBOLS
    
    for symbol in symbols:
        try:
            collect_data_for_symbol(symbol, days=7)
            time.sleep(1)  # Rate limiting
        except Exception as e:
            print(f"❌ Error collecting {symbol}: {e}")

def scheduled_collection():
    """
    Scheduled data collection job
    """
    print(f"\n🕒 Scheduled collection at {datetime.now()}")
    collect_all_symbols()
    print(f"✅ Scheduled collection completed at {datetime.now()}")

def main():
    parser = argparse.ArgumentParser(description="Collect market data")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol to collect")
    parser.add_argument("--days", type=int, default=30, help="Days of historical data")
    parser.add_argument("--schedule", action="store_true", help="Run as scheduled job")
    parser.add_argument("--interval", type=int, default=5, help="Schedule interval in minutes")
    
    args = parser.parse_args()
    
    if args.schedule:
        print(f"🚀 Starting scheduled data collection every {args.interval} minutes")
        print(f"📊 Symbols: {', '.join(config.SYMBOLS)}")
        
        # Run immediately
        scheduled_collection()
        
        # Schedule regular runs
        schedule.every(args.interval).minutes.do(scheduled_collection)
        
        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(1)
    
    else:
        # One-time collection
        if args.symbol == "ALL":
            collect_all_symbols()
        else:
            collect_data_for_symbol(args.symbol, args.days)

if __name__ == "__main__":
    main() 
