import unittest
import pandas as pd
from app.data_collector import DataCollector
from app.config import config

class TestDataCollector(unittest.TestCase):
    """Test data collection functionality"""
    
    def setUp(self):
        self.collector = DataCollector()
    
    def test_realtime_price(self):
        """Test getting real-time price"""
        price = self.collector.get_realtime_price("BTCUSDT")
        self.assertIsInstance(price, float)
        self.assertGreater(price, 0)
    
    def test_historical_data(self):
        """Test getting historical data"""
        df = self.collector.get_historical_data("BTCUSDT", "1h", 1)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        
        # Check required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            self.assertIn(col, df.columns)

if __name__ == '__main__':
    unittest.main() 
