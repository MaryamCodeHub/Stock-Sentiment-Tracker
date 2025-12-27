import unittest
from app.sentiment_analyzer import SentimentAnalyzer

class TestSentimentAnalyzer(unittest.TestCase):
    """Test sentiment analysis functionality"""
    
    def setUp(self):
        self.analyzer = SentimentAnalyzer()
    
    def test_vader_analysis(self):
        """Test VADER sentiment analysis"""
        result = self.analyzer.analyze("Bitcoin is amazing!", model="vader")
        
        self.assertIn('sentiment', result)
        self.assertIn('compound', result)
        self.assertIn('confidence', result)
        
        # Sentiment should be one of these
        self.assertIn(result['sentiment'], ['positive', 'negative', 'neutral'])
    
    def test_batch_analysis(self):
        """Test batch sentiment analysis"""
        texts = ["This is good", "This is bad", "This is neutral"]
        df = self.analyzer.analyze_batch(texts)
        
        self.assertEqual(len(df), 3)
        self.assertIn('sentiment', df.columns)

if __name__ == '__main__':
    unittest.main() 
