import unittest
import numpy as np
from app.model_trainer import ModelTrainer, FeatureEngineer

class TestModelTrainer(unittest.TestCase):
    """Test model training functionality"""
    
    def setUp(self):
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        self.df = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 105,
            'low': np.random.randn(100).cumsum() + 95,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.abs(np.random.randn(100) * 1000) + 1000
        })
    
    def test_feature_engineering(self):
        """Test feature engineering"""
        engineer = FeatureEngineer()
        df_features = engineer.add_technical_indicators(self.df)
        
        # Check that features were added
        self.assertGreater(len(df_features.columns), len(self.df.columns))
        self.assertIn('returns', df_features.columns)
        self.assertIn('rsi', df_features.columns)
    
    def test_model_training(self):
        """Test model training"""
        trainer = ModelTrainer(model_type="logistic")
        
        # Prepare data
        X_train, X_test, y_train, y_test, _ = trainer.prepare_data(
            self.df, horizon=1, test_size=0.2
        )
        
        # Train model
        model = trainer.train(X_train, y_train)
        
        # Should have a trained model
        self.assertIsNotNone(model)
        
        # Should be able to predict
        predictions = trainer.predict(X_test)
        self.assertEqual(len(predictions), len(X_test))

if __name__ == '__main__':
    unittest.main() 
