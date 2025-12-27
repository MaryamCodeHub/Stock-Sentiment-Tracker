import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import joblib
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Import ML libraries
try:
    import xgboost as xgb
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: xgboost not installed")

try:
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, mean_squared_error, r2_score
    )
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed")

from app.config import config
from app.data_collector import DataCollector
from app.sentiment_analyzer import SentimentAnalyzer

class FeatureEngineer:
    """Feature engineering for financial time series"""
    
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to DataFrame
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with added indicators
        """
        df = df.copy()
        
        # Ensure required columns
        if 'close' not in df.columns and 'Close' in df.columns:
            df.rename(columns={'Close': 'close'}, inplace=True)
        if 'open' not in df.columns and 'Open' in df.columns:
            df.rename(columns={'Open': 'open'}, inplace=True)
        if 'high' not in df.columns and 'High' in df.columns:
            df.rename(columns={'High': 'high'}, inplace=True)
        if 'low' not in df.columns and 'Low' in df.columns:
            df.rename(columns={'Low': 'low'}, inplace=True)
        if 'volume' not in df.columns and 'Volume' in df.columns:
            df.rename(columns={'Volume': 'volume'}, inplace=True)
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(365*24)
        
        # Price patterns
        df['high_low_pct'] = (df['high'] - df['low']) / df['close'] * 100
        df['close_open_pct'] = (df['close'] - df['open']) / df['open'] * 100
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'rolling_std_{window}'] = df['returns'].rolling(window).std()
            df[f'rolling_skew_{window}'] = df['returns'].rolling(window).skew()
            df[f'rolling_kurt_{window}'] = df['returns'].rolling(window).kurt()
        
        # Clean NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    @staticmethod
    def create_target(df: pd.DataFrame, horizon: int = 1, target_type: str = "direction") -> pd.Series:
        """
        Create target variable for prediction
        
        Args:
            df: DataFrame with price data
            horizon: Prediction horizon (periods ahead)
            target_type: "direction", "return", or "volatility"
        
        Returns:
            Target Series
        """
        if target_type == "direction":
            # Binary: 1 if price goes up, 0 if down
            future_price = df['close'].shift(-horizon)
            target = (future_price > df['close']).astype(int)
            target = target.iloc[:-horizon] if horizon > 0 else target
        
        elif target_type == "return":
            # Continuous: Percentage return
            future_price = df['close'].shift(-horizon)
            target = (future_price - df['close']) / df['close'] * 100
            target = target.iloc[:-horizon] if horizon > 0 else target
        
        elif target_type == "volatility":
            # Future volatility
            future_returns = df['close'].pct_change().shift(-horizon).rolling(horizon).std()
            target = future_returns * np.sqrt(365*24)  # Annualized
            target = target.iloc[:-horizon] if horizon > 0 else target
        
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
        
        return target

class ModelTrainer:
    """Train and manage ML models for stock prediction"""
    
    def __init__(self, model_type: str = "xgboost"):
        """
        Initialize model trainer
        
        Args:
            model_type: Type of model to train
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_engineer = FeatureEngineer()
        self.feature_importance = None
        
        print(f"✅ ModelTrainer initialized for {model_type}")
    
    def prepare_data(self, 
                    df: pd.DataFrame,
                    horizon: int = 1,
                    test_size: float = 0.2,
                    target_type: str = "direction") -> Tuple:
        """
        Prepare data for training
        
        Args:
            df: Input DataFrame
            horizon: Prediction horizon
            test_size: Test set proportion
            target_type: Type of target variable
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_names)
        """
        # Add technical indicators
        df_features = self.feature_engineer.add_technical_indicators(df)
        
        # Create target
        y = self.feature_engineer.create_target(df_features, horizon, target_type)
        
        # Align features with target (remove NaN rows)
        df_features = df_features.iloc[:len(y)]
        y = y.iloc[:len(df_features)]
        
        # Select features (exclude target columns)
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df_features.columns 
                       if col not in exclude_cols and not col.startswith('target')]
        
        X = df_features[feature_cols]
        
        # Time-series split (no shuffling!)
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # Scale features
        if self.scaler:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values
        
        return X_train_scaled, X_test_scaled, y_train.values, y_test.values, feature_cols
    
    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train XGBoost model
        """
        if not XGB_AVAILABLE:
            raise ImportError("xgboost not installed")
        
        # Define parameters based on problem type
        if len(np.unique(y_train)) == 2:  # Classification
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'n_estimators': 200,
                'learning_rate': 0.05,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1
            }
            self.model = XGBClassifier(**params)
        else:  # Regression
            params = {
                'objective': 'reg:squarederror',
                'n_estimators': 200,
                'learning_rate': 0.05,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1
            }
            self.model = XGBRegressor(**params)
        
        # Train with validation if provided
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
                early_stopping_rounds=20
            )
        else:
            self.model.fit(X_train, y_train)
        
        # Get feature importance
        self.feature_importance = self.model.feature_importances_
        
        print("✅ XGBoost model trained")
        return self.model
    
    def train_random_forest(self, X_train, y_train):
        """
        Train Random Forest model
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not installed")
        
        if len(np.unique(y_train)) == 2:  # Classification
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        else:  # Regression
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        
        self.model.fit(X_train, y_train)
        self.feature_importance = self.model.feature_importances_
        
        print("✅ Random Forest model trained")
        return self.model
    
    def train_logistic_regression(self, X_train, y_train):
        """
        Train Logistic Regression model
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not installed")
        
        self.model = LogisticRegression(
            C=1.0,
            penalty='l2',
            random_state=42,
            max_iter=1000,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        print("✅ Logistic Regression model trained")
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the selected model
        """
        if self.model_type == "xgboost":
            return self.train_xgboost(X_train, y_train, X_val, y_val)
        elif self.model_type == "random_forest":
            return self.train_random_forest(X_train, y_train)
        elif self.model_type == "logistic":
            return self.train_logistic_regression(X_train, y_train)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def predict(self, X):
        """
        Make predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities (for classification)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models without predict_proba, return binary predictions
            preds = self.predict(X)
            return np.column_stack([1 - preds, preds])
    
    def evaluate(self, X_test, y_test) -> Dict:
        """
        Evaluate model performance
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        y_pred = self.predict(X_test)
        
        metrics = {}
        
        # Check if classification or regression
        if len(np.unique(y_test)) == 2:  # Classification
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_test, y_pred, zero_division=0)
            metrics['f1'] = f1_score(y_test, y_pred, zero_division=0)
            
            # ROC AUC (need probabilities)
            if hasattr(self.model, 'predict_proba'):
                y_pred_proba = self.predict_proba(X_test)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()
            
        else:  # Regression
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = np.mean(np.abs(y_test - y_pred))
            metrics['r2'] = r2_score(y_test, y_pred)
            
            # Direction accuracy (for regression of returns)
            direction_acc = np.mean((y_test * y_pred) > 0)
            metrics['direction_accuracy'] = direction_acc
        
        return metrics
    
    def save_model(self, filename: str = None):
        """
        Save trained model to disk
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.model_type}_model_{timestamp}.joblib"
        
        # Prepare model data
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance,
            'training_date': datetime.now().isoformat()
        }
        
        # Save to models directory
        model_path = config.MODEL_DIR / filename
        joblib.dump(model_data, model_path)
        
        print(f"✅ Model saved to {model_path}")
        return model_path
    
    def load_model(self, filename: str):
        """
        Load trained model from disk
        """
        model_path = config.MODEL_DIR / filename
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.feature_importance = model_data['feature_importance']
        
        print(f"✅ Model loaded from {model_path}")
        return self
    
    def get_feature_importance(self, feature_names: List[str], top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance DataFrame
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance not available")
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Return top N features
        return importance_df.head(top_n)
    
    def train_complete_pipeline(self, 
                               symbol: str = "BTCUSDT",
                               horizon: int = 1,
                               days: int = 90) -> Dict:
        """
        Complete training pipeline from data collection to model saving
        """
        print(f"🚀 Starting training pipeline for {symbol}")
        
        # Step 1: Collect data
        print("📊 Collecting data...")
        collector = DataCollector()
        df = collector.get_historical_data(symbol, '1h', days)
        
        if df.empty:
            raise ValueError(f"No data collected for {symbol}")
        
        print(f"📈 Collected {len(df)} data points")
        
        # Step 2: Prepare data
        print("🔧 Preparing data...")
        X_train, X_test, y_train, y_test, feature_names = self.prepare_data(
            df, horizon=horizon, test_size=0.2
        )
        
        print(f"📊 Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"📈 Features: {len(feature_names)}")
        
        # Step 3: Train model
        print("🤖 Training model...")
        self.train(X_train, y_train)
        
        # Step 4: Evaluate
        print("📊 Evaluating model...")
        metrics = self.evaluate(X_test, y_test)
        
        # Step 5: Save model
        print("💾 Saving model...")
        model_path = self.save_model()
        
        # Step 6: Print results
        print("\n" + "="*50)
        print("TRAINING RESULTS")
        print("="*50)
        
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
        
        # Feature importance
        if self.feature_importance is not None:
            print("\nTop 10 Features:")
            importance_df = self.get_feature_importance(feature_names, top_n=10)
            for idx, row in importance_df.iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return {
            'metrics': metrics,
            'model_path': str(model_path),
            'num_features': len(feature_names),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        } 
