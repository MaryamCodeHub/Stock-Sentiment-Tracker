# 🎯 **PROJECT CONTEXT - Real-Time Stock Price & Sentiment Predictor**

## 📋 **PROJECT OVERVIEW**

**Project Name:** Real-Time Stock Price & Sentiment Predictor  
**Type:** Production-ready ML System for Financial Analytics  
**Purpose:** Analyze live cryptocurrency data + news sentiment for trading insights  

**Core Features:**
- 🔄 Real-time crypto price data (Binance/Yahoo Finance APIs)
- 📰 Sentiment analysis (VADER, FinBERT, TextBlob)
- 🤖 ML models (XGBoost, Random Forest, LSTM)
- 📊 Interactive Streamlit dashboard with Plotly
- 🐳 Docker containerization
- ⚡ Real-time predictions every 5 minutes

---

## 🏗️ **PROJECT ARCHITECTURE**

### **System Flow:**
```
User → Streamlit Dashboard → Data Collection → Sentiment Analysis → ML Model → Predictions
```

### **Data Pipeline:**
1. **Data Collection:** Binance API → data_collector.py → Redis Cache
2. **Sentiment Analysis:** News APIs → sentiment_analyzer.py → Sentiment Scores
3. **ML Pipeline:** Historical Data → model_trainer.py → Trained Models
4. **Dashboard:** All Data → main.py → Charts & Predictions

---

## 📁 **FILE STRUCTURE**

```
real-time-stock-sentiment/
├── 📁 app/                    # Main application
│   ├── main.py              # 🎯 Streamlit dashboard (ENTRY POINT)
│   ├── config.py            # ⚙️ Configuration (uses .env)
│   ├── data_collector.py    # 📡 Binance/Yahoo Finance API
│   ├── sentiment_analyzer.py # 😊 NLP sentiment analysis
│   ├── model_trainer.py     # 🤖 ML model training
│   └── utils.py             # 🔧 Utility functions
├── 📁 scripts/              # Automation scripts
│   ├── collect_data.py      # Data collection
│   ├── train_model.py       # Model training
│   └── run_pipeline.py      # Complete pipeline
├── 📁 tests/                # Unit tests
├── 📄 .env                  # 🔒 API KEYS (NEVER commit to git)
├── 📄 .env.example          # Template for env vars
├── 📄 requirements.txt      # Python dependencies
├── 📄 Dockerfile            # Docker config
├── 📄 docker-compose.yml    # Multi-container setup
└── 📄 README.md             # Documentation
```

---

## 🔧 **TECH STACK**

### **Core:**
- **Python 3.9+**, Streamlit 1.28+, Plotly 5.17+
- **Pandas 2.1+**, NumPy 1.24+, Scikit-learn 1.3+
- **Binance API** (python-binance), Yahoo Finance (yfinance)
- **XGBoost 1.7+**, TensorFlow 2.1+ (optional)
- **NLP:** Transformers 4.35+, NLTK, TextBlob
- **Database:** PostgreSQL (optional), Redis (caching)

### **Key Dependencies:**
```txt
python-binance==1.0.19
yfinance==0.2.28
streamlit==1.28.0
plotly==5.17.0
scikit-learn==1.3.0
xgboost==1.7.6
torch==2.1.0
transformers==4.35.0
pandas==2.1.3
```

---

## 🔐 **SECURITY & CONFIGURATION**

### **Environment Variables (.env file):**
```env
# 🔒 API KEYS (NEVER commit to GitHub!)
BINANCE_API_KEY=your_actual_key_here
BINANCE_API_SECRET=your_actual_secret_here

# Optional
NEWSAPI_KEY=your_newsapi_key
POSTGRES_DB=stockdb
POSTGRES_USER=admin
POSTGRES_PASSWORD=admin123
REDIS_HOST=localhost
```

### **Security Rules:**
1. ✅ **ALWAYS** use `os.getenv()` for API keys (NEVER hardcode)
2. ✅ Store real keys ONLY in `.env` file
3. ✅ `.env` is in `.gitignore` - NEVER commit to GitHub
4. ✅ Binance API: Read Only + Trading permissions, NO Withdrawal
5. ✅ Enable IP restrictions on Binance API

---

## 📊 **DATA STRUCTURES**

### **Price Data (DataFrame):**
```python
{
    'timestamp': pd.Timestamp,
    'open': float,
    'high': float,
    'low': float,
    'close': float,
    'volume': float,
    'returns': float,
    'rsi': float,
    'macd': float,
    # ... 20+ technical indicators
}
```

### **Sentiment Data:**
```python
{
    'text': str,
    'sentiment': str,  # 'positive', 'negative', 'neutral'
    'sentiment_score': float,  # -1.0 to +1.0
    'confidence': float,
    'model': str,  # 'vader', 'finbert', 'textblob'
    'timestamp': pd.Timestamp
}
```

### **Model Prediction:**
```python
{
    'symbol': str,
    'prediction': float,  # 0=down, 1=up
    'confidence': float,
    'horizon': int,  # hours ahead
    'timestamp': pd.Timestamp
}
```

---

## 🎯 **KEY CLASSES & FUNCTIONS**

### **1. Config (app/config.py)**
```python
# Usage: from app.config import config
config.BINANCE_API_KEY          # Get API key from .env
config.validate_config()        # Check required keys
config.SYMBOLS                  # ['BTCUSDT', 'ETHUSDT', ...]
config.get_database_url()       # PostgreSQL connection
```

### **2. DataCollector (app/data_collector.py)**
```python
collector = DataCollector()
collector.get_realtime_price('BTCUSDT')          # Current price
collector.get_historical_data('BTCUSDT', '1h', 7) # 7 days hourly data
collector.get_market_summary('BTCUSDT')          # 24h stats
```

### **3. SentimentAnalyzer (app/sentiment_analyzer.py)**
```python
analyzer = SentimentAnalyzer(model_type='vader')
result = analyzer.analyze("Bitcoin price surges!")
df = analyzer.analyze_batch(["text1", "text2"])
news_df, stats = analyzer.get_news_sentiment(limit=10)
```

### **4. ModelTrainer (app/model_trainer.py)**
```python
trainer = ModelTrainer(model_type='xgboost')
trainer.train_complete_pipeline(symbol='BTCUSDT', days=90)
predictions = trainer.predict(X_test)
metrics = trainer.evaluate(X_test, y_test)
trainer.save_model()  # Saves to ml_models/
```

### **5. Main Dashboard (app/main.py)**
```python
# Run with: streamlit run app/main.py
# Features: Real-time charts, sentiment gauges, predictions, news
```

---

## ⚙️ **CODING CONVENTIONS**

### **Naming:**
```python
# Variables: snake_case
current_price = 50000.0

# Classes: PascalCase
class DataCollector:

# Constants: UPPER_SNAKE_CASE
MAX_RETRIES = 3

# Functions: snake_case
def calculate_returns(prices):
```

### **Error Handling:**
```python
try:
    data = collector.get_historical_data(symbol, timeframe, days)
except ConnectionError as e:
    logger.error(f"API failed: {e}")
    return load_cached_data(symbol)
except Exception as e:
    logger.exception(f"Unexpected: {e}")
    raise
```

### **Type Hints (Always Use):**
```python
from typing import Dict, List, Optional

def get_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """Fetch data with type hints."""
    pass
```

---

## 🔄 **WORKFLOW**

### **Development:**
1. **Setup:** `python -m venv venv && source venv/bin/activate`
2. **Install:** `pip install -r requirements.txt`
3. **Configure:** `cp .env.example .env` (add real keys)
4. **Test:** `python scripts/collect_data.py --symbol BTCUSDT`
5. **Train:** `python scripts/train_model.py --symbol BTCUSDT`
6. **Run:** `streamlit run app/main.py`

### **Common Commands:**
```bash
# Train model for all symbols
python scripts/train_model.py --all

# Collect data every 5 minutes
python scripts/collect_data.py --schedule --interval 5

# Run complete pipeline
python scripts/run_pipeline.py

# Run tests
python -m pytest tests/
```

---

## 🐛 **TROUBLESHOOTING**

### **Common Issues:**
| Problem | Solution |
|---------|----------|
| **ModuleNotFoundError** | `pip install -r requirements.txt` |
| **Binance API Error** | Check `.env` file, verify API keys |
| **No data returned** | Check internet, API limits |
| **Streamlit not loading** | Check port 8501 |
| **Memory error** | Reduce `days` parameter |

### **Debug Commands:**
```python
# Test API keys
from app.config import config
print("Binance Key exists:", bool(config.BINANCE_API_KEY))

# Test data collection
from app.data_collector import DataCollector
dc = DataCollector()
print("BTC Price:", dc.get_realtime_price('BTCUSDT'))
```

---

## 🎯 **FOR COPILOT - CODING RULES**

### **ALWAYS Follow These:**
1. ✅ Use `config.BINANCE_API_KEY` not hardcoded keys
2. ✅ Add type hints to functions
3. ✅ Include error handling (try-except)
4. ✅ Use existing patterns from similar files
5. ✅ Follow project structure and naming
6. ✅ Add logging for important operations
7. ✅ Cache expensive API calls (Redis if available)
8. ✅ Validate inputs before processing
9. ✅ Document complex functions with comments
10. ✅ Test edge cases

### **When Creating New Files:**
- Start with module docstring explaining purpose
- Import from `app.config` for configuration
- Follow existing patterns in similar files
- Add to appropriate directory (app/, scripts/, tests/)
- Update this context file if adding major features

---

## 📞 **IMPORTANT LINKS**

### **Files:**
- **Entry Point:** `app/main.py`
- **Configuration:** `app/config.py` + `.env`
- **Data Collection:** `app/data_collector.py`
- **Sentiment:** `app/sentiment_analyzer.py`
- **ML Models:** `app/model_trainer.py`

### **External Docs:**
- **Binance API:** https://binance-docs.github.io/apidocs/
- **Streamlit:** https://docs.streamlit.io/
- **Plotly:** https://plotly.com/python/
- **XGBoost:** https://xgboost.readthedocs.io/

---

## 🎯 **PROJECT VALUES**
- ✅ **Security First:** Never expose API keys
- ✅ **Modular Design:** Each component independent
- ✅ **Production Ready:** Error handling, logging
- ✅ **User Friendly:** Clean dashboard, easy setup
- ✅ **Extensible:** Easy to add new features

---

**📌 KEEP THIS FILE OPEN IN VS CODE - COPILOT WILL USE IT FOR CONTEXT AUTOMATICALLY!**

**Now Copilot understands your entire project without needing explanations each time!** 🚀