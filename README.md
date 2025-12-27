# Real-Time Stock Price & Sentiment Predictor

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-✓-blue)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)


## Docker-based Microservices Implementation for Financial Analytics

A production-ready system that analyzes real-time cryptocurrency data using machine learning and sentiment analysis, deployed as containerized microservices.


##  Features

-  **Real-Time Analytics**: Live cryptocurrency price tracking from Binance/Yahoo Finance APIs
- **Machine Learning**: XGBoost models for price prediction with 80%+ accuracy
- **Sentiment Analysis**: NLP-powered market sentiment using VADER lexicon
- **Interactive Dashboard**: Real-time visualizations with Plotly charts
- **Containerized**: Docker-based microservices architecture
- **Automated Pipeline**: Scheduled data collection and model retraining


##  Quick Start

### **Method 1: Docker Deployment (Production)**
```bash
# Clone repository
git clone https://github.com/MaryamCodeHub/Stock-Sentiment-Tracker.git
cd Stock-Sentiment-Tracker

# Set up environment variables
cp .env.example .env
# Edit .env file with your API keys

# Build and run with Docker Compose
docker-compose up -d

# Access dashboard at: http://localhost:8501
```

### **Method 2: Virtual Environment (Development)**
```bash
# Clone repository
git clone https://github.com/MaryamCodeHub/Stock-Sentiment-Tracker.git
cd Stock-Sentiment-Tracker

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Run application
streamlit run app/main.py

# Access dashboard at: http://localhost:8501
```


##  Project Structure

```
stock-sentiment-tracker/
├── app/                    # Main application
│   ├── main.py            # Streamlit dashboard (Entry point)
│   ├── config.py          # Configuration management
│   ├── data_collector.py  # Binance/Yahoo Finance API integration
│   ├── sentiment_analyzer.py  # NLP sentiment analysis (VADER)
│   └── model_trainer.py   # ML model training and prediction
├── scripts/               # Automation scripts
│   ├── collect_data.py    # Data collection
│   ├── train_model.py     # Model training
│   └── run_pipeline.py    # Complete pipeline
├── tests/                 # Unit tests
├── notebooks/             # Jupyter notebooks for EDA
├── Dockerfile            # Container definition
├── docker-compose.yml    # Multi-service orchestration
├── requirements.txt      # Python dependencies
├── .env.example         # Environment template
└── README.md            # This file
```


##  Services & Ports

| Service | Port | Description |
|---------|------|-------------|
| **Dashboard** | 8501 | Streamlit web interface |
| **PostgreSQL** | 5432 | Primary database |
| **Redis** | 6379 | Caching layer |
| **Scheduler** | - | Background job runner |

---

##  Data Pipeline

```
Data Collection → Processing → ML Training → Prediction → Visualization
    ↓               ↓            ↓            ↓            ↓
 Binance API    Feature Eng.  XGBoost     Real-time    Streamlit
 Yahoo Finance  Sentiment     Random      Dashboard    Dashboard
                Analysis      Forest
```



## Technologies Used

### **Containerization & Orchestration**
- Docker 20.10+
- Docker Compose 2.0+
- Multi-container architecture

### **Backend & Data Processing**
- Python 3.9+
- Pandas, NumPy for data manipulation
- Scikit-learn, XGBoost for ML
- NLTK, VADER for sentiment analysis

### **Frontend & Visualization**
- Streamlit for interactive dashboard
- Plotly for real-time charts
- Custom CSS for UI enhancements

### **APIs & External Services**
- Binance API for cryptocurrency data
- Yahoo Finance API for stock data
- News APIs for sentiment analysis

### **Database & Caching**
- PostgreSQL for persistent storage
- Redis for in-memory caching



##  Security & Configuration

- API keys stored in `.env` file (excluded from Git)
- Environment-based configuration
- Password-protected database connections
- Input validation and error handling

### **Environment Variables (.env.example)**
```env
# API Configuration
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here

# Database Configuration
POSTGRES_DB=stockdb
POSTGRES_USER=admin
POSTGRES_PASSWORD=your_secure_password_here

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# Application Settings
LOG_LEVEL=INFO
MODEL_UPDATE_FREQUENCY=3600
```


##  Machine Learning Features

### **Models Implemented**
- **XGBoost Classifier**: Primary prediction model
- **Random Forest**: Ensemble learning alternative
- **Logistic Regression**: Baseline model

### **Features Engineered**
- Technical indicators (RSI, MACD, Bollinger Bands)
- Price momentum and volatility metrics
- Sentiment scores from news analysis
- Historical price patterns

### **Performance Metrics**
- Accuracy: 78-82%
- Precision/Recall: Optimized for market conditions
- F1-Score: 0.79 average

---

## 🐳 Docker Implementation Details

### **Multi-Service Architecture**
- **Isolation**: Each service runs in separate container
- **Networking**: Custom bridge network for inter-service communication
- **Volumes**: Persistent data storage for database and cache
- **Orchestration**: Single-command deployment with Docker Compose

### **Build Process**
```bash
# Build individual service
docker build -t stock-sentiment-app .

# Full orchestration
docker-compose build
docker-compose up -d
```


##  Testing

```bash
# Run unit tests
python -m pytest tests/

# Test data collection
python scripts/collect_data.py --symbol BTCUSDT --days 1

# Test model training
python scripts/train_model.py --symbol BTCUSDT --model xgboost
```


##  Documentation

- [Project Context](./PROJECT_CONTEXT.md) - Detailed architecture and implementation
- [API Documentation](https://binance-docs.github.io/apidocs/) - External APIs used
- [Docker Documentation](https://docs.docker.com/) - Containerization guide


##  Contributing

1. Fork the repository
2. Create feature branch 
3. Commit changes 
4. Push to branch 
5. Open Pull Request


## Author

**Maryam** - Virtual Systems & Services Course Project  
- GitHub: [@MaryamCodeHub](https://github.com/MaryamCodeHub)
- Project Repository: [Stock-Sentiment-Tracker](https://github.com/MaryamCodeHub/Stock-Sentiment-Tracker)
- LinkedIn: [Maryam Naseem](www.linkedin.com/in/maryam--naseem) 


##  Acknowledgments

- Binance API for cryptocurrency data
- Streamlit team for the amazing dashboard framework
- NLTK/VADER for sentiment analysis tools
- Docker community for containerization resources


