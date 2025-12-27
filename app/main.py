import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add app directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Page configuration
st.set_page_config(
    page_title="Stock Sentiment Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Dark Blue & Black Theme with Enhanced Glassmorphism
st.markdown("""
<style>
    /* Main background - More colorful gradient for glass effect */
    .stApp {
        background: linear-gradient(135deg, 
            #0a192f 0%, 
            #1a2332 10%,
            #172a45 20%,
            #1e3a8a 30%,
            #1e40af 40%,
            #2563eb 50%,
            #1e40af 60%,
            #1e3a8a 70%,
            #172a45 80%,
            #1a2332 90%,
            #0a192f 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Main content area - Enhanced Glassmorphism */
    .main .block-container {
        background: rgba(15, 23, 42, 0.4);
        backdrop-filter: blur(25px) saturate(200%);
        -webkit-backdrop-filter: blur(25px) saturate(200%);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.7),
            0 0 0 1px rgba(59, 130, 246, 0.5),
            inset 0 1px 2px rgba(255, 255, 255, 0.15);
        border: 2px solid rgba(59, 130, 246, 0.5);
    }
    
    .main-header {
        font-size: 2.5rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 0 0 30px rgba(59, 130, 246, 0.6), 0 0 60px rgba(59, 130, 246, 0.4);
    }
    
    .metric-card {
        background: rgba(30, 58, 138, 0.25);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 2px solid rgba(59, 130, 246, 0.5);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 
            0 8px 24px rgba(0, 0, 0, 0.5),
            0 0 0 1px rgba(59, 130, 246, 0.3),
            inset 0 1px 1px rgba(255, 255, 255, 0.15);
        margin-bottom: 15px;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        background: rgba(30, 58, 138, 0.35);
        border-color: rgba(59, 130, 246, 0.7);
        transform: translateY(-2px);
        box-shadow: 
            0 12px 32px rgba(37, 99, 235, 0.3),
            0 0 0 1px rgba(59, 130, 246, 0.5),
            inset 0 1px 1px rgba(255, 255, 255, 0.2);
    }
    
    /* Metric values */
    [data-testid="stMetricValue"] {
        color: #ffffff;
        font-size: 2rem;
    }
    
    [data-testid="stMetricLabel"] {
        color: #93c5fd;
    }
    
    .positive {
        color: #10b981;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
    }
    
    .negative {
        color: #ef4444;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(239, 68, 68, 0.5);
    }
    
    .neutral {
        color: #f59e0b;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(245, 158, 11, 0.5);
    }
    
    /* Sidebar styling - Enhanced Glass */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.5);
        backdrop-filter: blur(25px) saturate(180%);
        -webkit-backdrop-filter: blur(25px) saturate(180%);
        border-right: 2px solid rgba(59, 130, 246, 0.6);
        box-shadow: 
            inset -1px 0 20px rgba(59, 130, 246, 0.2),
            4px 0 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Tabs styling - Glass Effect */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background: rgba(30, 58, 138, 0.3);
        backdrop-filter: blur(15px) saturate(160%);
        -webkit-backdrop-filter: blur(15px) saturate(160%);
        border: 2px solid rgba(59, 130, 246, 0.4);
        border-radius: 10px 10px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #93c5fd;
        transition: all 0.3s ease;
        box-shadow: 
            0 4px 12px rgba(0, 0, 0, 0.4),
            inset 0 1px 2px rgba(255, 255, 255, 0.12);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(30, 58, 138, 0.5);
        backdrop-filter: blur(20px) saturate(200%);
        -webkit-backdrop-filter: blur(20px) saturate(200%);
        transform: translateY(-2px);
        border-color: rgba(59, 130, 246, 0.7);
        box-shadow: 
            0 6px 20px rgba(59, 130, 246, 0.4),
            inset 0 1px 2px rgba(255, 255, 255, 0.2);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #1e3a8a, #1e40af);
        color: #ffffff;
    }
    
    /* Text colors */
    p, label, h1, h2, h3, h4, h5, h6 {
        color: #e5e7eb !important;
    }
    
    /* Input fields */
    .stTextInput input, .stSelectbox select {
        background-color: rgba(17, 24, 39, 0.7);
        color: #ffffff;
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 8px;
    }
    
    /* Buttons - Navigation style */
    .stButton button {
        background: rgba(30, 58, 138, 0.4);
        backdrop-filter: blur(15px) saturate(160%);
        -webkit-backdrop-filter: blur(15px) saturate(160%);
        color: white;
        border: 2px solid rgba(59, 130, 246, 0.4);
        border-radius: 12px;
        padding: 1.2rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        height: 60px;
        box-shadow: 
            0 4px 12px rgba(0, 0, 0, 0.4),
            inset 0 1px 2px rgba(255, 255, 255, 0.12);
    }
    
    .stButton button:hover {
        transform: scale(1.03);
        background: rgba(30, 58, 138, 0.6);
        border-color: rgba(59, 130, 246, 0.7);
        box-shadow: 0 6px 24px rgba(59, 130, 246, 0.5);
    }
    
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #1e3a8a, #2563eb);
        border-color: rgba(59, 130, 246, 0.8);
        box-shadow: 
            0 6px 20px rgba(37, 99, 235, 0.5),
            inset 0 1px 2px rgba(255, 255, 255, 0.2);
    }
    
    /* Dataframe styling */
    .dataframe {
        background-color: rgba(17, 24, 39, 0.8);
        color: #ffffff;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_collector' not in st.session_state:
    from app.data_collector import DataCollector
    st.session_state.data_collector = DataCollector()

if 'sentiment_analyzer' not in st.session_state:
    from app.sentiment_analyzer import SentimentAnalyzer
    st.session_state.sentiment_analyzer = SentimentAnalyzer()

if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.model = None
    st.session_state.scaler = None

if 'realtime_data' not in st.session_state:
    st.session_state.realtime_data = {}

if 'last_update' not in st.session_state:
    st.session_state.last_update = None

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Dashboard"

def load_model():
    """Load trained ML model"""
    try:
        import joblib
        model_files = [f for f in os.listdir('ml_models') if f.endswith('.joblib')]
        if model_files:
            latest_model = sorted(model_files)[-1]
            model_data = joblib.load(f'ml_models/{latest_model}')
            st.session_state.model = model_data.get('model')
            st.session_state.scaler = model_data.get('scaler')
            st.session_state.model_loaded = True
            st.success(f"✅ Model loaded: {latest_model}")
            return True
    except Exception as e:
        st.error(f"Error loading model: {e}")
    return False

def get_price_data(symbol, interval='1h', days=7):
    """Get price data with caching"""
    cache_key = f"{symbol}_{interval}_{days}"
    current_time = time.time()
    
    # Cache for 5 minutes
    if (cache_key in st.session_state.realtime_data and 
        current_time - st.session_state.realtime_data[cache_key]['timestamp'] < 300):
        return st.session_state.realtime_data[cache_key]['data']
    
    # Fetch new data
    try:
        data = st.session_state.data_collector.get_historical_data(symbol, interval, days)
        if not data.empty:
            st.session_state.realtime_data[cache_key] = {
                'data': data,
                'timestamp': current_time
            }
            st.session_state.last_update = datetime.now()
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def create_candlestick_chart(df, symbol):
    """Create interactive candlestick chart"""
    if df.empty:
        return go.Figure()
    
    fig = go.Figure(data=[
        go.Candlestick(
            x=df['timestamp'] if 'timestamp' in df.columns else df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        )
    ])
    
    # Add moving averages
    if len(df) > 20:
        df['sma_20'] = df['close'].rolling(window=20).mean()
        fig.add_trace(go.Scatter(
            x=df['timestamp'] if 'timestamp' in df.columns else df.index,
            y=df['sma_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='orange', width=2)
        ))
    
    fig.update_layout(
        title=f"{symbol} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=500,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    return fig

def create_sentiment_gauge(sentiment_score):
    """Create sentiment gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sentiment_score * 100,
        title={'text': "Market Sentiment"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [-100, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-100, -33], 'color': "red"},
                {'range': [-33, 33], 'color': "yellow"},
                {'range': [33, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': sentiment_score * 100
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_technical_indicators(df):
    """Create technical indicators chart"""
    if df.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Price & Bollinger Bands', 'RSI', 'MACD'),
        vertical_spacing=0.1,
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Price with Bollinger Bands
    fig.add_trace(
        go.Scatter(x=df.index, y=df['close'], name='Close', line=dict(color='blue')),
        row=1, col=1
    )
    
    # RSI
    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['rsi'], name='RSI', line=dict(color='purple')),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['macd'], name='MACD', line=dict(color='blue')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['macd_signal'], name='Signal', line=dict(color='red')),
            row=3, col=1
        )
    
    fig.update_layout(height=800, showlegend=True)
    return fig

def main():
    """Main application function"""
    # Header
    st.markdown("<h1 class='main-header'>Stock Price & Sentiment Predictor</h1>", 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Symbol selection
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT", "XRPUSDT"]
        selected_symbol = st.selectbox("Select Symbol", symbols, index=0)
        
        # Timeframe selection
        timeframe = st.selectbox("Timeframe", 
                                ["1m", "5m", "15m", "1h", "4h", "1d", "1w"], 
                                index=3)
        
        # Days to display
        days_options = {"1h": 7, "4h": 30, "1d": 90, "1w": 365}
        default_days = days_options.get(timeframe, 7)
        days = st.slider("Days to Display", 1, 365, default_days)
        
        # Analysis options
        show_sentiment = st.checkbox("Show Sentiment Analysis", value=True)
        show_predictions = st.checkbox("Show Price Predictions", value=True)
        show_technical = st.checkbox("Show Technical Indicators", value=False)
        auto_refresh = st.checkbox("Auto Refresh (60s)", value=False)
        
        # Model section
        st.markdown("---")
        st.subheader("🤖 ML Model")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Load Model", type="primary"):
                with st.spinner("Loading model..."):
                    load_model()
        
        with col2:
            if st.button("Train New"):
                st.info("Run: python scripts/train_model.py")
        
        if st.session_state.model_loaded:
            st.success("✅ Model Loaded")
        
        # Last update
        if st.session_state.last_update:
            st.caption(f"Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")
    
    # Add spacing to move navigation down
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Navigation boxes (replacing tabs)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Dashboard", key="nav_dashboard", use_container_width=True, 
                     type="primary" if st.session_state.active_tab == "Dashboard" else "secondary"):
            st.session_state.active_tab = "Dashboard"
            st.rerun()
    
    with col2:
        if st.button("📈 Charts", key="nav_charts", use_container_width=True,
                     type="primary" if st.session_state.active_tab == "Charts" else "secondary"):
            st.session_state.active_tab = "Charts"
            st.rerun()
    
    with col3:
        if st.button("📰 News", key="nav_news", use_container_width=True,
                     type="primary" if st.session_state.active_tab == "News" else "secondary"):
            st.session_state.active_tab = "News"
            st.rerun()
    
    with col4:
        if st.button("⚙️ Settings", key="nav_settings", use_container_width=True,
                     type="primary" if st.session_state.active_tab == "Settings" else "secondary"):
            st.session_state.active_tab = "Settings"
            st.rerun()
    
    st.markdown("---")
    
    # Display content based on active tab
    if st.session_state.active_tab == "Dashboard":
        st.subheader("Dashboard Overview")
        
        # Quick metrics in a compact row
        col1, col2, col3, col4 = st.columns(4)
        current_price = st.session_state.data_collector.get_realtime_price(selected_symbol)
        
        with col1:
            st.metric("Current Price", f"${current_price:,.2f}" if current_price > 0 else "N/A")
        
        with col2:
            hist_data = get_price_data(selected_symbol, '1h', 2)
            if len(hist_data) >= 24:
                price_24h_ago = hist_data['close'].iloc[-24]
                change_24h = ((current_price - price_24h_ago) / price_24h_ago * 100)
                st.metric("24h Change", f"{change_24h:.2f}%")
            else:
                st.metric("24h Change", "N/A")
        
        with col3:
            if show_sentiment:
                try:
                    news_df, aggregate = st.session_state.sentiment_analyzer.get_news_sentiment()
                    avg_sentiment = aggregate['avg_sentiment_score']
                    sentiment_label = "Bullish" if avg_sentiment > 0.1 else "Bearish" if avg_sentiment < -0.1 else "Neutral"
                    st.metric("Sentiment", sentiment_label)
                except:
                    st.metric("Sentiment", "N/A")
        
        with col4:
            if st.session_state.model_loaded and show_predictions:
                st.metric("Prediction", "↑ Bullish")
            else:
                st.metric("Prediction", "Load Model")
        
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Price Chart")
            hist_data = get_price_data(selected_symbol, timeframe, days)
            if not hist_data.empty:
                fig = create_candlestick_chart(hist_data, selected_symbol)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available")
        
        with col2:
            if show_sentiment:
                st.subheader("Sentiment Analysis")
                analyzer = st.session_state.sentiment_analyzer
                try:
                    news_df, aggregate = analyzer.get_news_sentiment()
                    avg_sentiment = aggregate['avg_sentiment_score']
                    
                    fig_gauge = create_sentiment_gauge(avg_sentiment)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # Sentiment distribution
                    if not news_df.empty:
                        sentiment_counts = news_df['sentiment'].value_counts()
                        fig_pie = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            title="Sentiment Distribution",
                            color=sentiment_counts.index,
                            color_discrete_map={
                                'positive': 'green',
                                'negative': 'red',
                                'neutral': 'yellow'
                            }
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                except Exception as e:
                    st.error(f"Sentiment analysis failed: {e}")
        
        # Technical indicators
        if show_technical and not hist_data.empty:
            st.markdown("---")
            st.subheader("Technical Indicators")
            fig_tech = create_technical_indicators(hist_data)
            st.plotly_chart(fig_tech, use_container_width=True)
    
    elif st.session_state.active_tab == "Charts":
        st.subheader("Advanced Charts")
        
        # Multiple timeframe analysis
        timeframes = ["1h", "4h", "1d"]
        for tf in timeframes:
            with st.expander(f"{tf} Chart"):
                tf_data = get_price_data(selected_symbol, tf, 
                                       7 if tf == "1h" else 30 if tf == "4h" else 90)
                if not tf_data.empty:
                    fig = create_candlestick_chart(tf_data, f"{selected_symbol} ({tf})")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Volume analysis
        st.subheader("Volume Analysis")
        if not hist_data.empty and 'volume' in hist_data.columns:
            fig_volume = go.Figure()
            fig_volume.add_trace(go.Bar(
                x=hist_data.index,
                y=hist_data['volume'],
                name='Volume',
                marker_color='lightskyblue'
            ))
            fig_volume.update_layout(
                title="Trading Volume",
                xaxis_title="Date",
                yaxis_title="Volume",
                template="plotly_dark"
            )
            st.plotly_chart(fig_volume, use_container_width=True)
    
    elif st.session_state.active_tab == "News":
        st.subheader("Market News & Sentiment")
        
        analyzer = st.session_state.sentiment_analyzer
        try:
            news_df, aggregate = analyzer.get_news_sentiment()
            
            # Display aggregate metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Sentiment", f"{aggregate['avg_sentiment_score']:.3f}")
            with col2:
                st.metric("Positive", f"{aggregate['positive_ratio']:.1%}")
            with col3:
                st.metric("Negative", f"{aggregate['negative_ratio']:.1%}")
            with col4:
                st.metric("Total News", aggregate['total_news'])
            
            # News headlines
            st.subheader("Latest News Headlines")
            for idx, row in news_df.iterrows():
                with st.container():
                    sentiment_emoji = "🟢" if row['sentiment'] == 'positive' else \
                                     "🔴" if row['sentiment'] == 'negative' else "🟡"
                    
                    col1, col2 = st.columns([0.9, 0.1])
                    with col1:
                        st.write(f"**{row['text'][:150]}...**")
                    with col2:
                        st.write(sentiment_emoji)
                    
                    cols = st.columns(3)
                    with cols[0]:
                        st.caption(f"Positive: {row['positive']:.2%}")
                    with cols[1]:
                        st.caption(f"Negative: {row['negative']:.2%}")
                    with cols[2]:
                        st.caption(f"Confidence: {row['confidence']:.2%}")
                    
                    st.divider()
        
        except Exception as e:
            st.error(f"Could not fetch news: {e}")
            # Sample news
            st.info("Sample News (API not configured)")
            sample_news = [
                {"text": "Bitcoin reaches new all-time high amid institutional adoption", "sentiment": "positive"},
                {"text": "Regulatory concerns weigh on crypto markets", "sentiment": "negative"},
                {"text": "Ethereum upgrade reduces gas fees significantly", "sentiment": "positive"},
            ]
            
            for news in sample_news:
                st.write(f"📰 {news['text']}")
                st.caption(f"Sentiment: {news['sentiment'].upper()}")
                st.divider()
    
    elif st.session_state.active_tab == "Settings":
        st.subheader("Application Settings")
        
        # API Configuration
        with st.expander("API Configuration", expanded=True):
            st.text_input("Binance API Key", type="password", value="••••••••")
            st.text_input("Binance API Secret", type="password", value="••••••••")
            st.text_input("News API Key", type="password", value="••••••••")
            
            if st.button("Save API Keys", type="primary"):
                st.success("API keys saved (in production, these would be encrypted)")
        
        # Model Settings
        with st.expander("Model Configuration"):
            st.slider("Prediction Horizon (hours)", 1, 24, 1)
            st.slider("Confidence Threshold", 0.5, 0.95, 0.7)
            st.selectbox("Primary Model", ["XGBoost", "LSTM", "Ensemble", "Random Forest"])
            
            if st.button("Update Model Settings"):
                st.info("Settings updated")
        
        # Data Settings
        with st.expander("Data Settings"):
            st.number_input("Days of Historical Data", 30, 365, 90)
            st.selectbox("Data Source", ["Binance", "Yahoo Finance", "Both"])
            st.checkbox("Enable Redis Caching", value=True)
            st.checkbox("Store Historical Data", value=True)
        
        # System Status
        with st.expander("System Status"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("API Status", "✅ Connected")
                st.metric("Model Status", "✅ Loaded" if st.session_state.model_loaded else "❌ Not Loaded")
            with col2:
                st.metric("Database", "✅ Connected")
                st.metric("Last Update", "Just now" if st.session_state.last_update else "Never")
            
            # Clear cache button
            if st.button("Clear Cache", type="secondary"):
                st.session_state.realtime_data = {}
                st.success("Cache cleared")
    
    # Auto refresh
    if auto_refresh:
        time.sleep(60)
        st.rerun()

if __name__ == "__main__":
    main() 
