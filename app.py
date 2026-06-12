import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Tesla Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

# ---------------- CUSTOM CSS FOR STOCK MARKET THEME ----------------

st.markdown("""
<style>
/* Main background with gradient */
.stApp {
    background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f132c 100%);
}

/* Animated ticker tape */
@keyframes ticker {
    0% { transform: translateX(100%); }
    100% { transform: translateX(-100%); }
}

.ticker-tape {
    background: linear-gradient(90deg, #1a472a 0%, #2e7d32 50%, #1a472a 100%);
    color: #4caf50;
    padding: 8px;
    border-radius: 5px;
    margin-bottom: 20px;
    overflow: hidden;
    white-space: nowrap;
    border: 1px solid #4caf50;
    box-shadow: 0 0 10px rgba(76,175,80,0.3);
}

.ticker-content {
    display: inline-block;
    animation: ticker 25s linear infinite;
    font-family: monospace;
    font-weight: bold;
    font-size: 16px;
}

.ticker-content span {
    margin: 0 20px;
}

/* Glowing text effect */
@keyframes glow {
    0% { text-shadow: 0 0 5px #4caf50; }
    50% { text-shadow: 0 0 20px #4caf50, 0 0 30px #2e7d32; }
    100% { text-shadow: 0 0 5px #4caf50; }
}

.glow-text {
    animation: glow 2s ease-in-out infinite;
}

/* Stock card styling */
.stock-card {
    background: linear-gradient(135deg, rgba(30,40,60,0.9) 0%, rgba(20,30,50,0.9) 100%);
    border-radius: 15px;
    padding: 20px;
    margin: 10px 0;
    border: 1px solid rgba(76,175,80,0.3);
    backdrop-filter: blur(10px);
    transition: transform 0.3s, box-shadow 0.3s;
}

.stock-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 30px rgba(76,175,80,0.2);
    border-color: #4caf50;
}

/* Blinking cursor for live feel */
@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0; }
}

.live-badge {
    background: #e82127;
    color: white;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 12px;
    font-weight: bold;
    display: inline-block;
    animation: blink 1s infinite;
}

/* Main text */
h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
    color: white !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f132c 0%, #1a1f3a 100%);
    border-right: 1px solid #2e7d32;
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: #4caf50 !important;
    font-size: 32px !important;
    font-weight: bold !important;
}

[data-testid="stMetricLabel"] {
    color: #E82127 !important;
    font-weight: bold !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2e7d32 0%, #1a472a 100%);
    color: white;
    border: none;
    transition: all 0.3s;
}

.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 15px rgba(76,175,80,0.5);
}

/* Dataframe */
.dataframe {
    background: rgba(20,30,50,0.8) !important;
    border-radius: 10px !important;
}

/* Success message */
.stSuccess {
    background: linear-gradient(135deg, #1a472a, #2e7d32) !important;
    border: none !important;
}

/* Select box */
.stSelectbox label {
    color: white !important;
}

/* Progress bar */
.stProgress > div > div {
    background-color: #4caf50 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HELPER FUNCTIONS ----------------

def create_ticker_tape():
    """Create animated ticker tape with stock prices"""
    stocks = [
        "🚗 TSLA: $248.50 ▲ +2.3%",
        "📱 AAPL: $175.34 ▼ -0.5%",
        "💻 MSFT: $378.85 ▲ +1.2%",
        "🔍 GOOGL: $138.21 ▲ +0.8%",
        "🛒 AMZN: $145.67 ▼ -0.3%",
        "🐦 META: $335.45 ▲ +1.7%",
        "⚡ NVDA: $485.67 ▲ +3.2%",
        "💊 JNJ: $156.78 ▼ -0.2%",
        "🏦 JPM: $142.34 ▲ +0.6%"
    ]
    
    ticker_html = f"""
    <div class="ticker-tape">
        <div class="ticker-content">
            {' '.join([f'<span>{stock}</span>' for stock in stocks])}
            {' '.join([f'<span>{stock}</span>' for stock in stocks])}
        </div>
    </div>
    """
    st.markdown(ticker_html, unsafe_allow_html=True)

def display_live_clock():
    """Display live clock and market status"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Check if market is open (9:30 AM - 4:00 PM EST)
    current_hour = datetime.now().hour
    market_status = "OPEN" if 9 <= current_hour <= 16 else "CLOSED"
    status_color = "#4caf50" if market_status == "OPEN" else "#e82127"
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown(f"""
        <div style="text-align: center; background: rgba(0,0,0,0.5); padding: 10px; border-radius: 10px; margin: 10px 0;">
            <span style="color: white;">🕐 Market Time: </span>
            <span style="color: #4caf50; font-weight: bold;">{current_time}</span>
            <span style="color: {status_color}; margin-left: 10px;">● Market {market_status}</span>
            <span class="live-badge" style="margin-left: 10px;">LIVE</span>
        </div>
        """, unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    df = pd.read_csv("TSLA.csv")
    return df

try:
    df = load_data()
    data_loaded = True
except:
    st.error("❌ Error loading TSLA.csv file. Please make sure the file exists in the same directory.")
    data_loaded = False
    df = pd.DataFrame()

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_tsla_model():
    try:
        model = load_model("simple_rnn.keras")
        return model, True
    except:
        return None, False

if data_loaded:
    model, model_loaded = load_tsla_model()
else:
    model, model_loaded = None, False

# ---------------- PREDICTION FUNCTION ----------------

def future_prediction(model, last_sequence, future_days, scaler):
    temp_input = last_sequence.flatten().tolist()
    predictions = []
    
    progress_bar = st.progress(0)
    
    for i in range(future_days):
        x_input = np.array(temp_input[-60:])
        x_input = x_input.reshape(1,60,1)
        yhat = model.predict(x_input, verbose=0)
        pred = yhat[0][0]
        predictions.append(pred)
        temp_input.append(pred)
        progress_bar.progress((i + 1) / future_days)
    
    predictions = np.array(predictions).reshape(-1,1)
    return scaler.inverse_transform(predictions)

# ---------------- SIDEBAR ----------------

# Try to add Tesla logo (optional)
try:
    st.sidebar.image("https://www.freepnglogos.com/uploads/tesla-logo-png-20.png", width=150)
except:
    st.sidebar.markdown("# 🚗 TESLA")

st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "📊 Dataset",
        "📈 EDA",
        "🤖 Model Performance",
        "🔮 Prediction",
        "📋 Conclusion"
    ]
)

# Remove emoji from page name for logic
page_clean = page.split(" ")[1] if len(page.split(" ")) > 1 else page

# Check if data and model are loaded before proceeding
if not data_loaded:
    st.error("⚠️ Cannot proceed without data file. Please upload TSLA.csv")
    st.stop()

if page_clean != "Home" and not model_loaded:
    st.error("⚠️ Model file 'simple_rnn.keras' not found. Please check the file path.")
    st.stop()

# ---------------- HOME PAGE ----------------

if page_clean == "Home" or page == "🏠 Home":
    create_ticker_tape()
    display_live_clock()
    
    st.title("🚗 Tesla Stock Price Prediction")
    st.markdown("<h2 style='text-align: center;' class='glow-text'>Using SimpleRNN and LSTM</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate some real metrics
    current_price = df['Close'].iloc[-1]
    if len(df) > 1:
        price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
    else:
        price_change = 0
    
    with col1:
        st.metric("📈 Records", f"{len(df):,}")
    
    with col2:
        st.metric("📊 Features", len(df.columns))
    
    with col3:
        st.metric("🏆 Best Model", "SimpleRNN")
    
    with col4:
        st.metric("💰 Current Price", f"${current_price:.2f}", f"{price_change:+.2f}%")
    
    st.markdown("---")
    
    # Project Overview
    st.subheader("🚗 Project Overview")
    
    st.markdown("""
    <div class='stock-card'>
        <h4>🎯 Objective</h4>
        <p>This project predicts Tesla stock closing prices using Deep Learning models including SimpleRNN and LSTM.</p>
        
        <h4>📅 Forecast Horizon</h4>
        <p>The model forecasts future stock prices for:</p>
        <ul>
            <li>✅ 1 Day</li>
            <li>✅ 5 Days</li>
            <li>✅ 10 Days</li>
        </ul>
        
        <h4>🏆 Best Performance</h4>
        <p>The best performing model was <strong style='color: #4caf50'>SimpleRNN</strong> with <strong style='color: #4caf50'>97.31% R² Score</strong></p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- DATASET PAGE ----------------

elif page_clean == "Dataset" or page == "📊 Dataset":
    create_ticker_tape()
    
    st.title("📊 Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class='stock-card'>
            <h4>📋 Dataset Information</h4>
            <p><strong>Shape:</strong> {df.shape[0]} rows × {df.shape[1]} columns</p>
            <p><strong>Features:</strong> {', '.join(df.columns[:5])}...</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='stock-card'>
            <h4>📅 Data Coverage</h4>
            <p><strong>Total Records:</strong> {len(df)}</p>
            <p><strong>Date Range:</strong> First to Last available</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    tab1, tab2 = st.tabs(["📊 Statistics", "🔍 Missing Values"])
    
    with tab1:
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
    
    with tab2:
        st.subheader("Missing Values")
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing Values': df.isnull().sum(),
            'Data Type': df.dtypes.values
        })
        st.dataframe(missing_df, use_container_width=True)

# ---------------- EDA PAGE ----------------

elif page_clean == "EDA" or page == "📈 EDA":
    create_ticker_tape()
    
    st.title("📈 Exploratory Data Analysis")
    
    st.subheader("📉 Tesla Stock Price Trends")
    
    # Price chart with volume
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(df.index, df['Close'], color='#4caf50', linewidth=2, label='Close Price')
    ax1.fill_between(df.index, df['Close'].min(), df['Close'], alpha=0.3, color='#4caf50')
    ax1.set_ylabel('Price (USD)', color='white')
    ax1.set_title('Tesla Closing Price History', color='white', fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_facecolor('#0E1117')
    
    ax2.bar(df.index, df['Volume'], color='#e82127', alpha=0.7, label='Volume')
    ax2.set_ylabel('Volume', color='white')
    ax2.set_xlabel('Trading Days', color='white')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_facecolor('#0E1117')
    
    fig.patch.set_facecolor('#0E1117')
    st.pyplot(fig)
    
    # Moving averages
    st.subheader("📊 Moving Average Analysis")
    
    temp = df.copy()
    temp['MA20'] = temp['Close'].rolling(20).mean()
    temp['MA50'] = temp['Close'].rolling(50).mean()
    temp['MA200'] = temp['Close'].rolling(200).mean()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(temp.index, temp['Close'], label='Close Price', linewidth=1.5, alpha=0.7)
    ax.plot(temp.index, temp['MA20'], label='MA20', linewidth=2)
    ax.plot(temp.index, temp['MA50'], label='MA50', linewidth=2)
    ax.plot(temp.index, temp['MA200'], label='MA200', linewidth=2)
    ax.set_title('Moving Averages (20, 50, 200 days)', color='white', fontsize=14)
    ax.set_xlabel('Trading Days', color='white')
    ax.set_ylabel('Price (USD)', color='white')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_facecolor('#0E1117')
    fig.patch.set_facecolor('#0E1117')
    st.pyplot(fig)

# ---------------- MODEL PERFORMANCE PAGE ----------------

elif page_clean == "Model" or page == "🤖 Model Performance":
    create_ticker_tape()
    
    st.title("🤖 Model Performance Analysis")
    
    comparison = pd.DataFrame({
        "Model": ["SimpleRNN", "LSTM"],
        "RMSE": [11.95, 25.53],
        "MAE": [8.11, 20.70],
        "R2 Score": ["97.31%", "87.71%"]
    })
    
    st.dataframe(comparison, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        metrics = ['RMSE', 'MAE']
        simple_rnn_values = [11.95, 8.11]
        lstm_values = [25.53, 20.70]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, simple_rnn_values, width, label='SimpleRNN', color='#4caf50')
        ax.bar(x + width/2, lstm_values, width, label='LSTM', color='#e82127')
        
        ax.set_xlabel('Metrics', color='white')
        ax.set_ylabel('Error Value', color='white')
        ax.set_title('Model Performance Comparison', color='white')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_facecolor('#0E1117')
        fig.patch.set_facecolor('#0E1117')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(6, 5))
        models = ['SimpleRNN', 'LSTM']
        r2_scores = [0.9731, 0.8771]
        colors = ['#4caf50', '#e82127']
        
        bars = ax.bar(models, r2_scores, color=colors, alpha=0.8)
        ax.set_ylim(0.8, 1.0)
        ax.set_ylabel('R² Score', color='white')
        ax.set_title('Model Accuracy Comparison', color='white', fontsize=14)
        ax.set_facecolor('#0E1117')
        fig.patch.set_facecolor('#0E1117')
        
        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.003,
                    f'{score:.2%}', ha='center', va='bottom', color='white', fontweight='bold')
        
        st.pyplot(fig)
    
    st.success("🎉 **Winner: SimpleRNN** - Achieved 97.31% accuracy in predicting Tesla stock prices!")

# ---------------- PREDICTION PAGE ----------------

elif page_clean == "Prediction" or page == "🔮 Prediction":
    create_ticker_tape()
    
    st.title("🔮 Future Stock Price Prediction")
    
    if not model_loaded:
        st.error("❌ Model not loaded. Please check if 'simple_rnn.keras' exists.")
        st.stop()
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close']])
    last_60_days = scaled_data[-60:]
    last_60_days = last_60_days.reshape(1, 60, 1)
    
    days = st.selectbox(
        "📅 Select Forecast Horizon",
        [1, 5, 10],
        format_func=lambda x: f"{x} Day{'s' if x > 1 else ''}"
    )
    
    st.markdown("---")
    
    if st.button("🔮 Generate Prediction", use_container_width=True):
        with st.spinner("Analyzing market data and generating predictions..."):
            result = future_prediction(model, last_60_days, days, scaler)
        
        st.success(f"✅ Prediction generated successfully for next {days} day(s)!")
        
        st.subheader(f"📊 Forecast for Next {days} Day(s)")
        
        prediction_df = pd.DataFrame(result, columns=["Predicted Price ($)"])
        prediction_df.index = [f"Day {i+1}" for i in range(days)]
        
        st.dataframe(prediction_df, use_container_width=True)
        
        # Calculate predicted growth
        start_price = result[0][0]
        end_price = result[-1][0]
        growth = ((end_price - start_price) / start_price) * 100
        growth_color = "#4caf50" if growth > 0 else "#e82127"
        
        st.markdown(f"""
        <div class='stock-card'>
            <h4>📈 Prediction Summary</h4>
            <p><strong>Start Price (Day 1):</strong> <span style='color: #4caf50;'>${start_price:.2f}</span></p>
            <p><strong>End Price (Day {days}):</strong> <span style='color: {growth_color};'>${end_price:.2f}</span></p>
            <p><strong>Expected Change:</strong> <span style='color: {growth_color};'>{growth:+.2f}%</span></p>
            <p><strong>Market Sentiment:</strong> <span style='color: {growth_color};'>{'🟢 BULLISH' if growth > 0 else '🔴 BEARISH'}</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Historical context (last 30 days)
        historical = df['Close'].iloc[-30:].values
        ax.plot(range(-30, 0), historical, color='#e82127', linewidth=2, label='Historical (Last 30 days)')
        
        # Prediction
        pred_days = range(1, days + 1)
        ax.plot(pred_days, result.flatten(), color='#4caf50', linewidth=2, marker='o', markersize=8, label='Predicted')
        
        ax.axvline(x=0, color='white', linestyle='--', alpha=0.5)
        ax.set_title(f'Tesla Stock Price Forecast - Next {days} Days', color='white', fontsize=14)
        ax.set_xlabel('Days', color='white')
        ax.set_ylabel('Price (USD)', color='white')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_facecolor('#0E1117')
        fig.patch.set_facecolor('#0E1117')
        
        st.pyplot(fig)

# ---------------- CONCLUSION PAGE ----------------

elif page_clean == "Conclusion" or page == "📋 Conclusion":
    create_ticker_tape()
    
    st.title("📋 Project Conclusion")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class='stock-card'>
            <h3 style='color: #4caf50;'>✅ Key Achievements</h3>
            <ul>
                <li>Successfully predicted Tesla stock prices using Deep Learning</li>
                <li>Achieved <strong>97.31% R² Score</strong> with SimpleRNN</li>
                <li>RMSE reduced from 25.53 to <strong>11.95</strong></li>
                <li>MAE reduced from 20.70 to <strong>8.11</strong></li>
                <li>Interactive web application for real-time predictions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='stock-card'>
            <h3 style='color: #ff9800;'>📈 Future Enhancements</h3>
            <ul>
                <li>Sentiment analysis from news/social media</li>
                <li>Additional technical indicators (RSI, MACD)</li>
                <li>Ensemble methods for better accuracy</li>
                <li>Real-time data fetching from APIs</li>
                <li>Risk assessment metrics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class='stock-card'>
        <h4>⚠️ Disclaimer</h4>
        <p style='color: #ff9800; font-size: 12px;'>This tool is for educational purposes only. Stock market investments carry risks. 
        Always consult with financial advisors before making investment decisions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <p style='color: #4caf50;'>🚗 Built with Streamlit • TensorFlow • Python 🐍</p>
        <p style='color: #666; font-size: 12px;'>© 2024 Tesla Stock Prediction Project</p>
    </div>
    """, unsafe_allow_html=True)