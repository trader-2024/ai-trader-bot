import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="AI Trading Bot Pro", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);}
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-size: 1.1em;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 1rem;
    }
    .trade-signal {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: bold;
        text-align: center;
        font-size: 1.3em;
    }
    .buy-signal {background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white;}
    .sell-signal {background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: white;}
    .hold-signal {background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white;}
    .info-box {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'live_trading' not in st.session_state:
    st.session_state.live_trading = False
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {
        'cash': 100000,
        'holdings': 0,
        'trades': [],
        'entry_price': 0
    }

st.title("🤖 AI Intraday Trading Bot Pro")
st.markdown("### Advanced ML-Powered Trading System")

st.sidebar.header("🎛️ Configuration")

mode = st.sidebar.radio("Select Mode", ["📊 Train Model", "🚀 Live Trading", "📈 Backtest", "📁 Data Manager"])

st.sidebar.markdown("---")

symbols = {
    "Indian Stocks (NSE)": {
        "RELIANCE.NS": "Reliance", "TCS.NS": "TCS", "INFY.NS": "Infosys",
        "HDFCBANK.NS": "HDFC Bank", "ICICIBANK.NS": "ICICI Bank", "SBIN.NS": "SBI"
    },
    "US Stocks": {"AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google", "TSLA": "Tesla"},
    "Crypto": {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum"}
}

stock_category = st.sidebar.selectbox("📊 Category", list(symbols.keys()))
symbol_dict = symbols[stock_category]
symbol_name = st.sidebar.selectbox("Symbol", list(symbol_dict.keys()), 
                                   format_func=lambda x: f"{x} - {symbol_dict[x]}")

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Trading Parameters")
initial_capital = st.sidebar.number_input("Initial Capital (₹)", value=100000, step=10000, min_value=10000)
position_size = st.sidebar.slider("Position Size (%)", 10, 100, 50)
stop_loss = st.sidebar.slider("Stop Loss (%)", 1, 10, 3)
take_profit = st.sidebar.slider("Take Profit (%)", 1, 20, 5)

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 ML Parameters")
lookback_period = st.sidebar.slider("Lookback Period", 10, 100, 30)
prediction_confidence = st.sidebar.slider("Min Confidence (%)", 50, 95, 70)

timeframe = st.sidebar.selectbox("⏰ Timeframe", ["1m", "5m", "15m", "30m", "1h"], index=2)

@st.cache_data
def fetch_intraday_data(symbol, days=60, interval='15m'):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        data = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
        if hasattr(data.columns, 'levels'):
            data.columns = data.columns.get_level_values(0)
        if not data.empty:
            data = data.reset_index()
            if 'Datetime' in data.columns:
                data.rename(columns={'Datetime': 'Date'}, inplace=True)
        return data
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def load_csv_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns: {', '.join(missing_cols)}")
            return None
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None

def merge_data(auto_data, csv_data):
    if auto_data is None:
        return csv_data
    if csv_data is None:
        return auto_data
    combined = pd.concat([csv_data, auto_data], ignore_index=True)
    combined = combined.sort_values('Date').drop_duplicates(subset=['Date'], keep='first')
    return combined

def calculate_technical_indicators(data):
    df = data.copy()
    df['Returns'] = df['Close'].pct_change()
    
    for period in [5, 10, 20, 50]:
        df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
        df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + bb_std * 2
    df['BB_Lower'] = df['BB_Middle'] - bb_std * 2
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stochastic'] = ((df['Close'] - low_14) / (high_14 - low_14)) * 100
    
    return df

def create_ml_features(data, lookback):
    df = data.copy()
    features = []
    
    for i in range(lookback, len(df)):
        feature_row = []
        feature_row.append(df['Close'].iloc[i] / df['Close'].iloc[i-1] - 1)
        feature_row.append(df['Close'].iloc[i] / df['SMA_20'].iloc[i] - 1)
        feature_row.append(df['SMA_5'].iloc[i] / df['SMA_20'].iloc[i] - 1)
        feature_row.append(df['EMA_10'].iloc[i] / df['EMA_20'].iloc[i] - 1)
        feature_row.append(df['RSI'].iloc[i] / 100)
        feature_row.append(df['MACD'].iloc[i])
        feature_row.append(df['MACD_Hist'].iloc[i])
        
        bb_position = (df['Close'].iloc[i] - df['BB_Lower'].iloc[i]) / (df['BB_Upper'].iloc[i] - df['BB_Lower'].iloc[i])
        feature_row.append(bb_position)
        feature_row.append(df['BB_Width'].iloc[i])
        feature_row.append(df['Volume_Ratio'].iloc[i])
        feature_row.append(df['ROC'].iloc[i] / 100)
        feature_row.append(df['Momentum'].iloc[i] / df['Close'].iloc[i])
        feature_row.append(df['Volatility'].iloc[i])
        feature_row.append(df['ATR'].iloc[i] / df['Close'].iloc[i])
        feature_row.append(df['Stochastic'].iloc[i] / 100)
        
        features.append(feature_row)
    
    return np.array(features)

def create_labels(data, lookback, future_periods=3, threshold=0.5):
    df = data.copy()
    labels = []
    
    for i in range(lookback, len(df) - future_periods):
        current_price = df['Close'].iloc[i]
        future_max = df['High'].iloc[i+1:i+future_periods+1].max()
        future_min = df['Low'].iloc[i+1:i+future_periods+1].min()
        
        upside = (future_max - current_price) / current_price * 100
        downside = (current_price - future_min) / current_price * 100
        
        if upside > threshold and upside > downside:
            labels.append(1)
        elif downside > threshold and downside > upside:
            labels.append(-1)
        else:
            labels.append(0)
    
    return np.array(labels)

def train_ml_model(data, lookback):
    with st.spinner("🧠 Training AI model..."):
        df = calculate_technical_indicators(data)
        df = df.dropna()
        
        if len(df) < lookback + 50:
            st.error("Not enough data. Need at least 50+ candles.")
            return None, None, 0, 0, None
        
        X = create_ml_features(df, lookback)
        y = create_labels(df, lookback)
        
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, 
                                      min_samples_leaf=2, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        
        train_accuracy = model.score(X_train_scaled, y_train) * 100
        test_accuracy = model.score(X_test_scaled, y_test) * 100
        
        return model, scaler, train_accuracy, test_accuracy, df

def predict_signal(model, scaler, current_data, lookback):
    df = calculate_technical_indicators(current_data)
    df = df.dropna()
    
    if len(df) < lookback:
        return 0, 0
    
    X = create_ml_features(df, lookback)
    if len(X) == 0:
        return 0, 0
    
    X_scaled = scaler.transform(X[-1:])
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    confidence = max(probabilities) * 100
    
    return prediction, confidence

if mode == "📁 Data Manager":
    st.header("📁 Data Manager")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📥 Upload CSV Data")
        st.markdown("""
        <div class="info-box">
        <b>CSV Format:</b><br>
        Date, Open, High, Low, Close, Volume<br><br>
        <b>Download from:</b> TradingView, NSE, Investing.com
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        
        if uploaded_file:
            csv_data = load_csv_data(uploaded_file)
            if csv_data is not None:
                st.session_state.uploaded_data = csv_data
                st.success(f"✅ Loaded: {len(csv_data)} rows")
                st.dataframe(csv_data.head(10), use_container_width=True)
    
    with col2:
        st.markdown("### 🌐 Auto-Fetch Data")
        st.markdown("""
        <div class="info-box">
        <b>Yahoo Finance:</b><br>
        Last 60-90 days intraday<br>
        Auto-merges with CSV!
        </div>
        """, unsafe_allow_html=True)
        
        auto_days = st.slider("Days", 7, 90, 60)
        
        if st.button("🔄 Fetch"):
            auto_data = fetch_intraday_data(symbol_name, auto_days, timeframe)
            if auto_data is not None:
                st.success(f"✅ Fetched: {len(auto_data)} rows")
                st.dataframe(auto_data.head(10), use_container_width=True)
                
                if st.session_state.uploaded_data is not None:
                    merged = merge_data(auto_data, st.session_state.uploaded_data)
                    st.session_state.historical_data = merged
                    st.success(f"✅ Merged: {len(merged)} rows")
                else:
                    st.session_state.historical_data = auto_data

elif mode == "📊 Train Model":
    st.header("📊 Train AI Model")
    
    data_source = st.radio("Data Source", ["🌐 Auto-Fetch", "📁 CSV Only", "🔗 Merged"])
    
    if st.button("🚀 Start Training", type="primary"):
        data_to_use = None
        
        if data_source == "🌐 Auto-Fetch":
            data_to_use = fetch_intraday_data(symbol_name, 60, timeframe)
        elif data_source == "📁 CSV Only":
            data_to_use = st.session_state.uploaded_data
        elif data_source == "🔗 Merged":
            if st.session_state.uploaded_data is None:
                st.error("Upload CSV first!")
                st.stop()
            auto_data = fetch_intraday_data(symbol_name, 60, timeframe)
            data_to_use = merge_data(auto_data, st.session_state.uploaded_data)
        
        if data_to_use is not None and len(data_to_use) > 100:
            result = train_ml_model(data_to_use, lookback_period)
            
            if result[0] is not None:
                model, scaler, train_acc, test_acc, processed_data = result
                
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.historical_data = processed_data
                st.session_state.model_trained = True
                
                st.success("✅ Model trained!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Train Accuracy", f"{train_acc:.2f}%")
                with col2:
                    st.metric("Test Accuracy", f"{test_acc:.2f}%")
                with col3:
                    st.metric("Data Points", len(processed_data))

elif mode == "🚀 Live Trading":
    st.header("🚀 Live Paper Trading")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Train model first!")
        st.stop()
    
    portfolio = st.session_state.portfolio
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>💰 Cash</h4>
            <h2>₹{portfolio['cash']:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 Holdings</h4>
            <h2>{portfolio['holdings']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📈 Trades</h4>
            <h2>{len(portfolio['trades'])}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        current_data = fetch_intraday_data(symbol_name, 7, timeframe)
        if current_data is not None and not current_data.empty:
            current_price = current_data['Close'].iloc[-1]
            portfolio_value = portfolio['cash'] + (portfolio['holdings'] * current_price)
            pnl = portfolio_value - initial_capital
            pnl_color = "green" if pnl >= 0 else "red"
        else:
            pnl = 0
            pnl_color = "gray"
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>💵 P&L</h4>
            <h2 style="color: {pnl_color}">₹{pnl:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("🔄 Refresh"):
        st.rerun()
    
    if current_data is not None and not current_data.empty:
        signal, confidence = predict_signal(st.session_state.model, st.session_state.scaler, 
                                          current_data, lookback_period)
        
        signal_text = {1: "🟢 STRONG BUY", 0: "🟡 HOLD", -1: "🔴 STRONG SELL"}
        signal_class = {1: "buy-signal", 0: "hold-signal", -1: "sell-signal"}
        
        st.markdown(f"""
        <div class="trade-signal {signal_class[signal]}">
            {signal_text[signal]}<br>
            <small>Price: ₹{current_price:.2f} | Confidence: {confidence:.1f}%</small>
        </div>
        """, unsafe_allow_html=True)

elif mode == "📈 Backtest":
    st.header("📈 Backtest Strategy")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Train model first!")
        st.stop()
    
    st.info("Backtest feature - Use training data for testing")

st.markdown("---")
st.warning("⚠️ Paper trading only. Educational purposes. Not financial advice.")