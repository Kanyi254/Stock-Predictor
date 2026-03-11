import os
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import talib as ta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from alpha_vantage.fundamentaldata import FundamentalData
from stocknews import StockNews

# -----------------------
# Helpers & Caching
# -----------------------
@st.cache_data(ttl=60 * 5)
def download_data(ticker, start_date, end_date, interval):
    try:
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        st.error(f"Failed to download data: {e}")
        return pd.DataFrame()

def add_technical_indicators(df):
    d = df.copy()
    # Use 'Close' if 'Adj Close' not available
    price_col = 'Adj Close' if 'Adj Close' in d.columns else 'Close'

    if price_col in d.columns:
        # Moving averages
        d['SMA20'] = ta.SMA(d[price_col], timeperiod=20)
        d['SMA50'] = ta.SMA(d[price_col], timeperiod=50)
        d['EMA20'] = ta.EMA(d[price_col], timeperiod=20)

        # Momentum indicators
        d['RSI14'] = ta.RSI(d[price_col], timeperiod=14)
        d['MOM10'] = ta.MOM(d[price_col], timeperiod=10)
        d['ROC10'] = ta.ROC(d[price_col], timeperiod=10)

        # MACD
        macd, macdsignal, macdhist = ta.MACD(d[price_col], fastperiod=12, slowperiod=26, signalperiod=9)
        d['MACD'] = macd
        d['MACD_Signal'] = macdsignal
        d['MACD_Hist'] = macdhist

        # Bollinger Bands
        upper, middle, lower = ta.BBANDS(d[price_col], timeperiod=20)
        d['BB_Upper'] = upper
        d['BB_Middle'] = middle
        d['BB_Lower'] = lower
        d['BB_Width'] = (upper - lower) / middle

    if 'High' in d.columns and 'Low' in d.columns and 'Close' in d.columns:
        # Stochastic
        k, d_stoch = ta.STOCHF(d['High'], d['Low'], d['Close'], fastk_period=14, fastd_period=3)
        d['STOCH_K'] = k
        d['STOCH_D'] = d_stoch

        # Parabolic SAR
        d['SAR'] = ta.SAR(d['High'], d['Low'], acceleration=0.02, maximum=0.2)

        # ATR (Average True Range)
        d['ATR'] = ta.ATR(d['High'], d['Low'], d['Close'], timeperiod=14)

        # ADX (Average Directional Index)
        d['ADX'] = ta.ADX(d['High'], d['Low'], d['Close'], timeperiod=14)

    # Ensure 'Adj Close' exists for modeling (use Close if not available)
    if 'Adj Close' not in d.columns and 'Close' in d.columns:
        d['Adj Close'] = d['Close']

    return d


def calculate_percentage_accuracy(y_true, y_pred):
    # MAPE but safe when y_true contains zeros
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return (np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

# -----------------------
# Modeling helpers
# -----------------------

def prepare_model_data(df):
    # Expects df to include 'Adj Close' and a DatetimeIndex
    model_df = df[['Adj Close']].copy()

    # Ensure we have a datetime index
    if not isinstance(model_df.index, pd.DatetimeIndex):
        return None

    model_df = model_df.reset_index()

    # Clean up column names - handle any duplicates or unnamed columns
    if model_df.columns[0] in ['index', 'Date', '']:
        model_df.rename(columns={model_df.columns[0]: 'Date'}, inplace=True)
    elif 'Date' not in model_df.columns:
        model_df.rename(columns={model_df.columns[0]: 'Date'}, inplace=True)

    # Remove any duplicate Date columns if they exist
    if 'Date' in model_df.columns:
        date_cols = [col for col in model_df.columns if col == 'Date']
        if len(date_cols) > 1:
            # Keep only the first Date column
            cols_to_keep = ['Date'] + [col for col in model_df.columns if col != 'Date']
            model_df = model_df[cols_to_keep]

    model_df['Date'] = pd.to_datetime(model_df['Date'])
    model_df['Days'] = (model_df['Date'] - model_df['Date'].min()).dt.days

    # Add lag features (more lags)
    model_df['Lag1'] = model_df['Adj Close'].shift(1)
    model_df['Lag2'] = model_df['Adj Close'].shift(2)
    model_df['Lag3'] = model_df['Adj Close'].shift(3)
    model_df['Lag5'] = model_df['Adj Close'].shift(5)
    model_df['Lag10'] = model_df['Adj Close'].shift(10)
    model_df['Lag20'] = model_df['Adj Close'].shift(20)

    # Add rolling statistics (multiple windows)
    model_df['RollMean5'] = model_df['Adj Close'].rolling(window=5).mean()
    model_df['RollMean10'] = model_df['Adj Close'].rolling(window=10).mean()
    model_df['RollMean20'] = model_df['Adj Close'].rolling(window=20).mean()
    model_df['RollStd5'] = model_df['Adj Close'].rolling(window=5).std()
    model_df['RollStd10'] = model_df['Adj Close'].rolling(window=10).std()
    model_df['RollStd20'] = model_df['Adj Close'].rolling(window=20).std()

    # Price momentum
    model_df['Returns_1d'] = model_df['Adj Close'].pct_change(1)
    model_df['Returns_5d'] = model_df['Adj Close'].pct_change(5)
    model_df['Returns_10d'] = model_df['Adj Close'].pct_change(10)

    # Add TA features computed from the original df (aligned by date)
    ta_df = add_technical_indicators(df)
    ta_df = ta_df.reset_index()

    # Clean up TA dataframe column names
    if ta_df.columns[0] in ['index', 'Date', '']:
        ta_df.rename(columns={ta_df.columns[0]: 'Date_TA'}, inplace=True)
    elif 'Date' not in ta_df.columns:
        ta_df.rename(columns={ta_df.columns[0]: 'Date_TA'}, inplace=True)
    else:
        ta_df.rename(columns={'Date': 'Date_TA'}, inplace=True)

    # Convert to datetime
    ta_df['Date_TA'] = pd.to_datetime(ta_df['Date_TA'])

    # Select relevant TA indicators (exclude Date_TA from this list)
    ta_cols = ['SMA20', 'SMA50', 'EMA20', 'RSI14', 'MACD', 'MACD_Signal',
               'MACD_Hist', 'MOM10', 'ROC10', 'BB_Width', 'ATR', 'ADX']
    available_ta_cols = [col for col in ta_cols if col in ta_df.columns]

    # Merge using left_on and right_on to avoid column conflicts
    ta_df_subset = ta_df[['Date_TA'] + available_ta_cols].copy()
    model_df = model_df.merge(ta_df_subset, left_on='Date', right_on='Date_TA', how='left')

    # Drop the redundant Date_TA column
    if 'Date_TA' in model_df.columns:
        model_df.drop('Date_TA', axis=1, inplace=True)

    model_df.dropna(inplace=True)
    return model_df


def train_models(X_train, y_train, X_test, y_test):
    results = {}

    # Random Forest with tuned hyperparameters
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results['rf'] = {
        'model': rf,
        'pred': rf_pred,
        'mae': mean_absolute_error(y_test, rf_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'mape': calculate_percentage_accuracy(y_test, rf_pred)
    }

    # XGBoost with tuned hyperparameters
    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42,
        verbosity=0,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    results['xgb'] = {
        'model': xgb,
        'pred': xgb_pred,
        'mae': mean_absolute_error(y_test, xgb_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, xgb_pred)),
        'mape': calculate_percentage_accuracy(y_test, xgb_pred)
    }

    return results

# -----------------------
# Plotting helpers
# -----------------------

def plot_price_and_indicators(df, ticker, indicators):
    # Build a 3-row subplot: price, oscillator (RSI/Stochastic/MACD), volume (optional)
    rows = 3
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.25, 0.15], vertical_spacing=0.03,
                        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]])

    # Price (candlestick)
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                  low=df['Low'], close=df['Close'], name='PRICE',
                                  increasing=dict(line=dict(color='#00d084', width=1), fillcolor='#00d084'),
                                  decreasing=dict(line=dict(color='#ff3b5c', width=1), fillcolor='#ff3b5c')),
                  row=1, col=1)

    # Overlay moving averages
    if 'SMA' in indicators and 'SMA20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], mode='lines', name='SMA20',
                                line=dict(color='#ff9f00', width=1.2)), row=1, col=1)
    if 'SMA' in indicators and 'SMA50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name='SMA50',
                                line=dict(color='#cc7a00', width=1.8, dash='dot')), row=1, col=1)
    if 'EMA' in indicators and 'EMA20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], mode='lines', name='EMA20',
                                line=dict(color='#00c8e0', width=1.2)), row=1, col=1)

    # Fibonacci lines (simple static between min and max in range)
    if 'Fibonacci' in indicators:
        max_price = df['High'].max()
        min_price = df['Low'].min()
        diff = max_price - min_price
        levels = [1.0, 0.618, 0.5, 0.382, 0.0]
        fib_colors = ['#ff3b5c', '#ff9f00', '#e8e0d0', '#00d084', '#0088ff']
        for lvl, color in zip(levels, fib_colors):
            price = max_price - lvl * diff
            fig.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[price, price],
                                    mode='lines', showlegend=True, name=f'FIB {lvl}',
                                    line=dict(dash='dash', color=color, width=0.8)), row=1, col=1)

    # Bollinger Bands
    if 'Bollinger Bands' in indicators and 'BB_Upper' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], mode='lines', name='BB UPPER',
                                line=dict(color='rgba(0,136,255,0.5)', width=1, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], mode='lines', name='BB MID',
                                line=dict(color='rgba(0,136,255,0.3)', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], mode='lines', name='BB LOWER',
                                fill='tonexty', fillcolor='rgba(0,136,255,0.04)',
                                line=dict(color='rgba(0,136,255,0.5)', width=1, dash='dot')), row=1, col=1)

    # Parabolic SAR
    if 'Parabolic SAR' in indicators and 'SAR' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SAR'], mode='markers', name='SAR',
                                marker=dict(color='#00c8e0', size=3, symbol='circle')), row=1, col=1)

    # Oscillators row
    if 'RSI' in indicators and 'RSI14' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI14'], mode='lines', name='RSI14',
                                line=dict(color='#ff9f00', width=1.2)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ff3b5c", row=2, col=1, opacity=0.4)
        fig.add_hline(y=30, line_dash="dash", line_color="#00d084", row=2, col=1, opacity=0.4)

    if 'Stochastic' in indicators and 'STOCH_K' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['STOCH_K'], mode='lines', name='%K',
                                line=dict(color='#00c8e0', width=1.2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['STOCH_D'], mode='lines', name='%D',
                                line=dict(color='#cc7a00', width=1.2)), row=2, col=1)

    if 'MACD' in indicators and 'MACD' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD',
                                line=dict(color='#0088ff', width=1.2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], mode='lines', name='MACD SIG',
                                line=dict(color='#ff3b5c', width=1.2)), row=2, col=1)

    # Volume row
    if 'Volume' in df.columns:
        vol_colors = ['#ff3b5c' if close < open else '#00d084'
                  for close, open in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='VOLUME',
                            marker_color=vol_colors, opacity=0.6, showlegend=False), row=3, col=1)

    fig.update_layout(
        height=900,
        title=dict(
            text=f"<b>{ticker}</b>  ·  PRICE & INDICATORS",
            font=dict(family="IBM Plex Mono", size=12, color="#ff9f00"),
            x=0.01,
        ),
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0d0d0d',
        font=dict(family="IBM Plex Mono", size=10, color="#8a8070"),
        xaxis_rangeslider_visible=False,
        legend=dict(
            bgcolor='rgba(15,15,15,0.9)',
            bordercolor='#2a2a2a',
            borderwidth=1,
            font=dict(family="IBM Plex Mono", size=9, color="#8a8070"),
        ),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    # Style all axes
    axis_style = dict(
        gridcolor='#1a1a1a',
        gridwidth=1,
        zerolinecolor='#2a2a2a',
        tickfont=dict(family="IBM Plex Mono", size=9, color="#6a6060"),
        linecolor='#2a2a2a',
        showgrid=True,
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    fig.update_xaxes(title_text="DATE", row=3, col=1,
                     title_font=dict(family="IBM Plex Mono", size=8, color="#504840"))
    fig.update_yaxes(title_text="PRICE", row=1, col=1,
                     title_font=dict(family="IBM Plex Mono", size=8, color="#504840"))
    fig.update_yaxes(title_text="INDICATOR", row=2, col=1,
                     title_font=dict(family="IBM Plex Mono", size=8, color="#504840"))
    fig.update_yaxes(title_text="VOLUME", row=3, col=1,
                     title_font=dict(family="IBM Plex Mono", size=8, color="#504840"))

    return fig

# -----------------------
# Streamlit UI
# -----------------------

st.set_page_config(layout='wide', page_title='TERMINAL // MARKET ANALYTICS', page_icon='📊')

# ── Bloomberg-style CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-primary:   #0a0a0a;
    --bg-panel:     #0f0f0f;
    --bg-card:      #141414;
    --bg-elevated:  #1a1a1a;
    --border:       #2a2a2a;
    --border-bright:#3a3a3a;
    --amber:        #ff9f00;
    --amber-dim:    #cc7a00;
    --amber-faint:  #1a0f00;
    --green:        #00d084;
    --red:          #ff3b5c;
    --blue:         #0088ff;
    --cyan:         #00c8e0;
    --text-primary: #e8e0d0;
    --text-secondary:#8a8070;
    --text-dim:     #504840;
    --font-mono:    'IBM Plex Mono', monospace;
    --font-sans:    'IBM Plex Sans', sans-serif;
}

/* ── Global reset ── */
html, body, [class*="css"], .stApp {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: var(--bg-panel) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
    font-family: var(--font-mono) !important;
    color: var(--text-primary) !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stDateInput label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stCheckbox label {
    color: var(--amber) !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stMultiSelect > div > div {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: 2px !important;
    color: var(--text-primary) !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: var(--amber) !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 4px !important;
    margin-top: 16px !important;
}

/* ── Main headers ── */
h1, h2, h3 {
    font-family: var(--font-mono) !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-top: 2px solid var(--amber) !important;
    border-radius: 0 !important;
    padding: 12px 16px !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
    font-family: var(--font-mono) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--amber) !important;
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    font-family: var(--font-mono) !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.75rem !important;
    font-family: var(--font-mono) !important;
}

/* ── Dataframe ── */
.stDataFrame, [data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
}
.stDataFrame thead tr th {
    background-color: var(--bg-elevated) !important;
    color: var(--amber) !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid var(--amber-dim) !important;
}
.stDataFrame tbody tr td {
    color: var(--text-primary) !important;
    font-size: 0.72rem !important;
    border-color: var(--border) !important;
}
.stDataFrame tbody tr:hover td {
    background-color: var(--amber-faint) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background-color: var(--bg-panel) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    color: var(--text-secondary) !important;
    border: none !important;
    border-right: 1px solid var(--border) !important;
    border-radius: 0 !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-family: var(--font-mono) !important;
    padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
    background-color: var(--amber-faint) !important;
    color: var(--amber) !important;
    border-bottom: 2px solid var(--amber) !important;
}

/* ── Alerts / info boxes ── */
.stAlert {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-bright) !important;
    border-left: 3px solid var(--amber) !important;
    border-radius: 0 !important;
    color: var(--text-primary) !important;
    font-size: 0.75rem !important;
    font-family: var(--font-mono) !important;
}

/* ── Buttons ── */
.stButton > button {
    background-color: var(--amber) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
}
.stButton > button:hover {
    background-color: var(--amber-dim) !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: var(--amber) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-bright); }

/* ── Section dividers ── */
hr { border-color: var(--border) !important; }

/* ── Warning / error ── */
[data-testid="stNotificationContentWarning"] {
    background-color: #1a1000 !important;
    border-left: 3px solid #ff9f00 !important;
    border-radius: 0 !important;
}
[data-testid="stNotificationContentError"] {
    background-color: #1a0008 !important;
    border-left: 3px solid var(--red) !important;
    border-radius: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Terminal header ───────────────────────────────────────────────────────────
import datetime as _dt
_now = _dt.datetime.utcnow()
st.markdown(f"""
<div style="
    background: #0f0f0f;
    border-bottom: 1px solid #2a2a2a;
    padding: 10px 20px;
    margin: -1rem -1rem 1.5rem -1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
">
    <div style="display:flex; align-items:center; gap:24px;">
        <span style="font-family:'IBM Plex Mono',monospace; color:#ff9f00; font-size:1.05rem; font-weight:700; letter-spacing:0.15em;">
            ▸ MARKET TERMINAL
        </span>
        <span style="font-family:'IBM Plex Mono',monospace; color:#504840; font-size:0.6rem; letter-spacing:0.2em; text-transform:uppercase;">
            MULTI-ASSET ANALYTICS SYSTEM v2.0
        </span>
    </div>
    <div style="font-family:'IBM Plex Mono',monospace; color:#504840; font-size:0.6rem; letter-spacing:0.15em;">
        {_now.strftime("UTC %Y-%m-%d  %H:%M:%S")}  ●  LIVE
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar branding ─────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="
    font-family:'IBM Plex Mono',monospace;
    color:#ff9f00;
    font-size:0.7rem;
    font-weight:700;
    letter-spacing:0.25em;
    text-transform:uppercase;
    padding: 8px 0 12px 0;
    border-bottom: 1px solid #2a2a2a;
    margin-bottom: 8px;
">▸ CONTROL PANEL</div>
""", unsafe_allow_html=True)

# Add asset category badge
category_colors = {
    'Stocks': '🔵',
    'Commodities': '🟡',
    'Forex': '🟢',
    'Volatility Indices': '🔴',
    'Crypto': '🟣'
}

# Sidebar inputs
st.sidebar.header('Asset Selection')

# Asset category selection
asset_category = st.sidebar.selectbox(
    'Asset Category',
    ['Stocks', 'Commodities', 'Forex', 'Volatility Indices', 'Crypto']
)

# Predefined symbols for each category
asset_symbols = {
    'Stocks': {
        'Apple': 'AAPL',
        'Microsoft': 'MSFT',
        'Google': 'GOOGL',
        'Amazon': 'AMZN',
        'Tesla': 'TSLA',
        'Meta': 'META',
        'Netflix': 'NFLX',
        'NVIDIA': 'NVDA'
    },
    'Commodities': {
        'Gold': 'GC=F',
        'Silver': 'SI=F',
        'Crude Oil (WTI)': 'CL=F',
        'Brent Oil': 'BZ=F',
        'Natural Gas': 'NG=F',
        'Copper': 'HG=F',
        'Platinum': 'PL=F',
        'Corn': 'ZC=F',
        'Wheat': 'ZW=F',
        'Soybeans': 'ZS=F'
    },
    'Forex': {
        'EUR/USD': 'EURUSD=X',
        'GBP/USD': 'GBPUSD=X',
        'USD/JPY': 'USDJPY=X',
        'USD/CHF': 'USDCHF=X',
        'AUD/USD': 'AUDUSD=X',
        'USD/CAD': 'USDCAD=X',
        'NZD/USD': 'NZDUSD=X',
        'EUR/GBP': 'EURGBP=X',
        'EUR/JPY': 'EURJPY=X',
        'GBP/JPY': 'GBPJPY=X'
    },
    'Volatility Indices': {
        'VIX (CBOE)': '^VIX',
        'VIX3M': '^VIX3M',
        'VIX6M': '^VIX6M',
        'VVIX': '^VVIX',
        'VXN (Nasdaq)': '^VXN',
        'RVX (Russell)': '^RVX',
        'SKEW': '^SKEW'
    },
    'Crypto': {
        'Bitcoin': 'BTC-USD',
        'Ethereum': 'ETH-USD',
        'Binance Coin': 'BNB-USD',
        'Cardano': 'ADA-USD',
        'Solana': 'SOL-USD',
        'XRP': 'XRP-USD',
        'Polkadot': 'DOT-USD',
        'Dogecoin': 'DOGE-USD'
    }
}

# Symbol selection
selected_name = st.sidebar.selectbox(
    f'Select {asset_category}',
    list(asset_symbols[asset_category].keys())
)
ticker = asset_symbols[asset_category][selected_name]

# Option for custom ticker
use_custom = st.sidebar.checkbox('Use Custom Ticker')
if use_custom:
    ticker = st.sidebar.text_input('Custom Ticker', value=ticker)

st.sidebar.write(f"**Selected Ticker:** {ticker}")

# Date and interval inputs
st.sidebar.header('Time Period')
start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2023-01-01'))
end_date = st.sidebar.date_input('End Date', value=pd.to_datetime('today'))
interval = st.sidebar.selectbox('Interval', options=['1d', '1h', '30m', '5m'], index=0)

# Chart settings
st.sidebar.header('Chart Settings')
chart_type = st.sidebar.radio('Chart Type', ['Candlestick', 'Line'])
indicators = st.sidebar.multiselect('Indicators', ['SMA', 'EMA', 'RSI', 'Stochastic', 'MACD', 'Bollinger Bands', 'Parabolic SAR', 'Fibonacci'], default=['SMA', 'EMA', 'RSI'])

# Forecast settings
st.sidebar.header('Forecast Settings')
forecast_days = st.sidebar.number_input('Forecast Days', min_value=1, max_value=90, value=14)

# Data fetch
if ticker:
    # Display current selection
    cat_icon = category_colors.get(asset_category, '📊')
    st.markdown(f"""
    <div style="
        background:#0f0f0f;
        border:1px solid #2a2a2a;
        border-left:3px solid #ff9f00;
        padding:10px 18px;
        margin-bottom:16px;
        display:flex;
        align-items:center;
        justify-content:space-between;
        font-family:'IBM Plex Mono',monospace;
    ">
        <div>
            <span style="color:#ff9f00;font-weight:700;font-size:1.1rem;letter-spacing:0.1em;">{ticker}</span>
            <span style="color:#504840;font-size:0.65rem;margin-left:16px;letter-spacing:0.15em;text-transform:uppercase;">{selected_name}  ·  {asset_category}</span>
        </div>
        <div style="color:#504840;font-size:0.6rem;letter-spacing:0.12em;">MARKET DATA  ·  {cat_icon}</div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner('Downloading data...'):
        df = download_data(ticker, start_date, end_date, interval)

    if df.empty:
        st.warning('No data for the selected ticker / date range / interval')
    else:
        # Add indicators
        df = add_technical_indicators(df)

        # Main chart
        st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;color:#ff9f00;font-size:0.7rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;border-bottom:1px solid #2a2a2a;padding-bottom:6px;margin:20px 0 12px 0;">▸ {selected_name}  /  PRICE CHART</div>', unsafe_allow_html=True)
        fig = plot_price_and_indicators(df, f"{selected_name} ({ticker})", indicators)
        st.plotly_chart(fig, use_container_width=True)

        # Price table and stats
        st.markdown('<div style="font-family:IBM Plex Mono,monospace;color:#ff9f00;font-size:0.7rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;border-bottom:1px solid #2a2a2a;padding-bottom:6px;margin:20px 0 12px 0;">▸ PRICE DATA  /  STATISTICS</div>', unsafe_allow_html=True)

        # Show key metrics in cards
        col1, col2, col3, col4 = st.columns(4)

        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0

        with col1:
            st.metric(
                "Current Price",
                f"${current_price:.2f}" if asset_category != 'Forex' else f"{current_price:.5f}",
                f"{price_change_pct:+.2f}%"
            )
        with col2:
            st.metric("High (Period)", f"${df['High'].max():.2f}" if asset_category != 'Forex' else f"{df['High'].max():.5f}")
        with col3:
            st.metric("Low (Period)", f"${df['Low'].min():.2f}" if asset_category != 'Forex' else f"{df['Low'].min():.5f}")
        with col4:
            avg_volume = df['Volume'].mean() if 'Volume' in df.columns else 0
            st.metric("Avg Volume", f"{avg_volume:,.0f}")

        # Build price_df with available columns
        available_cols = []
        desired_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for col in desired_cols:
            if col in df.columns:
                available_cols.append(col)

        price_df = pd.DataFrame()
        if not available_cols:
            st.error("No price columns found in the data")
        else:
            price_df = df[available_cols].copy()

            # Use 'Close' if 'Adj Close' not available
            price_col = 'Adj Close' if 'Adj Close' in price_df.columns else 'Close'
            price_df['Pct Change'] = price_df[price_col].pct_change()
            price_df.dropna(inplace=True)
            st.dataframe(price_df.tail(200))

            annual_return = price_df['Pct Change'].mean() * 252 * 100
            annual_vol = price_df['Pct Change'].std() * np.sqrt(252) * 100

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric('Annual Return (approx)', f"{annual_return:.2f}%")
            with col2:
                st.metric('Annual Volatility (approx)', f"{annual_vol:.2f}%")
            with col3:
                sharpe = annual_return / annual_vol if annual_vol != 0 else 0
                st.metric('Sharpe Ratio (approx)', f"{sharpe:.2f}")

        # Add specific insights based on asset category
        st.markdown('<div style="font-family:IBM Plex Mono,monospace;color:#ff9f00;font-size:0.65rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;border-bottom:1px solid #2a2a2a;padding-bottom:5px;margin:16px 0 10px 0;">▸ MARKET INSIGHTS</div>', unsafe_allow_html=True)
        if asset_category == 'Volatility Indices':
            st.info(f"""
            **VIX Interpretation:**
            - Current Level: {current_price:.2f}
            - VIX < 12: Low volatility (complacent market)
            - VIX 12-20: Normal volatility
            - VIX 20-30: Elevated volatility (market stress)
            - VIX > 30: High volatility (fear/panic)
            """)
        elif asset_category == 'Forex':
            st.info(f"""
            **Forex Trading:**
            - Current Rate: {current_price:.5f}
            - Daily Change: {price_change_pct:+.2f}%
            - Forex markets are highly liquid and trade 24/5
            - Consider economic calendars and central bank policies
            """)
        elif asset_category == 'Commodities':
            st.info(f"""
            **Commodity Analysis:**
            - Current Price: ${current_price:.2f}
            - Daily Change: {price_change_pct:+.2f}%
            - Commodities are influenced by supply/demand, geopolitics, and USD strength
            """)
        elif asset_category == 'Crypto':
            st.info(f"""
            **Cryptocurrency:**
            - Current Price: ${current_price:.2f}
            - Daily Change: {price_change_pct:+.2f}%
            - High volatility asset class - trade 24/7
            - Consider market sentiment and regulatory news
            """)

        # Modeling
        st.markdown('<div style="font-family:IBM Plex Mono,monospace;color:#ff9f00;font-size:0.7rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;border-bottom:1px solid #2a2a2a;padding-bottom:6px;margin:20px 0 12px 0;">▸ ML FORECASTING  /  RF + XGB + ENSEMBLE</div>', unsafe_allow_html=True)

        # Add warning for certain asset classes
        if asset_category in ['Volatility Indices', 'Crypto']:
            st.warning(f"⚠️ {asset_category} can be highly volatile and unpredictable. Use forecasts with caution.")

        model_df = prepare_model_data(df)
        if model_df is None or model_df.empty:
            st.error('Not enough data to prepare modeling dataset.')
        else:
            # Train/test split without shuffling (time-series aware)
            split_idx = int(len(model_df) * 0.8)
            train = model_df.iloc[:split_idx]
            test = model_df.iloc[split_idx:]

            # Define features - dynamically select available ones
            all_features = ['Days', 'Lag1', 'Lag2', 'Lag3', 'Lag5', 'Lag10', 'Lag20',
                          'RollMean5', 'RollMean10', 'RollMean20',
                          'RollStd5', 'RollStd10', 'RollStd20',
                          'Returns_1d', 'Returns_5d', 'Returns_10d',
                          'SMA20', 'SMA50', 'EMA20', 'RSI14', 'MACD', 'MACD_Signal',
                          'MACD_Hist', 'MOM10', 'ROC10', 'BB_Width', 'ATR', 'ADX']
            features = [f for f in all_features if f in model_df.columns]

            st.write(f"Using {len(features)} features for modeling")

            X_train = train[features]
            y_train = train['Adj Close']
            X_test = test[features]
            y_test = test['Adj Close']

            with st.spinner('Training models...'):
                results = train_models(X_train, y_train, X_test, y_test)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div style="font-family:IBM Plex Mono,monospace;color:#00d084;font-size:0.65rem;font-weight:700;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:8px;">◆ RANDOM FOREST</div>', unsafe_allow_html=True)
                st.write(f"MAE: ${results['rf']['mae']:.2f}")
                st.write(f"RMSE: ${results['rf']['rmse']:.2f}")
                st.write(f"MAPE (error): {results['rf']['mape']:.2f}%")

                # Feature importance
                if len(features) > 0:
                    importance_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': results['rf']['model'].feature_importances_
                    }).sort_values('Importance', ascending=False).head(10)
                    st.write("Top 10 Important Features:")
                    st.bar_chart(importance_df.set_index('Feature'))

            with col2:
                st.markdown('<div style="font-family:IBM Plex Mono,monospace;color:#0088ff;font-size:0.65rem;font-weight:700;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:8px;">◆ XGBOOST</div>', unsafe_allow_html=True)
                st.write(f"MAE: ${results['xgb']['mae']:.2f}")
                st.write(f"RMSE: ${results['xgb']['rmse']:.2f}")
                st.write(f"MAPE (error): {results['xgb']['mape']:.2f}%")

                # Feature importance
                if len(features) > 0:
                    importance_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': results['xgb']['model'].feature_importances_
                    }).sort_values('Importance', ascending=False).head(10)
                    st.write("Top 10 Important Features:")
                    st.bar_chart(importance_df.set_index('Feature'))

            # Plot predictions against actuals for test set
            pred_df = test[['Date', 'Adj Close']].copy()
            pred_df['RF_Pred'] = results['rf']['pred']
            pred_df['XGB_Pred'] = results['xgb']['pred']

            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Adj Close'],
                                         mode='lines+markers', name='ACTUAL',
                                         line=dict(color='#ff9f00', width=2),
                                         marker=dict(size=4)))
            fig_pred.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['RF_Pred'],
                                         mode='lines', name='RF PRED',
                                         line=dict(color='#00d084', width=1.5, dash='dot')))
            fig_pred.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['XGB_Pred'],
                                         mode='lines', name='XGB PRED',
                                         line=dict(color='#0088ff', width=1.5, dash='dot')))
            fig_pred.update_layout(
                title=dict(text="<b>MODEL PREDICTIONS</b>  ·  TEST SET", font=dict(family="IBM Plex Mono", size=11, color="#ff9f00"), x=0.01),
                xaxis_title='DATE', yaxis_title='PRICE',
                paper_bgcolor='#0a0a0a', plot_bgcolor='#0d0d0d',
                font=dict(family="IBM Plex Mono", size=10, color="#8a8070"),
                legend=dict(bgcolor='rgba(15,15,15,0.9)', bordercolor='#2a2a2a', borderwidth=1, font=dict(family="IBM Plex Mono", size=9, color="#8a8070")),
                margin=dict(l=10, r=10, t=50, b=10),
                xaxis=dict(gridcolor='#1a1a1a', linecolor='#2a2a2a', tickfont=dict(family="IBM Plex Mono", size=9, color="#6a6060")),
                yaxis=dict(gridcolor='#1a1a1a', linecolor='#2a2a2a', tickfont=dict(family="IBM Plex Mono", size=9, color="#6a6060")),
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            # Forecast next N days using Days feature
            last_date = model_df['Date'].max()
            future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, forecast_days + 1)]
            future_days = [(d - model_df['Date'].min()).days for d in future_dates]

            # Build future features using last known values
            last_row = model_df.iloc[-1]
            last_close = last_row['Adj Close']

            future_rows = []
            # Use a rolling window of recent prices for iterative forecast
            recent_prices = list(model_df['Adj Close'].values[-20:])

            for fd, day in zip(future_dates, future_days):
                row = {'Date': fd, 'Days': day}

                # Use the most recently predicted price for lags
                lag_prices = recent_prices[-20:]  # keep a buffer

                row['Lag1'] = lag_prices[-1]
                row['Lag2'] = lag_prices[-2] if len(lag_prices) >= 2 else lag_prices[-1]
                row['Lag3'] = lag_prices[-3] if len(lag_prices) >= 3 else lag_prices[-1]
                row['Lag5'] = lag_prices[-5] if len(lag_prices) >= 5 else lag_prices[-1]
                row['Lag10'] = lag_prices[-10] if len(lag_prices) >= 10 else lag_prices[-1]
                row['Lag20'] = lag_prices[-20] if len(lag_prices) >= 20 else lag_prices[-1]

                # Rolling stats from the dynamic price window
                row['RollMean5'] = np.mean(lag_prices[-5:])
                row['RollMean10'] = np.mean(lag_prices[-10:])
                row['RollMean20'] = np.mean(lag_prices[-20:])
                row['RollStd5'] = np.std(lag_prices[-5:]) if len(lag_prices) >= 5 else 0
                row['RollStd10'] = np.std(lag_prices[-10:]) if len(lag_prices) >= 10 else 0
                row['RollStd20'] = np.std(lag_prices[-20:]) if len(lag_prices) >= 20 else 0

                # Return features
                row['Returns_1d'] = (lag_prices[-1] - lag_prices[-2]) / lag_prices[-2] if len(lag_prices) >= 2 and lag_prices[-2] != 0 else 0
                row['Returns_5d'] = (lag_prices[-1] - lag_prices[-5]) / lag_prices[-5] if len(lag_prices) >= 5 and lag_prices[-5] != 0 else 0
                row['Returns_10d'] = (lag_prices[-1] - lag_prices[-10]) / lag_prices[-10] if len(lag_prices) >= 10 and lag_prices[-10] != 0 else 0

                # TA indicators - use last known values (can't recompute without OHLC history)
                for ta_col in ['SMA20', 'SMA50', 'EMA20', 'RSI14', 'MACD',
                              'MACD_Signal', 'MACD_Hist', 'MOM10', 'ROC10',
                              'BB_Width', 'ATR', 'ADX']:
                    if ta_col in features:
                        row[ta_col] = last_row.get(ta_col, np.nan)

                future_rows.append(row)

                # Predict next price and append to rolling window for next iteration
                row_df = pd.DataFrame([row])[features]
                rf_next = results['rf']['model'].predict(row_df)[0]
                xgb_next = results['xgb']['model'].predict(row_df)[0]
                ensemble_next = (rf_next + xgb_next) / 2
                recent_prices.append(ensemble_next)

            future_X = pd.DataFrame(future_rows)[features]
            rf_future = results['rf']['model'].predict(future_X)
            xgb_future = results['xgb']['model'].predict(future_X)
            ensemble_future = (rf_future + xgb_future) / 2

            future_out = pd.DataFrame({'Date': future_dates, 'RF': rf_future, 'XGB': xgb_future, 'Ensemble': ensemble_future})

            # Calculate forecast statistics
            current_price = df['Adj Close'].iloc[-1]
            avg_forecast = ensemble_future[-1]
            forecast_change = ((avg_forecast - current_price) / current_price) * 100

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
            with col2:
                st.metric(f"Forecast ({forecast_days}d)", f"${avg_forecast:.2f}", f"{forecast_change:+.2f}%")
            with col3:
                forecast_direction = "📈 Bullish" if forecast_change > 0 else "📉 Bearish"
                st.metric("Direction", forecast_direction)

            st.dataframe(future_out, use_container_width=True)

            fig_future = go.Figure()
            fig_future.add_trace(go.Scatter(x=df.index, y=df['Adj Close'],
                                           mode='lines', name='HISTORICAL',
                                           line=dict(color='#504840', width=1.5)))
            fig_future.add_trace(go.Scatter(x=future_out['Date'], y=future_out['RF'],
                                           mode='lines+markers', name='RF FORECAST',
                                           line=dict(color='#00d084', dash='dot', width=1.5),
                                           marker=dict(size=4)))
            fig_future.add_trace(go.Scatter(x=future_out['Date'], y=future_out['XGB'],
                                           mode='lines+markers', name='XGB FORECAST',
                                           line=dict(color='#0088ff', dash='dot', width=1.5),
                                           marker=dict(size=4)))
            fig_future.add_trace(go.Scatter(x=future_out['Date'], y=future_out['Ensemble'],
                                           mode='lines+markers', name='ENSEMBLE',
                                           line=dict(color='#ff9f00', width=2.5),
                                           marker=dict(size=5)))
            fig_future.update_layout(
                title=dict(text="<b>PRICE FORECAST</b>  ·  RF + XGB + ENSEMBLE", font=dict(family="IBM Plex Mono", size=11, color="#ff9f00"), x=0.01),
                xaxis_title='DATE', yaxis_title='PRICE',
                paper_bgcolor='#0a0a0a', plot_bgcolor='#0d0d0d',
                font=dict(family="IBM Plex Mono", size=10, color="#8a8070"),
                legend=dict(bgcolor='rgba(15,15,15,0.9)', bordercolor='#2a2a2a', borderwidth=1, font=dict(family="IBM Plex Mono", size=9, color="#8a8070")),
                margin=dict(l=10, r=10, t=50, b=10),
                xaxis=dict(gridcolor='#1a1a1a', linecolor='#2a2a2a', tickfont=dict(family="IBM Plex Mono", size=9, color="#6a6060")),
                yaxis=dict(gridcolor='#1a1a1a', linecolor='#2a2a2a', tickfont=dict(family="IBM Plex Mono", size=9, color="#6a6060")),
            )
            st.plotly_chart(fig_future, use_container_width=True)

        # Fundamentals and News tabs
        tab1, tab2, tab3 = st.tabs(['Fundamentals', 'Top News', 'Asset Info'])

        with tab1:
            st.markdown('<div style="font-family:IBM Plex Mono,monospace;color:#ff9f00;font-size:0.65rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;margin-bottom:8px;">▸ FUNDAMENTAL DATA  /  ALPHAVANTAGE</div>', unsafe_allow_html=True)
            if asset_category not in ['Stocks']:
                st.info(f'Fundamental data is primarily available for stocks. {asset_category} may have limited fundamental data.')

            av_key = st.secrets.get('ALPHAVANTAGE_API_KEY') if 'ALPHAVANTAGE_API_KEY' in st.secrets else os.getenv('ALPHAVANTAGE_API_KEY')
            if not av_key:
                st.info('Set ALPHAVANTAGE_API_KEY in Streamlit secrets or environment variables to view fundamentals.')
            else:
                try:
                    fd = FundamentalData(av_key, output_format='pandas')
                    bs = fd.get_balance_sheet_annual(ticker)[0]
                    st.write(bs.T)
                except Exception as e:
                    st.error(f'Failed to fetch fundamentals: {e}')

        with tab2:
            st.markdown('<div style="font-family:IBM Plex Mono,monospace;color:#ff9f00;font-size:0.65rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;margin-bottom:8px;">▸ NEWS FEED  /  SENTIMENT ANALYSIS</div>', unsafe_allow_html=True)
            try:
                sn = StockNews(ticker, save_news=False)
                news_df = sn.read_rss()
                st.write(news_df[['published', 'title', 'summary', 'sentiment_title', 'sentiment_summary']].head(10))
            except Exception as e:
                st.error(f'Failed to fetch news: {e}')

        with tab3:
            st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;color:#ff9f00;font-size:0.65rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;margin-bottom:8px;">▸ ASSET PROFILE  /  {selected_name}</div>', unsafe_allow_html=True)

            asset_info = {
                'Stocks': 'Equities representing ownership in publicly traded companies.',
                'Commodities': 'Physical goods including metals, energy, and agricultural products.',
                'Forex': 'Foreign exchange pairs showing relative value between currencies.',
                'Volatility Indices': 'Measures of market volatility and investor sentiment.',
                'Crypto': 'Digital/virtual currencies using cryptography for security.'
            }

            st.write(f"**Category:** {asset_category}")
            st.write(f"**Ticker:** {ticker}")
            st.write(f"**Description:** {asset_info.get(asset_category, 'Financial instrument')}")

            # Add trading hours info
            st.markdown('<div style="font-family:IBM Plex Mono,monospace;color:#ff9f00;font-size:0.65rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;margin-bottom:8px;">▸ TRADING INFORMATION</div>', unsafe_allow_html=True)
            trading_hours = {
                'Stocks': '9:30 AM - 4:00 PM ET (Monday-Friday)',
                'Commodities': 'Varies by commodity and exchange (often 24-hour markets)',
                'Forex': '24 hours (Sunday 5 PM - Friday 5 PM ET)',
                'Volatility Indices': 'Based on options market hours',
                'Crypto': '24/7/365'
            }
            st.info(f"**Trading Hours:** {trading_hours.get(asset_category, 'Varies by instrument')}")

else:
    st.markdown("""
    <div style="
        text-align:center;
        padding: 80px 40px;
        font-family:'IBM Plex Mono',monospace;
    ">
        <div style="color:#2a2a2a;font-size:4rem;margin-bottom:16px;">▸</div>
        <div style="color:#504840;font-size:0.7rem;letter-spacing:0.3em;text-transform:uppercase;">
            SELECT A TICKER IN THE CONTROL PANEL TO BEGIN
        </div>
        <div style="color:#2a2a2a;font-size:0.55rem;letter-spacing:0.2em;margin-top:12px;">
            MULTI-ASSET ANALYTICS  ·  RF + XGB FORECASTING  ·  TECHNICAL INDICATORS
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------
# End
# -----------------------
