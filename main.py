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

@st.cache_data()
def add_technical_indicators(df):
    d = df.copy()
    # Use 'Close' if 'Adj Close' not available
    price_col = 'Adj Close' if 'Adj Close' in d.columns else 'Close'
    
    if price_col in d.columns:
        d['SMA20'] = ta.SMA(d[price_col], timeperiod=20)
        d['EMA20'] = ta.EMA(d[price_col], timeperiod=20)
        d['RSI14'] = ta.RSI(d[price_col], timeperiod=14)
        macd, macdsignal, macdhist = ta.MACD(d[price_col], fastperiod=12, slowperiod=26, signalperiod=9)
        d['MACD'] = macd
        d['MACD_Signal'] = macdsignal
        d['MACD_Hist'] = macdhist
        
    if 'High' in d.columns and 'Low' in d.columns and 'Close' in d.columns:
        k, d_stoch = ta.STOCHF(d['High'], d['Low'], d['Close'], fastk_period=14, fastd_period=3)
        d['STOCH_K'] = k
        d['STOCH_D'] = d_stoch
        d['SAR'] = ta.SAR(d['High'], d['Low'], acceleration=0.02, maximum=0.2)
    
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
    model_df.rename(columns={'index': 'Date'}, inplace=True)
    
    # Handle both 'index' and 'Date' column names from reset_index
    date_col = 'Date' if 'Date' in model_df.columns else model_df.columns[0]
    if date_col != 'Date':
        model_df.rename(columns={date_col: 'Date'}, inplace=True)
    
    model_df['Date'] = pd.to_datetime(model_df['Date'])
    model_df['Days'] = (model_df['Date'] - model_df['Date'].min()).dt.days

    # Add lag features
    model_df['Lag1'] = model_df['Adj Close'].shift(1)
    model_df['Lag5'] = model_df['Adj Close'].shift(5)
    model_df['Lag10'] = model_df['Adj Close'].shift(10)

    # Add simple rolling statistics
    model_df['RollMean7'] = model_df['Adj Close'].rolling(window=7).mean()
    model_df['RollStd7'] = model_df['Adj Close'].rolling(window=7).std()

    # Add TA features computed from the original df (aligned by date)
    ta_df = add_technical_indicators(df)
    ta_df = ta_df.reset_index()
    
    # Handle date column name from reset_index
    ta_date_col = 'Date' if 'Date' in ta_df.columns else ta_df.columns[0]
    if ta_date_col != 'Date':
        ta_df.rename(columns={ta_date_col: 'Date'}, inplace=True)
    
    ta_df = ta_df[['Date', 'SMA20', 'EMA20', 'RSI14', 'MACD', 'MACD_Signal']]
    model_df = model_df.merge(ta_df, on='Date', how='left')

    model_df.dropna(inplace=True)
    return model_df


def train_models(X_train, y_train, X_test, y_test):
    results = {}

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results['rf'] = {
        'model': rf,
        'pred': rf_pred,
        'mae': mean_absolute_error(y_test, rf_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'mape': calculate_percentage_accuracy(y_test, rf_pred)
    }

    xgb = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
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
                                  low=df['Low'], close=df['Close'], name='Price'), 
                  row=1, col=1)

    # Overlay moving averages
    if 'SMA' in indicators and 'SMA20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], mode='lines', name='SMA20', 
                                line=dict(color='orange')), row=1, col=1)
    if 'EMA' in indicators and 'EMA20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], mode='lines', name='EMA20',
                                line=dict(color='purple')), row=1, col=1)

    # Fibonacci lines (simple static between min and max in range)
    if 'Fibonacci' in indicators:
        max_price = df['High'].max()
        min_price = df['Low'].min()
        diff = max_price - min_price
        levels = [1.0, 0.618, 0.5, 0.382, 0.0]
        colors = ['red', 'orange', 'yellow', 'green', 'blue']
        for lvl, color in zip(levels, colors):
            price = max_price - lvl * diff
            fig.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[price, price], 
                                    mode='lines', showlegend=True, name=f'Fib {lvl}',
                                    line=dict(dash='dash', color=color)), row=1, col=1)

    # Oscillators row
    if 'RSI' in indicators and 'RSI14' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI14'], mode='lines', name='RSI14',
                                line=dict(color='blue')), row=2, col=1)
        # Add RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, opacity=0.5)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, opacity=0.5)
        
    if 'Stochastic' in indicators and 'STOCH_K' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['STOCH_K'], mode='lines', name='%K',
                                line=dict(color='purple')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['STOCH_D'], mode='lines', name='%D',
                                line=dict(color='orange')), row=2, col=1)
        
    if 'MACD' in indicators and 'MACD' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD',
                                line=dict(color='blue')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], mode='lines', name='MACD Signal',
                                line=dict(color='red')), row=2, col=1)

    # Volume row
    if 'Volume' in df.columns:
        colors = ['red' if close < open else 'green' 
                  for close, open in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', 
                            marker_color=colors, showlegend=False), row=3, col=1)

    fig.update_layout(height=900, title_text=f"{ticker} — Price & Indicators", 
                     xaxis_rangeslider_visible=False)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Indicator Value", row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=3, col=1)
    
    return fig

# -----------------------
# Streamlit UI
# -----------------------

st.set_page_config(layout='wide', page_title='Streamlit Stock Dashboard')
st.title('Streamlit Stock Dashboard — Refactored')

# Sidebar inputs
ticker = st.sidebar.text_input('Ticker', value='AAPL')
start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2023-01-01'))
end_date = st.sidebar.date_input('End Date', value=pd.to_datetime('today'))
interval = st.sidebar.selectbox('Interval', options=['1d', '1h', '30m', '5m'], index=0)
chart_type = st.sidebar.radio('Chart Type', ['Candlestick', 'Line'])
indicators = st.sidebar.multiselect('Indicators', ['SMA', 'EMA', 'RSI', 'Stochastic', 'MACD', 'Parabolic SAR', 'Fibonacci'], default=['SMA', 'EMA'])
forecast_days = st.sidebar.number_input('Forecast Days', min_value=1, max_value=90, value=14)

# Data fetch
if ticker:
    with st.spinner('Downloading data...'):
        df = download_data(ticker, start_date, end_date, interval)

    if df.empty:
        st.warning('No data for the selected ticker / date range / interval')
    else:
        # Add indicators
        df = add_technical_indicators(df)

        # Main chart
        st.header('Price Chart')
        fig = plot_price_and_indicators(df, ticker, indicators)
        st.plotly_chart(fig, use_container_width=True)

        # Price table and stats
        st.header('Price Data & Statistics')
        
        # Debug: show available columns
        st.write("Available columns:", df.columns.tolist())
        
        # Build price_df with available columns
        available_cols = []
        desired_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for col in desired_cols:
            if col in df.columns:
                available_cols.append(col)
        
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
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric('Annual Return (approx)', f"{annual_return:.2f}%")
        with col2:
            st.metric('Annual Volatility (approx)', f"{annual_vol:.2f}%")

        # Modeling
        st.header('Forecasting (RF & XGB)')
        model_df = prepare_model_data(df)
        if model_df is None or model_df.empty:
            st.error('Not enough data to prepare modeling dataset.')
        else:
            # Train/test split without shuffling (time-series aware)
            split_idx = int(len(model_df) * 0.8)
            train = model_df.iloc[:split_idx]
            test = model_df.iloc[split_idx:]

            features = ['Days', 'Lag1', 'Lag5', 'Lag10', 'RollMean7', 'RollStd7', 'SMA20', 'EMA20', 'RSI14', 'MACD', 'MACD_Signal']
            X_train = train[features]
            y_train = train['Adj Close']
            X_test = test[features]
            y_test = test['Adj Close']

            with st.spinner('Training models...'):
                results = train_models(X_train, y_train, X_test, y_test)

            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader('Random Forest')
                st.write(f"MAE: {results['rf']['mae']:.4f}")
                st.write(f"RMSE: {results['rf']['rmse']:.4f}")
                st.write(f"MAPE: {results['rf']['mape']:.2f}%")
            
            with col2:
                st.subheader('XGBoost')
                st.write(f"MAE: {results['xgb']['mae']:.4f}")
                st.write(f"RMSE: {results['xgb']['rmse']:.4f}")
                st.write(f"MAPE: {results['xgb']['mape']:.2f}%")

            # Plot predictions against actuals for test set
            pred_df = test[['Date', 'Adj Close']].copy()
            pred_df['RF_Pred'] = results['rf']['pred']
            pred_df['XGB_Pred'] = results['xgb']['pred']

            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Adj Close'], 
                                         mode='lines+markers', name='Actual',
                                         line=dict(color='blue')))
            fig_pred.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['RF_Pred'], 
                                         mode='lines', name='RF Pred',
                                         line=dict(color='green')))
            fig_pred.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['XGB_Pred'], 
                                         mode='lines', name='XGB Pred',
                                         line=dict(color='red')))
            fig_pred.update_layout(title='Model Predictions vs Actual (Test set)', 
                                  xaxis_title='Date', yaxis_title='Adj Close')
            st.plotly_chart(fig_pred, use_container_width=True)

            # Forecast next N days using Days feature
            last_date = model_df['Date'].max()
            future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, forecast_days + 1)]
            future_days = [(d - model_df['Date'].min()).days for d in future_dates]

            # Build a minimal feature frame for future (we'll use last known lags/rolling values)
            last_row = model_df.iloc[-1]
            future_rows = []
            for fd, day in zip(future_dates, future_days):
                future_rows.append({
                    'Date': fd,
                    'Days': day,
                    'Lag1': last_row['Adj Close'],
                    'Lag5': last_row['Adj Close'],
                    'Lag10': last_row['Adj Close'],
                    'RollMean7': last_row['RollMean7'],
                    'RollStd7': last_row['RollStd7'],
                    'SMA20': last_row.get('SMA20', np.nan),
                    'EMA20': last_row.get('EMA20', np.nan),
                    'RSI14': last_row.get('RSI14', np.nan),
                    'MACD': last_row.get('MACD', np.nan),
                    'MACD_Signal': last_row.get('MACD_Signal', np.nan)
                })
            future_X = pd.DataFrame(future_rows)[features]
            rf_future = results['rf']['model'].predict(future_X)
            xgb_future = results['xgb']['model'].predict(future_X)

            future_out = pd.DataFrame({'Date': future_dates, 'RF': rf_future, 'XGB': xgb_future})
            st.write(future_out)

            fig_future = go.Figure()
            fig_future.add_trace(go.Scatter(x=df.index, y=df['Adj Close'], 
                                           mode='lines', name='Historical',
                                           line=dict(color='blue')))
            fig_future.add_trace(go.Scatter(x=future_out['Date'], y=future_out['RF'], 
                                           mode='lines+markers', name='RF Forecast',
                                           line=dict(color='green')))
            fig_future.add_trace(go.Scatter(x=future_out['Date'], y=future_out['XGB'], 
                                           mode='lines+markers', name='XGB Forecast',
                                           line=dict(color='red')))
            fig_future.update_layout(title='Forecasts', xaxis_title='Date', yaxis_title='Adj Close')
            st.plotly_chart(fig_future, use_container_width=True)

        # Fundamentals and News tabs
        tab1, tab2 = st.tabs(['Fundamentals', 'Top News'])

        with tab1:
            st.subheader('Fundamental Data (AlphaVantage)')
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
            st.subheader('Latest News & Sentiment')
            try:
                sn = StockNews(ticker, save_news=False)
                news_df = sn.read_rss()
                st.write(news_df[['published', 'title', 'summary', 'sentiment_title', 'sentiment_summary']].head(10))
            except Exception as e:
                st.error(f'Failed to fetch news: {e}')

else:
    st.info('Enter a ticker in the sidebar to begin.')

# -----------------------
# End
# -----------------------
