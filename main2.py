import os
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import talib as ta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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

    # ── VWAP (resets daily) ──────────────────────────────────────────
    if 'Volume' in d.columns and 'High' in d.columns:
        typical_price = (d['High'] + d['Low'] + d['Close']) / 3
        d['VWAP'] = (typical_price * d['Volume']).cumsum() / d['Volume'].cumsum()

    return d


def compute_signals(df):
    """
    Generate BUY / SELL signals from multiple confluence indicators.
    Returns df with signal columns and a summary score per bar.
    """
    d = df.copy()
    price_col = 'Adj Close' if 'Adj Close' in d.columns else 'Close'

    # ── RSI signals ──────────────────────────────────────────────────
    d['sig_rsi_buy']  = (d['RSI14'] < 30) & (d['RSI14'].shift(1) >= 30) if 'RSI14' in d.columns else False
    d['sig_rsi_sell'] = (d['RSI14'] > 70) & (d['RSI14'].shift(1) <= 70) if 'RSI14' in d.columns else False

    # ── MACD crossover signals ────────────────────────────────────────
    if 'MACD' in d.columns and 'MACD_Signal' in d.columns:
        d['sig_macd_buy']  = (d['MACD'] > d['MACD_Signal']) & (d['MACD'].shift(1) <= d['MACD_Signal'].shift(1))
        d['sig_macd_sell'] = (d['MACD'] < d['MACD_Signal']) & (d['MACD'].shift(1) >= d['MACD_Signal'].shift(1))
    else:
        d['sig_macd_buy'] = d['sig_macd_sell'] = False

    # ── EMA crossover (price crosses EMA20) ──────────────────────────
    if 'EMA20' in d.columns:
        d['sig_ema_buy']  = (d[price_col] > d['EMA20']) & (d[price_col].shift(1) <= d['EMA20'].shift(1))
        d['sig_ema_sell'] = (d[price_col] < d['EMA20']) & (d[price_col].shift(1) >= d['EMA20'].shift(1))
    else:
        d['sig_ema_buy'] = d['sig_ema_sell'] = False

    # ── Stochastic oversold/overbought ───────────────────────────────
    if 'STOCH_K' in d.columns:
        d['sig_stoch_buy']  = (d['STOCH_K'] < 20) & (d['STOCH_K'].shift(1) >= 20)
        d['sig_stoch_sell'] = (d['STOCH_K'] > 80) & (d['STOCH_K'].shift(1) <= 80)
    else:
        d['sig_stoch_buy'] = d['sig_stoch_sell'] = False

    # ── Bollinger Band squeeze breakout ──────────────────────────────
    if 'BB_Upper' in d.columns:
        d['sig_bb_buy']  = d['Close'] < d['BB_Lower']
        d['sig_bb_sell'] = d['Close'] > d['BB_Upper']
    else:
        d['sig_bb_buy'] = d['sig_bb_sell'] = False

    # ── SAR flip ─────────────────────────────────────────────────────
    if 'SAR' in d.columns:
        d['sig_sar_buy']  = (d['Close'] > d['SAR']) & (d['Close'].shift(1) <= d['SAR'].shift(1))
        d['sig_sar_sell'] = (d['Close'] < d['SAR']) & (d['Close'].shift(1) >= d['SAR'].shift(1))
    else:
        d['sig_sar_buy'] = d['sig_sar_sell'] = False

    # ── Confluence score (-6 to +6) ───────────────────────────────────
    buy_cols  = ['sig_rsi_buy',  'sig_macd_buy',  'sig_ema_buy',  'sig_stoch_buy',  'sig_bb_buy',  'sig_sar_buy']
    sell_cols = ['sig_rsi_sell', 'sig_macd_sell', 'sig_ema_sell', 'sig_stoch_sell', 'sig_bb_sell', 'sig_sar_sell']
    d['confluence_buy']  = d[buy_cols].sum(axis=1)
    d['confluence_sell'] = d[sell_cols].sum(axis=1)
    d['confluence_score'] = d['confluence_buy'] - d['confluence_sell']  # positive = bullish

    # ── Strong signals: confluence >= 2 ──────────────────────────────
    d['strong_buy']  = d['confluence_buy']  >= 2
    d['strong_sell'] = d['confluence_sell'] >= 2

    return d


def detect_support_resistance(df, window=20, n_levels=5):
    """
    Detect support and resistance from rolling pivot highs/lows.
    Returns list of price levels.
    """
    highs = df['High'].rolling(window=window, center=True).max()
    lows  = df['Low'].rolling(window=window, center=True).min()

    resistance = df['High'][df['High'] == highs].dropna().values
    support    = df['Low'][df['Low']  == lows].dropna().values

    # Cluster levels (merge levels within 0.5% of each other)
    def cluster(levels, tol=0.005):
        if len(levels) == 0:
            return []
        levels = sorted(set(levels))
        clustered = [levels[0]]
        for lvl in levels[1:]:
            if (lvl - clustered[-1]) / clustered[-1] > tol:
                clustered.append(lvl)
        return clustered

    r_levels = cluster(resistance)
    s_levels = cluster(support)

    # Return closest levels to current price
    cur = df['Close'].iloc[-1]
    r_levels = sorted(r_levels, key=lambda x: abs(x - cur))[:n_levels]
    s_levels = sorted(s_levels, key=lambda x: abs(x - cur))[:n_levels]
    return s_levels, r_levels


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


def train_models(X_train, y_train, X_test, y_test, rf_params, xgb_params, n_cv_splits=5):
    """
    Train RF and XGB with:
    - User-supplied hyperparameters
    - TimeSeriesSplit cross-validation (no data leakage)
    - Walk-forward validation MAE
    - RF confidence intervals via per-tree predictions
    """
    results = {}
    tscv = TimeSeriesSplit(n_splits=n_cv_splits)

    # ── Random Forest ────────────────────────────────────────────────
    rf = RandomForestRegressor(**rf_params, random_state=42, n_jobs=-1)

    # Walk-forward CV MAE
    rf_cv_maes = []
    for tr_idx, val_idx in tscv.split(X_train):
        Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        ytr, yval = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        rf_cv = RandomForestRegressor(**rf_params, random_state=42, n_jobs=-1)
        rf_cv.fit(Xtr, ytr)
        rf_cv_maes.append(mean_absolute_error(yval, rf_cv.predict(Xval)))

    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    # Per-tree predictions for confidence interval
    rf_tree_preds = np.array([tree.predict(X_test) for tree in rf.estimators_])
    rf_pred_std = rf_tree_preds.std(axis=0)

    results['rf'] = {
        'model': rf,
        'pred': rf_pred,
        'pred_std': rf_pred_std,
        'mae': mean_absolute_error(y_test, rf_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'mape': calculate_percentage_accuracy(y_test, rf_pred),
        'cv_mae_mean': np.mean(rf_cv_maes),
        'cv_mae_std': np.std(rf_cv_maes),
    }

    # ── XGBoost ──────────────────────────────────────────────────────
    xgb = XGBRegressor(**xgb_params, random_state=42, verbosity=0, n_jobs=-1)

    xgb_cv_maes = []
    for tr_idx, val_idx in tscv.split(X_train):
        Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        ytr, yval = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        xgb_cv = XGBRegressor(**xgb_params, random_state=42, verbosity=0, n_jobs=-1)
        xgb_cv.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        xgb_cv_maes.append(mean_absolute_error(yval, xgb_cv.predict(Xval)))

    xgb.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False)
    xgb_pred = xgb.predict(X_test)

    results['xgb'] = {
        'model': xgb,
        'pred': xgb_pred,
        'mae': mean_absolute_error(y_test, xgb_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, xgb_pred)),
        'mape': calculate_percentage_accuracy(y_test, xgb_pred),
        'cv_mae_mean': np.mean(xgb_cv_maes),
        'cv_mae_std': np.std(xgb_cv_maes),
    }

    return results

# -----------------------
# Plotting helpers
# -----------------------

def plot_price_and_indicators(df, ticker, indicators, show_signals=True, show_sr=True):
    """
    4-row trading chart:
      Row 1: Candlestick + overlays + buy/sell signals + S/R + VWAP
      Row 2: Oscillator (RSI or Stochastic)
      Row 3: MACD + Histogram
      Row 4: Volume
    """
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'

    # Compute signals & S/R if needed
    sig_df = compute_signals(df) if show_signals else df
    s_levels, r_levels = detect_support_resistance(df) if show_sr else ([], [])

    rows = 4
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        row_heights=[0.52, 0.18, 0.17, 0.13],
        vertical_spacing=0.025,
        specs=[[{"secondary_y": False}]] * rows
    )

    # ── ROW 1: Candlestick ───────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='PRICE',
        increasing=dict(line=dict(color='#00d084', width=1), fillcolor='#00d084'),
        decreasing=dict(line=dict(color='#ff3b5c', width=1), fillcolor='#ff3b5c')
    ), row=1, col=1)

    # ── Moving averages ──────────────────────────────────────────────
    if 'SMA' in indicators and 'SMA20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], mode='lines', name='SMA20',
                                 line=dict(color='#ff9f00', width=1.2)), row=1, col=1)
    if 'SMA' in indicators and 'SMA50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name='SMA50',
                                 line=dict(color='#cc7a00', width=1.8, dash='dot')), row=1, col=1)
    if 'EMA' in indicators and 'EMA20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], mode='lines', name='EMA20',
                                 line=dict(color='#00c8e0', width=1.2)), row=1, col=1)

    # ── VWAP ─────────────────────────────────────────────────────────
    if 'VWAP' in df.columns and 'VWAP' in indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines', name='VWAP',
                                 line=dict(color='#c084fc', width=1.5, dash='dot')), row=1, col=1)

    # ── Bollinger Bands ──────────────────────────────────────────────
    if 'Bollinger Bands' in indicators and 'BB_Upper' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], mode='lines', name='BB UPPER',
                                 line=dict(color='rgba(0,136,255,0.4)', width=1, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], mode='lines', name='BB MID',
                                 line=dict(color='rgba(0,136,255,0.25)', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], mode='lines', name='BB LOWER',
                                 fill='tonexty', fillcolor='rgba(0,136,255,0.04)',
                                 line=dict(color='rgba(0,136,255,0.4)', width=1, dash='dot')), row=1, col=1)

    # ── Parabolic SAR ────────────────────────────────────────────────
    if 'Parabolic SAR' in indicators and 'SAR' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SAR'], mode='markers', name='SAR',
                                 marker=dict(color='#00c8e0', size=3, symbol='circle')), row=1, col=1)

    # ── Fibonacci ────────────────────────────────────────────────────
    if 'Fibonacci' in indicators:
        max_price = df['High'].max()
        min_price = df['Low'].min()
        diff = max_price - min_price
        levels = [1.0, 0.618, 0.5, 0.382, 0.0]
        fib_colors = ['#ff3b5c', '#ff9f00', '#e8e0d0', '#00d084', '#0088ff']
        for lvl, color in zip(levels, fib_colors):
            price = max_price - lvl * diff
            fig.add_trace(go.Scatter(
                x=[df.index[0], df.index[-1]], y=[price, price],
                mode='lines', showlegend=True, name=f'FIB {lvl}',
                line=dict(dash='dash', color=color, width=0.8)
            ), row=1, col=1)

    # ── Support & Resistance ──────────────────────────────────────────
    if show_sr:
        for lvl in s_levels:
            fig.add_hline(y=lvl, line_dash="dot", line_color="rgba(0,208,132,0.35)",
                          line_width=1, row=1, col=1)
            fig.add_annotation(x=df.index[-1], y=lvl, text=f" S {lvl:.2f}",
                               showarrow=False, font=dict(size=8, color="#00d084", family="IBM Plex Mono"),
                               xanchor="left", row=1, col=1)
        for lvl in r_levels:
            fig.add_hline(y=lvl, line_dash="dot", line_color="rgba(255,59,92,0.35)",
                          line_width=1, row=1, col=1)
            fig.add_annotation(x=df.index[-1], y=lvl, text=f" R {lvl:.2f}",
                               showarrow=False, font=dict(size=8, color="#ff3b5c", family="IBM Plex Mono"),
                               xanchor="left", row=1, col=1)

    # ── Buy / Sell signal arrows ──────────────────────────────────────
    if show_signals:
        buy_mask  = sig_df['strong_buy']
        sell_mask = sig_df['strong_sell']

        buy_dates  = sig_df.index[buy_mask]
        buy_prices = df['Low'][buy_mask] * 0.993   # slightly below bar
        sell_dates  = sig_df.index[sell_mask]
        sell_prices = df['High'][sell_mask] * 1.007  # slightly above bar

        if len(buy_dates) > 0:
            fig.add_trace(go.Scatter(
                x=buy_dates, y=buy_prices, mode='markers', name='BUY SIGNAL',
                marker=dict(symbol='triangle-up', size=12, color='#00d084',
                            line=dict(color='#003a22', width=1)),
            ), row=1, col=1)

        if len(sell_dates) > 0:
            fig.add_trace(go.Scatter(
                x=sell_dates, y=sell_prices, mode='markers', name='SELL SIGNAL',
                marker=dict(symbol='triangle-down', size=12, color='#ff3b5c',
                            line=dict(color='#3a0010', width=1)),
            ), row=1, col=1)

    # ── ROW 2: RSI or Stochastic ──────────────────────────────────────
    if 'RSI' in indicators and 'RSI14' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI14'], mode='lines', name='RSI14',
                                 line=dict(color='#ff9f00', width=1.2)), row=2, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor='rgba(255,59,92,0.05)', line_width=0, row=2, col=1)
        fig.add_hrect(y0=0,  y1=30,  fillcolor='rgba(0,208,132,0.05)', line_width=0, row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ff3b5c", row=2, col=1, opacity=0.4)
        fig.add_hline(y=50, line_dash="dash", line_color="#2a2a2a", row=2, col=1, opacity=0.8)
        fig.add_hline(y=30, line_dash="dash", line_color="#00d084", row=2, col=1, opacity=0.4)

    if 'Stochastic' in indicators and 'STOCH_K' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['STOCH_K'], mode='lines', name='%K',
                                 line=dict(color='#00c8e0', width=1.2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['STOCH_D'], mode='lines', name='%D',
                                 line=dict(color='#cc7a00', width=1.2)), row=2, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="#ff3b5c", row=2, col=1, opacity=0.4)
        fig.add_hline(y=20, line_dash="dash", line_color="#00d084", row=2, col=1, opacity=0.4)

    # ── ROW 3: MACD + Histogram ───────────────────────────────────────
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD',
                                 line=dict(color='#0088ff', width=1.2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], mode='lines', name='SIGNAL',
                                 line=dict(color='#ff3b5c', width=1.2)), row=3, col=1)
        # MACD Histogram with colour by sign
        hist_colors = ['#00d084' if v >= 0 else '#ff3b5c' for v in df['MACD_Hist'].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='HIST',
                             marker_color=hist_colors, opacity=0.6, showlegend=False), row=3, col=1)
        fig.add_hline(y=0, line_color='#2a2a2a', line_width=1, row=3, col=1)

    # ── ROW 4: Volume ────────────────────────────────────────────────
    if 'Volume' in df.columns:
        vol_colors = ['#ff3b5c' if c < o else '#00d084'
                      for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='VOLUME',
                             marker_color=vol_colors, opacity=0.55, showlegend=False), row=4, col=1)
        # Volume MA
        vol_ma = df['Volume'].rolling(20).mean()
        fig.add_trace(go.Scatter(x=df.index, y=vol_ma, mode='lines', name='VOL MA20',
                                 line=dict(color='#ff9f00', width=1, dash='dot'), showlegend=False), row=4, col=1)

    # ── Layout ───────────────────────────────────────────────────────
    fig.update_layout(
        height=1000,
        title=dict(text=f"<b>{ticker}</b>  ·  TRADING CHART",
                   font=dict(family="IBM Plex Mono", size=12, color="#ff9f00"), x=0.01),
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0d0d0d',
        font=dict(family="IBM Plex Mono", size=10, color="#8a8070"),
        xaxis_rangeslider_visible=False,
        legend=dict(bgcolor='rgba(10,10,10,0.9)', bordercolor='#2a2a2a', borderwidth=1,
                    font=dict(family="IBM Plex Mono", size=9, color="#8a8070"),
                    orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0),
        margin=dict(l=10, r=80, t=60, b=10),
        hovermode='x unified',
        hoverlabel=dict(bgcolor='#141414', bordercolor='#2a2a2a',
                        font=dict(family="IBM Plex Mono", size=10, color="#e8e0d0")),
    )
    axis_style = dict(gridcolor='#161616', gridwidth=1, zerolinecolor='#2a2a2a',
                      tickfont=dict(family="IBM Plex Mono", size=9, color="#6a6060"),
                      linecolor='#2a2a2a', showgrid=True)
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    fig.update_yaxes(title_text="PRICE",     title_font=dict(size=8, color="#504840"), row=1, col=1)
    fig.update_yaxes(title_text="OSC",       title_font=dict(size=8, color="#504840"), row=2, col=1)
    fig.update_yaxes(title_text="MACD",      title_font=dict(size=8, color="#504840"), row=3, col=1)
    fig.update_yaxes(title_text="VOL",       title_font=dict(size=8, color="#504840"), row=4, col=1)
    fig.update_xaxes(title_text="DATE", title_font=dict(size=8, color="#504840"), row=4, col=1)

    return fig, sig_df

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
    'Kenyan Stocks (NSE)': '🇰🇪',
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
    ['Stocks', 'Kenyan Stocks (NSE)', 'Commodities', 'Forex', 'Volatility Indices', 'Crypto']
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
    'Kenyan Stocks (NSE)': {
        'Safaricom': 'SCOM.NR',
        'Equity Group': 'EQTY.NR',
        'KCB Group': 'KCB.NR',
        'East African Breweries': 'EABL.NR',
        'Co-operative Bank': 'COOP.NR',
        'Standard Chartered Kenya': 'SCBK.NR',
        'Bamburi Cement': 'BAMB.NR',
        'Nation Media Group': 'NMG.NR',
        'Stanbic Holdings': 'CFC.NR',
        'British American Tobacco Kenya': 'BAT.NR',
        'Diamond Trust Bank': 'DTK.NR',
        'ABSA Bank Kenya': 'ABSA.NR',
        'Jubilee Holdings': 'JUB.NR',
        'Kenya Airways': 'KQ.NR',
        'Kenol Kobil': 'KENO.NR',
        'Longhorn Publishers': 'LKL.NR',
        'Total Energies Kenya': 'TOTE.NR',
        'Kakuzi': 'KUKZ.NR',
        'Williamson Tea': 'WTK.NR',
        'Carbacid Investments': 'CARB.NR',
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
indicators = st.sidebar.multiselect('Indicators', ['SMA', 'EMA', 'VWAP', 'RSI', 'Stochastic', 'MACD', 'Bollinger Bands', 'Parabolic SAR', 'Fibonacci'], default=['SMA', 'EMA', 'RSI', 'MACD'])
show_signals = st.sidebar.toggle('Show Buy/Sell Signals', value=True)
show_sr      = st.sidebar.toggle('Show Support & Resistance', value=True)

# Forecast settings
st.sidebar.header('Forecast Settings')
forecast_days = st.sidebar.number_input('Forecast Days', min_value=1, max_value=90, value=14)
train_split_pct = st.sidebar.slider('Train Split %', min_value=60, max_value=90, value=80, step=5,
    help='Percentage of data used for training vs. test evaluation')
n_cv_splits = st.sidebar.slider('CV Folds (TimeSeriesSplit)', min_value=2, max_value=10, value=5,
    help='Number of walk-forward cross-validation folds')

st.sidebar.header('RF Hyperparameters')
rf_n_estimators = st.sidebar.slider('RF: n_estimators', 50, 500, 300, step=50)
rf_max_depth = st.sidebar.slider('RF: max_depth', 3, 30, 15)
rf_min_samples_leaf = st.sidebar.slider('RF: min_samples_leaf', 1, 10, 2)
rf_max_features = st.sidebar.selectbox('RF: max_features', ['sqrt', 'log2', None], index=0)

st.sidebar.header('XGB Hyperparameters')
xgb_n_estimators = st.sidebar.slider('XGB: n_estimators', 50, 500, 300, step=50)
xgb_max_depth = st.sidebar.slider('XGB: max_depth', 2, 12, 7)
xgb_learning_rate = st.sidebar.select_slider('XGB: learning_rate',
    options=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3], value=0.05)
xgb_subsample = st.sidebar.slider('XGB: subsample', 0.5, 1.0, 0.8, step=0.05)
xgb_colsample = st.sidebar.slider('XGB: colsample_bytree', 0.5, 1.0, 0.8, step=0.05)
xgb_reg_alpha = st.sidebar.select_slider('XGB: reg_alpha (L1)',
    options=[0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0], value=0.1)
xgb_reg_lambda = st.sidebar.select_slider('XGB: reg_lambda (L2)',
    options=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0], value=1.0)

# Build param dicts to pass to train_models
rf_params = dict(
    n_estimators=rf_n_estimators,
    max_depth=rf_max_depth,
    min_samples_split=5,
    min_samples_leaf=rf_min_samples_leaf,
    max_features=rf_max_features,
)
xgb_params = dict(
    n_estimators=xgb_n_estimators,
    max_depth=xgb_max_depth,
    learning_rate=xgb_learning_rate,
    subsample=xgb_subsample,
    colsample_bytree=xgb_colsample,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=xgb_reg_alpha,
    reg_lambda=xgb_reg_lambda,
)

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
        st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;color:#ff9f00;font-size:0.7rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;border-bottom:1px solid #2a2a2a;padding-bottom:6px;margin:20px 0 12px 0;">▸ {selected_name}  /  TRADING CHART</div>', unsafe_allow_html=True)
        fig, sig_df = plot_price_and_indicators(df, f"{selected_name} ({ticker})", indicators, show_signals, show_sr)
        st.plotly_chart(fig, use_container_width=True)

        # ── Signal Dashboard ─────────────────────────────────────────
        st.markdown('<div style="font-family:IBM Plex Mono,monospace;color:#ff9f00;font-size:0.7rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;border-bottom:1px solid #2a2a2a;padding-bottom:6px;margin:20px 0 12px 0;">▸ SIGNAL DASHBOARD  /  CONFLUENCE ANALYSIS</div>', unsafe_allow_html=True)

        latest = sig_df.iloc[-1]
        score  = int(latest.get('confluence_score', 0))
        buy_ct = int(latest.get('confluence_buy', 0))
        sel_ct = int(latest.get('confluence_sell', 0))
        max_signals = 6

        # Overall rating
        if score >= 3:
            rating, rating_color, rating_bg = "STRONG BUY",  "#00d084", "#001a0e"
        elif score >= 1:
            rating, rating_color, rating_bg = "BUY",         "#00d084", "#001a0e"
        elif score <= -3:
            rating, rating_color, rating_bg = "STRONG SELL", "#ff3b5c", "#1a0008"
        elif score <= -1:
            rating, rating_color, rating_bg = "SELL",        "#ff3b5c", "#1a0008"
        else:
            rating, rating_color, rating_bg = "NEUTRAL",     "#8a8070", "#141414"

        bar_pct_buy  = int((buy_ct  / max_signals) * 100)
        bar_pct_sell = int((sel_ct  / max_signals) * 100)

        st.markdown(f"""
        <div style="display:flex;gap:16px;margin-bottom:16px;">
          <div style="flex:1;background:{rating_bg};border:1px solid #2a2a2a;border-left:3px solid {rating_color};
               padding:16px 20px;font-family:'IBM Plex Mono',monospace;">
            <div style="color:#504840;font-size:0.55rem;letter-spacing:0.2em;text-transform:uppercase;margin-bottom:4px;">OVERALL SIGNAL</div>
            <div style="color:{rating_color};font-size:1.6rem;font-weight:700;letter-spacing:0.1em;">{rating}</div>
            <div style="color:#504840;font-size:0.6rem;margin-top:4px;">Score: {score:+d}  ·  {buy_ct} bullish / {sel_ct} bearish indicators</div>
          </div>
          <div style="flex:2;background:#0f0f0f;border:1px solid #2a2a2a;padding:16px 20px;font-family:'IBM Plex Mono',monospace;">
            <div style="color:#504840;font-size:0.55rem;letter-spacing:0.2em;text-transform:uppercase;margin-bottom:10px;">INDICATOR BREAKDOWN</div>
            <div style="margin-bottom:8px;">
              <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                <span style="color:#00d084;font-size:0.65rem;">BUY  {buy_ct}/{max_signals}</span>
              </div>
              <div style="background:#1a1a1a;height:6px;border-radius:0;">
                <div style="background:#00d084;width:{bar_pct_buy}%;height:6px;"></div>
              </div>
            </div>
            <div>
              <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                <span style="color:#ff3b5c;font-size:0.65rem;">SELL  {sel_ct}/{max_signals}</span>
              </div>
              <div style="background:#1a1a1a;height:6px;border-radius:0;">
                <div style="background:#ff3b5c;width:{bar_pct_sell}%;height:6px;"></div>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Per-indicator status grid
        sig_names = ['RSI', 'MACD CROSS', 'EMA CROSS', 'STOCHASTIC', 'BOLLINGER', 'SAR FLIP']
        sig_buy_keys  = ['sig_rsi_buy',  'sig_macd_buy',  'sig_ema_buy',  'sig_stoch_buy',  'sig_bb_buy',  'sig_sar_buy']
        sig_sell_keys = ['sig_rsi_sell', 'sig_macd_sell', 'sig_ema_sell', 'sig_stoch_sell', 'sig_bb_sell', 'sig_sar_sell']

        cols_sig = st.columns(6)
        for i, (name, bk, sk) in enumerate(zip(sig_names, sig_buy_keys, sig_sell_keys)):
            is_buy  = bool(latest.get(bk,  False))
            is_sell = bool(latest.get(sk, False))
            if is_buy:
                color, label, bg = '#00d084', '▲ BUY', '#001a0e'
            elif is_sell:
                color, label, bg = '#ff3b5c', '▼ SELL', '#1a0008'
            else:
                color, label, bg = '#504840', '— NEUT', '#0f0f0f'
            with cols_sig[i]:
                st.markdown(f"""
                <div style="background:{bg};border:1px solid #2a2a2a;padding:10px 8px;
                     font-family:'IBM Plex Mono',monospace;text-align:center;">
                  <div style="color:#504840;font-size:0.5rem;letter-spacing:0.12em;margin-bottom:4px;">{name}</div>
                  <div style="color:{color};font-size:0.75rem;font-weight:700;">{label}</div>
                </div>""", unsafe_allow_html=True)

        # ── Signal History Table ───────────────────────────────────────
        with st.expander("📋 SIGNAL HISTORY  —  All buy/sell occurrences"):
            sig_hist = sig_df[sig_df['strong_buy'] | sig_df['strong_sell']].copy()
            if not sig_hist.empty:
                sig_hist['Type'] = sig_hist.apply(
                    lambda r: 'BUY' if r.get('strong_buy', False) else 'SELL', axis=1)
                price_col_name = 'Adj Close' if 'Adj Close' in sig_hist.columns else 'Close'
                sig_hist['Price']      = sig_hist[price_col_name].round(3)
                sig_hist['Buy Sigs']   = sig_hist['confluence_buy'].astype(int)
                sig_hist['Sell Sigs']  = sig_hist['confluence_sell'].astype(int)
                sig_hist['Net Score']  = sig_hist['confluence_score'].astype(int)
                display_cols = ['Type', 'Price', 'Buy Sigs', 'Sell Sigs', 'Net Score']
                st.dataframe(
                    sig_hist[display_cols].tail(50).sort_index(ascending=False),
                    use_container_width=True
                )
            else:
                st.info("No strong signals in the selected period.")

        # Price table and stats
        st.markdown('<div style="font-family:IBM Plex Mono,monospace;color:#ff9f00;font-size:0.7rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;border-bottom:1px solid #2a2a2a;padding-bottom:6px;margin:20px 0 12px 0;">▸ PRICE DATA  /  STATISTICS</div>', unsafe_allow_html=True)

        # Show key metrics in cards
        col1, col2, col3, col4 = st.columns(4)

        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0

        # Determine currency prefix
        if asset_category == 'Forex':
            fmt = lambda v: f"{v:.5f}"
        elif asset_category == 'Kenyan Stocks (NSE)':
            fmt = lambda v: f"KES {v:.2f}"
        else:
            fmt = lambda v: f"${v:.2f}"

        with col1:
            st.metric("Current Price", fmt(current_price), f"{price_change_pct:+.2f}%")
        with col2:
            st.metric("High (Period)", fmt(df['High'].max()))
        with col3:
            st.metric("Low (Period)", fmt(df['Low'].min()))
        with col4:
            avg_volume = df['Volume'].mean() if 'Volume' in df.columns else 0
            st.metric("Avg Volume", f"{avg_volume:,.0f}")

        # ── Risk / Reward Calculator ───────────────────────────────────
        st.markdown('<div style="font-family:IBM Plex Mono,monospace;color:#ff9f00;font-size:0.7rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;border-bottom:1px solid #2a2a2a;padding-bottom:6px;margin:20px 0 12px 0;">▸ RISK CALCULATOR  /  POSITION SIZING</div>', unsafe_allow_html=True)

        rr_col1, rr_col2, rr_col3, rr_col4 = st.columns(4)
        with rr_col1:
            rr_entry = st.number_input('Entry Price', value=float(round(current_price, 4)), format="%.4f")
        with rr_col2:
            rr_stop  = st.number_input('Stop Loss',   value=float(round(current_price * 0.97, 4)), format="%.4f")
        with rr_col3:
            rr_target = st.number_input('Take Profit', value=float(round(current_price * 1.06, 4)), format="%.4f")
        with rr_col4:
            rr_capital = st.number_input('Account Size', value=10000.0, step=1000.0)

        risk_pct_input = st.slider('Risk per trade (%)', min_value=0.5, max_value=5.0, value=1.0, step=0.5)

        if rr_entry > 0 and rr_stop != rr_entry:
            risk_per_unit   = abs(rr_entry - rr_stop)
            reward_per_unit = abs(rr_target - rr_entry)
            rr_ratio        = reward_per_unit / risk_per_unit if risk_per_unit else 0
            max_risk_amt    = rr_capital * (risk_pct_input / 100)
            position_units  = max_risk_amt / risk_per_unit if risk_per_unit else 0
            position_value  = position_units * rr_entry
            potential_gain  = position_units * reward_per_unit
            potential_loss  = position_units * risk_per_unit

            rr_m1, rr_m2, rr_m3, rr_m4, rr_m5 = st.columns(5)
            rr_m1.metric("R:R Ratio",       f"1 : {rr_ratio:.2f}",
                         "✅ Good" if rr_ratio >= 2 else "⚠️ Low" if rr_ratio >= 1 else "❌ Poor")
            rr_m2.metric("Position Size",   f"{position_units:,.2f} units")
            rr_m3.metric("Position Value",  fmt(position_value))
            rr_m4.metric("Max Loss",        fmt(potential_loss),   f"-{risk_pct_input:.1f}%")
            rr_m5.metric("Max Gain",        fmt(potential_gain),   f"+{rr_ratio * risk_pct_input:.1f}%")

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
        elif asset_category == 'Kenyan Stocks (NSE)':
            st.info(f"""
            **Nairobi Securities Exchange (NSE):**
            - Current Price: KES {current_price:.2f}
            - Daily Change: {price_change_pct:+.2f}%
            - Trading Hours: 9:00 AM – 3:00 PM EAT (Mon–Fri)
            - Currency: Kenyan Shilling (KES)
            - Regulator: Capital Markets Authority (CMA)
            - Monitor CBK policy rates, KES/USD exchange rate, and East Africa macro trends
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
            split_idx = int(len(model_df) * (train_split_pct / 100))
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

            st.markdown(f"""
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#504840;
                 background:#0f0f0f;border:1px solid #1a1a1a;padding:8px 14px;margin-bottom:12px;
                 display:inline-block;">
              FEATURES: <span style="color:#ff9f00;font-weight:700;">{len(features)}</span>
              &nbsp;·&nbsp; TRAIN ROWS: <span style="color:#ff9f00;font-weight:700;">{len(train)}</span>
              &nbsp;·&nbsp; TEST ROWS: <span style="color:#ff9f00;font-weight:700;">{len(test)}</span>
              &nbsp;·&nbsp; CV FOLDS: <span style="color:#ff9f00;font-weight:700;">{n_cv_splits}</span>
            </div>""", unsafe_allow_html=True)

            X_train = train[features]
            y_train = train['Adj Close']
            X_test = test[features]
            y_test = test['Adj Close']

            with st.spinner('Training models with cross-validation...'):
                results = train_models(X_train, y_train, X_test, y_test,
                                       rf_params, xgb_params, n_cv_splits)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div style="font-family:IBM Plex Mono,monospace;color:#00d084;font-size:0.65rem;font-weight:700;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:8px;">◆ RANDOM FOREST</div>', unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("MAE", f"{results['rf']['mae']:.3f}")
                m2.metric("RMSE", f"{results['rf']['rmse']:.3f}")
                m3.metric("MAPE", f"{results['rf']['mape']:.2f}%")
                st.markdown(f"""
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#504840;
                     background:#0f0f0f;border:1px solid #1a1a1a;padding:8px 12px;margin:6px 0;">
                  CV MAE (walk-forward): <span style="color:#00d084">{results['rf']['cv_mae_mean']:.3f}</span>
                  <span style="color:#2a2a2a"> ± </span>
                  <span style="color:#8a8070">{results['rf']['cv_mae_std']:.3f}</span>
                </div>""", unsafe_allow_html=True)
                if len(features) > 0:
                    importance_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': results['rf']['model'].feature_importances_
                    }).sort_values('Importance', ascending=False).head(10)
                    st.markdown('<div style="font-family:IBM Plex Mono,monospace;color:#504840;font-size:0.6rem;letter-spacing:0.12em;text-transform:uppercase;margin-top:10px;">TOP FEATURES</div>', unsafe_allow_html=True)
                    st.bar_chart(importance_df.set_index('Feature'))

            with col2:
                st.markdown('<div style="font-family:IBM Plex Mono,monospace;color:#0088ff;font-size:0.65rem;font-weight:700;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:8px;">◆ XGBOOST</div>', unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("MAE", f"{results['xgb']['mae']:.3f}")
                m2.metric("RMSE", f"{results['xgb']['rmse']:.3f}")
                m3.metric("MAPE", f"{results['xgb']['mape']:.2f}%")
                st.markdown(f"""
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#504840;
                     background:#0f0f0f;border:1px solid #1a1a1a;padding:8px 12px;margin:6px 0;">
                  CV MAE (walk-forward): <span style="color:#0088ff">{results['xgb']['cv_mae_mean']:.3f}</span>
                  <span style="color:#2a2a2a"> ± </span>
                  <span style="color:#8a8070">{results['xgb']['cv_mae_std']:.3f}</span>
                </div>""", unsafe_allow_html=True)
                if len(features) > 0:
                    importance_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': results['xgb']['model'].feature_importances_
                    }).sort_values('Importance', ascending=False).head(10)
                    st.markdown('<div style="font-family:IBM Plex Mono,monospace;color:#504840;font-size:0.6rem;letter-spacing:0.12em;text-transform:uppercase;margin-top:10px;">TOP FEATURES</div>', unsafe_allow_html=True)
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

            # RF confidence interval via per-tree std across future steps
            rf_tree_future = np.array([
                tree.predict(future_X) for tree in results['rf']['model'].estimators_
            ])
            rf_future_std = rf_tree_future.std(axis=0)
            ensemble_upper = ensemble_future + 1.96 * rf_future_std
            ensemble_lower = ensemble_future - 1.96 * rf_future_std

            future_out = pd.DataFrame({
                'Date': future_dates,
                'RF': rf_future,
                'XGB': xgb_future,
                'Ensemble': ensemble_future,
                'CI_Upper (95%)': ensemble_upper,
                'CI_Lower (95%)': ensemble_lower,
            })

            # Calculate forecast statistics
            current_price = df['Adj Close'].iloc[-1]
            avg_forecast = ensemble_future[-1]
            forecast_change = ((avg_forecast - current_price) / current_price) * 100

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                price_lbl = f"KES {current_price:.2f}" if asset_category == 'Kenyan Stocks (NSE)' else f"${current_price:.2f}"
                st.metric("Current Price", price_lbl)
            with col2:
                fcast_lbl = f"KES {avg_forecast:.2f}" if asset_category == 'Kenyan Stocks (NSE)' else f"${avg_forecast:.2f}"
                st.metric(f"Ensemble ({forecast_days}d)", fcast_lbl, f"{forecast_change:+.2f}%")
            with col3:
                ci_range = ensemble_upper[-1] - ensemble_lower[-1]
                ci_lbl = f"KES {ci_range:.2f}" if asset_category == 'Kenyan Stocks (NSE)' else f"${ci_range:.2f}"
                st.metric("95% CI Width", ci_lbl)
            with col4:
                forecast_direction = "📈 Bullish" if forecast_change > 0 else "📉 Bearish"
                st.metric("Signal", forecast_direction)

            with st.expander("📋 Full Forecast Table"):
                st.dataframe(future_out.set_index('Date').style.format({
                    c: '{:.3f}' for c in ['RF','XGB','Ensemble','CI_Upper (95%)','CI_Lower (95%)']
                }), use_container_width=True)

            fig_future = go.Figure()
            fig_future.add_trace(go.Scatter(x=df.index, y=df['Adj Close'],
                                           mode='lines', name='HISTORICAL',
                                           line=dict(color='#504840', width=1.5)))
            # 95% CI upper band (invisible, used as fill reference)
            fig_future.add_trace(go.Scatter(
                x=future_out['Date'], y=future_out['CI_Upper (95%)'],
                mode='lines', name='95% CI', showlegend=False,
                line=dict(color='rgba(255,159,0,0)', width=0),
            ))
            # 95% CI lower band with fill
            fig_future.add_trace(go.Scatter(
                x=future_out['Date'], y=future_out['CI_Lower (95%)'],
                mode='lines', name='95% CI Band',
                fill='tonexty', fillcolor='rgba(255,159,0,0.08)',
                line=dict(color='rgba(255,159,0,0)', width=0),
            ))
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
            if asset_category not in ['Stocks', 'Kenyan Stocks (NSE)']:
                st.info(f'Fundamental data is primarily available for stocks. {asset_category} may have limited fundamental data.')
            if asset_category == 'Kenyan Stocks (NSE)':
                st.info('Note: AlphaVantage coverage for NSE-listed stocks is limited. Use the price chart and technical indicators for analysis.')

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
                # For NSE tickers, strip the .NR suffix for news search
                news_ticker = ticker.replace('.NR', '') if ticker.endswith('.NR') else ticker
                sn = StockNews(news_ticker, save_news=False)
                news_df = sn.read_rss()
                st.write(news_df[['published', 'title', 'summary', 'sentiment_title', 'sentiment_summary']].head(10))
            except Exception as e:
                st.error(f'Failed to fetch news: {e}')

        with tab3:
            st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;color:#ff9f00;font-size:0.65rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;margin-bottom:8px;">▸ ASSET PROFILE  /  {selected_name}</div>', unsafe_allow_html=True)

            asset_info = {
                'Stocks': 'Equities representing ownership in publicly traded companies.',
                'Kenyan Stocks (NSE)': 'Equities listed on the Nairobi Securities Exchange (NSE), Kenya\'s principal stock exchange. Prices quoted in Kenyan Shillings (KES).',
                'Commodities': 'Physical goods including metals, energy, and agricultural products.',
                'Forex': 'Foreign exchange pairs showing relative value between currencies.',
                'Volatility Indices': 'Measures of market volatility and investor sentiment.',
                'Crypto': 'Digital/virtual currencies using cryptography for security.'
            }

            st.write(f"**Category:** {asset_category}")
            st.write(f"**Ticker:** {ticker}")
            st.write(f"**Description:** {asset_info.get(asset_category, 'Financial instrument')}")

            if asset_category == 'Kenyan Stocks (NSE)':
                st.markdown("""
                <div style="
                    background:#0f0f0f;border:1px solid #2a2a2a;border-left:3px solid #ff9f00;
                    padding:12px 16px;margin:10px 0;font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#8a8070;
                ">
                    <div style="color:#ff9f00;font-size:0.6rem;letter-spacing:0.2em;text-transform:uppercase;margin-bottom:6px;">▸ NSE MARKET INFO</div>
                    <div>Exchange: <span style="color:#e8e0d0;">Nairobi Securities Exchange (NSE)</span></div>
                    <div>Currency: <span style="color:#e8e0d0;">Kenyan Shilling (KES)</span></div>
                    <div>Regulator: <span style="color:#e8e0d0;">Capital Markets Authority (CMA)</span></div>
                    <div>Index: <span style="color:#e8e0d0;">NSE 20, NSE 25, NASI</span></div>
                    <div>Website: <span style="color:#ff9f00;">www.nse.co.ke</span></div>
                </div>
                """, unsafe_allow_html=True)

            # Add trading hours info
            st.markdown('<div style="font-family:IBM Plex Mono,monospace;color:#ff9f00;font-size:0.65rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;margin-bottom:8px;">▸ TRADING INFORMATION</div>', unsafe_allow_html=True)
            trading_hours = {
                'Stocks': '9:30 AM - 4:00 PM ET (Monday-Friday)',
                'Kenyan Stocks (NSE)': '9:00 AM - 3:00 PM EAT / 6:00 AM - 12:00 PM UTC (Monday-Friday)',
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
