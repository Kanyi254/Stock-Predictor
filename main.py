import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import talib
from sklearn.model_selection import train_test_split
import xgboost as xgb

st.title('Stock Dashboard with Advanced Modeling')

# Sidebar for user inputs
ticker = st.sidebar.text_input('Ticker')
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')

# Timeframe selection
timeframe = st.sidebar.selectbox('Select Timeframe', ['5-min', '30-min', '1-hr', '4-hrs', '24-hrs'])
timeframes_map = {
    '5-min': '5m',
    '30-min': '30m',
    '1-hr': '1h',
    '4-hrs': '4h',
    '24-hrs': '1d'
}
chart_type = st.sidebar.radio('Chart Type', ['Line', 'Candlestick'])

# Download data
if ticker:
    data = yf.download(ticker, start=start_date, end=end_date, interval=timeframes_map[timeframe])

    if not data.empty:
        # Plot Line or Candlestick Chart
        if chart_type == 'Line':
            fig = px.line(data, x=data.index, y='Adj Close', title=f"{ticker} ({timeframe})")
        elif chart_type == 'Candlestick':
            fig = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=ticker
            )])
            fig.update_layout(title=f"{ticker} ({timeframe})")
        st.plotly_chart(fig)

        # Add Indicators (Moving Averages, Bollinger Bands, RSI)
        st.subheader('Technical Indicators')

        # Simple Moving Averages
        data['SMA_20'] = data['Adj Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Adj Close'].rolling(window=50).mean()

        # Bollinger Bands
        data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = talib.BBANDS(data['Adj Close'], timeperiod=20)

        # Relative Strength Index (RSI)
        data['RSI'] = talib.RSI(data['Adj Close'], timeperiod=14)

        # Plot indicators
        st.line_chart(data[['Adj Close', 'SMA_20', 'SMA_50']], use_container_width=True)
        st.line_chart(data[['Adj Close', 'BB_Upper', 'BB_Middle', 'BB_Lower']], use_container_width=True)
        st.line_chart(data[['RSI']], use_container_width=True)

        # Show Technical Indicators Data
        st.write(data[['Adj Close', 'SMA_20', 'SMA_50', 'BB_Upper', 'BB_Lower', 'RSI']].tail())

        # Advanced Modeling with RandomForest and XGBoost
        st.subheader('Price Prediction Model (RandomForest and XGBoost)')

        # Feature Engineering for Modeling
        model_data = data[['Adj Close', 'SMA_20', 'SMA_50', 'BB_Upper', 'BB_Lower', 'RSI']].dropna().reset_index()
        model_data['Days'] = (model_data['Date'] - model_data['Date'].min()).dt.days

        # Train-Test Split
        X = model_data[['Days', 'SMA_20', 'SMA_50', 'BB_Upper', 'BB_Lower', 'RSI']]
        y = model_data['Adj Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # RandomForest Regression Model
        rf_model = RandomForestRegressor(n_estimators=100)
        rf_model.fit(X_train, y_train)
        rf_predictions = rf_model.predict(X_test)

        # XGBoost Regression Model
        xgb_model = xgb.XGBRegressor(n_estimators=100)
        xgb_model.fit(X_train, y_train)
        xgb_predictions = xgb_model.predict(X_test)

        # Evaluate the models
        rf_mae = mean_absolute_error(y_test, rf_predictions)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
        xgb_mae = mean_absolute_error(y_test, xgb_predictions)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))

        st.write(f"**RandomForest Model - MAE:** {rf_mae:.2f}, **RMSE:** {rf_rmse:.2f}")
        st.write(f"**XGBoost Model - MAE:** {xgb_mae:.2f}, **RMSE:** {xgb_rmse:.2f}")

        # Plot Actual vs Predicted
        model_data_test = model_data.iloc[X_test.index]
        model_data_test['RF_Predictions'] = rf_predictions
        model_data_test['XGB_Predictions'] = xgb_predictions

        fig_model = px.line(model_data_test, x='Date', y=['Adj Close', 'RF_Predictions', 'XGB_Predictions'], 
                            title="Actual vs Predicted Prices")
        st.plotly_chart(fig_model)
    
    else:
        st.write("No data found for the selected ticker and date range.")
else:
    st.write("Please enter a ticker symbol.")

# Add Pricing, Fundamental Data, and News Tabs
pricing_data, fundamental_data, news = st.tabs(["Pricing Data", "Fundamental Data", "Top 10 News"])

with pricing_data:
    st.header('Price Movements')
    if not data.empty:
        data2 = data.copy()
        data2['% Change'] = data['Adj Close'].pct_change()
        data2.dropna(inplace=True)
        st.write(data2)
        annual_return = data2['% Change'].mean() * 252 * 100
        st.write(f'Annual Return is {annual_return:.2f}%')
        stdev = data2['% Change'].std() * np.sqrt(252) * 100
        st.write(f'Standard Deviation is {stdev:.2f}%')

# Add Fundamental Data (Alpha Vantage)
from alpha_vantage.fundamentaldata import FundamentalData
with fundamental_data:
    key = 'GET YOUR OWN API KEY DUMBBELL'
    fd = FundamentalData(key, output_format='pandas')
    st.subheader('Balance Sheet')
    balance_sheet = fd.get_balance_sheet_annual(ticker)[0]
    bs = balance_sheet.T[2:]
    bs.columns = list(balance_sheet.T.iloc[0])
    st.write(bs)
    st.subheader('Income Statement')
    income_statement = fd.get_income_statement_annual(ticker)[0]
    is1 = income_statement.T[2:]
    is1.columns = list(income_statement.T.iloc[0])
    st.write(is1)
    st.subheader('Cash Flow Statement')
    cash_flow = fd.get_cash_flow_annual(ticker)[0]
    cf = cash_flow.T[2:]
    cf.columns = list(cash_flow.T.iloc[0])
    st.write(cf)

# Add Stock News (StockNews API)
from stocknews import StockNews
with news:
    st.header(f'Stock News of {ticker}')
    sn = StockNews(ticker, save_news=False)
    df_news = sn.read_rss()
    for i in range(10):
        st.subheader(f'News {i+1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        title_sentiment = df_news['sentiment_title'][i]
        st.write(f'Title Sentiment: {title_sentiment}')
        news_sentiment = df_news['sentiment_summary'][i]
        st.write(f'News Sentiment: {news_sentiment}')
