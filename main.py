import streamlit as st 
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd 
import yfinance as yf 
import numpy as np 
from alpha_vantage.fundamentaldata import FundamentalData
from stocknews import StockNews
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Function to check if 'Date' column exists
def check_and_prepare_data(df):
    if 'Date' not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df['Date'] = df.index
        else:
            st.error("The 'Date' column is missing from model_data, and DateIndex is not present.")
            return None
    else:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if df['Date'].isna().any():
            st.warning("Some dates could not be parsed.")
    return df

# Function to calculate percentage accuracy
def calculate_percentage_accuracy(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

# Streamlit app
st.title('Stock Dashboard')
ticker = st.sidebar.text_input('Ticker')
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')
chart_type = st.sidebar.radio('Chart Type', ['Line', 'Candlestick'])
timeframe = st.sidebar.selectbox('Timeframe', ['1d', '5m', '30m', '1h', '4h'])
forecast_days = st.sidebar.number_input('Days to Forecast', min_value=1, value=7)

if ticker:
    data = yf.download(ticker, start=start_date, end=end_date, interval=timeframe)
    if not data.empty:
        if chart_type == 'Line':
            fig = px.line(data, x=data.index, y='Adj Close', title=ticker)
        elif chart_type == 'Candlestick':
            fig = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=ticker
            )])
            fig.update_layout(title=ticker)
        st.plotly_chart(fig)

        # Pricing Data
        st.header('Price Movements')
        data2 = data.copy()
        data2['% Change'] = data['Adj Close'].pct_change()
        data2.dropna(inplace=True)
        st.write(data2)
        annual_return = data2['% Change'].mean() * 252 * 100
        st.write(f'Annual Return is {annual_return:.2f}%')
        stdev = data2['% Change'].std() * np.sqrt(252) * 100
        st.write(f'Standard Deviation is {stdev:.2f}%')

        # Modeling
        st.header('Modeling')
        model_data = data[['Adj Close']].reset_index()
        model_data = check_and_prepare_data(model_data)
        
        if model_data is not None:
            model_data['Days'] = (model_data['Date'] - model_data['Date'].min()).dt.days

            X = model_data[['Days']]
            y = model_data['Adj Close']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            rf_predictions = rf_model.predict(X_test)
            rf_mae = mean_absolute_error(y_test, rf_predictions)
            rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
            rf_mape = calculate_percentage_accuracy(y_test, rf_predictions)

            xgb_model = XGBRegressor(n_estimators=100, random_state=42)
            xgb_model.fit(X_train, y_train)
            xgb_predictions = xgb_model.predict(X_test)
            xgb_mae = mean_absolute_error(y_test, xgb_predictions)
            xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
            xgb_mape = calculate_percentage_accuracy(y_test, xgb_predictions)

            st.subheader('RandomForest Regressor Performance')
            st.write(f'MAE: {rf_mae:.2f}')
            st.write(f'RMSE: {rf_rmse:.2f}')
            st.write(f'MAPE: {rf_mape:.2f}%')
            
            st.subheader('XGBoost Regressor Performance')
            st.write(f'MAE: {xgb_mae:.2f}')
            st.write(f'RMSE: {xgb_rmse:.2f}')
            st.write(f'MAPE: {xgb_mape:.2f}%')

            # Plot Predictions vs Actuals
            fig_rf = go.Figure()
            fig_rf.add_trace(go.Scatter(x=X_test['Days'], y=y_test, mode='markers', name='Actual'))
            fig_rf.add_trace(go.Scatter(x=X_test['Days'], y=rf_predictions, mode='lines', name='RF Predictions'))
            fig_rf.update_layout(title='RandomForest Predictions vs Actual', xaxis_title='Days', yaxis_title='Adj Close')
            st.plotly_chart(fig_rf)

            fig_xgb = go.Figure()
            fig_xgb.add_trace(go.Scatter(x=X_test['Days'], y=y_test, mode='markers', name='Actual'))
            fig_xgb.add_trace(go.Scatter(x=X_test['Days'], y=xgb_predictions, mode='lines', name='XGB Predictions'))
            fig_xgb.update_layout(title='XGBoost Predictions vs Actual', xaxis_title='Days', yaxis_title='Adj Close')
            st.plotly_chart(fig_xgb)

            # Future Predictions
            st.header('Future Predictions')
            last_date = data.index[-1]
            future_dates = [last_date + pd.DateOffset(days=i) for i in range(1, forecast_days + 1)]

            future_X = pd.DataFrame({
                'Days': [(date - model_data['Date'].min()).days for date in future_dates]
            })

            rf_future_preds = rf_model.predict(future_X)
            xgb_future_preds = xgb_model.predict(future_X)
            
            future_df = pd.DataFrame({
                'Date': future_dates,
                'RF Predictions': rf_future_preds,
                'XGB Predictions': xgb_future_preds
            })
            st.write(future_df)
            
            # Plot Future Predictions
            fig_future = go.Figure()
            fig_future.add_trace(go.Scatter(x=data.index, y=data['Adj Close'], mode='lines', name='Historical'))
            fig_future.add_trace(go.Scatter(x=future_df['Date'], y=future_df['RF Predictions'], mode='lines', name='RF Future Predictions'))
            fig_future.add_trace(go.Scatter(x=future_df['Date'], y=future_df['XGB Predictions'], mode='lines', name='XGB Future Predictions'))
            fig_future.update_layout(title='Future Predictions', xaxis_title='Date', yaxis_title='Adj Close')
            st.plotly_chart(fig_future)

    else:
        st.write("No data found for the selected ticker and date range.")
else:
    st.write("Please enter a ticker symbol.")

# Fundamental Data
fundamental_data, news = st.tabs(["Fundamental Data", "Top 10 News"])

with fundamental_data:
    if ticker:
        st.subheader('Balance Sheet')
        key = '5N1BBRGV7IWAG0ZA'
        fd = FundamentalData(key, output_format='pandas')
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

with news:
    if ticker:
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
    else:
        st.write("Please enter a ticker symbol.")
