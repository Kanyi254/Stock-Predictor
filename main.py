import math
import numpy as np
import datetime as dt
import yfinance as yf
import streamlit as st
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU
import plotly.graph_objs as go

# ------------------ Helper Functions -------------------

# Load stock data for multiple tickers
def get_multiple_stock_data(tickers, start, end):
    stock_data = {}
    for ticker in tickers:
        stock_data[ticker] = yf.download(ticker, start=start, end=end)
    return stock_data

# Plot the predictions and actual data using Plotly
def plot_predictions_plotly(actual, predicted, title="Stock Price Prediction"):
    fig = go.Figure()
    
    # Actual stock prices
    fig.add_trace(go.Scatter(x=actual.index, y=actual.values, mode='lines', name='Actual'))
    
    # Predicted stock prices
    fig.add_trace(go.Scatter(x=actual.index, y=predicted, mode='lines', name='Predicted'))

    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)

# Backtesting strategy
def backtest_strategy(predictions, actual, initial_capital=10000):
    capital = initial_capital
    position = 0  # No stock held initially
    shares = 0

    for i in range(1, len(predictions)):
        if predictions[i] > actual[i-1] and position == 0:  # Buy signal
            shares = capital // actual[i-1]
            capital -= shares * actual[i-1]
            position = 1
        elif predictions[i] < actual[i-1] and position == 1:  # Sell signal
            capital += shares * actual[i-1]
            position = 0

    # Final capital after selling any remaining shares
    if position == 1:
        capital += shares * actual[-1]

    return capital

# Prepare and scale data for machine learning models
def prepare_data(df):
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()

    X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = df['Target']
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    
    return X_train, X_test, y_train, y_test, scaler

# Build and train the LSTM model
def build_lstm_model(input_shape, learning_rate):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Build and train the GRU model
def build_gru_model(input_shape, learning_rate):
    model = Sequential()
    model.add(GRU(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Build and train the XGBoost model
def build_xgboost_model(X_train, y_train):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {'objective': 'reg:squarederror'}
    xgb_model = xgb.train(params, dtrain, num_boost_round=100)
    return xgb_model

# ------------------ Streamlit UI -------------------

st.title('Stock Price Predictor with Backtesting')

# User input for stock tickers and date range
tickers = st.text_input("Enter Stock Tickers (comma-separated)", "AAPL, GOOG")
start_date = st.date_input("Start Date", dt.date(2020, 1, 1))
end_date = st.date_input("End Date", dt.date.today())

# Model selection
model_choice = st.selectbox('Select Model', ['Linear Regression', 'LSTM', 'GRU', 'XGBoost'])

# Hyperparameter inputs
epochs = st.slider('Number of epochs', min_value=10, max_value=100, value=10)
batch_size = st.slider('Batch size', min_value=16, max_value=128, value=32)
learning_rate = st.number_input('Learning rate', min_value=0.0001, max_value=0.1, value=0.001)

# Button to trigger prediction
if st.button('Predict and Backtest'):
    # Fetch stock data
    data_dict = get_multiple_stock_data(tickers.split(','), start_date, end_date)
    
    for ticker, df in data_dict.items():
        st.subheader(f'Predictions for {ticker}')
        
        # Prepare data for prediction
        X_train, X_test, y_train, y_test, scaler = prepare_data(df)
        
        # Select and train model based on user choice
        if model_choice == 'LSTM':
            X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
            model = build_lstm_model((X_train_reshaped.shape[1], X_train_reshaped.shape[2]), learning_rate)
            model.fit(X_train_reshaped, y_train, epochs=epochs, batch_size=batch_size)
            predictions = model.predict(X_test_reshaped)
        elif model_choice == 'GRU':
            X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
            model = build_gru_model((X_train_reshaped.shape[1], X_train_reshaped.shape[2]), learning_rate)
            model.fit(X_train_reshaped, y_train, epochs=epochs, batch_size=batch_size)
            predictions = model.predict(X_test_reshaped)
        elif model_choice == 'XGBoost':
            model = build_xgboost_model(X_train, y_train)
            dtest = xgb.DMatrix(X_test)
            predictions = model.predict(dtest)
        else:  # Linear Regression
            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
        
        # Scale back the predictions
        predictions = scaler.inverse_transform(np.column_stack((np.zeros_like(predictions), predictions)))[:, 1]
        
        # Backtest strategy
        final_capital = backtest_strategy(predictions, df['Close'][-len(predictions):])
        st.write(f"Final capital after backtesting: ${final_capital:.2f}")
        
        # Plot the results
        plot_predictions_plotly(df['Close'][-len(predictions):], predictions)
