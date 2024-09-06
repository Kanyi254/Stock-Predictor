import streamlit as st 
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd 
import yfinance as yf 
import numpy as np 

st.title('Stock Dashboard')
ticker = st.sidebar.text_input('Ticker')
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End date')
chart_type = st.sidebar.radio('Chart Type', ['Line', 'Candlestick'])

if ticker:
    data = yf.download(ticker, start=start_date, end=end_date)
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
    else:
        st.write("No data found for the selected ticker and date range.")
else:
    st.write("Please enter a ticker symbol.")

pricing_data, fundamental_data, news = st.tabs(["Pricing Data", "Fundamental Data", "Top 10 News"])

with pricing_data:
    st.header('Price Movements')
    data2 = data.copy()
    data2['% Change'] = data['Adj Close'].pct_change()
    data2.dropna(inplace=True)
    st.write(data2)
    annual_return = data2['% Change'].mean() * 252 * 100
    st.write(f'Annual Return is {annual_return:.2f}%')
    stdev = data2['% Change'].std() * np.sqrt(252) * 100
    st.write(f'Standard Deviation is {stdev:.2f}%')

from alpha_vantage.fundamentaldata import FundamentalData
with fundamental_data:
    key = 'YOUR_ALPHA_VANTAGE_API_KEY'
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
