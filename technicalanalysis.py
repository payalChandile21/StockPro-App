import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD

# Function to fetch stock data
def get_stock_data(ticker, period='1y'):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df

# Function to calculate trends
def trend(x):
    if x > -0.5 and x <= 0.5:
        return 'Slight or No change'
    elif x > 0.5 and x <= 1:
        return 'Slight Positive'
    elif x > -1 and x <= -0.5:
        return 'Slight Negative'
    elif x > 1 and x <= 3:
        return 'Positive'
    elif x > -3 and x <= -1:
        return 'Negative'
    elif x > 3 and x <= 7:
        return 'Among top gainers'
    elif x > -7 and x <= -3:
        return 'Among top losers'
    elif x > 7:
        return 'Bull run'
    elif x <= -7:
        return 'Bear drop'

# Function to calculate RSI buy and sell signals
def rsi_signals(df):
    rsi = RSIIndicator(df['Close'], window=14)
    df['RSI'] = rsi.rsi()
    df['Buy_Signal_RSI'] = df['RSI'] < 40
    df['Sell_Signal_RSI'] = df['RSI'] > 70
    return df

# Function to calculate Bollinger Bands buy and sell signals
def bollinger_signals(df):
    bb = BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['Buy_Signal_BB'] = df['Close'] < df['BB_Lower']
    df['Sell_Signal_BB'] = df['Close'] > df['BB_Upper']
    return df

# Function to calculate MACD buy and sell signals
def macd_signals(df):
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['Buy_Signal_MACD'] = df['MACD'] > df['MACD_Signal']
    df['Sell_Signal_MACD'] = df['MACD'] < df['MACD_Signal']
    return df

# Streamlit App
st.title("Stock Market Technical Analysis")
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT)", "AAPL")
period = st.selectbox("Select Time Period", ['1mo', '3mo', '6mo', '1y', '2y', '5y'], index=3)

if st.button("Analyze"):
    df = get_stock_data(ticker, period)
    df['Day_Perc_Change'] = df['Close'].pct_change() * 100
    df.dropna(inplace=True)
    df['Trend'] = df['Day_Perc_Change'].apply(lambda x: trend(x))
    df = rsi_signals(df)
    df = bollinger_signals(df)
    df = macd_signals(df)

    # Plot Market Trends
    st.subheader("Market Trend Analysis")
    fig, ax = plt.subplots()
    df['Trend'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
    st.pyplot(fig)

    # Plot RSI with Buy/Sell Signals
    st.subheader("RSI Indicator with Buy/Sell Signals")
    fig, ax = plt.subplots()
    ax.plot(df.index, df['RSI'], label='RSI', color='blue')
    ax.axhline(70, linestyle='--', color='red', label='Overbought')
    ax.axhline(30, linestyle='--', color='green', label='Oversold')
    ax.scatter(df.index[df['Buy_Signal_RSI']], df['RSI'][df['Buy_Signal_RSI']], marker='^', color='green', label='Buy Signal', alpha=1)
    ax.scatter(df.index[df['Sell_Signal_RSI']], df['RSI'][df['Sell_Signal_RSI']], marker='v', color='red', label='Sell Signal', alpha=1)
    ax.legend()
    st.pyplot(fig)

    # Plot Bollinger Bands with Buy/Sell Signals
    st.subheader("Bollinger Bands with Buy/Sell Signals")
    fig, ax = plt.subplots()
    ax.plot(df.index, df['Close'], label='Close Price', color='black')
    ax.plot(df.index, df['BB_Upper'], label='Upper Band', color='red')
    ax.plot(df.index, df['BB_Lower'], label='Lower Band', color='green')
    ax.fill_between(df.index, df['BB_Lower'], df['BB_Upper'], alpha=0.1)
    ax.scatter(df.index[df['Buy_Signal_BB']], df['Close'][df['Buy_Signal_BB']], marker='^', color='green', label='Buy Signal', alpha=1)
    ax.scatter(df.index[df['Sell_Signal_BB']], df['Close'][df['Sell_Signal_BB']], marker='v', color='red', label='Sell Signal', alpha=1)
    ax.legend()
    st.pyplot(fig)

    # Plot MACD with Buy/Sell Signals
    st.subheader("MACD Indicator with Buy/Sell Signals")
    fig, ax = plt.subplots()
    ax.plot(df.index, df['MACD'], label='MACD', color='blue')
    ax.plot(df.index, df['MACD_Signal'], label='Signal Line', color='red')
    ax.scatter(df.index[df['Buy_Signal_MACD']], df['MACD'][df['Buy_Signal_MACD']], marker='^', color='green', label='Buy Signal', alpha=1)
    ax.scatter(df.index[df['Sell_Signal_MACD']], df['MACD'][df['Sell_Signal_MACD']], marker='v', color='red', label='Sell Signal', alpha=1)
    ax.legend()
    st.pyplot(fig)

    # Moving Averages
    st.subheader("Moving Averages")
    df['50_SMA'] = df['Close'].rolling(window=50, min_periods=1).mean()
    df['200_SMA'] = df['Close'].rolling(window=200, min_periods=1).mean()
    fig, ax = plt.subplots()
    df['Close'].plot(ax=ax, color='k', label='Close Price')
    df['50_SMA'].plot(ax=ax, color='b', label='50-day SMA')
    df['200_SMA'].plot(ax=ax, color='g', label='200-day SMA')
    ax.legend()
    st.pyplot(fig)

    st.write("Data Preview:")
    st.dataframe(df.tail())
