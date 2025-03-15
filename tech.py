import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# Load Data (User should upload CSV)
st.title("Stock Market Technical Analysis")
uploaded_file = st.file_uploader("Upload a CSV file with 'Close' prices", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
    st.write("Data Preview:", df.head())

    # Trend Analysis
    def trend(x):
        if -0.5 < x <= 0.5: return 'Slight or No change'
        elif 0.5 < x <= 1: return 'Slight Positive'
        elif -1 < x <= -0.5: return 'Slight Negative'
        elif 1 < x <= 3: return 'Positive'
        elif -3 < x <= -1: return 'Negative'
        elif 3 < x <= 7: return 'Among top gainers'
        elif -7 < x <= -3: return 'Among top losers'
        elif x > 7: return 'Bull run'
        else: return 'Bear drop'

    df['Trend'] = df['Close'].pct_change().apply(lambda x: trend(x))
    st.subheader("Trend Analysis")
    fig, ax = plt.subplots()
    df['Trend'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
    ax.set_ylabel('')
    st.pyplot(fig)

    # Moving Averages
    st.subheader("Simple & Exponential Moving Averages")
    df['50_SMA'] = df['Close'].rolling(window=50, min_periods=1).mean()
    df['200_SMA'] = df['Close'].rolling(window=200, min_periods=1).mean()
    df['50_EMA'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['200_EMA'] = df['Close'].ewm(span=200, adjust=False).mean()

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df.index, df['Close'], label='Close Price', color='black')
    ax.plot(df.index, df['50_SMA'], label='50-day SMA', color='blue')
    ax.plot(df.index, df['200_SMA'], label='200-day SMA', color='green')
    ax.plot(df.index, df['50_EMA'], label='50-day EMA', linestyle='dashed', color='blue')
    ax.plot(df.index, df['200_EMA'], label='200-day EMA', linestyle='dashed', color='green')
    ax.legend()
    st.pyplot(fig)

    # RSI
    st.subheader("Relative Strength Index (RSI)")
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df.index, df['RSI'], label='RSI', color='blue')
    ax.axhline(30, linestyle='--', color='red')
    ax.axhline(70, linestyle='--', color='green')
    ax.legend()
    st.pyplot(fig)

    # Bollinger Bands
    st.subheader("Bollinger Bands")
    indicator_bb = BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Middle'] = indicator_bb.bollinger_mavg()
    df['BB_Upper'] = indicator_bb.bollinger_hband()
    df['BB_Lower'] = indicator_bb.bollinger_lband()

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df.index, df['Close'], label='Close Price', color='black')
    ax.plot(df.index, df['BB_Middle'], label='Middle Band', color='blue')
    ax.plot(df.index, df['BB_Upper'], label='Upper Band', color='green')
    ax.plot(df.index, df['BB_Lower'], label='Lower Band', color='red')
    ax.fill_between(df.index, df['BB_Lower'], df['BB_Upper'], color='gray', alpha=0.3)
    ax.legend()
    st.pyplot(fig)
