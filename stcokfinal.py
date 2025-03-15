import streamlit as st
from datetime import date, datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD

# Configure page
st.set_page_config(
    page_title="Stock Analysis & Forecast App",
    page_icon="📈",
    layout="wide"
)

# App title and description
st.title('📈 Stock Analysis & Forecast App')
st.markdown("""
This app provides comprehensive stock analysis using technical indicators and forecasts future prices using Facebook's Prophet model.
""")

# Define available stock options
stocks = {
    "Google": "GOOGL",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Tesla": "TSLA",
    "Tata Motors": "TATAMOTORS.NS"
}

def fetch_data_from_yfinance(ticker, period='5y'):
    """Fetch stock data using yfinance"""
    try:
        # Get data for the specified period
        end_date = datetime.now()

        if period == '5y' or period == '1825d':
            start_date = end_date - timedelta(days=1825)
        else:
            # For technical analysis periods
            df = yf.download(ticker, period=period)
            return df

        # Download data for forecast
        df = yf.download(ticker, start=start_date, end=end_date)

        if not df.empty and len(df) > 30:
            return df
        return None
    except Exception as e:
        st.error(f"Error fetching data from Yahoo Finance: {str(e)}")
        return None

def get_sample_data(ticker):
    """Generate sample stock data for demonstration"""
    seed_value = sum(ord(c) for c in ticker)
    np.random.seed(seed_value)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=1825)
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')

    # Set initial price based on ticker
    start_price = {
        'AAPL': 150,
        'GOOGL': 2800,
        'MSFT': 300,
        'TSLA': 800
    }.get(ticker, 100)

    # Generate price data
    daily_returns = np.random.normal(0.0005, 0.02, len(date_range))
    price_series = start_price * (1 + daily_returns).cumprod()

    # Create DataFrame
    df = pd.DataFrame(index=date_range)
    df['Close'] = price_series
    df['Open'] = df['Close'].shift(1) * (1 + np.random.normal(0, 0.005, len(df)))
    df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + abs(np.random.normal(0, 0.003, len(df))))
    df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - abs(np.random.normal(0, 0.003, len(df))))
    df['Adj Close'] = df['Close']
    df['Volume'] = np.random.randint(100000, 10000000, len(df))

    return df.fillna(method='bfill')

def prepare_dataframe(df):
    """Prepare dataframe for Prophet model"""
    df = df.reset_index()
    result_df = pd.DataFrame()
    result_df['ds'] = df['Date']
    result_df['y'] = df['Close']
    return result_df

@st.cache_data(show_spinner=False)
def fetch_stock_data(stock_ticker, period='5y'):
    """Fetch stock data with fallback to sample data"""
    with st.spinner(f"Fetching data for {stock_ticker}..."):
        df = fetch_data_from_yfinance(stock_ticker, period)
        if df is not None and not df.empty and len(df) > 30:
            st.success(f"Successfully loaded data for {stock_ticker}")
            if period == '5y' or period == '1825d':
                return prepare_dataframe(df), df
            return df

        st.warning(f"Using sample data for {stock_ticker}")
        sample_data = get_sample_data(stock_ticker)
        if period == '5y' or period == '1825d':
            return prepare_dataframe(sample_data), sample_data
        return sample_data

# Technical Analysis Functions
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

def rsi_signals(df):
    rsi = RSIIndicator(df['Close'], window=14)
    df['RSI'] = rsi.rsi()
    df['Buy_Signal_RSI'] = df['RSI'] < 40
    df['Sell_Signal_RSI'] = df['RSI'] > 70
    return df

def bollinger_signals(df):
    bb = BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['Buy_Signal_BB'] = df['Close'] < df['BB_Lower']
    df['Sell_Signal_BB'] = df['Close'] > df['BB_Upper']
    return df

def macd_signals(df):
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['Buy_Signal_MACD'] = df['MACD'] > df['MACD_Signal']
    df['Sell_Signal_MACD'] = df['MACD'] < df['MACD_Signal']
    return df

def perform_technical_analysis(df):
    """Calculate all technical indicators and signals"""
    df['Day_Perc_Change'] = df['Close'].pct_change() * 100
    df.dropna(inplace=True)
    df['Trend'] = df['Day_Perc_Change'].apply(lambda x: trend(x))
    df = rsi_signals(df)
    df = bollinger_signals(df)
    df = macd_signals(df)

    # Add moving averages
    df['50_SMA'] = df['Close'].rolling(window=50, min_periods=1).mean()
    df['200_SMA'] = df['Close'].rolling(window=200, min_periods=1).mean()
    return df

def display_technical_analysis(df, ticker):
    """Display technical analysis charts and data"""
    st.header(f"Technical Analysis for {ticker}")

    # Market Trend Analysis
    st.subheader("Market Trend Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        df['Trend'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, colors=plt.cm.Paired.colors)
        plt.title('Distribution of Price Movements')
        st.pyplot(fig)

    with col2:
        st.markdown("### Trend Interpretation")
        st.write("""
        This pie chart shows the distribution of daily price movements categorized by trend strength.
        - **Bull run/Bear drop**: Extreme price movements (>7%)
        - **Top gainers/losers**: Strong movements (3-7%)
        - **Positive/Negative**: Moderate movements (1-3%)
        - **Slight changes**: Minor movements (0.5-1%)
        - **No change**: Minimal price movement (<0.5%)
        """)

    # RSI with Buy/Sell Signals
    st.subheader("RSI Indicator with Buy/Sell Signals")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['RSI'], label='RSI', color='blue')
    ax.axhline(70, linestyle='--', color='red', label='Overbought (70)')
    ax.axhline(30, linestyle='--', color='green', label='Oversold (30)')
    ax.scatter(df.index[df['Buy_Signal_RSI']], df['RSI'][df['Buy_Signal_RSI']],
              marker='^', color='green', label='Buy Signal', alpha=1)
    ax.scatter(df.index[df['Sell_Signal_RSI']], df['RSI'][df['Sell_Signal_RSI']],
              marker='v', color='red', label='Sell Signal', alpha=1)
    ax.set_title('Relative Strength Index (RSI)')
    ax.set_ylabel('RSI Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Bollinger Bands with Buy/Sell Signals
    st.subheader("Bollinger Bands with Buy/Sell Signals")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['Close'], label='Close Price', color='black')
    ax.plot(df.index, df['BB_Upper'], label='Upper Band', color='red')
    ax.plot(df.index, df['BB_Lower'], label='Lower Band', color='green')
    ax.fill_between(df.index, df['BB_Lower'], df['BB_Upper'], alpha=0.1)
    ax.scatter(df.index[df['Buy_Signal_BB']], df['Close'][df['Buy_Signal_BB']],
              marker='^', color='green', label='Buy Signal', alpha=1)
    ax.scatter(df.index[df['Sell_Signal_BB']], df['Close'][df['Sell_Signal_BB']],
              marker='v', color='red', label='Sell Signal', alpha=1)
    ax.set_title('Bollinger Bands')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # MACD with Buy/Sell Signals
    st.subheader("MACD Indicator with Buy/Sell Signals")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['MACD'], label='MACD', color='blue')
    ax.plot(df.index, df['MACD_Signal'], label='Signal Line', color='red')
    ax.bar(df.index, df['MACD'] - df['MACD_Signal'], color=df.apply(
        lambda x: 'green' if x['MACD'] > x['MACD_Signal'] else 'red', axis=1),
        alpha=0.5, label='Histogram')
    ax.set_title('Moving Average Convergence Divergence (MACD)')
    ax.set_ylabel('MACD Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Moving Averages
    st.subheader("Moving Averages")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['Close'], color='k', label='Close Price')
    ax.plot(df.index, df['50_SMA'], color='b', label='50-day SMA')
    ax.plot(df.index, df['200_SMA'], color='g', label='200-day SMA')
    # Highlight golden cross and death cross
    golden_cross = (df['50_SMA'] > df['200_SMA']) & (df['50_SMA'].shift() <= df['200_SMA'].shift())
    death_cross = (df['50_SMA'] < df['200_SMA']) & (df['50_SMA'].shift() >= df['200_SMA'].shift())
    if golden_cross.any():
        golden_dates = df.index[golden_cross]
        for date in golden_dates:
            ax.axvline(x=date, color='gold', linestyle='--', alpha=0.7)
    if death_cross.any():
        death_dates = df.index[death_cross]
        for date in death_dates:
            ax.axvline(x=date, color='darkred', linestyle='--', alpha=0.7)
    ax.set_title('Moving Averages (50-day and 200-day)')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Technical Analysis Summary
    st.subheader("Technical Analysis Summary")

    latest = df.iloc[-1]
    col1, col2, col3 = st.columns(3)

    with col1:
        rsi_status = "Overbought" if latest['RSI'] > 70 else "Oversold" if latest['RSI'] < 30 else "Neutral"
        st.metric("RSI Value", f"{latest['RSI']:.2f}", delta=rsi_status)

    with col2:
        bb_position = ((latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower'])) * 100
        bb_status = "Upper Band" if bb_position > 80 else "Lower Band" if bb_position < 20 else "Middle"
        st.metric("Bollinger Position", f"{bb_position:.1f}%", delta=bb_status)

    with col3:
        macd_signal = "Bullish" if latest['MACD'] > latest['MACD_Signal'] else "Bearish"
        macd_diff = latest['MACD'] - latest['MACD_Signal']
        st.metric("MACD Signal", macd_signal, delta=f"{macd_diff:.4f}")

    # Technical signals summary
    signal_df = pd.DataFrame({
        'Indicator': ['RSI', 'Bollinger Bands', 'MACD', 'Moving Averages'],
        'Value': [
            f"{latest['RSI']:.2f}",
            f"Width: {((latest['BB_Upper'] - latest['BB_Lower'])/latest['Close']*100):.2f}%",
            f"MACD: {latest['MACD']:.4f}, Signal: {latest['MACD_Signal']:.4f}",
            f"50-day: {latest['50_SMA']:.2f}, 200-day: {latest['200_SMA']:.2f}"
        ],
        'Signal': [
            "Buy" if latest['Buy_Signal_RSI'] else "Sell" if latest['Sell_Signal_RSI'] else "Hold",
            "Buy" if latest['Buy_Signal_BB'] else "Sell" if latest['Sell_Signal_BB'] else "Hold",
            "Buy" if latest['Buy_Signal_MACD'] else "Sell",
            "Bullish" if latest['50_SMA'] > latest['200_SMA'] else "Bearish"
        ]
    })

    st.table(signal_df)

    # Data Preview
    st.subheader("Recent Data and Indicators")
    st.dataframe(df.tail())

# Sidebar inputs
st.sidebar.header('Settings')
selected_stock = st.sidebar.selectbox("Select Stock", list(stocks.keys()))
analysis_mode = st.sidebar.selectbox("Analysis Mode", ["Forecast", "Technical Analysis", "Both"])

# Technical analysis settings
if analysis_mode in ["Technical Analysis", "Both"]:
    st.sidebar.subheader("Technical Analysis Settings")
    ta_period = st.sidebar.selectbox(
        "Technical Analysis Period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3
    )

# Forecast settings
if analysis_mode in ["Forecast", "Both"]:
    st.sidebar.subheader("Forecast Settings")
    n_years = st.sidebar.slider("Prediction Years", 1, 4, 2)
    period = n_years * 365

# Main app flow
try:
    # Fetch data
    ticker = stocks[selected_stock]

    if analysis_mode == "Forecast":
        df_train, raw_data = fetch_stock_data(ticker)
        show_forecast = True
        show_technical = False
    elif analysis_mode == "Technical Analysis":
        df_tech = fetch_stock_data(ticker, ta_period)
        df_tech = perform_technical_analysis(df_tech)
        show_forecast = False
        show_technical = True
    else:  # Both
        df_train, raw_data = fetch_stock_data(ticker)
        df_tech = fetch_stock_data(ticker, ta_period)
        df_tech = perform_technical_analysis(df_tech)
        show_forecast = True
        show_technical = True

    # Display Technical Analysis if selected
    if show_technical:
        display_technical_analysis(df_tech, selected_stock)

        if not show_forecast:
            # Disclaimer
            st.info("""
            📊 **Disclaimer**: This technical analysis is for educational purposes only. Stock markets are influenced by many factors
            not captured in these indicators. Always conduct thorough research and consult with financial advisors before making
            investment decisions.
            """)

    # Display Forecast if selected
    if show_forecast:
        st.header("Stock Price Forecast")

        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${df_train['y'].iloc[-1]:.2f}")
        with col2:
            st.metric("Trading Days", len(df_train))
        with col3:
            price_change = ((df_train['y'].iloc[-1] - df_train['y'].iloc[0]) / df_train['y'].iloc[0]) * 100
            st.metric("Total Return", f"{price_change:.1f}%")

        # Historical price chart
        st.subheader("Historical Price Data")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_train["ds"],
            y=df_train["y"],
            name="Price",
            line=dict(color='royalblue', width=1.5)
        ))

        if len(df_train) >= 30:
            ma30 = df_train['y'].rolling(window=30).mean()
            fig.add_trace(go.Scatter(
                x=df_train["ds"][29:],
                y=ma30[29:],
                name="30-Day MA",
                line=dict(color='orange', width=1.5, dash='dot')
            ))

        fig.update_layout(
            title=f"{selected_stock} Price History",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_white",
            xaxis_rangeslider_visible=True
        )
        st.plotly_chart(fig, use_container_width=True)

        # Forecast
        st.subheader("Price Forecast")
        with st.spinner("Generating forecast..."):
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            m.fit(df_train)
            future = m.make_future_dataframe(periods=period)
            forecast = m.predict(future)

        # Forecast results
        col1, col2 = st.columns(2)
        with col1:
            st.write("Latest Predictions")
            forecast_tail = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
            forecast_tail.columns = ['Date', 'Forecast', 'Lower', 'Upper']
            st.dataframe(forecast_tail)

        with col2:
            current_price = df_train['y'].iloc[-1]
            last_prediction = forecast['yhat'].iloc[-1]
            price_change = ((last_prediction - current_price) / current_price) * 100
            st.metric(
                "Predicted Change",
                f"{price_change:.1f}%",
                delta=f"${last_prediction - current_price:.2f}"
            )

        # Forecast plot
        fig1 = plot_plotly(m, forecast)
        fig1.update_layout(
            title=f"{selected_stock} Forecast - {n_years} Years",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_white"
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Components plot
        st.subheader("Forecast Components")
        fig2 = m.plot_components(forecast)
        st.write(fig2)

        # Disclaimer for forecast
        st.info("""
        📊 **Disclaimer**: This forecast is for educational purposes only. Stock markets are influenced by many factors
        not captured in this model. Always conduct thorough research and consult with financial advisors before making
        investment decisions.
        """)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.code(f"Error details: {str(e)}")

# Footer
st.markdown("""
---
Created with ❤️ using Streamlit, Facebook Prophet, and Technical Indicators
""")
