import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Function to fetch stock data
def get_stock_data(ticker, period="1mo", interval="1d"):
    stock = yf.Ticker(ticker)
    real_time_data = stock.history(period=period, interval=interval)

    if not real_time_data.empty:
        last_price = real_time_data['Close'].iloc[-1]  # Get last closing price
        opening_price = real_time_data['Open'].iloc[0]  # Get first opening price
        return real_time_data, last_price, opening_price
    else:
        return None, None, None

# Streamlit UI
st.set_page_config(page_title="Stock Market Dashboard", layout="wide")
st.title("📈 Real-Time Stock Market Dashboard")

# User input for stock ticker
ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, TSLA, GOOG):", "AAPL")

if st.button("Get Stock Data"):
    data, last_price, opening_price = get_stock_data(ticker)

    if data is not None:
        st.markdown(f"### **Stock Data for {ticker}**")
        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="📌 Last Price", value=f"${last_price:.2f}")
        with col2:
            st.metric(label="📌 Opening Price", value=f"${opening_price:.2f}")

        # Line Chart
        st.subheader("📊 Stock Price Trend")
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
        fig_line.update_layout(title=f"Stock Price Trend for {ticker}", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_line, use_container_width=True)

        # Candlestick Chart
        st.subheader("🕯️ Candlestick Chart")
        fig_candle = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Candlestick"
        )])
        fig_candle.update_layout(title=f"Candlestick Chart for {ticker}", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_candle, use_container_width=True)

        # Show Raw Data
        with st.expander("📜 View Raw Data"):
            st.dataframe(data)

    else:
        st.error("Stock data could not be retrieved. Please check the ticker symbol.")

