# Stock Analysis & Forecast App

This project is a **Stock Analysis & Forecast App** built using **Streamlit**, **yfinance**, and **Facebook Prophet**. It allows users to analyze stock trends, apply technical indicators, and forecast future stock prices.

## Features
- 📈 **Stock Analysis**: Visualize stock trends using RSI, Bollinger Bands, MACD, and Moving Averages.
- 🔮 **Stock Forecasting**: Predict future stock prices using Facebook Prophet.
- 🛠️ **Customizable Settings**: Choose different analysis periods and forecast years.
- 📊 **Interactive Visuals**: Display dynamic plots using Plotly and Matplotlib.

## Installation
To run this project locally:
1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd stock-analysis-app
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run stcokfinal.py
   ```

## Dependencies
- **Python 3.8+**
- **Streamlit**
- **yfinance**
- **Prophet** (for forecasting)
- **Plotly**
- **TA-Lib** (for technical analysis)
- **Matplotlib**
- **Pandas**
- **NumPy**

## Usage
1. Select a stock from the sidebar (e.g., Google, Apple, Microsoft).
2. Choose an **Analysis Mode**:
   - **Forecast**: Predict future stock prices.
   - **Technical Analysis**: Analyze stock trends with RSI, MACD, and more.
   - **Both**: Combine both forecasts and technical analysis.
3. Adjust analysis settings like time period and forecast years.
4. View the results with interactive visualizations and data insights.

## Technical Analysis Indicators
- **RSI (Relative Strength Index)** for overbought/oversold signals.
- **Bollinger Bands** to track volatility and potential price reversals.
- **MACD (Moving Average Convergence Divergence)** for trend direction.
- **Moving Averages** (50-day and 200-day) for identifying golden crosses and death crosses.

## Forecasting Model
- The app uses **Facebook Prophet** for forecasting.
- The model considers yearly and weekly seasonality with changepoint adjustments for better accuracy.

## Disclaimer
This app is for **educational purposes only**. The predictions and technical analysis are based on historical data and may not reflect real-world outcomes. Please consult financial experts before making investment decisions.

## Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests.

---
Created with ❤️ using Streamlit, Facebook Prophet, and Technical Indicators.

