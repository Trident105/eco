import streamlit as st
import yfinance as yf
import numpy as np
import numpy_financial as npf
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

# **Streamlit Web App 标题**
st.title("📈 股票 & ETF 投资分析工具")
st.write("请输入股票代码，进行 IRR、RMSE、Sharpe Ratio 计算")

# **用户输入股票代码**
stock_codes = st.text_input("请输入股票代码（台股：2330, 00646 / 美股：AAPL, TSLA, SPY）：").upper()

# **检查用户是否输入数据**
if stock_codes:
    stock_list = [s.strip() for s in stock_codes.split(',')]  # **处理多个股票代码**
    results = []

    for stock_code in stock_list:
        # **判断台股 or 美股**
        ticker = f"{stock_code}.TW" if stock_code.isdigit() else stock_code
        stock = yf.Ticker(ticker)
        hist = stock.history(period="max")

        # **如果数据为空，显示警告**
        if hist.empty:
            st.warning(f"⚠ 无法获取 {stock_code} 的数据，请检查代码是否正确。")
            continue

        # **获取股息数据**
        dividends = stock.dividends

        # **计算年化回报率**
        initial_price = hist['Close'].iloc[0]
        final_price = hist['Close'].iloc[-1]
        total_dividends = dividends.sum() if not dividends.empty else 0
        years = (hist.index[-1] - hist.index[0]).days / 365.25

        total_return = (final_price + total_dividends - initial_price) / initial_price
        annualized_return = (1 + total_return) ** (1 / years) - 1

        # **计算波动率 & Sharpe Ratio**
        daily_returns = hist['Close'].pct_change().dropna()
        annual_volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.02) / annual_volatility if annual_volatility > 0 else np.nan

        # **计算最大回撤**
        cumulative_returns = (1 + daily_returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()

        # **计算 20 年后的资产**
        expected_monthly_return = (1 + annualized_return) ** (1/12) - 1
        total_value = 0
        total_assets = []

        for i in range(20 * 12):
            total_value = (total_value + 15000) * (1 + expected_monthly_return)
            total_assets.append(total_value)

        # **计算 IRR**
        cash_flows = [-15000] * (20 * 12)
        cash_flows.append(total_value)
        computed_irr = npf.irr(cash_flows) * 12

        # **计算 RMSE**
        theoretical_values = [15000 * ((1 + expected_monthly_return) ** i - 1) / expected_monthly_return for i in range(20 * 12 + 1)]
        mse = mean_squared_error(cash_flows[:-1], theoretical_values[:-1])
        rmse = np.sqrt(mse)

        # **风险评级**
        risk_level = "Low Risk" if rmse <= 3_000_000 else "Medium Risk" if rmse <= 5_000_000 else "High Risk"

        # **存储数据**
        results.append([stock_code, computed_irr, total_value / 1e4, rmse / 1e4, sharpe_ratio, max_drawdown, risk_level, total_assets])

    # **展示结果**
    if results:
        df = pd.DataFrame(results, columns=["Stock/ETF", "Annualized IRR", "Total Assets (10K TWD)", "RMSE (10K TWD)", "Sharpe Ratio", "Max Drawdown", "Risk Level", "Total Assets Curve"])
        st.write("📊 **Investment Results:**")
        st.dataframe(df.drop(columns=["Total Assets Curve"]))

        # **绘制资产累积曲线**
        plt.figure(figsize=(10, 5))
        for i, row in df.iterrows():
            plt.plot(range(1, 20 * 12 + 1), np.array(row["Total Assets Curve"]) / 1e4, label=row["Stock/ETF"])
        plt.xlabel("Investment Months")
        plt.ylabel("Total Accumulated Assets (10K TWD)")
        plt.title("20-Year Asset Growth Trend")
        plt.legend()
        plt.grid()
        st.pyplot(plt)
