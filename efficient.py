import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="QuantPro Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .metric-card {
        background-color: #0E1117;
        border: 1px solid #303030;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 300;
    }
    .stDataFrame {
        border: 1px solid #303030;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

@st.cache_data
def get_currency_rate():
    try:
        data = yf.Ticker("INR=X").history(period="1d")
        if not data.empty:
            return data['Close'].iloc[-1]
        return 84.0 
    except:
        return 84.0 

def fetch_data(tickers, period="2y"):
    if not tickers:
        return pd.DataFrame()
    try:
        data = yf.download(tickers, period=period, auto_adjust=False, threads=True)
        if 'Adj Close' in data.columns:
            return data['Adj Close']
        elif 'Close' in data.columns:
            return data['Close']
        return data.iloc[:, 0] if not data.empty else pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# UPDATED: Included Risk-Free Rate of 6.5%
def calculate_portfolio_metrics(weights, mean_returns, cov_matrix, risk_free_rate=0.065):
    returns = np.sum(mean_returns * weights) * 252
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    # Correct Sharpe Formula: (Rp - Rf) / Sigma
    sharpe = (returns - risk_free_rate) / std_dev
    return returns, std_dev, sharpe

# --- Main App ---

def main():
    st.title("📈 QuantPro: The Efficient Frontier Engine")
    st.markdown("### *Institutional Grade Portfolio Analytics*")

    # --- Sidebar: File Upload & Macro Settings ---
    with st.sidebar:
        st.header("📂 Data Import")
        uploaded_file = st.file_uploader("Upload Portfolio (CSV, Excel, or JSON)", type=["csv", "xlsx", "json"])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.json'):
                    up_df = pd.read_json(uploaded_file)
                elif uploaded_file.name.endswith('.csv'):
                    up_df = pd.read_csv(uploaded_file)
                else:
                    up_df = pd.read_excel(uploaded_file)
                
                # Standardizing column names for the app
                required_cols = ["Ticker", "Shares", "Avg Cost"]
                if all(col in up_df.columns for col in required_cols):
                    st.session_state.portfolio_df = up_df[required_cols + (["Currency"] if "Currency" in up_df.columns else [])]
                    st.success("File Loaded!")
                else:
                    st.error("File must contain: Ticker, Shares, Avg Cost")
            except Exception as e:
                st.error(f"Error: {e}")

        st.divider()
        st.header("🌍 Macro Intelligence")
        country = st.selectbox("Select Your Region", ["India", "USA", "Global"])
        risk_free_rate = st.number_input("Risk Free Rate (Rf)", value=0.065, step=0.005, format="%.3f")
        
        st.divider()
        st.write("Current USD/INR Rate: ₹" + str(round(get_currency_rate(), 2)))

    # --- Section 1: Bulk Portfolio Input ---
    st.subheader("💼 Portfolio Management")

    if 'portfolio_df' not in st.session_state:
        data = {
            "Ticker": ["TATASTEEL.NS", "SBIN.NS", "VEDL.NS", "GOLDBEES.NS"],
            "Shares": [20.0, 10.0, 14.0, 155.0],
            "Avg Cost": [171.74, 881.58, 524.11, 58.0],
            "Currency": ["INR", "INR", "INR", "INR"]
        }
        st.session_state.portfolio_df = pd.DataFrame(data)

    edited_df = st.data_editor(
        st.session_state.portfolio_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker Symbol", validate="^[A-Za-z0-9\-\.]+$"),
            "Shares": st.column_config.NumberColumn("Shares Held", min_value=0.0, format="%.2f"),
            "Avg Cost": st.column_config.NumberColumn("Avg Buy Price", min_value=0.0, format="₹%.2f"),
            "Currency": st.column_config.SelectboxColumn("Currency", options=["INR", "USD"], required=True)
        },
        key="portfolio_editor"
    )

    portfolio_clean = edited_df[edited_df["Ticker"].str.len() > 1].copy()
    if portfolio_clean.empty:
        st.warning("⚠️ Please add stocks or upload a file.")
        st.stop()

    tickers_list = portfolio_clean['Ticker'].unique().tolist()
    usd_rate = get_currency_rate()
    
    def calculate_invested(row):
        val = row['Shares'] * row['Avg Cost']
        return val * usd_rate if row['Currency'] == 'USD' else val

    portfolio_clean['Invested_INR'] = portfolio_clean.apply(calculate_invested, axis=1)

    # --- Section 2: Data Fetching ---
    if len(tickers_list) > 0:
        market_data = fetch_data(tickers_list)
        if market_data.empty:
            st.error("❌ No data found. Check ticker formats (e.g., RELIANCE.NS).")
            st.stop()

        if isinstance(market_data, pd.Series):
            market_data = market_data.to_frame(name=tickers_list[0])

        current_prices = market_data.iloc[-1]
        current_vals = []
        for _, row in portfolio_clean.iterrows():
            ticker = row['Ticker']
            price = current_prices[ticker] if ticker in current_prices else 0
            val = price * row['Shares']
            val = val * usd_rate if row['Currency'] == 'USD' else val
            current_vals.append(val)
        
        portfolio_clean['Current_Value_INR'] = current_vals
        portfolio_clean['Weight'] = portfolio_clean['Current_Value_INR'] / portfolio_clean['Current_Value_INR'].sum()

        # --- Section 3: Performance Metrics ---
        st.divider()
        c1, c2, c3 = st.columns(3)
        total_inv = portfolio_clean['Invested_INR'].sum()
        total_curr = portfolio_clean['Current_Value_INR'].sum()
        pnl_val = total_curr - total_inv
        pnl_pct = (pnl_val / total_inv) * 100 if total_inv > 0 else 0

        c1.metric("Total Invested", f"₹{total_inv:,.0f}")
        c2.metric("Current Value", f"₹{total_curr:,.0f}")
        c3.metric("Net P/L", f"₹{pnl_val:,.0f}", f"{pnl_pct:.2f}%")

        # --- Section 4: Risk Analytics ---
        if len(tickers_list) > 1:
            st.subheader("🔬 Deep Risk Analysis")
            log_returns = np.log(market_data / market_data.shift(1)).dropna()
            log_returns = log_returns.loc[:, ~log_returns.columns.duplicated()]
            mean_returns = log_returns.mean()
            cov_matrix = log_returns.cov()

            # B. Efficient Frontier
            st.subheader("🚀 Efficient Frontier")
            col_graph, col_controls = st.columns([2, 1])

            with col_graph:
                num_portfolios = 3000
                results = np.zeros((3, num_portfolios))
                unique_tickers = log_returns.columns.tolist()
                n_assets = len(unique_tickers)

                for i in range(num_portfolios):
                    weights = np.random.random(n_assets)
                    weights /= np.sum(weights)
                    pret, pstd, psharpe = calculate_portfolio_metrics(weights, mean_returns, cov_matrix, risk_free_rate)
                    results[0,i] = pstd
                    results[1,i] = pret
                    results[2,i] = psharpe

                grouped_weights = portfolio_clean.groupby('Ticker')['Weight'].sum()
                current_w_array = np.array([grouped_weights.get(t, 0) for t in unique_tickers])
                cur_ret, cur_std, cur_sharpe = calculate_portfolio_metrics(current_w_array, mean_returns, cov_matrix, risk_free_rate)

                fig_ef = go.Figure()
                fig_ef.add_trace(go.Scatter(
                    x=results[0,:], y=results[1,:], mode='markers',
                    marker=dict(color=results[2,:], colorscale='Viridis', size=4, showscale=True, colorbar=dict(title=f"Sharpe (Rf={risk_free_rate*100}%)")),
                    name='Simulated'
                ))
                fig_ef.add_trace(go.Scatter(
                    x=[cur_std], y=[cur_ret], mode='markers+text',
                    marker=dict(color='red', size=15, symbol='star'),
                    text=["YOU"], textposition="top center", name='Current'
                ))
                fig_ef.update_layout(title="Risk vs Return Optimization", xaxis_title="Risk (Annual Vol)", yaxis_title="Return (Annualized)", template="plotly_dark", height=500)
                st.plotly_chart(fig_ef, use_container_width=True)

            with col_controls:
                st.markdown("### 🎛️ Optimization Metrics")
                st.metric("Portfolio Sharpe Ratio", f"{cur_sharpe:.2f}")
                st.metric("Annualized Volatility", f"{cur_std*100:.2f}%")
                st.info(f"The Sharpe ratio is now correctly calculated by subtracting the {risk_free_rate*100}% Risk-Free Rate from your returns.")

            # C. Tornado Sensitivity
            st.subheader("🌪️ Macro Sensitivity")
            implied_beta = cur_std / 0.15 
            scenarios = {"Market Crash (-10%)": -0.10 * implied_beta, "Bull Run (+10%)": 0.10 * implied_beta, "Rate Hike": -0.05 * (implied_beta * 1.2)}
            impacts = [total_curr * move for move in scenarios.values()]
            fig_tor = go.Figure(go.Bar(x=impacts, y=list(scenarios.keys()), orientation='h', marker_color=['red' if x < 0 else 'green' for x in impacts], text=[f"₹{x:,.0f}" for x in impacts], textposition="auto"))
            fig_tor.update_layout(template="plotly_dark")
            st.plotly_chart(fig_tor, use_container_width=True)

    st.divider()
    json_data = portfolio_clean.to_json(orient="records")
    st.download_button("💾 Save Portfolio", json_data, "my_portfolio.json", "application/json")

if __name__ == "__main__":
    main()
