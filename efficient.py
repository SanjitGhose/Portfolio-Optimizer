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

# --- Custom CSS for Institutional Look ---
st.markdown("""
<style>
    .stApp { background-color: #000000; color: #FFFFFF; }
    .stDataFrame { border: 1px solid #303030; border-radius: 5px; }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; font-weight: 300; }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

@st.cache_data
def get_currency_rate():
    try:
        data = yf.Ticker("INR=X").history(period="1d")
        return data['Close'].iloc[-1] if not data.empty else 84.0
    except: return 84.0 

def fetch_data(tickers, period="2y"):
    if not tickers: return pd.DataFrame()
    try:
        data = yf.download(tickers, period=period, auto_adjust=False, threads=True)
        return data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    except: return pd.DataFrame()

def calculate_portfolio_metrics(weights, mean_returns, cov_matrix, risk_free_rate=0.065):
    returns = np.sum(mean_returns * weights) * 252
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    # Corrected Sharpe Ratio: Subtracting the 6.5% Hurdle
    sharpe = (returns - risk_free_rate) / std_dev
    return returns, std_dev, sharpe

# --- Main App Logic ---

def main():
    st.title("📈 QuantPro: The Efficient Frontier Engine")
    st.markdown("### *Institutional Grade Portfolio Analytics*")

    # --- Sidebar: Document Import & Macro Settings ---
    with st.sidebar:
        st.header("📂 Data Import")
        uploaded_file = st.file_uploader("Upload Portfolio (CSV, XLSX, JSON)", type=["csv", "xlsx", "json"])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.json'):
                    up_df = pd.read_json(uploaded_file)
                elif uploaded_file.name.endswith('.csv'):
                    up_df = pd.read_csv(uploaded_file)
                else:
                    up_df = pd.read_excel(uploaded_file)
                
                # Validation: Ensuring basic columns exist
                if "Ticker" in up_df.columns and "Shares" in up_df.columns:
                    st.session_state.portfolio_df = up_df
                    st.success("Portfolio Uploaded!")
                else:
                    st.error("Missing 'Ticker' or 'Shares' columns.")
            except Exception as e:
                st.error(f"Upload Error: {e}")

        st.divider()
        st.header("🌍 Risk Settings")
        # Defaulting to your specified 6.5% Risk-Free Rate
        risk_free_rate = st.number_input("Risk Free Rate (Rf)", value=0.065, step=0.005, format="%.3f")
        st.info(f"Using {risk_free_rate*100}% Hurdle Rate")
        
        usd_rate = get_currency_rate()
        st.write(f"Live USD/INR: ₹{round(usd_rate, 2)}")

    # --- Section 1: Portfolio Input ---
    st.subheader("💼 Portfolio Composition")

    if 'portfolio_df' not in st.session_state:
        st.session_state.portfolio_df = pd.DataFrame({
            "Ticker": ["TATASTEEL.NS", "SBIN.NS", "VEDL.NS", "GOLDBEES.NS"],
            "Shares": [20.0, 10.0, 14.0, 155.0],
            "Avg Cost": [171.74, 881.58, 524.11, 58.0],
            "Currency": ["INR", "INR", "INR", "INR"]
        })

    edited_df = st.data_editor(st.session_state.portfolio_df, num_rows="dynamic", use_container_width=True)
    portfolio_clean = edited_df[edited_df["Ticker"].str.len() > 1].copy()

    if not portfolio_clean.empty:
        tickers = portfolio_clean['Ticker'].unique().tolist()
        market_data = fetch_data(tickers)
        
        if not market_data.empty:
            # Re-calculate Market Weights
            current_prices = market_data.iloc[-1]
            vals = []
            for _, row in portfolio_clean.iterrows():
                price = current_prices[row['Ticker']] if row['Ticker'] in current_prices else 0
                v = price * row['Shares']
                vals.append(v * usd_rate if row['Currency'] == 'USD' else v)
            
            portfolio_clean['Current_Value'] = vals
            portfolio_clean['Weight'] = portfolio_clean['Current_Value'] / portfolio_clean['Current_Value'].sum()

            # PnL Summary
            total_inv = (portfolio_clean['Shares'] * portfolio_clean['Avg Cost']).sum()
            total_curr = portfolio_clean['Current_Value'].sum()
            
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Invested Capital", f"₹{total_inv:,.0f}")
            m2.metric("Portfolio Value", f"₹{total_curr:,.0f}")
            m3.metric("Absolute P/L", f"₹{total_curr-total_inv:,.0f}", f"{((total_curr-total_inv)/total_inv)*100:.2f}%")

            # --- Section 2: Correlation Heatmap ---
            st.subheader("🔬 Risk Engine: Correlation Matrix")
            log_returns = np.log(market_data / market_data.shift(1)).dropna()
            corr_matrix = log_returns.corr()
            
            fig_corr = px.imshow(
                corr_matrix, text_auto=".2f", aspect="auto",
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                title="Asset Correlation (Interdependency Map)"
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            # --- Section 3: Efficient Frontier ---
            st.subheader("🚀 Efficient Frontier Optimization")
            col_graph, col_stats = st.columns([2, 1])

            mean_returns = log_returns.mean()
            cov_matrix = log_returns.cov()
            
            num_portfolios = 3000
            n_assets = len(tickers)
            results = np.zeros((3, num_portfolios))

            for i in range(num_portfolios):
                w = np.random.random(n_assets)
                w /= np.sum(w)
                pret, pstd, psharpe = calculate_portfolio_metrics(w, mean_returns, cov_matrix, risk_free_rate)
                results[0,i], results[1,i], results[2,i] = pstd, pret, psharpe

            # Current Portfolio Position
            curr_w = np.array([portfolio_clean.groupby('Ticker')['Weight'].sum()[t] for t in tickers])
            c_ret, c_std, c_sharpe = calculate_portfolio_metrics(curr_w, mean_returns, cov_matrix, risk_free_rate)

            with col_graph:
                fig_ef = px.scatter(
                    x=results[0,:], y=results[1,:], color=results[2,:],
                    labels={'x': 'Risk (Annual Vol)', 'y': 'Return (Annualized)', 'color': 'Sharpe'},
                    color_continuous_scale='Viridis', template="plotly_dark"
                )
                fig_ef.add_trace(go.Scatter(
                    x=[c_std], y=[c_ret], mode='markers+text',
                    marker=dict(color='red', size=15, symbol='star'),
                    text=["YOUR PORTFOLIO"], textposition="top center", name="Current"
                ))
                st.plotly_chart(fig_ef, use_container_width=True)

            with col_stats:
                st.markdown("#### Portfolio Health")
                st.metric("Risk-Adjusted Sharpe", f"{c_sharpe:.2f}")
                st.metric("Projected Annual Volatility", f"{c_std*100:.2f}%")
                st.metric("Projected Annual Return", f"{c_ret*100:.2f}%")
                
                st.write("---")
                st.caption(f"Note: Sharpe ratio accounts for a {risk_free_rate*100}% risk-free hurdle.")

if __name__ == "__main__":
    main()
                    
