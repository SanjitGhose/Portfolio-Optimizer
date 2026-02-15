import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="QuantPro: Institutional Analytics",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- High-Density CSS ---
st.markdown("""
<style>
    .stApp { background-color: #000000; color: #FFFFFF; }
    
    /* Metrics Styling */
    div[data-testid="stMetric"] {
        background-color: #111;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 8px;
    }

    /* SCROLLABLE MATRIX CONTAINER */
    .matrix-scroll-wrapper {
        overflow-x: auto;
        overflow-y: auto;
        max-height: 800px;
        border: 1px solid #333;
        padding: 5px;
        background-color: #000;
    }
    
    /* Ensure the Plotly iframe doesn't get squashed */
    .element-container iframe { width: 100% !important; }
</style>
""", unsafe_allow_html=True)

# --- Core Logic Functions ---

@st.cache_data(ttl=3600)
def get_currency_rate():
    try:
        data = yf.Ticker("INR=X").history(period="1d")
        return data['Close'].iloc[-1] if not data.empty else 84.0
    except: return 84.0 

def fetch_data(tickers, period="2y"):
    if not tickers: return pd.DataFrame()
    try:
        data = yf.download(tickers, period=period, auto_adjust=False, threads=True)
        if 'Adj Close' in data.columns: return data['Adj Close']
        elif 'Close' in data.columns: return data['Close']
        return data.iloc[:, 0] if not data.empty else pd.DataFrame()
    except: return pd.DataFrame()

def calculate_portfolio_metrics(weights, mean_returns, cov_matrix, rf_rate):
    returns = np.sum(mean_returns * weights) * 252
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe = (returns - rf_rate) / std_dev if std_dev != 0 else 0
    return returns, std_dev, sharpe

# --- Main Application ---

def main():
    st.title("💎 QuantPro Terminal")
    st.caption("Risk Intelligence & Multi-Asset Optimization Engine")

    # --- Sidebar: Upload & Config ---
    with st.sidebar:
        st.header("⚙️ System Config")
        rf_rate = st.number_input("Bond Rate (Risk-Free %)", value=0.065, step=0.005, format="%.3f")
        
        st.divider()
        st.subheader("📂 Import Portfolio")
        uploaded_file = st.file_uploader("CSV, Excel, or JSON", type=["csv", "xlsx", "json"])
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'): df_up = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.json'): df_up = pd.read_json(uploaded_file)
                else: df_up = pd.read_excel(uploaded_file)
                st.session_state.portfolio_df = df_up
                st.success("Loaded Portfolio!")
            except: st.error("Format Error")

        st.divider()
        usd_rate = get_currency_rate()
        st.metric("USD/INR", f"₹{usd_rate:.2f}")

    # --- Portfolio Initialization ---
    if 'portfolio_df' not in st.session_state:
        data = {
            "Ticker": ["OLAELEC.NS", "BAJAJHFL.NS", "CESC.NS", "KOTAKIT.NS", "TATASILV.NS", "KALYANKJIL.NS", "ITC.NS", "CASTROLIND.NS", "GAIL.NS", "REDINGTON.NS", "ADANIPOWER.NS", "TMPV.NS", "GROWW.NS", "BSLNIFTY.NS", "PHARMABEES.NS", "GROWWMETAL.NS", "TATAGOLD.NS", "TATASTEEL.NS", "VEDL.NS", "SBIN.NS"],
            "Shares": [31, 20, 20, 70, 123, 10, 7, 20, 20, 10, 10, 10, 12, 72, 100, 195, 155, 20, 14, 10],
            "Avg Cost": [37.86, 109.5, 176.18, 40.19, 27.4, 473.05, 351.99, 204.65, 177.22, 273.55, 152.04, 391.37, 175.32, 29.49, 22.38, 10.74, 13.53, 171.74, 524.11, 881.58],
            "Currency": ["INR"] * 20
        }
        st.session_state.portfolio_df = pd.DataFrame(data)

    # --- Section 1: Dynamic Editor ---
    with st.expander("💼 Portfolio Data Editor", expanded=False):
        edited_df = st.data_editor(st.session_state.portfolio_df, num_rows="dynamic", use_container_width=True)

    tickers = edited_df['Ticker'].unique().tolist()
    
    with st.spinner("🚀 Market Data Sync..."):
        market_data = fetch_data(tickers)
        if market_data.empty: st.stop()

    # --- Math Processing ---
    current_prices = market_data.iloc[-1]
    def get_val(row):
        p = current_prices.get(row['Ticker'], 0)
        v = p * row['Shares']
        return v * usd_rate if row['Currency'] == "USD" else v

    edited_df['Current_Value'] = edited_df.apply(get_val, axis=1)
    total_val = edited_df['Current_Value'].sum()
    total_inv = (edited_df['Shares'] * edited_df['Avg Cost']).sum()
    pnl_pct = ((total_val - total_inv) / total_inv) * 100

    log_returns = np.log(market_data / market_data.shift(1)).dropna()
    mean_ret = log_returns.mean()
    cov_mat = log_returns.cov()
    
    weights = np.array([edited_df[edited_df['Ticker']==t]['Current_Value'].sum()/total_val for t in market_data.columns])
    ann_ret, ann_std, ann_sharpe = calculate_portfolio_metrics(weights, mean_ret, cov_mat, rf_rate)

    # --- Section 2: Metrics & Commentary ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Value", f"₹{total_val:,.0f}")
    c2.metric("Net P&L", f"{pnl_pct:.2f}%")
    c3.metric("Sharpe Ratio", f"{ann_sharpe:.2f}")
    c4.metric("Volatility", f"{ann_std*100:.1f}%")

    st.subheader("💡 Intelligence Dashboard")
    if pnl_pct > 0:
        st.success(f"**Doing Good!** 🚀 Portfolio is outperforming the bond rate by {pnl_pct:.1f}%.")
    else:
        st.warning("**Under Pressure.** Portfolio is in a drawdown. Correlation analysis required.")

    # --- Section 3: High Density Tabs ---
    tab1, tab2, tab3 = st.tabs(["🚀 Frontier", "🔬 Legible Matrix", "🌪️ Stress Test"])

    with tab1:
        st.subheader("Efficient Frontier")
        num_sim = 1500
        results = np.zeros((3, num_sim))
        for i in range(num_sim):
            w = np.random.random(len(tickers)); w /= np.sum(w)
            r, s, sh = calculate_portfolio_metrics(w, mean_ret, cov_mat, rf_rate)
            results[0,i], results[1,i], results[2,i] = s, r, sh
        
        fig_ef = px.scatter(x=results[0,:], y=results[1,:], color=results[2,:], 
                            labels={'x':'Risk','y':'Return','color':'Sharpe'}, template="plotly_dark")
        fig_ef.add_trace(go.Scatter(x=[ann_std], y=[ann_ret], mode='markers+text', text=["YOUR PORTFOLIO"], 
                                    marker=dict(color='white', size=15, symbol='star')))
        st.plotly_chart(fig_ef, use_container_width=True)
        

    with tab2:
        st.subheader("🔬 Institutional Correlation Matrix")
        st.caption("Every ticker visible. Swipe horizontally or vertically to navigate the full grid.")
        
        corr_matrix = log_returns.corr()
        num_assets = len(tickers)
        
        # Matrix Optimization: 11px font + 1800px wide + Auto-scroll
        fig_corr = px.imshow(
            corr_matrix, text_auto=".2f", aspect="auto", 
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            width=max(1200, num_assets * 55), # Dynamic width based on asset count
            height=max(800, num_assets * 55)   # Dynamic height
        )
        
        fig_corr.update_layout(
            font=dict(size=11, color="white"),
            margin=dict(l=150, r=50, t=100, b=150),
            template="plotly_dark",
            coloraxis_showscale=False
        )
        fig_corr.update_xaxes(side="top", tickangle=-45)
        
        # Legibility Wrapper
        st.markdown('<div class="matrix-scroll-wrapper">', unsafe_allow_html=True)
        st.plotly_chart(fig_corr, use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)
        

    with tab3:
        st.subheader("🌪️ Risk Sensitivity (AI Boom Update)")
        col_in, col_gr = st.columns([1, 2])
        with col_in:
            ai_boom = st.slider("🤖 Tech AI Boom %", 0, 100, 40)
            crash = st.slider("📉 Market Crash %", -50, 0, -20)
        
        with col_gr:
            beta = ann_std / 0.15 # Approx beta to Nifty/Market
            scenarios = {
                f"Market Crash ({crash}%)": (crash/100) * beta,
                f"AI Boom (+{ai_boom}%)": (ai_boom/100) * (beta * 1.6)
            }
            impacts = [total_val * v for v in scenarios.values()]
            fig_tor = go.Figure(go.Bar(
                x=impacts, y=list(scenarios.keys()), orientation='h',
                marker_color=['#FF4B4B' if x < 0 else '#00CC96' for x in impacts],
                text=[f"₹{x:,.0f}" for x in impacts], textposition="auto"
            ))
            fig_tor.update_layout(template="plotly_dark", height=400, xaxis_title="P&L Impact (INR)")
            st.plotly_chart(fig_tor, use_container_width=True)
            

if __name__ == "__main__":
    main()
