import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- Page Setup ---
st.set_page_config(
    page_title="QuantPro: Institutional Analytics",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced Styling ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    div[data-testid="stMetric"] {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #1E1E1E; border-radius: 4px; color: #FFF; }
    .stTabs [aria-selected="true"] { background-color: #4F8BF9; color: #FFF; }
    /* Make Plotly containers more expansive */
    .js-plotly-plot { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# --- Core Logic Functions ---
@st.cache_data(ttl=3600)
def fetch_data(tickers, period="2y"):
    try:
        data = yf.download(tickers, period=period, auto_adjust=False, threads=True)
        if isinstance(data.columns, pd.MultiIndex):
            return data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
        return data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    except: return pd.DataFrame()

def calculate_portfolio_metrics(weights, mean_returns, cov_matrix, rf):
    returns = np.sum(mean_returns * weights) * 252
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe = (returns - rf) / std_dev
    return returns, std_dev, sharpe

# --- Main App ---
def main():
    st.title("💎 QuantPro: Portfolio Intelligence")
    
    with st.sidebar:
        st.header("⚙️ Settings")
        rf_rate = st.number_input("Risk Free Rate", value=0.065, format="%.3f")
        # Load default Tickers
        tickers = ["OLAELEC.NS", "BAJAJHFL.NS", "CESC.NS", "IT.NS", "TATSILV.NS", "KALYANKJIL.NS", "ITC.NS", "CASTROLIND.NS", "GAIL.NS", "REDINGTON.NS", "ADANIPOWER.NS", "TMPV.NS", "GROWW.NS", "BSLNIFTY.NS", "PHARMABEES.NS", "GROWWMETAL.NS", "TATAGOLD.NS", "TATASTEEL.NS", "VEDL.NS", "SBIN.NS"]

    with st.spinner("🚀 Analyzing Market Data..."):
        market_data = fetch_data(tickers)
        if market_data.empty: st.stop()
        
        log_returns = np.log(market_data / market_data.shift(1)).dropna()
        mean_returns = log_returns.mean()
        cov_matrix = log_returns.cov()
        
        # Default weighting (equal)
        num_assets = len(market_data.columns)
        weights = np.array([1/num_assets] * num_assets)
        curr_ret, curr_std, curr_sharpe = calculate_portfolio_metrics(weights, mean_returns, cov_matrix, rf_rate)

    tab1, tab2, tab3 = st.tabs(["🚀 Efficient Frontier", "🔬 Correlation Matrix", "🌪️ Stress Test"])

    # --- TAB 1: FRONTIER WITH GLOWING STAR ---
    with tab1:
        st.subheader("Frontier Optimization")
        num_portfolios = 2000
        results = np.zeros((3, num_portfolios))
        for i in range(num_portfolios):
            w = np.random.random(num_assets); w /= np.sum(w)
            r, s, sh = calculate_portfolio_metrics(w, mean_returns, cov_matrix, rf_rate)
            results[0,i], results[1,i], results[2,i] = s, r, sh
        
        fig_ef = px.scatter(x=results[0,:], y=results[1,:], color=results[2,:],
                            labels={'x': 'Risk', 'y': 'Return', 'color': 'Sharpe'},
                            template="plotly_dark", color_continuous_scale='Plasma')
        
        # THE FIX: High-Visibility Star with Glow Effect
        fig_ef.add_trace(go.Scatter(
            x=[curr_std], y=[curr_ret],
            mode='markers+text',
            text=["⭐ CURRENT PORTFOLIO"],
            textposition="top center",
            marker=dict(color='#00FFCC', size=25, symbol='star', 
                        line=dict(color='white', width=2)),
            name="You are here"
        ))
        st.plotly_chart(fig_ef, use_container_width=True)
        

    # --- TAB 2: FULL-SCREEN LEGIBLE MATRIX ---
    with tab2:
        st.subheader("🔬 Asset Correlation Matrix")
        st.info("💡 Pro Tip: Click the 'Expand' arrows on the top right of the chart to cover the whole screen.")
        
        corr_matrix = log_returns.corr()
        
        # CRITICAL LEGIBILITY: Height scales to number of tickers (approx 45px per ticker)
        matrix_height = max(850, num_assets * 45)
        
        fig_corr = px.imshow(
            corr_matrix, 
            text_auto=".2f", 
            aspect="auto", 
            color_continuous_scale="RdBu_r", 
            zmin=-1, zmax=1,
            height=matrix_height
        )
        
        fig_corr.update_layout(
            font=dict(size=14, color="white"), # Larger font for readability
            margin=dict(l=150, r=50, t=100, b=150),
            xaxis_nticks=num_assets,
            yaxis_nticks=num_assets,
            template="plotly_dark"
        )
        
        fig_corr.update_xaxes(tickangle=-45, tickfont=dict(size=13))
        fig_corr.update_yaxes(tickfont=dict(size=13))
        
        # Display the chart - it will now be much taller, requiring the user to scroll or expand
        st.plotly_chart(fig_corr, use_container_width=True)
        

    # --- TAB 3: TORNADO ---
    with tab3:
        st.subheader("🌪️ Risk Tornado")
        c1, c2 = st.columns([1, 2])
        with c1:
            crash = st.slider("Crash Scenario %", -50, 0, -20)
            bull = st.slider("Bull Scenario %", 0, 50, 20)
        
        with c2:
            sens = curr_std / 0.15
            scenarios = {"Crash": crash * sens, "Bull": bull * sens}
            fig_tor = go.Figure(go.Bar(
                x=list(scenarios.values()), y=list(scenarios.keys()), 
                orientation='h', marker_color=['#FF4B4B' if x < 0 else '#00CC96' for x in scenarios.values()],
                text=[f"{x:.1f}%" for x in scenarios.values()], textposition="auto"
            ))
            fig_tor.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_tor, use_container_width=True)
            

if __name__ == "__main__":
    main()
    
