import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- 1. PAGE SETUP & PROFESSIONAL STYLING ---
st.set_page_config(page_title="QuantPro Elite", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    div[data-testid="stMetric"] {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #4F8BF9;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #1E1E1E;
        border-radius: 5px;
        padding: 10px 20px;
        color: white;
    }
    .stTabs [aria-selected="true"] { background-color: #4F8BF9; }
</style>
""", unsafe_allow_html=True)

# --- 2. DATA PROCESSING ENGINE ---
@st.cache_data(ttl=3600)
def fetch_market_data(tickers):
    try:
        # Fetch data
        df = yf.download(tickers, period="2y")['Adj Close']
        # If only one ticker is fetched, it returns a Series. Convert to DataFrame.
        if isinstance(df, pd.Series):
            df = df.to_frame()
        # Drop columns that are entirely NaN
        df = df.dropna(axis=1, how='all')
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def calculate_metrics(weights, returns_df, rf):
    # Ensure weights match the number of columns in returns_df
    ann_ret = np.sum(returns_df.mean() * weights) * 252
    ann_risk = np.sqrt(np.dot(weights.T, np.dot(returns_df.cov() * 252, weights)))
    sharpe = (ann_ret - rf) / ann_risk
    return ann_ret, ann_risk, sharpe

# --- 3. MAIN APPLICATION ---
def main():
    st.title("💎 QuantPro: Institutional Portfolio Intelligence")
    st.caption("Real-time Risk Analytics & Optimization Engine")

    with st.sidebar:
        st.header("⚙️ Macro Inputs")
        rf_rate = st.number_input("Risk Free Rate (Rf)", value=0.065, step=0.005, format="%.3f")
        st.info(f"Hurdle Rate: {rf_rate*100:.1f}% (India 10Y Bond)")

    # Your Specific Ticker List
    tickers = [
        "OLAELEC.NS", "BAJAJHFL.NS", "CESC.NS", "KOTAKIT.NS", "TATSILV.NS", 
        "KALYANKJIL.NS", "ITC.NS", "CASTROLIND.NS", "GAIL.NS", "REDINGTON.NS", 
        "ADANIPOWER.NS", "TMPV.NS", "GROWW.NS", "BSLNIFTY.NS", "PHARMABEES.NS", 
        "GROWWMETAL.NS", "TATAGOLD.NS", "TATASTEEL.NS", "VEDL.NS", "SBIN.NS"
    ]

    with st.spinner("Analyzing Market Trends..."):
        data = fetch_market_data(tickers)
        
        if data.empty:
            st.error("No data found. Check your internet or tickers.")
            return
        
        # Log returns
        returns = np.log(data / data.shift(1)).dropna()
        
        # --- CRITICAL FIX: ALIGN WEIGHTS WITH DOWNLOADED DATA ---
        active_tickers = returns.columns.tolist()
        num_active = len(active_tickers)
        
        if num_active < len(tickers):
            missing = list(set(tickers) - set(active_tickers))
            st.warning(f"Note: Data unavailable for {len(missing)} tickers: {', '.join(missing)}")
        
        # Create weights based ONLY on active tickers
        weights = np.array([1/num_active] * num_active)

    # Calculate Portfolio Stats
    ann_ret, ann_risk, sharpe = calculate_metrics(weights, returns, rf_rate)

    # --- 4. TOP SUMMARY METRICS ---
    m1, m2, m3 = st.columns(3)
    m1.metric("Annualized Return", f"{ann_ret*100:.2f}%")
    m2.metric("Annualized Risk (Vol)", f"{ann_risk*100:.2f}%")
    m3.metric("True Sharpe Ratio", f"{sharpe:.2f}")

    st.divider()

    # --- 5. THE THREE ANALYTICS TABS ---
    tab1, tab2, tab3 = st.tabs(["🚀 Efficient Frontier", "🔬 Covariance Matrix", "🌪️ Stress Test"])

    with tab1:
        st.subheader("Efficient Frontier Optimization")
        col_left, col_right = st.columns([3, 1])
        
        with col_left:
            num_portfolios = 1000
            results = np.zeros((3, num_portfolios))
            for i in range(num_portfolios):
                w = np.random.random(num_active)
                w /= np.sum(w)
                r, s, sh = calculate_metrics(w, returns, rf_rate)
                results[0,i], results[1,i], results[2,i] = s, r, sh
            
            fig_ef = px.scatter(
                x=results[0,:], y=results[1,:], color=results[2,:],
                labels={'x': 'Annualized Risk', 'y': 'Annualized Return', 'color': 'Sharpe Ratio'},
                template="plotly_dark", color_continuous_scale='Viridis'
            )
            fig_ef.add_trace(go.Scatter(x=[ann_risk], y=[ann_ret], mode='markers', 
                                        marker=dict(color='white', size=15, symbol='star'), name="Current"))
            st.plotly_chart(fig_ef, use_container_width=True)
            

        with col_right:
            st.markdown("#### **The Sharpe Formula**")
            st.latex(r"Sharpe = \frac{R_p - R_f}{\sigma_p}")
            st.write(f"- **Rp**: {ann_ret*100:.1f}%\n- **Rf**: {rf_rate*100:.1f}%\n- **σp**: {ann_risk*100:.1f}%")

    with tab2:
        st.subheader("🔬 Asset Correlation Matrix")
        corr_matrix = returns.corr()
        dynamic_height = max(500, num_active * 35) 
        fig_corr = px.imshow(
            corr_matrix, text_auto=".2f", aspect="auto", 
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1, height=dynamic_height 
        )
        fig_corr.update_layout(margin=dict(l=100, r=20, t=50, b=100), font=dict(size=10), template="plotly_dark")
        fig_corr.update_xaxes(side="bottom", tickangle=-45)
        st.plotly_chart(fig_corr, use_container_width=True)
        

    with tab3:
        st.subheader("🌪️ Interactive Stress Testing")
        c1, c2 = st.columns([1, 2])
        with c1:
            crash_val = st.slider("📉 Market Crash %", -50, 0, -20)
            bull_val = st.slider("📈 Bull Run %", 0, 50, 20)
            rate_impact = st.slider("🏦 Rate Hike Impact %", -20, 0, -5)

        with c2:
            scenarios = {"Market Crash": crash_val, "Bull Run": bull_val, "Rate Hike": rate_impact}
            sensitivity = ann_risk / 0.15
            impacts = [val * sensitivity for val in scenarios.values()] 
            fig_tor = go.Figure(go.Bar(
                x=impacts, y=list(scenarios.keys()), orientation='h',
                marker_color=['#FF4B4B' if x < 0 else '#00CC96' for x in impacts],
                text=[f"{x:.1f}% Portfolio Change" for x in impacts], textposition="auto"
            ))
            fig_tor.update_layout(template="plotly_dark", height=400, xaxis_title="P&L (%)")
            st.plotly_chart(fig_tor, use_container_width=True)
            

if __name__ == "__main__":
    main()
