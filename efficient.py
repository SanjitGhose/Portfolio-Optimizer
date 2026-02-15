import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="QuantPro: Institutional Analytics",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    div[data-testid="stMetric"] {
        background-color: #1E1E1E;
        padding: 15px; border-radius: 10px; border: 1px solid #333;
    }
    /* SCROLLABLE CONTAINER FOR MATRIX */
    .matrix-container {
        overflow-x: auto;
        overflow-y: hidden;
        width: 100%;
    }
    .element-container iframe { height: auto !important; }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
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
        if isinstance(data.columns, pd.MultiIndex):
            return data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
        return data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    except: return pd.DataFrame()

def calculate_portfolio_metrics(weights, mean_returns, cov_matrix, risk_free_rate):
    returns = np.sum(mean_returns * weights) * 252
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe = (returns - risk_free_rate) / std_dev
    return returns, std_dev, sharpe

# --- Main App ---
def main():
    st.title("💎 QuantPro: Portfolio Intelligence")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        rf_rate = st.number_input("Risk Free Rate (Rf)", value=0.065, step=0.005, format="%.3f")
        usd_rate = get_currency_rate()
        st.metric("USD/INR Rate", f"₹{usd_rate:.2f}")

    if 'portfolio_df' not in st.session_state:
        data = {
            "Ticker": ["OLAELEC.NS", "BAJAJHFL.NS", "CESC.NS", "IT.NS", "TATASILV.NS", "KALYANKJIL.NS", "ITC.NS", "CASTROLIND.NS", "GAIL.NS", "REDINGTON.NS", "ADANIPOWER.NS", "TMPV.NS", "GROWW.NS", "BSLNIFTY.NS", "PHARMABEES.NS", "GROWWMETAL.NS", "TATAGOLD.NS", "TATASTEEL.NS", "VEDL.NS", "SBIN.NS"],
            "Shares": [31, 20, 20, 70, 123, 10, 7, 20, 20, 10, 10, 10, 12, 72, 100, 195, 155, 20, 14, 10],
            "Avg Cost": [37.86, 109.5, 176.18, 40.19, 27.4, 473.05, 351.99, 204.65, 177.22, 273.55, 152.04, 391.37, 175.32, 29.49, 22.38, 10.74, 13.53, 171.74, 524.11, 881.58],
            "Currency": ["INR"] * 20
        }
        st.session_state.portfolio_df = pd.DataFrame(data)

    with st.expander("💼 View / Edit Portfolio Holdings", expanded=False):
        edited_df = st.data_editor(st.session_state.portfolio_df, num_rows="dynamic", use_container_width=True)

    tickers = edited_df['Ticker'].unique().tolist()
    with st.spinner("🚀 Crunching Market Data..."):
        market_data = fetch_data(tickers)
        if market_data.empty: st.stop()

    # Calculations
    current_prices = market_data.iloc[-1]
    def get_current_val(row):
        price = current_prices.get(row['Ticker'], 0)
        val = price * row['Shares']
        return val * usd_rate if str(row.get('Currency')).upper() == 'USD' else val

    edited_df['Current_Value'] = edited_df.apply(get_current_val, axis=1)
    total_value = edited_df['Current_Value'].sum()
    total_invested = (edited_df['Shares'] * edited_df['Avg Cost']).sum()
    pnl_pct = ((total_value - total_invested) / total_invested) * 100
    
    log_returns = np.log(market_data / market_data.shift(1)).dropna()
    mean_returns = log_returns.mean()
    cov_matrix = log_returns.cov()
    
    weights = np.array([edited_df[edited_df['Ticker']==t]['Current_Value'].sum()/total_value for t in market_data.columns])
    curr_ret, curr_std, curr_sharpe = calculate_portfolio_metrics(weights, mean_returns, cov_matrix, rf_rate)

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Value", f"₹{total_value:,.0f}")
    c2.metric("📈 Net P&L", f"{pnl_pct:.2f}%")
    c3.metric("⚡ Sharpe", f"{curr_sharpe:.2f}")
    c4.metric("🌪️ Volatility", f"{curr_std*100:.1f}%")

    # Intelligence
    st.subheader("💡 Quant Intelligence")
    if pnl_pct > 0: st.success(f"**Doing Good!** 🚀 Outperforming capital costs by {pnl_pct:.1f}%.")
    else: st.warning("**Under Pressure.** Drawdown detected in current holdings.")

    tab1, tab2, tab3 = st.tabs(["🚀 Frontier", "🔬 Legible Matrix", "🌪️ Stress Test"])

    with tab1:
        num_port = 1500
        res = np.zeros((3, num_port))
        for i in range(num_port):
            w = np.random.random(len(tickers)); w /= np.sum(w)
            r, s, sh = calculate_portfolio_metrics(w, mean_returns, cov_matrix, rf_rate)
            res[0,i], res[1,i], res[2,i] = s, r, sh
        fig_ef = px.scatter(x=res[0,:], y=res[1,:], color=res[2,:], template="plotly_dark", color_continuous_scale='Spectral_r')
        fig_ef.add_trace(go.Scatter(x=[curr_std], y=[curr_ret], mode='markers+text', text=["YOUR PORTFOLIO"], marker=dict(color='white', size=15, symbol='star')))
        st.plotly_chart(fig_ef, use_container_width=True)

    with tab2:
        st.subheader("🔬 Asset Correlation Matrix")
        st.caption("Scroll right to see all 20+ assets. Fonts optimized for readability.")
        
        corr_matrix = log_returns.corr()
        
        # KEY ADJUSTMENTS: Auto-aspect for width, scrollable container, and smaller font
        fig_corr = px.imshow(
            corr_matrix, text_auto=".2f", aspect="auto", 
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            width=1800, height=1000 # Massive canvas
        )
        
        fig_corr.update_layout(
            font=dict(size=11, color="white"), # Smaller font to prevent overlap
            margin=dict(l=150, r=50, t=50, b=150),
            template="plotly_dark",
            coloraxis_showscale=False
        )
        fig_corr.update_xaxes(tickangle=-45)
        
        st.markdown('<div class="matrix-container">', unsafe_allow_html=True)
        st.plotly_chart(fig_corr, use_container_width=False) # container_width=False allows the 1800px width to trigger scroll
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.subheader("🌪️ Risk Tornado")
        c_in, c_ch = st.columns([1, 2])
        with c_in:
            crash = st.slider("Crash %", -50, 0, -20)
            ai_boom = st.slider("🤖 Tech AI Boom %", 0, 100, 40)
        with c_ch:
            beta = curr_std / 0.15
            scen = {f"Crash ({crash}%)": (crash/100)*beta, f"AI Boom (+{ai_boom}%)": (ai_boom/100)*(beta*1.5)}
            fig_tor = go.Figure(go.Bar(x=[total_value*v for v in scen.values()], y=list(scen.keys()), orientation='h',
                                      marker_color=['#FF4B4B' if x<0 else '#00CC96' for x in scen.values()]))
            fig_tor.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_tor, use_container_width=True)

if __name__ == "__main__":
    main()
        
