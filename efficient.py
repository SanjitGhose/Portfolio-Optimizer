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
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; font-weight: 300; }
    h1 { color: #4F8BF9; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #1E1E1E; border-radius: 4px; color: #FFF; }
    .stTabs [aria-selected="true"] { background-color: #4F8BF9; color: #FFF; }
    
    /* Ensure the plot container allows for full height expansion */
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
            if 'Adj Close' in data.columns.get_level_values(0):
                return data['Adj Close']
            elif 'Close' in data.columns.get_level_values(0):
                return data['Close']
        elif 'Adj Close' in data.columns:
            return data['Adj Close']
        elif 'Close' in data.columns:
            return data['Close']
        return data.iloc[:, 0] if not data.empty else pd.DataFrame()
    except: return pd.DataFrame()

def calculate_portfolio_metrics(weights, mean_returns, cov_matrix, risk_free_rate):
    returns = np.sum(mean_returns * weights) * 252
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe = (returns - risk_free_rate) / std_dev
    return returns, std_dev, sharpe

# --- Main App ---

def main():
    st.title("💎 QuantPro: Portfolio Intelligence")
    st.caption(" institutional grade risk analytics & optimization engine")

    with st.sidebar:
        st.header("⚙️ Configuration")
        uploaded_file = st.file_uploader("📂 Upload Portfolio", type=["csv", "xlsx", "json"])
        st.divider()
        st.subheader("🌍 Macro Inputs")
        risk_free_rate = st.number_input("Risk Free Rate (Rf)", value=0.065, step=0.005, format="%.3f")
        usd_rate = get_currency_rate()
        st.metric("USD/INR Rate", f"₹{usd_rate:.2f}")

    # Data Loading
    if 'portfolio_df' not in st.session_state:
        data = {
            "Ticker": ["OLAELEC.NS", "BAJAJHFL.NS", "CESC.NS", "IT.NS", "TATSILV.NS", "KALYANKJIL.NS", "ITC.NS", "CASTROLIND.NS", "GAIL.NS", "REDINGTON.NS", "ADANIPOWER.NS", "TMPV.NS", "GROWW.NS", "BSLNIFTY.NS", "PHARMABEES.NS", "GROWWMETAL.NS", "TATAGOLD.NS", "TATASTEEL.NS", "VEDL.NS", "SBIN.NS"],
            "Shares": [31, 20, 20, 70, 123, 10, 7, 20, 20, 10, 10, 10, 12, 72, 100, 195, 155, 20, 14, 10],
            "Avg Cost": [37.86, 109.5, 176.18, 40.19, 27.4, 473.05, 351.99, 204.65, 177.22, 273.55, 152.04, 391.37, 175.32, 29.49, 22.38, 10.74, 13.53, 171.74, 524.11, 881.58],
            "Currency": ["INR"] * 20
        }
        st.session_state.portfolio_df = pd.DataFrame(data)

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.json'): df = pd.read_json(uploaded_file)
            else: df = pd.read_excel(uploaded_file)
            st.session_state.portfolio_df = df
        except: st.sidebar.error("❌ Invalid File")

    with st.expander("💼 View / Edit Portfolio Holdings", expanded=False):
        edited_df = st.data_editor(st.session_state.portfolio_df, num_rows="dynamic", use_container_width=True)

    portfolio_clean = edited_df.copy()
    if portfolio_clean.empty: st.stop()

    tickers = portfolio_clean['Ticker'].unique().tolist()
    
    with st.spinner("🚀 Crunching Market Data..."):
        market_data = fetch_data(tickers)
    
    if market_data.empty:
        st.error("Could not fetch market data.")
        st.stop()

    current_prices = market_data.iloc[-1]
    def get_current_val(row):
        price = current_prices.get(row['Ticker'], 0)
        val = price * row['Shares']
        return val * usd_rate if str(row.get('Currency')).upper() == 'USD' else val

    portfolio_clean['Current_Value'] = portfolio_clean.apply(get_current_val, axis=1)
    total_value = portfolio_clean['Current_Value'].sum()
    portfolio_clean['Weight'] = portfolio_clean['Current_Value'] / total_value
    
    total_invested = (portfolio_clean['Shares'] * portfolio_clean['Avg Cost']).sum()
    pnl = total_value - total_invested

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Portfolio Value", f"₹{total_value:,.0f}")
    c2.metric("💸 Invested Capital", f"₹{total_invested:,.0f}")
    c3.metric("📈 Net P&L", f"₹{pnl:,.0f}", f"{(pnl/total_invested)*100:.2f}%")
    
    log_returns = np.log(market_data / market_data.shift(1)).dropna()
    mean_returns = log_returns.mean()
    cov_matrix = log_returns.cov()
    
    curr_weights = np.array([portfolio_clean[portfolio_clean['Ticker']==t]['Weight'].sum() for t in market_data.columns])
    curr_ret, curr_std, curr_sharpe = calculate_portfolio_metrics(curr_weights, mean_returns, cov_matrix, risk_free_rate)
    
    c4.metric("⚡ True Sharpe Ratio", f"{curr_sharpe:.2f}")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["🚀 Efficient Frontier", "🔬 Covariance Matrix", "🌪️ Stress Test"])

    # --- TAB 1: EFFICIENT FRONTIER ---
    with tab1:
        st.subheader("Efficient Frontier Optimization")
        c_left, c_right = st.columns([3, 1])
        with c_left:
            num_portfolios = 2000
            n_assets = len(market_data.columns)
            results = np.zeros((3, num_portfolios))
            for i in range(num_portfolios):
                w = np.random.random(n_assets)
                w /= np.sum(w)
                r, s, sh = calculate_portfolio_metrics(w, mean_returns, cov_matrix, risk_free_rate)
                results[0,i], results[1,i], results[2,i] = s, r, sh
            
            fig_ef = px.scatter(x=results[0,:], y=results[1,:], color=results[2,:],
                                labels={'x': 'Annualized Risk', 'y': 'Annualized Return', 'color': 'Sharpe'},
                                template="plotly_dark", color_continuous_scale='Spectral_r')
            
            fig_ef.add_trace(go.Scatter(
                x=[curr_std], y=[curr_ret], 
                mode='markers+text',
                text=["YOUR PORTFOLIO"],
                textposition="top center",
                marker=dict(color='white', size=20, symbol='star', line=dict(color='red', width=3)),
                name="Current Status"
            ))
            st.plotly_chart(fig_ef, use_container_width=True)

        with c_right:
            st.markdown("#### The Sharpe Formula")
            st.latex(r"Sharpe = \frac{R_p - R_f}{\sigma_p}")
            st.info(f"**Ann. Return:** {curr_ret*100:.2f}%")
            st.error(f"**Ann. Risk:** {curr_std*100:.2f}%")

    # --- TAB 2: LEGIBLE COVARIANCE MATRIX (MAXIMIZED VIEW) ---
    with tab2:
        st.subheader("🔬 Asset Correlation Matrix")
        st.caption("Red = Positive Correlation | Blue = Negative Correlation. Full details visible upon expansion.")
        
        corr_matrix = log_returns.corr()
        num_assets = len(market_data.columns)
        
        # INCREASED DYNAMIC HEIGHT: 50 pixels per asset ensures labels and cells are huge
        dynamic_height = max(1000, num_assets * 50)
        
        fig_corr = px.imshow(
            corr_matrix, 
            text_auto=".2f", 
            aspect="equal", # Keeps cells square so they don't stretch awkwardly
            color_continuous_scale="RdBu_r", 
            zmin=-1, zmax=1,
            height=dynamic_height
        )
        
        fig_corr.update_layout(
            font=dict(size=14, color="white"),
            margin=dict(l=150, r=50, t=100, b=150), 
            xaxis_nticks=num_assets,
            yaxis_nticks=num_assets,
            template="plotly_dark",
            coloraxis_colorbar=dict(title="Corr", thickness=20)
        )
        
        fig_corr.update_xaxes(tickangle=-45, side="bottom", tickfont=dict(size=13))
        fig_corr.update_yaxes(tickfont=dict(size=13))
        
        # Set use_container_width to True to let it fill the center column, 
        # but the height will drive the "Expansion" clarity.
        st.plotly_chart(fig_corr, use_container_width=True)
        
    # --- TAB 3: TORNADO STRESS TEST ---
    with tab3:
        st.subheader("🌪️ Interactive Stress Testing")
        col_inputs, col_chart = st.columns([1, 2])
        with col_inputs:
            crash_pct = st.slider("📉 Crash %", -50, 0, -15)
            bull_pct = st.slider("📈 Bull %", 0, 50, 15)
            rate_hike = st.slider("🏦 Rate Hike Impact %", -20, 0, -5)

        with col_chart:
            beta = curr_std / 0.15
            scenarios = {f"Crash ({crash_pct}%)": (crash_pct/100) * beta,
                         f"Bull (+{bull_pct}%)": (bull_pct/100) * beta,
                         "Rate Hike Shock": (rate_hike/100) * (beta * 1.2)}
            impacts = [total_value * factor for factor in scenarios.values()]
            fig_tor = go.Figure(go.Bar(x=impacts, y=list(scenarios.keys()), orientation='h',
                                      marker_color=['#FF4B4B' if x < 0 else '#00CC96' for x in impacts],
                                      text=[f"₹{x:,.0f}" for x in impacts], textposition="auto"))
            fig_tor.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_tor, use_container_width=True)

if __name__ == "__main__":
    main()
                         
