import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration (Must be first) ---
st.set_page_config(
    page_title="QuantPro: Institutional Analytics",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling for "Standout" Look ---
st.markdown("""
<style>
    /* Dark Theme Optimization */
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    
    /* Metrics Styling */
    div[data-testid="stMetric"] {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Headers */
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; font-weight: 300; }
    h1 { color: #4F8BF9; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #1E1E1E; border-radius: 4px; color: #FFF; }
    .stTabs [aria-selected="true"] { background-color: #4F8BF9; color: #FFF; }
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
        # Handle multi-level columns if multiple tickers
        if isinstance(data.columns, pd.MultiIndex):
            # Prioritize Adj Close, then Close
            if 'Adj Close' in data.columns.get_level_values(0):
                return data['Adj Close']
            elif 'Close' in data.columns.get_level_values(0):
                return data['Close']
        # Handle single level columns
        elif 'Adj Close' in data.columns:
            return data['Adj Close']
        elif 'Close' in data.columns:
            return data['Close']
        
        # Fallback for single ticker
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

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File Upload
        uploaded_file = st.file_uploader("📂 Upload Portfolio (CSV/Excel)", type=["csv", "xlsx", "json"])
        
        st.divider()
        st.subheader("🌍 Macro Inputs")
        risk_free_rate = st.number_input("Risk Free Rate (Rf)", value=0.065, step=0.005, format="%.3f", help="Current Indian 10y Bond Yield")
        st.caption(f"Hurdle Rate: **{risk_free_rate*100:.1f}%**")
        
        usd_rate = get_currency_rate()
        st.metric("USD/INR Rate", f"₹{usd_rate:.2f}")

    # --- Data Loading ---
    if 'portfolio_df' not in st.session_state:
        # Default Data (Your provided portfolio)
        data = {
            "Ticker": ["OLAELEC.NS", "BAJAJHFL.NS", "CESC.NS", "IT.NS", "TATSILV.NS", "KALYANKJIL.NS", "ITC.NS", "CASTROLIND.NS", "GAIL.NS", "REDINGTON.NS", "ADANIPOWER.NS", "TMPV.NS", "GROWW.NS", "BSLNIFTY.NS", "PHARMABEES.NS", "GROWWMETAL.NS", "TATAGOLD.NS", "TATASTEEL.NS", "VEDL.NS", "SBIN.NS"],
            "Shares": [31, 20, 20, 70, 123, 10, 7, 20, 20, 10, 10, 10, 12, 72, 100, 195, 155, 20, 14, 10],
            "Avg Cost": [37.86, 109.5, 176.18, 40.19, 27.4, 473.05, 351.99, 204.65, 177.22, 273.55, 152.04, 391.37, 175.32, 29.49, 22.38, 10.74, 13.53, 171.74, 524.11, 881.58],
            "Currency": ["INR"] * 20
        }
        st.session_state.portfolio_df = pd.DataFrame(data)

    # Handle Upload
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.json'): df = pd.read_json(uploaded_file)
            else: df = pd.read_excel(uploaded_file)
            st.session_state.portfolio_df = df
            st.sidebar.success("✅ Portfolio Loaded")
        except: st.sidebar.error("❌ Invalid File")

    # Display Data Editor
    with st.expander("💼 View / Edit Portfolio Holdings", expanded=False):
        edited_df = st.data_editor(st.session_state.portfolio_df, num_rows="dynamic", use_container_width=True)

    # --- Processing ---
    portfolio_clean = edited_df.copy()
    if portfolio_clean.empty: st.stop()

    tickers = portfolio_clean['Ticker'].unique().tolist()
    
    with st.spinner("🚀 Crunching Market Data..."):
        market_data = fetch_data(tickers)
    
    if market_data.empty:
        st.error("Could not fetch market data. Check ticker symbols.")
        st.stop()

    # Calculate Current Value & Weights
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

    # --- Top Level Metrics ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Portfolio Value", f"₹{total_value:,.0f}")
    c2.metric("💸 Invested Capital", f"₹{total_invested:,.0f}")
    c3.metric("📈 Net P&L", f"₹{pnl:,.0f}", f"{(pnl/total_invested)*100:.2f}%")
    
    # --- Analytics Engine ---
    log_returns = np.log(market_data / market_data.shift(1)).dropna()
    mean_returns = log_returns.mean()
    cov_matrix = log_returns.cov()
    
    # Current Stats
    curr_weights = np.array([portfolio_clean[portfolio_clean['Ticker']==t]['Weight'].sum() for t in market_data.columns])
    curr_ret, curr_std, curr_sharpe = calculate_portfolio_metrics(curr_weights, mean_returns, cov_matrix, risk_free_rate)
    
    c4.metric("⚡ True Sharpe Ratio", f"{curr_sharpe:.2f}", help=f"Adjusted for {risk_free_rate*100}% Risk-Free Rate")

    st.divider()

    # --- TABS INTERFACE ---
    tab1, tab2, tab3 = st.tabs(["🚀 Efficient Frontier", "🔬 Covariance Matrix", "🌪️ Stress Test (Tornado)"])

    # --- TAB 1: EFFICIENT FRONTIER ---
    with tab1:
        c_left, c_right = st.columns([3, 1])
        
        with c_left:
            # Monte Carlo Simulation
            num_portfolios = 3000
            n_assets = len(market_data.columns)
            results = np.zeros((3, num_portfolios))
            
            for i in range(num_portfolios):
                w = np.random.random(n_assets)
                w /= np.sum(w)
                r, s, sh = calculate_portfolio_metrics(w, mean_returns, cov_matrix, risk_free_rate)
                results[0,i], results[1,i], results[2,i] = s, r, sh
            
            # Plot
            fig_ef = px.scatter(
                x=results[0,:], y=results[1,:], color=results[2,:],
                labels={'x': 'Annualized Risk (Volatility)', 'y': 'Annualized Return', 'color': 'Sharpe'},
                title="Efficient Frontier Optimization",
                color_continuous_scale='Spectral_r', template="plotly_dark"
            )
            # Add Current Portfolio Star
            fig_ef.add_trace(go.Scatter(
                x=[curr_std], y=[curr_ret], mode='markers+text',
                marker=dict(color='white', size=18, symbol='star', line=dict(color='red', width=2)),
                text=["YOU"], textposition="top center", name="Current Portfolio"
            ))
            st.plotly_chart(fig_ef, use_container_width=True)

        with c_right:
            st.markdown("### 📊 Metrics")
            st.write("Performance stats based on last 2 years of daily data.")
            
            st.info(f"**Annualized Return:** {curr_ret*100:.2f}%")
            st.error(f"**Annualized Risk:** {curr_std*100:.2f}%")
            
            st.markdown("#### The Formula")
            st.latex(r"Sharpe = \frac{R_p - R_f}{\sigma_p}")
            st.caption(f"Where $R_f$ (Risk Free) = {risk_free_rate*100}%")
            
            st.markdown("---")
            if curr_sharpe > 1.0:
                st.success("✅ **Great!** You are generating solid excess returns.")
            elif curr_sharpe > 0.5:
                st.warning("⚠️ **Good**, but room for optimization.")
            else:
                st.error("❌ **Risk Alert:** Return is not justifying the volatility.")

    # --- TAB 2: COVARIANCE MATRIX ---
    with tab2:
    st.subheader("🔬 Asset Correlation Matrix")
    st.caption("A larger scale ensures labels don't overlap. Red = High Correlation, Blue = Diversification.")
    
    # Calculate Correlation
    corr_matrix = returns.corr()
    
    # --- DYNAMIC LEGIBILITY UPDATES ---
    # 1. Calculate dynamic height (approx 35 pixels per ticker)
    num_tickers = len(tickers)
    dynamic_height = max(500, num_tickers * 35) 
    
    fig_corr = px.imshow(
        corr_matrix, 
        text_auto=".2f", # Show values with 2 decimals
        aspect="auto", 
        color_continuous_scale="RdBu_r", 
        zmin=-1, zmax=1,
        height=dynamic_height # Apply dynamic height
    )
    
    # 2. Refine Layout for Readability
    fig_corr.update_layout(
        margin=dict(l=100, r=20, t=50, b=100), # Add margins for long ticker names
        font=dict(size=10), # Set a readable base font size
        xaxis_nticks=num_tickers,
        yaxis_nticks=num_tickers
    )
    
    # 3. Rotate X-axis labels to prevent overlapping
    fig_corr.update_xaxes(side="bottom", tickangle=-45)
    
    st.plotly_chart(fig_corr, use_container_width=True)

    # --- TAB 3: TORNADO STRESS TEST ---
    with tab3:
        st.subheader("🌪️ Interactive Stress Testing")
        st.caption("Adjust the sliders to simulate market scenarios and see P&L Impact.")
        
        col_inputs, col_chart = st.columns([1, 2])
        
        with col_inputs:
            st.markdown("#### 🎛️ Scenario Inputs")
            crash_pct = st.slider("📉 Market Crash Severity", min_value=-50, max_value=0, value=-15, step=5)
            bull_pct = st.slider("📈 Bull Run Magnitude", min_value=0, max_value=50, value=15, step=5)
            rate_hike_pct = st.slider("🏦 Rate Hike Impact (Banks/Auto)", min_value=-20, max_value=0, value=-5, step=1)
            tech_boom_pct = st.slider("🤖 Tech/AI Rally", min_value=0, max_value=30, value=10, step=2)

        with col_chart:
            # Calculate Beta (approximate using Portfolio Vol / Market Vol 15%)
            beta = curr_std / 0.15
            
            scenarios = {
                f"Market Crash ({crash_pct}%)": (crash_pct/100) * beta,
                f"Bull Run (+{bull_pct}%)": (bull_pct/100) * beta,
                "Interest Rate Hike Shock": (rate_hike_pct/100) * (beta * 1.2), # Banks hurt more
                "Tech/Innovation Boom": (tech_boom_pct/100) * (beta * 0.9)
            }
            
            impacts = [total_value * factor for factor in scenarios.values()]
            names = list(scenarios.keys())
            colors = ['#FF4B4B' if x < 0 else '#00CC96' for x in impacts]
            
            fig_tor = go.Figure(go.Bar(
                x=impacts, y=names, orientation='h',
                marker_color=colors,
                text=[f"₹{x:,.0f}" for x in impacts],
                textposition="auto"
            ))
            
            fig_tor.update_layout(
                title="Projected P&L Impact (INR)",
                xaxis_title="Profit / Loss",
                template="plotly_dark",
                yaxis={'categoryorder':'total ascending'},
                height=400
            )
            st.plotly_chart(fig_tor, use_container_width=True)

if __name__ == "__main__":
    main()
    


