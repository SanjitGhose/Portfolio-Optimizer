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
    /* Make the data editor look cleaner */
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
        # Get USD to INR rate
        data = yf.Ticker("INR=X").history(period="1d")
        if not data.empty:
            return data['Close'].iloc[-1]
        return 84.0 # Fallback
    except:
        return 84.0 

def fetch_data(tickers, period="2y"):
    """
    Robust data fetching that handles single vs multiple tickers 
    and the missing 'Adj Close' bug.
    """
    if not tickers:
        return pd.DataFrame()
    
    try:
        # auto_adjust=False guarantees we get 'Adj Close' column
        # threads=True speeds up bulk downloading
        data = yf.download(tickers, period=period, auto_adjust=False, threads=True)
        
        # Scenario 1: 'Adj Close' is in columns (Standard case)
        if 'Adj Close' in data.columns:
            return data['Adj Close']
        
        # Scenario 2: Columns are 'Close' because auto_adjust failed to disable or API change
        elif 'Close' in data.columns:
            return data['Close']
            
        # Scenario 3: Flat DataFrame (Single ticker edge case)
        # If we asked for 1 ticker, yf might return just columns like [Open, Close...]
        # We need to return it as a Series or DataFrame correctly
        return data.iloc[:, 0] if not data.empty else pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def calculate_portfolio_metrics(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return returns, std_dev

# --- Main App ---

def main():
    st.title("📈 QuantPro: The Efficient Frontier Engine")
    st.markdown("### *Institutional Grade Portfolio Analytics*")

    # --- Sidebar: Macro Settings ---
    with st.sidebar:
        st.header("🌍 Macro Intelligence")
        country = st.selectbox("Select Your Region", ["India", "USA", "Global"])
        
        st.divider()
        st.info("💡 **Macro Strategy**")
        if country == "India":
            st.write("**Tailwinds:** Capex Cycle (Rail/Defence), Banking Credit Growth, Mfg PLI.")
            st.write("**Sector Picks:** Infra, PSU Banks, Cap Goods.")
        elif country == "USA":
            st.write("**Tailwinds:** AI Infrastructure, Rate Cut Hopes.")
            st.write("**Sector Picks:** Semiconductors, Big Tech, Utilities.")
        else:
            st.write("**Global:** Gold & Silver as volatility hedge.")
        
        st.divider()
        st.write("Current USD/INR Rate: ₹" + str(round(get_currency_rate(), 2)))

    # --- Section 1: Bulk Portfolio Input ---
    st.subheader("💼 Portfolio Management")
    st.markdown("Add all your assets below. You can copy-paste from Excel.")

    # Initialize Session State for DataFrame
    if 'portfolio_df' not in st.session_state:
        # Default starting data
        data = {
            "Ticker": ["TATASTEEL.NS", "SBIN.NS", "VEDL.NS", "GOLDBEES.NS"],
            "Shares": [20, 10, 14, 155],
            "Avg Cost": [171.74, 881.58, 524.11, 58.0],
            "Currency": ["INR", "INR", "INR", "INR"]
        }
        st.session_state.portfolio_df = pd.DataFrame(data)

    # The NEW Bulk Editor
    edited_df = st.data_editor(
        st.session_state.portfolio_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn(
                "Ticker Symbol", 
                help="Yahoo Finance Ticker (e.g., RELIANCE.NS, AAPL)",
                validate="^[A-Za-z0-9\-\.]+$"
            ),
            "Shares": st.column_config.NumberColumn("Shares Held", min_value=0.01, format="%.2f"),
            "Avg Cost": st.column_config.NumberColumn("Avg Buy Price", min_value=0.0, format="₹%.2f"),
            "Currency": st.column_config.SelectboxColumn("Currency", options=["INR", "USD"], required=True)
        },
        key="portfolio_editor"
    )

    # --- Validation & Processing ---
    # Filter out empty rows
    portfolio_clean = edited_df[edited_df["Ticker"].str.len() > 1].copy()
    
    if portfolio_clean.empty:
        st.warning("⚠️ Please add at least one stock to begin.")
        st.stop()

    tickers_list = portfolio_clean['Ticker'].unique().tolist()
    
    # Process Currency & Values
    usd_rate = get_currency_rate()
    
    def calculate_invested(row):
        val = row['Shares'] * row['Avg Cost']
        return val * usd_rate if row['Currency'] == 'USD' else val

    portfolio_clean['Invested_INR'] = portfolio_clean.apply(calculate_invested, axis=1)

    # --- Section 2: Data Fetching ---
    if len(tickers_list) > 0:
        # Fetch Data
        market_data = fetch_data(tickers_list)
        
        if market_data.empty:
            st.error("❌ Failed to download market data. Check your ticker symbols.")
            st.stop()

        # Handle formatting (ensure we have a DataFrame even for 1 stock)
        if isinstance(market_data, pd.Series):
            market_data = market_data.to_frame(name=tickers_list[0])

        # Get Current Prices (Last available row)
        current_prices = market_data.iloc[-1]

        # Map current prices back to portfolio
        current_vals = []
        for index, row in portfolio_clean.iterrows():
            ticker = row['Ticker']
            try:
                # Handle cases where ticker might be missing in download
                if ticker in current_prices:
                    price = current_prices[ticker]
                    val = price * row['Shares']
                    val = val * usd_rate if row['Currency'] == 'USD' else val
                    current_vals.append(val)
                else:
                    current_vals.append(0)
            except:
                current_vals.append(0)
        
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

        # --- Section 4: Risk Analytics (The Quant Stuff) ---
        if len(tickers_list) > 1:
            st.subheader("🔬 Deep Risk Analysis")
            
            # Returns Calculation
            log_returns = np.log(market_data / market_data.shift(1)).dropna()
            
            # Handle duplicates in log_returns if user entered same ticker twice
            log_returns = log_returns.loc[:, ~log_returns.columns.duplicated()]
            
            mean_returns = log_returns.mean()
            cov_matrix = log_returns.cov()

            # A. Covariance Heatmap
            with st.expander("Show Covariance Matrix (Correlation Heatmap)", expanded=True):
                corr_matrix = log_returns.corr()
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1,
                    title="Asset Correlation (Red = Moves Together, Blue = Hedge)"
                )
                st.plotly_chart(fig_corr, use_container_width=True)

            # B. Efficient Frontier
            st.subheader("🚀 Efficient Frontier")
            col_graph, col_controls = st.columns([2, 1])

            with col_graph:
                # Monte Carlo Sim
                num_portfolios = 3000
                results = np.zeros((3, num_portfolios))
                
                # We need to ensure weights match the unique tickers in log_returns
                unique_tickers = log_returns.columns.tolist()
                n_assets = len(unique_tickers)

                for i in range(num_portfolios):
                    weights = np.random.random(n_assets)
                    weights /= np.sum(weights)
                    pret, pstd = calculate_portfolio_metrics(weights, mean_returns, cov_matrix)
                    results[0,i] = pstd
                    results[1,i] = pret
                    results[2,i] = pret / pstd # Sharpe

                # Current Portfolio Position
                # We need to aggregate weights by Ticker (in case user has 2 entries for same stock)
                grouped_weights = portfolio_clean.groupby('Ticker')['Weight'].sum()
                # Reorder weights to match log_returns columns
                current_w_array = np.array([grouped_weights.get(t, 0) for t in unique_tickers])
                
                cur_ret, cur_std = calculate_portfolio_metrics(current_w_array, mean_returns, cov_matrix)

                fig_ef = go.Figure()
                fig_ef.add_trace(go.Scatter(
                    x=results[0,:], y=results[1,:],
                    mode='markers',
                    marker=dict(color=results[2,:], colorscale='Viridis', size=4, showscale=True, colorbar=dict(title="Sharpe")),
                    name='Simulated'
                ))
                fig_ef.add_trace(go.Scatter(
                    x=[cur_std], y=[cur_ret],
                    mode='markers+text',
                    marker=dict(color='red', size=15, symbol='star'),
                    text=["YOU"], textposition="top center",
                    name='Current'
                ))
                fig_ef.update_layout(
                    title="Risk vs Return Optimization",
                    xaxis_title="Risk (Annualized Volatility)",
                    yaxis_title="Return (Annualized)",
                    template="plotly_dark",
                    height=500
                )
                st.plotly_chart(fig_ef, use_container_width=True)

            with col_controls:
                st.markdown("### 🎛️ Optimization Sandbox")
                st.info("Adjust weights below to see how the 'Red Star' moves.")
                
                sim_weights = {}
                # Create Sliders for Unique Tickers
                for t in unique_tickers:
                    # Get current weight
                    cw = grouped_weights.get(t, 0.0)
                    sim_weights[t] = st.slider(f"{t}", 0.0, 1.0, float(cw), 0.05)
                
                # Normalize sliders
                total_sim_w = sum(sim_weights.values())
                if total_sim_w > 0:
                    normalized_sim = np.array([sim_weights[t]/total_sim_w for t in unique_tickers])
                    
                    sim_ret, sim_std = calculate_portfolio_metrics(normalized_sim, mean_returns, cov_matrix)
                    
                    st.divider()
                    st.metric("Simulated Return", f"{sim_ret*100:.2f}%", delta=f"{(sim_ret-cur_ret)*100:.2f}%")
                    st.metric("Simulated Risk", f"{sim_std*100:.2f}%", delta=f"{(sim_std-cur_std)*100:.2f}%", delta_color="inverse")

            # C. Tornado Sensitivity
            st.subheader("🌪️ Macro Sensitivity (Tornado Chart)")
            
            # Simple Beta Estimation
            # If portfolio vol > market vol (15%), assume high beta
            implied_beta = cur_std / 0.15 
            
            scenarios = {
                "Market Crash (-10%)": -0.10 * implied_beta,
                "Bull Run (+10%)": 0.10 * implied_beta,
                "Interest Rate Hike": -0.05 * (implied_beta * 1.2), # Banks hurt more
                "Tech/AI Boom": 0.08 * (implied_beta * 0.9) # General lift
            }
            
            impacts = [total_curr * move for move in scenarios.values()]
            names = list(scenarios.keys())
            colors = ['red' if x < 0 else 'green' for x in impacts]
            
            fig_tor = go.Figure(go.Bar(
                x=impacts, y=names, orientation='h',
                marker_color=colors,
                text=[f"₹{x:,.0f}" for x in impacts],
                textposition="auto"
            ))
            fig_tor.update_layout(title="Estimated P&L Impact", template="plotly_dark")
            st.plotly_chart(fig_tor, use_container_width=True)

    # --- Save Button ---
    st.divider()
    json_data = portfolio_clean.to_json(orient="records")
    st.download_button("💾 Save Portfolio", json_data, "my_portfolio.json", "application/json")

if __name__ == "__main__":
    main()
