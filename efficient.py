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
    .stApp { background-color: #000000; color: #FFFFFF; }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; font-weight: 300; }
    .stDataFrame { border: 1px solid #303030; border-radius: 5px; }
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
        if 'Adj Close' in data.columns: return data['Adj Close']
        elif 'Close' in data.columns: return data['Close']
        return data.iloc[:, 0] if not data.empty else pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def calculate_portfolio_metrics(weights, mean_returns, cov_matrix, risk_free_rate):
    # Annualized Returns
    returns = np.sum(mean_returns * weights) * 252
    # Annualized Volatility
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    # Sharpe Ratio Calculation
    sharpe = (returns - risk_free_rate) / std_dev if std_dev != 0 else 0
    return returns, std_dev, sharpe

# --- Main App ---

def main():
    st.title("📈 QuantPro: The Efficient Frontier Engine")
    st.markdown("### *Institutional Grade Portfolio Analytics*")

    # --- Sidebar: Configuration & Upload ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # FEATURE: Dynamic Risk Free Rate (Bond Rate)
        rf_rate = st.number_input("Risk Free Rate (Bond Rate)", 
                                  value=0.065, 
                                  step=0.005, 
                                  format="%.3f",
                                  help="The return of a risk-free investment, typically the 10Y Govt Bond yield.")
        
        st.divider()
        st.header("📂 Data Management")
        
        # FEATURE: File Uploading
        uploaded_file = st.file_uploader("Upload Portfolio (CSV, XLSX, JSON)", type=["csv", "xlsx", "json"])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_upload = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df_upload = pd.read_json(uploaded_file)
                else:
                    df_upload = pd.read_excel(uploaded_file)
                
                # Ensure required columns exist
                required = ["Ticker", "Shares", "Avg Cost", "Currency"]
                if all(col in df_upload.columns for col in required):
                    st.session_state.portfolio_df = df_upload[required]
                    st.success("File Loaded Successfully!")
                else:
                    st.error(f"Missing columns. File must contain: {required}")
            except Exception as e:
                st.error(f"Error reading file: {e}")

        st.divider()
        st.header("🌍 Macro Intelligence")
        country = st.selectbox("Select Your Region", ["India", "USA", "Global"])
        st.write("Current USD/INR Rate: ₹" + str(round(get_currency_rate(), 2)))

    # --- Section 1: Portfolio Input ---
    st.subheader("💼 Portfolio Management")
    
    if 'portfolio_df' not in st.session_state:
        data = {
            "Ticker": ["TATASTEEL.NS", "SBIN.NS", "VEDL.NS", "GOLDBEES.NS"],
            "Shares": [20, 10, 14, 155],
            "Avg Cost": [171.74, 881.58, 524.11, 58.0],
            "Currency": ["INR", "INR", "INR", "INR"]
        }
        st.session_state.portfolio_df = pd.DataFrame(data)

    edited_df = st.data_editor(
        st.session_state.portfolio_df,
        num_rows="dynamic",
        use_container_width=True,
        key="portfolio_editor"
    )

    portfolio_clean = edited_df[edited_df["Ticker"].str.len() > 1].copy()
    if portfolio_clean.empty: st.stop()

    # --- Calculations & Data ---
    tickers_list = portfolio_clean['Ticker'].unique().tolist()
    usd_rate = get_currency_rate()
    
    def calculate_invested(row):
        val = row['Shares'] * row['Avg Cost']
        return val * usd_rate if row['Currency'] == 'USD' else val

    portfolio_clean['Invested_INR'] = portfolio_clean.apply(calculate_invested, axis=1)

    with st.spinner("Crunching Numbers..."):
        market_data = fetch_data(tickers_list)
        if market_data.empty: st.stop()
        
        current_prices = market_data.iloc[-1]
        current_vals = []
        for _, row in portfolio_clean.iterrows():
            price = current_prices.get(row['Ticker'], 0)
            val = (price * row['Shares']) * (usd_rate if row['Currency'] == 'USD' else 1)
            current_vals.append(val)
        
        portfolio_clean['Current_Value_INR'] = current_vals
        portfolio_clean['Weight'] = portfolio_clean['Current_Value_INR'] / portfolio_clean['Current_Value_INR'].sum()

    # --- Section 3: Metrics ---
    st.divider()
    c1, c2, c3 = st.columns(3)
    total_inv = portfolio_clean['Invested_INR'].sum()
    total_curr = portfolio_clean['Current_Value_INR'].sum()
    c1.metric("Total Invested", f"₹{total_inv:,.0f}")
    c2.metric("Current Value", f"₹{total_curr:,.0f}")
    c3.metric("Net P/L", f"₹{(total_curr - total_inv):,.0f}", f"{((total_curr-total_inv)/total_inv)*100:.2f}%")

    # --- Section 4: Risk Analytics ---
    if len(tickers_list) > 1:
        log_returns = np.log(market_data / market_data.shift(1)).dropna()
        log_returns = log_returns.loc[:, ~log_returns.columns.duplicated()]
        mean_returns = log_returns.mean()
        cov_matrix = log_returns.cov()

        # B. Efficient Frontier
        st.subheader("🚀 Efficient Frontier & Sharpe Analysis")
        
        # FORMULA CALLOUT
        st.latex(r"Sharpe\ Ratio = \frac{R_p - R_f}{\sigma_p}")

        col_graph, col_controls = st.columns([2, 1])

        with col_graph:
            num_portfolios = 3000
            results = np.zeros((3, num_portfolios))
            unique_tickers = log_returns.columns.tolist()
            
            for i in range(num_portfolios):
                weights = np.random.random(len(unique_tickers))
                weights /= np.sum(weights)
                pret, pstd, psharpe = calculate_portfolio_metrics(weights, mean_returns, cov_matrix, rf_rate)
                results[0,i] = pstd
                results[1,i] = pret
                results[2,i] = psharpe

            # Current Portfolio Metrics
            grouped_weights = portfolio_clean.groupby('Ticker')['Weight'].sum()
            current_w_array = np.array([grouped_weights.get(t, 0) for t in unique_tickers])
            cur_ret, cur_std, cur_sharpe = calculate_portfolio_metrics(current_w_array, mean_returns, cov_matrix, rf_rate)

            fig_ef = px.scatter(x=results[0,:], y=results[1,:], color=results[2,:],
                                labels={'x': 'Risk (Volatility)', 'y': 'Return', 'color': 'Sharpe Ratio'},
                                color_continuous_scale='Viridis', template="plotly_dark")
            
            fig_ef.add_trace(go.Scatter(x=[cur_std], y=[cur_ret], mode='markers+text',
                                        marker=dict(color='red', size=15, symbol='star'),
                                        text=["CURRENT PORTFOLIO"], textposition="top center"))
            st.plotly_chart(fig_ef, use_container_width=True)

        with col_controls:
            st.markdown("### 📊 Performance Summary")
            st.info(f"Using Risk-Free Rate: **{rf_rate*100:.2f}%**")
            st.metric("Portfolio Sharpe Ratio", f"{cur_sharpe:.2f}")
            st.metric("Annualized Volatility", f"{cur_std*100:.2f}%")
            
            # THE LEGIBILITY UPGRADE: Asset Correlation
            with st.expander("🔬 Correlation Matrix"):
                corr_matrix = log_returns.corr()
                fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="RdBu_r")
                st.plotly_chart(fig_corr, use_container_width=True)
                

    # Save Button
    json_data = portfolio_clean.to_json(orient="records")
    st.sidebar.download_button("💾 Export to JSON", json_data, "quant_portfolio.json", "application/json")

if __name__ == "__main__":
    main()
