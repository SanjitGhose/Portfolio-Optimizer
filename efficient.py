import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
import json
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="QuantPro: Portfolio Analytics",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .stApp { 
        background: linear-gradient(135deg, #0a0e1a 0%, #1a1f35 50%, #0f1425 100%);
        color: #e8eaed;
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1e2538, #252d45);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(100, 150, 255, 0.2);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: rgba(100, 150, 255, 0.5);
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.2);
    }
    
    h1 {
        font-weight: 700 !important;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(15, 20, 37, 0.4);
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 37, 56, 0.6);
        border-radius: 8px;
        padding: 10px 20px;
        border: 1px solid rgba(100, 150, 255, 0.15);
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.3), rgba(139, 92, 246, 0.3)) !important;
        border-color: rgba(100, 150, 255, 0.5) !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

@st.cache_data(ttl=3600)
def get_currency_rate():
    """Fetch current USD/INR exchange rate"""
    try:
        ticker = yf.Ticker("INR=X")
        data = ticker.history(period="1d")
        if not data.empty:
            return float(data['Close'].iloc[-1])
        return 84.0
    except:
        return 84.0

@st.cache_data(ttl=1800)
def fetch_data(tickers, period="2y"):
    """Fetch historical price data for given tickers"""
    if not tickers:
        return pd.DataFrame()
    
    try:
        # Download data
        data = yf.download(tickers, period=period, auto_adjust=True, threads=True, progress=False)
        
        # Handle single vs multiple tickers
        if len(tickers) == 1:
            if 'Close' in data.columns:
                result = pd.DataFrame({tickers[0]: data['Close']})
            else:
                result = pd.DataFrame()
        else:
            if isinstance(data.columns, pd.MultiIndex):
                result = data['Close'] if 'Close' in data.columns.get_level_values(0) else pd.DataFrame()
            else:
                result = data
        
        # Remove any NaN columns
        result = result.dropna(axis=1, how='all')
        
        return result
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

def calculate_portfolio_metrics(weights, mean_returns, cov_matrix, risk_free_rate):
    """Calculate portfolio return, volatility, and Sharpe ratio"""
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std if portfolio_std > 0 else 0
    
    return portfolio_return, portfolio_std, sharpe_ratio

def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate, target='sharpe'):
    """Optimize portfolio weights"""
    num_assets = len(mean_returns)
    
    def neg_sharpe(weights):
        ret, std, sharpe = calculate_portfolio_metrics(weights, mean_returns, cov_matrix, risk_free_rate)
        return -sharpe
    
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.array([1/num_assets] * num_assets)
    
    objective = neg_sharpe if target == 'sharpe' else portfolio_volatility
    
    try:
        result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x if result.success else initial_weights
    except:
        return initial_weights

def parse_uploaded_portfolio(uploaded_file):
    """Parse portfolio data from uploaded file (CSV, JSON, Excel only)"""
    try:
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        if file_type == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_type == 'json':
            data = json.load(uploaded_file)
            df = pd.DataFrame(data)
        elif file_type in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        else:
            st.error(f"Unsupported file type: {file_type}. Use CSV, JSON, or Excel.")
            return None
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.title()
        column_mapping = {
            'Symbol': 'Ticker',
            'Stock': 'Ticker',
            'Quantity': 'Shares',
            'Qty': 'Shares',
            'Units': 'Shares',
            'Average Cost': 'Avg Cost',
            'Avg Price': 'Avg Cost',
            'Cost': 'Avg Cost',
            'Price': 'Avg Cost',
        }
        df.rename(columns=column_mapping, inplace=True)
        
        # Ensure required columns
        required = ['Ticker', 'Shares', 'Avg Cost']
        if not all(col in df.columns for col in required):
            st.error(f"Missing required columns. Need: {', '.join(required)}")
            return None
        
        # Add Currency column if missing
        if 'Currency' not in df.columns:
            df['Currency'] = 'INR'
        
        # Convert to proper types
        df['Shares'] = pd.to_numeric(df['Shares'], errors='coerce')
        df['Avg Cost'] = pd.to_numeric(df['Avg Cost'], errors='coerce')
        
        # Remove invalid rows
        df = df.dropna(subset=['Ticker', 'Shares', 'Avg Cost'])
        
        return df
        
    except Exception as e:
        st.error(f"Error parsing file: {str(e)}")
        return None

# --- Main App ---
def main():
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0 2rem 0;'>
        <h1 style='margin-bottom: 0.5rem;'>💎 QuantPro: Portfolio Analytics</h1>
        <p style='font-size: 1.1rem; color: #9ca3af; font-weight: 500;'>
            Advanced Portfolio Analytics • Risk Management • Optimization
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.markdown("**Risk-Free Rate**")
        rf_rate = st.number_input(
            "Annual Rate (%)", 
            value=6.5, 
            min_value=0.0,
            max_value=20.0,
            step=0.5,
            help="India 10Y G-Sec yield ≈ 6.5%"
        ) / 100
        
        st.divider()
        
        # Currency rate
        usd_rate = get_currency_rate()
        st.metric("💱 USD/INR", f"₹{usd_rate:.2f}")
        
        st.divider()
        
        # File upload
        st.markdown("**📂 Import Portfolio**")
        uploaded_file = st.file_uploader(
            "Upload CSV, JSON, or Excel",
            type=['csv', 'json', 'xlsx', 'xls'],
            help="Required columns: Ticker, Shares, Avg Cost"
        )
        
        if uploaded_file:
            parsed_df = parse_uploaded_portfolio(uploaded_file)
            if parsed_df is not None:
                st.session_state.portfolio_df = parsed_df
                st.success(f"✅ Loaded {len(parsed_df)} positions")
        
        st.divider()
        
        # Tips
        with st.expander("💡 Quick Tips"):
            st.markdown("""
            **Indian Stocks:**
            - Add `.NS` for NSE stocks
            - Example: `RELIANCE.NS`, `TCS.NS`
            
            **Sharpe Ratio:**
            - < 1: Below average
            - 1-2: Good
            - > 2: Excellent
            """)

    # Initialize default portfolio
    if 'portfolio_df' not in st.session_state:
        st.info("👋 **Welcome!** A default portfolio has been loaded. Upload your own via the sidebar or edit below.")
        
        data = {
            "Ticker": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", 
                      "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "HINDUNILVR.NS", "KOTAKBANK.NS"],
            "Shares": [10, 15, 20, 12, 25, 30, 20, 18, 10, 15],
            "Avg Cost": [2500, 3200, 1400, 1600, 900, 400, 600, 800, 2400, 1800],
            "Currency": ["INR"] * 10
        }
        st.session_state.portfolio_df = pd.DataFrame(data)

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Assets", len(st.session_state.portfolio_df))
    with col2:
        st.metric("🌐 Source", "Yahoo Finance")
    with col3:
        st.metric("📈 Period", "2 Years")
    with col4:
        st.metric("🔄 Status", "Live", delta="✓")
    
    st.markdown("---")

    # Portfolio Editor
    with st.expander("💼 View / Edit Portfolio", expanded=False):
        edited_df = st.data_editor(
            st.session_state.portfolio_df, 
            num_rows="dynamic", 
            use_container_width=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", width="medium"),
                "Shares": st.column_config.NumberColumn("Shares", format="%d"),
                "Avg Cost": st.column_config.NumberColumn("Avg Cost", format="%.2f"),
                "Currency": st.column_config.SelectboxColumn("Currency", options=["INR", "USD"])
            }
        )
        st.session_state.portfolio_df = edited_df

    # Fetch data
    tickers = edited_df['Ticker'].unique().tolist()
    
    if not tickers:
        st.error("⚠️ No tickers in portfolio. Please add positions.")
        st.stop()
    
    with st.spinner("🚀 Fetching market data..."):
        market_data = fetch_data(tickers, period="2y")
        
        if market_data.empty:
            st.error("❌ Failed to fetch data. Check tickers (Indian stocks need .NS suffix)")
            st.stop()
        
        # Check missing tickers
        missing = set(tickers) - set(market_data.columns)
        if missing:
            st.warning(f"⚠️ No data for: {', '.join(missing)}")
            edited_df = edited_df[~edited_df['Ticker'].isin(missing)]
            tickers = [t for t in tickers if t not in missing]
        
        if edited_df.empty:
            st.error("No valid data available.")
            st.stop()
        
        st.success(f"✅ Data fetched: {len(market_data.columns)} assets, {len(market_data)} days")

    # Calculate metrics
    current_prices = market_data.iloc[-1]
    
    def get_current_val(row):
        price = current_prices.get(row['Ticker'], 0)
        val = price * row['Shares']
        return val * usd_rate if str(row.get('Currency', 'INR')).upper() == 'USD' else val

    edited_df['Current_Value'] = edited_df.apply(get_current_val, axis=1)
    total_value = edited_df['Current_Value'].sum()
    total_invested = (edited_df['Shares'] * edited_df['Avg Cost']).sum()
    pnl_abs = total_value - total_invested
    pnl_pct = (pnl_abs / total_invested) * 100 if total_invested > 0 else 0
    
    # Returns and covariance
    log_returns = np.log(market_data / market_data.shift(1)).dropna()
    mean_returns = log_returns.mean()
    cov_matrix = log_returns.cov()
    
    # Portfolio weights
    weights = np.array([edited_df[edited_df['Ticker']==t]['Current_Value'].sum()/total_value 
                       for t in market_data.columns])
    
    # Current metrics
    curr_ret, curr_std, curr_sharpe = calculate_portfolio_metrics(
        weights, mean_returns, cov_matrix, rf_rate
    )
    
    # Optimal portfolios
    optimal_sharpe_weights = optimize_portfolio(mean_returns, cov_matrix, rf_rate, 'sharpe')
    optimal_ret, optimal_std, optimal_sharpe = calculate_portfolio_metrics(
        optimal_sharpe_weights, mean_returns, cov_matrix, rf_rate
    )
    
    min_vol_weights = optimize_portfolio(mean_returns, cov_matrix, rf_rate, 'volatility')
    minvol_ret, minvol_std, minvol_sharpe = calculate_portfolio_metrics(
        min_vol_weights, mean_returns, cov_matrix, rf_rate
    )

    # Display Metrics
    st.markdown("---")
    st.markdown("## 📊 Portfolio Dashboard")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("💰 Portfolio Value", f"₹{total_value:,.0f}", delta=f"₹{pnl_abs:,.0f}")
    
    with col2:
        st.metric("📈 Total Return", f"{pnl_pct:.2f}%", delta=f"{pnl_pct:.2f}%")
    
    with col3:
        sharpe_delta = curr_sharpe - optimal_sharpe
        st.metric("⚡ Sharpe Ratio", f"{curr_sharpe:.3f}", delta=f"{sharpe_delta:.3f}")
    
    with col4:
        st.metric("🌪️ Volatility", f"{curr_std*100:.2f}%")
    
    with col5:
        st.metric("📊 Expected Return", f"{curr_ret*100:.2f}%")

    # Intelligence Brief
    st.markdown("---")
    st.markdown("## 💡 Intelligence Brief")
    
    if pnl_pct > 0:
        st.success(
            f"**Strong Performance** 🚀 Your portfolio gained **{pnl_pct:.2f}%** with a "
            f"Sharpe ratio of **{curr_sharpe:.3f}**."
        )
    else:
        st.warning(
            f"**Under Pressure** ⚠️ Current drawdown: **{pnl_pct:.2f}%**. "
            f"Consider rebalancing."
        )
    
    if curr_sharpe < optimal_sharpe:
        improvement = ((optimal_sharpe - curr_sharpe) / max(curr_sharpe, 0.001)) * 100
        st.info(
            f"**Optimization Opportunity** 💎 Potential **{improvement:.1f}%** Sharpe improvement "
            f"(from {curr_sharpe:.3f} to {optimal_sharpe:.3f}) through rebalancing."
        )

    # Tabs
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚀 Efficient Frontier", 
        "🔬 Correlation Matrix", 
        "🌪️ Stress Testing",
        "📊 Position Analysis"
    ])

    # TAB 1: Efficient Frontier
    with tab1:
        st.subheader("🚀 Efficient Frontier")
        st.caption("Monte Carlo simulation with 5,000 portfolios")
        
        # Monte Carlo
        num_portfolios = 5000
        results = np.zeros((3, num_portfolios))
        
        progress = st.progress(0)
        for i in range(num_portfolios):
            w = np.random.random(len(tickers))
            w /= np.sum(w)
            r, s, sh = calculate_portfolio_metrics(w, mean_returns, cov_matrix, rf_rate)
            results[0, i] = s
            results[1, i] = r
            results[2, i] = sh
            
            if i % 100 == 0:
                progress.progress((i + 1) / num_portfolios)
        
        progress.empty()
        
        # Plot
        fig_ef = go.Figure()
        
        fig_ef.add_trace(go.Scatter(
            x=results[0, :] * 100,
            y=results[1, :] * 100,
            mode='markers',
            name='Simulated',
            marker=dict(size=4, color=results[2, :], colorscale='Viridis', showscale=True, opacity=0.6),
            hovertemplate='Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
        ))
        
        fig_ef.add_trace(go.Scatter(
            x=[curr_std * 100], y=[curr_ret * 100],
            mode='markers+text', name='Your Portfolio',
            text=['YOUR<br>PORTFOLIO'], textposition='top center',
            marker=dict(color='#FFD700', size=20, symbol='star', line=dict(color='white', width=2))
        ))
        
        fig_ef.add_trace(go.Scatter(
            x=[optimal_std * 100], y=[optimal_ret * 100],
            mode='markers+text', name='Optimal',
            text=['OPTIMAL'], textposition='top center',
            marker=dict(color='#00FF00', size=18, symbol='diamond', line=dict(color='white', width=2))
        ))
        
        fig_ef.add_trace(go.Scatter(
            x=[minvol_std * 100], y=[minvol_ret * 100],
            mode='markers+text', name='Min Vol',
            text=['MIN VOL'], textposition='bottom center',
            marker=dict(color='#00CED1', size=16, symbol='square', line=dict(color='white', width=2))
        ))
        
        fig_ef.update_layout(
            template="plotly_dark",
            height=600,
            title="Efficient Frontier",
            xaxis_title="Volatility %",
            yaxis_title="Return %",
            showlegend=True
        )
        
        st.plotly_chart(fig_ef, use_container_width=True)
        
        # Optimal weights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Optimal Sharpe Weights**")
            opt_df = pd.DataFrame({
                'Ticker': tickers,
                'Optimal %': optimal_sharpe_weights * 100,
                'Current %': weights * 100
            }).sort_values('Optimal %', ascending=False)
            opt_df = opt_df[opt_df['Optimal %'] > 0.5]
            st.dataframe(opt_df, hide_index=True)
        
        with col2:
            st.markdown("**Min Volatility Weights**")
            minvol_df = pd.DataFrame({
                'Ticker': tickers,
                'Min Vol %': min_vol_weights * 100,
                'Current %': weights * 100
            }).sort_values('Min Vol %', ascending=False)
            minvol_df = minvol_df[minvol_df['Min Vol %'] > 0.5]
            st.dataframe(minvol_df, hide_index=True)

    # TAB 2: Correlation Matrix
    with tab2:
        st.subheader("🔬 Correlation Matrix")
        
        corr_matrix = log_returns.corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig_corr.update_layout(
            template="plotly_dark",
            height=600,
            title="Asset Correlation Matrix"
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # High correlations
        st.markdown("**High Correlation Pairs (>0.7)**")
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    corr_pairs.append({
                        'Asset 1': corr_matrix.columns[i],
                        'Asset 2': corr_matrix.columns[j],
                        'Correlation': corr_val
                    })
        
        if corr_pairs:
            st.dataframe(pd.DataFrame(corr_pairs), hide_index=True)
        else:
            st.success("Good diversification - no extreme correlations")

    # TAB 3: Stress Testing
    with tab3:
        st.subheader("🌪️ Stress Testing")
        
        portfolio_beta = curr_std / 0.15
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            crash = st.slider("Market Crash %", -50, -10, -25, 5)
            bull = st.slider("Bull Rally %", 10, 100, 50, 10)
            st.metric("Portfolio Beta", f"{portfolio_beta:.2f}")
        
        with col2:
            scenarios = {
                f"Market Crash ({crash}%)": crash / 100 * portfolio_beta,
                f"Bull Rally (+{bull}%)": bull / 100 * portfolio_beta * 0.9,
            }
            
            names = list(scenarios.keys())
            impacts = [scenarios[s] for s in names]
            values = [total_value * (1 + i) - total_value for i in impacts]
            colors = ['#FF4B4B' if v < 0 else '#00CC96' for v in values]
            
            fig = go.Figure(go.Bar(
                y=names, x=values, orientation='h',
                marker=dict(color=colors),
                text=[f"₹{v:,.0f}" for v in values],
                textposition='outside'
            ))
            
            fig.update_layout(
                template="plotly_dark",
                height=300,
                title="Stress Test Impact",
                xaxis_title="P&L (₹)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk metrics
        st.markdown("**Risk Metrics**")
        col1, col2, col3, col4 = st.columns(4)
        
        var_95 = curr_std * np.sqrt(1/252) * 1.645
        with col1:
            st.metric("VaR (95%, 1D)", f"-{var_95 * 100:.2f}%")
        with col2:
            st.metric("CVaR (95%)", f"-{var_95 * 1.25 * 100:.2f}%")
        with col3:
            st.metric("Max Drawdown", f"-{curr_std * 2 * 100:.1f}%")
        with col4:
            st.metric("Beta", f"{portfolio_beta:.2f}")

    # TAB 4: Position Analysis
    with tab4:
        st.subheader("📊 Position Analysis")
        
        position_data = []
        for _, row in edited_df.iterrows():
            ticker = row['Ticker']
            current_price = current_prices.get(ticker, 0)
            position_value = row['Current_Value']
            invested = row['Shares'] * row['Avg Cost']
            pnl = position_value - invested
            pnl_pct = (pnl / invested) * 100 if invested > 0 else 0
            weight = (position_value / total_value) * 100
            
            if ticker in log_returns.columns:
                ann_return = log_returns[ticker].mean() * 252 * 100
                ann_vol = log_returns[ticker].std() * np.sqrt(252) * 100
            else:
                ann_return = ann_vol = 0
            
            position_data.append({
                'Ticker': ticker,
                'Shares': row['Shares'],
                'Current Price': current_price,
                'Value': position_value,
                'Weight %': weight,
                'P&L': pnl,
                'P&L %': pnl_pct,
                'Return %': ann_return,
                'Vol %': ann_vol
            })
        
        pos_df = pd.DataFrame(position_data).sort_values('Value', ascending=False)
        st.dataframe(pos_df, use_container_width=True)
        
        # Pie chart
        fig_pie = px.pie(
            pos_df, values='Value', names='Ticker',
            title='Portfolio Allocation',
            hole=0.4
        )
        fig_pie.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h3 style='background: linear-gradient(135deg, #60a5fa, #a78bfa); 
                   -webkit-background-clip: text; 
                   -webkit-text-fill-color: transparent;'>
            💎 QuantPro Portfolio Analytics
        </h3>
        <p style='color: #9ca3af;'>
            Powered by Modern Portfolio Theory • Yahoo Finance Data
        </p>
        <p style='color: #6b7280; font-size: 0.875rem;'>
            <strong>Disclaimer:</strong> For informational purposes only. Not financial advice.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
