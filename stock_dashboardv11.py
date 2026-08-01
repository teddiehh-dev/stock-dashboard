import os
import sys
import json
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors
import concurrent.futures

# --- 1. App UI Initialization ---
st.set_page_config(page_title="AI Stock Competitor Dashboard v12", layout="wide")
st.title("📈 Advanced Quantitative AI Stock Dashboard (v12)")
st.caption("Created by Teddie Hutchings | Upgraded with Multithreaded Parallel Processing")

# NEW: Brief description of the program
st.markdown("""
**Welcome to the Advanced Quantitative AI Stock Dashboard.** 
This tool leverages Google's Gemini AI to dynamically identify direct market competitors for any target stock. 
It utilizes multithreaded processing to rapidly fetch historical pricing, calculate technical indicators, 
compare key financial fundamentals, and summarize real-time market sentiment from global news feeds.
""")
st.markdown("---")

# --- 2. Cached Business Logic Functions ---
@st.cache_data(ttl=3600)
def discover_competitors_via_ai(stock_query):
    """Queries Gemini to discover tickers, company names, and market classification."""
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    else:
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return {"error": "GEMINI_API_KEY not found in environment variables or secrets."}

    client = genai.Client(api_key=api_key)
    
    competitor_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "target_stock": types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "company_name": types.Schema(type=types.Type.STRING),
                    "ticker": types.Schema(type=types.Type.STRING),
                    "description": types.Schema(type=types.Type.STRING),
                    "field_category": types.Schema(type=types.Type.STRING),
                },
                required=["company_name", "ticker", "description", "field_category"]
            ),
            "direct_competitors": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "company_name": types.Schema(type=types.Type.STRING),
                        "ticker": types.Schema(type=types.Type.STRING)
                    },
                    required=["company_name", "ticker"]
                )
            )
        },
        required=["target_stock", "direct_competitors"]
    )
    
    try:
        response = client.models.generate_content(
            model='gemini-3.5-flash', 
            contents=(
                f"Find the stock ticker symbol for {stock_query}. "
                f"Provide a brief description of what {stock_query} does, its core business model, and its primary sector (maximum 100 words). "
                f"Identify exactly 5 DIRECT COMPETITORS operating in the exact same specific field/industry as {stock_query} and find their tickers."
            ),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=competitor_schema,
                temperature=0.0 
            )
        )
        return json.loads(response.text)

    except errors.APIError as e:
        return {"error": f"Google AI API Error (Status {e.code})"}
    except Exception as e:
        return {"error": f"Failed to parse AI structure: {str(e)}"}


def _process_single_ticker(ticker, time_frame, company_name):
    """Processes a single ticker. Designed to be run in parallel on separate threads."""
    ticker = ticker.strip().upper()
    stock_obj = yf.Ticker(ticker)
    
    hist = stock_obj.history(period=time_frame)
    growth_val = 0.0
    
    if not hist.empty:
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
            
        if 'Close' in hist.columns:
            close_series = hist['Close']
            if isinstance(close_series, pd.DataFrame):
                close_series = close_series.iloc[:, 0]
                
            clean_close = close_series.dropna()
            
            if len(clean_close) > 0:
                hist['SMA20'] = clean_close.rolling(window=20).mean()
                hist['SMA50'] = clean_close.rolling(window=50).mean()
                
                delta = clean_close.diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                ema_gain = gain.ewm(com=13, adjust=False).mean()
                ema_loss = loss.ewm(com=13, adjust=False).mean()
                rs = ema_gain / ema_loss.replace(0, 1e-9)
                hist['RSI14'] = 100 - (100 / (1 + rs))
                
                start_p = float(clean_close.values[0])
                end_p = float(clean_close.values[-1])
                growth_val = ((end_p - start_p) / start_p) * 100 if start_p != 0 else 0.0

    try:
        info = stock_obj.info
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
        pe_ratio = info.get('trailingPE', 'N/A')
        market_cap = info.get('marketCap', 'N/A')
        avg_volume = info.get('averageVolume', info.get('averageDailyVolume10Day', 'N/A'))
        # NEW: Fetch Dividend Yield
        div_yield = info.get('dividendYield', 'N/A')
    except Exception:
        current_price, pe_ratio, market_cap, avg_volume, div_yield = 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'

    if current_price == 'N/A' and not hist.empty and 'clean_close' in locals() and len(clean_close) > 0:
        current_price = float(clean_close.values[-1])
        
    if isinstance(current_price, (int, float)): current_price = f"${current_price:,.2f}"
    if isinstance(pe_ratio, float): pe_ratio = f"{pe_ratio:.2f}"
    if isinstance(market_cap, (int, float)): market_cap = f"${market_cap:,}"
    if isinstance(avg_volume, (int, float)): avg_volume = f"{avg_volume:,}"
    
    # NEW: Format Dividend Yield as a percentage
    if isinstance(div_yield, float): 
        div_yield = f"{div_yield * 100:.2f}%"
    elif div_yield == 'N/A' or div_yield is None:
        div_yield = "N/A"

    # NEW: Added Dividend Yield to the data dictionary
    fund_row = {
        "Ticker": ticker,
        "Company Name": company_name,
        "Current Price": current_price,
        "P/E Ratio": pe_ratio,
        "Market Cap": market_cap,
        "Avg Volume": avg_volume,
        "Dividend Yield": div_yield 
    }
    
    return ticker, hist, growth_val, fund_row


@st.cache_data(ttl=1800)
def fetch_financial_market_data(tickers_list, ticker_to_name_map, time_frame):
    """Orchestrates multithreaded data fetching for extreme performance."""
    historical_dfs = {}
    growth_metrics = {}
    fundamental_data = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(_process_single_ticker, ticker, time_frame, ticker_to_name_map.get(ticker, "Unknown"))
            for ticker in tickers_list
        ]
        
        for future in concurrent.futures.as_completed(futures):
            ticker, hist, growth_val, fund_row = future.result()
            
            if not hist.empty:
                historical_dfs[ticker] = hist
                growth_metrics[ticker] = growth_val
            fundamental_data.append(fund_row)

    return historical_dfs, growth_metrics, fundamental_data


@st.cache_data(ttl=3600)
def fetch_ai_summarized_news(ticker_symbol, context_type):
    """Downloads fresh market news feeds and translates them to sentiment capsules."""
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    else:
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        
    client = genai.Client(api_key=api_key)
    raw_news = yf.Ticker(ticker_symbol).news
    summarized_stories = []
    
    if not raw_news:
        return summarized_stories

    for story in raw_news[:3]:
        content = story.get('content', story)
        title = content.get('title', 'No Title Available')
        
        click_data = content.get('clickThroughUrl', {})
        link = click_data.get('url', content.get('url', '#')) if isinstance(click_data, dict) else click_data
        if not link: link = '#'
            
        provider_data = content.get('provider', {})
        publisher = provider_data.get('displayName', content.get('publisher', 'Unknown Source')) if isinstance(provider_data, dict) else provider_data
        
        try:
            prompt = (
                f"Analyze this financial news story headline: '{title}' published by {publisher}.\n"
                f"1. Summarize the key implications for {context_type} in exactly 100 words or less.\n"
                "2. Conclude the summary by explicitly classifying the market sentiment indicator as either [BULLISH], [BEARISH], or [NEUTRAL].\n"
                "Provide only the summary paragraph and indicator, no conversational fluff."
            )
            ai_resp = client.models.generate_content(model='gemini-3.5-flash', contents=prompt)
            summary_text = ai_resp.text.strip()
        except Exception:
            summary_text = "AI summary temporarily unavailable due to request limits. View original link."
        
        summarized_stories.append({"title": title, "link": link, "summary": summary_text, "source": publisher})
        
    return summarized_stories


# --- 3. Sidebar UI Layout Component ---
with st.sidebar:
    st.header("Dashboard Configuration")
    stock_choice = st.text_input("Enter Target Stock:", placeholder="e.g., Apple, Tesla, NVIDIA")
    time_frame = st.selectbox("Select Time Frame:", options=["1mo", "3mo", "6mo", "1y", "5y"], index=2) 
    analyze_button = st.button("Run Competitor Analysis", type="primary")

# --- 4. Pipeline Execution Controller ---
if analyze_button and stock_choice:
    with st.spinner('AI is analyzing the industry and mapping peer groups...'):
        ai_data = discover_competitors_via_ai(stock_choice)
        
        if "error" in ai_data:
            st.error(ai_data["error"])
            st.stop()
            
        target_ticker = ai_data['target_stock']['ticker'].upper()
        ticker_to_name = {target_ticker: ai_data['target_stock']['company_name']}
        ai_discovered_tickers = [target_ticker]
        
        for stock in ai_data.get("direct_competitors", []):
            ticker_to_name[stock["ticker"].upper()] = stock["company_name"]
            ai_discovered_tickers.append(stock["ticker"].upper())
            
        st.session_state['ai_discovered_tickers'] = ai_discovered_tickers
        st.session_state['ticker_to_name'] = ticker_to_name
        st.session_state['target_ticker'] = target_ticker
        st.session_state['target_desc'] = ai_data['target_stock'].get('description', '')
        st.session_state['field_category'] = ai_data['target_stock'].get('field_category', 'Unknown Sector')


# --- 5. Interactive Display Component ---
if 'target_ticker' in st.session_state:
    target_ticker = st.session_state['target_ticker']
    ticker_to_name = st.session_state['ticker_to_name']
    field_category = st.session_state['field_category']
    
    st.subheader(f"ℹ️ About {ticker_to_name[target_ticker]} ({target_ticker})")
    st.markdown(f"**Specific Industry Field / Category:** `{field_category}`")
    st.info(st.session_state['target_desc'])
    st.markdown("---")

    st.subheader("📈 Chart Visualization Controls")
    control_col1, control_col2, control_col3 = st.columns([2, 3, 2])
    
    with control_col1:
        chart_mode = st.radio("Select Graph Metric View:", options=["Percentage Growth (%)", "Stock Price ($)"], horizontal=True)
        
    with control_col2:
        selected_tickers = st.multiselect(
            "Select Tickers to Compare:",
            options=st.session_state['ai_discovered_tickers'],
            default=st.session_state['ai_discovered_tickers']
        )
        
    with control_col3:
        st.markdown("**Technical Overlays (Target Ticker Only):**")
        show_sma20 = st.checkbox("Show 20-Day SMA")
        show_sma50 = st.checkbox("Show 50-Day SMA")
        show_rsi = st.checkbox("Show RSI Subplot Panel")

    with st.spinner('Fetching live market data and computing technicals...'):
        historical_dfs, growth_metrics, fundamental_data = fetch_financial_market_data(
            selected_tickers, ticker_to_name, time_frame
        )
        
    filtered_fundamentals = [row for row in fundamental_data if row["Ticker"] in selected_tickers]

    st.subheader("📊 Performance Overview & Direct Competitors Key")
    if growth_metrics:
        visible_metrics = {k: v for k, v in growth_metrics.items() if k in selected_tickers}
        cols = st.columns(max(len(visible_metrics), 1))
        for i, ticker in enumerate(visible_metrics.keys()):
            with cols[i]:
                is_target = " (Target)" if ticker == target_ticker else ""
                st.metric(
                    label=f"{ticker_to_name.get(ticker, ticker)}{is_target}",
                    value=f"{visible_metrics[ticker]:.2f}%",
                    delta=f"{visible_metrics[ticker]:.2f}%"
                )
    
    st.markdown("### 📋 Live Key Fundamentals")
    
    with st.expander("📘 Financial Terminology Guide (What do these rows mean?)"):
        st.markdown("""
        * **P/E Ratio (Price-to-Earnings):** A primary valuation multiplier computed by dividing the current share price by its trailing 12-month earnings per share. High numbers indicate investors expect major future growth or that the stock is currently expensive.
        * **Market Cap (Market Capitalization):** The total aggregate net market dollar value of the firm's outstanding equity. It designates the total operational scale tier of the corporation (e.g., Mega Cap, Large Cap).
        * **Avg Volume (Average Trading Volume):** The standard rolling quantity of shares transacted on public markets daily. High liquidity indexes allow capital deployment changes without creating adverse slippage or flash volatility.
        * **Dividend Yield:** A financial ratio that shows how much a company pays out in dividends each year relative to its stock price. A value of N/A usually means the company does not currently pay a dividend.
        """)
        
    st.table(filtered_fundamentals)
    st.markdown("---")

    if show_rsi and target_ticker in historical_dfs and 'RSI14' in historical_dfs[target_ticker].columns:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
        rsi_subplot_active = True
    else:
        fig = go.Figure()
        rsi_subplot_active = False

    yaxis_title_string = "Stock Close Price (USD)"
    
    for ticker in selected_tickers:
        if ticker in historical_dfs:
            df = historical_dfs[ticker]
            close_col = df['Close']
            if isinstance(close_col, pd.DataFrame): 
                close_col = close_col.iloc[:, 0]
            clean_close_col = close_col.dropna()
            
            if len(clean_close_col) > 0:
                if chart_mode == "Percentage Growth (%)":
                    initial_price = float(clean_close_col.values[0])
                    y_data = ((close_col - initial_price) / initial_price) * 100 if initial_price != 0 else close_col * 0.0
                    yaxis_title_string = "Return Value Growth (%)"
                else:
                    y_data = close_col
                    yaxis_title_string = "Stock Close Price (USD)"
                    
                trace = go.Scatter(
                    x=df.index, y=y_data, mode='lines',
                    name=f"{ticker} ({ticker_to_name.get(ticker, ticker)})",
                    line=dict(width=3 if ticker == target_ticker else 1.5)
                )
                
                if rsi_subplot_active:
                    fig.add_trace(trace, row=1, col=1)
                else:
                    fig.add_trace(trace)

    if target_ticker in historical_dfs and chart_mode == "Stock Price ($)":
        target_df = historical_dfs[target_ticker]
        
        if show_sma20 and 'SMA20' in target_df.columns:
            sma20_trace = go.Scatter(x=target_df.index, y=target_df['SMA20'], mode='lines', name=f'{target_ticker} 20-Day SMA', line=dict(dash='dash', width=1.5))
            fig.add_trace(sma20_trace, row=1, col=1) if rsi_subplot_active else fig.add_trace(sma20_trace)
            
        if show_sma50 and 'SMA50' in target_df.columns:
            sma50_trace = go.Scatter(x=target_df.index, y=target_df['SMA50'], mode='lines', name=f'{target_ticker} 50-Day SMA', line=dict(dash='dot', width=1.5))
            fig.add_trace(sma50_trace, row=1, col=1) if rsi_subplot_active else fig.add_trace(sma50_trace)

    if rsi_subplot_active:
        target_df = historical_dfs[target_ticker]
        fig.add_trace(go.Scatter(x=target_df.index, y=target_df['RSI14'], mode='lines', name=f'{target_ticker} RSI (14)', line=dict(width=1.5)), row=2, col=1)
        
        fig.add_hline(y=70, line_dash="dash", line_width=1, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_width=1, row=2, col=1)
        fig.update_yaxes(title_text="RSI Value", range=[10, 90], row=2, col=1)

    if rsi_subplot_active:
        fig.update_layout(title=f"Comparative Performance & Market Indicators ({time_frame})", xaxis2_title="Date", yaxis_title=yaxis_title_string, hovermode="x unified", legend_title="Dashboard Key", height=650)
    else:
        fig.update_layout(title=f"Comparative Performance over Time Frame ({time_frame})", xaxis_title="Date", yaxis_title=yaxis_title_string, hovermode="x unified", legend_title="Dashboard Key", height=500)
        
    st.plotly_chart(fig, use_container_width=True)
    
    if rsi_subplot_active:
        st.info("""
        📊 **RSI (Relative Strength Index) Interpretation Guide:**
        * **RSI > 70 [Overbought Regime]:** Indicates asset momentum has expanded rapidly. Historically shows that the stock is potentially overvalued short-term and susceptible to local consolidation or localized technical sell-offs.
        * **RSI < 30 [Oversold Regime]:** Indicates heavy historical sell pressure that may be mathematically overextended. Suggests selling exhaustion where the underlying security may bounce back.
        * **30 to 70 [Neutral Equilibrium Bounds]:** Indicates regular price activity without overextended velocity changes.
        """)
        
    st.markdown("---")

    st.subheader("📰 Live AI Sentiment News Feed")
    news_col1, news_col2 = st.columns(2)
    
    with st.spinner('Aggregating and analyzing global sentiment...'):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_target = executor.submit(fetch_ai_summarized_news, target_ticker, target_ticker)
            future_macro = executor.submit(fetch_ai_summarized_news, target_ticker, field_category)
            
            target_news = future_target.result()
            macro_news = future_macro.result()
    
    with news_col1:
        st.markdown(f"#### 🎯 Target Stock Intelligence: **{target_ticker}**")
        if target_news:
            for story in target_news:
                st.markdown(f"🔗 **[{story['title']}]({story['link']})**")
                st.caption(f"Source: {story['source']}")
                st.info(story['summary'])
        else:
            st.write("No recent targeted equities market stories detected.")

    with news_col2:
        st.markdown(f"#### 🌐 Macro Field Intelligence: **{field_category}**")
        if macro_news:
            for story in macro_news:
                st.markdown(f"🔗 **[{story['title']}]({story['link']})**")
                st.caption(f"Source: {story['source']}")
                st.success(story['summary'])
        else:
            st.write("No matching vertical industry structural updates discovered.")
