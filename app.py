import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="StockSage AI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 StockSage AI")
st.caption("Multi-Market Stock Prediction System")
st.divider()

with st.sidebar:
    st.header("⚙️ Settings")
    market = st.selectbox("Market", ["US","India","Crypto"])
    options = {
        "US":     ["AAPL","MSFT","GOOGL","TSLA","NVDA","AMZN"],
        "India":  ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS"],
        "Crypto": ["BTC-USD","ETH-USD","BNB-USD","SOL-USD"]
    }
    ticker  = st.selectbox("Stock", options[market])
    custom  = st.text_input("Custom ticker")
    if custom:
        ticker = custom.upper()
    period  = st.selectbox("Period",
                ["1mo","3mo","6mo","1y","2y"], index=3)
    horizon = st.slider("Forecast Days", 1, 30, 1)
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

@st.cache_data(ttl=3600)
def get_data(ticker, period):
    try:
        import yfinance as yf
        df = yf.download(
            ticker, period=period,
            auto_adjust=True, progress=False
        )
        if df.empty:
            return pd.DataFrame()
        df.columns = [
            c[0] if isinstance(c, tuple) else c
            for c in df.columns
        ]
        close = df["Close"].squeeze()
        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        df["RSI"] = 100 - (100/(1+gain/(loss+1e-9)))
        # MACD
        e12 = close.ewm(span=12).mean()
        e26 = close.ewm(span=26).mean()
        df["MACD"]     = e12 - e26
        df["MACD_sig"] = df["MACD"].ewm(span=9).mean()
        # Bollinger
        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        df["BB_U"]  = sma + 2*std
        df["BB_L"]  = sma - 2*std
        df["SMA20"] = sma
        df["EMA20"] = close.ewm(span=20).mean()
        df["Ret"]   = close.pct_change()
        return df.dropna().reset_index()
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame()

tab1, tab2, tab3 = st.tabs([
    "📊 Chart","🔮 Predict","🔍 Screener"
])

with tab1:
    st.subheader(f"📊 {ticker}")
    with st.spinner("Loading data..."):
        df = get_data(ticker, period)
    if df.empty:
        st.error("No data found. Check ticker.")
    else:
        c  = float(df["Close"].iloc[-1])
        p  = float(df["Close"].iloc[-2])
        ch = (c-p)/p*100
        rsi = float(df["RSI"].iloc[-1])
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Price", f"${c:,.2f}", f"{ch:+.2f}%")
        m2.metric("High",  f"${float(df['High'].iloc[-1]):,.2f}")
        m3.metric("Low",   f"${float(df['Low'].iloc[-1]):,.2f}")
        m4.metric("RSI",   f"{rsi:.1f}",
            "⚠️OB" if rsi>70 else "💡OS" if rsi<30 else "✅OK")

        dfc = df.tail(90)
        dc  = "Date" if "Date" in dfc.columns else dfc.columns[0]
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.6,0.2,0.2],
            vertical_spacing=0.03
        )
        fig.add_trace(go.Candlestick(
            x=dfc[dc],
            open=dfc["Open"], high=dfc["High"],
            low=dfc["Low"],   close=dfc["Close"],
            increasing_line_color="#00FF88",
            decreasing_line_color="#FF4444",
            name="Price"
        ), row=1, col=1)
        if "SMA20" in dfc.columns:
            fig.add_trace(go.Scatter(
                x=dfc[dc], y=dfc["SMA20"],
                line=dict(color="#00BFFF",width=1.5),
                name="SMA20"
            ), row=1, col=1)
        if "BB_U" in dfc.columns:
            fig.add_trace(go.Scatter(
                x=dfc[dc], y=dfc["BB_U"],
                line=dict(color="violet",width=1,dash="dot"),
                name="BB+"
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=dfc[dc], y=dfc["BB_L"],
                line=dict(color="violet",width=1,dash="dot"),
                fill="tonexty",
                fillcolor="rgba(238,130,238,0.05)",
                name="BB-"
            ), row=1, col=1)
        vcols = [
            "#00FF88" if c>=o else "#FF4444"
            for c,o in zip(dfc["Close"],dfc["Open"])
        ]
        fig.add_trace(go.Bar(
            x=dfc[dc], y=dfc["Volume"],
            marker_color=vcols, name="Vol"
        ), row=2, col=1)
        if "RSI" in dfc.columns:
            fig.add_trace(go.Scatter(
                x=dfc[dc], y=dfc["RSI"],
                line=dict(color="gold",width=1.5),
                name="RSI"
            ), row=3, col=1)
            fig.add_hline(
                y=70,
                line=dict(color="red",dash="dash"),
                row=3, col=1
            )
            fig.add_hline(
                y=30,
                line=dict(color="green",dash="dash"),
                row=3, col=1
            )
        fig.update_layout(
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=580,
            margin=dict(l=0,r=0,t=20,b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader(f"🔮 Predict — {ticker}")
    if st.button("🚀 Run Prediction"):
        with st.spinner("Analysing..."):
            df2 = get_data(ticker, "6mo")
        if df2.empty:
            st.error("No data")
        else:
            close   = df2["Close"].squeeze().values
            current = float(close[-1])
            mom = float(close[-1]/close[-6]-1) \
                  if len(close)>5 else 0
            rsi2    = float(df2["RSI"].iloc[-1])
            rs      = -0.01 if rsi2>70 else \
                       0.01 if rsi2<30 else 0
            macd2   = float(df2["MACD"].iloc[-1])
            ms      = 0.005 if macd2>0 else -0.005
            sig     = (mom*0.5+rs+ms)*horizon
            pred    = current*(1+sig)
            ret     = (pred-current)/current*100
            c1,c2,c3 = st.columns(3)
            c1.metric("Current",   f"${current:,.2f}")
            c2.metric("Predicted", f"${pred:,.2f}",
                      f"{ret:+.2f}%")
            c3.metric("Signal",
                "🟢 BULLISH" if ret>0 else "🔴 BEARISH")
            st.info(
                f"RSI: {rsi2:.1f} | "
                f"Momentum: {mom*100:+.2f}% | "
                f"Days: {horizon}"
            )
    else:
        st.info("👆 Tap Run Prediction")

    st.divider()
    st.subheader("📅 7-Day Forecast")
    if st.button("Show 7-Day Table"):
        df3   = get_data(ticker, "6mo")
        close = df3["Close"].squeeze().values
        curr  = float(close[-1])
        rows  = []
        for h in range(1,8):
            mom = float(close[-1]/close[-6]-1) \
                  if len(close)>5 else 0
            rsi3 = float(df3["RSI"].iloc[-1])
            rs   = -0.01 if rsi3>70 else \
                    0.01 if rsi3<30 else 0
            macd3= float(df3["MACD"].iloc[-1])
            ms   = 0.005 if macd3>0 else -0.005
            sig  = (mom*0.5+rs+ms)*h
            pred = curr*(1+sig)
            ret  = (pred-curr)/curr*100
            rows.append({
                "Day":       f"+{h}",
                "Price":     f"${pred:,.2f}",
                "Return %":  f"{ret:+.2f}%",
                "Signal":    "🟢" if ret>0 else "🔴"
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True
        )

with tab3:
    st.subheader("🔍 Market Screener")
    sm = st.selectbox("Market",
                      ["US","India","Crypto"], key="sm")
    slists = {
        "US":    ["AAPL","MSFT","TSLA","NVDA","AMZN","META"],
        "India": ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS"],
        "Crypto":["BTC-USD","ETH-USD","BNB-USD","SOL-USD"]
    }
    if st.button("🔍 Scan Market"):
        results = []
        prog    = st.progress(0)
        total   = len(slists[sm])
        for i,t in enumerate(slists[sm]):
            prog.progress((i+1)/total)
            try:
                d = get_data(t,"1mo")
                if d.empty:
                    continue
                c_   = float(d["Close"].iloc[-1])
                p_   = float(d["Close"].iloc[-5]) \
                       if len(d)>5 else c_
                chg_ = (c_-p_)/p_*100
                rsi_ = float(d["RSI"].iloc[-1])
                sig_ = "🟢 Bullish" if rsi_<40 else \
                       "🔴 Bearish" if rsi_>65 else \
                       "🟡 Neutral"
                results.append({
                    "Ticker": t,
                    "Price":  f"${c_:,.2f}",
                    "5D Chg": f"{chg_:+.1f}%",
                    "RSI":    round(rsi_,1),
                    "Signal": sig_
                })
            except:
                pass
        prog.empty()
        if results:
            st.dataframe(
                pd.DataFrame(results),
                use_container_width=True
            )
        else:
            st.warning("No results")
    else:
        st.info("👆 Tap Scan Market")

st.divider()
st.caption(
    "📈 StockSage AI | College ML Project | "
    "Not financial advice"
      )
