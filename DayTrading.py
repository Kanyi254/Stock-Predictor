import os
import datetime as _dt
import requests
from io import StringIO

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import talib as ta
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(
    page_title="Swing Trader Terminal",
    page_icon="▸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&display=swap');
:root {
    --font-mono:'IBM Plex Mono','Courier New',monospace;
    --bg:#070707; --bg-panel:#0d0d0d; --bg-card:#0f0f0f; --bg-el:#141414;
    --border:#1e1e1e; --border2:#2a2a2a;
    --amber:#ff9f00; --amber-dim:#cc7a00; --amber-faint:rgba(255,159,0,0.06);
    --green:#00d084; --red:#ff3b5c; --blue:#0088ff; --cyan:#00c8e0; --purple:#c084fc;
    --tx:#e8e0d0; --tx2:#8a8070; --tx3:#504840;
}
html,body,[class*="css"]{font-family:var(--font-mono)!important;background:var(--bg)!important;color:var(--tx)!important;}
.stApp{background:var(--bg)!important;}
section[data-testid="stSidebar"]{background:var(--bg-panel)!important;border-right:1px solid var(--border)!important;}
section[data-testid="stSidebar"] *{font-family:var(--font-mono)!important;}
section[data-testid="stSidebar"] label{color:var(--tx2)!important;font-size:0.67rem!important;letter-spacing:0.07em!important;}
section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3{
    color:var(--amber)!important;font-size:0.58rem!important;letter-spacing:0.2em!important;
    text-transform:uppercase!important;border-bottom:1px solid var(--border)!important;
    padding-bottom:4px!important;margin-top:14px!important;}
[data-testid="metric-container"]{background:var(--bg-card)!important;border:1px solid var(--border)!important;
    border-top:2px solid var(--amber)!important;border-radius:0!important;padding:10px 14px!important;}
[data-testid="metric-container"] [data-testid="stMetricLabel"]{color:var(--tx2)!important;font-size:0.56rem!important;
    letter-spacing:0.14em!important;text-transform:uppercase!important;font-family:var(--font-mono)!important;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{color:var(--amber)!important;font-size:1.15rem!important;
    font-weight:600!important;font-family:var(--font-mono)!important;}
[data-testid="metric-container"] [data-testid="stMetricDelta"]{font-size:0.7rem!important;font-family:var(--font-mono)!important;}
.stTabs [data-baseweb="tab-list"]{background:var(--bg-panel)!important;border-bottom:1px solid var(--border)!important;gap:0!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--tx2)!important;border:none!important;
    border-right:1px solid var(--border)!important;border-radius:0!important;
    font-size:0.62rem!important;letter-spacing:0.12em!important;text-transform:uppercase!important;
    font-family:var(--font-mono)!important;padding:7px 16px!important;}
.stTabs [aria-selected="true"]{background:var(--amber-faint)!important;color:var(--amber)!important;border-bottom:2px solid var(--amber)!important;}
.stAlert{background:var(--bg-card)!important;border:1px solid var(--border2)!important;
    border-left:3px solid var(--amber)!important;border-radius:0!important;
    color:var(--tx)!important;font-size:0.72rem!important;font-family:var(--font-mono)!important;}
.stButton>button{background:var(--amber)!important;color:#000!important;border:none!important;
    border-radius:0!important;font-family:var(--font-mono)!important;font-size:0.67rem!important;
    letter-spacing:0.1em!important;text-transform:uppercase!important;font-weight:700!important;}
.stDataFrame thead tr th{background:var(--bg-el)!important;color:var(--amber)!important;
    font-size:0.61rem!important;letter-spacing:0.1em!important;text-transform:uppercase!important;}
.stDataFrame tbody tr td{color:var(--tx)!important;font-size:0.69rem!important;}
hr{border-color:var(--border)!important;}
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--border2);}
div[data-testid="stExpander"]{border:1px solid var(--border)!important;border-radius:0!important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ASSET CATALOGUE  (Forex + Crypto first, NSE last)
# ─────────────────────────────────────────────────────────────────────────────
ASSET_SYMBOLS = {
    'Forex': {
        'EUR/USD':'EURUSD=X','GBP/USD':'GBPUSD=X','USD/JPY':'USDJPY=X',
        'GBP/JPY':'GBPJPY=X','EUR/JPY':'EURJPY=X','AUD/USD':'AUDUSD=X',
        'USD/CAD':'USDCAD=X','USD/CHF':'USDCHF=X','NZD/USD':'NZDUSD=X',
        'EUR/GBP':'EURGBP=X','USD/KES':'USDKES=X','EUR/KES':'EURKES=X',
    },
    'Crypto': {
        'Bitcoin':'BTC-USD','Ethereum':'ETH-USD','Solana':'SOL-USD',
        'XRP':'XRP-USD','BNB':'BNB-USD','Cardano':'ADA-USD',
        'Dogecoin':'DOGE-USD','Polkadot':'DOT-USD','Avalanche':'AVAX-USD',
    },
    'Commodities': {
        'Gold':'GC=F','Silver':'SI=F','Crude Oil (WTI)':'CL=F',
        'Brent Oil':'BZ=F','Natural Gas':'NG=F','Copper':'HG=F',
    },
    'US Stocks': {
        'SPY (S&P 500)':'SPY','QQQ (Nasdaq)':'QQQ','Apple':'AAPL',
        'NVIDIA':'NVDA','Tesla':'TSLA','Microsoft':'MSFT','Meta':'META',
    },
    'Kenyan Stocks (NSE)': {
        'Safaricom':'SCOM.NR','Equity Group':'EQTY.NR','KCB Group':'KCB.NR',
        'EABL':'EABL.NR','Co-op Bank':'COOP.NR','ABSA Kenya':'ABSA.NR',
        'Stanbic':'CFC.NR','Kenya Airways':'KQ.NR','DTB':'DTK.NR',
        'Jubilee Holdings':'JUB.NR','Nation Media':'NMG.NR',
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_nse(symbol, start_date, end_date):
    sym = symbol.upper().replace('.NR','')
    hdrs = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    # afx.kwayisi.org
    try:
        r = requests.get(f'https://afx.kwayisi.org/nse/{sym.lower()}.html', headers=hdrs, timeout=12)
        if r.status_code == 200:
            for t in pd.read_html(StringIO(r.text)):
                cols = [str(c).lower() for c in t.columns]
                if any('date' in c for c in cols) and any('close' in c for c in cols):
                    ren = {}
                    for c in t.columns:
                        cl = str(c).lower()
                        if 'date' in cl: ren[c]='Date'
                        elif 'open' in cl: ren[c]='Open'
                        elif 'high' in cl: ren[c]='High'
                        elif 'low'  in cl: ren[c]='Low'
                        elif 'close' in cl: ren[c]='Close'
                        elif 'vol'  in cl: ren[c]='Volume'
                    t = t.rename(columns=ren)
                    if 'Date' in t.columns and 'Close' in t.columns:
                        t['Date'] = pd.to_datetime(t['Date'], dayfirst=True, errors='coerce')
                        t = t.dropna(subset=['Date']).set_index('Date').sort_index()
                        t = t.loc[str(start_date):str(end_date)]
                        for col in ['Open','High','Low']:
                            if col not in t.columns: t[col]=t['Close']
                        if 'Volume' not in t.columns: t['Volume']=0
                        for col in ['Open','High','Low','Close','Volume']:
                            t[col] = pd.to_numeric(t[col], errors='coerce')
                        t['Adj Close'] = t['Close']
                        t = t[['Open','High','Low','Close','Adj Close','Volume']].dropna(subset=['Close'])
                        if len(t) > 5: return t
    except Exception: pass
    # stooq fallback
    try:
        sd = start_date.strftime('%Y%m%d') if hasattr(start_date,'strftime') else str(start_date).replace('-','')
        ed = end_date.strftime('%Y%m%d')   if hasattr(end_date,'strftime')   else str(end_date).replace('-','')
        r = requests.get(f'https://stooq.com/q/d/l/?s={sym.lower()}.ke&d1={sd}&d2={ed}&i=d', headers=hdrs, timeout=12)
        if r.status_code == 200 and 'Date' in r.text[:100]:
            t = pd.read_csv(StringIO(r.text))
            t['Date'] = pd.to_datetime(t['Date'], errors='coerce')
            t = t.dropna(subset=['Date']).set_index('Date').sort_index()
            t.columns = [c.strip().title() for c in t.columns]
            if 'Close' in t.columns and len(t) > 5:
                for col in ['Open','High','Low']:
                    if col not in t.columns: t[col]=t['Close']
                if 'Volume' not in t.columns: t['Volume']=0
                t['Adj Close'] = t['Close']
                return t[['Open','High','Low','Close','Adj Close','Volume']].dropna(subset=['Close'])
    except Exception: pass
    return pd.DataFrame()


@st.cache_data(ttl=60)
def download_data(ticker, start_date, end_date, interval, asset_category=''):
    today = _dt.date.today()
    try:
        if asset_category == 'Kenyan Stocks (NSE)' or ticker.endswith('.NR'):
            df = _fetch_nse(ticker, start_date, end_date)
            if df.empty:
                st.warning(f"⚠️ No NSE data for **{ticker}**. Upload a CSV from your broker or nse.co.ke below.")
            return df
        intraday_max = {'1h':729,'4h':729,'30m':59,'15m':59,'5m':59,'1m':6}
        if interval in intraday_max:
            earliest = today - _dt.timedelta(days=intraday_max[interval])
            if start_date < earliest: start_date = earliest
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval,
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_convert('UTC').tz_localize(None)
        return df
    except Exception as e:
        st.error(f"Download failed: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
def add_indicators(df):
    d = df.copy()
    pc = 'Adj Close' if 'Adj Close' in d.columns else 'Close'
    if pc in d.columns:
        d['EMA9']  = ta.EMA(d[pc], timeperiod=9)
        d['EMA21'] = ta.EMA(d[pc], timeperiod=21)
        d['EMA50'] = ta.EMA(d[pc], timeperiod=50)
        d['EMA200']= ta.EMA(d[pc], timeperiod=200)
        d['SMA20'] = ta.SMA(d[pc], timeperiod=20)
        d['RSI14'] = ta.RSI(d[pc], timeperiod=14)
        d['RSI9']  = ta.RSI(d[pc], timeperiod=9)
        macd, sig, hist = ta.MACD(d[pc], 12, 26, 9)
        d['MACD']=macd; d['MACD_Signal']=sig; d['MACD_Hist']=hist
        u,m,l = ta.BBANDS(d[pc], timeperiod=20)
        d['BB_Upper']=u; d['BB_Middle']=m; d['BB_Lower']=l
        d['BB_Width']=(u-l)/m
        d['BB_Pct'] = (d[pc]-l)/(u-l)   # 0=at lower, 1=at upper

    if all(c in d.columns for c in ['High','Low','Close']):
        d['ATR']  = ta.ATR(d['High'],d['Low'],d['Close'], timeperiod=14)
        d['ATR50']= ta.ATR(d['High'],d['Low'],d['Close'], timeperiod=50)
        d['ADX']  = ta.ADX(d['High'],d['Low'],d['Close'], timeperiod=14)
        d['DI_plus'] = ta.PLUS_DI(d['High'],d['Low'],d['Close'], timeperiod=14)
        d['DI_minus']= ta.MINUS_DI(d['High'],d['Low'],d['Close'], timeperiod=14)
        d['SAR']  = ta.SAR(d['High'],d['Low'], acceleration=0.02, maximum=0.2)
        k,dk = ta.STOCHF(d['High'],d['Low'],d['Close'], fastk_period=14, fastd_period=3)
        d['STOCH_K']=k; d['STOCH_D']=dk
        # Ichimoku (basic)
        high9 = d['High'].rolling(9).max(); low9  = d['Low'].rolling(9).min()
        high26= d['High'].rolling(26).max(); low26 = d['Low'].rolling(26).min()
        d['ICH_Tenkan']  = (high9  + low9)  / 2
        d['ICH_Kijun']   = (high26 + low26) / 2
        d['ICH_SpanA']   = ((d['ICH_Tenkan'] + d['ICH_Kijun']) / 2).shift(26)
        high52 = d['High'].rolling(52).max(); low52 = d['Low'].rolling(52).min()
        d['ICH_SpanB']   = ((high52 + low52) / 2).shift(26)

        # VWAP (daily reset)
        if 'Volume' in d.columns:
            tp = (d['High']+d['Low']+d['Close'])/3
            if isinstance(d.index, pd.DatetimeIndex):
                d['_dt'] = d.index.date
                d['VWAP'] = (tp*d['Volume']).groupby(d['_dt']).cumsum() / d['Volume'].groupby(d['_dt']).cumsum().replace(0,np.nan)
                d.drop(columns=['_dt'], inplace=True)
            else:
                d['VWAP'] = (tp*d['Volume']).cumsum() / d['Volume'].cumsum()

    if 'Adj Close' not in d.columns and 'Close' in d.columns:
        d['Adj Close'] = d['Close']
    return d


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHTED SIGNAL ENGINE
# ─────────────────────────────────────────────────────────────────────────────
# Weights reflect importance for 1h-4h swing trading
SIGNAL_WEIGHTS = {
    'macd_cross': 2.0,    # trend-following, high weight
    'ema_cross':  2.0,    # EMA9/21 cross, strong for swing
    'adx_trend':  1.5,    # trend strength confirms
    'rsi':        1.5,    # momentum
    'ichimoku':   1.5,    # above/below cloud
    'vwap':       1.0,    # intraday bias
    'stoch':      1.0,    # shorter momentum
    'bb':         0.5,    # mean reversion only
    'sar':        0.5,    # lagging, lower weight
}
MAX_SCORE = sum(SIGNAL_WEIGHTS.values())  # 13.0

def compute_signals(df):
    d = df.copy()
    pc = 'Adj Close' if 'Adj Close' in d.columns else 'Close'

    def safe_cross_up(a, b):
        return (a > b) & (a.shift(1) <= b.shift(1))
    def safe_cross_dn(a, b):
        return (a < b) & (a.shift(1) >= b.shift(1))

    sigs = {}

    # MACD cross (weight 2.0)
    if 'MACD' in d.columns:
        sigs['macd_cross_buy']  = safe_cross_up(d['MACD'], d['MACD_Signal'])
        sigs['macd_cross_sell'] = safe_cross_dn(d['MACD'], d['MACD_Signal'])
    else:
        sigs['macd_cross_buy'] = sigs['macd_cross_sell'] = pd.Series(False, index=d.index)

    # EMA 9/21 cross (weight 2.0)
    if 'EMA9' in d.columns and 'EMA21' in d.columns:
        sigs['ema_cross_buy']  = safe_cross_up(d['EMA9'], d['EMA21'])
        sigs['ema_cross_sell'] = safe_cross_dn(d['EMA9'], d['EMA21'])
    else:
        sigs['ema_cross_buy'] = sigs['ema_cross_sell'] = pd.Series(False, index=d.index)

    # ADX trend direction (weight 1.5) — DI+/DI- cross when ADX > 20
    if 'DI_plus' in d.columns:
        strong = d['ADX'] > 20
        sigs['adx_trend_buy']  = safe_cross_up(d['DI_plus'], d['DI_minus']) & strong
        sigs['adx_trend_sell'] = safe_cross_dn(d['DI_plus'], d['DI_minus']) & strong
    else:
        sigs['adx_trend_buy'] = sigs['adx_trend_sell'] = pd.Series(False, index=d.index)

    # RSI (weight 1.5) — cross of 50 line (trend confirmation)
    if 'RSI14' in d.columns:
        sigs['rsi_buy']  = safe_cross_up(d['RSI14'], pd.Series(50, index=d.index))
        sigs['rsi_sell'] = safe_cross_dn(d['RSI14'], pd.Series(50, index=d.index))
    else:
        sigs['rsi_buy'] = sigs['rsi_sell'] = pd.Series(False, index=d.index)

    # Ichimoku — price vs cloud (weight 1.5)
    if 'ICH_SpanA' in d.columns and 'ICH_SpanB' in d.columns:
        cloud_top = d[['ICH_SpanA','ICH_SpanB']].max(axis=1)
        cloud_bot = d[['ICH_SpanA','ICH_SpanB']].min(axis=1)
        sigs['ichimoku_buy']  = safe_cross_up(d[pc], cloud_top)
        sigs['ichimoku_sell'] = safe_cross_dn(d[pc], cloud_bot)
    else:
        sigs['ichimoku_buy'] = sigs['ichimoku_sell'] = pd.Series(False, index=d.index)

    # VWAP cross (weight 1.0)
    if 'VWAP' in d.columns:
        sigs['vwap_buy']  = safe_cross_up(d[pc], d['VWAP'])
        sigs['vwap_sell'] = safe_cross_dn(d[pc], d['VWAP'])
    else:
        sigs['vwap_buy'] = sigs['vwap_sell'] = pd.Series(False, index=d.index)

    # Stochastic (weight 1.0)
    if 'STOCH_K' in d.columns:
        sigs['stoch_buy']  = (d['STOCH_K'] < 25) & (d['STOCH_K'].shift(1) >= 25)
        sigs['stoch_sell'] = (d['STOCH_K'] > 75) & (d['STOCH_K'].shift(1) <= 75)
    else:
        sigs['stoch_buy'] = sigs['stoch_sell'] = pd.Series(False, index=d.index)

    # Bollinger (weight 0.5)
    if 'BB_Lower' in d.columns:
        sigs['bb_buy']  = d['Close'] < d['BB_Lower']
        sigs['bb_sell'] = d['Close'] > d['BB_Upper']
    else:
        sigs['bb_buy'] = sigs['bb_sell'] = pd.Series(False, index=d.index)

    # SAR (weight 0.5)
    if 'SAR' in d.columns:
        sigs['sar_buy']  = safe_cross_up(d['Close'], d['SAR'])
        sigs['sar_sell'] = safe_cross_dn(d['Close'], d['SAR'])
    else:
        sigs['sar_buy'] = sigs['sar_sell'] = pd.Series(False, index=d.index)

    for k, v in sigs.items():
        d[f'sig_{k}'] = v.astype(bool)

    # Weighted scores
    buy_score  = pd.Series(0.0, index=d.index)
    sell_score = pd.Series(0.0, index=d.index)
    for sig_name, weight in SIGNAL_WEIGHTS.items():
        bk = f'sig_{sig_name}_buy'
        sk = f'sig_{sig_name}_sell'
        if bk in d.columns: buy_score  = buy_score  + d[bk].astype(float) * weight
        if sk in d.columns: sell_score = sell_score + d[sk].astype(float) * weight

    d['buy_score']  = buy_score
    d['sell_score'] = sell_score
    d['net_score']  = buy_score - sell_score
    d['buy_pct']    = (buy_score  / MAX_SCORE * 100).clip(0, 100)
    d['sell_pct']   = (sell_score / MAX_SCORE * 100).clip(0, 100)

    # Threshold: 30% of max score = actionable signal
    d['signal_buy']  = buy_score  >= MAX_SCORE * 0.30
    d['signal_sell'] = sell_score >= MAX_SCORE * 0.30
    return d


def detect_candle_patterns(df):
    o,h,l,c = df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values
    patterns = {
        'Hammer':ta.CDLHAMMER(o,h,l,c),
        'Inv Hammer':ta.CDLINVERTEDHAMMER(o,h,l,c),
        'Engulfing':ta.CDLENGULFING(o,h,l,c),
        'Morning Star':ta.CDLMORNINGSTAR(o,h,l,c,penetration=0),
        'Piercing':ta.CDLPIERCING(o,h,l,c),
        '3 White Sol':ta.CDL3WHITESOLDIERS(o,h,l,c),
        'Dragonfly Doji':ta.CDLDRAGONFLYDOJI(o,h,l,c),
        'Hanging Man':ta.CDLHANGINGMAN(o,h,l,c),
        'Shooting Star':ta.CDLSHOOTINGSTAR(o,h,l,c),
        'Evening Star':ta.CDLEVENINGSTAR(o,h,l,c,penetration=0),
        'Dark Cloud':ta.CDLDARKCLOUDCOVER(o,h,l,c,penetration=0),
        '3 Black Crows':ta.CDL3BLACKCROWS(o,h,l,c),
        'Gravestone Doji':ta.CDLGRAVESTONEDOJI(o,h,l,c),
        'Doji':ta.CDLDOJI(o,h,l,c),
        'Harami':ta.CDLHARAMI(o,h,l,c),
        'Marubozu':ta.CDLMARUBOZU(o,h,l,c),
    }
    return {nm: int(arr[-1]) for nm, arr in patterns.items() if len(arr)>0}


def detect_sr(df, window=10, n=4):
    highs = df['High'].rolling(window=window, center=True).max()
    lows  = df['Low'].rolling(window=window, center=True).min()
    res = df['High'][df['High']==highs].dropna().values
    sup = df['Low'][df['Low']==lows].dropna().values
    def cluster(lvls, tol=0.003):
        if not len(lvls): return []
        lvls = sorted(set(lvls)); out=[lvls[0]]
        for lv in lvls[1:]:
            if (lv-out[-1])/out[-1]>tol: out.append(lv)
        return out
    cur = df['Close'].iloc[-1]
    r = sorted(cluster(res), key=lambda x: abs(x-cur))[:n]
    s = sorted(cluster(sup), key=lambda x: abs(x-cur))[:n]
    return s, r


# ─────────────────────────────────────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────────────────────────────────────
def build_chart(df, title, indicators, show_signals, show_sr, show_ich,
                interval, entry_zones=None):
    pc = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    sig_df = compute_signals(df) if show_signals else df
    s_lvls, r_lvls = detect_sr(df) if show_sr else ([],[])

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.54,0.17,0.16,0.13], vertical_spacing=0.02,
        specs=[[{"secondary_y":False}]]*4)

    # Ichimoku cloud
    if show_ich and 'ICH_SpanA' in df.columns:
        for i in range(len(df)-1):
            if pd.isna(df['ICH_SpanA'].iloc[i]) or pd.isna(df['ICH_SpanB'].iloc[i]): continue
            a,b = df['ICH_SpanA'].iloc[i], df['ICH_SpanB'].iloc[i]
            col = 'rgba(0,208,132,0.06)' if a>=b else 'rgba(255,59,92,0.06)'
        # Plot as filled area
        fig.add_trace(go.Scatter(x=df.index, y=df['ICH_SpanA'], mode='lines',
            name='SPAN A', line=dict(color='rgba(0,208,132,0.3)',width=1),showlegend=False), row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['ICH_SpanB'], mode='lines',
            name='SPAN B', line=dict(color='rgba(255,59,92,0.3)',width=1),
            fill='tonexty', fillcolor='rgba(100,100,100,0.05)', showlegend=False), row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['ICH_Tenkan'], mode='lines',
            name='TENKAN', line=dict(color='rgba(0,200,224,0.5)',width=1)), row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['ICH_Kijun'], mode='lines',
            name='KIJUN', line=dict(color='rgba(255,159,0,0.5)',width=1,dash='dot')), row=1,col=1)

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='PRICE',
        increasing=dict(line=dict(color='#00d084',width=1),fillcolor='#00d084'),
        decreasing=dict(line=dict(color='#ff3b5c',width=1),fillcolor='#ff3b5c'),
        whiskerwidth=0.5), row=1,col=1)

    # EMAs
    if 'EMA' in indicators:
        ema_map = [('EMA9','#ff9f00',1.2),('EMA21','#00c8e0',1.2),('EMA50','#cc7a00',1.5),('EMA200','#504840',1.8)]
        for ema_name, clr, wid in ema_map:
            if ema_name in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[ema_name], mode='lines',
                    name=ema_name, line=dict(color=clr,width=wid)), row=1,col=1)

    # VWAP
    if 'VWAP' in indicators and 'VWAP' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines',
            name='VWAP', line=dict(color='#c084fc',width=2)), row=1,col=1)

    # Bollinger
    if 'Bollinger' in indicators and 'BB_Upper' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], mode='lines', name='BB+',
            line=dict(color='rgba(0,136,255,0.3)',width=1,dash='dot')), row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], mode='lines', name='BB mid',
            line=dict(color='rgba(0,136,255,0.2)',width=1)), row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], fill='tonexty',
            fillcolor='rgba(0,136,255,0.03)', mode='lines', name='BB-',
            line=dict(color='rgba(0,136,255,0.3)',width=1,dash='dot')), row=1,col=1)

    # SAR
    if 'SAR' in indicators and 'SAR' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SAR'], mode='markers', name='SAR',
            marker=dict(color='#00c8e0',size=3,symbol='circle')), row=1,col=1)

    # S/R
    if show_sr:
        for lvl in s_lvls:
            fig.add_hline(y=lvl, line_dash="dot", line_color="rgba(0,208,132,0.4)", line_width=1, row=1,col=1)
            fig.add_annotation(x=df.index[-1], y=lvl, text=f" S {lvl:.5g}", showarrow=False,
                font=dict(size=8,color="#00d084",family="IBM Plex Mono"), xanchor="left", row=1,col=1)
        for lvl in r_lvls:
            fig.add_hline(y=lvl, line_dash="dot", line_color="rgba(255,59,92,0.4)", line_width=1, row=1,col=1)
            fig.add_annotation(x=df.index[-1], y=lvl, text=f" R {lvl:.5g}", showarrow=False,
                font=dict(size=8,color="#ff3b5c",family="IBM Plex Mono"), xanchor="left", row=1,col=1)

    # Entry zones — draw SL/TP bands on chart
    if entry_zones:
        for ez in entry_zones:
            clr = '#00d084' if ez['dir']=='LONG' else '#ff3b5c'
            # TP band
            fig.add_hrect(y0=ez['entry'], y1=ez['tp'],
                fillcolor=f"rgba(0,208,132,0.06)", line_width=0, row=1,col=1)
            # SL band
            fig.add_hrect(y0=ez['sl'], y1=ez['entry'],
                fillcolor=f"rgba(255,59,92,0.06)", line_width=0, row=1,col=1)
            # Entry line
            fig.add_hline(y=ez['entry'], line_color=clr, line_width=1.5,
                line_dash="solid", row=1,col=1)
            fig.add_annotation(x=df.index[int(len(df)*0.01)], y=ez['entry'],
                text=f" ENTRY {ez['entry']:.5g}", showarrow=False,
                font=dict(size=8,color=clr,family="IBM Plex Mono"), xanchor="left", row=1,col=1)
            fig.add_hline(y=ez['tp'], line_color='rgba(0,208,132,0.6)', line_width=1,
                line_dash="dash", row=1,col=1)
            fig.add_annotation(x=df.index[int(len(df)*0.01)], y=ez['tp'],
                text=f" TP {ez['tp']:.5g}", showarrow=False,
                font=dict(size=8,color="#00d084",family="IBM Plex Mono"), xanchor="left", row=1,col=1)
            fig.add_hline(y=ez['sl'], line_color='rgba(255,59,92,0.6)', line_width=1,
                line_dash="dash", row=1,col=1)
            fig.add_annotation(x=df.index[int(len(df)*0.01)], y=ez['sl'],
                text=f" SL {ez['sl']:.5g}", showarrow=False,
                font=dict(size=8,color="#ff3b5c",family="IBM Plex Mono"), xanchor="left", row=1,col=1)

    # Buy/sell arrows
    if show_signals and 'signal_buy' in sig_df.columns:
        bm = sig_df['signal_buy']; sm = sig_df['signal_sell']
        if bm.any():
            fig.add_trace(go.Scatter(x=sig_df.index[bm], y=df['Low'][bm]*0.994, mode='markers',
                name='BUY', marker=dict(symbol='triangle-up',size=13,color='#00d084',
                line=dict(color='#003a22',width=1))), row=1,col=1)
        if sm.any():
            fig.add_trace(go.Scatter(x=sig_df.index[sm], y=df['High'][sm]*1.006, mode='markers',
                name='SELL', marker=dict(symbol='triangle-down',size=13,color='#ff3b5c',
                line=dict(color='#3a0010',width=1))), row=1,col=1)

    # Session shading
    if interval in ['5m','15m','1h'] and isinstance(df.index, pd.DatetimeIndex):
        for dd in sorted(set(df.index.date))[-5:]:
            fig.add_vrect(x0=pd.Timestamp(dd)+pd.Timedelta(hours=8),
                x1=pd.Timestamp(dd)+pd.Timedelta(hours=16,minutes=30),
                fillcolor='rgba(0,136,255,0.04)', line_width=0, row=1,col=1)
            fig.add_vrect(x0=pd.Timestamp(dd)+pd.Timedelta(hours=13,minutes=30),
                x1=pd.Timestamp(dd)+pd.Timedelta(hours=20),
                fillcolor='rgba(255,159,0,0.03)', line_width=0, row=1,col=1)

    # Row 2: RSI + Stoch
    if 'RSI14' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI14'], mode='lines', name='RSI14',
            line=dict(color='#ff9f00',width=1.3)), row=2,col=1)
        fig.add_hrect(y0=65,y1=100, fillcolor='rgba(255,59,92,0.04)', line_width=0, row=2,col=1)
        fig.add_hrect(y0=0, y1=35,  fillcolor='rgba(0,208,132,0.04)', line_width=0, row=2,col=1)
        for lvl,clr in [(65,'#ff3b5c'),(50,'#2a2a2a'),(35,'#00d084')]:
            fig.add_hline(y=lvl, line_dash="dash", line_color=clr, row=2,col=1, opacity=0.5)
    if 'STOCH_K' in df.columns and 'Stochastic' in indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df['STOCH_K'], mode='lines', name='%K',
            line=dict(color='#00c8e0',width=1)), row=2,col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['STOCH_D'], mode='lines', name='%D',
            line=dict(color='#cc7a00',width=1)), row=2,col=1)

    # Row 3: MACD
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD',
            line=dict(color='#0088ff',width=1.2)), row=3,col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], mode='lines', name='SIG',
            line=dict(color='#ff3b5c',width=1.2)), row=3,col=1)
        hc = ['#00d084' if v>=0 else '#ff3b5c' for v in df['MACD_Hist'].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='HIST',
            marker_color=hc, opacity=0.65, showlegend=False), row=3,col=1)
        fig.add_hline(y=0, line_color='#2a2a2a', row=3,col=1)

    # Row 4: Volume
    if 'Volume' in df.columns:
        vc = ['#ff3b5c' if c<o else '#00d084' for c,o in zip(df['Close'],df['Open'])]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='VOL',
            marker_color=vc, opacity=0.55, showlegend=False), row=4,col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Volume'].rolling(20).mean(),
            mode='lines', name='VOL MA', line=dict(color='#ff9f00',width=1,dash='dot'),
            showlegend=False), row=4,col=1)

    _ax = dict(gridcolor='#111',linecolor='#2a2a2a',
               tickfont=dict(family="IBM Plex Mono",size=9,color="#504840"), zeroline=False)
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(family="IBM Plex Mono",size=11,color="#ff9f00"), x=0.01),
        paper_bgcolor='#070707', plot_bgcolor='#0a0a0a',
        font=dict(family="IBM Plex Mono",size=10,color="#8a8070"),
        xaxis_rangeslider_visible=False,
        legend=dict(bgcolor='rgba(7,7,7,0.9)',bordercolor='#2a2a2a',borderwidth=1,
            font=dict(family="IBM Plex Mono",size=8,color="#8a8070"),
            orientation='h', y=1.02, x=0),
        margin=dict(l=8,r=8,t=40,b=8), height=720,
    )
    for i in range(1,5):
        fig.update_xaxes(**_ax, row=i,col=1)
        fig.update_yaxes(**_ax, row=i,col=1)
    fig.update_yaxes(title_text="PRICE", title_font=dict(size=8,color="#504840"), row=1,col=1)
    fig.update_yaxes(title_text="OSC",   title_font=dict(size=8,color="#504840"), row=2,col=1)
    fig.update_yaxes(title_text="MACD",  title_font=dict(size=8,color="#504840"), row=3,col=1)
    fig.update_yaxes(title_text="VOL",   title_font=dict(size=8,color="#504840"), row=4,col=1)
    return fig, sig_df


# ─────────────────────────────────────────────────────────────────────────────
# ALERT ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def check_alerts(df, sig_df, alert_conditions):
    """
    Evaluate user-defined alert conditions against current bar.
    Returns list of triggered alert dicts.
    """
    triggered = []
    last = df.iloc[-1]
    last_sig = sig_df.iloc[-1]
    pc = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    cur = last['Close']

    for cond in alert_conditions:
        t = cond['type']
        fired = False
        msg = ''

        if t == 'signal_buy' and bool(last_sig.get('signal_buy', False)):
            fired = True
            msg = f"BUY signal fired — score {last_sig['buy_pct']:.0f}%"
        elif t == 'signal_sell' and bool(last_sig.get('signal_sell', False)):
            fired = True
            msg = f"SELL signal fired — score {last_sig['sell_pct']:.0f}%"
        elif t == 'rsi_oversold' and 'RSI14' in df.columns:
            if last['RSI14'] < cond.get('value', 35):
                fired = True
                msg = f"RSI oversold: {last['RSI14']:.1f} < {cond.get('value',35)}"
        elif t == 'rsi_overbought' and 'RSI14' in df.columns:
            if last['RSI14'] > cond.get('value', 65):
                fired = True
                msg = f"RSI overbought: {last['RSI14']:.1f} > {cond.get('value',65)}"
        elif t == 'price_above' and cur > cond.get('value', 0):
            fired = True
            msg = f"Price {cur:.5g} crossed above {cond['value']:.5g}"
        elif t == 'price_below' and cur < cond.get('value', 0):
            fired = True
            msg = f"Price {cur:.5g} crossed below {cond['value']:.5g}"
        elif t == 'macd_cross_up' and bool(last_sig.get('sig_macd_cross_buy', False)):
            fired = True
            msg = "MACD bullish crossover"
        elif t == 'macd_cross_dn' and bool(last_sig.get('sig_macd_cross_sell', False)):
            fired = True
            msg = "MACD bearish crossover"
        elif t == 'ema_cross_up' and bool(last_sig.get('sig_ema_cross_buy', False)):
            fired = True
            msg = "EMA 9/21 bullish crossover"
        elif t == 'ema_cross_dn' and bool(last_sig.get('sig_ema_cross_sell', False)):
            fired = True
            msg = "EMA 9/21 bearish crossover"

        if fired:
            triggered.append({'label': cond.get('label', t), 'msg': msg,
                               'type': 'buy' if 'buy' in t or 'above' in t or 'oversold' in t else 'sell'})
    return triggered


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
_now = _dt.datetime.utcnow()
st.markdown(f"""
<div style="background:#0a0a0a;border-bottom:1px solid #1e1e1e;padding:8px 20px;
    margin:-1rem -1rem 1rem -1rem;display:flex;align-items:center;justify-content:space-between;">
  <div style="display:flex;align-items:center;gap:24px;">
    <span style="font-family:'IBM Plex Mono',monospace;color:#ff9f00;font-size:0.95rem;font-weight:700;letter-spacing:0.15em;">▸ SWING TRADER TERMINAL</span>
    <span style="font-family:'IBM Plex Mono',monospace;color:#2a2a2a;font-size:0.55rem;letter-spacing:0.2em;text-transform:uppercase;">FOREX · CRYPTO · NSE  ·  1H–4H INTRADAY</span>
  </div>
  <div style="font-family:'IBM Plex Mono',monospace;color:#504840;font-size:0.55rem;letter-spacing:0.15em;">
    UTC {_now.strftime("%Y-%m-%d  %H:%M:%S")}  ●  LIVE
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def sh(t): st.sidebar.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace;color:#ff9f00;font-size:0.57rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;border-bottom:1px solid #1e1e1e;padding-bottom:3px;margin:12px 0 6px 0;">▸ {t}</div>', unsafe_allow_html=True)

sh("CONTROL PANEL")

auto_refresh = st.sidebar.toggle('Auto Refresh (60s)', value=False)
if auto_refresh:
    st.markdown('<meta http-equiv="refresh" content="60">', unsafe_allow_html=True)
    st.sidebar.caption("⚡ Refreshes every 60s")

sh("ASSET")
asset_category = st.sidebar.selectbox('Category', list(ASSET_SYMBOLS.keys()))
selected_name  = st.sidebar.selectbox('Instrument', list(ASSET_SYMBOLS[asset_category].keys()))
ticker = ASSET_SYMBOLS[asset_category][selected_name]
if st.sidebar.checkbox('Custom Ticker'):
    ticker = st.sidebar.text_input('Symbol', value=ticker)
st.sidebar.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace;color:#504840;font-size:0.6rem;">{ticker}</div>', unsafe_allow_html=True)

# NSE CSV upload
nse_csv_df = None
if asset_category == 'Kenyan Stocks (NSE)':
    sh("NSE DATA")
    st.sidebar.info("NSE fetched from afx.kwayisi.org. Upload CSV as fallback.")
    nse_up = st.sidebar.file_uploader("NSE CSV", type=['csv'])
    if nse_up:
        try:
            nse_csv_df = pd.read_csv(nse_up)
            nse_csv_df.columns = [c.strip().title() for c in nse_csv_df.columns]
            dc = next((c for c in nse_csv_df.columns if 'Date' in c), None)
            if dc:
                nse_csv_df[dc] = pd.to_datetime(nse_csv_df[dc], dayfirst=True, errors='coerce')
                nse_csv_df = nse_csv_df.dropna(subset=[dc]).set_index(dc).sort_index()
                if 'Adj Close' not in nse_csv_df.columns and 'Close' in nse_csv_df.columns:
                    nse_csv_df['Adj Close'] = nse_csv_df['Close']
                st.sidebar.success(f"✅ {len(nse_csv_df)} rows")
        except Exception as e:
            st.sidebar.error(f"CSV error: {e}")

sh("TIMEFRAME")
interval = st.sidebar.selectbox('Primary Interval', ['1h','4h','1d','15m','5m'], index=0,
    help='1h or 4h for intraday swings')
_today = _dt.date.today()
_defs  = {'5m':5,'15m':10,'1h':45,'4h':90,'1d':365}
start_date = st.sidebar.date_input('From', value=_today - _dt.timedelta(days=_defs.get(interval,45)))
end_date   = st.sidebar.date_input('To',   value=_today)
if interval in ['15m','5m','1h','4h']:
    st.sidebar.caption(f"ℹ️ {interval}: yFinance max history auto-applied")

sh("INDICATORS")
indicators = st.sidebar.multiselect('Overlays',
    ['EMA','VWAP','Bollinger','SAR','Stochastic'], default=['EMA','VWAP'])
show_ich     = st.sidebar.toggle('Ichimoku Cloud', value=True)
show_signals = st.sidebar.toggle('Entry Signals', value=True)
show_sr      = st.sidebar.toggle('Support & Resistance', value=True)
show_entry_zones = st.sidebar.toggle('Draw Entry Zones on Chart', value=True)

sh("RISK")
account_size = st.sidebar.number_input('Account (USD/KES)', value=10000.0, step=500.0)
risk_pct     = st.sidebar.slider('Risk per Trade %', 0.5, 5.0, 1.0, step=0.5)
sl_atr_mult  = st.sidebar.slider('SL = N × ATR', 1.0, 3.0, 1.5, step=0.25)
tp_rr        = st.sidebar.slider('TP R:R Ratio', 1.0, 5.0, 2.5, step=0.25)

sh("ALERTS")
st.sidebar.caption("Alerts fire when conditions are met on the current bar.")
alert_types = {
    'Buy Signal (score ≥ 30%)': 'signal_buy',
    'Sell Signal (score ≥ 30%)': 'signal_sell',
    'MACD Bullish Cross': 'macd_cross_up',
    'MACD Bearish Cross': 'macd_cross_dn',
    'EMA 9/21 Bull Cross': 'ema_cross_up',
    'EMA 9/21 Bear Cross': 'ema_cross_dn',
    'RSI Oversold': 'rsi_oversold',
    'RSI Overbought': 'rsi_overbought',
}
selected_alerts = st.sidebar.multiselect('Active Alerts',
    list(alert_types.keys()),
    default=['Buy Signal (score ≥ 30%)', 'Sell Signal (score ≥ 30%)'])

alert_conditions = []
for lbl in selected_alerts:
    cond = {'type': alert_types[lbl], 'label': lbl}
    if 'RSI Oversold' in lbl:
        cond['value'] = st.sidebar.slider('RSI oversold threshold', 20, 45, 35)
    if 'RSI Overbought' in lbl:
        cond['value'] = st.sidebar.slider('RSI overbought threshold', 55, 80, 65)
    alert_conditions.append(cond)

price_alert_on = st.sidebar.toggle('Price Level Alert', value=False)
if price_alert_on:
    pa_price = st.sidebar.number_input('Alert price level', value=0.0, format="%.6g")
    pa_dir   = st.sidebar.radio('Direction', ['Above', 'Below'], horizontal=True)
    alert_conditions.append({
        'type': 'price_above' if pa_dir=='Above' else 'price_below',
        'value': pa_price, 'label': f'Price {pa_dir} {pa_price:.5g}'
    })


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if not ticker:
    st.info("Select a ticker in the sidebar.")
    st.stop()

# Banner
cat_icons = {'Forex':'🟢','Crypto':'🟣','Kenyan Stocks (NSE)':'🇰🇪',
             'Commodities':'🟡','US Stocks':'🔵','Volatility':'🔴'}
st.markdown(f"""
<div style="background:#0d0d0d;border:1px solid #1e1e1e;border-left:3px solid #ff9f00;
    padding:8px 16px;margin-bottom:12px;display:flex;align-items:center;justify-content:space-between;
    font-family:'IBM Plex Mono',monospace;">
  <div>
    <span style="color:#ff9f00;font-weight:700;font-size:1rem;letter-spacing:0.1em;">{ticker}</span>
    <span style="color:#504840;font-size:0.6rem;margin-left:12px;text-transform:uppercase;">{selected_name} · {asset_category} · {interval.upper()}</span>
  </div>
  <span style="color:#504840;font-size:0.58rem;">{cat_icons.get(asset_category,'📊')} {_now.strftime('%H:%M UTC')}</span>
</div>
""", unsafe_allow_html=True)

# ── Fetch
with st.spinner('Fetching market data...'):
    if nse_csv_df is not None and asset_category == 'Kenyan Stocks (NSE)':
        df = nse_csv_df.loc[str(start_date):str(end_date)].copy()
    else:
        df = download_data(ticker, start_date, end_date, interval, asset_category)

if df.empty:
    st.error("No data. Try a different date range, interval, or check the ticker symbol.")
    st.stop()

df     = add_indicators(df)
sig_df = compute_signals(df)
latest = sig_df.iloc[-1]
pc     = 'Adj Close' if 'Adj Close' in df.columns else 'Close'

cur_price  = df['Close'].iloc[-1]
prev_price = df['Close'].iloc[-2] if len(df) > 1 else cur_price
chg_pct    = (cur_price - prev_price) / prev_price * 100 if prev_price else 0
is_forex   = asset_category == 'Forex'
is_nse     = asset_category == 'Kenyan Stocks (NSE)'
fmt = (lambda v: f"{v:.5f}") if is_forex else (lambda v: f"KES {v:,.2f}") if is_nse else (lambda v: f"${v:,.4g}")

atr_now = df['ATR'].iloc[-1]  if ('ATR'   in df.columns and not pd.isna(df['ATR'].iloc[-1]))  else None
adx_now = df['ADX'].iloc[-1]  if ('ADX'   in df.columns and not pd.isna(df['ADX'].iloc[-1]))  else None
rsi_now = df['RSI14'].iloc[-1] if ('RSI14' in df.columns and not pd.isna(df['RSI14'].iloc[-1])) else None

# ── ATR-based entry zones
entry_zones = []
if show_entry_zones and atr_now:
    buy_score  = float(latest.get('buy_score',  0))
    sell_score = float(latest.get('sell_score', 0))
    if buy_score >= MAX_SCORE * 0.30:
        entry_zones.append({
            'dir': 'LONG',
            'entry': cur_price,
            'sl':    cur_price - sl_atr_mult * atr_now,
            'tp':    cur_price + tp_rr * sl_atr_mult * atr_now,
        })
    if sell_score >= MAX_SCORE * 0.30:
        entry_zones.append({
            'dir': 'SHORT',
            'entry': cur_price,
            'sl':    cur_price + sl_atr_mult * atr_now,
            'tp':    cur_price - tp_rr * sl_atr_mult * atr_now,
        })

# ── Check alerts
triggered_alerts = check_alerts(df, sig_df, alert_conditions)

# ═════════════════════════════════════════════════════════════════════════════
# ALERT BANNER
# ═════════════════════════════════════════════════════════════════════════════
if triggered_alerts:
    for alrt in triggered_alerts:
        clr = '#00d084' if alrt['type']=='buy' else '#ff3b5c'
        bg  = '#001a0e' if alrt['type']=='buy' else '#1a0008'
        st.markdown(f"""
        <div style="background:{bg};border:1px solid {clr};border-left:4px solid {clr};
            padding:10px 16px;margin-bottom:8px;font-family:'IBM Plex Mono',monospace;
            display:flex;align-items:center;gap:16px;">
          <span style="color:{clr};font-size:1.1rem;">{'▲' if alrt['type']=='buy' else '▼'}</span>
          <div>
            <div style="color:{clr};font-size:0.68rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;">⚡ ALERT: {alrt['label']}</div>
            <div style="color:#8a8070;font-size:0.62rem;margin-top:2px;">{alrt['msg']}</div>
          </div>
        </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL STRENGTH METER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;color:#ff9f00;font-size:0.65rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;border-bottom:1px solid #2a2a2a;padding-bottom:5px;margin:0 0 12px 0;">▸ SIGNAL STRENGTH</div>', unsafe_allow_html=True)

buy_pct  = float(latest.get('buy_pct',  0))
sell_pct = float(latest.get('sell_pct', 0))
net      = float(latest.get('net_score', 0))

if   net >= MAX_SCORE*0.35:  rating,rc,rbg = "STRONG BUY",  "#00d084","#001a0e"
elif net >= MAX_SCORE*0.20:  rating,rc,rbg = "BUY",          "#00d084","#001a0e"
elif net <= -MAX_SCORE*0.35: rating,rc,rbg = "STRONG SELL",  "#ff3b5c","#1a0008"
elif net <= -MAX_SCORE*0.20: rating,rc,rbg = "SELL",         "#ff3b5c","#1a0008"
else:                        rating,rc,rbg = "NEUTRAL",      "#8a8070","#141414"

col_sig, col_meter, col_stats = st.columns([1, 2, 1])

with col_sig:
    st.markdown(f"""
    <div style="background:{rbg};border:1px solid #2a2a2a;border-left:4px solid {rc};
         padding:16px 18px;font-family:'IBM Plex Mono',monospace;text-align:center;">
      <div style="color:#504840;font-size:0.5rem;letter-spacing:0.2em;margin-bottom:6px;">OVERALL SIGNAL</div>
      <div style="color:{rc};font-size:1.7rem;font-weight:700;letter-spacing:0.06em;">{rating}</div>
      <div style="color:#504840;font-size:0.56rem;margin-top:6px;">
        Net {net:+.1f} / {MAX_SCORE:.0f} pts
      </div>
    </div>""", unsafe_allow_html=True)

with col_meter:
    st.markdown(f"""
    <div style="background:#0d0d0d;border:1px solid #1e1e1e;padding:14px 16px;font-family:'IBM Plex Mono',monospace;">
      <div style="color:#504840;font-size:0.5rem;letter-spacing:0.18em;text-transform:uppercase;margin-bottom:10px;">WEIGHTED SCORE  ·  MAX {MAX_SCORE:.0f} PTS</div>
      <div style="margin-bottom:10px;">
        <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
          <span style="color:#00d084;font-size:0.62rem;">BUY STRENGTH</span>
          <span style="color:#00d084;font-size:0.62rem;">{buy_pct:.0f}%</span>
        </div>
        <div style="background:#111;height:8px;border-radius:0;position:relative;">
          <div style="background:linear-gradient(90deg,#003a22,#00d084);width:{buy_pct:.0f}%;height:8px;"></div>
        </div>
      </div>
      <div>
        <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
          <span style="color:#ff3b5c;font-size:0.62rem;">SELL STRENGTH</span>
          <span style="color:#ff3b5c;font-size:0.62rem;">{sell_pct:.0f}%</span>
        </div>
        <div style="background:#111;height:8px;border-radius:0;">
          <div style="background:linear-gradient(90deg,#3a0010,#ff3b5c);width:{sell_pct:.0f}%;height:8px;"></div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Individual signal grid (weighted)
    sig_display = [
        ('MACD',   'sig_macd_cross_buy',  'sig_macd_cross_sell',  2.0),
        ('EMA×',   'sig_ema_cross_buy',   'sig_ema_cross_sell',   2.0),
        ('ADX/DI', 'sig_adx_trend_buy',   'sig_adx_trend_sell',   1.5),
        ('RSI50',  'sig_rsi_buy',         'sig_rsi_sell',         1.5),
        ('ICHI',   'sig_ichimoku_buy',    'sig_ichimoku_sell',    1.5),
        ('VWAP',   'sig_vwap_buy',        'sig_vwap_sell',        1.0),
        ('STOCH',  'sig_stoch_buy',       'sig_stoch_sell',       1.0),
        ('BOLL',   'sig_bb_buy',          'sig_bb_sell',          0.5),
        ('SAR',    'sig_sar_buy',         'sig_sar_sell',         0.5),
    ]
    cols9 = st.columns(9)
    for i, (nm, bk, sk, wt) in enumerate(sig_display):
        ib = bool(latest.get(bk, False))
        is_ = bool(latest.get(sk, False))
        if ib:    clr,lbl,bg='#00d084','▲','#001a0e'
        elif is_: clr,lbl,bg='#ff3b5c','▼','#1a0008'
        else:     clr,lbl,bg='#2a2a2a','—','#0d0d0d'
        with cols9[i]:
            st.markdown(f"""
            <div style="background:{bg};border:1px solid #1a1a1a;padding:7px 3px;text-align:center;font-family:'IBM Plex Mono',monospace;">
              <div style="color:#2a2a2a;font-size:0.42rem;letter-spacing:0.08em;">{nm}</div>
              <div style="color:{clr};font-size:1rem;font-weight:700;">{lbl}</div>
              <div style="color:#2a2a2a;font-size:0.4rem;">{wt}×</div>
            </div>""", unsafe_allow_html=True)

with col_stats:
    ich_bias = '—'
    if 'ICH_SpanA' in df.columns and 'ICH_SpanB' in df.columns:
        cloud_top = max(df['ICH_SpanA'].iloc[-1], df['ICH_SpanB'].iloc[-1]) if not pd.isna(df['ICH_SpanA'].iloc[-1]) else None
        cloud_bot = min(df['ICH_SpanA'].iloc[-1], df['ICH_SpanB'].iloc[-1]) if not pd.isna(df['ICH_SpanA'].iloc[-1]) else None
        if cloud_top:
            if cur_price > cloud_top:   ich_bias='▲ ABOVE'
            elif cur_price < cloud_bot: ich_bias='▼ BELOW'
            else:                       ich_bias='— INSIDE'
    ich_color = '#00d084' if '▲' in ich_bias else '#ff3b5c' if '▼' in ich_bias else '#504840'

    ema_trend = '▲ BULL' if ('EMA9' in df.columns and 'EMA21' in df.columns and df['EMA9'].iloc[-1]>df['EMA21'].iloc[-1]) else '▼ BEAR'
    ema_color = '#00d084' if '▲' in ema_trend else '#ff3b5c'
    adx_strength = 'STRONG' if adx_now and adx_now > 25 else 'WEAK'
    adx_color    = '#00d084' if adx_now and adx_now > 25 else '#504840'

    st.markdown(f"""
    <div style="background:#0d0d0d;border:1px solid #1e1e1e;padding:12px 14px;font-family:'IBM Plex Mono',monospace;font-size:0.65rem;">
      <div style="color:#504840;font-size:0.48rem;letter-spacing:0.18em;margin-bottom:8px;">KEY READINGS</div>
      <div style="display:flex;justify-content:space-between;margin-bottom:5px;"><span style="color:#8a8070;">ATR</span><span style="color:#ff9f00;">{f'{atr_now:.4g}' if atr_now else '—'}</span></div>
      <div style="display:flex;justify-content:space-between;margin-bottom:5px;"><span style="color:#8a8070;">ADX</span><span style="color:{adx_color};">{f'{adx_now:.1f} {adx_strength}' if adx_now else '—'}</span></div>
      <div style="display:flex;justify-content:space-between;margin-bottom:5px;"><span style="color:#8a8070;">RSI</span><span style="color:{'#ff3b5c' if rsi_now and rsi_now>65 else '#00d084' if rsi_now and rsi_now<35 else '#ff9f00'};">{f'{rsi_now:.1f}' if rsi_now else '—'}</span></div>
      <div style="display:flex;justify-content:space-between;margin-bottom:5px;"><span style="color:#8a8070;">EMA</span><span style="color:{ema_color};">{ema_trend}</span></div>
      <div style="display:flex;justify-content:space-between;"><span style="color:#8a8070;">ICHI</span><span style="color:{ich_color};">{ich_bias}</span></div>
    </div>""", unsafe_allow_html=True)

# Price metrics
st.markdown('<div style="margin:10px 0 6px 0"></div>', unsafe_allow_html=True)
m1,m2,m3,m4,m5 = st.columns(5)
m1.metric("Price",        fmt(cur_price),        f"{chg_pct:+.3f}%")
m2.metric("Period High",  fmt(df['High'].max()))
m3.metric("Period Low",   fmt(df['Low'].min()))
m4.metric("ATR(14)",      f"{atr_now:.4g}" if atr_now else "N/A")
m5.metric("ADX",          f"{adx_now:.1f}" if adx_now else "N/A",
          "Trending" if adx_now and adx_now>25 else "Ranging")

# ═════════════════════════════════════════════════════════════════════════════
# MAIN CHART
# ═════════════════════════════════════════════════════════════════════════════
st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace;color:#ff9f00;font-size:0.65rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;border-bottom:1px solid #2a2a2a;padding-bottom:5px;margin:18px 0 10px 0;">▸ {selected_name}  ·  {interval.upper()} CHART</div>', unsafe_allow_html=True)

fig, sig_df = build_chart(df, f"{selected_name} ({ticker})  ·  {interval.upper()}",
                          indicators, show_signals, show_sr, show_ich,
                          interval, entry_zones if show_entry_zones else [])
st.plotly_chart(fig, use_container_width=True)

if interval in ['5m','15m','1h']:
    st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.55rem;color:#2a2a2a;margin:-6px 0 8px 0;display:flex;gap:16px;"><span><span style="color:#0088ff;">■</span> London 08:00–16:30 UTC</span><span><span style="color:#ff9f00;">■</span> New York 13:30–20:00 UTC</span></div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# MULTI-TIMEFRAME ANALYSIS  (1H → 4H → DAILY, always shown)
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;color:#ff9f00;font-size:0.65rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;border-bottom:1px solid #2a2a2a;padding-bottom:5px;margin:18px 0 10px 0;">▸ MULTI-TIMEFRAME BIAS  ·  1H → 4H → DAILY</div>', unsafe_allow_html=True)
st.caption("Trade in the direction where all 3 timeframes agree.")

MTF_FRAMES = [('1h','1 HOUR'), ('4h','4 HOUR'), ('1d','DAILY')]
_mtf_days  = {'1h':45, '4h':120, '1d':365}
mtf_cols   = st.columns(3)

mtf_results = {}
for col_i, (tf, tf_lbl) in enumerate(MTF_FRAMES):
    with mtf_cols[col_i]:
        try:
            _s  = _today - _dt.timedelta(days=_mtf_days[tf])
            _df = download_data(ticker, _s, _today, tf, asset_category)
            if _df.empty: raise ValueError("no data")
            _df  = add_indicators(_df)
            _sig = compute_signals(_df)
            _l   = _sig.iloc[-1]
            _net = float(_l.get('net_score', 0))
            _rsi = _df['RSI14'].iloc[-1] if 'RSI14' in _df.columns else None
            _adx = _df['ADX'].iloc[-1]   if 'ADX'   in _df.columns else None
            _e9  = _df['EMA9'].iloc[-1]  if 'EMA9'  in _df.columns else None
            _e21 = _df['EMA21'].iloc[-1] if 'EMA21' in _df.columns else None
            _macd_hist = _df['MACD_Hist'].iloc[-1] if 'MACD_Hist' in _df.columns else None
            _close = _df['Close'].iloc[-1]

            # Ichimoku bias
            _ich = '—'
            if 'ICH_SpanA' in _df.columns and not pd.isna(_df['ICH_SpanA'].iloc[-1]):
                _ct = max(_df['ICH_SpanA'].iloc[-1], _df['ICH_SpanB'].iloc[-1])
                _cb = min(_df['ICH_SpanA'].iloc[-1], _df['ICH_SpanB'].iloc[-1])
                _ich = '▲ ABOVE CLOUD' if _close>_ct else ('▼ BELOW CLOUD' if _close<_cb else '— IN CLOUD')

            if   _net >= MAX_SCORE*0.25: _bc,_bl,_bg='#00d084','▲ BULLISH','#001a0e'
            elif _net <= -MAX_SCORE*0.25:_bc,_bl,_bg='#ff3b5c','▼ BEARISH','#1a0008'
            else:                        _bc,_bl,_bg='#8a8070','— NEUTRAL','#0f0f0f'

            _tr = '▲ EMA9>21' if (_e9 and _e21 and _e9>_e21) else '▼ EMA9<21'
            _tc = '#00d084' if '▲' in _tr else '#ff3b5c'
            _mc_col = '#00d084' if _macd_hist and _macd_hist>0 else '#ff3b5c'
            _ich_col= '#00d084' if '▲' in _ich else '#ff3b5c' if '▼' in _ich else '#504840'

            is_primary = (tf == interval)
            border_style = f"border-top:3px solid {_bc}" if not is_primary else f"border-top:3px solid {_bc};border:2px solid {_bc}"

            st.markdown(f"""
            <div style="background:{_bg};{border_style};border:1px solid #2a2a2a;padding:14px 14px;font-family:'IBM Plex Mono',monospace;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                <span style="color:#504840;font-size:0.52rem;letter-spacing:0.18em;">{tf_lbl}{' ← CURRENT' if is_primary else ''}</span>
                <span style="color:{_bc};font-size:0.6rem;font-weight:700;">{_net:+.1f}pts</span>
              </div>
              <div style="color:{_bc};font-size:1.05rem;font-weight:700;margin-bottom:10px;">{_bl}</div>
              <div style="font-size:0.62rem;color:#8a8070;line-height:1.9;">
                <div style="display:flex;justify-content:space-between;"><span>RSI</span><span style="color:{'#ff3b5c' if _rsi and _rsi>65 else '#00d084' if _rsi and _rsi<35 else '#ff9f00'};">{f'{_rsi:.1f}' if _rsi else '—'}</span></div>
                <div style="display:flex;justify-content:space-between;"><span>ADX</span><span style="color:{'#00d084' if _adx and _adx>25 else '#504840'};">{f'{_adx:.1f}' if _adx else '—'}{'  STRONG' if _adx and _adx>25 else ''}</span></div>
                <div style="display:flex;justify-content:space-between;"><span>EMA</span><span style="color:{_tc};">{_tr}</span></div>
                <div style="display:flex;justify-content:space-between;"><span>MACD</span><span style="color:{_mc_col};">{'▲ Bullish' if _macd_hist and _macd_hist>0 else '▼ Bearish' if _macd_hist else '—'}</span></div>
                <div style="display:flex;justify-content:space-between;"><span>ICHI</span><span style="color:{_ich_col};font-size:0.58rem;">{_ich}</span></div>
              </div>
            </div>""", unsafe_allow_html=True)
            mtf_results[tf] = _bl

        except Exception:
            st.markdown(f'<div style="background:#0d0d0d;border:1px solid #1e1e1e;padding:14px;font-family:\'IBM Plex Mono\',monospace;text-align:center;"><div style="color:#504840;font-size:0.6rem;">{tf_lbl}<br>NO DATA</div></div>', unsafe_allow_html=True)
            mtf_results[tf] = 'NO DATA'

# MTF confluence summary
bulls = sum(1 for v in mtf_results.values() if '▲' in str(v))
bears = sum(1 for v in mtf_results.values() if '▼' in str(v))
if bulls == 3:
    st.markdown('<div style="background:#001a0e;border:1px solid #003a22;border-left:4px solid #00d084;padding:8px 16px;font-family:\'IBM Plex Mono\',monospace;font-size:0.65rem;color:#00d084;margin-top:8px;">✅ ALL 3 TIMEFRAMES BULLISH — High-confidence long setup</div>', unsafe_allow_html=True)
elif bears == 3:
    st.markdown('<div style="background:#1a0008;border:1px solid #3a0010;border-left:4px solid #ff3b5c;padding:8px 16px;font-family:\'IBM Plex Mono\',monospace;font-size:0.65rem;color:#ff3b5c;margin-top:8px;">✅ ALL 3 TIMEFRAMES BEARISH — High-confidence short setup</div>', unsafe_allow_html=True)
elif bulls == 2:
    st.markdown('<div style="background:#0d0d0d;border:1px solid #2a2a2a;border-left:4px solid #cc7a00;padding:8px 16px;font-family:\'IBM Plex Mono\',monospace;font-size:0.65rem;color:#cc7a00;margin-top:8px;">⚠️ 2/3 TIMEFRAMES BULLISH — Moderate long bias, wait for confirmation</div>', unsafe_allow_html=True)
elif bears == 2:
    st.markdown('<div style="background:#0d0d0d;border:1px solid #2a2a2a;border-left:4px solid #cc7a00;padding:8px 16px;font-family:\'IBM Plex Mono\',monospace;font-size:0.65rem;color:#cc7a00;margin-top:8px;">⚠️ 2/3 TIMEFRAMES BEARISH — Moderate short bias, wait for confirmation</div>', unsafe_allow_html=True)
else:
    st.markdown('<div style="background:#0d0d0d;border:1px solid #1e1e1e;border-left:4px solid #504840;padding:8px 16px;font-family:\'IBM Plex Mono\',monospace;font-size:0.65rem;color:#504840;margin-top:8px;">— MIXED SIGNALS — No clear directional bias. Sit on hands.</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# CANDLESTICK PATTERNS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;color:#ff9f00;font-size:0.65rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;border-bottom:1px solid #2a2a2a;padding-bottom:5px;margin:18px 0 10px 0;">▸ CANDLE PATTERNS  ·  LAST BAR</div>', unsafe_allow_html=True)
try:
    cdl = detect_candle_patterns(df)
    bullish_cdl = {k:v for k,v in cdl.items() if v>0}
    bearish_cdl = {k:v for k,v in cdl.items() if v<0}
    if bullish_cdl or bearish_cdl:
        cp1, cp2 = st.columns(2)
        with cp1:
            if bullish_cdl:
                for nm in bullish_cdl:
                    st.markdown(f'<div style="background:#001a0e;border-left:3px solid #00d084;padding:5px 10px;font-family:\'IBM Plex Mono\',monospace;font-size:0.67rem;color:#00d084;margin-bottom:3px;">▲ {nm}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<span style="color:#2a2a2a;font-size:0.62rem;font-family:\'IBM Plex Mono\',monospace;">No bullish patterns</span>', unsafe_allow_html=True)
        with cp2:
            if bearish_cdl:
                for nm in bearish_cdl:
                    st.markdown(f'<div style="background:#1a0008;border-left:3px solid #ff3b5c;padding:5px 10px;font-family:\'IBM Plex Mono\',monospace;font-size:0.67rem;color:#ff3b5c;margin-bottom:3px;">▼ {nm}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<span style="color:#2a2a2a;font-size:0.62rem;font-family:\'IBM Plex Mono\',monospace;">No bearish patterns</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#504840;font-size:0.67rem;font-family:\'IBM Plex Mono\',monospace;">No recognisable patterns on last bar.</span>', unsafe_allow_html=True)
except Exception as e:
    st.warning(f"Pattern scanner: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# TRADE CALCULATOR
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;color:#ff9f00;font-size:0.65rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;border-bottom:1px solid #2a2a2a;padding-bottom:5px;margin:18px 0 10px 0;">▸ TRADE CALCULATOR</div>', unsafe_allow_html=True)

if atr_now:
    st.markdown(f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.58rem;color:#504840;">ATR(14) = {atr_now:.4g}  ·  SL = {sl_atr_mult}× ATR = {sl_atr_mult*atr_now:.4g}  ·  TP = {tp_rr}:1 R:R</span>', unsafe_allow_html=True)

tc1,tc2,tc3,tc4 = st.columns(4)
with tc1: entry  = st.number_input('Entry', value=float(round(cur_price,8)), format="%.6g")
with tc2: stop   = st.number_input('Stop Loss', value=float(round(cur_price - (sl_atr_mult*atr_now if atr_now else cur_price*0.01), 8)), format="%.6g")
with tc3: target = st.number_input('Take Profit', value=float(round(cur_price + (tp_rr*sl_atr_mult*atr_now if atr_now else cur_price*0.02), 8)), format="%.6g")
with tc4: direction = st.selectbox('Direction', ['LONG','SHORT'])

if entry > 0 and stop != entry:
    risk_u   = abs(entry-stop)
    reward_u = abs(target-entry)
    rr       = reward_u/risk_u if risk_u else 0
    max_risk = account_size*(risk_pct/100)
    pos_size = max_risk/risk_u if risk_u else 0
    pos_val  = pos_size*entry

    tm1,tm2,tm3,tm4,tm5,tm6 = st.columns(6)
    tm1.metric("R:R",          f"1:{rr:.2f}", "✅" if rr>=2 else "⚠️" if rr>=1 else "❌")
    tm2.metric("Position",     f"{pos_size:,.3f}")
    tm3.metric("Value",        fmt(pos_val))
    tm4.metric("Max Loss",     fmt(pos_size*risk_u),   f"-{risk_pct:.1f}%")
    tm5.metric("Max Gain",     fmt(pos_size*reward_u), f"+{rr*risk_pct:.1f}%")
    tm6.metric("Breakeven",    fmt(entry + risk_u*0.1 if direction=='LONG' else entry - risk_u*0.1))

# ═════════════════════════════════════════════════════════════════════════════
# TABS: Signal History / OHLCV / News
# ═════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(['Signal History', 'Recent Bars', 'News'])

with tab1:
    hist = sig_df[sig_df['signal_buy'] | sig_df['signal_sell']].copy()
    if not hist.empty:
        hist['Dir']      = hist.apply(lambda r: '▲ BUY' if r.get('signal_buy') else '▼ SELL', axis=1)
        hist['Price']    = hist[pc].round(6)
        hist['Buy%']     = hist['buy_pct'].round(1)
        hist['Sell%']    = hist['sell_pct'].round(1)
        hist['Net Score']= hist['net_score'].round(2)
        st.dataframe(hist[['Dir','Price','Buy%','Sell%','Net Score']].tail(60).sort_index(ascending=False), use_container_width=True)
    else:
        st.info("No signals fired in this period. Try a wider date range.")

with tab2:
    show_cols = [c for c in ['Open','High','Low','Close','Volume','ATR','RSI14','MACD_Hist','VWAP','ADX'] if c in df.columns]
    st.dataframe(df[show_cols].tail(100).sort_index(ascending=False).round(5), use_container_width=True)

with tab3:
    try:
        nticker = ticker.replace('.NR','').replace('=X','').replace('-USD','')
        from stocknews import StockNews
        sn = StockNews(nticker, save_news=False)
        nd = sn.read_rss()
        st.dataframe(nd[['published','title','sentiment_title','sentiment_summary']].head(12), use_container_width=True)
    except Exception as e:
        st.info(f"News unavailable for {ticker}: {e}")
