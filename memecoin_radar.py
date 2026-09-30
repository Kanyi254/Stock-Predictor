"""
memecoin_radar.py  -  fast-mover screener for brand-new memecoin pools.

Data : GeckoTerminal public API (no key, ~30 calls/min)
Use  : from memecoin_radar import render_memecoin_radar   (called from main2.py)
"""
import requests
import pandas as pd
import streamlit as st

GT_BASE = "https://api.geckoterminal.com/api/v2"
HEADERS = {"Accept": "application/json;version=20230302"}

NETWORKS = {
    "Solana": "solana",
    "Base": "base",
    "Ethereum": "eth",
    "BNB Chain": "bsc",
    "Arbitrum": "arbitrum",
}
DEXSCREENER_SLUG = {"eth": "ethereum"}   # everything else matches


# ───────────────────────── helpers ─────────────────────────
def _f(x, default=0.0):
    try:
        return float(x) if x not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


# ───────────────────────── data ─────────────────────────
def _parse(p, net_id):
    a = p.get("attributes", {})
    created = pd.to_datetime(a.get("pool_created_at"), utc=True, errors="coerce")
    if pd.isna(created):
        return None
    vol = a.get("volume_usd") or {}
    chg = a.get("price_change_percentage") or {}
    tx = a.get("transactions") or {}
    h1, m5 = tx.get("h1") or {}, tx.get("m5") or {}
    base_id = (((p.get("relationships") or {}).get("base_token") or {}).get("data") or {}).get("id", "")
    name = a.get("name", "?")
    return {
        "symbol": name.split(" / ")[0],
        "pool": a.get("address", ""),
        "token": base_id.split("_", 1)[-1],
        "created": created,
        "liquidity_usd": _f(a.get("reserve_in_usd")),
        "fdv_usd": _f(a.get("fdv_usd")),
        "vol_m5": _f(vol.get("m5")),
        "vol_h1": _f(vol.get("h1")),
        "vol_h24": _f(vol.get("h24")),
        "chg_m5": _f(chg.get("m5")),
        "chg_h1": _f(chg.get("h1")),
        "chg_h6": _f(chg.get("h6")),
        "buys_h1": int(_f(h1.get("buys"))),
        "sells_h1": int(_f(h1.get("sells"))),
        "buyers_h1": int(_f(h1.get("buyers"))),
        "txns_m5": int(_f(m5.get("buys")) + _f(m5.get("sells"))),
        "net": net_id,
    }


def fetch_pools(network_id):
    """Newest pools (2 pages) + currently trending pools, de-duplicated."""
    calls = [
        (f"{GT_BASE}/networks/{network_id}/new_pools", {"page": 1}, True),
        (f"{GT_BASE}/networks/{network_id}/new_pools", {"page": 2}, True),
        (f"{GT_BASE}/networks/{network_id}/trending_pools", {"duration": "1h"}, False),
    ]
    rows, seen = [], set()
    for url, params, required in calls:
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=15)
            if r.status_code == 429:
                raise RuntimeError("rate limited (30 calls/min) - use a slower refresh")
            r.raise_for_status()
        except (requests.RequestException, RuntimeError):
            if required:
                raise
            continue                      # trending is a bonus, never fatal
        for p in r.json().get("data", []):
            row = _parse(p, network_id)
            if row and row["pool"] not in seen:
                seen.add(row["pool"])
                rows.append(row)
    return rows


# ───────────────────────── scoring ─────────────────────────
def momentum_score(r):
    """
    0-100 heat score. Rewards: rising price, accelerating volume, buyer-dominated
    flow, real turnover, freshness. Penalizes: can't-exit liquidity, no sells,
    too few buyers, and coins that already ran (you'd be buying the top).
    Returns (score, notes).
    """
    notes = []
    accel = (r["vol_m5"] * 12 / r["vol_h1"]) if r["vol_h1"] > 0 else 0.0
    total_tx = r["buys_h1"] + r["sells_h1"]
    buy_ratio = r["buys_h1"] / total_tx if total_tx else 0.5
    turnover = r["vol_h1"] / r["liquidity_usd"] if r["liquidity_usd"] > 0 else 0.0

    price = 15 * _clip(r["chg_m5"] / 20, 0, 1) + 15 * _clip(r["chg_h1"] / 100, 0, 1)
    volume = 25 * _clip((accel - 1) / 3, 0, 1)
    flow = 25 * _clip((buy_ratio - 0.5) / 0.3, 0, 1)
    turn = 10 * _clip(turnover / 2, 0, 1)
    fresh = 10 * _clip(1 - r["age_h"] / 12, 0, 1)
    score = price + volume + flow + turn + fresh

    if r["liquidity_usd"] < 10_000:
        score -= 20; notes.append("thin liquidity (hard to exit)")
    if r["buys_h1"] >= 10 and r["sells_h1"] == 0:
        score = min(score - 30, 25); notes.append("no sells (honeypot?)")
    if r["buyers_h1"] < 15:
        score -= 10; notes.append("few buyers")
    if r["chg_h1"] > 300 or r["chg_h6"] > 1000:
        score -= 15; notes.append("already pumped (late)")
    if r["fdv_usd"] > 0 and r["liquidity_usd"] / r["fdv_usd"] < 0.01:
        notes.append("liq << FDV")
    if accel > 1.5:
        notes.append("volume accelerating")
    return round(_clip(score, 0, 100), 1), ", ".join(notes) if notes else "-"


def scan(network_names, max_age_h=24, min_liq=5000, min_vol_h1=500, fetch=fetch_pools):
    rows, errors = [], []
    for name in network_names:
        try:
            rows += [{**r, "network": name} for r in fetch(NETWORKS[name])]
        except Exception as e:
            errors.append(f"{name}: {e}")
    if not rows:
        return pd.DataFrame(), errors

    df = pd.DataFrame(rows)
    df["age_h"] = (pd.Timestamp.now(tz="UTC") - df["created"]).dt.total_seconds() / 3600
    df = df[(df["age_h"] <= max_age_h)
            & (df["liquidity_usd"] >= min_liq)
            & (df["vol_h1"] >= min_vol_h1)].copy()
    if df.empty:
        return df, errors

    scored = df.apply(momentum_score, axis=1, result_type="expand")
    df["score"], df["notes"] = scored[0], scored[1]
    df["signal"] = df["score"].apply(lambda s: "🔥 HOT" if s >= 65 else ("👀 WATCH" if s >= 45 else ""))
    slug = df["net"].replace(DEXSCREENER_SLUG)
    df["chart"] = "https://dexscreener.com/" + slug + "/" + df["pool"]
    df["safety"] = df.apply(
        lambda r: f"https://rugcheck.xyz/tokens/{r['token']}" if r["net"] == "solana" else None, axis=1)
    df["key"] = df["net"] + ":" + df["pool"]
    return df, errors


# ───────────────────────── Streamlit UI ─────────────────────────
_HDR = ("font-family:IBM Plex Mono,monospace;color:#ff9f00;font-size:0.7rem;font-weight:700;"
        "letter-spacing:0.2em;text-transform:uppercase;border-bottom:1px solid #2a2a2a;"
        "padding-bottom:6px;margin:20px 0 12px 0;")


def render_memecoin_radar():
    st.sidebar.header("Memecoin Radar")
    nets = st.sidebar.multiselect("Networks", list(NETWORKS), default=["Solana", "Base"])
    max_age = st.sidebar.slider("Max pool age (hours)", 1, 24, 24)
    min_liq = st.sidebar.number_input("Min liquidity (USD)", 0, 1_000_000, 10_000, step=1000,
                                      help="Below ~$10k you usually can't sell without heavy slippage.")
    min_vol = st.sidebar.number_input("Min 1h volume (USD)", 0, 10_000_000, 1000, step=500)
    alert_at = st.sidebar.slider("Alert when heat score >=", 40, 90, 65)
    sort_by = st.sidebar.radio("Sort by", ["Heat score", "Newest", "1h change"])
    refresh = st.sidebar.select_slider("Auto-refresh (seconds)", [60, 120, 300], value=60)

    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=refresh * 1000, key="radar_refresh")
    except ImportError:
        st.sidebar.info("pip install streamlit-autorefresh for auto-refresh.")
        if st.sidebar.button("Refresh now"):
            st.cache_data.clear()
            st.rerun()

    st.markdown(f'<div style="{_HDR}">▸ MEMECOIN RADAR  /  FAST MOVERS  ·  POOLS &lt; {max_age}H OLD</div>',
                unsafe_allow_html=True)
    if not nets:
        st.info("Pick at least one network in the sidebar.")
        return

    cached_fetch = st.cache_data(ttl=60, show_spinner=False)(fetch_pools)
    with st.spinner("Scanning new pools..."):
        df, errors = scan(nets, max_age, min_liq, min_vol, fetch=cached_fetch)
    for e in errors:
        st.warning(e)
    if df.empty:
        st.info("Nothing passes the filters right now. Try lowering min liquidity or volume.")
        return

    # in-app alerts: fire once per pool when it crosses the heat threshold
    alerted = st.session_state.setdefault("radar_alerted", set())
    hot_new = df[(df["score"] >= alert_at) & (~df["key"].isin(alerted))]
    for _, r in hot_new.sort_values("score", ascending=False).head(5).iterrows():
        st.toast(f"🔥 {r['symbol']} on {r['network']} | heat {r['score']:.0f} | "
                 f"{r['chg_m5']:+.0f}% 5m | age {r['age_h']:.1f}h")
    alerted.update(hot_new["key"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Pools scanned", len(df))
    c2.metric("🔥 Hot now", int((df["score"] >= 65).sum()))
    c3.metric("Alerts this session", len(alerted))

    order = {"Heat score": ("score", False), "Newest": ("age_h", True), "1h change": ("chg_h1", False)}[sort_by]
    df = df.sort_values(order[0], ascending=order[1]).reset_index(drop=True)

    view = df[["signal", "symbol", "network", "age_h", "score", "chg_m5", "chg_h1", "chg_h6",
               "liquidity_usd", "vol_h1", "buys_h1", "sells_h1", "buyers_h1", "notes", "chart", "safety"]]
    st.dataframe(
        view, use_container_width=True, hide_index=True,
        column_config={
            "signal": "Signal",
            "age_h": st.column_config.NumberColumn("Age (h)", format="%.1f"),
            "score": st.column_config.ProgressColumn("Heat", min_value=0, max_value=100, format="%.0f"),
            "chg_m5": st.column_config.NumberColumn("5m %", format="%+.1f"),
            "chg_h1": st.column_config.NumberColumn("1h %", format="%+.0f"),
            "chg_h6": st.column_config.NumberColumn("6h %", format="%+.0f"),
            "liquidity_usd": st.column_config.NumberColumn("Liquidity", format="$%.0f"),
            "vol_h1": st.column_config.NumberColumn("Vol 1h", format="$%.0f"),
            "buys_h1": "Buys 1h", "sells_h1": "Sells 1h", "buyers_h1": "Buyers 1h",
            "notes": "Notes",
            "chart": st.column_config.LinkColumn("Chart", display_text="open"),
            "safety": st.column_config.LinkColumn("Safety", display_text="rugcheck"),
        },
    )
    st.caption("Heat score is a screener, not a prediction. Most new memecoins go to zero. "
               "Always check the contract and confirm you can sell before buying.")
