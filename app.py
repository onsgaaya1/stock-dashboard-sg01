import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

st.set_page_config(page_title="Stock Dashboard SG01", layout="wide")
st.title("Tableau de Bord Boursier - SG01 Finance & Trading")
st.markdown("**Etudiant(e) :** Ons Gaaya | **Dataset :** 9000+ Tickers (1962-2024)")
st.markdown("---")

@st.cache_data
def load_data():
    file_id = "17GFmaHT1o6oMHgV8Uu9tGHHAnp8v08YG"
    local_path = "all_stock_data.csv"

    if not os.path.exists(local_path):
        import gdown
        gdown.download(id=file_id, output=local_path, quiet=False)

    # Load last 600k rows to get the most recent data instead of oldest
    df = pd.read_csv(local_path)
    df = df.tail(600000)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.rename(columns={"Stock Splits": "Stock_Splits"})
    df = df.drop(columns=["Dividends", "Stock_Splits"], errors="ignore")

    TICKERS = ["AAPL","MSFT","GOOGL","AMZN","TSLA","META","NVDA",
               "JPM","BAC","GS","XOM","WMT","KO","PFE","NFLX"]
    df = df[df["Ticker"].isin(TICKERS)].copy()
    df = df.sort_values(["Ticker","Date"]).reset_index(drop=True)
    df = df[(df["Close"] > 0) & (df["Volume"] >= 0)].dropna(subset=["Open","High","Low","Close","Volume"])

    # --- Basic indicators via transform (safe, keeps index) ---
    df["MA_20"]         = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(20).mean())
    df["MA_50"]         = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(50).mean())
    df["Daily_Return"]  = df.groupby("Ticker")["Close"].transform(lambda x: x.pct_change())
    df["Volatility_20"] = df.groupby("Ticker")["Daily_Return"].transform(lambda x: x.rolling(20).std() * np.sqrt(252))
    df["Price_Range"]   = df["High"] - df["Low"]

    def rsi(s):
        d = s.diff()
        g = d.clip(lower=0).rolling(14).mean()
        l = (-d.clip(upper=0)).rolling(14).mean()
        rs = g / l.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    df["RSI_14"] = df.groupby("Ticker")["Close"].transform(rsi)

    # --- MACD via transform (avoids groupby apply column drop issue) ---
    def macd_line(s):
        ef = s.ewm(span=12, adjust=False).mean()
        es = s.ewm(span=26, adjust=False).mean()
        return ef - es

    def signal_line(s):
        m = macd_line(s)
        return m.ewm(span=9, adjust=False).mean()

    df["MACD"]        = df.groupby("Ticker")["Close"].transform(macd_line)
    df["MACD_Signal"] = df.groupby("Ticker")["Close"].transform(signal_line)
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    # --- Bollinger Bands via transform ---
    def bb_upper(s):
        ma = s.rolling(20).mean()
        return ma + 2 * s.rolling(20).std()

    def bb_lower(s):
        ma = s.rolling(20).mean()
        return ma - 2 * s.rolling(20).std()

    df["BB_Upper"] = df.groupby("Ticker")["Close"].transform(bb_upper)
    df["BB_Lower"] = df.groupby("Ticker")["Close"].transform(bb_lower)
    df["BB_Width"]  = (df["BB_Upper"] - df["BB_Lower"]) / df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(20).mean())

    df["Target"] = df.groupby("Ticker")["Close"].transform(lambda x: x.shift(-1))

    return df

with st.spinner("Chargement et traitement des donnees..."):
    df = load_data()
st.success(f"✅ {len(df):,} lignes chargees!")

st.sidebar.header("Parametres")
ticker   = st.sidebar.selectbox("Choisir un ticker :", sorted(df["Ticker"].unique()), index=0)
date_min = df["Date"].min().date()
date_max = df["Date"].max().date()
default_start = max(date_min, (pd.Timestamp(date_max) - pd.DateOffset(years=3)).date())
start, end = st.sidebar.date_input("Periode :", [default_start, date_max], min_value=date_min, max_value=date_max)

df_t = df[(df["Ticker"]==ticker) & (df["Date"]>=pd.Timestamp(start)) & (df["Date"]<=pd.Timestamp(end))].sort_values("Date")
if df_t.empty:
    st.error("Aucune donnee pour cette selection."); st.stop()

last  = df_t["Close"].iloc[-1]
first = df_t["Close"].iloc[0]
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Prix actuel",  f"${last:,.2f}")
c2.metric("Variation",    f"{((last-first)/first*100):+.2f}%")
c3.metric("Volume moyen", f"{df_t['Volume'].mean():,.0f}")
c4.metric("Volatilite",   f"{df_t['Volatility_20'].mean()*100:.1f}%" if df_t['Volatility_20'].notna().any() else "N/A")
c5.metric("RSI actuel",   f"{df_t['RSI_14'].dropna().iloc[-1]:.1f}" if df_t['RSI_14'].notna().any() else "N/A")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["Prix & MA", "RSI", "MACD", "Modele ML"])

with tab1:
    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(df_t["Date"], df_t["Close"], label="Cloture", color="black",  lw=1.0)
    ax.plot(df_t["Date"], df_t["MA_20"], label="MA 20",   color="blue",   lw=1.2)
    ax.plot(df_t["Date"], df_t["MA_50"], label="MA 50",   color="orange", lw=1.2)
    ax.fill_between(df_t["Date"], df_t["BB_Lower"], df_t["BB_Upper"], alpha=0.1, color="purple", label="Bollinger")
    ax.set_ylabel("Prix ($)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title(f"{ticker} - Prix + MA + Bollinger", fontweight="bold")
    st.pyplot(fig)

with tab2:
    df_rsi = df_t.dropna(subset=["RSI_14"])
    if len(df_rsi) > 0:
        fig, ax = plt.subplots(figsize=(14,3))
        ax.plot(df_rsi["Date"], df_rsi["RSI_14"], color="purple", lw=1.0)
        ax.axhline(70, color="red",   linestyle="--", lw=1, label="Surachat (70)")
        ax.axhline(30, color="green", linestyle="--", lw=1, label="Survente (30)")
        ax.fill_between(df_rsi["Date"], df_rsi["RSI_14"], 70, where=df_rsi["RSI_14"]>70, alpha=0.2, color="red")
        ax.fill_between(df_rsi["Date"], df_rsi["RSI_14"], 30, where=df_rsi["RSI_14"]<30, alpha=0.2, color="green")
        ax.set_ylim(0,100); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_title(f"{ticker} - RSI (14j)", fontweight="bold")
        st.pyplot(fig)
    else:
        st.info("Pas assez de donnees pour le RSI.")

with tab3:
    df_macd = df_t.dropna(subset=["MACD","MACD_Signal"])
    if len(df_macd) > 0:
        fig, (a1,a2) = plt.subplots(2,1,figsize=(14,6),sharex=True)
        a1.plot(df_macd["Date"], df_macd["Close"], color="black", lw=1.0, label="Cloture")
        a1.fill_between(df_macd["Date"], df_macd["BB_Lower"], df_macd["BB_Upper"], alpha=0.15, color="purple", label="Bollinger")
        a1.legend(fontsize=8); a1.grid(True,alpha=0.3); a1.set_ylabel("Prix ($)")
        a2.plot(df_macd["Date"], df_macd["MACD"],        label="MACD",   color="blue",   lw=1.0)
        a2.plot(df_macd["Date"], df_macd["MACD_Signal"], label="Signal", color="orange", lw=1.0)
        colors = ["green" if v >= 0 else "red" for v in df_macd["MACD_Hist"].fillna(0)]
        a2.bar(df_macd["Date"], df_macd["MACD_Hist"], color=colors, alpha=0.4, width=1)
        a2.axhline(0, color="black", lw=0.5); a2.legend(fontsize=8); a2.grid(True,alpha=0.3); a2.set_ylabel("MACD")
        plt.tight_layout(); st.pyplot(fig)
    else:
        st.info("Pas assez de donnees pour le MACD.")

with tab4:
    FEATURES = [f for f in ["Open","High","Low","Close","Volume","MA_20","MA_50",
                             "RSI_14","Volatility_20","Price_Range","MACD","MACD_Signal","BB_Width"]
                if f in df_t.columns]
    df_ml = df_t.copy()
    df_ml["Target"] = df_ml["Close"].shift(-1)
    df_ml = df_ml.dropna(subset=FEATURES + ["Target"])
    if len(df_ml) > 100:
        X = df_ml[FEATURES]; y = df_ml["Target"]
        sp = int(len(X) * 0.8)
        sc = StandardScaler()
        Xtr = sc.fit_transform(X.iloc[:sp]); Xte = sc.transform(X.iloc[sp:])
        model = LinearRegression(); model.fit(Xtr, y.iloc[:sp])
        yp = model.predict(Xte)
        mae  = mean_absolute_error(y.iloc[sp:], yp)
        rmse = np.sqrt(mean_squared_error(y.iloc[sp:], yp))
        r2   = r2_score(y.iloc[sp:], yp)
        mc1,mc2,mc3 = st.columns(3)
        mc1.metric("MAE",  f"{mae:.4f}$")
        mc2.metric("RMSE", f"{rmse:.4f}$")
        mc3.metric("R²",   f"{r2:.4f}")
        fig, ax = plt.subplots(figsize=(14,4))
        ax.plot(df_ml.iloc[sp:]["Date"].values, y.iloc[sp:].values, label="Reel",   color="blue", lw=1.2)
        ax.plot(df_ml.iloc[sp:]["Date"].values, yp,                 label="Predit", color="red",  lw=1.2, alpha=0.8)
        ax.set_title(f"{ticker} - Regression Lineaire : Reel vs Predit", fontweight="bold")
        ax.set_ylabel("Prix ($)"); ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    else:
        st.warning("Pas assez de donnees pour entrainer le modele.")

st.markdown("---")
st.caption("Mini-Projet SG01 - Ons Gaaya | 9000+ Tickers of Stock Market Data (Full History) | Kaggle")
