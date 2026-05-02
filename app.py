import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model    import LinearRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Stock Dashboard SG01", layout="wide")
st.title("Tableau de Bord Boursier - SG01 Finance & Trading")
st.markdown("**Etudiant(e) :** Ons Gaaya | **Dataset :** 9000+ Tickers (1962-2024)")
st.markdown("---")

@st.cache_data
def load_data():
    # Chargement CSV — 600 000 lignes pour couvrir les 15 tickers selectionnes
    df = pd.read_csv("all_stock_data.csv", nrows=600000)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.rename(columns={"Stock Splits": "Stock_Splits"})
    df = df.drop(columns=["Dividends", "Stock_Splits"], errors="ignore")
    TICKERS = ["AAPL","MSFT","GOOGL","AMZN","TSLA","META","NVDA","JPM","BAC","GS","XOM","WMT","KO","PFE","NFLX"]
    df = df[df["Ticker"].isin(TICKERS)].copy()
    df = df.sort_values(["Ticker","Date"]).reset_index(drop=True)
    df = df[(df["Close"] > 0) & (df["Volume"] >= 0)].dropna(subset=["Open","High","Low","Close","Volume"])
    df["MA_20"]         = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(20).mean())
    df["MA_50"]         = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(50).mean())
    df["Daily_Return"]  = df.groupby("Ticker")["Close"].transform(lambda x: x.pct_change())
    df["Volatility_20"] = df.groupby("Ticker")["Daily_Return"].transform(lambda x: x.rolling(20).std() * np.sqrt(252))
    def rsi(s):
        d = s.diff(); g = d.clip(lower=0).rolling(14).mean(); l = -d.clip(upper=0).rolling(14).mean()
        return 100 - (100/(1+g/l))
    df["RSI_14"] = df.groupby("Ticker")["Close"].transform(rsi)
    def macd_fn(grp):
        ef = grp["Close"].ewm(span=12,adjust=False).mean(); es = grp["Close"].ewm(span=26,adjust=False).mean()
        m = ef-es; sig = m.ewm(span=9,adjust=False).mean()
        grp["MACD"]=m.values; grp["MACD_Signal"]=sig.values; grp["MACD_Hist"]=(m-sig).values; return grp
    df = df.groupby("Ticker",group_keys=False).apply(macd_fn)
    def bb(g):
        ma=g["Close"].rolling(20).mean(); std=g["Close"].rolling(20).std()
        g["BB_Upper"]=(ma+2*std).values; g["BB_Lower"]=(ma-2*std).values; g["BB_Width"]=((ma+2*std-(ma-2*std))/ma).values; return g
    df = df.groupby("Ticker",group_keys=False).apply(bb)
    df["Price_Range"] = df["High"] - df["Low"]
    df["Target"]      = df.groupby("Ticker")["Close"].transform(lambda x: x.shift(-1))
    return df

with st.spinner("Chargement et traitement des donnees..."):
    df = load_data()
st.success(f"OK {len(df):,} lignes chargees!")

st.sidebar.header("Parametres")
ticker   = st.sidebar.selectbox("Choisir un ticker :", sorted(df["Ticker"].unique()), index=0)
date_min = df["Date"].min().date()
date_max = df["Date"].max().date()
start, end = st.sidebar.date_input("Periode :", [date_min, date_max], min_value=date_min, max_value=date_max)

df_t = df[(df["Ticker"]==ticker)&(df["Date"]>=pd.Timestamp(start))&(df["Date"]<=pd.Timestamp(end))].sort_values("Date")
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
    ax.plot(df_t["Date"], df_t["Close"],  label="Cloture", color="black",  lw=1.0)
    ax.plot(df_t["Date"], df_t["MA_20"],  label="MA 20",   color="blue",   lw=1.2)
    ax.plot(df_t["Date"], df_t["MA_50"],  label="MA 50",   color="orange", lw=1.2)
    if "BB_Upper" in df_t.columns:
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
        if "BB_Upper" in df_macd.columns:
            a1.fill_between(df_macd["Date"],df_macd["BB_Lower"],df_macd["BB_Upper"],alpha=0.15,color="purple",label="Bollinger")
        a1.legend(fontsize=8); a1.grid(True,alpha=0.3); a1.set_ylabel("Prix ($)")
        a2.plot(df_macd["Date"],df_macd["MACD"],        label="MACD",  color="blue",  lw=1.0)
        a2.plot(df_macd["Date"],df_macd["MACD_Signal"], label="Signal",color="orange",lw=1.0)
        a2.bar(df_macd["Date"],df_macd["MACD_Hist"],
               color=["green" if v>=0 else "red" for v in df_macd["MACD_Hist"].fillna(0)],alpha=0.4,width=1)
        a2.axhline(0,color="black",lw=0.5); a2.legend(fontsize=8); a2.grid(True,alpha=0.3); a2.set_ylabel("MACD")
        plt.tight_layout(); st.pyplot(fig)
    else:
        st.info("Pas assez de donnees pour le MACD.")

with tab4:
    FEATURES_ST = [f for f in ["Open","High","Low","Close","Volume","MA_20","MA_50","RSI_14","Volatility_20","Price_Range","MACD","MACD_Signal","BB_Width"] if f in df_t.columns]
    df_ml2 = df_t.copy(); df_ml2["Target"] = df_ml2["Close"].shift(-1)
    df_ml2 = df_ml2.dropna(subset=FEATURES_ST+["Target"])
    if len(df_ml2) > 100:
        X2 = df_ml2[FEATURES_ST]; y2 = df_ml2["Target"]
        sp = int(len(X2)*0.8)
        sc2 = StandardScaler(); Xtr=sc2.fit_transform(X2.iloc[:sp]); Xte=sc2.transform(X2.iloc[sp:])
        m2 = LinearRegression(); m2.fit(Xtr, y2.iloc[:sp])
        yp = m2.predict(Xte)
        mae2=mean_absolute_error(y2.iloc[sp:],yp); rmse2=np.sqrt(mean_squared_error(y2.iloc[sp:],yp)); r2_2=r2_score(y2.iloc[sp:],yp)
        mc1,mc2,mc3 = st.columns(3)
        mc1.metric("MAE",f"{mae2:.4f}$"); mc2.metric("RMSE",f"{rmse2:.4f}$"); mc3.metric("R2",f"{r2_2:.4f}")
        fig, ax = plt.subplots(figsize=(14,4))
        ax.plot(df_ml2.iloc[sp:]["Date"].values, y2.iloc[sp:].values, label="Reel",  color="blue", lw=1.2)
        ax.plot(df_ml2.iloc[sp:]["Date"].values, yp,                  label="Predit",color="red",  lw=1.2, alpha=0.8)
        ax.set_title(f"{ticker} - Regression Lineaire : Reel vs Predit", fontweight="bold")
        ax.set_ylabel("Prix ($)"); ax.legend(); ax.grid(True,alpha=0.3)
        st.pyplot(fig)
    else:
        st.warning("Pas assez de donnees pour entrainer le modele.")

st.markdown("---")
st.caption("Mini-Projet SG01 - Ons Gaaya | 9000+ Tickers of Stock Market Data (Full History) | Kaggle")
