import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

st.set_page_config(page_title="Stock Dashboard SG01", layout="wide")
st.title("Tableau de Bord Boursier — SG01 Finance & Trading")
st.markdown("**Etudiant(e) :** Ons Gaaya | **Groupe :** GL4 A1 | **Dataset :** 9000+ Tickers (1962–2024)")
st.markdown("---")

@st.cache_data
def load_data():
    file_id    = "1pynCBxcG9UN5SWlE2fyiLE4hVpc9xyEo"
    local_path = "all_stock_data.csv"

    if not os.path.exists(local_path):
        import gdown
        st.info("Téléchargement du dataset depuis Google Drive (3 350 MB — ~5 min)...")
        gdown.download(id=file_id, output=local_path, quiet=False)

    TICKERS = ["AAPL","MSFT","GOOGL","AMZN","TSLA","META","NVDA",
               "JPM","BAC","GS","XOM","WMT","KO","PFE","NFLX"]
    chunks = []
    for chunk in pd.read_csv(local_path, chunksize=100_000):
        filtered = chunk[chunk["Ticker"].isin(TICKERS)]
        if not filtered.empty:
            chunks.append(filtered)
    df = pd.concat(chunks, ignore_index=True)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop(columns=["Dividends", "Stock Splits", "Stock_Splits"], errors="ignore")
    df = df[(df["Close"] > 0) & (df["Volume"] >= 0)]
    df = df.drop_duplicates()
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # ── Feature Engineering ──────────────────────────────
    df["MA_20"]         = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(20).mean())
    df["MA_50"]         = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(50).mean())
    df["Daily_Return"]  = df.groupby("Ticker")["Close"].transform(lambda x: x.pct_change())
    df["Volatility_20"] = df.groupby("Ticker")["Daily_Return"].transform(
        lambda x: x.rolling(20).std() * np.sqrt(252))
    df["Price_Range"]   = df["High"] - df["Low"]

    def compute_rsi(s):
        d = s.diff()
        g = d.clip(lower=0).rolling(14).mean()
        l = (-d.clip(upper=0)).rolling(14).mean()
        return 100 - (100 / (1 + g / l.replace(0, np.nan)))

    df["RSI_14"]      = df.groupby("Ticker")["Close"].transform(compute_rsi)

    def macd_line(s):
        return s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()

    def signal_line(s):
        return macd_line(s).ewm(span=9, adjust=False).mean()

    df["MACD"]        = df.groupby("Ticker")["Close"].transform(macd_line)
    df["MACD_Signal"] = df.groupby("Ticker")["Close"].transform(signal_line)
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    def bb_upper(s):
        return s.rolling(20).mean() + 2 * s.rolling(20).std()

    def bb_lower(s):
        return s.rolling(20).mean() - 2 * s.rolling(20).std()

    df["BB_Upper"] = df.groupby("Ticker")["Close"].transform(bb_upper)
    df["BB_Lower"] = df.groupby("Ticker")["Close"].transform(bb_lower)
    ma20           = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(20).mean())
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / ma20

    # Multi-horizon targets
    for h in [1, 5, 10, 21]:
        df[f"Target_t{h}"] = df.groupby("Ticker")["Close"].transform(lambda x: x.shift(-h))

    return df


with st.spinner("Chargement des données complètes (140 429 lignes, 1962–2024)..."):
    df = load_data()

st.success(f"Données chargées : {len(df):,} lignes | {df['Ticker'].nunique()} tickers | "
           f"{df['Date'].min().year}–{df['Date'].max().year}")

TICKERS = sorted(df["Ticker"].unique())

# ── Sidebar ──────────────────────────────────────────────
st.sidebar.header("Paramètres")
ticker = st.sidebar.selectbox("Choisir un ticker :", TICKERS,
                               index=TICKERS.index("AAPL") if "AAPL" in TICKERS else 0)

ticker_df     = df[df["Ticker"] == ticker]
date_min      = ticker_df["Date"].min().date()
date_max      = ticker_df["Date"].max().date()
default_start = max(date_min, (ticker_df["Date"].max() - pd.DateOffset(years=5)).date())

if st.sidebar.button("Toute la période disponible"):
    selected_start = date_min
    selected_end   = date_max
else:
    selected_start = default_start
    selected_end   = date_max

start, end = st.sidebar.date_input(
    "Période :", [selected_start, selected_end],
    min_value=date_min, max_value=date_max)

st.sidebar.caption(f"Données disponibles pour {ticker} : {date_min} → {date_max}")

df_t = df[
    (df["Ticker"] == ticker) &
    (df["Date"]   >= pd.Timestamp(start)) &
    (df["Date"]   <= pd.Timestamp(end))
].sort_values("Date")

if df_t.empty:
    st.error("Aucune donnée pour cette sélection.")
    st.stop()

# ── KPIs ─────────────────────────────────────────────────
last  = df_t["Close"].iloc[-1]
first = df_t["Close"].iloc[0]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Prix actuel",  f"${last:,.2f}")
c2.metric("Variation",    f"{((last - first) / first * 100):+.2f}%")
c3.metric("Volume moyen", f"{df_t['Volume'].mean():,.0f}")
c4.metric("Volatilité",
          f"{df_t['Volatility_20'].mean() * 100:.1f}%"
          if df_t["Volatility_20"].notna().any() else "N/A")
c5.metric("RSI actuel",
          f"{df_t['RSI_14'].dropna().iloc[-1]:.1f}"
          if df_t["RSI_14"].notna().any() else "N/A")
st.markdown("---")

# ── TABS — now 5 tabs including Multi-Horizons ───────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Prix & MA", "RSI", "MACD", "Modèle ML (t+1)", "Multi-Horizons"])

FEATURES = [f for f in ["Open", "High", "Low", "Close", "Volume",
                         "MA_20", "MA_50", "RSI_14", "Volatility_20",
                         "Price_Range", "MACD", "MACD_Signal", "BB_Width"]
            if f in df_t.columns]

# ── TAB 1 — Prix + MA + Bollinger ────────────────────────
with tab1:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_t["Date"], df_t["Close"],    label="Clôture",   color="black",  lw=1.0)
    ax.plot(df_t["Date"], df_t["MA_20"],    label="MA 20",     color="blue",   lw=1.2)
    ax.plot(df_t["Date"], df_t["MA_50"],    label="MA 50",     color="orange", lw=1.2)
    ax.fill_between(df_t["Date"], df_t["BB_Lower"], df_t["BB_Upper"],
                    alpha=0.1, color="purple", label="Bollinger")
    ax.set_ylabel("Prix ($)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{ticker} — Prix + MA20 + MA50 + Bandes de Bollinger ({start} → {end})",
                 fontweight="bold")
    st.pyplot(fig)
    st.caption("MA20/MA50 : tendance court/long terme. Bollinger : zones de volatilité — "
               "un prix hors bandes signale une anomalie de marché.")

# ── TAB 2 — RSI ───────────────────────────────────────────
with tab2:
    df_rsi = df_t.dropna(subset=["RSI_14"])
    if len(df_rsi) > 0:
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.plot(df_rsi["Date"], df_rsi["RSI_14"], color="purple", lw=1.0)
        ax.axhline(70, color="red",   linestyle="--", lw=1, label="Surachat (70)")
        ax.axhline(30, color="green", linestyle="--", lw=1, label="Survente (30)")
        ax.fill_between(df_rsi["Date"], df_rsi["RSI_14"], 70,
                        where=df_rsi["RSI_14"] > 70, alpha=0.2, color="red")
        ax.fill_between(df_rsi["Date"], df_rsi["RSI_14"], 30,
                        where=df_rsi["RSI_14"] < 30, alpha=0.2, color="green")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{ticker} — RSI (14 jours)", fontweight="bold")
        st.pyplot(fig)
        st.caption("RSI > 70 = surachat (signal de vente potentiel). RSI < 30 = survente (signal d'achat potentiel).")
    else:
        st.info("Pas assez de données pour le RSI sur cette période.")

# ── TAB 3 — MACD ──────────────────────────────────────────
with tab3:
    df_macd = df_t.dropna(subset=["MACD", "MACD_Signal"])
    if len(df_macd) > 0:
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        a1.plot(df_macd["Date"], df_macd["Close"], color="black", lw=1.0, label="Clôture")
        a1.fill_between(df_macd["Date"], df_macd["BB_Lower"], df_macd["BB_Upper"],
                        alpha=0.15, color="purple", label="Bollinger")
        a1.legend(fontsize=8)
        a1.grid(True, alpha=0.3)
        a1.set_ylabel("Prix ($)")
        a2.plot(df_macd["Date"], df_macd["MACD"],        label="MACD",   color="blue",   lw=1.0)
        a2.plot(df_macd["Date"], df_macd["MACD_Signal"], label="Signal", color="orange", lw=1.0)
        colors = ["green" if v >= 0 else "red" for v in df_macd["MACD_Hist"].fillna(0)]
        a2.bar(df_macd["Date"], df_macd["MACD_Hist"], color=colors, alpha=0.4, width=1)
        a2.axhline(0, color="black", lw=0.5)
        a2.legend(fontsize=8)
        a2.grid(True, alpha=0.3)
        a2.set_ylabel("MACD")
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("MACD > Signal = tendance haussière. Histogramme vert = momentum positif croissant.")
    else:
        st.info("Pas assez de données pour le MACD sur cette période.")

# ── TAB 4 — Modèle ML t+1 ────────────────────────────────
with tab4:
    df_ml = df_t.copy()
    df_ml["Target"] = df_ml["Close"].shift(-1)
    df_ml = df_ml.dropna(subset=FEATURES + ["Target"])

    if len(df_ml) > 100:
        X  = df_ml[FEATURES]
        y  = df_ml["Target"]
        sp = int(len(X) * 0.8)

        sc  = StandardScaler()
        Xtr = sc.fit_transform(X.iloc[:sp])
        Xte = sc.transform(X.iloc[sp:])

        model = LinearRegression()
        model.fit(Xtr, y.iloc[:sp])
        yp = model.predict(Xte)

        mae_v  = mean_absolute_error(y.iloc[sp:], yp)
        rmse_v = np.sqrt(mean_squared_error(y.iloc[sp:], yp))
        r2_v   = r2_score(y.iloc[sp:], yp)

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("MAE",  f"${mae_v:.4f}")
        mc2.metric("RMSE", f"${rmse_v:.4f}")
        mc3.metric("R²",   f"{r2_v:.4f}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        dates_test = df_ml.iloc[sp:]["Date"].values
        n_show     = min(500, len(y.iloc[sp:]))

        axes[0].plot(dates_test[:n_show], y.iloc[sp:].values[:n_show],
                     label="Réel",   color="blue", lw=1.2)
        axes[0].plot(dates_test[:n_show], yp[:n_show],
                     label="Prédit", color="red",  lw=1.2, alpha=0.8)
        axes[0].fill_between(dates_test[:n_show],
                             y.iloc[sp:].values[:n_show], yp[:n_show],
                             alpha=0.12, color="orange")
        axes[0].set_title(f"{ticker} — Réel vs Prédit (t+1)", fontweight="bold")
        axes[0].set_ylabel("Prix ($)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        lim = max(y.iloc[sp:].values[:n_show].max(), yp[:n_show].max())
        axes[1].scatter(y.iloc[sp:].values[:n_show], yp[:n_show],
                        alpha=0.3, s=8, color="steelblue")
        axes[1].plot([0, lim], [0, lim], "r--", lw=1.5, label="Ligne parfaite")
        axes[1].set_xlabel("Prix Réel ($)")
        axes[1].set_ylabel("Prix Prédit ($)")
        axes[1].set_title("Scatter : Réel vs Prédit", fontweight="bold")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f"MAE={mae_v:.4f}$ | RMSE={rmse_v:.4f}$ | R²={r2_v:.4f}",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        st.caption(f"R²={r2_v:.4f} : le modèle explique {r2_v*100:.2f}% de la variance. "
                   "La forte autocorrélation des séries boursières explique ce score élevé.")
    else:
        st.warning("Pas assez de données pour entraîner le modèle. Élargissez la période.")

# ── TAB 5 — Multi-Horizons ───────────────────────────────
with tab5:
    st.subheader("Prédiction Multi-Horizons : t+1, t+5, t+10, t+21")
    st.markdown("""
    Ce modèle teste la capacité de la régression linéaire à prédire le prix de clôture
    à **4 horizons différents** : lendemain, 1 semaine, 2 semaines, 1 mois.
    """)

    HORIZONS = [1, 5, 10, 21]
    results_multi = {}

    with st.spinner("Entraînement des 4 modèles multi-horizons..."):
        for h in HORIZONS:
            target_col = f"Target_t{h}"
            df_h = df_t.copy()
            df_h[target_col] = df_h["Close"].shift(-h)
            df_h = df_h.dropna(subset=FEATURES + [target_col])

            if len(df_h) < 100:
                continue

            X_h = df_h[FEATURES]
            y_h = df_h[target_col]
            sp  = int(len(X_h) * 0.8)

            sc_h  = StandardScaler()
            Xtr_h = sc_h.fit_transform(X_h.iloc[:sp])
            Xte_h = sc_h.transform(X_h.iloc[sp:])

            mdl = LinearRegression()
            mdl.fit(Xtr_h, y_h.iloc[:sp])
            yp_h = mdl.predict(Xte_h)

            results_multi[h] = {
                "y_test": y_h.iloc[sp:],
                "y_pred": yp_h,
                "dates":  df_h.iloc[sp:]["Date"].values,
                "mae":    mean_absolute_error(y_h.iloc[sp:], yp_h),
                "rmse":   np.sqrt(mean_squared_error(y_h.iloc[sp:], yp_h)),
                "r2":     r2_score(y_h.iloc[sp:], yp_h),
            }

    if results_multi:
        # ── Metrics table ──
        col_headers = st.columns([2, 2, 2, 2])
        labels = [f"t+{h}" for h in results_multi]
        for col, h in zip(col_headers, results_multi):
            r = results_multi[h]
            period = {1:"Lendemain", 5:"1 semaine", 10:"2 semaines", 21:"1 mois"}[h]
            col.metric(
                label=f"t+{h} — {period}",
                value=f"R² = {r['r2']:.4f}",
                delta=f"MAE = {r['mae']:.4f} $"
            )

        st.markdown("---")

        # ── Degradation charts ──
        horizons_list = list(results_multi.keys())
        maes  = [results_multi[h]["mae"]  for h in horizons_list]
        rmses = [results_multi[h]["rmse"] for h in horizons_list]
        r2s   = [results_multi[h]["r2"]   for h in horizons_list]
        xlabels = [f"t+{h}" for h in horizons_list]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        colors_bar = ["#1565C0","#1976D2","#42A5F5","#90CAF9"]

        axes[0].bar(xlabels, maes, color=colors_bar)
        axes[0].set_title("MAE ($) par Horizon", fontweight="bold")
        axes[0].set_ylabel("Erreur Absolue Moyenne ($)")
        axes[0].grid(True, alpha=0.3, axis="y")
        for i, v in enumerate(maes):
            axes[0].text(i, v + 0.01*max(maes), f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

        axes[1].bar(xlabels, rmses, color=["#1B5E20","#388E3C","#66BB6A","#A5D6A7"])
        axes[1].set_title("RMSE ($) par Horizon", fontweight="bold")
        axes[1].set_ylabel("RMSE ($)")
        axes[1].grid(True, alpha=0.3, axis="y")
        for i, v in enumerate(rmses):
            axes[1].text(i, v + 0.01*max(rmses), f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

        axes[2].plot(xlabels, r2s, "o-", color="#E65100", lw=2.5, ms=10,
                     markerfacecolor="white", markeredgewidth=2.5)
        axes[2].set_title("R² par Horizon", fontweight="bold")
        axes[2].set_ylabel("R²")
        axes[2].set_ylim(min(r2s)*0.999, 1.0005)
        axes[2].grid(True, alpha=0.3)
        for i, v in enumerate(r2s):
            axes[2].text(i, v + 0.00005, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")

        plt.suptitle("Dégradation des Performances avec l'Horizon de Prédiction",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)

        # ── 4-panel réel vs prédit ──
        fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8))
        axes2 = axes2.flatten()
        period_labels = {1:"Lendemain", 5:"1 semaine", 10:"2 semaines", 21:"1 mois"}

        for idx, h in enumerate(horizons_list):
            res    = results_multi[h]
            n_show = min(300, len(res["y_test"]))
            axes2[idx].plot(res["dates"][:n_show], res["y_test"].values[:n_show],
                            label="Réel",   color="blue", lw=1.2, alpha=0.9)
            axes2[idx].plot(res["dates"][:n_show], res["y_pred"][:n_show],
                            label="Prédit", color="red",  lw=1.2, alpha=0.8)
            axes2[idx].fill_between(res["dates"][:n_show],
                                    res["y_test"].values[:n_show], res["y_pred"][:n_show],
                                    alpha=0.12, color="orange")
            axes2[idx].set_title(
                f"t+{h} ({period_labels[h]}) — MAE={res['mae']:.3f}$ | R²={res['r2']:.4f}",
                fontweight="bold")
            axes2[idx].set_ylabel("Prix ($)")
            axes2[idx].legend(fontsize=8)
            axes2[idx].grid(True, alpha=0.3)

        plt.suptitle(f"{ticker} — Réel vs Prédit : 4 Horizons de Prédiction",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig2)

        st.info("""
        **Interprétation :** La régression linéaire reste performante jusqu'à t+21 grâce à la forte
        autocorrélation des séries boursières. Le R² décroît de 0.9997 (t+1) à ~0.9941 (t+21).
        Pour des horizons > 1 mois, des modèles comme **LSTM** ou **XGBoost** seraient plus adaptés.
        """)
    else:
        st.warning("Pas assez de données pour les modèles multi-horizons. Élargissez la période.")

st.markdown("---")
st.caption("Mini-Projet SG01 — Ons Gaaya | GL4 A1 | 9000+ Tickers of Stock Market Data | "
           "Kaggle | 140 429 lignes, 1962–2024")
