"""
 Prédiction du Trafic Urbain — Smart City
Application Streamlit complète
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, json
from sklearn.preprocessing import LabelEncoder

# ──────────────────────────────────────────────
# CONFIG PAGE
# ──────────────────────────────────────────────
st.set_page_config(
    page_title=" Prédiction du Trafic Urbain",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(90deg, #1a73e8, #0d47a1);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.2rem;
    }
    .subtitle { text-align: center; color: #5f6368; font-size: 1rem; margin-bottom: 2rem; }
    .result-box {
        background: linear-gradient(135deg, #0d47a1, #1565c0);
        color: white; border-radius: 16px; padding: 2rem;
        text-align: center; margin: 1rem 0;
    }
    .result-volume { font-size: 3.5rem; font-weight: 900; }
    .result-label  { font-size: 1rem; opacity: 0.85; }
    .sidebar-section { font-weight: 700; color: #1a73e8; font-size: 1rem; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# CHARGEMENT MODÈLES & META
# ──────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("model_rf.pkl", "rb") as f: rf = pickle.load(f)
    with open("model_gb.pkl", "rb") as f: gb = pickle.load(f)
    with open("model_meta.json", "r") as f: meta = json.load(f)
    return rf, gb, meta

@st.cache_data
def load_data():
    df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")
    df["date_time"]   = pd.to_datetime(df["date_time"])
    df = df.sort_values("date_time").reset_index(drop=True)
    df["hour"]        = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["month"]       = df["date_time"].dt.month
    df["year"]        = df["date_time"].dt.year
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday"]  = df["holiday"].notna().astype(int)
    df["traffic_rolling_3h"]  = df["traffic_volume"].rolling(3,  min_periods=1).mean()
    df["traffic_rolling_24h"] = df["traffic_volume"].rolling(24, min_periods=1).mean()
    return df

rf_model, gb_model, meta = load_models()
df = load_data()

FEATURES       = meta["features"]
WEATHER_CLASSES = meta["weather_classes"]
MOY_HIST       = df["traffic_volume"].mean()

le = LabelEncoder()
le.classes_ = np.array(WEATHER_CLASSES)

MODELS = {
    " Random Forest (Recommandé — R²=0.987)": rf_model,
    " Gradient Boosting (R²=0.975)": gb_model,
}


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("##  Traffic Predictor")
    st.markdown("---")

    st.markdown('<p class="sidebar-section"> Modèle ML</p>', unsafe_allow_html=True)
    model_name     = st.selectbox("Modèle", list(MODELS.keys()), label_visibility="collapsed")
    selected_model = MODELS[model_name]

    st.markdown("---")
    st.markdown('<p class="sidebar-section"> Date & Heure</p>', unsafe_allow_html=True)
    hour        = st.slider("Heure", 0, 23, 8, format="%dh")
    day_of_week = st.selectbox("Jour", ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"])
    month       = st.selectbox("Mois", ["Janvier","Février","Mars","Avril","Mai","Juin",
                                         "Juillet","Août","Septembre","Octobre","Novembre","Décembre"])
    year        = st.selectbox("Année", [2022,2023,2024,2025,2026], index=3)
    is_holiday  = st.checkbox("Jour férié ")

    st.markdown("---")
    st.markdown('<p class="sidebar-section"> Météo</p>', unsafe_allow_html=True)
    temp_c      = st.slider("Température (°C)", -30, 45, 15)
    weather_main = st.selectbox("Condition météo", WEATHER_CLASSES)
    clouds_all   = st.slider("Couverture nuageuse (%)", 0, 100, 20)
    rain_1h      = st.slider("Pluie (mm/h)", 0.0, 50.0, 0.0, step=0.5)
    snow_1h      = st.slider("Neige (mm/h)", 0.0, 10.0, 0.0, step=0.1)

    st.markdown("---")
    st.markdown('<p class="sidebar-section"> Contexte trafic</p>', unsafe_allow_html=True)
    rolling_3h  = st.slider("Trafic moyen 3h (véh/h)", 0, 7500, int(MOY_HIST), step=100)
    rolling_24h = st.slider("Trafic moyen 24h (véh/h)", 0, 7500, int(MOY_HIST), step=100)


# ──────────────────────────────────────────────
# FONCTION PRÉDICTION
# ──────────────────────────────────────────────
def make_prediction(model):
    dow_map   = {"Lundi":0,"Mardi":1,"Mercredi":2,"Jeudi":3,"Vendredi":4,"Samedi":5,"Dimanche":6}
    month_map = {"Janvier":1,"Février":2,"Mars":3,"Avril":4,"Mai":5,"Juin":6,
                 "Juillet":7,"Août":8,"Septembre":9,"Octobre":10,"Novembre":11,"Décembre":12}
    dow_val   = dow_map[day_of_week]
    month_val = month_map[month]
    w_enc     = le.transform([weather_main])[0]
    is_wknd   = 1 if dow_val >= 5 else 0

    row = pd.DataFrame([{
        "hour": hour, "day_of_week": dow_val, "month": month_val, "year": year,
        "is_weekend": is_wknd, "is_holiday": int(is_holiday),
        "temp": temp_c + 273.15, "rain_1h": rain_1h, "snow_1h": snow_1h,
        "clouds_all": clouds_all, "weather_encoded": w_enc,
        "traffic_rolling_3h": rolling_3h, "traffic_rolling_24h": rolling_24h
    }])[FEATURES]
    return int(model.predict(row)[0])

def traffic_level(vol):
    if vol < 2000:   return " Faible",  "Trafic fluide"
    elif vol < 4500: return " Modéré",  "Circulation normale"
    else:            return " Dense",   "Trafic chargé"


# ──────────────────────────────────────────────
# PAGE PRINCIPALE
# ──────────────────────────────────────────────
st.markdown('<h1 class="main-title"> Prédiction du Trafic Urbain</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Smart City · Machine Learning · Interstate 94, Minnesota, USA</p>',
            unsafe_allow_html=True)

tabs = st.tabs([" Prédiction", "Exploration", "Performance des Modèles", " Importance Variables", " Rapport"])


# ──── TAB 1 : PRÉDICTION ────
with tabs[0]:
    col1, col2 = st.columns([1.3, 1], gap="large")

    with col1:
        st.subheader(" Paramètres sélectionnés")
        c1, c2, c3 = st.columns(3)
        c1.metric(" Heure",   f"{hour}h00")
        c2.metric(" Jour",    day_of_week[:3])
        c3.metric(" Temp.",  f"{temp_c}°C")
        c1.metric(" Météo",  weather_main)
        c2.metric("Pluie", f"{rain_1h} mm/h")
        c3.metric(" Neige",  f"{snow_1h} mm/h")

        vol             = make_prediction(selected_model)
        level, desc     = traffic_level(vol)
        pct             = min(vol / 7280, 1.0)
        bar_color       = "#2e7d32" if pct < 0.35 else ("#f57f17" if pct < 0.70 else "#c62828")

        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Volume de trafic prédit</div>
            <div class="result-volume">{vol:,}</div>
            <div class="result-label">véhicules / heure</div>
            <br>
            <div style="font-size:1.4rem;font-weight:700;">{level}</div>
            <div style="opacity:0.8;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:0.5rem;">
            <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#666;">
                <span>0</span><span>Max : 7 280</span></div>
            <div style="background:#e0e0e0;border-radius:8px;height:14px;margin-top:4px;">
                <div style="width:{pct*100:.1f}%;background:{bar_color};height:100%;border-radius:8px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader(" Prévision sur 24h")
        dow_map2   = {"Lundi":0,"Mardi":1,"Mercredi":2,"Jeudi":3,"Vendredi":4,"Samedi":5,"Dimanche":6}
        month_map2 = {"Janvier":1,"Février":2,"Mars":3,"Avril":4,"Mai":5,"Juin":6,
                      "Juillet":7,"Août":8,"Septembre":9,"Octobre":10,"Novembre":11,"Décembre":12}
        w_enc = le.transform([weather_main])[0]
        dow_v = dow_map2[day_of_week]; month_v = month_map2[month]
        is_wk = 1 if dow_v >= 5 else 0

        rows = []
        for h in range(24):
            rows.append({"hour":h,"day_of_week":dow_v,"month":month_v,"year":year,
                         "is_weekend":is_wk,"is_holiday":int(is_holiday),
                         "temp":temp_c+273.15,"rain_1h":rain_1h,"snow_1h":snow_1h,
                         "clouds_all":clouds_all,"weather_encoded":w_enc,
                         "traffic_rolling_3h":rolling_3h,"traffic_rolling_24h":rolling_24h})
        X_day   = pd.DataFrame(rows)[FEATURES]
        preds_d = selected_model.predict(X_day).astype(int)

        chart_df = pd.DataFrame({"Heure": range(24), "Trafic prédit": preds_d})
        st.line_chart(chart_df.set_index("Heure"), height=280, use_container_width=True)

        peak_h = int(np.argmax(preds_d))
        st.info(f"**Pic de trafic :** {peak_h}h00 → {preds_d[peak_h]:,} véh/h")


# ──── TAB 2 : EXPLORATION ────
with tabs[1]:
    st.subheader(" Aperçu du dataset")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Lignes",   f"{len(df):,}")
    c2.metric("Colonnes", len(df.columns))
    c3.metric("Période",  "2012–2018")
    c4.metric("Fréquence","Horaire")

    st.dataframe(df.head(50), use_container_width=True, height=280)
    st.markdown("---")
    ca, cb = st.columns(2)

    with ca:
        st.subheader(" Trafic moyen par heure")
        hourly = df.groupby("hour")["traffic_volume"].mean().reset_index()
        st.bar_chart(hourly.set_index("hour"), height=260)

    with cb:
        st.subheader(" Trafic moyen par jour")
        days_fr = ["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"]
        daily   = df.groupby("day_of_week")["traffic_volume"].mean().reset_index()
        daily["jour"] = daily["day_of_week"].map(lambda x: days_fr[x])
        st.bar_chart(daily.set_index("jour")[["traffic_volume"]], height=260)

    st.subheader(" Trafic moyen par condition météo")
    wt = df.groupby("weather_main")["traffic_volume"].mean().sort_values(ascending=False)
    st.bar_chart(wt, height=240)

    st.subheader(" Statistiques descriptives")
    st.dataframe(df[["temp","rain_1h","snow_1h","clouds_all","traffic_volume",
                      "traffic_rolling_3h","traffic_rolling_24h"]].describe().round(2),
                 use_container_width=True)


# ──── TAB 3 : PERFORMANCE ────
with tabs[2]:
    st.subheader(" Comparaison des modèles")
    results = meta["results"]
    for name, r in sorted(results.items(), key=lambda x: -x[1]["R2"]):
        best = r["R2"] == max(v["R2"] for v in results.values())
        badge = " Meilleur modèle" if best else ""
        bg = "#e8f5e9" if best else "#f5f5f5"
        st.markdown(f"""
        <div style="background:{bg};border-radius:12px;padding:1rem 1.5rem;margin:0.5rem 0;
                    border-left:5px solid {'#2e7d32' if best else '#9e9e9e'};">
            <b style="font-size:1.1rem;">{name}</b> {badge}<br>
            <span style="color:#555;">MAE : <b>{r['MAE']:.0f}</b> véh/h &nbsp;|&nbsp;
            RMSE : <b>{r['RMSE']:.0f}</b> véh/h &nbsp;|&nbsp;
            R² : <b>{r['R2']:.4f}</b></span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Réel vs Prédit — Random Forest (200 points)")
    sample_df = pd.DataFrame({
        "Réel":       meta["y_test_sample"][:200],
        "Prédit (RF)":[int(x) for x in meta["pred_sample"][:200]]
    })
    st.line_chart(sample_df, height=320)


# ──── TAB 4 : IMPORTANCE ────
with tabs[3]:
    st.subheader(" Importance des variables — Random Forest")
    fi = meta["feature_importance"]
    fi_df = pd.DataFrame(list(fi.items()), columns=["Variable","Importance"])
    fi_df = fi_df.sort_values("Importance", ascending=False)

    labels_fr = {
        "hour":"Heure de la journée","day_of_week":"Jour de la semaine",
        "is_weekend":"Week-end","temp":"Température","month":"Mois","year":"Année",
        "clouds_all":"Couverture nuageuse","weather_encoded":"Condition météo",
        "rain_1h":"Pluie (mm/h)","snow_1h":"Neige (mm/h)","is_holiday":"Jour férié",
        "traffic_rolling_3h":"Moyenne glissante 3h ★",
        "traffic_rolling_24h":"Moyenne glissante 24h"
    }
    fi_df["Variable FR"]    = fi_df["Variable"].map(labels_fr)
    fi_df["Importance (%)"] = (fi_df["Importance"] * 100).round(2)

    st.bar_chart(fi_df.set_index("Variable FR")["Importance (%)"], height=380)
    st.dataframe(fi_df[["Variable FR","Importance (%)"]].rename(columns={"Variable FR":"Variable"}),
                 use_container_width=True, hide_index=True)

    st.info("""
    **💡 Interprétation :**
    - **Moyenne glissante 3h** (78.8%) : signal le plus fort — autocorrélation temporelle du trafic.
    - **Heure** (19.7%) : structure les pics matin/soir.
    - **La météo** a un impact secondaire mais réel.
    """)


# ──── TAB 5 : RAPPORT ────
with tabs[4]:
    st.subheader(" Rapport Synthétique du Modèle")

    st.markdown("### 1. Objectif")
    st.info("Prédire le volume horaire de trafic routier (véhicules/heure) sur l'Interstate 94 "
            "à partir de l'heure, du jour de la semaine et des conditions météorologiques, "
            "dans un contexte Smart City de gestion intelligente du trafic.")

    st.markdown("### 2. Méthode")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Dataset :**")
        st.markdown("- 48 204 enregistrements horaires\n- Période : 2012–2018\n- Interstate 94, Minnesota")
        st.markdown("**Features créées :**")
        st.markdown("- Variables temporelles (heure, jour, mois)\n"
                    "- Variables binaires (is_weekend, is_holiday)\n"
                    "- **Moyennes glissantes 3h et 24h** ← clé\n"
                    "- Encodage météo (LabelEncoder)")
    with col2:
        st.markdown("**Modèles comparés :**")
        st.markdown("1. Régression Linéaire (baseline)\n"
                    "2. Gradient Boosting\n"
                    "3. **Random Forest** ← retenu")
        st.markdown("**Évaluation :**")
        st.markdown("- MAE, RMSE, R²\n- Split 80/20\n- Validation croisée 5-fold")

    st.markdown("### 3. Résultats")
    res_df = pd.DataFrame([
        {"Modèle":"Régression Linéaire", "MAE":477,  "RMSE":672,  "R²":0.886},
        {"Modèle":"Gradient Boosting",   "MAE":211,  "RMSE":314,  "R²":0.975},
        {"Modèle":"Random Forest ",    "MAE":129,  "RMSE":224,  "R²":0.987},
    ])
    st.dataframe(res_df, use_container_width=True, hide_index=True)
    st.success("**Meilleur modèle : Random Forest — R² = 0.987** (98.7% de la variance expliquée, erreur moyenne ±129 véh/h)")

    st.markdown("### 4. Limites")
    st.warning(
        "-  Ne prend pas en compte les accidents et incidents routiers\n"
        "-  Ne modélise pas les travaux et déviations de circulation\n"
        "-  Limité à l'Interstate 94 (faible généralisation géographique)\n"
        "-  La rolling_3h nécessite de connaître le trafic récent\n"
        "-  Données antérieures à 2018 (pas de comportement post-COVID)"
    )


# ──── FOOTER ────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#9e9e9e;font-size:0.82rem;'>"
    "Smart City Traffic Prediction · Random Forest R²=0.987 · "
    "Dataset : UCI Metro Interstate Traffic Volume (2012–2018)"
    "</div>", unsafe_allow_html=True
)
