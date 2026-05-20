import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Prédiction du Diabète", page_icon="🩺", layout="wide")

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ============================================================
# DONNÉES + MODÈLES
# ============================================================
@st.cache_data
def load_and_train():
    df = pd.read_csv('diabetes.csv')
    cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols] = df[cols].replace(0, np.nan)
    df[cols] = df[cols].fillna(df[cols].mean())
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LogisticRegression(random_state=42, class_weight='balanced')
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    lr.fit(X_train, y_train); acc_lr = accuracy_score(y_test, lr.predict(X_test))
    rf.fit(X_train, y_train); acc_rf = accuracy_score(y_test, rf.predict(X_test))
    gb.fit(X_train, y_train); acc_gb = accuracy_score(y_test, gb.predict(X_test))
    return gb, acc_lr, acc_rf, acc_gb

model, acc_lr, acc_rf, acc_gb = load_and_train()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚡ Informations</div>', unsafe_allow_html=True)
    st.markdown("---")

    glucose     = st.slider("Glucose (mg/dL)",    0, 200, 148)
    bmi         = st.slider("BMI",               0.0, 67.1, 33.6)
    age         = st.slider("Âge",               21, 81, 50)
    insulin     = st.slider("Insuline (µU/mL)",   0, 846, 80)
    pregnancies = st.slider("Grossesses",         0, 17, 6)
    blood_press = st.slider("Pression artérielle",0, 122, 72)
    skin        = st.slider("Épaisseur peau",     0, 99, 35)
    dpf         = st.slider("Facteur génétique",  0.0, 2.42, 0.627)

    # Metrics sidebar
    st.markdown("---")
    g_pct = int((glucose / 200) * 100)
    b_pct = int((bmi / 67.1) * 100)
    a_pct = int(((age - 21) / 60) * 100)
    i_pct = int((insulin / 846) * 100)

    st.markdown(f"""
    <div class="sb-metric">
        <div class="sb-metric-label">Glucose</div>
        <div class="sb-metric-val">{glucose} mg/dL</div>
        <div class="sb-bar-bg"><div class="sb-bar" style="width:{g_pct}%"></div></div>
    </div>
    <div class="sb-metric">
        <div class="sb-metric-label">BMI</div>
        <div class="sb-metric-val">{bmi}</div>
        <div class="sb-bar-bg"><div class="sb-bar" style="width:{b_pct}%"></div></div>
    </div>
    <div class="sb-metric">
        <div class="sb-metric-label">Âge</div>
        <div class="sb-metric-val">{age} ans</div>
        <div class="sb-bar-bg"><div class="sb-bar" style="width:{a_pct}%"></div></div>
    </div>
    <div class="sb-metric">
        <div class="sb-metric-label">Insuline</div>
        <div class="sb-metric-val">{insulin} µU/mL</div>
        <div class="sb-bar-bg"><div class="sb-bar" style="width:{i_pct}%"></div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    predict_btn = st.button("⊕ Prédire")

# ============================================================
# MAIN
# ============================================================
st.markdown('<div class="main-title">🩺 Prédiction du Diabète</div>', unsafe_allow_html=True)
st.markdown('<div class="main-sub">Pima Indians Diabetes Dataset</div>', unsafe_allow_html=True)

# Accuracy big card
st.markdown(f"""
<div class="acc-big-card">
    <div>
        <div class="acc-number">{acc_gb*100:.2f}%</div>
        <div class="acc-label">Meilleure accuracy</div>
    </div>
    <div class="acc-badge">Gradient Boosting</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# RÉSULTAT
# ============================================================
if predict_btn:
    with st.spinner("Analyse en cours..."):
        time.sleep(1)

    patient = np.array([[pregnancies, glucose, blood_press, skin, insulin, bmi, dpf, age]])
    prediction  = model.predict(patient)[0]
    probabilite = model.predict_proba(patient)[0][1] * 100

    if prediction == 1:
        st.markdown(f"""
        <div class="result-diab">
            <div class="result-icon-diab">⚠</div>
            <div class="result-title-diab">Diabétique détecté</div>
            <div class="result-prob">Probabilité : {probabilite:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-norm">
            <div class="result-icon-norm">✓</div>
            <div class="result-title-norm">Non Diabétique</div>
            <div class="result-prob">Probabilité de diabète : {probabilite:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="waiting-card">
        <div class="waiting-icon">🩺</div>
        <div class="waiting-text">
            Ajustez les paramètres à gauche<br>
            puis cliquez sur <strong style="color:#3fb0ff">Prédire</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# COMPARAISON MODÈLES
# ============================================================
st.markdown(f"""
<div class="model-grid">
    <div class="model-card">
        <div class="model-name">Log. Reg.</div>
        <div class="model-acc">{acc_lr*100:.2f}%</div>
    </div>
    <div class="model-card">
        <div class="model-name">Rand. Forest</div>
        <div class="model-acc">{acc_rf*100:.2f}%</div>
    </div>
    <div class="model-card best">
        <div class="model-name">Grad. Boost</div>
        <div class="model-acc best-acc">{acc_gb*100:.2f}%</div>
    </div>
</div>
""", unsafe_allow_html=True)
