import streamlit as st
import pickle       
import numpy as np      
import time   

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Prédiction du Diabète", page_icon="🩺", layout="wide")

def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # Évite que le code plante si le fichier CSS est absent pendant un test
        pass

load_css("style.css")

# ============================================================
# CHARGER LE MODÈLE PKL
# ============================================================
@st.cache_resource
def load_model():
    # Simulation ou chargement réel
    try:
        with open('model.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['acc_lr'], data['acc_rf'], data['acc_gb']
    except FileNotFoundError:
        # Valeurs de secours pour le test si model.pkl n'est pas là
        return None, 0.76, 0.78, 0.82

model, acc_lr, acc_rf, acc_gb = load_model()

# ============================================================
# SIDEBAR (Uniquement pour les entrées)
# ============================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚡ Informations</div>', unsafe_allow_html=True)
    st.markdown("---")

    glucose     = st.slider("Glucose (mg/dL)",     0, 200, 148)
    bmi         = st.slider("BMI",                0.0, 67.1, 33.6)
    age         = st.slider("Âge",                21, 81, 50)
    insulin     = st.slider("Insuline (µU/mL)",    0, 846, 80)
    pregnancies = st.slider("Grossesses",          0, 17, 6)
    blood_press = st.slider("Pression artérielle", 0, 122, 72)
    skin        = st.slider("Épaisseur peau",      0, 99, 35)
    dpf         = st.slider("Facteur génétique",   0.0, 2.42, 0.627)

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

# ============================================================
# MAIN
# ============================================================
st.markdown('<div class="main-title">🩺 Prédiction du Diabète</div>', unsafe_allow_html=True)
st.markdown('<div class="main-sub">Pima Indians Diabetes Dataset</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="acc-big-card">
    <div>
        <div class="acc-number">{acc_gb*100:.2f}%</div>
        <div class="acc-label">Meilleure accuracy</div>
    </div>
    <div class="acc-badge">Gradient Boosting</div>
</div>
""", unsafe_allow_html=True)

# On place le bouton au centre pour éviter le bug de rafraîchissement de la sidebar
predict_btn = st.button("⊕ Lancer l'Analyse Médicale", use_container_width=True)

# ============================================================
# RÉSULTAT
# ============================================================
if predict_btn and model is not None:
    with st.spinner("Analyse des constantes en cours..."):
        time.sleep(0.5) # Réduit à 0.5s pour une meilleure fluidité

    patient     = np.array([[pregnancies, glucose, blood_press, skin, insulin, bmi, dpf, age]])
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
elif model is None:
    st.warning("Fichier 'model.pkl' introuvable. Mode démonstration actif.")
else:
    st.markdown("""
    <div class="waiting-card">
        <div class="waiting-icon">🩺</div>
        <div class="waiting-text">
            Ajustez les paramètres dans le panneau de gauche<br>
            puis cliquez sur le bouton <strong>Lancer l'Analyse Médicale</strong> ci-dessus.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# COMPARAISON MODÈLES
# ============================================================
st.markdown("### 📊 Performance des algorithmes entraînés")
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