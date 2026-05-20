import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ============================================================
# CHARGER ET NETTOYER LES DONNÉES
# ============================================================
df = pd.read_csv('diabetes.csv')
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols] = df[cols].replace(0, np.nan)
df[cols] = df[cols].fillna(df[cols].mean())

X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ============================================================
# ENTRAÎNER LES 3 MODÈLES
# ============================================================
lr = LogisticRegression(random_state=42, class_weight='balanced')
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

lr.fit(X_train, y_train); acc_lr = accuracy_score(y_test, lr.predict(X_test))
rf.fit(X_train, y_train); acc_rf = accuracy_score(y_test, rf.predict(X_test))
gb.fit(X_train, y_train); acc_gb = accuracy_score(y_test, gb.predict(X_test))

print(f"✅ Régression Logistique : {acc_lr*100:.2f}%")
print(f"✅ Random Forest         : {acc_rf*100:.2f}%")
print(f"✅ Gradient Boosting     : {acc_gb*100:.2f}%")

# ============================================================
# SAUVEGARDER DANS model.pkl
# ============================================================
data = {
    'model': gb,
    'acc_lr': acc_lr,
    'acc_rf': acc_rf,
    'acc_gb': acc_gb
}

with open('model.pkl', 'wb') as f:
    pickle.dump(data, f)

print("✅ model.pkl sauvegardé avec succès !")
