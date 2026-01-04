import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib 


# KONFIGURATION

DATA_PATH = "d:/Lidar-projekt/data/processed/frame_features_dataset.csv"
MODEL_PATH = "d:/Lidar-projekt/models/frame_classifier.pkl"


# 1. DATEN VORBEREITEN

try:
    df = pd.read_csv(DATA_PATH)
    print(f"Datensatz erfolgreich geladen. {len(df)} Frames.")
except FileNotFoundError:
    print(f" Fehler: Datensatz nicht gefunden unter {DATA_PATH}. Bitte zuerst 'test_load.py' ausführen.")
    exit()

# Features (X) und Label (Y) definieren
# Wir schließen nicht-numerische Spalten und die ID aus.
X = df.drop(columns=['frame_id', 'scan_type', 'is_wet'])
Y = df['is_wet']

# Trainings- und Test-Sets erstellen
# 80% Training, 20% Test
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

print(f"Trainingsdaten: {len(X_train)} Frames, Testdaten: {len(X_test)} Frames.")


# 2. MODELLTRAINING (Random Forest)

print("\nStarte das Training des Random Forest Classifiers...")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)
print(" Training abgeschlossen.")


# 3. EVALUATION

Y_pred = model.predict(X_test)

print(" MODELL-EVALUATION (Test-Set)")

# Klassifikationsbericht
print(classification_report(Y_test, Y_pred, target_names=['DRY', 'WET']))

# Confusion Matrix
cm = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix:")
print(f"[[DRY_true DRY_false]\n [WET_false WET_true]]")
print(cm)

# Feature Importance
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
top_5_features = feature_importances.nlargest(5)
print("\n--- Top 5 Wichtigste Features zur Klassifizierung ---")
print(top_5_features.to_markdown(numalign="left", stralign="left"))


# 4. MODELL SPEICHERN

import os
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True) 

joblib.dump(model, MODEL_PATH)
print(f"\n Modell erfolgreich gespeichert unter {MODEL_PATH}")