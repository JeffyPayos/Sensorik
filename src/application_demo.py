import numpy as np
import pandas as pd
import joblib 
import os
from src.loader import load_csv_scandata
from src.feature_extractor import extract_features 
from src.viewer import show_frame_2d 
import matplotlib.pyplot as plt


# KONFIGURATION & MODELLE

MODEL_PATH = "d:/Lidar-projekt/models/frame_classifier.pkl"
WET_FILE_PATH = 'd:/Lidar-projekt/data/raw/wet_20251207160801_20251207160821.csv'
FRAME_INDEX = 199 # Frame zur Demonstration

# Schwellenwerte für das physikalische Filtering, basierend auf der Analyse
NEAR_DISTANCE_THRESHOLD_MM = 2000 # Rauschen ist oft nah (2m)
MIN_RSSI_WET = 100 # Im Wet-Scan ist der RSSI-Mittelwert extrem niedrig (ca. 127)

# 1. MODELL LADEN

try:
    classifier = joblib.load(MODEL_PATH)
    print(f" Klassifikationsmodell erfolgreich geladen: {MODEL_PATH}")
except FileNotFoundError:
    print(f" Fehler: Modell nicht gefunden unter {MODEL_PATH}. Bitte zuerst 'ml_classifier.py' ausführen.")
    exit()


# 2. FRAME LADEN UND KLASSIFIZIEREN

print(f"\nLade Frame {FRAME_INDEX} aus dem WET-Datensatz...")
all_frames_wet = load_csv_scandata(WET_FILE_PATH)
if not all_frames_wet or FRAME_INDEX >= len(all_frames_wet):
    print(f"❌ Fehler: Frame {FRAME_INDEX} nicht gefunden.")
    exit()

original_frame = all_frames_wet[FRAME_INDEX]
print(f"Original Frame geladen: {len(original_frame)} Punkte.")

# Feature-Extraktion für den einzelnen Frame 
df_features_single = extract_features([original_frame], scan_type="UNKNOWN")

# Bereinigung: NaN-Werte im Test-Frame auffüllen
# Hier füllen wir einfach mit 0 auf, da es nur ein Test ist.
X_test_frame = df_features_single.drop(columns=['frame_id', 'scan_type']).fillna(0)

# Klassifizierung
prediction = classifier.predict(X_test_frame)
prediction_label = "WET" if prediction[0] == 1 else "DRY"
print(f"Modell-Vorhersage für Frame {FRAME_INDEX}: >>> {prediction_label} <<<")


# 3. FILTER-LOGIK ANWENDEN

filtered_frame = original_frame.copy() 
noise_removed_count = 0

if prediction_label == "WET":
    print("Filter wird angewendet: Entferne Nahfeld-Punkte mit geringem RSSI.")
    
    # Trenne Distanz und RSSI
    distance = original_frame[:, 2] # Distanz in mm
    rssi = original_frame[:, 3]     # RSSI
    
    #  FILTER-REGEL BASIEREND AUF FEATURE-ANALYSE
    # Wir filtern Punkte, die sehr nah sind UND eine schwache Reflektion haben (typisch für Nebel/Spritzer)
    is_noise = (distance < NEAR_DISTANCE_THRESHOLD_MM) & (rssi < MIN_RSSI_WET)
    
    # Wähle alle Punkte, die NICHT als Rauschen identifiziert wurden
    filtered_frame = original_frame[~is_noise]
    noise_removed_count = np.sum(is_noise)
    
    print(f" Filterung abgeschlossen: {noise_removed_count} Rauschpunkte entfernt.")
else:
    print("Frame ist DRY, keine Filterung notwendig.")



# 4. VISUALISIERUNG (Original vs. Gefiltert)

# Wir verwenden die 2D-Visualisierung, um den Unterschied klar zu zeigen
print("\nStarte Visualisierungs-Vergleich...")

# Plot-Setup
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# Hilfsfunktion, um den 2D Plot zu erstellen
def plot_frame_2d(ax, frame, title):
    if frame.size == 0:
        ax.set_title(title + " (Leer)")
        return
        
    theta = frame[:, 0]
    r = frame[:, 2] / 1000.0
    rssi = frame[:, 3]

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    sc = ax.scatter(x, y, c=rssi, s=10, cmap="viridis", alpha=0.6)
    
    ax.set_xlim(-5, 10)
    ax.set_ylim(-5, 10) 
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    
    return sc

# 1. Original Plot
plot_frame_2d(axes[0], original_frame, f"Original (WET) – {len(original_frame)} Pkt.")

# 2. Gefilterter Plot
plot_frame_2d(axes[1], filtered_frame, f"Gefiltert (Noise: {noise_removed_count} Pkt.) – {len(filtered_frame)} Pkt.")

fig.suptitle(f"Frame {FRAME_INDEX}: Rauschfilter-Demonstration (Klassifiziert als: {prediction_label})", fontsize=16)
plt.tight_layout()
plt.show()