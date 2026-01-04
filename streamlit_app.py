import streamlit as st
import numpy as np
import pandas as pd
import joblib 
import os
import sys
import tempfile
import matplotlib.pyplot as plt


project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root) 

try:
    from src.loader import load_csv_scandata
    from src.feature_extractor import extract_features
except ModuleNotFoundError:
    st.error(" Fehler: Die Module 'src/loader.py' oder 'src/feature_extractor.py' wurden nicht gefunden. Stellen Sie sicher, dass 'src' im Projekt-Root liegt.")
    sys.exit(1)


# 1. KONFIGURATION & MODELLE

MODEL_PATH = "d:/Lidar-projekt/models/frame_classifier.pkl"
NEAR_DISTANCE_THRESHOLD_MM = 2000 # Schwellwert für Nahfeldrauschen (2 Meter)
MIN_RSSI_WET = 100 # Schwellwert für schwachen RSSI

# Lade das Modell einmal (Streamlit-Cache für Performance)
@st.cache_resource
def load_classifier():
    try:
        classifier = joblib.load(MODEL_PATH)
        return classifier
    except FileNotFoundError:
        st.error(f" Fehler: Klassifikationsmodell nicht gefunden unter {MODEL_PATH}. Bitte ml_classifier.py ausführen, um es zu erstellen.")
        return None

classifier = load_classifier()

# Wiederverwendete Plot-Funktion (aus application_demo.py)
def plot_frame_2d(ax, frame, title):
    if frame.size == 0:
        ax.set_title(title + " (Leer)")
        return
        
    # Die ersten 4 Spalten sind Theta, Phi, Distanz, RSSI
    theta = frame[:, 0]
    r = frame[:, 2] / 1000.0 # Distanz in Meter
    rssi = frame[:, 3]

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Scatte plot
    ax.scatter(x, y, c=rssi, s=10, cmap="viridis", alpha=0.7)
    
    ax.set_xlim(-5, 10)
    ax.set_ylim(-5, 10) 
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    

# 2. HAUPTFUNKTION DER STREAMLIT-APP

st.title("🌧️ Lidar-Rauschfilter (KI-gesteuert)")
st.write("Laden Sie eine Lidar-Rohdaten-CSV-Datei hoch, um sie automatisch auf Nebel/Wasser-Rauschen zu prüfen und zu bereinigen.")

if classifier is None:
    st.stop()

uploaded_file = st.file_uploader("Wählen Sie eine Lidar CSV-Datei", type=['csv'])

if uploaded_file is not None:
    # 1. Temporäre Speicherung der hochgeladenen Datei
    # Streamlit benötigt einen physikalischen Pfad, um ihn an die Loader-Funktion zu übergeben
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    st.info(f"Datei '{uploaded_file.name}' wurde hochgeladen und wird verarbeitet...")
    
    try:
        # 2. Daten laden
        all_frames = load_csv_scandata(tmp_file_path)
        total_frames = len(all_frames)

        if total_frames == 0:
            st.warning("Keine Frames in der Datei gefunden. Bitte prüfen Sie das Format.")
            st.stop()
            
        # 3. Features extrahieren und Klassifizieren
        df_features = extract_features(all_frames, scan_type="STREAMLIT")
        X_features = df_features.drop(columns=['frame_id', 'scan_type']).fillna(0)
        predictions = classifier.predict(X_features)
        
        total_wet_frames = np.sum(predictions)
        st.metric(label="Frames als Rauschen (WET) klassifiziert", value=f"{total_wet_frames} / {total_frames}")

        # 4. Filterung anwenden
        filtered_frames = []
        total_noise_removed = 0
        
        for i, (frame, is_wet) in enumerate(zip(all_frames, predictions)):
            current_frame = frame.copy()
            
            if is_wet == 1:
                # Annahme: Lidar-Frames sind (Theta, Phi, Distanz_mm, RSSI, is_valid, timestamp)
                distance = frame[:, 2] 
                rssi = frame[:, 3]     
                
                # Filterregel: Nah und schwaches Signal
                is_noise = (distance < NEAR_DISTANCE_THRESHOLD_MM) & (rssi < MIN_RSSI_WET)
                
                # Nur die NICHT-Rauschpunkte behalten
                current_frame = frame[~is_noise]
                total_noise_removed += np.sum(is_noise)
                
            filtered_frames.append(current_frame)
        
        st.success(f"Filterung abgeschlossen: Insgesamt {total_noise_removed} Rauschpunkte entfernt.")

        # 5. Visualisierung
        st.header("Visualisierung des Filters (Beispiel Frame)")
        
        # Wähle den ersten Frame, der als WET klassifiziert wurde (oder Frame 0)
        wet_indices = np.where(predictions == 1)[0]
        example_frame_index = wet_indices[0] if len(wet_indices) > 0 else 0
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        plot_frame_2d(axes[0], all_frames[example_frame_index], f"Original (Frame {example_frame_index})")
        plot_frame_2d(axes[1], filtered_frames[example_frame_index], f"Gefiltert (Frame {example_frame_index})")
        
        fig.suptitle(f"Frame {example_frame_index}: {('WET' if predictions[example_frame_index] else 'DRY')} - Filterung", fontsize=16)
        st.pyplot(fig)

        # 6. Download vorbereiten 
        st.header("Bereinigte Daten speichern")
        # Alle gefilterten Frames zu einem einzigen Array verketten
        filtered_data_flat = np.concatenate(filtered_frames, axis=0)
        
      
        df_export = pd.DataFrame(
            filtered_data_flat, 
            columns=['theta', 'phi', 'distance_mm', 'rssi'] 
        )
        
        # Exportiere als CSV und biete den Download an
        csv_data = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=" Gefilterte Daten herunterladen (.csv)",
            data=csv_data,
            file_name=uploaded_file.name.replace('.csv', '_CLEANED.csv'),
            mime='text/csv',
        )
        
    except Exception as e:
        st.error(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
        st.exception(e) # Zeigt detaillierten Fehler in der App an
        
    finally:
        # Wichtig: Temporäre Datei löschen
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
             os.unlink(tmp_file_path)