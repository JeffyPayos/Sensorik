import numpy as np
import pandas as pd # NEU: Für DataFrame-Handhabung der Features
from src.loader import load_csv_scandata
from src.viewer import show_frame_2d, animate_2d
from src.feature_extractor import extract_features, compare_wet_dry_stats # NEU: Feature-Logik

FILE_WET = "d:/Lidar-projekt/data/raw/wet_20251207160801_20251207160821.csv"
FILE_DRY = "d:/Lidar-projekt/data/raw/dry_20251207161846_20251207161906.csv"

# =================================================================
# 1. Laden der Daten
# =================================================================
print(f"Starte das Laden der WET-Datei: {FILE_WET}")
frames_wet = load_csv_scandata(FILE_WET)
print(f"🥳 Wet Frames geladen: {len(frames_wet)}")

print(f"\nStarte das Laden der DRY-Datei: {FILE_DRY}")
frames_dry = load_csv_scandata(FILE_DRY)
print(f"🥳 Dry Frames geladen: {len(frames_dry)}")

# =================================================================
# 2. Feature-Extraktion
# =================================================================
if frames_wet:
    df_wet_features = extract_features(frames_wet, scan_type="WET")
    print(f"\n✅ {len(df_wet_features)} WET-Frames in Features umgewandelt.")
    print("\n--- Erste WET Feature-Reihen ---")
    print(df_wet_features.head(3).to_markdown(numalign="left", stralign="left"))
else:
    df_wet_features = pd.DataFrame()

if frames_dry:
    df_dry_features = extract_features(frames_dry, scan_type="DRY")
    print(f"\n✅ {len(df_dry_features)} DRY-Frames in Features umgewandelt.")
else:
    df_dry_features = pd.DataFrame()


# =================================================================
# 3. Quantifizierung (Vergleich)
# =================================================================
if not df_wet_features.empty and not df_dry_features.empty:
    compare_wet_dry_stats(df_wet_features, df_dry_features)

# =================================================================
# 4. Erstellung des Trainingsdatensatzes (NEU)
# =================================================================
if not df_wet_features.empty and not df_dry_features.empty:
    
    # 1. Labels hinzufügen: DRY = 0 (Kein Rauschen/Objekt), WET = 1 (Rauschen/Objekt-Mischung)
    # HINWEIS: Wir labeln hier den FRAME. Später (Issue 5) labeln wir den einzelnen PUNKT.
    # Für den ersten ML-Test (Frame-Klassifizierung) ist das ausreichend.
    df_wet_features['is_wet'] = 1
    df_dry_features['is_wet'] = 0
    
    # 2. Kombinieren der Datensätze
    df_training_data = pd.concat([df_wet_features, df_dry_features], ignore_index=True)
    
    # 3. Features bereinigen (NaNs aus channel-spezifischen Features entfernen, falls aufgetreten)
    df_training_data = df_training_data.fillna(df_training_data.mean(numeric_only=True))

    # 4. Export des finalen Datensatzes
    OUTPUT_PATH = "d:/Lidar-projekt/data/processed/frame_features_dataset.csv"
    df_training_data.to_csv(OUTPUT_PATH, index=False)
    
    print("\n=======================================================")
    print(f"✅ Trainingsdatensatz erfolgreich exportiert:")
    print(f"   Gesamtframes: {len(df_training_data)}")
    print(f"   DRY (Label 0): {len(df_dry_features)} Frames")
    print(f"   WET (Label 1): {len(df_wet_features)} Frames")
    print(f"   Exportpfad: {OUTPUT_PATH}")
    print("=======================================================\n")
    
    #


# =================================================================
# 4. Visualisierung (optional - Animation nur für einen Satz)
# =================================================================
# # Wenn Sie die Animation sehen möchten, die Zeilenkommentare entfernen:
print("Zeige Wet Frame 0 ...")
show_frame_2d(frames_wet[0], "Wet Frame 0 (2D Projektion)")

print("Zeige Dry Frame 0 ...")
show_frame_2d(frames_dry[0], "Dry Frame 0 (2D Projektion)")

animate_2d(frames_wet, "Wet Animation")
animate_2d(frames_dry, "Dry Animation")
