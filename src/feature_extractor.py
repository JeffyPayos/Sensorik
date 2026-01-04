import numpy as np
import pandas as pd
from typing import List

# Konfiguration (kann später in eine config.py verschoben werden)
NUM_CHANNELS = 16
# Schwellenwert: Alles unter 2000 mm (2 Meter) ist potenzielles Nahfeld-Rauschen bei Nebel/Regen
NEAR_DISTANCE_THRESHOLD_MM = 2000 


def extract_features(frames: List[np.ndarray], scan_type: str) -> pd.DataFrame:
    """
    Extrahiert statistische Merkmale aus einer Liste von Lidar-Frames.

    Jeder Frame ist ein Nx4-Array: (H-angle, V-index, distance, RSSI)
    """
    all_features = []
    
    # 1. Feature-Extraktion pro Frame
    for idx, frame in enumerate(frames):
        
        if frame.size == 0:
            continue
            
        distance = frame[:, 2]  # Distanz in mm
        rssi = frame[:, 3]      # RSSI-Wert
        channel_index = frame[:, 1].astype(int) # Kanalindex (0-15)
        
        frame_features = {
            'frame_id': idx,
            'scan_type': scan_type,
            'valid_points_count': len(frame),
        }
        
        # --- Globale Features ---
        frame_features['global_mean_distance'] = np.mean(distance)
        frame_features['global_std_distance'] = np.std(distance)
        frame_features['global_mean_rssi'] = np.mean(rssi)
        frame_features['global_std_rssi'] = np.std(rssi)
        
        # Verhältnis naher / ferner Punkte (Schlüssel-Feature für Rauschen)
        near_points_count = np.sum(distance < NEAR_DISTANCE_THRESHOLD_MM)
        frame_features['near_points_ratio'] = near_points_count / len(frame)

        # Optional: Bounding Box (Hier nur für die Distanz im R-Winkel)
        frame_features['dist_range_mm'] = np.max(distance) - np.min(distance)
        
        # --- Kanal-basierte Features (zur späteren Verfeinerung) ---
        df_frame = pd.DataFrame({'distance': distance, 'rssi': rssi, 'channel': channel_index})
        
        # Wir fokussieren uns nur auf 2-3 repräsentative Kanäle, um den Feature-Vektor klein zu halten.
        # Später können alle 16 Kanäle verwendet werden.
        representative_channels = [0, 8, 15] 

        for ch in representative_channels:
            ch_data = df_frame[df_frame['channel'] == ch]
            
            if not ch_data.empty:
                frame_features[f'ch_{ch}_mean_dist'] = ch_data['distance'].mean()
                frame_features[f'ch_{ch}_std_dist'] = ch_data['distance'].std()
                frame_features[f'ch_{ch}_mean_rssi'] = ch_data['rssi'].mean()
            else:
                # Setze NaN, wenn der Kanal im Frame keine Daten hat
                frame_features[f'ch_{ch}_mean_dist'] = np.nan
                frame_features[f'ch_{ch}_std_dist'] = np.nan
                frame_features[f'ch_{ch}_mean_rssi'] = np.nan
            
        all_features.append(frame_features)

    # Konvertierung in einen DataFrame
    df_features = pd.DataFrame(all_features)
    
    # Die Spalte 'scan_type' als letztes Feature für das Labeling (noch keine binären Labels)
    return df_features


def compare_wet_dry_stats(df_wet: pd.DataFrame, df_dry: pd.DataFrame):
    """
    Führt einen einfachen statistischen Vergleich der Features (Mean & Std) durch.
    """
    if df_wet.empty or df_dry.empty:
        return
        
    print("\n=======================================================")
    print("📈 Feature-Vergleich: Wet vs. Dry (Quantifizierung)")
    print("=======================================================")
    
    # Kombiniere die Mittelwerte und Standardabweichungen der beiden Datensätze
    df_comp = pd.concat([df_wet.drop(columns=['scan_type', 'frame_id']).mean().rename('WET_Mean'), 
                         df_dry.drop(columns=['scan_type', 'frame_id']).mean().rename('DRY_Mean'),
                         df_wet.drop(columns=['scan_type', 'frame_id']).std().rename('WET_Std'), 
                         df_dry.drop(columns=['scan_type', 'frame_id']).std().rename('DRY_Std')], axis=1)
    
    # Füge eine Spalte für die Differenz der Mittelwerte hinzu
    df_comp['Mean_Diff'] = df_comp['WET_Mean'] - df_comp['DRY_Mean']
    
    # Sortiere nach der Differenz für die signifikantesten Features
    df_comp = df_comp.sort_values(by='Mean_Diff', key=lambda x: np.abs(x), ascending=False)
    
    print("\n--- Top 10 Features mit größtem Unterschied (Wet vs. Dry) ---")
    print(df_comp.head(10).round(2).to_markdown(numalign="left", stralign="left"))

    print("\n➡️ Die signifikantesten Unterschiede sollten typischerweise sein:")
    print("   1. 'near_points_ratio' (WET > DRY): Mehr Rauschen im Nahfeld.")
    print("   2. 'global_std_rssi' (WET > DRY): Stärkere Schwankung der Intensität durch Wasserpartikel.")
    print("   3. 'global_mean_distance' (WET < DRY): Im Wet-Scan wird der Laser oft früher reflektiert.")
    print("=======================================================\n")