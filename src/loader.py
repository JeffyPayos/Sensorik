import pandas as pd
import numpy as np

# KONFIGURATION

NUM_CHANNELS = 16
ANGLE_INCREMENT_DEG = 0.5
POINTS_PER_LINE = 36
CHANNEL_BLOCK_LENGTH = NUM_CHANNELS * POINTS_PER_LINE  

# Startpunkte in Kopfzeile
DIST_START_COLUMN = 'picoScan (SN 24490061).ScanData.aDataChannel16[0].aData[0]'
RSSI_START_COLUMN = 'picoScan (SN 24490061).ScanData.aDataChannel8[0].aData[0]'



# LADEN DER LIDAR-DATEN

def load_csv_scandata(path):
    frames = []

    try:
        # Header lesen
        df_header = pd.read_csv(path, sep=',', encoding='latin-1', nrows=0)
        clean_header = [col.strip().strip('"') for col in df_header.columns]

        # Startindex suchen
        dist_start = clean_header.index(DIST_START_COLUMN)
        rssi_start = clean_header.index(RSSI_START_COLUMN)

        print(f" Distanz-Datenblock startet bei Spalte: {dist_start}")
        print(f" RSSI-Datenblock startet bei Spalte: {rssi_start}")

    except Exception as e:
        print(" Header-Analyse fehlgeschlagen:", e)
        return []

    # Block-Enden berechnen
    dist_end = dist_start + CHANNEL_BLOCK_LENGTH
    rssi_end = rssi_start + CHANNEL_BLOCK_LENGTH

    print(f" Scan-Struktur: {NUM_CHANNELS} Linien × {POINTS_PER_LINE} Punkte")

    # CSV-Daten laden
    df_raw = pd.read_csv(path, sep=',', encoding='latin-1',
                         skipinitialspace=True, header=None, low_memory=False)

    # Blöcke extrahieren
    df_dist = df_raw.iloc[:, dist_start:dist_end]
    df_rssi = df_raw.iloc[:, rssi_start:rssi_end]

    df_dist = df_dist.apply(pd.to_numeric, errors='coerce').astype(np.float64)
    df_rssi = df_rssi.apply(pd.to_numeric, errors='coerce').astype(np.float64)

    dist_array = df_dist.values
    rssi_array = df_rssi.values

    # Basiswinkel
    base_angles = np.deg2rad(np.arange(POINTS_PER_LINE) * ANGLE_INCREMENT_DEG)

    # FRAMES BILDEN
    for line_idx in range(len(dist_array)):

        frame_points = []

        for ch in range(NUM_CHANNELS):
            s = ch * POINTS_PER_LINE
            e = s + POINTS_PER_LINE

            dist_vals = dist_array[line_idx, s:e]
            rssi_vals = rssi_array[line_idx, s:e]

            valid = ~np.isnan(dist_vals)
            if not valid.any():
                continue

            dist_vals = dist_vals[valid]
            rssi_vals = rssi_vals[valid]
            ang = base_angles[valid]

            z = np.full_like(dist_vals, ch)  

            pts = np.vstack((ang, z, dist_vals, rssi_vals)).T
            frame_points.append(pts)

        if frame_points:
            frames.append(np.concatenate(frame_points, axis=0))

    return frames 
