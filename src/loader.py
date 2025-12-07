import pandas as pd
import numpy as np

# =================================================================
# ⭐ WICHTIGE KONFIGURATION
# =================================================================
NUM_CHANNELS = 16 
ANGLE_INCREMENT_DEG = 0.5 
POINTS_PER_LINE = 36 # Basierend auf 0.5 Grad Auflösung und typischer 288-Grad-Abtastung (576 Punkte gesamt / 16 Kanäle)
CHANNEL_BLOCK_LENGTH = NUM_CHANNELS * POINTS_PER_LINE # 576 Punkte

# Startspalte Distanz (Channel 16, Index 0)
DIST_START_COLUMN = 'picoScan (SN 24490061).ScanData.aDataChannel16[0].aData[0]' 
# Startspalte Intensität (Channel 8, Index 0) - KORREKT
RSSI_START_COLUMN = 'picoScan (SN 24490061).ScanData.aDataChannel8[0].aData[0]' 


def load_csv_scandata(path):
    """
    Lädt und parst alle 16 Lidar-Kanäle (Distanz & Intensität) aus der CSV-Datei.
    """
    frames = []

    try:
        # 1. Lade Header zur Bestimmung der Indizes
        df_header = pd.read_csv(path, sep=',', encoding='latin-1', nrows=0)
        clean_header = [col.strip().strip('"') for col in df_header.columns]

        try:
            # Finde die Distanz-Startspalte
            dist_start_index = clean_header.index(DIST_START_COLUMN)
            # Finde die RSSI-Startspalte
            rssi_start_index = clean_header.index(RSSI_START_COLUMN)
            
            print(f"✅ Distanz-Datenblock startet bei Spalte: {dist_start_index}")
            print(f"✅ RSSI-Datenblock startet bei Spalte: {rssi_start_index}")

        except ValueError as e:
            # Fehlerbehandlung: Zeigt die tatsächlich fehlende Spalte an
            # Dieser Teil sollte nun nicht mehr fehlschlagen, da wir die Namen kennen
            print(f"❌ Fehler: Startspalte nicht gefunden. Details: {e}")
            return []
            
        # 2. Definiere die Blocklänge (Fixiert auf 576 Punkte gesamt / 36 pro Linie)
        points_per_line = POINTS_PER_LINE
        channel_block_length = CHANNEL_BLOCK_LENGTH
        
        # Bestimme die Endindizes der beiden getrennten Blöcke
        dist_end_index = dist_start_index + channel_block_length
        rssi_end_index = rssi_start_index + channel_block_length
        
        print(f"✅ Scan-Struktur: {NUM_CHANNELS} Linien mit je {points_per_line} horizontalen Schritten.")

        # 3. Lade alle relevanten Daten (Die Blöcke sind nicht zusammenhängend!)
        
        # Lade die gesamten Daten der CSV
        # header=None ist wichtig für die iloc-Indizierung, die wir verwenden
        df_raw = pd.read_csv(
            path, 
            sep=',',            
            encoding='latin-1', 
            skipinitialspace=True,
            header=None,
            low_memory=False    
        )
        
        # Extrahiere Distanzblock (Index 25 bis 601)
        df_dist = df_raw.iloc[:, dist_start_index : dist_end_index]
        # Extrahiere RSSI-Block (Index 16603 bis 17179)
        df_rssi = df_raw.iloc[:, rssi_start_index : rssi_end_index]

        # Führe sie horizontal zusammen: [DistanzBlock | RSSIBlock]
        df_data_block = pd.concat([df_dist.reset_index(drop=True), 
                                   df_rssi.reset_index(drop=True)], axis=1)

        # Konvertiere den gesamten Block
        df_data_block = df_data_block.apply(pd.to_numeric, errors='coerce').astype(np.float64)
        raw_scan_array = df_data_block.values

        if raw_scan_array.size == 0 or np.all(np.isnan(raw_scan_array)):
            print("❌ Fehler: Alle Daten sind NaN oder die Blöcke wurden falsch geladen.")
            return []
            
        # 4. Erstellung der Frames (Kombination aller 32 Kanäle)
        for scan_values in raw_scan_array:
            
            frame_points = []
            base_angles = np.deg2rad(np.arange(points_per_line) * ANGLE_INCREMENT_DEG)
            
            # Die Offsets sind nun fix: Distanz beginnt bei 0, RSSI beginnt bei channel_block_length
            
            for channel_index in range(NUM_CHANNELS):
                
                # Berechnung der Start-/End-Indizes für Distanz (innerhalb des zusammengefügten Arrays)
                dist_start = (channel_index * points_per_line) 
                dist_end = dist_start + points_per_line
                
                # Berechnung der Start-/End-Indizes für RSSI (innerhalb des zusammengefügten Arrays)
                rssi_start = channel_block_length + (channel_index * points_per_line)
                rssi_end = rssi_start + points_per_line
                
                # Extrahieren der Werte
                dist_values = scan_values[dist_start:dist_end]
                rssi_values = scan_values[rssi_start:rssi_end]

                # Filterung: Entferne NaN aus beiden Arrays gleichzeitig
                valid_mask = ~np.isnan(dist_values) & ~np.isnan(rssi_values)
                
                filtered_dist = dist_values[valid_mask]
                filtered_rssi = rssi_values[valid_mask]
                filtered_angles = base_angles[valid_mask] # Winkelgröße muss mit Distanz übereinstimmen

                if filtered_dist.size == 0:
                    continue
                
                # V-Winkel Index als Platzhalter für die Z-Achse
                vertical_angle_index = np.full_like(filtered_dist, channel_index)
                
                # Stapeln: (H-Winkel, V-Index, Distanz, RSSI)
                points = np.vstack((filtered_angles, vertical_angle_index, filtered_dist, filtered_rssi)).T
                frame_points.append(points)

            if frame_points:
                # Kombiniere alle 16 Kanäle zu einem einzigen Frame
                frames.append(np.concatenate(frame_points, axis=0))

    except FileNotFoundError:
        print(f"❌ Fehler: Datei nicht gefunden unter '{path}'")
        return []
    except Exception as e:
        print(f"Ein unerwarteter Fehler beim Laden der Daten: {e}")
        return []
    
    return frames