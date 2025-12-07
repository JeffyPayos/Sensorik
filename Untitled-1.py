import pandas as pd
import re

# Definieren Sie den Pfad zu Ihrer Datei
FILE_PATH = 'd:/Lidar-projekt/data/raw/wet_20251207160801_20251207160821.csv'
SEARCH_PATTERN = 'picoScan (SN 24490061).ScanData.aDataChannel'

print(f"Lese Header der Datei: {FILE_PATH.split('/')[-1]}")
print("---")

try:
    # Lade nur die Header-Zeile (erste Zeile)
    df_header = pd.read_csv(FILE_PATH, sep=',', encoding='latin-1', nrows=0)
    
    # Bereinige die Spaltennamen
    clean_header = [col.strip().strip('"') for col in df_header.columns]
    
    found_channels = []
    
    # Filtere alle Kanalnamen, die mit dem Suchmuster beginnen
    for col in clean_header:
        if col.startswith(SEARCH_PATTERN):
            found_channels.append(col)
            
    if not found_channels:
        print(f"❌ KEINE Spalten mit dem Muster '{SEARCH_PATTERN}' gefunden.")
        
    else:
        # Finde den Index der Distanz-Startspalte (wissen wir bereits)
        dist_start_col = 'picoScan (SN 24490061).ScanData.aDataChannel16[0].aData[0]'
        dist_index = -1
        try:
            dist_index = clean_header.index(dist_start_col)
            print(f"✅ Distanz-Startspalte ({dist_start_col}) gefunden bei Index {dist_index}.")
        except ValueError:
            print("⚠️ Distanz-Startspalte (Channel 16) nicht gefunden.")

        print("\n--- Gefundene aDataChannelXX-Spalten: ---")
        
        # 1. Sammle alle einzigartigen Channel-Startspalten
        start_columns = set()
        for col in found_channels:
            # Wir suchen nur die erste Spalte eines Datenblocks: [0].aData[0]
            if col.endswith('[0].aData[0]'):
                start_columns.add(col)
                
        # Sortiere die Spalten nach ihrem tatsächlichen Index in der CSV
        start_columns_list = sorted(list(start_columns), 
                                    key=lambda x: clean_header.index(x) if x in clean_header else len(clean_header))
        
        # 2. Gib die relevanten Startspalten aus
        print("\nMögliche Startspalten für einen Datenblock:")
        for col in start_columns_list:
            if col in clean_header:
                index = clean_header.index(col)
                
                # Finde die Kanalnummer außerhalb des f-strings (Korrektur des Syntaxfehlers)
                channel_match = re.search(r'aDataChannel(\d+)', col)
                channel_number = channel_match.group(1) if channel_match else '??'
                
                # Markiere Channel 16
                label = " (DISTANZ-START)" if col == dist_start_col else ""
                
                print(f" - Channel {channel_number}: {col} [Index: {index}]{label}")
            
        print("\n--- Ende der Spaltenanalyse ---")
        print("\n➡️ Die RSSI-Startspalte muss EINER dieser 'aDataChannelXX[0].aData[0]'-Einträge sein und NACH der Distanz-Startspalte liegen.")

except FileNotFoundError:
    print(f"❌ Fehler: Datei nicht gefunden unter '{FILE_PATH}'")
except Exception as e:
    print(f"Ein unerwarteter Fehler beim Lesen der Daten: {e}")