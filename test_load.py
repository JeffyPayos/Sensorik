import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
from mpl_toolkits.mplot3d import Axes3D # Import für 3D
from src.loader import load_csv_scandata 

# Definieren Sie den Pfad zu Ihrer Datei
FILE_PATH = 'd:/Lidar-projekt/data/raw/wet_20251207160801_20251207160821.csv'

print(f"Starte das Laden der Datei: {FILE_PATH.split('/')[-1]}")

# Laden der Daten
frames = load_csv_scandata(FILE_PATH)

# =================================================================
# 🚀 Animationsfunktion für 3D-Punktwolken
# =================================================================

def animate_scans_3d(frames):
    """
    Erstellt eine 3D-Animation aller geladenen Lidar-Frames. 
    """
    if not frames:
        return
        
    num_frames = len(frames)
    
    # Sammle RSSI-Werte für die globale Farb-Skala (Spalte 3)
    all_rssi = np.concatenate([frame[:, 3] for frame in frames])
    RSSI_MIN = np.percentile(all_rssi[all_rssi > 0], 5) if all_rssi.size > 0 else 0
    RSSI_MAX = np.percentile(all_rssi, 95) if all_rssi.size > 0 else 5000
    if RSSI_MAX == 0: RSSI_MAX = 5000 
    print(f"Farbskala (RSSI): {RSSI_MIN:.0f} bis {RSSI_MAX:.0f}")

    # 1. Figur und 3D-Achsen einrichten
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d') 
    
    # Setze Achsen-Grenzen
    ax.set_xlim(-10, 10) 
    ax.set_ylim(-10, 10)
    ax.set_zlim(-5, 5) 
    
    ax.set_xlabel("X (Meter)")
    ax.set_ylabel("Y (Meter)")
    ax.set_zlabel("V-Winkel Index") 

    # 2. Initialen Scatter-Plot erstellen
    current_frame = frames[0]
    
    # Spalten: H-Winkel (0), V-Index (1), Distanz (2), RSSI (3)
    theta = current_frame[:, 0]
    vertical_index = current_frame[:, 1] 
    r = current_frame[:, 2] / 1000.0 # Konvertierung zu Meter
    rssi = current_frame[:, 3]
    
    # Kartesische 3D-Koordinaten:
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = vertical_index 
    
    scatter = ax.scatter(x, y, z, s=5, c=rssi, cmap='viridis', 
                         vmin=RSSI_MIN, vmax=RSSI_MAX) 
    
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.7)
    cbar.set_label('RSSI / Intensität')
    
    # 3. Die Update-Funktion
    def update(frame_index):
        """Aktualisiert die Datenpunkte für den Frame frame_index."""
        current_frame = frames[frame_index]
        
        theta = current_frame[:, 0]
        vertical_index = current_frame[:, 1] 
        r = current_frame[:, 2] / 1000.0 # Meter
        rssi = current_frame[:, 3]
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = vertical_index
        
        # Aktualisiere die Daten
        scatter._offsets3d = (x, y, z) 
        scatter.set_array(rssi)
        
        ax.set_title(f"3D Lidar Point Cloud | Frame {frame_index + 1}/{num_frames} ({len(x)} Punkte)")
        
        return scatter,

    # 4. FuncAnimation erstellen und starten
    ani = FuncAnimation(
        fig, 
        update, 
        frames=num_frames, 
        interval=50, 
        blit=False, 
        repeat=True 
    )

    print("Starte Matplotlib 3D Animation (Alle 16 Kanäle)...")
    plt.show()


# =================================================================
# 🏁 Hauptausführung
# =================================================================
if frames:
    print(f"🥳 {len(frames)} Frames erfolgreich geladen.")
    animate_scans_3d(frames)
else:
    print("Die Daten konnten nicht geladen werden oder es wurden keine gültigen Frames gefunden.")