import numpy as np
import matplotlib.pyplot as plt

def plot_frame(frame, frame_index=0, unit="mm"):
    """
    Visualisiert einen einzelnen LiDAR-Frame als Streudiagramm (Polar → Kartesisch).
    """
    if len(frame) == 0:
        print("Keine Daten zum Plotten vorhanden.")
        return

    # Die Daten sind bereits in [Winkel, Distanz] Format
    angles = frame[:, 0]
    distances = frame[:, 1]
    
    # Der PicoScan gibt die Distanz oft in Millimetern (mm) aus. 
    # Wenn 'unit' auf 'm' gesetzt ist, konvertieren wir die Distanz.
    scale_factor = 1.0
    if unit.lower() == "m":
        scale_factor = 1000.0
    
    # Polar → Kartesisch: x = r*cos(phi), y = r*sin(phi)
    x = distances / scale_factor * np.cos(angles)
    y = distances / scale_factor * np.sin(angles)

    plt.figure(figsize=(8, 8))
    # Farbkodierung nach Distanz hinzugefügt, um die Visualisierung zu verbessern
    plt.scatter(x, y, s=5, c=distances, cmap='viridis') 
    plt.colorbar(label=f'Distanz [{unit}]')
    
    plt.xlabel(f"x [{unit}]") 
    plt.ylabel(f"y [{unit}]")
    plt.title(f"LiDAR Frame {frame_index} (Kartesisch)")
    plt.axis("equal")
    plt.grid(True)
    plt.show()