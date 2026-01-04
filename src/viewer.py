import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D



# 2D EINZELFRAME

def show_frame_2d(frame, title="2D Frame"):
    theta = frame[:, 0]
    r = frame[:, 2] / 1000.0
    rssi = frame[:, 3]

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    plt.figure(figsize=(8, 8))
    sc = plt.scatter(x, y, c=rssi, s=10, cmap="viridis")
    plt.colorbar(sc, label="RSSI")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(title)
    plt.grid(True)
    plt.axis("equal")
    plt.show()



# 2D ANIMATION

def animate_2d(frames, title="2D Animation", interval=80):
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter([], [], s=5)

    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_aspect('equal')
    ax.grid(True)

    def update(idx):
        frame = frames[idx]
        theta = frame[:, 0]
        r = frame[:, 2] / 1000.0
        rssi = frame[:, 3]

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        scatter.set_offsets(np.c_[x, y])
        scatter.set_array(rssi)
        ax.set_title(f"{title} – Frame {idx+1}/{len(frames)}")
        return scatter,

    ani = FuncAnimation(fig, update, frames=len(frames),
                        interval=interval, repeat=True)
    plt.colorbar(scatter, label="RSSI")
    plt.show()



# 3D EINZELFRAME

def show_frame_3d(frame, title="3D Frame"):
    theta = frame[:, 0]
    z = frame[:, 1]
    r = frame[:, 2] / 1000.0
    rssi = frame[:, 3]

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(x, y, z, c=rssi, cmap="viridis", s=10)
    plt.colorbar(sc, label="RSSI")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("Channel")
    ax.set_title(title)
    plt.show()
