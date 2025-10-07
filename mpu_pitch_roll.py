import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# ----- CONFIG -----
PORT = "COM9"  # change to your COM port
BAUD = 115200
WINDOW = 200  # not strictly needed for 3D but could keep for smoothing

ser = serial.Serial(PORT, BAUD, timeout=1)

pitch_buf = []
roll_buf = []

# ----- 3D PLOT SETUP -----
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')

# Cube definition centered at origin
cube_definition = np.array([
    [-1, -1, -0.1],
    [-1,  1, -0.1],
    [1,  1, -0.1],
    [1, -1, -0.1],
    [-1, -1,  0.1],
    [-1,  1,  0.1],
    [1,  1,  0.1],
    [1, -1,  0.1]
])

# Edges of the cube
edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
    [4, 5], [5, 6], [6, 7], [7, 4],  # top
    [0, 4], [1, 5], [2, 6], [3, 7]   # verticals
]

lines = [ax.plot([], [], [], 'b', linewidth=2)[0] for _ in edges]

ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("3D Visualization of Pitch & Roll (Yaw Ignored)")

# ----- HELPERS -----


def rotation_matrix(pitch, roll):
    """Rotation matrix for pitch (Y-axis) and roll (X-axis)."""
    pitch = np.radians(pitch)
    roll = np.radians(roll)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    return Ry @ Rx  # Apply roll then pitch


def parse_line(line):
    """Parse 'pitch,roll' from serial."""
    try:
        parts = line.strip().split(',')
        if len(parts) < 2:
            return None, None
        pitch = float(parts[0])
        roll = float(parts[1])
        return pitch, roll
    except:
        return None, None


def update(frame):
    # Read a few lines from serial
    for _ in range(5):
        raw = ser.readline().decode(errors='ignore')
        if not raw:
            continue
        pitch, roll = parse_line(raw)
        if pitch is None:
            continue
        pitch_buf.append(pitch)
        roll_buf.append(roll)
        # Keep buffers limited
        if len(pitch_buf) > WINDOW:
            pitch_buf.pop(0)
            roll_buf.pop(0)

    if not pitch_buf:
        return lines

    pitch = pitch_buf[-1]
    roll = roll_buf[-1]

    R = rotation_matrix(pitch, roll)
    rotated = cube_definition @ R.T

    for idx, edge in enumerate(edges):
        start, end = edge
        lines[idx].set_data([rotated[start][0], rotated[end][0]],
                            [rotated[start][1], rotated[end][1]])
        lines[idx].set_3d_properties([rotated[start][2], rotated[end][2]])

    return lines


# ----- ANIMATE -----
ani = FuncAnimation(fig, update, interval=30, blit=False)
plt.show()
