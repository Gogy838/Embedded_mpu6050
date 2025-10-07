import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ----- CONFIG -----
PORT = "COM10"  # Change to your COM port
BAUD = 115200
ser = serial.Serial(PORT, BAUD, timeout=1)

# ----- HELPER FUNCTIONS -----


def parse_line(line):
    """Parse 'yaw,pitch,roll' line from serial."""
    try:
        y, p, r = line.strip().split(',')
        return float(y), float(p), float(r)
    except:
        return None, None, None


def rotation_matrix(yaw, pitch, roll):
    """Return combined rotation matrix from yaw, pitch, roll (degrees)."""
    y, p, r = np.radians([yaw, pitch, roll])
    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y),  np.cos(y), 0],
                   [0, 0, 1]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)],
                   [0, 1, 0],
                   [-np.sin(p), 0, np.cos(p)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r), np.cos(r)]])
    return Rz @ Ry @ Rx


def create_sphere(center, radius, res=12):
    """Return vertex coordinates for a sphere."""
    phi, theta = np.mgrid[0:np.pi:complex(0, res), 0:2*np.pi:complex(0, res)]
    x = center[0] + radius*np.sin(phi)*np.cos(theta)
    y = center[1] + radius*np.sin(phi)*np.sin(theta)
    z = center[2] + radius*np.cos(phi)
    verts = np.stack((x, y, z), axis=-1)
    # Create faces (quads)
    faces = []
    for i in range(res-1):
        for j in range(res-1):
            quad = [verts[i, j], verts[i+1, j], verts[i+1, j+1], verts[i, j+1]]
            faces.append(quad)
    return faces


def create_cone(base_center, height, radius, res=12):
    """Return list of triangular faces for a cone."""
    theta = np.linspace(0, 2*np.pi, res, endpoint=False)
    x = base_center[0] + radius*np.cos(theta)
    y = base_center[1] + radius*np.sin(theta)
    z = np.full_like(x, base_center[2])
    tip = np.array([base_center[0], base_center[1], base_center[2]+height])
    faces = [np.vstack(([x[i], y[i], z[i]],
                        [x[(i+1) % res], y[(i+1) % res], z[(i+1) % res]],
                        tip)) for i in range(res)]
    return faces


# ----- CREATE DUCK -----
body_faces = create_sphere([0, 0, 0], 0.25)
head_faces = create_sphere([0, 0, 0.35], 0.12)
beak_faces = create_cone([0, 0.18, 0.35], 0.07, 0.07)

# ----- FIGURE -----
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-0.5, 0.8])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Duck Tilt (Yaw, Pitch, Roll)")

# Create Poly3DCollections once
body_poly = Poly3DCollection(body_faces, facecolor="yellow", edgecolor="k")
head_poly = Poly3DCollection(head_faces, facecolor="yellow", edgecolor="k")
beak_poly = Poly3DCollection(beak_faces, facecolor="red", edgecolor="k")
ax.add_collection3d(body_poly)
ax.add_collection3d(head_poly)
ax.add_collection3d(beak_poly)

# Text display
yaw_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)
pitch_text = ax.text2D(0.02, 0.90, "", transform=ax.transAxes)
roll_text = ax.text2D(0.02, 0.85, "", transform=ax.transAxes)

# ----- UPDATE FUNCTION -----


def update(frame):
    # Read multiple lines to keep up with sensor
    for _ in range(3):
        raw = ser.readline().decode(errors="ignore")
        yaw, pitch, roll = parse_line(raw)
        if yaw is not None:
            break
    else:
        return  # skip if no valid data

    # Update text
    yaw_text.set_text(f"Yaw: {yaw:.1f}°")
    pitch_text.set_text(f"Pitch: {pitch:.1f}°")
    roll_text.set_text(f"Roll: {roll:.1f}°")

    # Compute rotation
    R = rotation_matrix(yaw, pitch, roll)

    # Rotate Poly3DCollections directly
    def rotate_poly(poly):
        verts = poly.get_verts()
        new_verts = [(v @ R.T) for v in verts]
        poly.set_verts(new_verts)

    rotate_poly(body_poly)
    rotate_poly(head_poly)
    rotate_poly(beak_poly)


# ----- ANIMATION -----
ani = FuncAnimation(fig, update, interval=20, cache_frame_data=False)
plt.show()
