import pychrono.core as chrono
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
import time

# -----------------------
# Physics system
# -----------------------
system = chrono.ChSystemNSC()

body = chrono.ChBodyEasyBox(1, 1, 1, 1000, True, True)
body.SetPos(chrono.ChVector3d(0, 0, 0))
system.Add(body)

body.SetFixed(False)

# Create a constant torque load
load_container = chrono.ChLoadContainer()
system.Add(load_container)

torque = chrono.ChVector3d(0, 0, 50)
load = chrono.ChLoadBodyTorque(body, torque, False)
load_container.Add(load)


ground = chrono.ChBodyEasyCylinder()

# -----------------------
# MeshCat viewer
# -----------------------
vis = meshcat.Visualizer().open()

# ---- FIX: custom grid (works on all MeshCat versions) ----
def make_grid(n=20, spacing=0.1):
    pts = []
    for i in range(-n, n+1):
        x = i * spacing
        pts.append([x, -n*spacing, 0])
        pts.append([x,  n*spacing, 0])
        pts.append([-n*spacing, x, 0])
        pts.append([ n*spacing, x, 0])
    return np.array(pts).T

grid_points = make_grid()
vis["world"].set_object(
    g.LineSegments(
        g.PointsGeometry(grid_points),
        g.LineBasicMaterial(color=0x888888)
    )
)

# Create box in MeshCat
vis["box"].set_object(g.Box([1, 1, 1]))


# -----------------------
# Simulation loop
# -----------------------
dt = 0.01

while True:
    system.DoStepDynamics(dt)

    pos = body.GetPos()
    rot = body.GetRot()

    T = tf.translation_matrix([pos.x, pos.y, pos.z])

    q = [rot.e0, rot.e1, rot.e2, rot.e3]
    R = tf.quaternion_matrix(q)

    transform = T @ R
    vis["box"].set_transform(transform)

    time.sleep(dt)
