import numpy as np
from sympy.physics.mechanics import RigidBody, ReferenceFrame, Point, SphericalJoint
from pydy.viz import Scene, VisualizationFrame, Cylinder, Sphere

# ── Frames ─────────────────────────────────────────────────────────────────
ground_frame = ReferenceFrame('ground_frame')
link1_frame  = ReferenceFrame('link1_frame')
link2_frame  = ReferenceFrame('link2_frame')
link3_frame  = ReferenceFrame('link3_frame')

# ── Points ─────────────────────────────────────────────────────────────────
ground_origin = Point('ground_origin')
link1_mc      = Point('link1_mc')
link2_mc      = Point('link2_mc')
link3_mc      = Point('link3_mc')

# ── Bodies ─────────────────────────────────────────────────────────────────
ground = RigidBody('ground', ground_origin, ground_frame)
link1  = RigidBody('link1',  link1_mc,      link1_frame)
link2  = RigidBody('link2',  link2_mc,      link2_frame)
link3  = RigidBody('link3',  link3_mc,      link3_frame)

# ── Joints ─────────────────────────────────────────────────────────────────
j1 = SphericalJoint(
    'j1', ground, link1,
    child_point=-link1_frame.y * 0.5,
)
j2 = SphericalJoint(
    'j2', link1, link2,
    parent_point= link1_frame.y * 0.5,
    child_point =-link2_frame.y * 0.5,
)
j3 = SphericalJoint(
    'j3', link2, link3,
    parent_point= link2_frame.y * 0.5,
    child_point =-link3_frame.y * 0.5,
)

# ── Shapes ─────────────────────────────────────────────────────────────────
link1_shape = Cylinder(length=1.0, radius=0.05, color='indianred')
link2_shape = Cylinder(length=1.0, radius=0.05, color='steelblue')
link3_shape = Cylinder(length=1.0, radius=0.05, color='seagreen')
sphere1 = Sphere(radius=0.08, color='gold')
sphere2 = Sphere(radius=0.08, color='gold')
sphere3 = Sphere(radius=0.08, color='gold')

# ── Explicit joint locations ────────────────────────────────────────────────
j2_loc = link1_mc.locatenew('j2_loc', link1_frame.y * 0.5)
j3_loc = link2_mc.locatenew('j3_loc', link2_frame.y * 0.5)

# ── Visualization frames ────────────────────────────────────────────────────
vf_link1 = VisualizationFrame('link1_vf', link1_frame, link1_mc, link1_shape)
vf_link2 = VisualizationFrame('link2_vf', link2_frame, link2_mc, link2_shape)
vf_link3 = VisualizationFrame('link3_vf', link3_frame, link3_mc, link3_shape)
vf_j1    = VisualizationFrame('j1_vf',    ground_frame, ground_origin, sphere1)
vf_j2    = VisualizationFrame('j2_vf',    link1_frame,  j2_loc,        sphere2)
vf_j3    = VisualizationFrame('j3_vf',    link2_frame,  j3_loc,        sphere3)

# ── Scene ───────────────────────────────────────────────────────────────────
scene = Scene(
    ground_frame, ground_origin,
    vf_link1, vf_link2, vf_link3,
    vf_j1, vf_j2, vf_j3,
)

# ── Dummy trajectory ────────────────────────────────────────────────────────
coords = list(j1.coordinates) + list(j2.coordinates) + list(j3.coordinates)
speeds = list(j1.speeds)      + list(j2.speeds)      + list(j3.speeds)
n = len(coords) + len(speeds)

times  = np.array([0.0, 0.1])
states = np.zeros((2, n))

# Print what the scene actually expects before assigning
scene.states_symbols = coords + speeds
times  = np.array([0.0, 0.1])
states = np.zeros((2, len(scene.states_symbols)))
scene.constants = {}
scene.states_trajectories = states
scene.times               = times



scene.display()