import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

from sympy import (
    symbols,
    cos,
    sin,
    pi,
    Rational,
    lambdify,
)

from sympy.physics.mechanics import (
    dynamicsymbols,
    ReferenceFrame,
    Point,
    RigidBody,
    inertia,
    SphericalJoint,
    PrismaticJoint,
    System,
    KanesMethod,
)


# =============================================================================
# Parameters
# =============================================================================

n_beams = 3

r, h, L0 = symbols('r h L0', positive=True)

M = symbols('M', positive=True)

Ixx, Iyy, Izz = symbols('Ixx Iyy Izz', positive=True)


# =============================================================================
# Bodies
# =============================================================================

body_A = RigidBody('body_A')

body_B = RigidBody('body_B')
body_B.mass = M
body_B.inertia = (
    inertia(body_B.frame, Ixx, Iyy, Izz),
    body_B.masscenter,
)

body_C = RigidBody('body_C')
body_C.mass = M
body_C.inertia = (
    inertia(body_C.frame, Ixx, Iyy, Izz),
    body_C.masscenter,
)


# =============================================================================
# Build beams
# =============================================================================

all_joints = []
beam_links = []


def make_spr_beam(
    label,
    parent_body,
    child_body,
    root_attach_vec,
    tip_attach_vec,
    nominal_beam_vec,
):

    link_A = RigidBody(f'lA_{label}')
    link_B = RigidBody(f'lB_{label}')

    link_A.mass = M / 100
    link_B.mass = M / 100

    link_A.inertia = (
        inertia(link_A.frame, Ixx/100, Iyy/100, Izz/100),
        link_A.masscenter,
    )

    link_B.inertia = (
        inertia(link_B.frame, Ixx/100, Iyy/100, Izz/100),
        link_B.masscenter,
    )

    jA = SphericalJoint(
        f'sA_{label}',
        parent_body,
        link_A,
        parent_interframe=nominal_beam_vec,
        parent_point=root_attach_vec,
        child_point=-L0 * link_A.frame.x,
    )

    jP = PrismaticJoint(
        f'p_{label}',
        link_A,
        link_B,
        joint_axis=link_A.frame.x,
        parent_point=L0 * link_A.frame.x,
        child_point=link_A.masscenter.locatenew(
            f'{label}_prismatic_child',
            0 * link_A.frame.x
        )
    )

    jB = SphericalJoint(
        f'sB_{label}',
        link_B,
        child_body,
        parent_point=0,
        child_point=tip_attach_vec,
    )

    return [jA, jP, jB], [link_A, link_B]


# -----------------------------------------------------------------------------
# Group A → B
# -----------------------------------------------------------------------------

for i in range(n_beams):

    phi_r = 2 * pi * i / n_beams
    phi_t = 2 * pi * (i + 1) / n_beams

    root_vec = (
        r * cos(phi_r) * body_A.frame.x
        + r * sin(phi_r) * body_A.frame.y
    )

    tip_vec = (
        r * cos(phi_t) * body_B.frame.x
        + r * sin(phi_t) * body_B.frame.y
    )

    beam_dir = (
        (r * (cos(phi_t) - cos(phi_r))) * body_A.frame.x
        + (r * (sin(phi_t) - sin(phi_r))) * body_A.frame.y
        + (h / 2) * body_A.frame.z
    )

    joints, links = make_spr_beam(
        f'ab{i}',
        body_A,
        body_B,
        root_vec,
        tip_vec,
        beam_dir,
    )

    all_joints.extend(joints)
    beam_links.extend(links)


# -----------------------------------------------------------------------------
# Group C → B
# -----------------------------------------------------------------------------

for i in range(n_beams):

    phi_r = 2 * pi * (i - Rational(1, 2)) / n_beams
    phi_t = 2 * pi * (i + Rational(1, 2)) / n_beams

    root_vec = (
        r * cos(phi_r) * body_C.frame.x
        + r * sin(phi_r) * body_C.frame.y
    )

    tip_vec = (
        r * cos(phi_t) * body_B.frame.x
        + r * sin(phi_t) * body_B.frame.y
    )

    beam_dir = (
        (r * (cos(phi_t) - cos(phi_r))) * body_C.frame.x
        + (r * (sin(phi_t) - sin(phi_r))) * body_C.frame.y
        + (-h / 2) * body_C.frame.z
    )

    joints, links = make_spr_beam(
        f'cb{i}',
        body_C,
        body_B,
        root_vec,
        tip_vec,
        beam_dir,
    )

    all_joints.extend(joints)
    beam_links.extend(links)


# =============================================================================
# SymPy system
# =============================================================================

sys = System(body_A.frame, body_A.masscenter)
sys.add_joints(*all_joints)

print(f"Generalized coordinates: {len(sys.q)}")
print(f"Generalized speeds:      {len(sys.u)}")


# =============================================================================
# Numerical parameter values
# =============================================================================

constants = {
    r: 0.10,
    h: 0.30,
    L0: 0.20,
    M: 1.0,
    Ixx: 0.01,
    Iyy: 0.01,
    Izz: 0.01,
}


# =============================================================================
# Build forward kinematics functions
# =============================================================================

world = body_A.frame
origin = body_A.masscenter

all_bodies = [
    body_A,
    body_B,
    body_C,
    *beam_links,
]

fk_functions = {}

for body in all_bodies:

    pos = body.masscenter.pos_from(origin).to_matrix(world)

    rot = body.frame.dcm(world)

    fk_expr = pos.col_join(rot.reshape(9, 1))

    fk_functions[body.name] = lambdify(
        [sys.q],
        fk_expr.subs(constants),
        modules='numpy',
    )


# =============================================================================
# Create fake motion trajectory
# =============================================================================

nq = len(sys.q)

T = 10.0
N = 400

times = np.linspace(0, T, N)

q_traj = np.zeros((N, nq))

for i in range(min(nq, 12)):
    q_traj[:, i] = 0.15 * np.sin(2.0 * times + i * 0.3)


# =============================================================================
# MeshCat setup
# =============================================================================

vis = meshcat.Visualizer().open()

print("MeshCat opened in browser")


# -----------------------------------------------------------------------------
# World axes
# -----------------------------------------------------------------------------

vis['world_axes'].set_object(
    g.LineSegments(
        g.PointsGeometry(
            position=np.array([
                [0, 0, 0], [1, 0, 0],
                [0, 0, 0], [0, 1, 0],
                [0, 0, 0], [0, 0, 1],
            ]).T,
            color=np.array([
                [1, 0, 0], [1, 0, 0],
                [0, 1, 0], [0, 1, 0],
                [0, 0, 1], [0, 0, 1],
            ]).T,
        ),
        g.LineBasicMaterial(vertexColors=True),
    )
)


# -----------------------------------------------------------------------------
# Body geometry
# -----------------------------------------------------------------------------

for body in all_bodies:

    if body.name.startswith('lA') or body.name.startswith('lB'):

        vis[body.name].set_object(
            g.Cylinder(
                height=0.2,
                radius=0.01,
            ),
            g.MeshLambertMaterial(color=0xcc3333),
        )

    else:

        vis[body.name].set_object(
            g.Sphere(0.04),
            g.MeshLambertMaterial(color=0x3366cc),
        )


# =============================================================================
# Animation loop
# =============================================================================

print("Starting animation...")

fps = 60

dt = 1.0 / fps

for k in range(N):

    qk = q_traj[k]

    for body in all_bodies:

        data = np.array(
            fk_functions[body.name](qk),
            dtype=float,
        ).flatten()

        px, py, pz = data[:3]

        R = data[3:].reshape(3, 3)

        Tmat = np.eye(4)

        Tmat[:3, :3] = R
        Tmat[:3, 3] = [px, py, pz]

        vis[body.name].set_transform(Tmat)

    time.sleep(dt)

print("Done.")