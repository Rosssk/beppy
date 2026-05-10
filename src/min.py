"""
PRBM (Pseudo-Rigid-Body Model) — RPR Compliant Link
with Meshcat 3-D Animation
====================================================
Topology:
  Ground ──[q1]──● Arm-A (L_r) ──[s]── Arm-B (L_r) ──[q2]──●── Free Body (m_B)

  Each R–P joint  →  PRBM torsional spring k_t  +  rotational DOF
  Prismatic P      →  axial spring k_p            +  translational DOF s

DOFs (3):  q1 [rad],  q2 [rad],  s [m]
States (6): y = [q1, q2, s, u1, u2, us]   (ui = dqi/dt)

Sections
--------
  I   - SymPy / Kane's Method  (symbolic)
  II  - Numerical EOM (from symbolic output)
  III - SciPy ODE integration
  IV  - Meshcat 3-D animation

Usage:
    python3 prbm_rpr_anim.py
    Open the printed URL in a browser.

Note on the RPR singularity
----------------------------
When q2 = 0 (arms collinear) the 3x3 dynamic mass matrix is rank-deficient.
A small Tikhonov regularisation eps=1e-6 on the diagonal keeps the integrator
stable near that configuration with negligible accuracy impact.

References:
  Howell, L. L. (2001). Compliant Mechanisms. Wiley.
  PRBM characteristic radius factor Gamma ~ 0.85.
"""

# =============================================================================
#  I - SYMBOLIC (SymPy / Kane's Method)
# =============================================================================

from sympy import *
from sympy.physics.mechanics import *

print("=" * 62)
print("  PRBM RPR - Symbolic derivation (Kane's Method)")
print("=" * 62)

# 1. Symbols
L_r, k_t, k_p, m_B, g_sym = symbols('L_r k_t k_p m_B g', positive=True)

q1, q2, s  = dynamicsymbols('q1 q2 s')
u1, u2, us = dynamicsymbols('u1 u2 us')
t = dynamicsymbols._t

# 2. Reference frames
N = ReferenceFrame('N')
A = ReferenceFrame('A')
B = ReferenceFrame('B')

A.orient_axis(N, N.z, q1)
B.orient_axis(A, A.z, q2)
A.set_ang_vel(N, u1 * N.z)
B.set_ang_vel(A, u2 * N.z)

# 3. Points
O = Point('O');   O.set_vel(N, 0)

P1 = Point('P1')
P1.set_pos(O, L_r * A.x)
P1.v2pt_theory(O, N, A)

P2 = Point('P2')
P2.set_pos(P1, s * A.x)
P2.set_vel(N, P1.vel(N) + us * A.x)

P_tip = Point('P_tip')
P_tip.set_pos(P2, L_r * B.x)
P_tip.v2pt_theory(P2, N, B)

# 4. Bodies (tip mass only; arms are massless)
Free_Body = Particle('FreeBody', P_tip, m_B)

# 5. Kinematic DEs
kd_eqs = [u1 - q1.diff(t), u2 - q2.diff(t), us - s.diff(t)]

# 6. Applied loads
loads = [
    (P_tip, -m_B * g_sym * N.y),   # gravity on tip
    (A,     -k_t * q1 * N.z),       # torsional spring 1 on A
    (B,     -k_t * q2 * N.z),       # torsional spring 2 on B
    (A,      k_t * q2 * N.z),       # Newton-3 reaction on A
    (P2,    -k_p * s  * A.x),       # axial spring on slider
]

# 7. Kane's equations
kane = KanesMethod(N, q_ind=[q1,q2,s], u_ind=[u1,u2,us], kd_eqs=kd_eqs)
fr, frstar = kane.kanes_equations([Free_Body], loads)

eom = [trigsimp(e) for e in fr + frstar]
print("\n  Mass matrix M(q):")
pprint(trigsimp(kane.mass_matrix))
print("\n  Forcing vector f(q,u):")
pprint(trigsimp(kane.forcing))


# =============================================================================
#  II - NUMERICAL EOM
#       Hand-transcribed from the symbolic output above.
# =============================================================================

import numpy as np

def eom_rhs(t_val, y, Lr, kt, kp, mB, g, eps=1e-6, damp=0.003):
    """
    State derivative: [qdot; udot] for y = [q1, q2, s, u1, u2, us].

    Mass matrix and forcing from the Kane symbolic output (Section I).
    eps  : diagonal regularisation (avoids singular M at q2=0).
    damp : light viscous damping for numerical stability.
    """
    q1v, q2v, sv, u1v, u2v, usv = y
    c2, s2 = np.cos(q2v), np.sin(q2v)
    Lr2 = Lr * Lr

    M = np.array([
        [2*Lr2*mB*(1+c2) + eps,  Lr2*mB*(1+c2),         -Lr*mB*s2      ],
        [Lr2*mB*(1+c2),           Lr2*mB + eps,           -Lr*mB*s2      ],
        [-Lr*mB*s2,              -Lr*mB*s2,                mB + eps      ],
    ])

    f = np.array([
        ( 2*Lr2*mB*u1v*u2v*s2
          + Lr2*mB*u2v**2*s2
          - Lr*mB*g*(np.cos(q1v+q2v) + np.cos(q1v))
          - Lr*mB*u1v*usv*(c2 + 1)
          - kt*q1v - damp*u1v ),

        ( -Lr2*mB*u1v**2*s2
          - Lr*mB*g*np.cos(q1v+q2v)
          - Lr*mB*u1v*usv*c2
          - kt*q2v - damp*u2v ),

        ( Lr*mB*(u1v+u2v)**2*c2
          + Lr*mB*u1v**2
          - mB*g*np.sin(q1v)
          - kp*sv - damp*usv ),
    ])

    udot = np.linalg.solve(M, f)
    return [u1v, u2v, usv, udot[0], udot[1], udot[2]]


# =============================================================================
#  III - ODE INTEGRATION
# =============================================================================

from scipy.integrate import solve_ivp

params = dict(
    Lr=0.85*0.15,  # [m]       effective arm length (Gamma=0.85, half-length=0.15)
    kt=0.8,         # [N.m/rad] torsional spring
    kp=40.0,        # [N/m]     prismatic spring
    mB=0.05,        # [kg]      tip mass
    g=9.81,         # [m/s^2]
)

# ICs: small initial displacement; q2 != 0 avoids singularity at t=0
y0 = [0.30, 0.15, 0.0,  0.0, 0.0, 0.0]

T_end = 8.0
fps   = 30
t_eval = np.linspace(0, T_end, int(T_end * fps))

print("\n" + "=" * 62)
print(f"  Integrating {T_end} s, {len(t_eval)} frames …")
print("=" * 62)

sol = solve_ivp(
    lambda tv, yv: eom_rhs(tv, yv, **params),
    [0, T_end], y0,
    t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-8,
)
if not sol.success:
    raise RuntimeError(f"ODE solver failed: {sol.message}")

print(f"  q1 in [{sol.y[0].min():.3f}, {sol.y[0].max():.3f}] rad")
print(f"  q2 in [{sol.y[1].min():.3f}, {sol.y[1].max():.3f}] rad")
print(f"   s in [{sol.y[2].min():.4f}, {sol.y[2].max():.4f}] m")

# Forward kinematics
Lr = params['Lr']

def fwd_kin(q1v, q2v, sv):
    O_p  = np.zeros(3)
    ax   = np.array([np.cos(q1v), np.sin(q1v), 0.0])
    P1_p = O_p  + Lr * ax
    P2_p = P1_p + sv * ax
    bx   = np.array([np.cos(q1v+q2v), np.sin(q1v+q2v), 0.0])
    Pt_p = P2_p + Lr * bx
    return O_p, P1_p, P2_p, Pt_p

traj = [fwd_kin(sol.y[0,i], sol.y[1,i], sol.y[2,i]) for i in range(len(t_eval))]


# =============================================================================
#  IV - MESHCAT ANIMATION
# =============================================================================

import meshcat
import meshcat.geometry  as geom
import meshcat.animation as mcanim

print("\n" + "=" * 62)
print("  Meshcat visualiser")
print("=" * 62)

vis = meshcat.Visualizer()
print(f"\n  Open in browser:  {vis.url()}\n")

# Materials
def mat(color, opacity=1.0):
    return geom.MeshLambertMaterial(color=color, opacity=opacity)

C = dict(ground=0x444444, armA=0x1565C0, slider=0x2E7D32,
         armB=0xC62828, joint=0xF9A825, tip=0xFF5722)

R_arm  = 0.008;  R_sl = 0.005;  R_jt = 0.016;  R_tip = 0.024

# Geometry helpers
def look_along_y(vec, centre):
    """4x4 transform: place unit Cylinder (axis +Y) along vec, centred at centre."""
    length = np.linalg.norm(vec)
    if length < 1e-10:
        T = np.eye(4); T[:3,3] = centre; return T
    n = vec / length
    y = np.array([0.0, 1.0, 0.0])
    cross = np.cross(y, n)
    cn = np.linalg.norm(cross)
    if cn < 1e-10:
        R3 = np.eye(3) if np.dot(y, n) > 0 else -np.eye(3)
    else:
        ax_ = cross / cn
        ang = np.arccos(np.clip(np.dot(y, n), -1.0, 1.0))
        K = np.array([[0, -ax_[2], ax_[1]],
                      [ax_[2], 0, -ax_[0]],
                      [-ax_[1], ax_[0], 0]])
        R3 = np.eye(3) + np.sin(ang)*K + (1-np.cos(ang))*(K@K)
    T = np.eye(4); T[:3,:3] = R3; T[:3,3] = centre
    return T

def cyl_tf(p0, p1):
    vec = p1 - p0
    T   = look_along_y(vec, 0.5*(p0+p1))
    S   = np.diag([1.0, max(np.linalg.norm(vec), 1e-6), 1.0, 1.0])
    return T @ S

def transl(p):
    T = np.eye(4); T[:3,3] = p; return T

# Static scene: ground block + fixed origin pin
vis['scene/ground'].set_object(geom.Box([0.10, 0.015, 0.07]), mat(C['ground']))
vis['scene/ground'].set_transform(transl([-0.05, -0.0075, -0.035]))
vis['scene/pin_O'].set_object(geom.Sphere(R_jt), mat(C['joint']))
vis['scene/pin_O'].set_transform(transl([0, 0, 0]))

# Animated geometry (set once; transforms updated per frame)
vis['anim/armA'  ].set_object(geom.Cylinder(1.0, R_arm), mat(C['armA']))
vis['anim/slider'].set_object(geom.Cylinder(1.0, R_sl ), mat(C['slider']))
vis['anim/armB'  ].set_object(geom.Cylinder(1.0, R_arm), mat(C['armB']))
vis['anim/jt_P1' ].set_object(geom.Sphere(R_jt),         mat(C['joint']))
vis['anim/jt_P2' ].set_object(geom.Sphere(R_jt),         mat(C['joint']))
vis['anim/tip'   ].set_object(geom.Sphere(R_tip),         mat(C['tip']))

# Build animation
print(f"  Building {len(traj)}-frame animation at {fps} fps …")
anim = mcanim.Animation(default_framerate=fps)

for i, (O_p, P1_p, P2_p, Pt_p) in enumerate(traj):
    with anim.at_frame(vis, i) as fr:
        fr['anim/armA'  ].set_transform(cyl_tf(O_p, P1_p))

        ls = np.linalg.norm(P2_p - P1_p)
        if ls > 5e-4:
            fr['anim/slider'].set_transform(cyl_tf(P1_p, P2_p))
        else:
            fr['anim/slider'].set_transform(np.diag([1e-4, 1e-4, 1e-4, 1.0]))

        fr['anim/armB'  ].set_transform(cyl_tf(P2_p, Pt_p))
        fr['anim/jt_P1' ].set_transform(transl(P1_p))
        fr['anim/jt_P2' ].set_transform(transl(P2_p))
        fr['anim/tip'   ].set_transform(transl(Pt_p))

vis.set_animation(anim, play=True, repetitions=0)   # 0 = loop forever

print("  Animation live — looping continuously.\n")
print("  Colour legend:")
print("    Blue   = Rigid arm A  (ground side)")
print("    Green  = Prismatic slider  (P section)")
print("    Red    = Rigid arm B  (free-body side)")
print("    Amber  = PRBM torsional pin joint")
print("    Orange = Free-body tip mass (m_B)")
print("\n  Press Ctrl-C to stop the server.")

try:
    import time
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n  Server stopped.")