"""
RPR tripod – SymPy Mechanics rewrite
=====================================
Derives equations of motion via Kane's method, then lambdifies the
full RHS (mass matrix + forcing vector) into a single monolithic NumPy
call with CSE, eliminating redundant trig recomputation.

Structure
---------
  Step 1 : Reference frames and kinematics (ZYX Euler, two floating bodies)
  Step 2 : Attachment points as Point objects
  Step 3 : Beam particles (3-point lumped mass per beam, 6 beams)
  Step 4 : Disc rigid bodies (B2, B3)
  Step 5 : Kane's method → MM, forcing
  Step 6 : Potential energy (bending + axial) → generalised active forces
  Step 7 : Lambdify with CSE → fast NumPy eom()
  Step 8 : simulate() wrapper (same interface as original)

Compatibility
-------------
  Drop-in replacement for symbolic_derivation.py.
  Public API preserved:
    rotation_matrix(angles)
    potential_and_gradient(q)
    mass_matrix(q)
    eom(t, state, force_fn, ...)
    simulate(force_fn, ...)
    run_checks()

Notes on Kane's method formulation
-----------------------------------
  Generalised coordinates: q = [x2,y2,z2, phi2,th2,ps2, x3,y3,z3, phi3,th3,ps3]
  Generalised speeds:      u = dq/dt  (simple choice avoids auxiliary equations)
  Orientation convention:  ZYX body-fixed Euler (same as original code)

  The full 24x24 system is:
      MM_full * [qdot; udot]^T = forcing_full
  where the top 12 rows are kinematic (qdot - u = 0) and the bottom 12
  are dynamic.  We solve only for udot (= qddot) at each step.
"""

import sympy as sp
from sympy import symbols, cos, sin, sqrt, Matrix, Rational
from sympy import lambdify, trigsimp
import numpy as np
from scipy.integrate import solve_ivp
import time

# ─────────────────────────────────────────────────────────────────────────────
# Parameters  (identical to original)
# ─────────────────────────────────────────────────────────────────────────────

R_ATT          = 0.3
BEAM_ANGLES_NP = np.array([0.0, 2*np.pi/3, 4*np.pi/3])
B1_POS         = np.zeros(3)
L_BEAM         = 1.0
ZHAT_NP        = np.array([0., 0., 1.])

R_LOCAL = np.stack([
    R_ATT * np.array([np.cos(a), np.sin(a), 0.0])
    for a in BEAM_ANGLES_NP
])  # (3,3)

K_THETA  = 2.031
K_S      = 0.978
M_DISC   = 1.0
R_DISC   = 0.4
I_XX     = 0.25 * M_DISC * R_DISC**2
I_ZZ     = 0.50 * M_DISC * R_DISC**2
M_BEAM   = 0.1

_DISC_BLOCK = np.diag([M_DISC, M_DISC, M_DISC, I_XX, I_XX, I_ZZ])
M_DISCS = np.block([[_DISC_BLOCK, np.zeros((6,6))],
                    [np.zeros((6,6)), _DISC_BLOCK]])

# Lumped-mass quadrature fractions along each beam segment
_ALPHA1  = Rational(1, 3)
_ALPHA_P = Rational(1, 3)
_ALPHA2  = Rational(1, 3)
_BETA1   = Rational(1, 2)
_BETA_P  = Rational(1, 2)
_BETA2   = Rational(1, 2)




def rotmat(body_state):
    """
    Return a rotation matrix, takes the 1x6 body state
    """
    phi, theta, psi = body_state[3], body_state[4], body_state[5]
    R_X = Matrix([[1,0,0],[0,cos(phi),-sin(phi)],[0,sin(phi),cos(phi)]])
    R_Y = Matrix([[cos(theta),0,sin(theta)],[0,1,0],[-sin(theta),0,cos(theta)]])
    R_Z = Matrix([[cos(psi),-sin(psi),0],[sin(psi),cos(psi),0],[0,0,1]])
    return R_Z * R_Y * R_X

def global_pos(body_state, local_vec):
    """
    Transform local coordinate vector into global coordinate
    """
    body_pos = Matrix([body_state[:3]])
    return body_pos + local_vec*rotmat(body_state)

x2, y2, z2, psi2, theta2, phi2 = sp.symbols('x2 y2 z2 psi2, theta2, phi2')
x3, y3, z3, psi3, theta3, phi3 = sp.symbols('x3 y3 z3 psi3, theta3, phi3')
q_b1 = Matrix([[0,0,0,0,0,0]])
q_b2_0 = Matrix([[0,0,h/2,0,0,0]])
q_b2 = Matrix([[x2, y2, z2, psi2, theta2, phi2]])
q_b3_0 = Matrix([[0,0,h,0,0,0]])
q_b3 = Matrix([[x3, y3, z3, psi3, theta3, phi3]])

print("Step 1: Coordinates and frames...")
t0_build = time.time()
t = symbols('t')

# Generalised coordinates
x2, y2, z2, phi2, th2, ps2 = symbols('x2 y2 z2 phi2 theta2 psi2')
x3, y3, z3, phi3, th3, ps3 = symbols('x3 y3 z3 phi3 theta3 psi3')
q = [x2, y2, z2, phi2, th2, ps2,
     x3, y3, z3, phi3, th3, ps3]

# Generalised speeds (= qdot, simplest choice)
u2x, u2y, u2z, u2ph, u2th, u2ps = symbols('u2x u2y u2z u2ph u2th u2ps')
u3x, u3y, u3z, u3ph, u3th, u3ps = symbols('u3x u3y u3z u3ph u3th u3ps')
u = [u2x, u2y, u2z, u2ph, u2th, u2ps,
          u3x, u3y, u3z, u3ph, u3th, u3ps]


links = []
for i in range(3):





kd_eqs = [ui - qi.diff(t) for ui,qi in zip(u, q)]
_d2u   = {qi.diff(t): ui for qi,ui in zip(q, u)}

# Inertial frame
N = ReferenceFrame('N')

# Body frames with ZYX (body-fixed) orientation
# orient_body_fixed(parent, angles, 'ZYX') rotates: first Z by ps, then Y by th, then X by ph
B2 = ReferenceFrame('B2')
B2.orient_body_fixed(N, (ps2, th2, phi2), 'ZYX')

B3 = ReferenceFrame('B3')
B3.orient_body_fixed(N, (ps3, th3, phi3), 'ZYX')

# Angular velocities expressed in terms of u (not qdot)
B2.set_ang_vel(N, B2.ang_vel_in(N).subs(_d2u))
B3.set_ang_vel(N, B3.ang_vel_in(N).subs(_d2u))

print(f"  done ({time.time()-t0_build:.1f}s)")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Points and attachment geometry
# ─────────────────────────────────────────────────────────────────────────────

print("Step 2: Points and attachments...")

O = Point('O')
O.set_vel(N, 0)

P2 = Point('P2')
P2.set_pos(O, x2*N.x + y2*N.y + z2*N.z)
P2.set_vel(N, u2x*N.x + u2y*N.y + u2z*N.z)

P3 = Point('P3')
P3.set_pos(O, x3*N.x + y3*N.y + z3*N.z)
P3.set_vel(N, u3x*N.x + u3y*N.y + u3z*N.z)


def _make_attach(name, centre, frame, r_np):
    """Attachment point at centre + R*r_local."""
    rx, ry, rz = r_np
    pt = Point(name)
    pt.set_pos(centre, rx*frame.x + ry*frame.y + rz*frame.z)
    return pt


def _vel_from_pos(pt):
    """Set and return velocity by differentiating position, replacing qdot->u."""
    vel = pt.pos_from(O).diff(t, N).subs(_dot_to_u)
    pt.set_vel(N, vel)
    return vel


# Bay 1: A fixed on B1, B on B2
# Bay 2: A on B2,        B on B3
A1 = [_make_attach(f'A1_{i}', O,  N,  R_LOCAL[i]) for i in range(3)]
B1 = [_make_attach(f'B1_{i}', P2, B2, R_LOCAL[i]) for i in range(3)]
A2 = [_make_attach(f'A2_{i}', P2, B2, R_LOCAL[i]) for i in range(3)]
B2_att = [_make_attach(f'B2_{i}', P3, B3, R_LOCAL[i]) for i in range(3)]

for pt in A1:
    pt.set_vel(N, 0)

for pt in B1 + A2 + B2_att:
    _vel_from_pos(pt)

# Midpoints
M1 = []
M2 = []
for i in range(3):
    m = Point(f'M1_{i}')
    m.set_pos(O, Rational(1,2)*(A1[i].pos_from(O) + B1[i].pos_from(O)))
    _vel_from_pos(m)
    M1.append(m)

    m = Point(f'M2_{i}')
    m.set_pos(O, Rational(1,2)*(A2[i].pos_from(O) + B2_att[i].pos_from(O)))
    _vel_from_pos(m)
    M2.append(m)