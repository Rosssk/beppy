"""
Pseudo-Rigid Body Model (PRBM) with RPR links, flexure DOFs eliminated.

Structure:
    Body A (grounded) -> 3x RPR flexures -> Body B -> 3x RPR flexures -> Body C

RPR link geometry:
    - Rigid link from body attachment point (fixed in body frame)
    - Rotational spring at proximal end  -> angle theta = f(qB, qC)
    - Prismatic joint                    -> extension d = f(qB, qC)
    - Rotational spring at distal end    -> angle phi   = f(qB, qC)
    - Rigid link to next body attachment point

All flexure DOFs (d, theta, phi) are expressed as functions of the 12
body generalized coordinates. No constraint equations needed.

Generalized coordinates (12 total):
    qB = [xB, yB, zB, psiB, thB, phiB]   Body B (translation + ZYX Euler)
    qC = [xC, yC, zC, psiC, thC, phiC]   Body C
"""

import sympy as sp
import sympy.physics.mechanics as me
from sympy import cos, sin, pi, sqrt, symbols, Matrix, eye
from sympy.physics.mechanics import msubs

# ─────────────────────────────────────────────
# Symbols
# ─────────────────────────────────────────────
t = sp.Symbol('t')
h, r, L0 = symbols('h r L0', positive=True)   # L0 = beam rest length
k_rot, k_pri = symbols('k_rot k_pri', positive=True)
g_sym = symbols('g', positive=True)
m_B, m_C = symbols('m_B m_C', positive=True)
Ixx_B, Iyy_B, Izz_B = symbols('Ixx_B Iyy_B Izz_B', positive=True)
Ixx_C, Iyy_C, Izz_C = symbols('Ixx_C Iyy_C Izz_C', positive=True)

n = 3  # beams per layer

# ─────────────────────────────────────────────
# Generalized coordinates
# ─────────────────────────────────────────────
def make_qs(name):
    labels = ('x', 'y', 'z', 'psi', 'th', 'phi')
    q   = [me.dynamicsymbols(f'{name}_{c}') for c in labels]
    qd  = [qi.diff(t)                        for qi in q]
    qdd = [qi.diff(t, 2)                     for qi in q]
    return q, qd, qdd

qB, qdB, qddB = make_qs('B')
qC, qdC, qddC = make_qs('C')
xB, yB, zB, psiB, thB, phiB = qB
xC, yC, zC, psiC, thC, phiC = qC

q_all   = qB + qC
qd_all  = qdB + qdC
qdd_all = qddB + qddC

# ─────────────────────────────────────────────
# Rotation matrix: ZYX Euler (psi, th, phi)
# ─────────────────────────────────────────────
def Rzyx(psi, th, phi):
    """Rotation matrix from body frame to inertial frame."""
    Rz = Matrix([[cos(psi), -sin(psi), 0],
                 [sin(psi),  cos(psi), 0],
                 [0,         0,        1]])
    Ry = Matrix([[ cos(th), 0, sin(th)],
                 [0,        1, 0      ],
                 [-sin(th), 0, cos(th)]])
    Rx = Matrix([[1, 0,        0        ],
                 [0, cos(phi), -sin(phi)],
                 [0, sin(phi),  cos(phi)]])
    return Rz * Ry * Rx

RB = Rzyx(psiB, thB, phiB)
RC = Rzyx(psiC, thC, phiC)

# Body origins in inertial frame
pB = Matrix([xB, yB, h/2 + zB])
pC = Matrix([xC, yC, h     + zC])

# ─────────────────────────────────────────────
# Attachment points in inertial frame
# ─────────────────────────────────────────────
def attach_inertial(body_origin, R_body, local_vec):
    return body_origin + R_body * local_vec

def layer0_attachA(i):
    a = i/n * 2*pi
    return Matrix([r*cos(a), r*sin(a), 0])

def layer0_attachB(i):
    a = (i+1)/n * 2*pi
    return attach_inertial(pB, RB, Matrix([r*cos(a), r*sin(a), 0]))

def layer1_attachC(i):
    a = (i - sp.Rational(1,2))/n * 2*pi
    return attach_inertial(pC, RC, Matrix([r*cos(a), r*sin(a), 0]))

def layer1_attachB(i):
    a = (i + sp.Rational(1,2))/n * 2*pi
    return attach_inertial(pB, RB, Matrix([r*cos(a), r*sin(a), 0]))

# ─────────────────────────────────────────────
# Flexure kinematics: d, theta, phi as f(q)
#
# For RPR link between proximal point P (on body with R_prox)
# and distal point Q (on body with R_dist):
#   d     = |Q-P| - L0
#   theta = angle between beam vector and proximal body z-axis
#   phi   = angle between (-beam vector) and distal body z-axis
# ─────────────────────────────────────────────
def flexure_kinematics(P, Q, R_prox, R_dist):
    v      = Q - P
    length = sqrt(v.dot(v))
    d      = length - L0

    v_hat  = v / length
    z_prox = R_prox * Matrix([0, 0, 1])
    z_dist = R_dist * Matrix([0, 0, 1])

    # Use dot product directly in PE; avoid acos for cleaner derivatives
    # PE ~ k*(1 - cos(angle))  or  k*(v_hat.dot(z) - 1)^2
    # Here we use the squared-dot-product form (equivalent near equilibrium)
    cos_th = v_hat.dot(z_prox)
    cos_ph = (-v_hat).dot(z_dist)

    return d, cos_th, cos_ph

# ─────────────────────────────────────────────
# Potential energy
#
# Spring PE uses (1 - cos_angle)^2 * k_rot  for rotational springs
# (exact, avoids acos and its derivative singularities)
# and k_pri/2 * d^2 for prismatic spring
# ─────────────────────────────────────────────
V_spring = sp.Integer(0)

for i in range(n):
    # Layer 0: A -> B
    P0, Q0 = layer0_attachA(i), layer0_attachB(i)
    d0, c0_th, c0_ph = flexure_kinematics(P0, Q0, eye(3), RB)
    V_spring += sp.Rational(1,2)*k_pri*d0**2
    V_spring += k_rot*(1 - c0_th)**2
    V_spring += k_rot*(1 - c0_ph)**2

    # Layer 1: C -> B
    P1, Q1 = layer1_attachC(i), layer1_attachB(i)
    d1, c1_th, c1_ph = flexure_kinematics(P1, Q1, RC, RB)
    V_spring += sp.Rational(1,2)*k_pri*d1**2
    V_spring += k_rot*(1 - c1_th)**2
    V_spring += k_rot*(1 - c1_ph)**2

V_grav  = m_B*g_sym*(h/2 + zB) + m_C*g_sym*(h + zC)
V_total = V_spring + V_grav

# ─────────────────────────────────────────────
# Kinetic energy
# ─────────────────────────────────────────────
def omega_ZYX(psi, th, phi, dpsi, dth, dphi):
    """Angular velocity in inertial frame for ZYX Euler angles."""
    wx = dphi - dpsi*sin(th)
    wy = dth*cos(phi) + dpsi*cos(th)*sin(phi)
    wz = -dth*sin(phi) + dpsi*cos(th)*cos(phi)
    return Matrix([wx, wy, wz])

wB = omega_ZYX(psiB, thB, phiB, *qdB[3:])
wC = omega_ZYX(psiC, thC, phiC, *qdC[3:])

vB_vec = Matrix(qdB[:3])
vC_vec = Matrix(qdC[:3])

IB = Matrix([[Ixx_B, 0, 0], [0, Iyy_B, 0], [0, 0, Izz_B]])
IC = Matrix([[Ixx_C, 0, 0], [0, Iyy_C, 0], [0, 0, Izz_C]])

T_B = sp.Rational(1,2)*m_B*vB_vec.dot(vB_vec) + sp.Rational(1,2)*(wB.T*IB*wB)[0]
T_C = sp.Rational(1,2)*m_C*vC_vec.dot(vC_vec) + sp.Rational(1,2)*(wC.T*IC*wC)[0]
T_total = T_B + T_C

# ─────────────────────────────────────────────
# Lagrangian and EOMs
# ─────────────────────────────────────────────
L = T_total - V_total

print("Forming Lagrange equations (12 DOF)...")
LM = me.LagrangesMethod(L, q_all)
eoms = LM.form_lagranges_equations()
print("EOMs formed. Shape:", eoms.shape)

# ─────────────────────────────────────────────
# Numerical simulation
# ─────────────────────────────────────────────
import numpy as np
from scipy.integrate import solve_ivp

param_vals = {
    h: 0.10, r: 0.05, L0: 0.06,
    k_rot: 1.0, k_pri: 500.0, g_sym: 9.81,
    m_B: 0.1, m_C: 0.1,
    Ixx_B: 1e-5, Iyy_B: 1e-5, Izz_B: 2e-5,
    Ixx_C: 1e-5, Iyy_C: 1e-5, Izz_C: 2e-5,
}

print("Substituting parameters and extracting M, f...")
eoms_num = eoms.subs(param_vals)
qdd_vec  = Matrix(qdd_all)
M_mat    = eoms_num.jacobian(qdd_all)
f_vec    = -(eoms_num - M_mat * qdd_vec)

# qd_all and qdd_all are now q.diff(t) and q.diff(t,2), so they ARE the
# Derivative objects LagrangesMethod produces — no substitution needed.
# lambdify needs flat scalar symbols to substitute at runtime, so we create
# plain symbols as stand-ins and map Derivative objects -> plain symbols.
q_syms   = [sp.Symbol(f'q{i}')   for i in range(len(q_all))]
qd_syms  = [sp.Symbol(f'qd{i}')  for i in range(len(qd_all))]
qdd_syms = [sp.Symbol(f'qdd{i}') for i in range(len(qdd_all))]

flat_map = dict(zip(q_all + qd_all + qdd_all, q_syms + qd_syms + qdd_syms))
M_flat   = msubs(M_mat, flat_map)
f_flat   = msubs(f_vec, flat_map)

remaining = M_flat.atoms(sp.Derivative) | f_flat.atoms(sp.Derivative)
if remaining:
    raise RuntimeError(f"Unresolved Derivatives: {remaining}")

# f_flat should be free of qdd_syms after jacobian extraction.
# If any survive (numerical residual), zero them — they are artefacts.
zero_qdd = {s: sp.Integer(0) for s in qdd_syms}
f_flat = f_flat.subs(zero_qdd)

# Verify f_flat contains only q_syms and qd_syms
leftover = f_flat.free_symbols - set(q_syms) - set(qd_syms)
if leftover:
    raise RuntimeError(f"Unexpected symbols in f_flat: {leftover}")

print("Lambdifying M and f...")
args_sym = q_syms + qd_syms
M_func   = sp.lambdify(args_sym, M_flat, 'numpy')
f_func   = sp.lambdify(args_sym, f_flat, 'numpy')

def odes(t_val, state):
    nq        = len(q_all)
    args_vals = list(state[:nq]) + list(state[nq:])
    M = np.array(M_func(*args_vals), dtype=float)
    f = np.array(f_func(*args_vals), dtype=float).flatten()
    qdd = np.linalg.solve(M, f)
    return np.concatenate([state[nq:], qdd])

nq     = len(q_all)
state0 = np.zeros(2*nq)

print("Running simulation (t=0..0.5s)...")
sol = solve_ivp(odes, [0, 0.5], state0, method='RK45',
                max_step=1e-3, rtol=1e-6, atol=1e-9)
print(f"Done. Steps={len(sol.t)}, Success={sol.success}")
print(f"Final zB={sol.y[2,-1]:.6f} m")
print(f"Final zC={sol.y[8,-1]:.6f} m")

# ─────────────────────────────────────────────
# Animation
# ─────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Numerical parameter values (plain floats)
pv = {k: float(v) for k, v in param_vals.items()}

def Rzyx_num(psi, th, phi):
    """Numerical ZYX rotation matrix."""
    cp, sp_ = np.cos(psi), np.sin(psi)
    ct, st  = np.cos(th),  np.sin(th)
    cf, sf  = np.cos(phi), np.sin(phi)
    Rz = np.array([[cp, -sp_, 0], [sp_, cp, 0], [0, 0, 1]])
    Ry = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    Rx = np.array([[1, 0, 0], [0, cf, -sf], [0, sf, cf]])
    return Rz @ Ry @ Rx

def get_geometry(q_vals):
    """
    Given q_vals = [xB,yB,zB,psiB,thB,phiB, xC,yC,zC,psiC,thC,phiC],
    return:
      body_pts : (3,3) array  [A_origin, B_origin, C_origin]
      beams    : list of (P, Q) pairs (each a 3-vec) for all 6 beams
      discs    : list of (origin, R) for each body disc (for orientation viz)
    """
    h_  = pv[h];  r_  = pv[r]
    xB_, yB_, zB_ = q_vals[0:3]
    psiB_, thB_, phiB_ = q_vals[3:6]
    xC_, yC_, zC_ = q_vals[6:9]
    psiC_, thC_, phiC_ = q_vals[9:12]

    RB_ = Rzyx_num(psiB_, thB_, phiB_)
    RC_ = Rzyx_num(psiC_, thC_, phiC_)

    pA = np.array([0.0, 0.0, 0.0])
    pB_ = np.array([xB_, yB_, h_/2 + zB_])
    pC_ = np.array([xC_, yC_, h_   + zC_])

    beams = []
    for i in range(n):
        # Layer 0: A -> B
        aA = i/n * 2*np.pi
        aB0 = (i+1)/n * 2*np.pi
        PA = np.array([r_*np.cos(aA), r_*np.sin(aA), 0.0])
        QB = pB_ + RB_ @ np.array([r_*np.cos(aB0), r_*np.sin(aB0), 0.0])
        beams.append((PA, QB))

        # Layer 1: C -> B
        aC = (i - 0.5)/n * 2*np.pi
        aB1 = (i + 0.5)/n * 2*np.pi
        PC = pC_ + RC_ @ np.array([r_*np.cos(aC), r_*np.sin(aC), 0.0])
        QB1 = pB_ + RB_ @ np.array([r_*np.cos(aB1), r_*np.sin(aB1), 0.0])
        beams.append((PC, QB1))

    body_pts = np.array([pA, pB_, pC_])
    discs = [(pA, np.eye(3)), (pB_, RB_), (pC_, RC_)]
    return body_pts, beams, discs

def disc_circle(origin, R, radius, n_pts=40):
    """Points of a circle in the body xy-plane, in inertial frame."""
    angles = np.linspace(0, 2*np.pi, n_pts)
    local  = np.array([radius*np.cos(angles), radius*np.sin(angles),
                        np.zeros(n_pts)])        # (3, n_pts)
    world  = origin[:,None] + R @ local          # (3, n_pts)
    return world

# Downsample to ~200 frames for smooth animation
n_frames = min(200, len(sol.t))
frame_idx = np.linspace(0, len(sol.t)-1, n_frames, dtype=int)

fig = plt.figure(figsize=(7, 8))
ax  = fig.add_subplot(111, projection='3d')

h_val = pv[h]; r_val = pv[r]
pad   = r_val * 1.5
ax.set_xlim(-pad, pad); ax.set_ylim(-pad, pad)
ax.set_zlim(-0.02, h_val + 0.02)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title('PRBM Simulation')

# Colour scheme
BODY_COLORS  = ['#2196F3', '#FF9800', '#4CAF50']   # A, B, C
BEAM_COLORS  = ['#90CAF9']*3 + ['#FFCC80']*3        # layer0, layer1
DISC_ALPHA   = 0.25

# Static body-A disc (grounded)
body_pts0, beams0, discs0 = get_geometry(np.zeros(12))
circ_A = disc_circle(discs0[0][0], discs0[0][1], r_val)
ax.plot(circ_A[0], circ_A[1], circ_A[2], color=BODY_COLORS[0], lw=2)
ax.scatter(*discs0[0][0], color=BODY_COLORS[0], s=60, zorder=5)

# Dynamic artists
beam_lines = [ax.plot([], [], [], color=BEAM_COLORS[j], lw=2)[0]
              for j in range(2*n)]
disc_lines  = [ax.plot([], [], [], color=BODY_COLORS[k+1], lw=2)[0]
               for k in range(2)]          # B and C discs
body_dots   = [ax.scatter([], [], [], color=BODY_COLORS[k+1], s=60, zorder=5)
               for k in range(2)]
time_text   = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=9)

def init():
    for ln in beam_lines + disc_lines:
        ln.set_data([], [])
        ln.set_3d_properties([])
    return beam_lines + disc_lines

def update(frame):
    q_vals = sol.y[:12, frame_idx[frame]]
    body_pts, beams, discs = get_geometry(q_vals)

    # Beams
    for j, (P, Q) in enumerate(beams):
        beam_lines[j].set_data([P[0], Q[0]], [P[1], Q[1]])
        beam_lines[j].set_3d_properties([P[2], Q[2]])

    # Body B and C discs + dots
    for k, (origin, R) in enumerate(discs[1:]):
        circ = disc_circle(origin, R, r_val)
        disc_lines[k].set_data(circ[0], circ[1])
        disc_lines[k].set_3d_properties(circ[2])
        body_dots[k]._offsets3d = ([origin[0]], [origin[1]], [origin[2]])

    time_text.set_text(f't = {sol.t[frame_idx[frame]]:.3f} s')
    return beam_lines + disc_lines + [time_text]

ani = animation.FuncAnimation(fig, update, frames=n_frames,
                               init_func=init, interval=30, blit=False)

out_path = 'prbm_animation.gif'
print(f"Saving animation to {out_path} ...")
ani.save(out_path, writer='pillow', fps=30)
print("Saved.")
plt.show()