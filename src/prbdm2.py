"""
simulate_mechanism.py
=====================
Numerical simulation of the 3-disc compliant parallel mechanism.

Pipeline
--------
1. Build symbolic EOMs with Kane's method (from the user's model).
2. Lambdify M(q, u) and f(q, u) with concrete parameter values.
3. Integrate M·u̇ = f via scipy solve_ivp (Radau — stiff-safe).
4. Apply a random impulsive force to the top disc (B3) at t=0
   by perturbing the initial velocity state.
5. Save trajectory to mechanism_trajectory.npz for post-processing / animation.

Physical parameters
-------------------
All SI units.  Tune the constants block to match your hardware.

Dependencies
------------
    pip install sympy scipy numpy matplotlib
"""

import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve as la_solve
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Symbolic model (reuse the user's module) ────────────────────────────────
from sympy import symbols, lambdify, pi
from sympy.physics.vector import dynamicsymbols

# Import the model builder
from prbdm import main as build_eom   # the user's file, renamed

# ============================================================
#  PHYSICAL CONSTANTS  –  tune here
# ============================================================
PARAMS = dict(
    g      = 9.81,       # m/s²
    rho    = 2700.0,     # kg/m³  (aluminium)
    t_b    = 0.005,      # m      disc thickness
    r_b    = 0.050,      # m      disc radius  50 mm
    r_f    = 0.040,      # m      attachment radius on disc
    h      = 0.120,      # m      layer spacing
    t_f    = 0.001,      # m      (unused in EOM but kept for completeness)
    E_mod  = 70e9,       # Pa
    V_rat  = 0.33,
    gamma  = 0.85,       # PRBM characteristic radius factor
    k_th   = 0.50,       # N·m    bending stiffness
    k_s    = 5000.0,     # N/m    axial stiffness
)

# ============================================================
#  SIMULATION SETTINGS
# ============================================================
T_END   = 40.0          # s   — long enough to observe stability / decay
DT_SAVE = 0.005         # s   — output resolution
RTOL    = 1e-6
ATOL    = 1e-9

# Random impulse applied to top disc (B3) at t=0.
# Modelled as an instantaneous velocity change: Δv = J/m, Δω = τ/I
RNG_SEED   = 42
IMPULSE_SCALE = 0.15    # m/s  — scale of random translational impulse
TORQUE_SCALE  = 0.30    # rad/s — scale of random angular velocity kick

# ============================================================
#  1. BUILD SYMBOLIC MODEL
# ============================================================
print("Building symbolic EOMs … (may take ~60 s)")
t0 = time.time()
KM, M_sym, f_sym, _, _ = build_eom()
print(f"  Done in {time.time()-t0:.1f} s")

# Generalised coordinates and speeds (12 each: 6 for B2, 6 for B3)
q_syms = KM.q          # [x2,y2,z2,phi2,theta2,psi2, x3,y3,z3,phi3,theta3,psi3]
u_syms = KM.u          # matching speeds

# Collect all free symbols that need numerical values
all_free = M_sym.free_symbols | f_sym.free_symbols
# Remove dynamic symbols (they're the states, not parameters)
param_syms = sorted(all_free - set(q_syms) - set(u_syms),
                    key=lambda s: str(s))

print(f"  Parameter symbols: {[str(s) for s in param_syms]}")

# ============================================================
#  2. LAMBDIFY
# ============================================================
print("Lambdifying M and f …")
t0 = time.time()

state_syms = list(q_syms) + list(u_syms)   # 24 symbols

M_func = lambdify(state_syms + param_syms, M_sym, modules='numpy')
f_func = lambdify(state_syms + param_syms, f_sym, modules='numpy')

print(f"  Done in {time.time()-t0:.1f} s")

# Build ordered parameter value array to pass to lambdified functions
# (order must match param_syms, which is sorted alphabetically)
param_map = {str(s): PARAMS[str(s)] for s in param_syms}
param_vals = np.array([param_map[str(s)] for s in param_syms])

# Compute mass of one disc for impulse → velocity conversion
m_disc = PARAMS['rho'] * PARAMS['t_b'] * np.pi * PARAMS['r_b']**2
Ixx_disc = m_disc * PARAMS['r_b']**2 / 4.0

# ============================================================
#  3. INITIAL CONDITIONS
# ============================================================
# Equilibrium position
# B2 at (0, 0, h/2),  B3 at (0, 0, h)  — all angles zero, all vel zero
h = PARAMS['h']
q0 = np.array([
    0, 0, h/2, 0, 0, 0,   # B2: x2,y2,z2,phi2,theta2,psi2
    0, 0, h,   0, 0, 0,   # B3: x3,y3,z3,phi3,theta3,psi3
], dtype=float)

u0 = np.zeros(12, dtype=float)

# Random impulse on B3 → initial velocity perturbation
rng = np.random.default_rng(RNG_SEED)
dv  = rng.standard_normal(3) * IMPULSE_SCALE     # translational
dom = rng.standard_normal(3) * TORQUE_SCALE       # angular

# B3 velocities are u_syms[6:12] → state indices 12+6 : 12+12
# In the state vector [q(12), u(12)] the speeds start at index 12
u0[6:9]  = dv    # vx3, vy3, vz3
u0[9:12] = dom   # om1_3, om2_3, om3_3

print(f"\nRandom impulse applied to top disc (B3):")
print(f"  Δv  = {dv}  m/s")
print(f"  Δω  = {dom}  rad/s")

state0 = np.concatenate([q0, u0])   # length 24

# ============================================================
#  4. ODE RIGHT-HAND SIDE
# ============================================================
def rhs(t, state):
    q_num = state[:12]
    u_num = state[12:]

    args = list(q_num) + list(u_num) + list(param_vals)

    M_num = np.array(M_func(*args), dtype=float)
    f_num = np.array(f_func(*args), dtype=float).ravel()

    # Solve M·u̇ = f
    udot = la_solve(M_num, f_num)

    # KDEs: q̇ computed from kinematic differential equations
    # For the translational coords: q̇ = u directly (by construction in build_kdes)
    # For the rotational coords: already embedded in kdes → we can pull qdot
    # from KM.kindiffdict, but the cleanest route here is:
    #   qdot = KM.mass_matrix_full \ KM.forcing_full  (includes kdes)
    # To avoid re-lambdifying the full system, we reconstruct qdot from u
    # using the angle-rate relations inverted:
    #   The kdes define u = Γ(q)·qdot_rot, so qdot_rot = Γ⁻¹·u
    # For XYZ Euler angles (phi, theta, psi) the inverse of the body-rate map is:
    #   phi_dot   = (om1 cos_psi - om2 sin_psi) / cos_theta
    #   theta_dot =  om1 sin_psi + om2 cos_psi
    #   psi_dot   = (om2 sin_psi - om1 cos_psi) * tan_theta + om3
    # Applied separately for B2 (indices 3-5 of q, 3-5 of u) and
    # B3 (indices 9-11 of q, 9-11 of u).

    qdot = np.empty(12)

    for k in range(2):          # k=0 → B2,  k=1 → B3
        qs = 6*k;  us = 6*k    # slices into q_num and u_num
        # Translational
        qdot[qs:qs+3] = u_num[us:us+3]
        # Rotational
        phi, theta, psi = q_num[qs+3:qs+6]
        om1, om2, om3   = u_num[us+3:us+6]
        ct = np.cos(theta); tt = np.tan(theta)
        cp = np.cos(psi);   sp = np.sin(psi)
        qdot[qs+3] = ( om1*cp - om2*sp) / ct          # phi_dot
        qdot[qs+4] =   om1*sp + om2*cp                # theta_dot
        qdot[qs+5] = (-om1*cp + om2*sp)*tt + om3      # psi_dot

    return np.concatenate([qdot, udot])

# ============================================================
#  5. INTEGRATE
# ============================================================
t_eval = np.arange(0, T_END + DT_SAVE, DT_SAVE)

print(f"\nIntegrating over {T_END} s with Radau …")
t0 = time.time()

sol = solve_ivp(
    rhs,
    t_span   = (0.0, T_END),
    y0       = state0,
    method   = 'Radau',
    t_eval   = t_eval,
    rtol     = RTOL,
    atol     = ATOL,
    dense_output = False,
)

elapsed = time.time() - t0
print(f"  Solver finished in {elapsed:.1f} s")
print(f"  Status  : {sol.message}")
print(f"  Steps   : {sol.t.shape[0]} saved points")

if not sol.success:
    raise RuntimeError(f"Solver failed: {sol.message}")

# ============================================================
#  6. POST-PROCESS
# ============================================================
t_arr = sol.t
q_arr = sol.y[:12, :]    # (12, N)
u_arr = sol.y[12:, :]    # (12, N)

# Positions of disc centres in N
# B1 (fixed): origin (0, 0, 0)
# B2: q_arr[0:3]
# B3: q_arr[6:9]
pos_B1 = np.zeros((3, t_arr.size))
pos_B2 = q_arr[0:3, :]
pos_B3 = q_arr[6:9, :]

# Kinetic energy  T = ½ uᵀ M u  (approximate — uses diagonal mass terms)
# For a rough stability indicator we track ||u||
speed_B2 = np.linalg.norm(u_arr[0:6, :], axis=0)
speed_B3 = np.linalg.norm(u_arr[6:12, :], axis=0)

# ============================================================
#  7. SAVE TRAJECTORY
# ============================================================
np.savez_compressed(
    'mechanism_trajectory.npz',
    t      = t_arr,
    q      = q_arr,
    u      = u_arr,
    pos_B1 = pos_B1,
    pos_B2 = pos_B2,
    pos_B3 = pos_B3,
    params = np.array([PARAMS[k] for k in sorted(PARAMS)]),
    param_keys = np.array(sorted(PARAMS.keys())),
    impulse_dv  = dv,
    impulse_dom = dom,
)
print("\nTrajectory saved → mechanism_trajectory.npz")

# ============================================================
#  8. STABILITY DIAGNOSTIC PLOTS
# ============================================================
fig = plt.figure(figsize=(14, 10), facecolor='#0d1117')
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

DARK  = '#0d1117'
GRID  = '#21262d'
C1    = '#58a6ff'   # B2 colour
C2    = '#f78166'   # B3 colour
CREF  = '#3fb950'   # reference / equilibrium

def styled_ax(ax, title):
    ax.set_facecolor(DARK)
    ax.spines[:].set_color(GRID)
    ax.tick_params(colors='#8b949e', labelsize=8)
    ax.xaxis.label.set_color('#8b949e')
    ax.yaxis.label.set_color('#8b949e')
    ax.set_title(title, color='#e6edf3', fontsize=9, pad=6)
    ax.grid(color=GRID, linewidth=0.6)

# --- X-Y trajectory of disc centres (top view) ---
ax0 = fig.add_subplot(gs[0, 0])
styled_ax(ax0, 'Top-view trajectory (X-Y plane)')
ax0.plot(pos_B2[0], pos_B2[1], color=C1, lw=0.8, label='B2 (mid)')
ax0.plot(pos_B3[0], pos_B3[1], color=C2, lw=0.8, label='B3 (top)')
ax0.plot(0, 0, 'o', color=CREF, ms=5, label='B1 (fixed)')
ax0.set_xlabel('x  [m]'); ax0.set_ylabel('y  [m]')
ax0.legend(fontsize=7, labelcolor='#8b949e', facecolor=DARK,
           edgecolor=GRID)

# --- Z displacement vs time ---
ax1 = fig.add_subplot(gs[0, 1])
styled_ax(ax1, 'Vertical displacement vs time')
ax1.axhline(h/2, color=C1, lw=0.5, ls='--', alpha=0.4)
ax1.axhline(h,   color=C2, lw=0.5, ls='--', alpha=0.4)
ax1.plot(t_arr, pos_B2[2], color=C1, lw=0.8, label='B2 z')
ax1.plot(t_arr, pos_B3[2], color=C2, lw=0.8, label='B3 z')
ax1.set_xlabel('t  [s]'); ax1.set_ylabel('z  [m]')
ax1.legend(fontsize=7, labelcolor='#8b949e', facecolor=DARK, edgecolor=GRID)

# --- Euler angles B2 ---
ax2 = fig.add_subplot(gs[1, 0])
styled_ax(ax2, 'B2 Euler angles vs time')
labels = ['φ₂', 'θ₂', 'ψ₂']
colors = ['#79c0ff', '#d2a8ff', '#ffa657']
for k, (lb, col) in enumerate(zip(labels, colors)):
    ax2.plot(t_arr, np.degrees(q_arr[3+k]), color=col, lw=0.8, label=lb)
ax2.set_xlabel('t  [s]'); ax2.set_ylabel('angle  [°]')
ax2.legend(fontsize=7, labelcolor='#8b949e', facecolor=DARK, edgecolor=GRID)

# --- Euler angles B3 ---
ax3 = fig.add_subplot(gs[1, 1])
styled_ax(ax3, 'B3 Euler angles vs time')
for k, (lb, col) in enumerate(zip(['φ₃', 'θ₃', 'ψ₃'], colors)):
    ax3.plot(t_arr, np.degrees(q_arr[9+k]), color=col, lw=0.8, label=lb)
ax3.set_xlabel('t  [s]'); ax3.set_ylabel('angle  [°]')
ax3.legend(fontsize=7, labelcolor='#8b949e', facecolor=DARK, edgecolor=GRID)

# --- Generalised speed norm (stability proxy) ---
ax4 = fig.add_subplot(gs[2, 0])
styled_ax(ax4, '‖u‖ — speed norm (stability proxy)')
ax4.plot(t_arr, speed_B2, color=C1, lw=0.8, label='B2')
ax4.plot(t_arr, speed_B3, color=C2, lw=0.8, label='B3')
ax4.set_xlabel('t  [s]'); ax4.set_ylabel('‖u‖  [mixed]')
ax4.legend(fontsize=7, labelcolor='#8b949e', facecolor=DARK, edgecolor=GRID)

# --- X displacement envelope ---
ax5 = fig.add_subplot(gs[2, 1])
styled_ax(ax5, 'Lateral displacement envelope')
ax5.plot(t_arr, pos_B2[0], color=C1, lw=0.6, alpha=0.7, label='B2 x')
ax5.plot(t_arr, pos_B3[0], color=C2, lw=0.6, alpha=0.7, label='B3 x')
ax5.plot(t_arr, pos_B2[1], color=C1, lw=0.6, ls='--', alpha=0.4, label='B2 y')
ax5.plot(t_arr, pos_B3[1], color=C2, lw=0.6, ls='--', alpha=0.4, label='B3 y')
ax5.set_xlabel('t  [s]'); ax5.set_ylabel('displacement  [m]')
ax5.legend(fontsize=7, labelcolor='#8b949e', facecolor=DARK, edgecolor=GRID,
           ncol=2)

fig.suptitle(
    f'3-Disc Compliant Mechanism — Stability Simulation  '
    f'(impulse seed={RNG_SEED}, scale={IMPULSE_SCALE} m/s)',
    color='#e6edf3', fontsize=11, y=0.98
)

plt.savefig('mechanism_stability.png', dpi=150, bbox_inches='tight',
            facecolor=DARK)
print("Diagnostic plot saved → mechanism_stability.png")
plt.show()

print("\nDone.  Key outputs:")
print("  mechanism_trajectory.npz  — full state history")
print("  mechanism_stability.png   — 6-panel stability diagnostic")