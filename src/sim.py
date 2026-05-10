# simulate.py
#
# FAST simulation script for SymPy Mechanics systems
#
# Key optimizations:
#   - NO symbolic LUsolve()
#   - Numerical linear solve only
#   - Lambdified M and f separately
#   - cse=True
#   - Stiff solver (Radau)
#   - Progress bar
#

import cloudpickle
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from tqdm import tqdm
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Load precompiled functions
# ---------------------------------------------------------------------
with open('eoms.pkl', 'rb') as file:
    data = cloudpickle.load(file)

q = data['q']
u = data['u']

qdot_func = data['qdot_func']
M_func = data['M_func']
f_func = data['f_func']


# ---------------------------------------------------------------------
# ODE RHS
# ---------------------------------------------------------------------
def odefun(t, x, p):

    nq = len(q)

    q_vals = x[:nq]
    u_vals = x[nq:]

    args = list(q_vals) + list(u_vals) + list(p)

    # -------------------------------------------------------------
    # qdot
    # -------------------------------------------------------------
    qdot = np.asarray(
        qdot_func(*args),
        dtype=float
    ).flatten()

    # -------------------------------------------------------------
    # Evaluate M(q)
    # -------------------------------------------------------------
    M_eval = np.asarray(
        M_func(*args),
        dtype=float
    )

    # -------------------------------------------------------------
    # Evaluate forcing
    # -------------------------------------------------------------
    f_eval = np.asarray(
        f_func(*args),
        dtype=float
    ).flatten()

    # -------------------------------------------------------------
    # Numerical solve:
    #
    #     M udot = f
    # -------------------------------------------------------------
    udot = np.linalg.solve(M_eval, f_eval)

    return np.concatenate([qdot, udot])


# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
p = [
    9.81,       # g
    0.5,        # gamma
    5.0,        # k_th
    1000.0,     # k_s
    2.0e9,      # E_mod
    0.30,       # V_rat
    1200.0,     # rho
    0.10,       # h
    0.03,       # r_f
    0.05,       # r_b
    0.002,      # t_f
    0.005,      # t_b
]


# ---------------------------------------------------------------------
# Initial conditions
#
# State:
#   x = [q, u]
# ---------------------------------------------------------------------
nq = len(q)
nu = len(u)

x0 = np.zeros(nq + nu)

# Small perturbations
x0[3] = 0.05
x0[9] = -0.03


# ---------------------------------------------------------------------
# Simulation settings
# ---------------------------------------------------------------------
t0 = 0.0
tf = 5.0

# Chunk size for progress updates
dt_chunk = 0.05

times = np.arange(t0, tf + dt_chunk, dt_chunk)


# ---------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------
X = [x0.copy()]
T = [t0]

x_current = x0.copy()


# ---------------------------------------------------------------------
# Progress-tracked integration
# ---------------------------------------------------------------------
with tqdm(total=len(times) - 1, desc='Simulating') as pbar:

    for i in range(len(times) - 1):

        t_start = times[i]
        t_end = times[i + 1]

        sol = solve_ivp(
            lambda t, x: odefun(t, x, p),
            (t_start, t_end),
            x_current,

            # MUCH better for stiff mechanics systems
            method='Radau',

            # Relaxed tolerances for speed
            rtol=1e-4,
            atol=1e-6,
        )

        # Last state becomes next IC
        x_current = sol.y[:, -1]

        X.append(x_current.copy())
        T.append(t_end)

        pbar.update(1)


# ---------------------------------------------------------------------
# Convert to arrays
# ---------------------------------------------------------------------
T = np.array(T)
X = np.array(X).T


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
plt.figure(figsize=(10, 6))

# z positions
plt.plot(T, X[2], label='z2')
plt.plot(T, X[8], label='z3')

plt.xlabel('Time [s]')
plt.ylabel('Vertical Position [m]')

plt.title('PRBM Dynamics')

plt.grid(True)
plt.legend()

plt.show()