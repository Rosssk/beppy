import numpy as np
from scipy.integrate import solve_ivp
from sympy import symbols, Matrix, lambdify, acos, sqrt, atan2
from sympy.physics.mechanics import RigidBody, Inertia, inertia, KanesMethod
from sympy.physics.vector import ReferenceFrame, Point, dynamicsymbols
import plotly.graph_objects as go
from sympy.utilities.autowrap import autowrap

# -------------------------------------
# Symbolic derivation
# -------------------------------------
m, Iz, L0, Ks, gamma, Kth = symbols('m Iz L0 Ks gamma Kth')

x, y, theta = dynamicsymbols('x y theta')
u, v, omega = dynamicsymbols('u v omega')
ic = {
    x: L0, y: 0, theta: 0,
    u: 0, v: 0, omega: 0
}

N = ReferenceFrame('N')
O = Point('O')
O.set_vel(N, 0 * N.x)

# Create body
Nb = N.orientnew('Nb', 'Axis', [theta, N.z])
Nb.set_ang_vel(N, omega * N.z)
Pb = O.locatenew('Pb', x * N.x + y * N.y)
Pb.set_vel(N, u * N.x + v * N.y)
I = inertia(Nb, 0, 0, Iz)
B = RigidBody('B', Pb, Nb, m, (I, Pb))

# Set up PRBM RPR link
link_vec = O.pos_from(Pb)
# Create the end points of the rigid segments at each end of the PRBM link.
rigid_segment_a = O.locatenew('spring_a', -link_vec.express(N).subs(ic)*gamma)
rigid_segment_b = Pb.locatenew('spring_B', link_vec.express(Nb).subs(ic)*gamma)
# rigid_segment_b.set_vel(Nb, 0) # this point should be stationary in B, so if B rotates this point should look like its rotating in N

# Linear spring
spring_vec = rigid_segment_a.pos_from(rigid_segment_b)
length = sqrt(spring_vec.dot(spring_vec))
Fk = Ks * (length - length.subs(ic)) * spring_vec / length

# Hinge spring
link_vec_Nb = spring_vec.express(Nb)
link_vec_Nb_ic = link_vec_Nb.subs(ic)

link_vec_N = spring_vec.express(N)
link_vec_ic = link_vec_N.subs(ic)

e_current = link_vec_N / length
e_ref = link_vec_ic / sqrt(link_vec_ic.dot(link_vec_ic))
sin_phiA = e_ref.cross(e_current).dot(N.z)

cos_phiA = e_ref.dot(e_current)
phiA = atan2(sin_phiA, cos_phiA)

e_current_b = link_vec_Nb / length
# B-side: angle between link and body's local -x axis
e_ref_b = -Nb.x   # or whichever axis aligns with the link at rest
sin_phiB = e_ref_b.cross(e_current_b).dot(Nb.z)
cos_phiB = e_ref_b.dot(e_current_b)
phiB = atan2(sin_phiB, cos_phiB)

MkA = -Kth * 1000 * phiA * N.z
MkB = -Kth /100 * phiB * N.z
print(phiB)

link_unit = spring_vec.express(N) / length
# Perpendicular to link in 2D
perp = N.z.cross(link_unit)   # unit vector perpendicular to link

MkA_scalar = -Kth * phiA
F_from_MkA = (MkA_scalar / length) * perp
MkB_scalar = -Kth * phiB
F_from_MkB = (MkB_scalar / length) * perp   # same perp vector

kd = [
    x.diff() - u,
    y.diff() - v,
    theta.diff() - omega
]

KM = KanesMethod(N, [x, y, theta], [u, v, omega], kd)
BL = [B]

spring_vec = O.pos_from(Pb)
FL = [
    (rigid_segment_b, Fk), # Spring force
    (O, -Fk),
    # (Nb, MkA),  # reaction of ground-side torsional spring acts ON body  ← add this back
    # (Nb, -MkB),
    (Pb, -F_from_MkA),  # ← transverse restoring force from origin pin
    # (rigid_segment_b, -F_from_MkB),  # ← add this
    (O, F_from_MkB),  # ← reaction at ground pin
    (Pb, 9.81*m * -N.y), # Gravity
    (Pb, -0.5 * Pb.vel(N)) # Damping
]

(fr, frstar) = KM.kanes_equations(BL, FL)
MM = KM.mass_matrix_full
forcing = KM.forcing_full

states = Matrix([x, y, theta, u, v, omega])

rhs = MM.LUsolve(forcing)

# ============================================================
# Numerical Simulation
# ============================================================

L0_val    = 1.5         # 100 mm link
gamma_val = 0.45      # PRBM constant

# Thin steel strip: E=200GPa, b=10mm, h=0.5mm → EI = 200e9 * (0.01*0.0005³/12)
EI        = 200e9 * (0.01 * 0.0005**3 / 12)  # ≈ 2.08e-5 Nm²
Ks_val    = 3
Kth_val   = 0.05

m_val     = 0.01         # 10g body
Iz_val    = m_val * 0.05 ** 2 / 6  # ≈ 4.2e-6 kg·m² (small rectangular block)

params = (m_val, Iz_val, L0_val, Ks_val, Kth_val, gamma_val)

rhs = rhs.subs({
    m: m_val,
    L0: L0_val,
    Ks: Ks_val,
    Kth: Kth_val,
    Iz: Iz_val,
    gamma: gamma_val
})

rhs_func = lambdify([states], rhs, modules='numpy', cse=True)

A_vec = rigid_segment_a.pos_from(O).express(N).to_matrix(N)
B_vec = rigid_segment_b.pos_from(O).express(N).to_matrix(N)

geom_func = lambdify(
    (x, y, theta, L0, gamma),
    (A_vec, B_vec),
    modules='numpy'
)
from time import time

call_count = 0
t_start = time()
t_e = 20

def ode(t, state):
    global call_count
    call_count += 1
    if call_count % 100 == 0:
        print(f"t={t:.3f}/{t_e:.1f}  calls={call_count}  elapsed={time()-t_start:.1f}s", end='\r')
    return np.array(rhs_func(state), dtype=float).flatten()

x0 = np.array([
    L0_val,  # x at natural length
    0.0,     # y
    0.0,     # theta
    0.0,     # u
    0.0,     # v
    0.0,     # omega
])

sol = solve_ivp(
    ode,
    [0, t_e],
    x0,
    method="Radau",
    t_eval=np.linspace(0, t_e, 500),
    rtol=1e-4,
    atol=1e-4,
)

x_data = sol.y[0]
y_data = sol.y[1]
theta_data = sol.y[2]

# ============================================================
# Plotly Animation
# ============================================================

body_length = 0.7

frames = []

for k in range(len(sol.t)):
    xk = x_data[k]
    yk = y_data[k]
    th = theta_data[k]

    dx = body_length * np.cos(th)
    dy = body_length * np.sin(th)

    A_vec, B_vec = geom_func(xk, yk, th, L0_val, gamma_val)

    A_x, A_y, _ = np.array(A_vec, dtype=float).flatten()
    B_x, B_y, _ = np.array(B_vec, dtype=float).flatten()

    frame = go.Frame(
        data=[
            # Spring / link
            go.Scatter(
                x=[0, xk],
                y=[0, yk],
                mode='lines',
                line=dict(width=4),
                hoverinfo='skip'
            ),

            # Trajectory
            go.Scatter(
                x=x_data[:k + 1],
                y=y_data[:k + 1],
                mode='lines',
                line=dict(width=2),
                opacity=0.35,
                hoverinfo='skip'
            ),

            # Body center
            go.Scatter(
                x=[xk],
                y=[yk],
                mode='markers',
                marker=dict(size=18),
                hoverinfo='skip'
            ),

            # Orientation vector
            go.Scatter(
                x=[xk, xk + dx],
                y=[yk, yk + dy],
                mode='lines',
                line=dict(width=6),
                hoverinfo='skip'
            ),

            # -----------------------------
            # NEW: linkage points
            # -----------------------------

            go.Scatter(
                x=[A_x],
                y=[A_y],
                mode='markers',
                marker=dict(size=10),
                hoverinfo='skip'
            ),

            go.Scatter(
                x=[B_x],
                y=[B_y],
                mode='markers',
                marker=dict(size=10),
                hoverinfo='skip'
            ),
        ],
        name=str(k)
    )

    frames.append(frame)

fig = go.Figure(
    data=frames[0].data,
    frames=frames,
)

fig.update_layout(
    template='plotly_dark',
    title='Spring–Rigid-Body Dynamics',
    width=900,
    height=900,
    showlegend=False,

    xaxis=dict(
        range=[-5, 5],
        zeroline=False,
        scaleanchor='y',
    ),

    yaxis=dict(
        range=[-5, 5],
        zeroline=False,
    ),

    updatemenus=[
        dict(
            type='buttons',
            showactive=False,
            buttons=[
                dict(
                    label='▶ Play',
                    method='animate',
                    args=[
                        None,
                        {
                            'frame': {'duration': 20, 'redraw': True},
                            'fromcurrent': True,
                        }
                    ]
                )
            ]
        )
    ]
)

# Fixed origin marker
fig.add_trace(
    go.Scatter(
        x=[0],
        y=[0],
        mode='markers',
        marker=dict(size=12, symbol='x'),
        hoverinfo='skip'
    )
)

fig.show()
