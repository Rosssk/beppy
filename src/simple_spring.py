import numpy as np
from scipy.integrate import solve_ivp
from sympy import symbols, Matrix, lambdify
from sympy.physics.mechanics import RigidBody, Inertia, inertia, KanesMethod
from sympy.physics.vector import ReferenceFrame, Point, dynamicsymbols
import plotly.graph_objects as go

# -------------------------------------
# Symbolic derivation
# -------------------------------------
m, Iz, L0, Ks = symbols('m Iz L0 Ks')

N = ReferenceFrame('N')
O = Point('O')
O.set_vel(N, 0 * N.x)

x, y, theta = dynamicsymbols('x y theta')
u, v, omega = dynamicsymbols('u v omega')

Nb = N.orientnew('Nb', 'Axis', [theta, N.z])
Nb.set_ang_vel(N, omega * N.z)
Pb = O.locatenew('Pb', x * N.x + y * N.y)
Pb.set_vel(N, u * N.x + v * N.y)
I = inertia(Nb, 0, 0, Iz)
B = RigidBody('B', Pb, Nb, m, (I, Pb))

kd = [
    x.diff() - u,
    y.diff() - v,
    theta.diff() - omega
]

KM = KanesMethod(N, [x, y, theta], [u, v, omega], kd)
BL = [B]

spring_vec = O.pos_from(Pb)
FL = [
    (Pb, spring_vec.normalize()*(spring_vec.magnitude()-L0) * Ks), # Spring force
    (Pb, 9.81*m * -N.y), # Gravity
    (Pb, -0.5 * Pb.vel(N)) # Damping
]

(fr, frstar) = KM.kanes_equations(BL, FL)
MM = KM.mass_matrix_full
forcing = KM.forcing_full

states = Matrix([x, y, theta, u, v, omega])

rhs = MM.LUsolve(forcing)

rhs_func = lambdify(
    (states, m, Iz, L0, Ks),
    rhs,
    modules='numpy'
)


# ============================================================
# Numerical Simulation
# ============================================================

m_val = 1.0
Iz_val = 0.15
L0_val = 2.0
Ks_val = 12.0

params = (m_val, Iz_val, L0_val, Ks_val)


def ode(t, state):
    dx = rhs_func(state, *params)
    return np.array(dx, dtype=float).flatten()


x0 = np.array([
    3.2,
    0.4,
    0.0,
    0.0,
    1.5,
    3.0,
])

sol = solve_ivp(
    ode,
    [0, 20],
    x0,
    t_eval=np.linspace(0, 20, 500),
    rtol=1e-9,
    atol=1e-9,
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

    frame = go.Frame(
        data=[
            # Spring
            go.Scatter(
                x=[0, xk],
                y=[0, yk],
                mode='lines',
                line=dict(width=4),
                hoverinfo='skip'
            ),

            # Trajectory
            go.Scatter(
                x=x_data[:k+1],
                y=y_data[:k+1],
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
