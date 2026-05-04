import jax
import jax.numpy as jnp
from jax import grad, jacobian
from jax.scipy.optimize import minimize

gamma = 0.85
kappa_theta = 2.65


# ----------------------------
# Math utilities
# ----------------------------
def norm(v):
    return jnp.linalg.norm(v)


# def angle(a, b):
#     cosang = jnp.dot(a, b) / (norm(a) * norm(b))
#     cosang = jnp.clip(cosang, -1.0, 1.0)
#     return jnp.arccos(cosang)

def angle(a, b):
    cross = jnp.linalg.norm(jnp.cross(a, b))
    dotp = jnp.dot(a, b)
    return jnp.arctan2(cross, dotp)

# ----------------------------
# Rotations
# ----------------------------
def RX(t):
    return jnp.array([
        [1, 0, 0],
        [0, jnp.cos(t), -jnp.sin(t)],
        [0, jnp.sin(t), jnp.cos(t)]
    ])


def RY(t):
    return jnp.array([
        [jnp.cos(t), 0, jnp.sin(t)],
        [0, 1, 0],
        [-jnp.sin(t), 0, jnp.cos(t)]
    ])


def RZ(t):
    return jnp.array([
        [jnp.cos(t), -jnp.sin(t), 0],
        [jnp.sin(t), jnp.cos(t), 0],
        [0, 0, 1]
    ])


def rotmat(theta):
    return RX(theta[0]) @ RY(theta[1]) @ RZ(theta[2])


# ----------------------------
# System builder
# ----------------------------
def build_system(bodies, flexures, forces, A, E, I):
    return {
        "bodies": bodies,
        "flexures": flexures,
        "forces": forces,
        "A": A,
        "E": E,
        "I": I
    }


# ----------------------------
# Flexure precomputation
# ----------------------------
def init_flexure(bodyA, bodyB, attachA, attachB):
    xA, RA = bodyA
    xB, RB = bodyB

    pA = RA @ attachA + xA
    pB = RB @ attachB + xB

    d = pB - pA
    L0 = norm(d)

    unit = d / L0

    return {
        "attachA": attachA,
        "attachB": attachB,
        "springlen0": L0 * gamma,
        "spring_unitA0": RA.T @ unit,
        "spring_unitB0": RB.T @ unit
    }


# ----------------------------
# Flexure energy
# ----------------------------
def flexure_energy(xA, tA, xB, tB, f, A, E, I):
    kappa = gamma * kappa_theta * E * I / f["springlen0"]
    k = E * A / f["springlen0"]

    RA = rotmat(tA)
    RB = rotmat(tB)

    pA = RA @ f["attachA"] + xA
    pB = RB @ f["attachB"] + xB

    d = pB - pA
    eps = 1e-8
    L = jnp.linalg.norm(d) + eps

    # stretch
    u = L * gamma - f["springlen0"]
    u = jnp.clip(L * gamma - f["springlen0"], -1e3, 1e3)
    E_stretch = 0.5 * k * u**2

    # bending
    unit = d / L
    unitA = RA.T @ unit
    unitB = RB.T @ unit

    thetaA = angle(unitA, f["spring_unitA0"])
    thetaB = angle(unitB, f["spring_unitB0"])

    E_bend = 0.5 * kappa * (thetaA**2 + thetaB**2)

    return E_stretch + E_bend


# ----------------------------
# Total energy
# ----------------------------
def total_energy(x, system):
    energy = 0.0

    # flexures
    for f in system["flexures"]:
        iA = f["iA"]
        iB = f["iB"]

        xA = x[iA*6:iA*6+3]
        tA = x[iA*6+3:iA*6+6]

        xB = x[iB*6:iB*6+3]
        tB = x[iB*6+3:iB*6+6]

        energy += flexure_energy(
            xA, tA, xB, tB,
            f,
            system["A"], system["E"], system["I"]
        )

    # external forces
    for bf in system["forces"]:
        i = bf["i"]

        pos = x[i*6:i*6+3]
        ang = x[i*6+3:i*6+6]

        R = rotmat(ang)
        p = R @ bf["attach"] + pos

        energy -= jnp.dot(p, bf["force"])

    return energy


# ----------------------------
# Solver
# ----------------------------
def solve(system, x0):
    energy_grad = grad(total_energy)
    energy_hess = jacobian(energy_grad)
    import scipy
    result = scipy.optimize.minimize(
        total_energy,
        x0,
        args=(system,),
        jac=energy_grad,
        hess=energy_hess,
        method="trust-ncg",
        options={"maxiter": 1000}
    )

    return result

def show(system, x, dim=3):
    import numpy as np  # plotting prefers numpy

    def rotmat_np(theta):
        cx, cy, cz = np.cos(theta)
        sx, sy, sz = np.sin(theta)

        RX = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
        RY = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
        RZ = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
        return RX @ RY @ RZ

    bodies = []

    # reconstruct bodies
    n_bodies = len(x) // 6
    for i in range(n_bodies):
        pos = np.array(x[i*6:i*6+3])
        ang = np.array(x[i*6+3:i*6+6])
        R = rotmat_np(ang)
        bodies.append((pos, R))

    # flexure lines
    flex_lines = []
    for f in system["flexures"]:
        iA, iB = f["iA"], f["iB"]

        posA, RA = bodies[iA]
        posB, RB = bodies[iB]

        pA = RA @ np.array(f["attachA"]) + posA
        pB = RB @ np.array(f["attachB"]) + posB

        d = pB - pA

        sA = pA + d * (1 - gamma) / 2
        sB = pB - d * (1 - gamma) / 2

        line = np.vstack([pA, sA, sB, pB]).T
        flex_lines.append(line)

    # body lines (just origin for now)
    body_lines = []
    for pos, R in bodies:
        pts = np.array([pos, pos + R @ np.array([0,0,0])]).T
        body_lines.append(pts)

    # ------------------ plotting ------------------
    if dim == 2:
        import matplotlib.pyplot as plt

        for line in flex_lines:
            plt.plot(line[0], line[1], "o-")

        for line in body_lines:
            plt.plot(line[0], line[1], "-")

        plt.axis("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    else:
        import plotly.graph_objects as go

        fig = go.Figure()

        for line in flex_lines:
            fig.add_trace(go.Scatter3d(
                x=line[0], y=line[1], z=line[2],
                mode="lines+markers",
                name="flexure"
            ))

        for line in body_lines:
            fig.add_trace(go.Scatter3d(
                x=line[0], y=line[1], z=line[2],
                mode="lines",
                name="body"
            ))

        fig.update_layout(scene=dict(aspectmode="cube"))
        fig.show()


def main():
    mm = 1e-3
    cm = 1e-2

    h = 3 * cm
    r = 14 * mm
    n = 3

    # initial bodies (pos, rotation matrix)
    bodies = [
        (jnp.array([0, 0, 0]), jnp.eye(3)),        # A
        (jnp.array([0, 0, h/2]), jnp.eye(3)),      # B
        (jnp.array([0, 0, h]), jnp.eye(3)),        # C
    ]

    t = mm
    A = jnp.pi * t**2
    E = 1650e6
    I = jnp.pi * t**4 / 2

    flexures = []

    for i in range(n):
        # --- A → B ---
        a1 = jnp.array([
            r*jnp.cos(i/n*2*jnp.pi),
            r*jnp.sin(i/n*2*jnp.pi),
            0
        ])
        b1 = jnp.array([
            r*jnp.cos((i+1)/n*2*jnp.pi),
            r*jnp.sin((i+1)/n*2*jnp.pi),
            0
        ])

        f1 = init_flexure(bodies[0], bodies[1], a1, b1)
        f1["iA"], f1["iB"] = 0, 1
        flexures.append(f1)

        # --- C → B (THIS WAS MISSING) ---
        a2 = jnp.array([
            r*jnp.cos((i-0.5)/n*2*jnp.pi),
            r*jnp.sin((i-0.5)/n*2*jnp.pi),
            0
        ])
        b2 = jnp.array([
            r*jnp.cos((i+0.5)/n*2*jnp.pi),
            r*jnp.sin((i+0.5)/n*2*jnp.pi),
            0
        ])

        f2 = init_flexure(bodies[2], bodies[1], a2, b2)
        f2["iA"], f2["iB"] = 2, 1
        flexures.append(f2)

    forces = [{
        "i": 2,
        "force": jnp.array([0, 0, -4]),
        "attach": jnp.zeros(3)
    }]

    system = build_system(bodies, flexures, forces, A, E, I)

    # initial guess
    x0 = jnp.concatenate([
        jnp.array([0,0,0, 0,0,0]),
        jnp.array([0,0,h/2, 0,0,0]),
        jnp.array([0,0,h, 0,0,0])
    ])

    result = solve(system, x0)
    show(system, result.x, dim=3)
    
    print(result)


if __name__ == "__main__":
    main()