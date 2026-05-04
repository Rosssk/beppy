import jax
import jax.numpy as jnp
from jax import jacobian, jit
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)  # Use float64 for physical accuracy

gamma = 0.85
kappa_theta = 2.65
cm = 1e-2
mm = 1e-3


# ---------------------------------------------------------------------------
# Pure JAX math primitives — all differentiable
# ---------------------------------------------------------------------------

def RX(t):
    return jnp.array([[1, 0, 0], [0, jnp.cos(t), -jnp.sin(t)], [0, jnp.sin(t), jnp.cos(t)]])

def RY(t):
    return jnp.array([[jnp.cos(t), 0, jnp.sin(t)], [0, 1, 0], [-jnp.sin(t), 0, jnp.cos(t)]])

def RZ(t):
    return jnp.array([[jnp.cos(t), -jnp.sin(t), 0], [jnp.sin(t), jnp.cos(t), 0], [0, 0, 1]])

def rotmat(angles):
    return RX(angles[0]) @ RY(angles[1]) @ RZ(angles[2])

def safe_norm(x, eps=1e-30):
    """
    Euclidean norm with a small epsilon under the square root so that the
    gradient never produces 0/0 at x == 0.
    """
    return jnp.sqrt(jnp.dot(x, x) + eps)


def angle(a, b):
    """
    Angle between vectors a and b using atan2.

    Uses safe_norm so the backward pass is well-defined even when a == b
    (rest position), returning a gradient of zero rather than NaN.
    """
    cos_val = jnp.dot(a, b) / (safe_norm(a) * safe_norm(b))
    cross_norm = safe_norm(jnp.cross(a, b))
    return jnp.arctan2(cross_norm, cos_val)


# ---------------------------------------------------------------------------
# Pure functional energy — no class state, fully JAX-traceable
# ---------------------------------------------------------------------------

def flexure_energy(
    xA, thetaA, xB, thetaB,
    attachpoint_localA, attachpoint_localB,
    spring_unitA0, spring_unitB0,
    springlen0,
    A, E, I,
):
    """
    Potential energy of a single PRBM flexure.

    Parameters
    ----------
    xA, xB         : (3,) position of body A and B
    thetaA, thetaB : (3,) Euler angles of body A and B
    attachpoint_localA/B : (3,) flexure attachment point in body-local frame
    spring_unitA0/B0     : (3,) reference spring unit vectors (body-local)
    springlen0           : scalar, reference spring length
    A, E, I              : cross-section area, Young's modulus, 2nd moment of area
    """
    kappa = gamma * kappa_theta * E * I / springlen0
    k = E * A / springlen0

    rotmat_A = rotmat(thetaA)
    rotmat_B = rotmat(thetaB)

    attachpoint_globalA = rotmat_A @ attachpoint_localA + xA
    attachpoint_globalB = rotmat_B @ attachpoint_localB + xB
    flexure_global = attachpoint_globalB - attachpoint_globalA
    length = jnp.linalg.norm(flexure_global)

    # Axial stretch/compression energy
    u = length * gamma - springlen0
    energy_axial = k * u ** 2 / 2

    # Bending energy at each end
    spring_unit = flexure_global / length
    tA = angle(rotmat_A.T @ spring_unit, spring_unitA0)
    tB = angle(rotmat_B.T @ spring_unit, spring_unitB0)
    energy_bend = kappa * (tA ** 2 + tB ** 2) / 2

    return energy_axial + energy_bend


def body_energy(pos, angles, forces):
    """
    Potential energy of a body due to external forces.

    Parameters
    ----------
    pos    : (3,) body position
    angles : (3,) body Euler angles
    forces : list of (attachpoint_local, vector, _) tuples
    """
    R = rotmat(angles)
    energy = 0.0
    for attachpoint_local, vector, _ in forces:
        attachpoint_global = R @ jnp.array(attachpoint_local) + pos
        energy -= jnp.dot(attachpoint_global, jnp.array(vector))
    return energy


# ---------------------------------------------------------------------------
# Jacobian of the total energy w.r.t. the free-body state vector
# ---------------------------------------------------------------------------

def make_energy_and_jacobian(flexure_params, body_forces, free_body_indices, fixed_body_states, n_free):
    """
    Build a total energy function and its Jacobian over the optimisation state
    vector x of shape (n_free * 6,), where each body occupies a [pos, angles]
    slice of 6 elements.

    Parameters
    ----------
    flexure_params : list of dicts, each with keys:
        'iA', 'iB'    : int or None — index into free_body_indices (None = fixed)
        'stateA', 'stateB' : (pos, angles) for fixed bodies (ignored if free)
        'localA', 'localB' : attachment points in local frame
        'unitA0', 'unitB0' : reference unit vectors in body-local frame
        'springlen0'        : scalar
        'A', 'E', 'I'
    body_forces : list of (free_body_index, forces_list) — forces on free bodies
    free_body_indices : list of int labels (just used for ordering)
    fixed_body_states : unused — each flexure carries its own fixed state
    n_free : int, number of free bodies

    Returns
    -------
    total_energy : callable x -> scalar
    jac          : callable x -> (n_free*6,) Jacobian
    """
    def total_energy(x):
        e = 0.0

        for fp in flexure_params:
            # Resolve body A
            if fp['iA'] is None:
                xA, tA = fp['stateA']
            else:
                i = fp['iA']
                xA, tA = x[i*6:i*6+3], x[i*6+3:i*6+6]

            # Resolve body B
            if fp['iB'] is None:
                xB, tB = fp['stateB']
            else:
                i = fp['iB']
                xB, tB = x[i*6:i*6+3], x[i*6+3:i*6+6]

            e += flexure_energy(
                xA, tA, xB, tB,
                fp['localA'], fp['localB'],
                fp['unitA0'], fp['unitB0'],
                fp['springlen0'],
                fp['A'], fp['E'], fp['I'],
            )

        for body_idx, forces in body_forces:
            i = body_idx
            e += body_energy(x[i*6:i*6+3], x[i*6+3:i*6+6], forces)

        return e

    jac = jit(jacobian(total_energy))
    total_energy = jit(total_energy)

    return total_energy, jac


# ---------------------------------------------------------------------------
# Helper to build flexure_params from the PRBM-style setup dicts
# ---------------------------------------------------------------------------

def build_flexure_params(flexure_defs, body_states, free_body_names, A, E, I):
    """
    Convert a list of flexure definitions into the flat parameter dicts
    consumed by make_energy_and_jacobian.

    Parameters
    ----------
    flexure_defs : list of dicts with keys:
        'nameA', 'nameB'           : body names
        'attachpoint_localA/B'     : local attachment points (list/array)
    body_states : dict {name: (position0, angles0)}
    free_body_names : list of names that are free during optimisation
    A, E, I : material/section properties
    """
    params = []
    for fd in flexure_defs:
        nameA = fd['nameA']
        nameB = fd['nameB']
        localA = jnp.array(fd['attachpoint_localA'], dtype=float)
        localB = jnp.array(fd['attachpoint_localB'], dtype=float)

        posA0, angA0 = body_states[nameA]
        posB0, angB0 = body_states[nameB]
        posA0 = jnp.array(posA0, dtype=float)
        posB0 = jnp.array(posB0, dtype=float)
        angA0 = jnp.array(angA0, dtype=float)
        angB0 = jnp.array(angB0, dtype=float)

        RA0 = rotmat(angA0)
        RB0 = rotmat(angB0)
        attA0 = RA0 @ localA + posA0
        attB0 = RB0 @ localB + posB0
        flex0 = attB0 - attA0
        length0 = jnp.linalg.norm(flex0)
        unit0 = flex0 / length0

        params.append({
            'iA': free_body_names.index(nameA) if nameA in free_body_names else None,
            'iB': free_body_names.index(nameB) if nameB in free_body_names else None,
            'stateA': (posA0, angA0),
            'stateB': (posB0, angB0),
            'localA': localA,
            'localB': localB,
            'unitA0': RA0.T @ unit0,
            'unitB0': RB0.T @ unit0,
            'springlen0': float(length0 * gamma),
            'A': A, 'E': E, 'I': I,
        })
    return params


# ---------------------------------------------------------------------------
# Convenience solver — drop-in for PRBM.solve_pose
# ---------------------------------------------------------------------------

def solve_pose_jax(flexure_defs, body_states, body_forces_raw, free_body_names, A, E, I, x0=None):
    """
    Minimise total energy over free bodies using JAX Jacobian for gradient info.

    Parameters
    ----------
    flexure_defs    : see build_flexure_params
    body_states     : dict {name: (position, angles)}
    body_forces_raw : dict {name: forces_list}  — forces on free bodies
    free_body_names : list of free body names (order defines the state vector)
    A, E, I         : material properties
    x0              : optional initial state vector of shape (len(free_body_names)*6,)

    Returns
    -------
    result : scipy OptimizeResult
    states : dict {name: (position, angles)} for each free body
    """
    n_free = len(free_body_names)

    fp = build_flexure_params(flexure_defs, body_states, free_body_names, A, E, I)

    body_forces = []
    for i, name in enumerate(free_body_names):
        if name in body_forces_raw:
            body_forces.append((i, body_forces_raw[name]))

    energy_fn, jac_fn = make_energy_and_jacobian(fp, body_forces, free_body_names, {}, n_free)

    if x0 is None:
        x0 = []
        for name in free_body_names:
            pos, ang = body_states[name]
            x0 += list(pos) + list(ang)
        x0 = np.array(x0, dtype=float)

    result = minimize(
        lambda x: float(energy_fn(jnp.array(x))),
        x0,
        jac=lambda x: np.array(jac_fn(jnp.array(x))),
        method='L-BFGS-B',
    )

    states = {}
    x = result.x
    for i, name in enumerate(free_body_names):
        states[name] = (x[i*6:i*6+3], x[i*6+3:i*6+6])

    return result, states


# ---------------------------------------------------------------------------
# Example — matches the original main()
# ---------------------------------------------------------------------------

def main():
    A_val = 1e-3
    E = 1650e6
    t = 1e-3
    I = jnp.pi * t ** 4 / 2
    h = 3e-2
    body_states = {
        'A': ([0., 0., 0.], [0., 0., 0.]),
        'B': ([0., 0., h/2], [0., 0., 0.]),
        'C': ([0., 0., h], [0., 0., 0.]),
    }

    flexure_defs = []
    n = 3
    r = 14e-3
    for i in range(n):
        flexure_defs += {'nameA': 'A', 'attachpoint_localA': [r*jnp.cos(i/n*2*jnp.pi), r*jnp.sin(i/n*2*jnp.pi), 0],
                         'nameB': 'B', 'attachpoint_localB': [r*jnp.cos((i+1)/n*2*jnp.pi), r*jnp.sin((i+1)/n*2*jnp.pi), 0]}
        flexure_defs += {'nameA': 'B', 'attachpoint_localA': [r*jnp.cos((i-.5)/n*2*jnp.pi), r*jnp.sin((i-.5)/n*2*jnp.pi), 0],
                         'nameB': 'C', 'attachpoint_localB': [r*jnp.cos((i+.5)/n*2*jnp.pi), r*jnp.sin((i+.5)/n*2*jnp.pi), 0]}



    # Solve for B, C free, A fixed
    result, states = solve_pose_jax(
        flexure_defs, body_states,
        body_forces_raw={'C': [((0, 0, 0), (0, 0, -4))]},
        free_body_names=['B', 'C'],
        A=A_val, E=E, I=I,
    )

    print("Optimisation success:", result.success)
    print("Final energy:", result.fun)
    print("Body B position:", states['B'][0])
    print("Body B angles:  ", states['B'][1])

    # --- Jacobian example ---
    fp = build_flexure_params(flexure_defs, body_states, ['B'], A_val, E, I)
    energy_fn, jac_fn = make_energy_and_jacobian(fp, [], ['B'], {}, 1)

    x = jnp.array(list(states['B'][0]) + list(states['B'][1]))
    print("\nJacobian at solution (should be ~0):")
    print(jac_fn(x))

    # Jacobian at a displaced, non-equilibrium state
    x_displaced = jnp.array([0.3, 0.2, 5.0, 0.1, 0.05, 0.0])
    print("\nJacobian at displaced state:")
    print(jac_fn(x_displaced))


if __name__ == "__main__":
    main()