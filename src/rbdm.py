from types import SimpleNamespace

import diffrax
import jax.numpy as jnp
import jax
from matplotlib import pyplot as plt

jax.config.update("jax_enable_x64", True)

def rot_mat(phi):
    theta_sq = jnp.dot(phi, phi)

    K = jnp.array([
        [0, -phi[2], phi[1]],
        [phi[2], 0, -phi[0]],
        [-phi[1], phi[0], 0]
    ])

    # Taylor: sin(θ)/θ = 1 - θ²/6 + θ⁴/120 - ...
    # (1-cos θ)/θ² = 1/2 - θ²/24 + θ⁴/720 - ...
    # Use where() to switch between exact and Taylor, avoiding sqrt entirely
    safe_theta_sq = jnp.where(theta_sq > 1e-6, theta_sq, jnp.ones_like(theta_sq))
    safe_theta = jnp.sqrt(safe_theta_sq)

    a_exact = jnp.sin(safe_theta) / safe_theta
    b_exact = (1 - jnp.cos(safe_theta)) / safe_theta_sq

    # Taylor series (polynomial, no singularity, safe for autodiff)
    a_taylor = 1 - theta_sq / 6 + theta_sq ** 2 / 120
    b_taylor = 0.5 - theta_sq / 24 + theta_sq ** 2 / 720

    a = jnp.where(theta_sq > 1e-6, a_exact, a_taylor)
    b = jnp.where(theta_sq > 1e-6, b_exact, b_taylor)

    return jnp.eye(3) + a * K + b * (K @ K)


def phi_to_omega(phi, phidot):
    theta_sq = jnp.dot(phi, phi)

    K = jnp.array([
        [0, -phi[2], phi[1]],
        [phi[2], 0, -phi[0]],
        [-phi[1], phi[0], 0]
    ])

    safe_theta_sq = jnp.where(theta_sq > 1e-6, theta_sq, jnp.ones_like(theta_sq))
    safe_theta = jnp.sqrt(safe_theta_sq)

    # (1-cos θ)/θ²
    a_exact = (1 - jnp.cos(safe_theta)) / safe_theta_sq
    # (θ - sin θ)/θ³
    b_exact = (safe_theta - jnp.sin(safe_theta)) / (safe_theta_sq * safe_theta)

    # Taylor: (1-cos θ)/θ² = 1/2 - θ²/24 + ...
    # (θ - sin θ)/θ³ = 1/6 - θ²/120 + ...
    a_taylor = 0.5 - theta_sq / 24 + theta_sq ** 2 / 720
    b_taylor = 1 / 6 - theta_sq / 120 + theta_sq ** 2 / 5040

    a = jnp.where(theta_sq > 1e-6, a_exact, a_taylor)
    b = jnp.where(theta_sq > 1e-6, b_exact, b_taylor)

    T = jnp.eye(3) - a * K + b * (K @ K)
    return T @ phidot

def angle_between(u, v):
    cross = jnp.cross(u, v)
    cross_mag = jnp.sqrt(jnp.dot(cross, cross) + 1e-20)  # safe gradient at zero
    dot = jnp.dot(u, v)
    return jnp.arctan2(cross_mag, dot)

def decomp_state(q, i):
    return q[i * 6 : i * 6 + 3], q[i * 6 + 3 : i * 6 + 6]

def linkage_energy(q, qdot, params, body_a_i, rel_a, body_b_i, rel_b):
    body_a_pos, body_a_rot = decomp_state(q, body_a_i)
    body_a_vel, body_a_rotvel = decomp_state(qdot, body_a_i)
    body_a_mat = rot_mat(body_a_rot)
    body_a_omega = phi_to_omega(body_a_rot, body_a_rotvel)
    body_b_pos, body_b_rot = decomp_state(q, body_b_i)
    body_b_vel, body_b_rotvel = decomp_state(qdot, body_b_i)
    body_b_mat = rot_mat(body_b_rot)
    body_b_omega = phi_to_omega(body_b_rot, body_b_rotvel)

    body_a_pos0 = params.ic[body_a_i]['pos']
    body_a_mat0 = rot_mat(params.ic[body_a_i]['rot'])
    body_b_pos0 = params.ic[body_b_i]['pos']
    body_b_mat0 = rot_mat(params.ic[body_b_i]['rot'])

    attach_a0 = body_a_pos0 + body_a_mat0 @ rel_a
    attach_b0 = body_b_pos0 + body_b_mat0 @ rel_b
    flexure_vec0 = attach_b0 - attach_a0
    flexure_vec_l0 = jnp.sqrt(jnp.dot(flexure_vec0, flexure_vec0))

    # Create the spring points. Points welded to the body with fixed relative position based on the initial configuration
    rig_a_local = body_a_mat.T @ (flexure_vec0 * (1-params.gamma)/2)
    rig_b_local = body_b_mat.T @ (-flexure_vec0 * (1-params.gamma)/2)
    spring_a = body_a_pos + body_a_mat @ (rel_a + rig_a_local)
    spring_b = body_b_pos + body_b_mat @ (rel_b + rig_b_local)

    spring_vec = spring_b - spring_a

    # ------------------
    # Potential energy
    # ------------------
    spring_len = jnp.sqrt(jnp.dot(spring_vec, spring_vec))
    spring_len0 = flexure_vec_l0 * params.gamma
    k_s = params.E*params.A/spring_len0
    U_s = 0.5 * k_s * (spring_len - spring_len0)**2

    theta1 = angle_between(body_a_mat @ rig_a_local, spring_vec)
    theta2 = angle_between(body_b_mat @ rig_b_local, -spring_vec)
    k_th = params.gamma*params.kappa_theta*params.E*params.I/spring_len0
    U_th1 = 0.5 * k_th * theta1**2
    U_th2 = 0.5 * k_th * theta2**2

    rig_a_mass_pos = body_a_pos + body_a_mat @ (params.mu * rig_a_local)
    rig_b_mass_pos = body_b_pos + body_b_mat @ (params.mu * rig_b_local)
    spring_mass_pos = spring_a + spring_vec * 0.5
    Uh_rig = (rig_a_mass_pos[2] + rig_b_mass_pos[2]) * params.m_flex * (1 - params.alpha)/2 * 9.81
    Uh_spr = spring_mass_pos[2] * params.alpha * params.m_flex * 9.81

    U = U_s + U_th1 + U_th2 + Uh_rig + Uh_spr


    # ------------------
    # Kinetic energy
    # ------------------
    rig_a_mass_vel = body_a_vel + jnp.cross(body_a_omega, body_a_mat @ (params.mu * rig_a_local))
    rig_b_mass_vel = body_b_vel + jnp.cross(body_b_omega, body_b_mat @ (params.mu * rig_b_local))
    spring_a_vel = body_a_vel + jnp.cross(body_a_omega, body_a_mat @ rig_a_local)
    spring_b_vel = body_b_vel + jnp.cross(body_b_omega, body_b_mat @ rig_b_local)
    spring_center_vel = spring_a_vel + (spring_b_vel - spring_a_vel) * 0.5

    Ek_rig = 0.5 * ((1-params.alpha)/2 * params.m_flex * (jnp.dot(rig_a_mass_vel, rig_a_mass_vel) + jnp.dot(rig_b_mass_vel, rig_b_mass_vel)))
    Ek_spr = 0.5 * params.alpha * params.m_flex * jnp.dot(spring_center_vel, spring_center_vel)
    T = Ek_spr + Ek_rig

    return T, U

def body_energy(q, qdot, params, body_i):
    body_pos, body_rot = decomp_state(q, body_i)
    body_vel, body_rotvel = decomp_state(qdot, body_i)
    body_mat = rot_mat(body_rot)
    body_omega = phi_to_omega(body_rot, body_rotvel)

    # Potential: gravitational
    U = 9.81 * params.m_body * body_pos[2]

    # Kinetic: translational
    Ek_trans = 0.5 * params.m_body * jnp.dot(body_vel, body_vel)

    # Kinetic: rotational — I_body is in body frame, omega in world frame
    omega_body = body_mat.T @ body_omega
    Ek_rot = 0.5 * jnp.dot(params.I_body * omega_body, omega_body)
    T = Ek_rot + Ek_trans

    return T, U

def energy(q, qdot, params):
    T = 0
    U = 0
    for i in range(3):
        t, u = body_energy(q, qdot, params, i)
        T += t; U += u


    for i in range(3):
        rel1 = jnp.array([params.r * jnp.cos(i / 3 * 2 * jnp.pi), params.r * jnp.sin(i / 3 * 2 * jnp.pi), 0])
        rel2 = jnp.array([params.r * jnp.cos((i + 1) / 3 * 2 * jnp.pi), params.r * jnp.sin((i + 1) / 3 * 2 * jnp.pi), 0])
        rel3 = jnp.array([params.r * jnp.cos((i + 2) / 3 * 2 * jnp.pi), params.r * jnp.sin((i + 2) / 3 * 2 * jnp.pi), 0])

        t1, u1 = linkage_energy(q, qdot, params, 0, rel1, 1, rel2)
        t2, u2 = linkage_energy(q, qdot, params, 1, rel2, 2, rel3)
        T += t1 + t2; U += u1 + u2


    return T, U




# Sweeping parameters
params = SimpleNamespace(
    kappa_theta = 2.65,
    h=10e-3,
    r=8.773827e-3,
    # k_s=4,
    # k_th=0.4,
    m_body=0.01,
    E = 850e4,
    I = 0.1*jnp.pi*(1e-3)**4/2,
    A = jnp.pi*(1e-3)**2,

    gamma=0.85,     # fraction of RPR link length that is the center segment
    mu=0.5,         # fraction along RPR rigid section length where the weight is
    alpha=0.6,      # fraction of RPR link weight in center segment
    m_flex=0.001,    # flexure mass

    ic=[],
)

params.I_body = jnp.array([
    params.m_body * (3 * params.r ** 2 + params.h ** 2) / 12,  # Ixx
    params.m_body * (3 * params.r ** 2 + params.h ** 2) / 12,  # Iyy
    params.m_body * params.r ** 2 / 2  # Izz
])

params.ic = [
    {'pos': jnp.array([0, 0, 0]),           'rot': jnp.array([0, 0, 0])},
    {'pos': jnp.array([0, 0, params.h/2]),  'rot': jnp.array([0, 0, 0])},
    {'pos': jnp.array([0, 0, params.h]),    'rot': jnp.array([0, 0, 0])},
]

# Create initial state vectors for Jax
q0 = jnp.array([
    val for ic in params.ic for key in ('pos', 'rot') for val in ic[key]
])
qdot0 = jnp.zeros(18)


# ── forward_sim.py ────────────────────────────────────────────────────────────
import functools
import diffrax

def energy_reduced(q, qdot, params):
    """q and qdot are 12-DOF (bodies 1 and 2 only); body 0 fixed at IC."""
    q_full    = jnp.concatenate([jnp.zeros(6), q])
    qdot_full = jnp.concatenate([jnp.zeros(6), qdot])
    return energy(q_full, qdot_full, params)

@jax.jit
def eom_reduced(q, qdot):
    def L(q_, qd_):
        T, U = energy_reduced(q_, qd_, params)
        return T - U

    M = jax.jacfwd(jax.grad(lambda qd: L(q, qd)))(qdot)
    dLdq = jax.grad(lambda q_: L(q_, qdot))(q)
    _, coriolis = jax.jvp(
        lambda q_: jax.grad(lambda qd: L(q_, qd))(qdot),
        (q,), (qdot,)
    )
    rhs = dLdq - coriolis
    return jnp.linalg.solve(M, rhs)

@functools.partial(jax.jit, static_argnames=("num_steps",))
def simulate(q0, qdot0, t0=0.0, t1=1.0, dt0=1e-4, num_steps=500):
    state0 = jnp.concatenate([q0, qdot0])  # 24-element

    def vector_field(t, state, args):
        q, qdot = state[:12], state[12:]
        qddot = eom_reduced(q, qdot)
        return jnp.concatenate([qdot, qddot])

    term     = diffrax.ODETerm(vector_field)
    solver   = diffrax.Dopri5()
    saveat   = diffrax.SaveAt(ts=jnp.linspace(t0, t1, num_steps))
    stepsize = diffrax.PIDController(rtol=1e-6, atol=1e-9)
    sol = diffrax.diffeqsolve(
        term, solver,
        t0=t0, t1=t1, dt0=dt0,
        y0=state0,
        saveat=saveat,
        # stepsize_controller=stepsize,
        max_steps=10_000_000,
    )
    return sol.ts, sol.ys[:, :12], sol.ys[:, 12:]

if __name__ == "__main__":
    # Bodies 1 and 2 only (drop first 6 DOFs)
    q0_red    = q0[6:]
    qdot0_red = qdot0.at[6 + 2].set(0.01)[6:]  # perturb body 1 vz

    print("Compiling + running simulation …")
    ts, qs, qdots = simulate(q0_red, qdot0_red, t0=0.0, t1=0.5, dt0=1e-5, num_steps=500)
    print(f"Done. Saved {ts.shape[0]} time steps.")

    # Reconstruct full state for energy eval
    qs_full    = jnp.concatenate([jnp.zeros((ts.shape[0], 6)), qs],    axis=1)
    qdots_full = jnp.concatenate([jnp.zeros((ts.shape[0], 6)), qdots], axis=1)

    energies = jax.vmap(lambda q, qd: energy(q, qd, params))(qs_full, qdots_full)
    T_hist, U_hist = energies
    H_hist = T_hist + U_hist
    print(f"Energy drift: {float(H_hist[-1] - H_hist[0]):.2e} J")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(ts, H_hist)
    ax1.set_xlabel("t [s]"); ax1.set_ylabel("E [J]"); ax1.set_title("Energy conservation")
    for i, label in enumerate(["body 1 z", "body 2 z"]):
        ax2.plot(ts, qs[:, i*6+2], label=label)
    ax2.set_xlabel("t [s]"); ax2.set_ylabel("z [m]"); ax2.legend(); ax2.set_title("Body z-positions")
    plt.tight_layout()
    plt.savefig("sim_result.png", dpi=150)
    plt.show()