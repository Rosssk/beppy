from types import SimpleNamespace
import functools
import time

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

jax.config.update("jax_enable_x64", True)


# ─────────────────────────────────────────────────────────────────────────────
# Maths helpers
# ─────────────────────────────────────────────────────────────────────────────

def skew(v):
    return jnp.array([[ 0,    -v[2],  v[1]],
                      [ v[2],  0,    -v[0]],
                      [-v[1],  v[0],  0   ]])


def rot_mat(phi):
    theta_sq = jnp.dot(phi, phi)
    K = skew(phi)
    safe_tsq = jnp.where(theta_sq > 1e-6, theta_sq, jnp.ones_like(theta_sq))
    safe_t   = jnp.sqrt(safe_tsq)
    a = jnp.where(theta_sq > 1e-6, jnp.sin(safe_t)/safe_t,          1 - theta_sq/6  + theta_sq**2/120)
    b = jnp.where(theta_sq > 1e-6, (1-jnp.cos(safe_t))/safe_tsq,    0.5 - theta_sq/24 + theta_sq**2/720)
    return jnp.eye(3) + a*K + b*(K@K)


def T_mat(phi):
    """Maps phi_dot → body angular velocity omega.  omega = T(phi) @ phi_dot"""
    theta_sq = jnp.dot(phi, phi)
    K = skew(phi)
    safe_tsq = jnp.where(theta_sq > 1e-6, theta_sq, jnp.ones_like(theta_sq))
    safe_t   = jnp.sqrt(safe_tsq)
    a = jnp.where(theta_sq > 1e-6, (1-jnp.cos(safe_t))/safe_tsq,              0.5  - theta_sq/24  + theta_sq**2/720)
    b = jnp.where(theta_sq > 1e-6, (safe_t-jnp.sin(safe_t))/(safe_tsq*safe_t), 1/6  - theta_sq/120 + theta_sq**2/5040)
    return jnp.eye(3) - a*K + b*(K@K)


def angle_between(u, v):
    cross = jnp.cross(u, v)
    return jnp.arctan2(jnp.sqrt(jnp.dot(cross, cross) + 1e-20), jnp.dot(u, v))


def decomp_state(q, i):
    return q[i*6:i*6+3], q[i*6+3:i*6+6]


# ─────────────────────────────────────────────────────────────────────────────
# Energies
# ─────────────────────────────────────────────────────────────────────────────

def linkage_energy(q, qdot, params, body_a_i, rel_a, body_b_i, rel_b):
    body_a_pos, body_a_rot = decomp_state(q, body_a_i)
    body_a_vel, body_a_rotvel = decomp_state(qdot, body_a_i)
    body_a_mat = rot_mat(body_a_rot)
    body_a_omega = T_mat(body_a_rot) @ body_a_rotvel
    body_b_pos, body_b_rot = decomp_state(q, body_b_i)
    body_b_vel, body_b_rotvel = decomp_state(qdot, body_b_i)
    body_b_mat = rot_mat(body_b_rot)
    body_b_omega = T_mat(body_b_rot) @ body_b_rotvel

    pa0  = params.ic[body_a_i]['pos']
    Ra0  = rot_mat(params.ic[body_a_i]['rot'])
    pb0  = params.ic[body_b_i]['pos']
    Rb0  = rot_mat(params.ic[body_b_i]['rot'])

    flex_vec0  = (pb0 + Rb0 @ rel_b) - (pa0 + Ra0 @ rel_a)
    flex_len0  = jnp.sqrt(jnp.dot(flex_vec0, flex_vec0))

    rig_a_local = body_a_mat.T @ (flex_vec0 * (1 - params.gamma) / 2)
    rig_b_local = body_b_mat.T @ (-flex_vec0 * (1 - params.gamma) / 2)
    spring_a = body_a_pos + body_a_mat @ (rel_a + rig_a_local)
    spring_b = body_b_pos + body_b_mat @ (rel_b + rig_b_local)
    spring_vec = spring_b - spring_a

    # Potential
    spring_len  = jnp.sqrt(jnp.dot(spring_vec, spring_vec))
    spring_len0 = flex_len0 * params.gamma
    k_s  = params.E * params.A / spring_len0
    U_s  = 0.5 * k_s * (spring_len - spring_len0)**2

    theta1 = angle_between(body_a_mat @ rig_a_local,  spring_vec)
    theta2 = angle_between(body_b_mat @ rig_b_local, -spring_vec)
    k_th   = params.gamma * params.kappa_theta * params.E * params.I / spring_len0
    U_th   = 0.5 * k_th * (theta1**2 + theta2**2)

    rig_a_mpos = body_a_pos + body_a_mat @ (params.mu * rig_a_local)
    rig_b_mpos = body_b_pos + body_b_mat @ (params.mu * rig_b_local)
    spr_mpos   = spring_a + spring_vec * 0.5
    Uh = ((rig_a_mpos[2] + rig_b_mpos[2]) * params.m_flex * (1-params.alpha)/2
          + spr_mpos[2] * params.alpha * params.m_flex) * 9.81
    U = U_s + U_th + Uh

    # Kinetic
    rig_a_vel = body_a_vel + jnp.cross(body_a_omega, body_a_mat @ (params.mu * rig_a_local))
    rig_b_vel = body_b_vel + jnp.cross(body_b_omega, body_b_mat @ (params.mu * rig_b_local))
    spr_a_vel = body_a_vel + jnp.cross(body_a_omega, body_a_mat @ rig_a_local)
    spr_b_vel = body_b_vel + jnp.cross(body_b_omega, body_b_mat @ rig_b_local)
    spr_c_vel = (spr_a_vel + spr_b_vel) * 0.5

    Ek_rig = 0.5 * (1-params.alpha)/2 * params.m_flex * (
        jnp.dot(rig_a_vel, rig_a_vel) + jnp.dot(rig_b_vel, rig_b_vel))
    Ek_spr = 0.5 * params.alpha * params.m_flex * jnp.dot(spr_c_vel, spr_c_vel)
    T = Ek_rig + Ek_spr

    return T, U


def body_energy(q, qdot, params, body_i):
    pos, rot     = decomp_state(q, body_i)
    vel, rotvel  = decomp_state(qdot, body_i)
    R            = rot_mat(rot)
    omega        = T_mat(rot) @ rotvel
    omega_body   = R.T @ omega

    U      = 9.81 * params.m_body * pos[2]
    Ek_t   = 0.5 * params.m_body * jnp.dot(vel, vel)
    Ek_r   = 0.5 * jnp.dot(params.I_body * omega_body, omega_body)
    return Ek_t + Ek_r, U


def _link_rels(i, r):
    def rel(k):
        a = (i + k) / 3 * 2 * jnp.pi
        return jnp.array([r * jnp.cos(a), r * jnp.sin(a), 0.0])
    return rel(0), rel(1), rel(2)


def energy(q, qdot, params):
    T = U = 0.0
    for i in range(3):
        t, u = body_energy(q, qdot, params, i)
        T += t; U += u
    for i in range(3):
        ra, rb, rc = _link_rels(i, params.r)
        t1, u1 = linkage_energy(q, qdot, params, 0, ra, 1, rb)
        t2, u2 = linkage_energy(q, qdot, params, 1, rb, 2, rc)
        T += t1 + t2; U += u1 + u2
    return T, U


# ─────────────────────────────────────────────────────────────────────────────
# Reduced system (body 0 fixed)
# ─────────────────────────────────────────────────────────────────────────────

def energy_reduced(q, qdot, params):
    return energy(jnp.concatenate([jnp.zeros(6), q]),
                  jnp.concatenate([jnp.zeros(6), qdot]), params)


# ─────────────────────────────────────────────────────────────────────────────
# Fast mass matrix
# ─────────────────────────────────────────────────────────────────────────────

def potential(q_red, params):
    q = jnp.concatenate([jnp.zeros(6), q_red])
    U = 0.0

    # gravitational — bodies 1 and 2 only (body 0 fixed at z=0)
    for b in range(1, 3):
        pos, _ = decomp_state(q, b)
        U += 9.81 * params.m_body * pos[2]

    # flexure linkages
    r = params.r
    for i in range(3):
        def rel(k, i=i):
            return jnp.array([r*jnp.cos((i+k)/3*2*jnp.pi),
                               r*jnp.sin((i+k)/3*2*jnp.pi), 0.0])

        for ai, rel_a, bi, rel_b in [(0, rel(0), 1, rel(1)),
                                      (1, rel(1), 2, rel(2))]:
            pos_a, rot_a = decomp_state(q, ai)
            pos_b, rot_b = decomp_state(q, bi)
            Ra, Rb = rot_mat(rot_a), rot_mat(rot_b)

            pa0 = params.ic[ai]['pos'];  Ra0 = rot_mat(params.ic[ai]['rot'])
            pb0 = params.ic[bi]['pos'];  Rb0 = rot_mat(params.ic[bi]['rot'])
            fv0 = (pb0 + Rb0 @ rel_b) - (pa0 + Ra0 @ rel_a)
            fl0 = jnp.sqrt(jnp.dot(fv0, fv0))

            rig_a_local = Ra.T @ ( fv0 * (1 - params.gamma) / 2)
            rig_b_local = Rb.T @ (-fv0 * (1 - params.gamma) / 2)
            spr_a = pos_a + Ra @ (rel_a + rig_a_local)
            spr_b = pos_b + Rb @ (rel_b + rig_b_local)
            sv    = spr_b - spr_a

            slen0 = fl0 * params.gamma
            U += 0.5 * (params.E * params.A / slen0) * (jnp.sqrt(jnp.dot(sv, sv)) - slen0)**2

            k_th = params.gamma * params.kappa_theta * params.E * params.I / slen0
            U += 0.5 * k_th * (angle_between(Ra @ rig_a_local,  sv)**2 +
                                angle_between(Rb @ rig_b_local, -sv)**2)

            # gravitational PE of flexure masses
            rig_a_mpos = pos_a + Ra @ (params.mu * rig_a_local)
            rig_b_mpos = pos_b + Rb @ (params.mu * rig_b_local)
            spr_mpos   = spr_a + sv * 0.5
            U += ((rig_a_mpos[2] + rig_b_mpos[2]) * params.m_flex * (1 - params.alpha) / 2
                  + spr_mpos[2] * params.alpha * params.m_flex) * 9.81

    return U

def skew(v):
    return jnp.array([[ 0,    -v[2],  v[1]],
                      [ v[2],  0,    -v[0]],
                      [-v[1],  v[0],  0   ]])

def _point_mass_contrib(m, b_idx, r_world, Tm):
    """m * J^T J where J is the 3x12 velocity Jacobian for a point at r_world on body b_idx."""
    J = jnp.zeros((3, 12))
    J = J.at[:, b_idx*6  :b_idx*6+3].set(jnp.eye(3))
    J = J.at[:, b_idx*6+3:b_idx*6+6].set(-skew(r_world) @ Tm)
    return m * J.T @ J

def mass_matrix(q, params):
    """
    Fully analytical 12x12 mass matrix. Zero autodiff.
    Reduced state layout: [v1(3) | phidot1(3) | v2(3) | phidot2(3)]
    Body 0 fixed -> no columns. Body 1 -> b_idx=0, Body 2 -> b_idx=1.
    """
    M = jnp.zeros((12, 12))

    # ── rigid body blocks ────────────────────────────────────────────────────
    for b in range(2):
        _, rot = decomp_state(jnp.concatenate([jnp.zeros(6), q]), b + 1)
        R  = rot_mat(rot)
        Tm = T_mat(rot)
        idx = b * 6
        # translational
        M = M.at[idx:idx+3, idx:idx+3].add(params.m_body * jnp.eye(3))
        # rotational: T^T @ R @ diag(I_body) @ R^T @ T
        M = M.at[idx+3:idx+6, idx+3:idx+6].add(
            Tm.T @ (R @ jnp.diag(params.I_body) @ R.T) @ Tm)

    # ── flexure (RPR) blocks ─────────────────────────────────────────────────
    # Each linkage has 3 lumped masses:
    #   rig_a : m_rig  attached to body_a  at R_a @ (mu * rig_a_local)
    #   rig_b : m_rig  attached to body_b  at R_b @ (mu * rig_b_local)
    #   spring centre: velocity = 0.5*(v_spr_a + v_spr_b)
    #     -> expands to: diag m_spr/2 on each endpoint + cross term J_a^T J_b

    def add_linkage(M, ai, rel_a, bi, rel_b):
        # ai, bi are FULL body indices (0,1,2); bidx is reduced index (body0 fixed -> skip)
        q_full = jnp.concatenate([jnp.zeros(6), q])
        _, rot_a = decomp_state(q_full, ai);  Ra = rot_mat(rot_a);  Tma = T_mat(rot_a)
        _, rot_b = decomp_state(q_full, bi);  Rb = rot_mat(rot_b);  Tmb = T_mat(rot_b)

        # initial-config flex vector to get rig_*_local
        pa0 = params.ic[ai]['pos'];  Ra0 = rot_mat(params.ic[ai]['rot'])
        pb0 = params.ic[bi]['pos'];  Rb0 = rot_mat(params.ic[bi]['rot'])
        fv0 = (pb0 + Rb0 @ rel_b) - (pa0 + Ra0 @ rel_a)

        rig_a_local = Ra.T @ ( fv0 * (1 - params.gamma) / 2)
        rig_b_local = Rb.T @ (-fv0 * (1 - params.gamma) / 2)

        m_rig = params.m_flex * (1 - params.alpha) / 2
        m_spr = params.m_flex * params.alpha

        # world-frame attachment vectors
        rw_rig_a = Ra @ (params.mu * rig_a_local)
        rw_rig_b = Rb @ (params.mu * rig_b_local)
        rw_spr_a = Ra @ (rel_a + rig_a_local)
        rw_spr_b = Rb @ (rel_b + rig_b_local)

        bidx_a = ai - 1   # body 0 -> -1 (fixed, skip), body 1 -> 0, body 2 -> 1
        bidx_b = bi - 1

        if bidx_a >= 0:
            M = M + _point_mass_contrib(m_rig,   bidx_a, rw_rig_a, Tma)
            M = M + _point_mass_contrib(m_spr/2, bidx_a, rw_spr_a, Tma)
        if bidx_b >= 0:
            M = M + _point_mass_contrib(m_rig,   bidx_b, rw_rig_b, Tmb)
            M = M + _point_mass_contrib(m_spr/2, bidx_b, rw_spr_b, Tmb)

        # cross term: 0.5 * m_spr * (J_a^T J_b + J_b^T J_a)
        if bidx_a >= 0 and bidx_b >= 0:
            Ja = jnp.zeros((3, 12))
            Ja = Ja.at[:, bidx_a*6  :bidx_a*6+3].set(jnp.eye(3))
            Ja = Ja.at[:, bidx_a*6+3:bidx_a*6+6].set(-skew(rw_spr_a) @ Tma)
            Jb = jnp.zeros((3, 12))
            Jb = Jb.at[:, bidx_b*6  :bidx_b*6+3].set(jnp.eye(3))
            Jb = Jb.at[:, bidx_b*6+3:bidx_b*6+6].set(-skew(rw_spr_b) @ Tmb)
            M = M + (m_spr/2) * (Ja.T @ Jb + Jb.T @ Ja)

        return M

    r = params.r
    for i in range(3):
        def rel(k, i=i):
            return jnp.array([r*jnp.cos((i+k)/3*2*jnp.pi),
                               r*jnp.sin((i+k)/3*2*jnp.pi), 0.0])
        M = add_linkage(M, 0, rel(0), 1, rel(1))
        M = add_linkage(M, 1, rel(1), 2, rel(2))

    return M


# ─────────────────────────────────────────────────────────────────────────────
# Equations of motion
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def eom_reduced(q, qdot):
    M = mass_matrix(q, params)
    dUdq = jax.grad(potential)(q, params)

    _, dMdq_v = jax.jvp(lambda q_: mass_matrix(q_, params), (q,), (qdot,))

    coriolis = dMdq_v @ qdot - 0.5 * (dMdq_v.T @ qdot)

    return jnp.linalg.solve(M, -dUdq - coriolis)


# ─────────────────────────────────────────────────────────────────────────────
# Integrator
# ─────────────────────────────────────────────────────────────────────────────

@functools.partial(jax.jit, static_argnames=("num_steps",))
def simulate(q0, qdot0, t0=0.0, t1=1.0, dt0=1e-3, num_steps=500):
    state0 = jnp.concatenate([q0, qdot0])

    def vector_field(t, state, args):
        q, qdot = state[:12], state[12:]
        return jnp.concatenate([qdot, eom_reduced(q, qdot)])

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Tsit5(),
        t0=t0, t1=t1, dt0=dt0,
        y0=state0,
        saveat=diffrax.SaveAt(ts=jnp.linspace(t0, t1, num_steps)),
        max_steps=200_000,
    )
    return sol.ts, sol.ys[:, :12], sol.ys[:, 12:]


# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────

params = SimpleNamespace(
    kappa_theta=2.65,
    h=10e-3,
    r=8.773827e-3,
    m_body=0.01,
    E=850e6,
    I=0.1 * jnp.pi * (1e-3)**4 / 2,
    A=jnp.pi * (1e-3)**2,
    gamma=0.85,
    mu=0.5,
    alpha=0.6,
    m_flex=0.001,
    ic=[],
)
params.I_body = jnp.array([
    params.m_body * (3*params.r**2 + params.h**2) / 12,
    params.m_body * (3*params.r**2 + params.h**2) / 12,
    params.m_body * params.r**2 / 2,
])
params.ic = [
    {'pos': jnp.array([0.0, 0.0, 0.0]),          'rot': jnp.array([0.0, 0.0, 0.0])},
    {'pos': jnp.array([0.0, 0.0, params.h/2]),   'rot': jnp.array([0.0, 0.0, 0.0])},
    {'pos': jnp.array([0.0, 0.0, params.h]),     'rot': jnp.array([0.0, 0.0, 0.0])},
]

q0 = jnp.array([val for ic in params.ic for key in ('pos','rot') for val in ic[key]])
qdot0 = jnp.zeros(18)


# ─────────────────────────────────────────────────────────────────────────────
# MeshCat visualisation
# ─────────────────────────────────────────────────────────────────────────────

def np_rot_mat(phi):
    phi = np.asarray(phi, dtype=float)
    theta = np.linalg.norm(phi)
    if theta < 1e-9:
        return np.eye(3)
    k = phi / theta
    K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)


def homogeneous(R, p):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = p
    return T


def cylinder_between(vis, name, p0, p1, radius, color, opacity=0.9):
    p0, p1 = np.asarray(p0), np.asarray(p1)
    delta  = p1 - p0
    length = np.linalg.norm(delta)
    if length < 1e-10:
        return
    mid = (p0 + p1) / 2
    y   = np.array([0., 1., 0.])
    d   = delta / length
    cr  = np.cross(y, d)
    cn  = np.linalg.norm(cr)
    if cn < 1e-9:
        R = np.eye(3) if np.dot(y, d) > 0 else np.diag([1., -1., 1.])
    else:
        ax = cr / cn
        an = np.arctan2(cn, np.dot(y, d))
        K  = np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
        R  = np.eye(3) + np.sin(an)*K + (1-np.cos(an))*(K@K)
    vis[name].set_object(
        g.Cylinder(length, radius),
        g.MeshLambertMaterial(color=color, opacity=opacity),
    )
    vis[name].set_transform(homogeneous(R, mid))


BODY_COLORS  = [0x4fc3f7, 0xff8a65, 0xa5d6a7]
FLEX_COLOR   = 0xffd54f
RIGID_COLOR  = 0xb0bec5
BODY_H_VIZ   = float(params.h) * 0.55
BODY_R_VIZ   = float(params.r)
FLEX_RAD     = 0.0004
RIGID_RAD    = 0.0007


def build_scene(vis, params):
    # ground plane
    vis["grid"].set_object(
        g.Box([0.10, 0.001, 0.10]),
        g.MeshLambertMaterial(color=0x1a2030, opacity=0.5),
    )
    vis["grid"].set_transform(tf.translation_matrix([0, -0.001, 0]))

    # bodies — MeshCat cylinders are Y-aligned; rotate 90° around X to get Z-aligned
    R_body = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=float)
    for b in range(3):
        vis[f"body/{b}"].set_object(
            g.Cylinder(BODY_H_VIZ, BODY_R_VIZ),
            g.MeshLambertMaterial(color=BODY_COLORS[b], opacity=0.92),
        )

    # flexure placeholders (overwritten every frame)
    for i in range(3):
        for pair in ("ab", "bc"):
            for seg, col in (("rigid_a", RIGID_COLOR), ("flex", FLEX_COLOR), ("rigid_b", RIGID_COLOR)):
                vis[f"link/{i}/{pair}/{seg}"].set_object(
                    g.Cylinder(1e-9, FLEX_RAD),
                    g.MeshLambertMaterial(color=col, opacity=0.9),
                )

    # world axes
    for ax, col, vec in [("x",0xff4444,[1,0,0]),("y",0x44ff44,[0,1,0]),("z",0x4488ff,[0,0,1])]:
        cylinder_between(vis, f"axes/{ax}", [0,0,0], np.array(vec)*0.015, 0.0003, col)


def update_scene(vis, q_full, params):
    q_full = np.asarray(q_full)
    bpos, bmat = [], []
    for b in range(3):
        p = q_full[b*6:b*6+3]
        R = np_rot_mat(q_full[b*6+3:b*6+6])
        bpos.append(p); bmat.append(R)
        R_viz = R @ np.array([[1,0,0],[0,0,-1],[0,1,0]])
        vis[f"body/{b}"].set_transform(homogeneous(R_viz, p))

    def draw_linkage(prefix, ai, rel_a, bi, rel_b):
        Ra, pa = bmat[ai], bpos[ai]
        Rb, pb = bmat[bi], bpos[bi]
        pa0 = np.array(params.ic[ai]['pos'])
        Ra0 = np_rot_mat(np.array(params.ic[ai]['rot']))
        pb0 = np.array(params.ic[bi]['pos'])
        Rb0 = np_rot_mat(np.array(params.ic[bi]['rot']))
        fv0 = (pb0 + Rb0 @ rel_b) - (pa0 + Ra0 @ rel_a)
        g_ = params.gamma
        ral = Ra.T @ (fv0 * (1-g_)/2)
        rbl = Rb.T @ (-fv0 * (1-g_)/2)
        att_a  = pa + Ra @ rel_a
        spr_a  = pa + Ra @ (rel_a + ral)
        att_b  = pb + Rb @ rel_b
        spr_b  = pb + Rb @ (rel_b + rbl)
        cylinder_between(vis, f"{prefix}/rigid_a", att_a, spr_a, RIGID_RAD, RIGID_COLOR)
        cylinder_between(vis, f"{prefix}/flex",    spr_a, spr_b, FLEX_RAD,  FLEX_COLOR)
        cylinder_between(vis, f"{prefix}/rigid_b", spr_b, att_b, RIGID_RAD, RIGID_COLOR)

    r = float(params.r)
    for i in range(3):
        rels = [np.array([r*np.cos((i+k)/3*2*np.pi), r*np.sin((i+k)/3*2*np.pi), 0.]) for k in range(3)]
        draw_linkage(f"link/{i}/ab", 0, rels[0], 1, rels[1])
        draw_linkage(f"link/{i}/bc", 1, rels[1], 2, rels[2])


def visualize(ts, qs_full, params, speedup=0.3):
    vis = meshcat.Visualizer()
    vis.open()
    print(f"MeshCat URL: {vis.url()}")
    build_scene(vis, params)
    time.sleep(0.6)
    print("Animating … (Ctrl-C to stop, loops continuously)")
    try:
        while True:
            t_start = time.time()
            for t, q in zip(ts, qs_full):
                update_scene(vis, q, params)
                elapsed  = time.time() - t_start
                target   = (float(t) - float(ts[0])) / speedup
                if target - elapsed > 0:
                    time.sleep(target - elapsed)
            print(f"  replaying … (t_end={float(ts[-1]):.3f} s)")
    except KeyboardInterrupt:
        print("Stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    q0_red    = q0[6:]
    qdot0_red = qdot0.at[6+2].set(0.01)[6:]   # perturb body 1 vz

    print("Compiling + running simulation …")
    t_run = time.time()
    ts, qs, qdots = simulate(q0_red, qdot0_red, t0=0.0, t1=10, dt0=1e-4, num_steps=5000)
    ts.block_until_ready()
    print(f"Done in {time.time()-t_run:.2f}s  ({ts.shape[0]} steps)")

    # after simulate has run once (so everything is compiled)
    q_test = q0_red
    qd_test = qdot0_red

    # warmup
    _ = eom_reduced(q_test, qd_test).block_until_ready()

    import time

    t = time.time()
    for _ in range(500):
        _ = eom_reduced(q_test, qd_test).block_until_ready()
    print(f"500 eom calls: {time.time() - t:.2f}s")

    print("Second call (pure runtime) …")
    t0_wall = time.time()
    ts, qs, qdots = simulate(q0_red, qdot0_red, t0=0.0, t1=0.5, dt0=1e-4, num_steps=5000)
    ts.block_until_ready()
    print(f"  {time.time() - t0_wall:.2f}s")


    qs_full    = jnp.concatenate([jnp.zeros((ts.shape[0], 6)), qs],    axis=1)
    qdots_full = jnp.concatenate([jnp.zeros((ts.shape[0], 6)), qdots], axis=1)

    T_hist, U_hist = jax.vmap(lambda q, qd: energy(q, qd, params))(qs_full, qdots_full)
    H_hist = T_hist + U_hist
    print(f"Energy drift: {float(H_hist[-1]-H_hist[0]):.2e} J")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(ts, H_hist)
    ax1.set_xlabel("t [s]"); ax1.set_ylabel("E [J]"); ax1.set_title("Energy conservation")
    for i, label in enumerate(["body 1 z", "body 2 z"]):
        ax2.plot(ts, qs[:, i*6+2], label=label)
    ax2.set_xlabel("t [s]"); ax2.set_ylabel("z [m]"); ax2.legend()
    ax2.set_title("Body z-positions")
    plt.tight_layout()
    plt.savefig("sim_result.png", dpi=150)
    plt.show()

    visualize(ts, np.array(qs_full), params, speedup=0.3)