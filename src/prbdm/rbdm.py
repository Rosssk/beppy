"""
flexure_sim.py
==============
Lagrangian dynamics simulation of a three-body flexure-linkage mechanism.

Architecture overview
---------------------
Three rigid discs (bodies 0, 1, 2) are stacked along the Z-axis and
connected in pairs by six identical RPR (Rigid–Pseudo-rigid–Rigid)
flexure linkages arranged symmetrically at 120° intervals around each
disc's circumference.

Body 0 is held fixed.  Bodies 1 and 2 are free, giving 12 generalised
coordinates (6 per body: [x, y, z, φx, φy, φz] using the Rodrigues /
rotation-vector parameterisation).

Simulation pipeline
-------------------
1.  Define the Lagrangian  L = T − U  via ``energy`` /
    ``energy_reduced``.
2.  Build the mass matrix analytically with ``mass_matrix`` (no
    auto-diff for M, which keeps compilation fast).
3.  Evaluate the equations of motion with ``eom_reduced``, which uses
    JAX auto-diff only for the potential gradient and for the Christoffel
    (Coriolis/centrifugal) term.
4.  Integrate with Diffrax (Tsit5 adaptive solver) via ``simulate``.
5.  Visualise in MeshCat (3-D browser renderer) via ``visualize``.

Coordinate conventions
-----------------------
- All positions in metres, angles in radians, energies in joules.
- Gravity acts in the −Z direction (g = 9.81 m s⁻²).
- State vector layout for N bodies:
      [pos₀(3), rot₀(3), pos₁(3), rot₁(3), …]   shape (6N,)
  where ``rotᵢ`` is the Rodrigues rotation vector φ with ‖φ‖ = θ.

Dependencies
------------
- JAX / JAX numpy  (numerical computation + JIT + auto-diff)
- jaxtyping        (array shape annotations + runtime shape verification)
- beartype         (type checker backend used by jaxtyped, on non-JIT functions only)
- Diffrax          (ODE solver built on JAX)
- MeshCat          (3-D browser-based visualisation)
- Matplotlib       (energy / position plots)
"""

from __future__ import annotations

from _types import SimpleNamespace
import functools
import time
from typing import Tuple

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype
from matplotlib import pyplot as plt
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

# Use 64-bit floats throughout to keep energy drift small.
# jax.config.update("jax_enable_x64", True)

# ── JIT compilation strategy ─────────────────────────────────────────────────
# ``simulate`` and ``eom_reduced`` are each decorated with ``@jax.jit``.
#
# Modern JAX (jaxlib ≥ 0.4) handles nested JIT calls without Python re-entry:
# the outer JIT sees inner JIT functions as opaque but fully compiled XLA
# subgraphs.  This is preferable to inlining everything into one large graph
# because XLA can make suboptimal fusion/scheduling decisions on very large
# programs — two well-scoped kernels often run faster than one giant one.
#
# ``mass_matrix`` and ``potential`` are left bare (no ``@jax.jit``) so they
# inline into ``eom_reduced``'s compiled graph rather than adding further
# unnecessary boundaries.
#
# ── Runtime type-checking strategy ───────────────────────────────────────────
# @jaxtyped(typechecker=beartype) is applied ONLY to pure-Python math helpers
# (skew, rot_mat, T_mat, …).  These are called at the Python level with real
# arrays, so shape contracts are meaningful and overhead is negligible.
#
# Functions in the JIT-compiled graph (eom_reduced, mass_matrix, potential) are
# left without @jaxtyped:
#   1. During tracing, arguments are JAX tracers, not real arrays — shape checks
#      would see abstract values and either pass vacuously or error.
#   2. Any Python-level wrapper adds overhead that defeats JIT compilation.
#
# Type annotations on these functions are documentation / static-analysis only.

@jaxtyped(typechecker=beartype)
def decomp_state(q: Float[Array, "n"], i: int) -> Tuple[Vec3, Vec3]:
    """
    Extract the position and rotation vector of body *i* from a state vector.

    State layout::

        [pos_0(3), rot_0(3), pos_1(3), rot_1(3), …]

    Parameters
    ----------
    q : Float[Array, "n"]
        Concatenated state of N bodies (n = 6N).
    i : int
        Body index (0-based).

    Returns
    -------
    pos : Vec3
        World-frame position of body *i*.
    rot : Vec3
        Rodrigues rotation vector of body *i*.
    """
    return q[i*6:i*6+3], q[i*6+3:i*6+6]


# ─────────────────────────────────────────────────────────────────────────────
# Energies
# ─────────────────────────────────────────────────────────────────────────────

@jaxtyped(typechecker=beartype)
def linkage_energy(
    q:        FullState,
    qdot:     FullState,
    params:   SimpleNamespace,
    body_a_i: int,
    rel_a:    Vec3,
    body_b_i: int,
    rel_b:    Vec3,
) -> Tuple[Scalar, Scalar]:
    """
    Kinetic and potential energy of one RPR flexure linkage.

    Each linkage connects an attachment point on body A (body-local offset
    *rel_a*) to one on body B (*rel_b*).  The linkage is split into three
    segments:

    - **Rigid arm A** — fraction ``(1−γ)/2`` of the reference length,
      rigidly attached to body A.
    - **Flexible spring** — fraction ``γ``, connecting the tips of the
      two rigid arms.
    - **Rigid arm B** — fraction ``(1−γ)/2``, rigidly attached to body B.

    Potential energy contributions
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``U_s``
        Axial spring (Hooke): ``½ k_s (L − L₀)²``,
        with ``k_s = EA / L₀_spring``.
    ``U_th``
        Bending: ``½ k_θ (θ₁² + θ₂²)``, where θ₁ / θ₂ are the angles
        between each rigid arm and the spring axis.
    ``Uh``
        Gravitational PE of the three lumped flexure masses.

    Kinetic energy
    ~~~~~~~~~~~~~~
    Rigid-arm midpoint velocities and spring-midpoint velocity computed
    via the rigid-body velocity formula  ``v_P = v_cm + ω × r_P``.

    Parameters
    ----------
    q, qdot : FullState   shape (18,)
        Full generalised coordinates and velocities.
    params : SimpleNamespace
        Physical parameters (see module-level ``params``).
    body_a_i, body_b_i : int
        Full body indices (0, 1, or 2) for the two endpoint bodies.
    rel_a, rel_b : Vec3
        Attachment offsets expressed in each body's local frame [m].

    Returns
    -------
    T : Scalar   kinetic energy [J]
    U : Scalar   potential energy [J]
    """
    body_a_pos, body_a_rot = decomp_state(q, body_a_i)
    body_a_vel, body_a_rotvel = decomp_state(qdot, body_a_i)
    body_a_mat = rot_mat(body_a_rot)
    body_a_omega = T_mat(body_a_rot) @ body_a_rotvel
    body_b_pos, body_b_rot = decomp_state(q, body_b_i)
    body_b_vel, body_b_rotvel = decomp_state(qdot, body_b_i)
    body_b_mat = rot_mat(body_b_rot)
    body_b_omega = T_mat(body_b_rot) @ body_b_rotvel

    # ── Reference configuration ───────────────────────────────────────────────
    pa0 = params.ic[body_a_i]['pos']
    Ra0 = rot_mat(params.ic[body_a_i]['rot'])
    pb0 = params.ic[body_b_i]['pos']
    Rb0 = rot_mat(params.ic[body_b_i]['rot'])

    flex_vec0 = (pb0 + Rb0 @ rel_b) - (pa0 + Ra0 @ rel_a)   # undeformed flex vector
    flex_len0 = jnp.sqrt(jnp.dot(flex_vec0, flex_vec0))

    # Body-local rigid-arm vectors (constant direction in body frame by design)
    rig_a_local = body_a_mat.T @ (flex_vec0 * (1 - params.gamma) / 2)
    rig_b_local = body_b_mat.T @ (-flex_vec0 * (1 - params.gamma) / 2)

    # World-frame spring endpoints (tips of the rigid arms)
    spring_a = body_a_pos + body_a_mat @ (rel_a + rig_a_local)
    spring_b = body_b_pos + body_b_mat @ (rel_b + rig_b_local)
    spring_vec = spring_b - spring_a

    # ── Potential energy ──────────────────────────────────────────────────────
    spring_len  = jnp.sqrt(jnp.dot(spring_vec, spring_vec))
    spring_len0 = flex_len0 * params.gamma
    k_s  = params.E * params.A / spring_len0
    U_s  = 0.5 * k_s * (spring_len - spring_len0)**2

    theta1 = angle_between(body_a_mat @ rig_a_local,  spring_vec)
    theta2 = angle_between(body_b_mat @ rig_b_local, -spring_vec)
    k_th   = params.gamma * params.kappa_theta * params.E * params.I / spring_len0
    U_th   = 0.5 * k_th * (theta1**2 + theta2**2)

    # Lumped mass positions for gravitational PE
    rig_a_mpos = body_a_pos + body_a_mat @ (params.mu * rig_a_local)
    rig_b_mpos = body_b_pos + body_b_mat @ (params.mu * rig_b_local)
    spr_mpos   = spring_a + spring_vec * 0.5
    Uh = ((rig_a_mpos[2] + rig_b_mpos[2]) * params.m_flex * (1 - params.alpha) / 2
          + spr_mpos[2] * params.alpha * params.m_flex) * 9.81
    U = U_s + U_th + Uh

    # ── Kinetic energy ────────────────────────────────────────────────────────
    rig_a_vel = body_a_vel + jnp.cross(body_a_omega, body_a_mat @ (params.mu * rig_a_local))
    rig_b_vel = body_b_vel + jnp.cross(body_b_omega, body_b_mat @ (params.mu * rig_b_local))
    spr_a_vel = body_a_vel + jnp.cross(body_a_omega, body_a_mat @ rig_a_local)
    spr_b_vel = body_b_vel + jnp.cross(body_b_omega, body_b_mat @ rig_b_local)
    spr_c_vel = (spr_a_vel + spr_b_vel) * 0.5   # spring midpoint velocity

    Ek_rig = 0.5 * (1 - params.alpha) / 2 * params.m_flex * (
        jnp.dot(rig_a_vel, rig_a_vel) + jnp.dot(rig_b_vel, rig_b_vel))
    Ek_spr = 0.5 * params.alpha * params.m_flex * jnp.dot(spr_c_vel, spr_c_vel)
    T = Ek_rig + Ek_spr

    return T, U


@jaxtyped(typechecker=beartype)
def body_energy(
    q:      FullState,
    qdot:   FullState,
    params: SimpleNamespace,
    body_i: int,
) -> Tuple[Scalar, Scalar]:
    """
    Kinetic and gravitational potential energy of one rigid disc.

    The disc is modelled as a solid cylinder with principal moments of
    inertia ``params.I_body``.  Rotational kinetic energy is evaluated in
    the body frame::

        T_rot = ½ ωᵦ · (I_body ⊙ ωᵦ)   (element-wise product)

    where ``ωᵦ = Rᵀ ω`` is the body-frame angular velocity.

    Parameters
    ----------
    q, qdot : FullState   shape (18,)
    params  : SimpleNamespace
    body_i  : int   body index (0–2)

    Returns
    -------
    T : Scalar   kinetic energy  [J]
    U : Scalar   gravitational potential energy [J]
    """
    pos, rot    = decomp_state(q, body_i)
    vel, rotvel = decomp_state(qdot, body_i)
    R           = rot_mat(rot)
    omega       = T_mat(rot) @ rotvel       # world-frame angular velocity
    omega_body  = R.T @ omega               # body-frame angular velocity

    U    = 9.81 * params.m_body * pos[2]
    Ek_t = 0.5 * params.m_body * jnp.dot(vel, vel)
    Ek_r = 0.5 * jnp.dot(params.I_body * omega_body, omega_body)
    return Ek_t + Ek_r, U


def _link_rels(i: int, r: float) -> Tuple[Vec3, Vec3, Vec3]:
    """
    Return the three attachment-point offsets for linkage set *i*.

    Points are evenly spaced on a circle of radius *r* in the XY-plane at
    angular positions  ``(i+k) / 3 · 2π``  for k = 0, 1, 2.

    Parameters
    ----------
    i : int     linkage set index (0–2)
    r : float   attachment circle radius [m]

    Returns
    -------
    rel0, rel1, rel2 : Vec3
        Attachment offsets in the body-local frame [m].
    """
    def rel(k: int) -> Vec3:
        a = (i + k) / 3 * 2 * jnp.pi
        return jnp.array([r * jnp.cos(a), r * jnp.sin(a), 0.0])
    return rel(0), rel(1), rel(2)


def energy(
    q:      FullState,
    qdot:   FullState,
    params: SimpleNamespace,
) -> Tuple[Scalar, Scalar]:
    """
    Total Lagrangian energy (T, U) of the full 3-body system.

    Sums contributions from:

    - 3 rigid disc bodies (``body_energy``).
    - 3 linkage sets × 2 links each = 6 RPR flexure linkages
      (``linkage_energy``).

    Parameters
    ----------
    q     : FullState   shape (18,)  generalised coordinates
    qdot  : FullState   shape (18,)  generalised velocities
    params: SimpleNamespace

    Returns
    -------
    T : Scalar   total kinetic energy  [J]
    U : Scalar   total potential energy [J]
    """
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

def energy_reduced(
    q:      RedState,
    qdot:   RedState,
    params: SimpleNamespace,
) -> Tuple[Scalar, Scalar]:
    """
    Total energy with body 0 clamped to the origin.

    Prepends six zeros to the reduced 12-DOF state before delegating to
    ``energy``, so body 0 is always treated as fixed.

    Parameters
    ----------
    q    : RedState   shape (12,)  DOF of bodies 1 and 2
    qdot : RedState   shape (12,)  velocities of bodies 1 and 2
    params : SimpleNamespace

    Returns
    -------
    T, U : Scalar
    """
    return energy(jnp.concatenate([jnp.zeros(6), q]),
                  jnp.concatenate([jnp.zeros(6), qdot]), params)


# ─────────────────────────────────────────────────────────────────────────────
# Potential (for auto-diff) and analytical mass matrix
# ─────────────────────────────────────────────────────────────────────────────

def potential(q_red: RedState, params: SimpleNamespace) -> Scalar:
    """
    Total potential energy as a function of the reduced 12-DOF state only.

    This function has no ``qdot`` dependence, making it cheap to
    auto-differentiate for the generalised force vector ``−∂U/∂q``.

    Parameters
    ----------
    q_red  : RedState   shape (12,)
    params : SimpleNamespace

    Returns
    -------
    Scalar   total potential energy [J]
    """
    q = jnp.concatenate([jnp.zeros(6), q_red])
    U = 0.0

    # Gravitational PE — bodies 1 and 2 only (body 0 fixed at z = 0)
    for b in range(1, 3):
        pos, _ = decomp_state(q, b)
        U += 9.81 * params.m_body * pos[2]

    # Flexure PE — mirrors linkage_energy but without kinetic terms
    r = params.r
    for i in range(3):
        def rel(k: int, i: int = i) -> Vec3:
            return jnp.array([r * jnp.cos((i+k)/3*2*jnp.pi),
                               r * jnp.sin((i+k)/3*2*jnp.pi), 0.0])

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

            # Axial spring
            U += 0.5 * (params.E * params.A / slen0) * (jnp.sqrt(jnp.dot(sv, sv)) - slen0)**2

            # Bending
            k_th = params.gamma * params.kappa_theta * params.E * params.I / slen0
            U += 0.5 * k_th * (angle_between(Ra @ rig_a_local,  sv)**2 +
                                angle_between(Rb @ rig_b_local, -sv)**2)

            # Gravitational PE of flexure lumped masses
            rig_a_mpos = pos_a + Ra @ (params.mu * rig_a_local)
            rig_b_mpos = pos_b + Rb @ (params.mu * rig_b_local)
            spr_mpos   = spr_a + sv * 0.5
            U += ((rig_a_mpos[2] + rig_b_mpos[2]) * params.m_flex * (1 - params.alpha) / 2
                  + spr_mpos[2] * params.alpha * params.m_flex) * 9.81

    return U


@jaxtyped(typechecker=beartype)
def _point_mass_contrib(
    m:       float | Scalar,
    b_idx:   int,
    r_world: Vec3,
    Tm:      Mat33,
) -> MassMat:
    """
    Mass-matrix contribution ``m · Jᵀ J`` from a single point mass.

    The velocity Jacobian for a point at world-frame lever arm *r_world*
    relative to body *b_idx*'s centre of mass is::

        J = [… I₃ … | … −skew(r_world) · T_m … ]
                ↑                    ↑
            cols b_idx*6        cols b_idx*6+3

    (all other 3-column blocks are zero because body 0 is fixed.)

    Parameters
    ----------
    m       : scalar mass [kg]
    b_idx   : int     reduced body index (0 → body 1, 1 → body 2)
    r_world : Vec3    world-frame lever arm from body CM to mass point [m]
    Tm      : Mat33   ``T_mat`` evaluated at the body's rotation vector

    Returns
    -------
    MassMat   shape (12, 12)
    """
    J = jnp.zeros((3, 12))
    J = J.at[:, b_idx*6  :b_idx*6+3].set(jnp.eye(3))
    J = J.at[:, b_idx*6+3:b_idx*6+6].set(-skew(r_world) @ Tm)
    return m * J.T @ J


def mass_matrix(q: RedState, params: SimpleNamespace) -> MassMat:
    """
    Analytical 12×12 mass matrix (zero auto-differentiation).

    Reduced generalised-coordinate layout::

        [v₁(3) | φ̇₁(3) | v₂(3) | φ̇₂(3)]

    Body 0 is fixed → no columns.  Body 1 → column block 0 (rows/cols 0–5),
    Body 2 → column block 1 (rows/cols 6–11).

    Construction
    ~~~~~~~~~~~~
    1. **Rigid-body diagonal blocks**

       - Translational: ``m_body · I₃``
       - Rotational: ``Tᵀ · R · diag(I_body) · Rᵀ · T``

    2. **Flexure lumped-mass blocks** (per RPR linkage)

       Three lumped masses per linkage:

       - ``m_rig`` at each rigid-arm midpoint → ``_point_mass_contrib`` on
         the owning body.
       - ``m_spr/2`` at each spring endpoint → diagonal ``Jᵀ J`` terms.
       - ``m_spr/2`` cross terms between the two spring endpoints, because
         the midpoint velocity is the average::

             v_mid = ½(v_spr_a + v_spr_b)
             → ΔM_cross = ½ m_spr (Jaᵀ Jb + Jbᵀ Ja)

    Parameters
    ----------
    q      : RedState   shape (12,)  reduced generalised coordinates
    params : SimpleNamespace

    Returns
    -------
    MassMat   shape (12, 12)  symmetric positive-definite mass matrix
    """
    M = jnp.zeros((12, 12))

    # ── Rigid body contributions ──────────────────────────────────────────────
    for b in range(2):
        _, rot = decomp_state(jnp.concatenate([jnp.zeros(6), q]), b + 1)
        R  = rot_mat(rot)
        Tm = T_mat(rot)
        idx = b * 6
        M = M.at[idx:idx+3, idx:idx+3].add(params.m_body * jnp.eye(3))
        M = M.at[idx+3:idx+6, idx+3:idx+6].add(
            Tm.T @ (R @ jnp.diag(params.I_body) @ R.T) @ Tm)

    # ── Flexure (RPR) contributions ───────────────────────────────────────────

    def add_linkage(
        M:     MassMat,
        ai:    int,
        rel_a: Vec3,
        bi:    int,
        rel_b: Vec3,
    ) -> MassMat:
        """
        Accumulate mass-matrix contributions from one RPR linkage.

        Parameters
        ----------
        M            : MassMat   current mass matrix to update
        ai, bi       : int       full body indices (0, 1, 2)
        rel_a, rel_b : Vec3      body-local attachment offsets [m]

        Returns
        -------
        MassMat   updated mass matrix
        """
        q_full = jnp.concatenate([jnp.zeros(6), q])
        _, rot_a = decomp_state(q_full, ai);  Ra = rot_mat(rot_a);  Tma = T_mat(rot_a)
        _, rot_b = decomp_state(q_full, bi);  Rb = rot_mat(rot_b);  Tmb = T_mat(rot_b)

        pa0 = params.ic[ai]['pos'];  Ra0 = rot_mat(params.ic[ai]['rot'])
        pb0 = params.ic[bi]['pos'];  Rb0 = rot_mat(params.ic[bi]['rot'])
        fv0 = (pb0 + Rb0 @ rel_b) - (pa0 + Ra0 @ rel_a)

        rig_a_local = Ra.T @ ( fv0 * (1 - params.gamma) / 2)
        rig_b_local = Rb.T @ (-fv0 * (1 - params.gamma) / 2)

        m_rig = params.m_flex * (1 - params.alpha) / 2   # mass of each rigid-arm lump [kg]
        m_spr = params.m_flex * params.alpha              # mass of spring lump [kg]

        # World-frame lever arms from body CM to each mass attachment point
        rw_rig_a: Vec3 = Ra @ (params.mu * rig_a_local)
        rw_rig_b: Vec3 = Rb @ (params.mu * rig_b_local)
        rw_spr_a: Vec3 = Ra @ (rel_a + rig_a_local)
        rw_spr_b: Vec3 = Rb @ (rel_b + rig_b_local)

        # bidx = -1 means body 0 (fixed) → skip
        bidx_a = ai - 1
        bidx_b = bi - 1

        if bidx_a >= 0:
            M = M + _point_mass_contrib(m_rig,   bidx_a, rw_rig_a, Tma)
            M = M + _point_mass_contrib(m_spr/2, bidx_a, rw_spr_a, Tma)
        if bidx_b >= 0:
            M = M + _point_mass_contrib(m_rig,   bidx_b, rw_rig_b, Tmb)
            M = M + _point_mass_contrib(m_spr/2, bidx_b, rw_spr_b, Tmb)

        # Spring-midpoint cross term  ½ m_spr (Jaᵀ Jb + Jbᵀ Ja)
        if bidx_a >= 0 and bidx_b >= 0:
            Ja: Float[Array, "3 12"] = jnp.zeros((3, 12))
            Ja = Ja.at[:, bidx_a*6  :bidx_a*6+3].set(jnp.eye(3))
            Ja = Ja.at[:, bidx_a*6+3:bidx_a*6+6].set(-skew(rw_spr_a) @ Tma)
            Jb: Float[Array, "3 12"] = jnp.zeros((3, 12))
            Jb = Jb.at[:, bidx_b*6  :bidx_b*6+3].set(jnp.eye(3))
            Jb = Jb.at[:, bidx_b*6+3:bidx_b*6+6].set(-skew(rw_spr_b) @ Tmb)
            M = M + (m_spr/2) * (Ja.T @ Jb + Jb.T @ Ja)

        return M

    r = params.r
    for i in range(3):
        def rel(k: int, i: int = i) -> Vec3:
            return jnp.array([r*jnp.cos((i+k)/3*2*jnp.pi),
                               r*jnp.sin((i+k)/3*2*jnp.pi), 0.0])
        M = add_linkage(M, 0, rel(0), 1, rel(1))
        M = add_linkage(M, 1, rel(1), 2, rel(2))

    return M


# ─────────────────────────────────────────────────────────────────────────────
# Equations of motion
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def eom_reduced(q: RedState, qdot: RedState) -> RedState:
    """
    Generalised accelerations q̈ from the reduced Lagrangian EOM.

    Decorated with ``@jax.jit`` as a separate compiled unit.  Modern JAX
    (via ``jaxlib`` ≥ 0.4) handles nested JIT calls without Python re-entry
    — the outer ``simulate`` JIT sees this as an opaque but fully compiled
    XLA subgraph, which gives the XLA optimizer a smaller, better-defined
    unit to work with.  Inlining everything into one giant graph can
    paradoxically hurt runtime performance due to suboptimal XLA
    fusion/scheduling decisions on large programs.

    Solves the Euler–Lagrange equations in matrix form::

        M(q) q̈ = −∂U/∂q − C(q, q̇) q̇

    where the Coriolis/centrifugal term is::

        C(q, q̇) q̇ = (∂M/∂q · q̇) q̇ − ½ (∂M/∂q · q̇)ᵀ q̇

    Auto-diff is used only for:

    - ``jax.grad(potential)`` — one scalar backward pass (O(n) cost).
    - ``jax.jvp(mass_matrix)`` — one forward-mode pass for dM/dq · q̇.

    Parameters
    ----------
    q    : RedState   shape (12,)  reduced generalised coordinates
    qdot : RedState   shape (12,)  reduced generalised velocities

    Returns
    -------
    RedState   shape (12,)  generalised accelerations q̈
    """
    M    = mass_matrix(q, params)
    dUdq = jax.grad(potential)(q, params)

    # Directional derivative of M along qdot (forward-mode, cheap)
    _, dMdq_v = jax.jvp(lambda q_: mass_matrix(q_, params), (q,), (qdot,))

    coriolis = dMdq_v @ qdot - 0.5 * (dMdq_v.T @ qdot)

    return jnp.linalg.solve(M, -dUdq - coriolis)


# ─────────────────────────────────────────────────────────────────────────────
# Integrator
# ─────────────────────────────────────────────────────────────────────────────

@functools.partial(jax.jit, static_argnames=("num_steps",))
def simulate(
    q0:        RedState,
    qdot0:     RedState,
    t0:        float | int = 0.0,
    t1:        float | int = 1.0,
    dt0:       float | int = 1e-3,
    num_steps: int         = 500,
) -> Tuple[
    Float[Array, "num_steps"],
    Float[Array, "num_steps 12"],
    Float[Array, "num_steps 12"],
]:
    """
    Integrate the reduced equations of motion over ``[t0, t1]``.

    Uses Diffrax's ``Tsit5`` (Tsitouras 5th-order Runge–Kutta) adaptive
    solver.  The ODE state is the concatenation ``[q(12); qdot(12)]``.

    Parameters
    ----------
    q0, qdot0  : RedState   shape (12,)
        Initial generalised coordinates and velocities.
    t0, t1     : float | int
        Integration interval [s].  Integer literals (e.g. ``t1=10``) are
        accepted and promoted to float by JAX automatically.
    dt0        : float | int
        Suggested initial step size for the adaptive solver [s].
    num_steps  : int  *(static for JIT)*
        Number of uniformly-spaced output frames saved.

    Returns
    -------
    ts    : Float[Array, "num_steps"]        output times [s]
    qs    : Float[Array, "num_steps 12"]     generalised coordinates
    qdots : Float[Array, "num_steps 12"]     generalised velocities

    """
    state0 = jnp.concatenate([q0, qdot0])

    def vector_field(
        t:     Scalar,
        state: Float[Array, "24"],
        args:  None,
    ) -> Float[Array, "24"]:
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
# Physical parameters
# ─────────────────────────────────────────────────────────────────────────────

params = SimpleNamespace(
    # Bending-stiffness multiplier (dimensionless calibration constant)
    kappa_theta=2.65,
    # Disc height [m] and attachment-circle radius [m]
    h=10e-3,
    r=8.773827e-3,
    # Rigid-disc mass [kg]
    m_body=0.01,
    # Young's modulus of flexure wire [Pa]
    E=850e6,
    # Second moment of area [m⁴] and cross-sectional area [m²] of the wire
    I=0.1 * jnp.pi * (1e-3)**4 / 2,
    A=jnp.pi * (1e-3)**2,
    # Fraction of total flex length occupied by the spring segment (0–1)
    gamma=0.85,
    # Normalised position of rigid-arm lumped mass along the arm (0 = root, 1 = tip)
    mu=0.5,
    # Fraction of total flexure mass assigned to the spring lump (rest split between arms)
    alpha=0.6,
    # Total flexure mass per linkage [kg]
    m_flex=0.001,
    ic=[],  # filled below
)

# Principal moments of inertia for a solid cylinder:
#   Ixx = Iyy = (m/12)(3r² + h²),  Izz = m r² / 2
params.I_body = jnp.array([
    params.m_body * (3*params.r**2 + params.h**2) / 12,
    params.m_body * (3*params.r**2 + params.h**2) / 12,
    params.m_body * params.r**2 / 2,
])

# Reference configuration: body 0 at origin, body 1 at h/2, body 2 at h
params.ic = [
    {'pos': jnp.array([0.0, 0.0, 0.0]),          'rot': jnp.array([0.0, 0.0, 0.0])},
    {'pos': jnp.array([0.0, 0.0, params.h/2]),   'rot': jnp.array([0.0, 0.0, 0.0])},
    {'pos': jnp.array([0.0, 0.0, params.h]),     'rot': jnp.array([0.0, 0.0, 0.0])},
]

# Full 18-DOF initial state assembled from the reference configuration
q0    = jnp.array([val for ic in params.ic for key in ('pos', 'rot') for val in ic[key]])
qdot0 = jnp.zeros(18)


# ─────────────────────────────────────────────────────────────────────────────
# MeshCat 3-D visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def np_rot_mat(phi: np.ndarray) -> np.ndarray:
    """
    NumPy (non-JAX) Rodrigues rotation matrix, for use in MeshCat callbacks.

    Identical maths to ``rot_mat`` but returns a plain ``numpy.ndarray``
    for compatibility with MeshCat's transform API.

    Parameters
    ----------
    phi : np.ndarray, shape (3,)

    Returns
    -------
    np.ndarray, shape (3, 3)
    """
    phi   = np.asarray(phi, dtype=float)
    theta = np.linalg.norm(phi)
    if theta < 1e-9:
        return np.eye(3)
    k = phi / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K@K)


def homogeneous(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Assemble a 4×4 homogeneous transformation matrix.

    Parameters
    ----------
    R : np.ndarray, shape (3, 3)   rotation matrix
    p : np.ndarray, shape (3,)     translation vector [m]

    Returns
    -------
    np.ndarray, shape (4, 4)
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = p
    return T


def cylinder_between(
    vis:     meshcat.Visualizer,
    name:    str,
    p0:      np.ndarray,
    p1:      np.ndarray,
    radius:  float,
    color:   int,
    opacity: float = 0.9,
) -> None:
    """
    Place a MeshCat cylinder object spanning two world-frame endpoints.

    MeshCat cylinders are aligned along their local Y-axis.  This helper
    computes the rotation that maps Y → (p1 − p0) and positions the
    cylinder centred between *p0* and *p1*.

    Parameters
    ----------
    vis     : meshcat.Visualizer
    name    : str    scene-tree path (e.g. ``"link/0/ab/flex"``)
    p0, p1  : np.ndarray (3,)   world-frame endpoints [m]
    radius  : float  cylinder radius [m]
    color   : int    0xRRGGBB hex colour
    opacity : float  transparency (default 0.9)
    """
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
        # Parallel or anti-parallel — identity or flip
        R = np.eye(3) if np.dot(y, d) > 0 else np.diag([1., -1., 1.])
    else:
        ax = cr / cn
        an = np.arctan2(cn, np.dot(y, d))
        K  = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R  = np.eye(3) + np.sin(an)*K + (1 - np.cos(an))*(K@K)
    vis[name].set_object(
        g.Cylinder(length, radius),
        g.MeshLambertMaterial(color=color, opacity=opacity),
    )
    vis[name].set_transform(homogeneous(R, mid))


# ── Visualisation constants ───────────────────────────────────────────────────
BODY_COLORS: list[int] = [0x4fc3f7, 0xff8a65, 0xa5d6a7]  # blue / orange / green
FLEX_COLOR:  int       = 0xffd54f    # yellow  — flexible spring segments
RIGID_COLOR: int       = 0xb0bec5    # grey    — rigid arm segments
BODY_H_VIZ:  float     = float(params.h) * 0.55  # slightly shorter than h for aesthetics
BODY_R_VIZ:  float     = float(params.r)
FLEX_RAD:    float     = 0.0004      # spring wire visual radius [m]
RIGID_RAD:   float     = 0.0007      # rigid arm visual radius  [m]


def build_scene(vis: meshcat.Visualizer, params: SimpleNamespace) -> None:
    """
    Create all static and placeholder MeshCat objects.

    Called once at visualisation startup.  Flexure segments are
    initialised with degenerate (1 nm) geometry and overwritten every
    frame by ``update_scene``.

    Scene tree
    ~~~~~~~~~~
    ``grid``
        Thin dark ground plane.
    ``body/0``, ``body/1``, ``body/2``
        Coloured cylinders for the three discs.
    ``link/{i}/{ab|bc}/{rigid_a|flex|rigid_b}``
        Flexure segment placeholders (3 linkage sets × 2 pairs × 3 segs).
    ``axes/{x|y|z}``
        World-frame coordinate axes (15 mm long).

    Parameters
    ----------
    vis    : meshcat.Visualizer
    params : SimpleNamespace
    """
    # Ground plane
    vis["grid"].set_object(
        g.Box([0.10, 0.001, 0.10]),
        g.MeshLambertMaterial(color=0x1a2030, opacity=0.5),
    )
    vis["grid"].set_transform(tf.translation_matrix([0, -0.001, 0]))

    # Bodies — MeshCat cylinders are Y-up; post-multiply to align with Z-up world axis
    for b in range(3):
        vis[f"body/{b}"].set_object(
            g.Cylinder(BODY_H_VIZ, BODY_R_VIZ),
            g.MeshLambertMaterial(color=BODY_COLORS[b], opacity=0.92),
        )

    # Flexure segment placeholders
    for i in range(3):
        for pair in ("ab", "bc"):
            for seg, col in (("rigid_a", RIGID_COLOR), ("flex", FLEX_COLOR), ("rigid_b", RIGID_COLOR)):
                vis[f"link/{i}/{pair}/{seg}"].set_object(
                    g.Cylinder(1e-9, FLEX_RAD),
                    g.MeshLambertMaterial(color=col, opacity=0.9),
                )

    # World-frame coordinate axes
    for ax, col, vec in [("x", 0xff4444, [1, 0, 0]),
                         ("y", 0x44ff44, [0, 1, 0]),
                         ("z", 0x4488ff, [0, 0, 1])]:
        cylinder_between(vis, f"axes/{ax}", [0, 0, 0], np.array(vec)*0.015, 0.0003, col)


def update_scene(
    vis:    meshcat.Visualizer,
    q_full: np.ndarray,
    params: SimpleNamespace,
) -> None:
    """
    Update all MeshCat transforms for one simulation frame.

    Recomputes body poses and linkage geometry from *q_full* and calls
    ``set_transform`` on every scene object.

    Parameters
    ----------
    vis    : meshcat.Visualizer
    q_full : np.ndarray, shape (18,)   full 18-DOF state vector
    params : SimpleNamespace
    """
    q_full = np.asarray(q_full)
    bpos: list[np.ndarray] = []
    bmat: list[np.ndarray] = []

    for b in range(3):
        p = q_full[b*6:b*6+3]
        R = np_rot_mat(q_full[b*6+3:b*6+6])
        bpos.append(p); bmat.append(R)
        # MeshCat cylinders are Y-up; post-multiply to align with body Z-axis
        R_viz = R @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        vis[f"body/{b}"].set_transform(homogeneous(R_viz, p))

    def draw_linkage(
        prefix: str,
        ai:     int,
        rel_a:  np.ndarray,
        bi:     int,
        rel_b:  np.ndarray,
    ) -> None:
        """Draw rigid_a / flex / rigid_b cylinders for one RPR linkage."""
        Ra, pa = bmat[ai], bpos[ai]
        Rb, pb = bmat[bi], bpos[bi]
        pa0 = np.array(params.ic[ai]['pos'])
        Ra0 = np_rot_mat(np.array(params.ic[ai]['rot']))
        pb0 = np.array(params.ic[bi]['pos'])
        Rb0 = np_rot_mat(np.array(params.ic[bi]['rot']))
        fv0 = (pb0 + Rb0 @ rel_b) - (pa0 + Ra0 @ rel_a)
        g_  = params.gamma
        ral = Ra.T @ (fv0 * (1 - g_) / 2)    # rigid arm A direction in body frame
        rbl = Rb.T @ (-fv0 * (1 - g_) / 2)   # rigid arm B direction in body frame
        att_a = pa + Ra @ rel_a               # attachment point on body A
        spr_a = pa + Ra @ (rel_a + ral)        # spring start (tip of rigid arm A)
        att_b = pb + Rb @ rel_b               # attachment point on body B
        spr_b = pb + Rb @ (rel_b + rbl)        # spring end   (tip of rigid arm B)
        cylinder_between(vis, f"{prefix}/rigid_a", att_a, spr_a, RIGID_RAD, RIGID_COLOR)
        cylinder_between(vis, f"{prefix}/flex",    spr_a, spr_b, FLEX_RAD,  FLEX_COLOR)
        cylinder_between(vis, f"{prefix}/rigid_b", spr_b, att_b, RIGID_RAD, RIGID_COLOR)

    r = float(params.r)
    for i in range(3):
        rels = [np.array([r*np.cos((i+k)/3*2*np.pi),
                          r*np.sin((i+k)/3*2*np.pi), 0.]) for k in range(3)]
        draw_linkage(f"link/{i}/ab", 0, rels[0], 1, rels[1])
        draw_linkage(f"link/{i}/bc", 1, rels[1], 2, rels[2])


def visualize(
    ts:       np.ndarray,
    qs_full:  np.ndarray,
    params:   SimpleNamespace,
    speedup:  float = 0.3,
) -> None:
    """
    Play back a simulation trajectory in the MeshCat browser viewer.

    Opens a MeshCat server (prints the URL), builds the scene, then loops
    continuously through the trajectory at *speedup* × real-time speed
    until the user presses Ctrl-C.

    Parameters
    ----------
    ts       : np.ndarray, shape (N,)    time stamps [s]
    qs_full  : np.ndarray, shape (N, 18) full state trajectories
    params   : SimpleNamespace
    speedup  : float
        Playback speed relative to real time.  ``0.3`` means 3× slower.
    """
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
                elapsed = time.time() - t_start
                target  = (float(t) - float(ts[0])) / speedup
                if target - elapsed > 0:
                    time.sleep(target - elapsed)
            print(f"  replaying … (t_end={float(ts[-1]):.3f} s)")
    except KeyboardInterrupt:
        print("Stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Initial condition: small vertical impulse on body 1 ──────────────────
    q0_red    = q0[6:]                        # drop fixed body 0 (first 6 DOF)
    qdot0_red = qdot0.at[6+2].set(0.01)[6:]  # perturb body 1 vz by 0.01 m s⁻¹

    # ── Compile + integrate (includes JIT warm-up) ────────────────────────────
    print("Compiling + running simulation …")
    t_run = time.time()
    ts, qs, qdots = simulate(q0_red, qdot0_red, t0=0.0, t1=10, dt0=1e-4, num_steps=5000)
    ts.block_until_ready()
    print(f"Done in {time.time()-t_run:.2f}s  ({ts.shape[0]} steps)")

    # ── EOM throughput benchmark ──────────────────────────────────────────────
    q_test  = q0_red
    qd_test = qdot0_red
    _ = eom_reduced(q_test, qd_test).block_until_ready()  # warm-up

    import time  # re-import ensures we time only the loop, not JIT

    t = time.time()
    for _ in range(500):
        _ = eom_reduced(q_test, qd_test).block_until_ready()
    print(f"500 eom calls: {time.time() - t:.2f}s")

    # ── Pure runtime (post-compilation) ──────────────────────────────────────
    print("Second call (pure runtime) …")
    t0_wall = time.time()
    ts, qs, qdots = simulate(q0_red, qdot0_red, t0=0.0, t1=0.5, dt0=1e-4, num_steps=5000)
    ts.block_until_ready()
    print(f"  {time.time() - t0_wall:.2f}s")

    # ── Energy conservation check ─────────────────────────────────────────────
    qs_full    = jnp.concatenate([jnp.zeros((ts.shape[0], 6)), qs],    axis=1)
    qdots_full = jnp.concatenate([jnp.zeros((ts.shape[0], 6)), qdots], axis=1)

    T_hist, U_hist = jax.vmap(lambda q, qd: energy(q, qd, params))(qs_full, qdots_full)
    H_hist = T_hist + U_hist
    print(f"Energy drift: {float(H_hist[-1] - H_hist[0]):.2e} J")

    # ── Plots ─────────────────────────────────────────────────────────────────
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

    # ── 3-D browser animation ─────────────────────────────────────────────────
    visualize(ts, np.array(qs_full), params, speedup=0.3)