from collections import namedtuple
from typing import Annotated, NamedTuple
import jax.numpy as jnp

from prbdm.types import BodyState, FullState, FullStateDot, Vec3, Scalar
from prbdm.math import T_mat, angle_between, rot_mat


class Params(NamedTuple):
    # Material
    e_mod: float  # Young's modulus
    v_rat: float  # Poisson's ratio
    rho: float  # density

    # PRB
    k_th: float
    k_ex: float
    gamma: float
    alpha: float
    mu: float

    # Structural
    m_body: float
    r_attach: float  # flexure attachment radius
    r_flex: float  # flexure radius

    # Initial condition stuff
    flexure_l0: float  # Unstretched length of flexure
    flex_p_spring_l: tuple[  # Local offsets of spring connection point
        tuple[Vec3, Vec3, Vec3], tuple[Vec3, Vec3, Vec3]
    ]
    flex_p_attach_l: tuple[  # Local offsets of flexure connection point
        tuple[Vec3, Vec3, Vec3], tuple[Vec3, Vec3, Vec3]
    ]


class FlexureState(NamedTuple):
    rig_a: Vec3  # Position at the end of the rigid section connected to the lower body
    mass_a: Vec3  # Position of mass along rigid section
    v_mass_a: Vec3  # Velocity of mass
    w_mass_a: Vec3  # Rot. vel. of body around mass
    theta_a: Scalar

    rig_b: Vec3  # Position at the end of the rigid section connected to the upper body
    mass_b: Vec3  # etc..
    v_mass_b: Vec3
    w_mass_b: Vec3
    theta_b: Scalar

    spring_middle: Vec3  # (Mass) point at the center of the prismatic spring beam
    v_spr: Vec3  # Velocity of spring mass point
    w_spr: Vec3  # Rot. vel. of spring mass point


def flexure_state(
    s: FullState, s_dot: FullStateDot, p: Params, b: int, i: int
) -> FlexureState:
    """Get the state of the i-th flexure with lower body index b
    """
    # Flexure: attach_a -(rig_a)-> spring_a -(spring_vec)-> spring_b -(rig_b)-> attach_b
    # mass_a and mass_b are along rig_a and rig_b

    # --------------------------------
    # Positions:
    # --------------------------------
    body_a = s[b]
    body_b = s[b + 1]
    rotmat_a = rot_mat(body_a.rot)
    rotmat_b = rot_mat(body_b.rot)

    # Flexure joint positions
    attach_a_local = p.flex_p_attach_l[b][i]
    attach_a: Vec3 = body_a.pos + rotmat_a @ attach_a_local
    attach_b_local = p.flex_p_attach_l[b][(i + 1) % 3]
    attach_b: Vec3 = body_b.pos - rotmat_b @ attach_b_local
    spring_a_local = p.flex_p_spring_l[b][i]
    spring_a: Vec3 = body_a.pos + rotmat_a @ spring_a_local
    spring_b_local = p.flex_p_spring_l[b][(i + 1) % 3]
    spring_b: Vec3 = body_b.pos + rotmat_b @ spring_b_local

    # Relative link vectors
    rig_a: Vec3 = spring_a - attach_a
    rig_a_local: Vec3 = spring_a_local - attach_a_local
    rig_b: Vec3 = attach_b - spring_b
    rig_b_local: Vec3 = attach_b_local - spring_b_local
    spring_v: Vec3 = spring_b - spring_a

    # World frame positions of masses along links
    mass_a = attach_a + rig_a * p.mu
    mass_b = attach_b - rig_b * p.mu
    spring_center = attach_a + spring_v / 2

    # --------------------------------
    # Angles:
    # --------------------------------
    theta_a = angle_between(rig_a, spring_v)
    theta_b = angle_between(spring_v, rig_b)

    # --------------------------------
    # Speeds:
    # --------------------------------
    body_a_d = s_dot[b]
    body_a_omega = T_mat(body_a.rot) @ body_a_d.rotvel
    body_b_d = s_dot[b + 1]
    body_b_omega = T_mat(body_b.rot) @ body_b_d.rotvel

    v_mass_a = body_a_d.vel + jnp.cross(body_a_omega, rotmat_a @ (p.mu * rig_a_local))
    v_mass_b = body_b_d.vel + jnp.cross(body_b_omega, rotmat_b @ -(p.mu * rig_b_local))
    v_spring_a = body_a_d.vel + jnp.cross(body_a_omega, rotmat_a @ rig_a_local)
    v_spring_b = body_b_d.vel + jnp.cross(body_b_omega, rotmat_b @ -rig_b_local)
    v_spring_center = (v_spring_a + v_spring_b) * 0.5

    def rod_omega(v_tip_a: Vec3, v_tip_b: Vec3, rod_vec: Vec3) -> Vec3:
        """Angular velocity of a rod from its two endpoint velocities."""
        rod_len_sq = jnp.dot(rod_vec, rod_vec)
        v_rel = v_tip_b - v_tip_a
        return jnp.cross(rod_vec, v_rel) / rod_len_sq

    # Angular velocity of rig_a rod (attach_a -> spring_a)
    v_attach_a = body_a_d.vel + jnp.cross(body_a_omega, rotmat_a @ attach_a_local)
    w_rig_a = rod_omega(v_attach_a, v_spring_a, rig_a)

    # Angular velocity of rig_b rod (spring_b -> attach_b)
    v_attach_b = body_b_d.vel + jnp.cross(body_b_omega, rotmat_b @ -attach_b_local)
    w_rig_b = rod_omega(v_spring_b, v_attach_b, rig_b)

    # Angular velocity of spring rod (spring_a -> spring_b)
    w_spring = rod_omega(v_spring_a, v_spring_b, spring_v)

    w_mass_a = w_rig_a   # mass_a rides on rig_a
    w_mass_b = w_rig_b   # mass_b rides on rig_b
    w_spr_c  = w_spring  # spring center rides on spring_v rod

    return FlexureState(
        rig_a,
        mass_a,
        v_mass_a,
        w_mass_a,
        theta_a,
        rig_b,
        mass_b,
        v_mass_b,
        w_mass_b,
        theta_b,
        spring_center,
        v_spring_center,
        w_spr_c,
    )
