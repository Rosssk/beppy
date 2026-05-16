from beartype import beartype
from jaxtyping import jaxtyped
import jax.numpy as jnp

from prbdm.flexure import Params
from prbdm.math import T_mat, rot_mat
from prbdm.types import FullState, FullStateDot, Mat33, Scalar, Vec3


@jaxtyped(typechecker=beartype)
def body_energy(
    q: FullState, qdot: FullStateDot, p: Params, i: int
) -> tuple[Scalar, Scalar]:
    body = q[i]
    bodydot = qdot[i]
    R: Mat33 = rot_mat(body.rot)
    omega: Vec3 = T_mat(body.rot) @ bodydot.rotvel  # world-frame angular velocity
    omega_body: Vec3 = R.T @ omega  # body-frame angular velocity

    U = 9.81 * p.m_body * body.pos[2]
    Ek_t = 0.5 * p.m_body * jnp.dot(bodydot.vel, bodydot.vel)
    Ek_r = 0  # todo
    return Ek_t + Ek_r, U


@jaxtyped(typechecker=beartype)
def energy(q: FullState, qdot: FullStateDot, p: Params) -> tuple[Scalar, Scalar]:
    T: Scalar = jnp.zeros(())
    U: Scalar = jnp.zeros(())

    for i in range(3):
        t, u = body_energy(q, qdot, p, i)
        T += t
        U += u

    # for i in range(2):
    #     for j in range(3):
    #         flexure_energy(q, qdot,)

    return T, U
