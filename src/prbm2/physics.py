import jax.numpy as jnp

from .compile import Compiled


def rotmat_batch(angles):
    rx, ry, rz = angles[:, 0], angles[:, 1], angles[:, 2]

    cx, sx = jnp.cos(rx), jnp.sin(rx)
    cy, sy = jnp.cos(ry), jnp.sin(ry)
    cz, sz = jnp.cos(rz), jnp.sin(rz)

    Rx = jnp.stack([
        jnp.stack([jnp.ones_like(cx), 0*cx, 0*cx], axis=-1),
        jnp.stack([0*cx, cx, -sx], axis=-1),
        jnp.stack([0*cx, sx, cx], axis=-1),
    ], axis=-2)

    Ry = jnp.stack([
        jnp.stack([cy, 0*cy, sy], axis=-1),
        jnp.stack([0*cy, jnp.ones_like(cy), 0*cy], axis=-1),
        jnp.stack([-sy, 0*cy, cy], axis=-1),
    ], axis=-2)

    Rz = jnp.stack([
        jnp.stack([cz, -sz, 0*cz], axis=-1),
        jnp.stack([sz, cz, 0*cz], axis=-1),
        jnp.stack([0*cz, 0*cz, jnp.ones_like(cz)], axis=-1),
    ], axis=-2)

    return Rx @ Ry @ Rz


def energy_fn(state, compiled: Compiled):
    N = compiled.init_pos.shape[0]
    state = state.reshape(N, 6)

    pos = state[:, :3]
    ang = state[:, 3:]

    R = rotmat_batch(ang)

    # ---- flexures ----
    if compiled.flex_idx.shape[0] > 0:
        ia = compiled.flex_idx[:, 0]
        ib = compiled.flex_idx[:, 1]

        pa = pos[ia]
        pb = pos[ib]

        Ra = R[ia]
        Rb = R[ib]

        attach_a = compiled.attach[:, 0]
        attach_b = compiled.attach[:, 1]

        ga = pa + jnp.einsum("nij,nj->ni", Ra, attach_a)
        gb = pb + jnp.einsum("nij,nj->ni", Rb, attach_b)

        vec = gb - ga
        length = jnp.linalg.norm(vec, axis=1)

        energy_flex = jnp.sum((length - 1.0) ** 2)
    else:
        energy_flex = 0.0

    # ---- forces ----
    if compiled.force_body.shape[0] > 0:
        fb = compiled.force_body
        pf = pos[fb]
        Rf = R[fb]

        gf = pf + jnp.einsum("nij,nj->ni", Rf, compiled.force_attach)
        energy_force = -jnp.sum(jnp.sum(gf * compiled.force_vec, axis=1))
    else:
        energy_force = 0.0

    return energy_flex + energy_force