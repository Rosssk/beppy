from dataclasses import dataclass

import jax.numpy as jnp

from .model import Model


@dataclass(frozen=True)
class Compiled:
    body_index: dict[str, int]

    # initial state
    init_pos: jnp.ndarray  # (N, 3)
    init_ang: jnp.ndarray  # (N, 3)

    # flexures
    flex_idx: jnp.ndarray   # (M, 2)
    attach: jnp.ndarray     # (M, 2, 3)

    # forces
    force_body: jnp.ndarray   # (F,)
    force_vec: jnp.ndarray    # (F, 3)
    force_attach: jnp.ndarray # (F, 3)


def compile_model(model: Model) -> Compiled:
    names = list(model.bodies.keys())
    body_index = {n: i for i, n in enumerate(names)}

    # bodies
    pos = jnp.array([model.bodies[n].position for n in names], dtype=float)
    ang = jnp.array([model.bodies[n].angles for n in names], dtype=float)

    # flexures
    if model.flexures:
        flex_idx = jnp.array([
            [body_index[f.a], body_index[f.b]]
            for f in model.flexures
        ], dtype=int)

        attach = jnp.array([
            [f.attach_a, f.attach_b]
            for f in model.flexures
        ], dtype=float)
    else:
        flex_idx = jnp.zeros((0, 2), dtype=int)
        attach = jnp.zeros((0, 2, 3), dtype=float)

    # forces
    if model.forces:
        force_body = jnp.array([body_index[f.body] for f in model.forces], dtype=int)
        force_vec = jnp.array([f.vector for f in model.forces], dtype=float)
        force_attach = jnp.array([f.attach for f in model.forces], dtype=float)
    else:
        force_body = jnp.zeros((0,), dtype=int)
        force_vec = jnp.zeros((0, 3), dtype=float)
        force_attach = jnp.zeros((0, 3), dtype=float)

    return Compiled(
        body_index=body_index,
        init_pos=pos,
        init_ang=ang,
        flex_idx=flex_idx,
        attach=attach,
        force_body=force_body,
        force_vec=force_vec,
        force_attach=force_attach,
    )