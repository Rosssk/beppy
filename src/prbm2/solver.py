import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize

from .compile import Compiled
from .physics import energy_fn


def solve(compiled: Compiled):
    x0 = jnp.concatenate(
        [compiled.init_pos, compiled.init_ang],
        axis=1
    ).reshape(-1)

    energy_grad = jax.jit(
        jax.value_and_grad(lambda s: energy_fn(s, compiled))
    )

    def objective(x_np: np.ndarray):
        x = jnp.asarray(x_np)
        e, g = energy_grad(x)
        return float(e), np.array(g)

    result = scipy.optimize.minimize(
        objective,
        x0=np.array(x0),
        jac=True,
        method="L-BFGS-B",
    )

    return result