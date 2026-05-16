from beartype import beartype
from jaxtyping import jaxtyped
import jax.numpy as jnp

from _types import Mat33, Vec3, Scalar


@jaxtyped(typechecker=beartype)
def skew(v: Vec3) -> Mat33:
    """
    3×3 skew-symmetric (cross-product) matrix for vector *v*.

    Satisfies  ``skew(v) @ w == jnp.cross(v, w)``  for any w ∈ R³.

    Parameters
    ----------
    v : Vec3
        Input 3-vector.

    Returns
    -------
    Mat33
        Skew-symmetric matrix K such that K @ w = v × w.
    """
    return jnp.array([[ 0,    -v[2],  v[1]],
                      [ v[2],  0,    -v[0]],
                      [-v[1],  v[0],  0   ]])


@jaxtyped(typechecker=beartype)
def rot_mat(phi: Vec3) -> Mat33:
    """
    Rodrigues rotation matrix from a rotation vector *phi*.

    ``phi`` encodes the rotation axis (unit vector) scaled by the rotation
    angle θ = ‖phi‖.  Near θ = 0 a Taylor expansion is used to avoid 0/0.

    Formula::

        R = I + sin(θ)/θ · K + (1 − cos θ)/θ² · K²
        K = skew(phi)

    Parameters
    ----------
    phi : Vec3
        Rotation vector.  Direction = axis, magnitude = angle [rad].

    Returns
    -------
    Mat33
        Orthogonal rotation matrix with det = +1.
    """
    theta_sq = jnp.dot(phi, phi)
    K = skew(phi)
    safe_tsq = jnp.where(theta_sq > 1e-6, theta_sq, jnp.ones_like(theta_sq))
    safe_t   = jnp.sqrt(safe_tsq)
    # Taylor-safe coefficients  a = sin θ / θ,  b = (1 − cos θ) / θ²
    a = jnp.where(theta_sq > 1e-6, jnp.sin(safe_t)/safe_t,
                  1 - theta_sq/6  + theta_sq**2/120)
    b = jnp.where(theta_sq > 1e-6, (1-jnp.cos(safe_t))/safe_tsq,
                  0.5 - theta_sq/24 + theta_sq**2/720)
    return jnp.eye(3) + a*K + b*(K@K)


@jaxtyped(typechecker=beartype)
def T_mat(phi: Vec3) -> Mat33:
    """
    Kinematic map from Rodrigues-parameter rate to body angular velocity.

    Given the Rodrigues parameterisation, the world-frame angular velocity
    satisfies  ``ω = T(φ) @ φ̇``.

    Formula::

        T = I − (1 − cos θ)/θ² · K + (θ − sin θ)/θ³ · K²

    Near θ = 0 the Taylor expansion avoids 0/0.

    Parameters
    ----------
    phi : Vec3
        Current rotation vector.

    Returns
    -------
    Mat33
        Kinematic matrix T(φ).
    """
    theta_sq = jnp.dot(phi, phi)
    K = skew(phi)
    safe_tsq = jnp.where(theta_sq > 1e-6, theta_sq, jnp.ones_like(theta_sq))
    safe_t   = jnp.sqrt(safe_tsq)
    a = jnp.where(theta_sq > 1e-6, (1-jnp.cos(safe_t))/safe_tsq,
                  0.5  - theta_sq/24  + theta_sq**2/720)
    b = jnp.where(theta_sq > 1e-6, (safe_t-jnp.sin(safe_t))/(safe_tsq*safe_t),
                  1/6  - theta_sq/120 + theta_sq**2/5040)
    return jnp.eye(3) - a*K + b*(K@K)


@jaxtyped(typechecker=beartype)
def angle_between(u: Vec3, v: Vec3) -> Scalar:
    """
    Angle in [0, π] between two 3-D vectors *u* and *v*.

    Uses ``atan2(‖u × v‖, u·v)`` for numerical robustness near 0 and π.
    A small epsilon (1e-20) prevents an exact-zero cross-product norm from
    causing a NaN gradient through ``sqrt``.

    Parameters
    ----------
    u, v : Vec3

    Returns
    -------
    Scalar
        Angle in radians.
    """
    cross = jnp.cross(u, v)
    return jnp.arctan2(jnp.sqrt(jnp.dot(cross, cross) + 1e-20), jnp.dot(u, v))
