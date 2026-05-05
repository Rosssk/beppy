from dataclasses import dataclass, field
from typing import Final, NamedTuple

import jax
import jax.numpy as jnp
from beartype.claw import beartype_package
from jax import Array
from jaxtyping import Float, install_import_hook

# PRBM constants
GAMMA: Final[float] = 0.85
KAPPA_THETA: Final[float] = 2.65

Vec3 = Float[Array, "3"]
Mat33 = Float[Array, "3 3"]

# TODO: disable when necessary
install_import_hook("prbm", typechecker="beartype.beartype")
beartype_package("prbm")


def rotmat(angles: Vec3) -> Mat33:
    """
    Compute a combined rotation matrix Rx @ Ry @ Rz for Euler angles [rx, ry, rz].

    Parameters
    ----------
    angles : Float[Array, "3"]
        Rotation angles [rx, ry, rz] in radians.

    Returns
    -------
    Float[Array, "3 3"]
    """
    rx_val, ry_val, rz_val = angles

    rx = jnp.array([[1, 0, 0], [0, jnp.cos(rx_val), -jnp.sin(rx_val)], [0, jnp.sin(rx_val), jnp.cos(rx_val)], ])
    ry = jnp.array([[jnp.cos(ry_val), 0, jnp.sin(ry_val)], [0, 1, 0], [-jnp.sin(ry_val), 0, jnp.cos(ry_val)], ])
    rz = jnp.array([[jnp.cos(rz_val), -jnp.sin(rz_val), 0], [jnp.sin(rz_val), jnp.cos(rz_val), 0], [0, 0, 1], ])
    return rx @ ry @ rz


class Force(NamedTuple):
    vector: Vec3
    attach_point_local: Vec3


@jax.tree_util.register_pytree_node_class
@dataclass
class Body:
    """
    Represents a rigid body in the PRBM.

    Attributes
    ----------
    name : str
        Unique identifier. Static — never changes after construction.
    position_0 : Vec3
        Reference position in global frame. Static.
    position : Vec3
        Current position in global frame.
    angles : Vec3
        Current Euler angles [rx, ry, rz] in radians.
    points : list[Vec3]
        Tracked points attached to the body, in local coordinates.
    """

    name: Final[str]
    position_0: Final[Vec3]

    position: Vec3
    angles: Vec3

    forces: list[Force] = field(default_factory=list)
    points: list[Vec3] = field(default_factory=lambda: [jnp.zeros(3)])

    def tree_flatten(self):
        leaves = (self.position, self.angles)
        aux = (self.name, self.position_0, self.forces, self.points)
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        name, position_0, forces, points = aux
        position, angles = leaves
        return cls(name=name, position_0=position_0, position=position,
                   angles=angles, forces=forces, points=points)


@jax.tree_util.register_pytree_node_class
@dataclass
class Flexure:
    """
    Compliant flexure between two rigid bodies, modelled as a linear spring
    connected to the bodies by torsional springs.
    """

    body_a: Body
    body_b: Body
    attach_point_a_local: Final[Vec3]
    attach_point_b_local: Final[Vec3]
    gamma: Final[float]

    # Derived by __post_init__
    spring_len_0: float = field(init=False)
    spring_a_dir_0: Vec3 = field(init=False)
    spring_b_dir_0: Vec3 = field(init=False)

    def __post_init__(self) -> None:
        rot_a = rotmat(self.body_a.angles)
        rot_b = rotmat(self.body_b.angles)

        attach_a_global = self.body_a.position + rot_a @ self.attach_point_a_local
        attach_b_global = self.body_b.position + rot_b @ self.attach_point_b_local

        flexure_vec = attach_b_global - attach_a_global
        length_0 = jnp.linalg.norm(flexure_vec)

        spring_unit = flexure_vec / length_0
        self.spring_len_0 = float(length_0 * self.gamma)
        self.spring_a_dir_0 = rot_a.T @ spring_unit
        self.spring_b_dir_0 = rot_b.T @ spring_unit

    #
    # Methods for jax pytree conversion
    #
    @staticmethod
    def flatten(f: Flexure):
        leaves = (f.spring_a_dir_0, f.spring_b_dir_0, f.body_a, f.body_b)
        aux = (f.attach_point_a_local, f.attach_point_b_local, f.gamma, f.spring_len_0)
        return leaves, aux

    @staticmethod
    def unflatten(aux, leaves):
        attach_a, attach_b, gamma, spring_len_0 = aux
        spring_a_dir_0, spring_b_dir_0, body_a, body_b = leaves
        f = object.__new__(Flexure)
        object.__setattr__(f, "body_a", body_a)
        object.__setattr__(f, "body_b", body_b)
        object.__setattr__(f, "attach_point_a_local", attach_a)
        object.__setattr__(f, "attach_point_b_local", attach_b)
        object.__setattr__(f, "gamma", gamma)
        object.__setattr__(f, "spring_len_0", spring_len_0)
        object.__setattr__(f, "spring_a_dir_0", spring_a_dir_0)
        object.__setattr__(f, "spring_b_dir_0", spring_b_dir_0)
        return f


class PRBM:
    """
    Pseudo-Rigid-Body Model (PRBM) container.

    Manages a collection of rigid bodies connected by compliant flexures.
    """

    def __init__(self, gamma: float = GAMMA) -> None:
        self.gamma = gamma
        self.bodies: dict[str, Body] = {}
        self.flexures: dict[str, Flexure] = {}

    def add_body(self, name: str, position: Vec3 | tuple[float, float, float] | None = None) -> None:
        """
        Add a rigid body to the model.
        """
        pos = jnp.zeros(3) if position is None else jnp.asarray(position, dtype=float)
        body = Body(name=name, position_0=pos, position=pos, angles=jnp.zeros(3))
        self.bodies[name] = body

    def add_flexure(self, body_name_a: str, attach_point_a_local: Vec3 | tuple[float, float, float], body_name_b: str,
                    attach_point_b_local: Vec3 | tuple[float, float, float], ) -> None:
        """
        Add a flexure (compliant spring element) between two bodies.
        """
        body_a = self.bodies[body_name_a]
        body_b = self.bodies[body_name_b]
        attach_point_a_local = jnp.asarray(attach_point_a_local, dtype=float)
        attach_point_b_local = jnp.asarray(attach_point_b_local, dtype=float)

        flexure = Flexure(body_a, body_b, attach_point_a_local, attach_point_b_local, self.gamma)

        # Add attachment points to respective bodies for display purposes
        body_a.points.append(attach_point_a_local)
        body_b.points.append(attach_point_b_local)

        name = self._unique_flexure_name(body_name_a, body_name_b)
        self.flexures[name] = flexure

    def add_force(self, body_name: str, force: Vec3 | tuple[float, float, float], attach_point_local: Vec3 | tuple[float, float, float] | None = None, ) -> None:
        body = self.bodies[body_name]
        pos = jnp.zeros(3) if attach_point_local is None else jnp.asarray(attach_point_local, dtype=float)
        force = jnp.asarray(force, dtype=float)
        body.forces.append((force, pos))

    def _unique_flexure_name(self, name_a: str, name_b: str) -> str:
        """Return a unique key for a flexure between two bodies."""
        base = name_a + name_b
        count = sum(1 for key in self.flexures if key.startswith(base))
        return f"{base}{count}"
