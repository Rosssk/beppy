from dataclasses import dataclass, field
from typing import Final, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy
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


def _angle_between(a: Vec3, b: Vec3) -> Float[Array, ""]:
    """Angle in radians between two unit vectors."""
    return jnp.arccos(jnp.clip(jnp.dot(a, b), -1.0, 1.0))


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
        if length_0 == 0.0:
            raise ValueError(
                f"Attachment points of flexure between '{self.body_a.name}' and "
                f"'{self.body_b.name}' coincide; cannot determine spring direction."
            )

        spring_unit = flexure_vec / length_0
        self.spring_len_0 = float(length_0 * self.gamma)
        self.spring_a_dir_0 = rot_a.T @ spring_unit
        self.spring_b_dir_0 = rot_b.T @ spring_unit

    def tree_flatten(self):
        leaves = (self.spring_a_dir_0, self.spring_b_dir_0, self.body_a, self.body_b)
        aux = (self.attach_point_a_local, self.attach_point_b_local, self.gamma, self.spring_len_0)
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        attach_a, attach_b, gamma, spring_len_0 = aux
        spring_a_dir_0, spring_b_dir_0, body_a, body_b = leaves
        f = object.__new__(cls)
        object.__setattr__(f, "body_a", body_a)
        object.__setattr__(f, "body_b", body_b)
        object.__setattr__(f, "attach_point_a_local", attach_a)
        object.__setattr__(f, "attach_point_b_local", attach_b)
        object.__setattr__(f, "gamma", gamma)
        object.__setattr__(f, "spring_len_0", spring_len_0)
        object.__setattr__(f, "spring_a_dir_0", spring_a_dir_0)
        object.__setattr__(f, "spring_b_dir_0", spring_b_dir_0)
        return f

    def energy(self, A: float, E: float, I: float) -> Float[Array, ""]:
        """
        Potential energy stored in this flexure given current body states.

        Parameters
        ----------
        A : float
            Cross-sectional area.
        E : float
            Young's modulus.
        I : float
            Second moment of area.
        """
        kappa = self.gamma * KAPPA_THETA * E * I / self.spring_len_0
        k = E * A / self.spring_len_0

        rot_a = rotmat(self.body_a.angles)
        rot_b = rotmat(self.body_b.angles)

        attach_a_global = self.body_a.position + rot_a @ self.attach_point_a_local
        attach_b_global = self.body_b.position + rot_b @ self.attach_point_b_local

        flexure_vec = attach_b_global - attach_a_global
        length = jnp.linalg.norm(flexure_vec)
        spring_unit = flexure_vec / length

        # Axial stretch/compression energy
        u = length * self.gamma - self.spring_len_0
        energy_axial = k * u ** 2 / 2

        # Torsional energy at each attachment
        spring_unit_a = rot_a.T @ spring_unit
        spring_unit_b = rot_b.T @ spring_unit
        theta_a = _angle_between(spring_unit_a, self.spring_a_dir_0)
        theta_b = _angle_between(spring_unit_b, self.spring_b_dir_0)
        energy_torsion = kappa * (theta_a ** 2 + theta_b ** 2) / 2

        return energy_axial + energy_torsion


class PRBM:
    """
    Pseudo-Rigid-Body Model (PRBM) container.

    Manages a collection of rigid bodies connected by compliant flexures,
    and provides energy minimisation to solve for equilibrium poses.
    """

    def __init__(self, gamma: float = GAMMA) -> None:
        self.gamma = gamma
        self.bodies: dict[str, Body] = {}
        self.flexures: dict[str, Flexure] = {}

    def add_body(self, name: str, position: Vec3 | tuple[float, float, float] | None = None) -> None:
        """Add a rigid body to the model."""
        pos = jnp.zeros(3) if position is None else jnp.asarray(position, dtype=float)
        body = Body(name=name, position_0=pos, position=pos, angles=jnp.zeros(3))
        self.bodies[name] = body

    def add_flexure(
            self,
            body_name_a: str,
            attach_point_a_local: Vec3 | tuple[float, float, float],
            body_name_b: str,
            attach_point_b_local: Vec3 | tuple[float, float, float],
    ) -> None:
        """Add a flexure between two bodies."""
        body_a = self.bodies[body_name_a]
        body_b = self.bodies[body_name_b]
        attach_a = jnp.asarray(attach_point_a_local, dtype=float)
        attach_b = jnp.asarray(attach_point_b_local, dtype=float)

        flexure = Flexure(body_a, body_b, attach_a, attach_b, self.gamma)
        body_a.points.append(attach_a)
        body_b.points.append(attach_b)

        name = self._unique_flexure_name(body_name_a, body_name_b)
        self.flexures[name] = flexure

    def add_force(
            self,
            body_name: str,
            force: Vec3 | tuple[float, float, float],
            attach_point_local: Vec3 | tuple[float, float, float] | None = None,
    ) -> None:
        """Apply an external force to a body."""
        body = self.bodies[body_name]
        attach = jnp.zeros(3) if attach_point_local is None else jnp.asarray(attach_point_local, dtype=float)
        body.forces.append(Force(
            vector=jnp.asarray(force, dtype=float),
            attach_point_local=attach,
        ))

    def _body_force_energy(self, body: Body) -> Float[Array, ""]:
        """Potential energy of a body due to its external forces."""
        def force_energy(f: Force) -> Float[Array, ""]:
            attach_global = rotmat(body.angles) @ f.attach_point_local + body.position
            return -jnp.dot(attach_global, f.vector)

        return jnp.sum(jnp.array([force_energy(f) for f in body.forces])) if body.forces else jnp.array(0.0)

    def total_energy(self, A: float, E: float, I: float) -> Float[Array, ""]:
        """Total potential energy of the model in its current state."""
        flexure_energy = sum((f.energy(A, E, I) for f in self.flexures.values()), jnp.array(0.0))
        body_energy = sum((self._body_force_energy(b) for b in self.bodies.values()), jnp.array(0.0))
        return flexure_energy + body_energy

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def _pack_state(self, body_names: list[str]) -> Float[Array, "n"]:
        """Flatten positions and angles of the given bodies into a 1D vector."""
        return jnp.concatenate([
            jnp.concatenate([self.bodies[n].position, self.bodies[n].angles])
            for n in body_names
        ])

    def _unpack_state(self, body_names: list[str], state: Float[Array, "n"]) -> None:
        """Write a flat state vector back into the bodies."""
        for i, name in enumerate(body_names):
            self.bodies[name].position = state[i * 6: i * 6 + 3]
            self.bodies[name].angles = state[i * 6 + 3: i * 6 + 6]

    def solve_pose(
            self,
            body_names: list[str],
            A: float,
            E: float,
            I: float,
            method: str | None = None,
            x0: Float[Array, "n"] | None = None,
            options: dict | None = None,
    ) -> scipy.optimize.OptimizeResult:
        """
        Solve for the equilibrium pose of the given bodies by minimising total energy.

        Parameters
        ----------
        body_names : list[str]
            Names of the bodies whose pose is to be optimised. Bodies not listed
            are treated as fixed.
        A, E, I : float
            Cross-sectional area, Young's modulus, and second moment of area.
        method : str, optional
            Scipy minimisation method. Defaults to L-BFGS-B.
        x0 : array, optional
            Initial state vector. Defaults to current body states.
        options : dict, optional
            Passed directly to scipy.optimize.minimize.

        Returns
        -------
        scipy.optimize.OptimizeResult
        """
        if x0 is None:
            x0 = self._pack_state(body_names)

        def objective(state_np: np.ndarray) -> tuple[float, np.ndarray]:
            state = jnp.asarray(state_np)
            self._unpack_state(body_names, state)
            energy, grad = jax.value_and_grad(
                lambda s: (self._unpack_state(body_names, s) or self.total_energy(A, E, I))
            )(state)
            return float(energy), np.array(grad)

        result = scipy.optimize.minimize( 
            objective,
            x0=np.array(x0),
            jac=True,
            method=method or "L-BFGS-B",
            options=options,
        )

        # Write the solution back into the bodies
        self._unpack_state(body_names, jnp.asarray(result.x))
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unique_flexure_name(self, name_a: str, name_b: str) -> str:
        base = name_a + name_b
        count = sum(1 for key in self.flexures if key.startswith(base))
        return f"{base}{count}"
