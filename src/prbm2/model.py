# pyright: strict

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

# PRBM constants
GAMMA: float = 0.85
KAPPA_THETA: float = 2.65

Vec3 = NDArray[np.float64]
Mat3 = NDArray[np.float64]


def as_vec3(v: ArrayLike) -> Vec3:
    arr: Vec3 = np.array(v, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"Expected 3D vector, got shape {arr.shape}")
    arr.flags.writeable = False
    return arr


def as_mat3(m: ArrayLike) -> Mat3:
    arr: Mat3 = np.array(m, dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix, got shape {arr.shape}")
    arr.flags.writeable = False
    return arr


def make_rotmat(angles: Vec3) -> Mat3:
    angles = angles
    rx_val, ry_val, rz_val = map(float, angles)

    rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx_val), -np.sin(rx_val)],
        [0, np.sin(rx_val), np.cos(rx_val)],
    ])
    ry = np.array([
        [np.cos(ry_val), 0, np.sin(ry_val)],
        [0, 1, 0],
        [-np.sin(ry_val), 0, np.cos(ry_val)],
    ])
    rz = np.array([
        [np.cos(rz_val), -np.sin(rz_val), 0],
        [np.sin(rz_val), np.cos(rz_val), 0],
        [0, 0, 1],
    ])
    return as_mat3(rx @ ry @ rz)


@dataclass(frozen=True, slots=True)
class ForceDef:
    body: str
    force: Vec3
    attach_local: Vec3


@dataclass(frozen=True, slots=True)
class BodyDef:
    name: str
    position: Vec3
    angles: Vec3
    rotmat: Mat3

    @classmethod
    def create(cls, name: str, position: ArrayLike | None, angles: ArrayLike | None = None) -> BodyDef:
        angles = as_vec3(np.zeros(3) if angles is None else angles)
        position = as_vec3(np.zeros(3) if position is None else position)
        return BodyDef(name, position, angles, as_mat3(make_rotmat(angles)))


@dataclass(frozen=True, slots=True)
class FlexureDef:
    body_a: str
    body_b: str
    attach_a_local: Vec3
    attach_b_local: Vec3

    spring_l0: float
    spring_dir_a_local: Vec3
    spring_dir_b_local: Vec3

    @classmethod
    def create(cls, body_a: BodyDef, body_b: BodyDef, attach_a_local: ArrayLike, attach_b_local: ArrayLike,
               spring_l0: float | None = None) -> FlexureDef:
        attach_a_local = as_vec3(attach_a_local)
        attach_b_local = as_vec3(attach_b_local)
        attach_a_global = body_a.position + body_a.rotmat @ attach_a_local
        attach_b_global = body_b.position + body_b.rotmat @ attach_b_local

        flexure_vec = attach_b_global - attach_a_global
        length_0 = np.linalg.norm(flexure_vec)

        if length_0 < 1e-12:
            raise ValueError(
                f"Attachment points of flexure between '{body_a}' and '{body_b}' coincide"
            )

        spring_unit_vector = flexure_vec / length_0
        spring_unit_a_local = as_vec3(body_a.rotmat.T @ spring_unit_vector)
        spring_unit_b_local = as_vec3(body_b.rotmat.T @ (-spring_unit_vector))

        if spring_l0 is None:
            spring_l0 = float(length_0) * GAMMA

        return FlexureDef(body_a.name, body_b.name, attach_a_local, attach_b_local, spring_l0, spring_unit_a_local,
                          spring_unit_b_local)


class Model:
    def __init__(self) -> None:
        self.bodies: dict[str, BodyDef] = {}
        self.flexures: list[FlexureDef] = []
        self.forces: list[ForceDef] = []

    def _get_body(self, name: str) -> BodyDef:
        if name not in self.bodies:
            raise KeyError(f"No body named {name!r}. Existing bodies: {list(self.bodies)}")
        return self.bodies[name]

    def add_body(self, name: str, position: ArrayLike | None = None,) -> None:
        if name in self.bodies:
            raise ValueError(f"A body named {name!r} already exists")

        self.bodies[name] = BodyDef.create(name, position)

    def add_flexure(self, body_a: str, attach_a_local: ArrayLike, body_b: str, attach_b_local: ArrayLike,
                    spring_l0: float | None = None, ) -> None:
        body_a_ = self._get_body(body_a)
        body_b_ = self._get_body(body_b)
        self.flexures.append(FlexureDef.create(body_a_, body_b_, attach_a_local, attach_b_local, spring_l0))

    def add_force(
            self,
            body: str,
            vector: ArrayLike,
            attach_local: ArrayLike | None = None,
    ) -> None:
        body_ = self._get_body(body)
        vector_ = as_vec3(vector)
        attach_local = as_vec3(np.zeros(3) if attach_local is None else attach_local)
        self.forces.append(ForceDef(body_.name, vector_, attach_local))
