# pyright: strict

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Vec3 = NDArray[np.float64]

def _check_str(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"'{name}' must be a str, got {type(value).__name__!r}")
    if not value.strip():
        raise ValueError(f"'{name}' must not be empty or whitespace-only")
    return value

def _as_vec3(v: ArrayLike) -> Vec3:
    arr: Vec3 = np.asarray(v, dtype=np.float64)

    if arr.shape != (3,):
        raise ValueError(f"Expected 3D vector, got shape {arr.shape}")

    return arr


class Force:
    def __init__(self, vector: ArrayLike, local: ArrayLike):
        self.vector: Vec3 = _as_vec3(vector)
        self.attach_local: Vec3 = _as_vec3(local)

class Body:
    def __init__(self, name: str, position: ArrayLike):
        self.name: str = _check_str(name, "name")
        self.position: Vec3 = _as_vec3(position)
        self.position0: Vec3 = self.position.copy()
        self.angles: Vec3 = np.zeros(3, dtype=np.float64)
        self.forces: list[Force] = []

class Flexure:
    def __init__(self, body_a: Body, attach_a_local: ArrayLike, body_b: Body, attach_b_local: ArrayLike):
        if body_a.name == body_b.name:
            raise ValueError(f"Cannot create flexure between bodies with the same name. {body_a.name!r} = {body_b.name!r}")

        self.body_a: Body = body_a
        self.attach_a_local: Vec3 = _as_vec3(attach_a_local)
        self.body_b: Body = body_b
        self.attach_b_local: Vec3 = _as_vec3(attach_b_local)


class Model:
    def __init__(self) -> None:
        self.bodies: dict[str, Body] = {}
        self.flexures: dict[str, Flexure] = {}

    def _get_body(self, name: str) -> Body:
        if name not in self.bodies:
            raise KeyError(f"No body named {name!r}. Existing bodies: {list(self.bodies)}")
        return self.bodies[name]

    def add_body(
        self,
        name: str,
        position: ArrayLike | None = None,
    ) -> None:
        if name in self.bodies:
            raise ValueError(f"A body named {name!r} already exists")

        position = np.zeros(3) if position is None else position
        self.bodies[name] = Body(name, position)

    def add_flexure(
        self,
        body_a: str,
        attach_a_local: ArrayLike,
        body_b: str,
        attach_b_local: ArrayLike,
    ) -> None:
        body_a_ = self._get_body(body_a)
        body_b_ = self._get_body(b)

        base = body_a + body_b
        count = sum(1 for key in self.flexures if key.startswith(base))
        name = f"{base}|{count}"

        self.flexures[name] = (Flexure(body_a_, attach_a_local, body_b_, attach_b_local))

    def add_force(
        self,
        body: str,
        vector: ArrayLike,
        attach_local: ArrayLike | None = None,
    ) -> None:
        body_ = self._get_body(body)
        attach_local = np.zeros(3) if attach_local is None else attach_local
        body_.forces.append(Force(vector, attach_local))
