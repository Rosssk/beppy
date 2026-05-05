from dataclasses import dataclass


@dataclass
class Body:
    name: str
    position0: tuple[float, float, float]
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angles: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class Flexure:
    a: str
    b: str
    attach_a: tuple[float, float, float]
    attach_b: tuple[float, float, float]


@dataclass
class Force:
    vector: tuple[float, float, float]
    attach: tuple[float, float, float] = (0.0, 0.0, 0.0)


class Model:
    def __init__(self):
        self.bodies: dict[str, Body] = {}
        self.flexures: list[Flexure] = []
        self.forces: list[Force] = []

    def add_body(self, name: str, position: tuple[float, float, float] = (0, 0, 0),
                 angles: tuple[float, float, float] = (0, 0, 0)) -> None:
        self.bodies[name] = Body(name, position, angles)

    def add_flexure(self, a: str, attach_a, b: str, attach_b) -> None:
        self.flexures.append(Flexure(a, b, attach_a, attach_b))

    def add_force(self, body: str, vector, attach=(0, 0, 0)) -> None:
        self.forces.append(Force(vector, attach))
