from typing import NamedTuple, Literal
import numpy as np
from numpy import array, sin, cos, zeros
from numpy.linalg import norm
from numpy.typing import NDArray

gamma = 0.85
kappa_theta = 2.65

Vec3 = tuple[float, float, float] | np.ndarray[tuple[Literal[3]], np.dtype[np.floating]]
Number = float | np.floating


# rotation matrix for a list of angles in 3d, or one angle in 2d
def rotmat(angles):
    rx = array([[1, 0, 0], [0, cos(angles), -sin(angles)], [0, sin(angles), cos(angles)]])
    ry = array([[cos(angles), 0, sin(angles)], [0, 1, 0], [-sin(angles), 0, cos(angles)]])
    rz = array([[cos(angles), -sin(angles), 0], [sin(angles), cos(angles), 0], [0, 0, 1]])
    return rx @ ry @ rz


class Flexure(NamedTuple):
    BODY_A: str
    BODY_B: str
    ATTACH_POINT_A_LOCAL: NDArray
    ATTACH_POINT_B_LOCAL: NDArray
    SPRING_LEN_0: Number
    SPRING_A_DIR_0: NDArray
    SPRING_B_DIR_0: NDArray

    def show(self):
        print("hey!")


class Body(NamedTuple):
    NAME: str
    POSITION_0: NDArray

    points: list[NDArray]
    position: NDArray
    angles: NDArray


class PRBM:
    """
    Contains useful functions for setting up a pseudo rigid body model.
    These functions are just for convenience: it might be easier to set up something manual if you have a certain use case.
    """

    def __init__(self):
        self.bodies: dict[str, Body] = {}
        self.flexures: dict[str, Flexure] = {}

    def add_body(self, name, position=None):
        if position is None:
            position = zeros(3)

        self.bodies[name] = Body(name, position, [zeros(3)], position, zeros(3))

    def add_flexure(self, body_name_a: str, attach_point_a_local: Vec3, body_name_b: str, attach_point_b_local: Vec3):
        attach_point_a_local = array(attach_point_a_local)
        attach_point_b_local = array(attach_point_b_local)

        body_a = self.bodies[body_name_a]
        body_b = self.bodies[body_name_a]

        # Get unique name for flexure
        name = body_name_a + body_name_b
        occurrence = 0
        for other_name in self.flexures.keys():
            if name in other_name:
                occurrence += 1
        name = name + str(occurrence)

        # Calculate initial values
        rotmat_a = rotmat(body_a.angles)
        rotmat_b = rotmat(body_b.angles)
        attach_point_a_global = body_a.position + rotmat_a @ attach_point_a_local
        attach_point_b_global = body_b.position + rotmat_b @ attach_point_b_local

        flexure_vector_global = attach_point_b_global - attach_point_a_global
        length0 = norm(flexure_vector_global)

        spring_len0 = length0 * gamma

        spring_unit0 = flexure_vector_global / length0
        spring_unit_a0 = rotmat_a.T @ spring_unit0
        spring_unit_b0 = rotmat_b.T @ spring_unit0

        flexure = Flexure(body_a.NAME, body_b.NAME, attach_point_a_local, attach_point_b_local, spring_len0,
                          spring_unit_a0, spring_unit_b0)

        self.flexures.
