# pyright: strict

from __future__ import annotations
import numpy as np
from typing import List, Literal

from numpy.linalg import norm
from numpy import dot, sin, cos, acos
from numpy.typing import NDArray
from scipy.optimize import minimize

type Matrix[N: int, M: int] = np.ndarray[
    tuple[N, M],
    np.dtype[np.floating | np.int_],
]

type Mat3 = Matrix[Literal[3], Literal[3]]
type Vec3 = Matrix[Literal[3], Literal[1]]

gamma = .85
kappa_theta = 2.65
cm = 1e-2
mm = 1e-3

def angle(a: Vec3, b: Vec3, atol: float = 1e-6):  # angle between vector a and b
    if norm(a - b) < atol:
        return 0
    return acos(dot(a, b)/norm(a)/norm(b))

def rotmat(angles: Vec3) -> Mat3:
    return _RX(angles[0])@_RY(angles[1])@_RZ(angles[2])

def _RX(t: float) -> Mat3: # 3d rotation matrix around x
    return np.array([[1, 0, 0], [0, cos(t), -sin(t)], [0, sin(t), cos(t)]])

def _RY(t: float) -> Mat3: # 3d rotation matrix around y
    return np.array([[cos(t), 0, sin(t)], [0, 1, 0], [-sin(t), 0, cos(t)]])

def _RZ(t: float) -> Mat3: # 3d rotation matrix around z
    return np.array([[cos(t), -sin(t), 0], [sin(t), cos(t), 0], [0, 0, 1]])

class Flexure:
    '''
    Class containing the data of a PRBM flexure and the energy function

    Contains:
    - data, updated by the attached body.move()
    - energy(), returns the potential energy of the flexure
    '''
    def __init__(self, bodyA: Body, attachpoint_localA: Vec3, bodyB: Body, attachpoint_localB: Vec3):
        # constant
        self.attachpoint_localA = attachpoint_localA
        self.attachpoint_localB = attachpoint_localB

        # do not need to be remembered besides len0, which is constant
        attachpoint_globalA0 = bodyA.position + self.attachpoint_localA
        attachpoint_globalB0 = bodyB.position + self.attachpoint_localB

        vector_global0 = attachpoint_globalB0 - attachpoint_globalA0
        self.len0 = norm(vector_global0)
        unitvector_global0 = vector_global0/self.len0

        springpoint_globalA0 = attachpoint_globalA0 + (1 - gamma)/2*vector_global0
        springpoint_globalB0 = attachpoint_globalB0 - (1 - gamma)/2*vector_global0

        # constant
        self.springlen0 = norm(springpoint_globalB0 - springpoint_globalA0)

        # variable, updated by Body.move()
        self.attachpoint_globalA = attachpoint_globalA0
        self.attachpoint_globalB = attachpoint_globalB0

        self.springpoint_globalA = springpoint_globalA0
        self.springpoint_globalB = springpoint_globalB0

        # constant
        self.springpoint_localA = springpoint_globalA0 - bodyA.position
        self.springpoint_localB = springpoint_globalB0 - bodyB.position

        # variable, updated by Body.move()
        self.unitvector_globalA = unitvector_global0
        self.unitvector_globalB = unitvector_global0

        # constant
        self.unitvector_localA = unitvector_global0
        self.unitvector_localB = unitvector_global0

    def energy(self, A: float, E: float, I: float):
        # calculate and return energy of the flexure
        kappa = gamma*kappa_theta*E*I/self.len0
        k = E*A/self.springlen0
        vector_global = self.springpoint_globalB - self.springpoint_globalA
        springlen = norm(vector_global)
        unitvector_global = vector_global/springlen

        thetaA = angle(self.unitvector_globalA, unitvector_global)
        thetaB = angle(self.unitvector_globalB, unitvector_global)

        energyA = kappa*thetaA**2/2
        energyB = kappa*thetaB**2/2

        deltaS = springlen - self.springlen0
        energyAB = k*deltaS**2/2

        energy = energyA + energyB + energyAB

        return energy

class Body:
    '''
    Class containing data and functions of a PRBM rigidbody

    Contains:
    - data, updated by self.move()
    - move(), updates the attached flexures
    - energy(), calculates and returns energy due to external forces
    '''
    def __init__(self, name: str, position0: Vec3, dim: Literal[2, 3] = 3):
        self.name = name
        self.position0 = position0
        self.position = self.position0
        self.angles0 = np.array([0, 0, 0])
        self.angles = self.angles0
        self.dim = dim

        self.rotmat = rotmat(self.angles)

        self.flexures: list[Flexure] = []
        self.which: list[bool] = []

        self.forces: List[tuple[Vec3, Vec3]] = []
        self.points = []

    def move(self, position: Vec3, angles: Vec3 | None = None):
        self.position = position
        if angles is None:
            self.angles = self.angles0
        else:
            self.angles = angles

        self.rotmat = rotmat(self.angles)

        # then update all flexure attached
        for flexure, which in zip(self.flexures, self.which):
            if which:
                flexure.attachpoint_globalA = self.rotmat@flexure.attachpoint_localA + self.position
                flexure.springpoint_globalA = self.rotmat@flexure.springpoint_localA + self.position
                flexure.unitvector_globalA = self.rotmat@flexure.unitvector_localA

            else:
                flexure.attachpoint_globalB = self.rotmat@flexure.attachpoint_localB + self.position
                flexure.springpoint_globalB = self.rotmat@flexure.springpoint_localB + self.position
                flexure.unitvector_globalB = self.rotmat@flexure.unitvector_localB

    def energy(self) -> float:
        # calculate the potential energy of a body due to external forces
        energy = 0
        for attachpoint_local, vector in self.forces:
            attachpoint_global = self.rotmat@attachpoint_local + self.position
            energy -= dot(attachpoint_global, vector)
        return energy

class PRBM:
    '''
    Contains useful functions for setting up a pseudo rigid body model.
    These functions are just for convenience: it might be easier to set up something manual if you have a certain use case.
    '''
    def __init__(self):
        self.bodies: dict[str, Body] = {}
        self.flexures: dict[str, Flexure] = {}

    def add_body(self, name: str, position: Vec3 | None =None):
        if position is None:
            position = np.array([0, 0, 0])

        self.bodies[name] = Body(name, position)


    def add_flexure(self, bodynameA: str, attachpoint_localA: Vec3, bodynameB: str, attachpoint_localB: Vec3):
        bodyA = self.bodies[bodynameA]
        bodyB = self.bodies[bodynameB]

        name = bodynameA + bodynameB
        occurrence = 0
        for othername in self.flexures.keys():
            if name in othername: occurrence += 1

        name = name + str(occurrence)

        flexure = Flexure(bodyA, attachpoint_localA, bodyB, attachpoint_localB)
        self.flexures[name] = flexure

        bodyA.flexures.append(flexure)
        bodyA.which.append(True)

        bodyB.flexures.append(flexure)
        bodyB.which.append(False)

    def move(self, bodyname: str, position: Vec3, angles: Vec3 | None = None):
        # move a body

        if angles is None:
            self.bodies[bodyname].move(position)
        else:
            self.bodies[bodyname].move(position, angles)

    def print(self, A: float | None = None, E: float | None = None, I: float | None = None):
        for (bodyname, body) in self.bodies.items():
            print('Body', bodyname)
            print('Position', body.position)
            print('Angles', body.angles)
            print('Energy', body.energy())
            print()

        if A is not None and E is not None and I is not None:
            for (flexurename, flexure) in self.flexures.items():
                print('Flexure', flexurename, flexure.energy(A, E, I))

    def energy(self, A: float, E: float, I: float):
        # calculate and return the potential energy of the prbm
        flexure_energies = [flexure.energy(A, E, I) for flexure in self.flexures.values()]
        body_energies = [body.energy() for body in self.bodies.values()]
        return sum(flexure_energies) + sum(body_energies)

    # NOTE: added forces and torques are fixed in the global reference frame
    def add_force(self, bodyname: str, vector: Vec3, attachpoint_local: Vec3 | None = None):
        body = self.bodies[bodyname]
        if attachpoint_local is None:
            attachpoint_local = np.array([0, 0, 0])

        body.forces.append((attachpoint_local, vector))

    def add_torque_x(self, bodyname: str, Mx: float):
        self.add_force(bodyname, np.array([0, 0, -Mx]))
        self.add_force(bodyname, np.array([0, 0, Mx]), np.array([0, 1, 0]))

    def add_torque_y(self, bodyname: str, My: float):
        self.add_force(bodyname, np.array([-My, 0, 0]))
        self.add_force(bodyname, np.array([My, 0, 0]), np.array([0, 0, 1]))

    def add_torque_z(self, bodyname: str, Mz: float):
        self.add_force(bodyname, np.array([0, -Mz, 0]))
        self.add_force(bodyname, np.array([0, Mz, 0]), np.array([1, 0, 0]))

    def add_torque(self, bodyname: str, torque: Vec3):
        self.add_torque_x(bodyname, torque[0])
        self.add_torque_y(bodyname, torque[1])
        self.add_torque_z(bodyname, torque[2])

    def solve_pose(self, bodynames: list[str], A: float, E: float, I: float, method: None = None):
        # solve pose of all bodies in the list bodynames
        # using minimization of the total energy

        free_bodies: list[Body] = []

        x0: NDArray[np.floating] = np.array([])

        for bodyname in bodynames:
            body = self.bodies[bodyname]
            free_bodies.append(body)
            x0 = np.append(x0, body.position)
            x0 = np.append(x0, body.angles)

        n_free_bodies = len(free_bodies)

        def optimize_function(x: NDArray[np.floating]):
            for i in range(n_free_bodies):
                s1 = 6*i
                s2 = 6*i + 3
                s3 = 6*i + 6
                p = x[s1:s2]
                a = x[s2:s3]
                free_bodies[i].move(p, a)
            return self.energy(A, E, I)

        self.solution = minimize(optimize_function, x0, method=method)

        x = self.solution.x
        optimize_function(x) # move bodies to the optimal pose
