from numpy import zeros, array, sin, cos, acos, dot, pi
from numpy.linalg import norm
from scipy.optimize import minimize

gamma = 0.85
kappa_theta = 2.65
cm = 1e-2
mm = 1e-3


def angle(a, b, atol=1e-6):  # angle between vector a and b
    if norm(a - b) < atol:
        return 0
    return acos(dot(a, b) / norm(a) / norm(b))


# rotation matrix for a list of angles in 3d, or one angle in 2d
def rotmat(angles):
    return RX(angles[0]) @ RY(angles[1]) @ RZ(angles[2])

def RX(t):  # 3d rotation matrix around x
    return array([[1, 0, 0], [0, cos(t), -sin(t)], [0, sin(t), cos(t)]])


def RY(t):  # 3d rotation matrix around y
    return array([[cos(t), 0, sin(t)], [0, 1, 0], [-sin(t), 0, cos(t)]])


def RZ(t):  # 3d rotation matrix around z
    return array([[cos(t), -sin(t), 0], [sin(t), cos(t), 0], [0, 0, 1]])


class Flexure:
    """
    Class containing the data of a PRBM flexure and the energy function

    Contains:
    - data, updated by the attached body.move()
    - energy(), returns the potential energy of the flexure
    """

    def __init__(self, bodyA, attachpoint_localA, bodyB, attachpoint_localB):
        self.bodyA = bodyA
        self.bodyB = bodyB

        # Flexure reference constants
        self.attachpoint_localA = array(attachpoint_localA)
        self.attachpoint_localB = array(attachpoint_localB)

        attachpoint_globalA0 = bodyA.position + bodyA.rotmat@self.attachpoint_localA
        attachpoint_globalB0 = bodyB.position + bodyB.rotmat@self.attachpoint_localB
        flexure_global0 = attachpoint_globalB0 - attachpoint_globalA0
        length0 = norm(flexure_global0)

        self.springlen0 = length0*gamma

        spring_unit0 = flexure_global0 / length0
        self.spring_unitA0 = bodyA.rotmat.T@spring_unit0
        self.spring_unitB0 = bodyB.rotmat.T@spring_unit0

        # Set during calculations but not read
        self.attachpoint_globalA = attachpoint_globalA0
        self.attachpoint_globalB = attachpoint_globalB0
        self.springpoint_globalA = attachpoint_globalA0 + flexure_global0 * (1 - gamma)/2
        self.springpoint_globalB = attachpoint_globalB0 + flexure_global0 * -(1 - gamma)/2

    def energy_func(self, free_bodies, bodies, A, E, I):
        if self.bodyA.name in free_bodies and self.bodyB.name in free_bodies:
            a = free_bodies.index(self.bodyA.name)
            b = free_bodies.index(self.bodyB.name)
            return lambda x : self.energy(x[a*6:a*6+3], x[a*6+3:a*6+6], x[b*6:b*6+3], x[b*6+3:b*6+6], A, E, I)
        elif self.bodyB.name in free_bodies:
            a = bodies[self.bodyA.name]
            b = free_bodies.index(self.bodyB.name)
            return lambda x : self.energy(a.position, a.angles, x[b*6:b*6+3], x[b*6+3:b*6+6], A, E, I)
        elif self.bodyB.name in free_bodies:
            a = free_bodies.index(self.bodyA.name)
            b = bodies[self.bodyB.name]
            return lambda x : self.energy(x[a*6:a*6+3], x[a*6+3:a*6+6], b.position, b.angles, A, E, I)
        else:
            a = bodies[self.bodyA.name]
            b = bodies[self.bodyB.name]
            return lambda x : self.energy(a.position, a.angles, b.position, b.angles, A, E, I)    

    def energy(self, xA, thetaA, xB, thetaB, A, E, I):
        # xA and xB are the positions of body A and B
        # thetaA and thetaB are the rotations of A and B w.r.t. the global coordinate axes
        kappa = gamma * kappa_theta * E * I / self.springlen0
        k = E * A / self.springlen0

        rotmat_A = rotmat(thetaA)
        rotmat_B = rotmat(thetaB)

        attachpoint_globalA = rotmat_A@self.attachpoint_localA + xA
        attachpoint_globalB = rotmat_B@self.attachpoint_localB + xB
        flexure_global = attachpoint_globalB - attachpoint_globalA
        length = norm(flexure_global)

        # Calculate energy caused by compression/stretch of flexure
        u = length*gamma - self.springlen0   
        energy_AB = k * u**2 / 2

        # Calculate energy caused by the angle change of the flexure attachements
        spring_unit = flexure_global / length
        spring_unitA = rotmat_A.T@spring_unit
        spring_unitB = rotmat_B.T@spring_unit

        thetaA = angle(spring_unitA, self.spring_unitA0)
        thetaB = angle(spring_unitB, self.spring_unitB0)

        energyA = kappa * thetaA**2 / 2
        energyB = kappa * thetaB**2 / 2

        return energy_AB + energyA + energyB

    def update(self):
        # Update the flexure positions/rotations based on the body positions/rotations
        attachpoint_globalA = self.bodyA.rotmat@self.attachpoint_localA + self.bodyA.position
        attachpoint_globalB = self.bodyB.rotmat@self.attachpoint_localB + self.bodyB.position
        flexure_global = attachpoint_globalB - attachpoint_globalA

        self.attachpoint_globalA = attachpoint_globalA
        self.attachpoint_globalB = attachpoint_globalB
        self.springpoint_globalA = self.attachpoint_globalA + flexure_global * (1 - gamma)/2
        self.springpoint_globalB = self.attachpoint_globalB + flexure_global * -(1 - gamma)/2
        


class Body:
    """
    Class containing data and functions of a PRBM rigidbody

    Contains:
    - data, updated by self.move()
    - move(), updates the attached flexures
    - energy(), calculates and returns energy due to external forces
    """

    def __init__(self, name, position0):
        self.name = name
        self.position0 = array(position0)
        self.position = array(position0)

        self.angles0 = zeros(3)
        self.angles = zeros(3)

        self.rotmat = rotmat(self.angles)

        self.flexures = []
        self.which = []

        self.forces = []
        self.points = [zeros(3)]

    def move(self, position, angles=None):
        # move a body to a pose
        self.position = array(position)
        if angles is None:
            self.angles = self.angles0
        else:
            self.angles = angles
    
        self.rotmat = rotmat(self.angles)

        for flexure in self.flexures:
            flexure.update()

    def energy_func(self, free_bodies):
        if self.name not in free_bodies:
            return lambda x : 0

        i = free_bodies.index(self.name)
        return lambda x : self.energy(x[i*6:i*6+3], x[i*6+3:i*6+6])

    def energy(self, pos, angles):
        # calculate the potential energy of a body due to external forces
        energy = 0
        for attachpoint_local, vector, attachpoint_global0 in self.forces:
            attachpoint_global = rotmat(angles) @ attachpoint_local + pos
            energy -= dot(attachpoint_global, vector)

        return energy

    def line(self):
        return [self.rotmat @ array(point) + self.position for point in self.points]


class PRBM:
    """
    Contains useful functions for setting up a pseudo rigid body model.
    These functions are just for convenience: it might be easier to set up something manual if you have a certain use case.
    """
    def __init__(self, dim):
        self.dim = dim
        self.bodies = {}
        self.bodynames = []
        self.flexures = {}
        self.flexurenames = []

    def add_body(self, name, position=None):
        self.bodies[name] = Body(name, position)
        self.bodynames.append(name)
        self.bodies[name].points.append(zeros(self.dim))

    def add_flexure(self, bodynameA, attachpoint_localA, bodynameB, attachpoint_localB):
        bodyA = self.bodies[bodynameA]
        bodyB = self.bodies[bodynameB]

        name = bodynameA + bodynameB
        occurrence = 0
        for othername in self.flexurenames:
            if name in othername:
                occurrence += 1
        name = name + str(occurrence)
        self.flexurenames.append(name)

        flexure = Flexure(bodyA, attachpoint_localA, bodyB, attachpoint_localB)
        self.flexures[name] = flexure

        bodyA.flexures.append(flexure)
        bodyA.which.append(True)
        bodyA.points.append(attachpoint_localA)

        bodyB.flexures.append(flexure)
        bodyB.which.append(False)
        bodyB.points.append(attachpoint_localB)

    def move(self, bodyname, position, angles=None):
        # move a body
        if angles is None:
            self.bodies[bodyname].move(position)
        else:
            self.bodies[bodyname].move(position, angles)

    def print(self, A=None, E=None, I=None):
        for bodyname in self.bodynames:
            print("Body", bodyname)
            print("Position", self.bodies[bodyname].position)
            print("Angles", self.bodies[bodyname].angles)
            print("Energy", self.bodies[bodyname].energy())
            print()
        if A is not None and E is not None and I is not None:
            for flexurename in self.flexurenames:
                print(
                    "Flexure", flexurename, self.flexures[flexurename].energy(A, E, I)
                )

    def show(self, args=None):
        # plot the prbm
        lines = [
            array(
                [
                    flexure.attachpoint_globalA,
                    flexure.springpoint_globalA,
                    flexure.springpoint_globalB,
                    flexure.attachpoint_globalB,
                ]
            ).T
            for flexure in self.flexures.values()
        ]

        lines2 = [array(body.line()).T for body in self.bodies.values()]

        if self.dim == 2:
            import matplotlib

            matplotlib.use("TkAgg")
            import matplotlib.pyplot as mp

            for line2, name2 in zip(lines2, self.bodynames):
                mp.plot(line2[0], line2[1], "-", label=name2)

            for line, name in zip(lines, self.flexurenames):
                mp.plot(line[0], line[1], "o-", label=name)

            mp.xlabel("x (m)")
            mp.ylabel("y (m)")
            mp.legend()
            mp.axis("equal")
            mp.show()

        if self.dim == 3:
            import plotly.graph_objects as pg

            fig = pg.Figure()

            for line2, name2 in zip(lines2, self.bodynames):
                fig.add_trace(
                    pg.Scatter3d(
                        x=line2[0], y=line2[1], z=line2[2], mode="lines", name=name2
                    )
                )

            for line, name in zip(lines, self.flexurenames):
                fig.add_trace(
                    pg.Scatter3d(
                        x=line[0], y=line[1], z=line[2], mode="lines+markers", name=name
                    )
                )

            fig.update_layout(scene=dict(aspectmode="cube"))
            fig.show()

    def energy_funcs(self, free_bodies, A, E, I):
        flexure_energy_funcs = [
            flexure.energy_func(free_bodies, self.bodies, A, E, I) for flexure in self.flexures.values()
        ]

        body_energy_funcs = [
            body.energy_func(free_bodies) for body in self.bodies.values()
        ]

        return flexure_energy_funcs + body_energy_funcs

    # NOTE: added forces and torques are fixed in the global reference frame
    def add_force(self, bodyname, vector, attachpoint_local=None):
        body = self.bodies[bodyname]
        if attachpoint_local is None:
            attachpoint_local = zeros(self.dim)
        else:
            body.points.append(attachpoint_local)

        attachpoint_global0 = rotmat(body.angles)@attachpoint_local + body.position
        body.forces.append((attachpoint_local, vector, attachpoint_global0))

    def add_torque_2D(self, bodyname, torque):
        self.add_force(bodyname, (0, -torque))
        self.add_force(bodyname, (0, torque), (0, 1))

    def add_torque_x(self, bodyname, Mx):
        self.add_force(bodyname, (0, 0, -Mx))
        self.add_force(bodyname, (0, 0, Mx), (0, 1, 0))

    def add_torque_y(self, bodyname, My):
        self.add_force(bodyname, (-My, 0, 0))
        self.add_force(bodyname, (My, 0, 0), (0, 0, 1))

    def add_torque_z(self, bodyname, Mz):
        self.add_force(bodyname, (0, -Mz, 0))
        self.add_force(bodyname, (0, Mz, 0), (1, 0, 0))

    def add_torque(self, bodyname, torque):
        if self.dim == 2:
            self.add_torque_2D(bodyname, torque)
        else:
            self.add_torque_x(bodyname, torque[0])
            self.add_torque_y(bodyname, torque[1])
            self.add_torque_z(bodyname, torque[2])

    def solve_pose(self, bodynames, A, E, I, method=None, x0=None, options=None):
        # solve pose of all bodies in the list bodynames
        # using minimization of the total energy
        free_bodies = []
        x0_def = []
        for bodyname in bodynames:
            body = self.bodies[bodyname]
            free_bodies.append(body.name)
            x0_def = x0_def + list(body.position) + list(body.angles)

        if x0 is None:
            x0 = x0_def
        elif len(x0) != len(x0_def):
            raise ValueError("Inconsistent dimensions for initial guess")

        energy_funcs = self.energy_funcs(free_bodies, A, E, I)

        def optimize_function(x):
            s = 0
            for f in energy_funcs:
                s += f(x)
            return s

        self.solution = minimize(optimize_function, x0, method=method, options=options)

        x = self.solution.x
        for i in range(len(free_bodies)):
            pos = x[i*6:i*6+3]
            rot = x[i*6+3:i*6+6]
            self.bodies[free_bodies[i]].move(pos, rot)


def main():
    p = PRBM(3)
    p.add_body("A", (0, 0, 0))
    p.add_body("B", (0, 0, 1))
    p.add_flexure("A", (0, 0, 0), "B", (0, 0, 0))

    flex = p.flexures["AB0"]

    A = 1e-3
    E = 250e6
    t = 1e-3
    I = pi * t**4 / 2

    p.move("B", (0, 0, 5))

    p.show()


if __name__ == "__main__":
    main()
