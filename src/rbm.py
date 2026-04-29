#
# (Modified) from https://github.com/dmsyaifoel/pyRBM/tree/main
#

from numpy import zeros, array, sin, cos, acos, dot
from numpy.linalg import norm
from scipy.optimize import minimize

gamma = .85
kappa_theta = 2.65
cm = 1e-2
mm = 1e-3

def angle(a, b, atol=1e-6):  # angle between vector a and b
  if norm(a - b) < atol:
    return 0
  return acos(dot(a, b)/norm(a)/norm(b))

# 2d rotation matrix for angle t
def R(t):
  return array([[cos(t), -sin(t)], [sin(t), cos(t)]])

# rotation matrix for a list of angles in 3d, or one angle in 2d
def rotmat(angles, dim):
  if dim == 2: 
    return R(angles)
  elif dim == 3: 
    return RX(angles[0])@RY(angles[1])@RZ(angles[2])
  else: 
    raise ValueError("Unsupported dimension")

def RX(t): # 3d rotation matrix around x
  return array([[1, 0, 0], [0, cos(t), -sin(t)], [0, sin(t), cos(t)]])

def RY(t): # 3d rotation matrix around y
  return array([[cos(t), 0, sin(t)], [0, 1, 0], [-sin(t), 0, cos(t)]])

def RZ(t): # 3d rotation matrix around z
  return array([[cos(t), -sin(t), 0], [sin(t), cos(t), 0], [0, 0, 1]])

class Flexure:
  '''
  Class containing the data of a PRBM flexure and the energy function

  Contains:
  - data, updated by the attached body.move()
  - energy(), returns the potential energy of the flexure
  '''
  def __init__(self, bodyA, attachpoint_localA, bodyB, attachpoint_localB):
    # constant
    self.attachpoint_localA = array(attachpoint_localA)
    self.attachpoint_localB = array(attachpoint_localB)

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

  def energy(self, A, E, I):
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
  def __init__(self, name, position0, dim):

    self.name = name
    self.position0 = array(position0)
    self.position = array(position0)
    self.dim = dim

    if self.dim == 2:
      self.angles0 = 0
      self.angles = 0

    if self.dim == 3:
      self.angles0 = zeros(3)
      self.angles = zeros(3)

    self.rotmat = rotmat(self.angles, self.dim)

    self.flexures = []
    self.which = []

    self.forces = []
    self.points = [zeros(dim)]

  def move(self, position, angles=None):
    # move a body to a pose
    self.position = array(position)
    if angles is None: self.angles = self.angles0
    else: self.angles = angles

    self.rotmat = rotmat(self.angles, self.dim)

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

  def energy(self):
    # calculate the potential energy of a body due to external forces
    energy = 0
    for attachpoint_local, vector in self.forces:
      attachpoint_global = self.rotmat@attachpoint_local + self.position
      energy -= dot(attachpoint_global, vector)
    return energy

  def line(self):
    return [self.rotmat@array(point) + self.position for point in self.points]


class PRBM:
  '''
  Contains useful functions for setting up a pseudo rigid body model.
  These functions are just for convenience: it might be easier to set up something manual if you have a certain use case.
  '''
  def __init__(self, dim):
    self.dim = dim
    self.bodies = {}
    self.bodynames = []
    self.flexures = {}
    self.flexurenames = []

  def add_body(self, name, position=None):
    self.bodies[name] = Body(name, position, self.dim)
    self.bodynames.append(name)
    self.bodies[name].points.append(zeros(self.dim))

  def add_flexure(self, bodynameA, attachpoint_localA, bodynameB, attachpoint_localB):
    bodyA = self.bodies[bodynameA]
    bodyB = self.bodies[bodynameB]

    name = bodynameA + bodynameB
    occurrence = 0
    for othername in self.flexurenames:
      if name in othername: occurrence += 1
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
      print('Body', bodyname)
      print('Position', self.bodies[bodyname].position)
      print('Angles', self.bodies[bodyname].angles)
      print('Energy', self.bodies[bodyname].energy())
      print()
    if A is not None and E is not None and I is not None:
      for flexurename in self.flexurenames:
        print('Flexure', flexurename, self.flexures[flexurename].energy(A, E, I))

  def show(self, args=None):
    # plot the prbm

    lines = [array([flexure.attachpoint_globalA,
                       flexure.springpoint_globalA,
                       flexure.springpoint_globalB,
                       flexure.attachpoint_globalB]).T for flexure in self.flexures.values()]

    lines2 = [array(body.line()).T for body in self.bodies.values()]

    if self.dim == 2:
      import matplotlib
      matplotlib.use('TkAgg')
      import matplotlib.pyplot as mp

      for line2, name2 in zip(lines2, self.bodynames):
        mp.plot(line2[0], line2[1], '-', label=name2)

      for line, name in zip(lines, self.flexurenames):
        mp.plot(line[0], line[1], 'o-', label=name)

      mp.xlabel('x (m)')
      mp.ylabel('y (m)')
      mp.legend()
      mp.axis('equal')
      mp.show()

    if self.dim == 3:
      import plotly.graph_objects as pg
      fig = pg.Figure()

      for line2, name2 in zip(lines2, self.bodynames):
        fig.add_trace(pg.Scatter3d(x=line2[0], y=line2[1], z=line2[2], mode='lines', name=name2))

      for line, name in zip(lines, self.flexurenames):
        fig.add_trace(pg.Scatter3d(x=line[0], y=line[1], z=line[2], mode='lines+markers', name=name))

      fig.update_layout(scene=dict(aspectmode='cube'))
      fig.show()

  def energy(self, A, E, I):
    # calculate and return the potential energy of the prbm
    flexure_energies = [flexure.energy(A, E, I) for flexure in self.flexures.values()]
    body_energies = [body.energy() for body in self.bodies.values()]
    return sum(flexure_energies) + sum(body_energies)

  # NOTE: added forces and torques are fixed in the global reference frame
  def add_force(self, bodyname, vector, attachpoint_local=None):
    body = self.bodies[bodyname]
    if attachpoint_local is None:
      attachpoint_local = zeros(self.dim)
    else:
      body.points.append(attachpoint_local)
    body.forces.append((attachpoint_local, vector))



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
    if self.dim == 2: self.add_torque_2D(bodyname, torque)
    else:
      self.add_torque_x(bodyname, torque[0])
      self.add_torque_y(bodyname, torque[1])
      self.add_torque_z(bodyname, torque[2])

  def solve_pose(self, bodynames, A, E, I, method=None, x0=None):
    # solve pose of all bodies in the list bodynames
    # using minimization of the total energy

    free_bodies = []
    x0_def = []
    for bodyname in bodynames:
      body = self.bodies[bodyname]
      free_bodies.append(body)
      if self.dim == 2: x0_def = x0_def + list(body.position) + [body.angles]
      if self.dim == 3: x0_def = x0_def + list(body.position) + list(body.angles)

    if x0 is None:
      x0 = x0_def
    elif len(x0) != len(x0_def):
      raise ValueError("Inconsistent dimensions for initial guess")

    n_free_bodies = len(free_bodies)

    def optimize_function(x):
      for i in range(n_free_bodies):
        if self.dim == 2:
          s1 = 3*i
          s2 = 3*i + 2
          p = x[s1:s2]
          a = x[s2]
        elif self.dim == 3:
          s1 = 6*i
          s2 = 6*i + 3
          s3 = 6*i + 6
          p = x[s1:s2]
          a = x[s2:s3]
        else:
          raise ValueError("Unsupported dimension")
        free_bodies[i].move(p, a)
      return self.energy(A, E, I)

    self.solution = minimize(optimize_function, x0, method=method)

    x = self.solution.x
    optimize_function(x) # move bodies to the optimal pose