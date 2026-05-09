import numpy
import sympy as sp
from sympy import sin, cos, pi, Matrix

r = 1
gamma = 0.2
h = 1

mu_rigid = 0.5
alpha_rigid = 0.2
alpha_prism = 1 - alpha_rigid * 2

def norm(vec):
    return sp.sqrt((vec * vec.T)[0])

def rotmat(body_state):
    """
    Return a rotation matrix, takes the 1x6 body state
    """
    phi, theta, psi = body_state[3], body_state[4], body_state[5]
    R_X = Matrix([[1,0,0],[0,cos(phi),-sin(phi)],[0,sin(phi),cos(phi)]])
    R_Y = Matrix([[cos(theta),0,sin(theta)],[0,1,0],[-sin(theta),0,cos(theta)]])
    R_Z = Matrix([[cos(psi),-sin(psi),0],[sin(psi),cos(psi),0],[0,0,1]])
    return R_Z * R_Y * R_X

def global_pos(body_state, local_vec):
    """
    Transform local coordinate vector into global coordinate
    """
    body_pos = Matrix([body_state[:3]])
    return body_pos + local_vec*rotmat(body_state)

# Body states
x2, y2, z2, psi2, theta2, phi2 = sp.symbols('x2 y2 z2 psi2, theta2, phi2')
x3, y3, z3, psi3, theta3, phi3 = sp.symbols('x3 y3 z3 psi3, theta3, phi3')
q_b1 = q_b1_0 = Matrix([[0,0,0,0,0,0]])
q_b2_0 = Matrix([[0,0,h/2,0,0,0]])
q_b2 = Matrix([[x2, y2, z2, psi2, theta2, phi2]])
q_b3_0 = Matrix([[0,0,h,0,0,0]])
q_b3 = Matrix([[x3, y3, z3, psi3, theta3, phi3]])

# List of (attachment_point_1, attachment_point_2)
n = 3
beams = []
for i in range(n):
    attach_local_1 = Matrix([[r*cos(i/n*2*pi), r*sin(i/n*2*pi), 0]])
    attach_local_2 = Matrix([[r*cos((i + 1)/n*2*pi), r*sin((i + 1)/n*2*pi), 0]])
    attach_global_a0 = global_pos(q_b1_0, attach_local_1)
    attach_global_b0 = global_pos(q_b2_0, attach_local_2)

    beam0 = attach_global_b0 - attach_global_a0
    length0 = sp.trigsimp(norm(beam0))

    start = beam0 / length0 * length0 * gamma # add mass location


    print(length0.evalf())









