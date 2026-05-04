# pyright: strict

from rbm import PRBM
import numpy as np
from numpy import cos, sin, pi

def init_rbm():
    mm = 1e-3
    cm = 1e-2

    p = PRBM(3)

    p.add_body('A', (0, 0, 0))
    p.add_body('B', (0, 0, 15*mm))
    p.add_body('C', (0, 0, 30*mm))

    n = 3
    r = 14*mm

    for i in range(n):
        p.add_flexure('A', (r*cos(i/n*2*pi), r*sin(i/n*2*pi), 0),
                      'B', (r*cos((i + 1)/n*2*pi), r*sin((i + 1)/n*2*pi), 0))
        p.add_flexure('C', (r*cos((i - .5)/n*2*pi), r*sin((i - .5)/n*2*pi), 0),
                      'B', (r*cos((i + .5)/n*2*pi), r*sin((i + .5)/n*2*pi), 0))
    
    return p

def solve_module(force_vec, x0):
    mm = 1e-3
    cm = 1e-2
    t = 1e-3
    A = 1*pi*t**2
    E = 3650e6 # Pa
    I = 10*pi*t**4/2

    p = init_rbm()
    p.add_force('C', force_vec)
    p.solve_pose('BC', A, E, I, None, x0)
    
    shifted_center = p.bodies['B'].rotmat@np.array([0, 0, -3*cm]) + p.bodies['B'].position
    shifted_offset = p.bodies['B'].rotmat@np.array([0, 15*mm, -3*cm]) + p.bodies['B'].position
    return (shifted_center, shifted_offset)