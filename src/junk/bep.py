from forces1 import ForceGenerator
import numpy as np
from numpy import cos, sin, pi
from rbm import PRBM
import matplotlib.pyplot as plt
from tqdm import tqdm

side_length = 1e-2
t_height = np.sqrt(3)/2 * side_length
mm = 1e-3 # m
cm = 1e-2 # m

depth = 10*mm

def init_rbm():
    p = PRBM(3)

    p.add_body('A', (0, 0, 0))
    p.add_body('B', (0, 0, 0.5*depth))
    p.add_body('C', (0, 0, depth))

    n = 3
    r = 8.773827*mm

    for i in range(n):
        p.add_flexure('A', (r*cos(i/n*2*pi), r*sin(i/n*2*pi), 0),
                      'B', (r*cos((i + 1)/n*2*pi), r*sin((i + 1)/n*2*pi), 0))
        p.add_flexure('C', (r*cos((i - .5)/n*2*pi), r*sin((i - .5)/n*2*pi), 0),
                      'B', (r*cos((i + .5)/n*2*pi), r*sin((i + .5)/n*2*pi), 0))
    
    return p

_prev_guess = None
def solve_module(force_vec):
    global _prev_guess
    t = mm
    A = pi*t**2
    E = 850e6 # Pa
    I = 0.1*pi*t**4/2

    p = init_rbm()
    p.add_force('C', force_vec)
    if force_vec != [0, 0, 0]:
        p.solve_pose('BC', A, E, I, x0=_prev_guess)
        _prev_guess = p.solution.x
    
    l = 4*mm
    c1 = p.bodies['B'].rotmat@np.array([0, l, -0.5*depth]) + p.bodies['B'].position
    c2 = p.bodies['B'].rotmat@np.array([-l*cos(pi/6), -l*sin(pi/6), -0.5*depth]) + p.bodies['B'].position
    c3 = p.bodies['B'].rotmat@np.array([l*cos(pi/6), -l*sin(pi/6), -0.5*depth]) + p.bodies['B'].position
    return (c1, c2, c3)


def main():
    print("test")


if __name__ == "__main__":
    main()
            

