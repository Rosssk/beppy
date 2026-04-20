from forcegen import ForceGenerator
import numpy as np
from numpy import cos, sin, pi
from rbm import PRBM
import matplotlib.pyplot as plt
from tqdm import tqdm

side_length = 1e-2
t_height = np.sqrt(3)/2 * side_length
mm = 1e-3 # m
cm = 1e-2 # m

def init_rbm():
    p = PRBM(3)

    p.add_body('A', (0, 0, 0))
    p.add_body('B', (0, 0, 15*mm))
    p.add_body('C', (0, 0, 3*cm))

    n = 3
    r = 14*mm

    for i in range(n):
        p.add_flexure('A', (r*cos(i/n*2*pi), r*sin(i/n*2*pi), 0),
                      'B', (r*cos((i + 1)/n*2*pi), r*sin((i + 1)/n*2*pi), 0))
        p.add_flexure('C', (r*cos((i - .5)/n*2*pi), r*sin((i - .5)/n*2*pi), 0),
                      'B', (r*cos((i + .5)/n*2*pi), r*sin((i + .5)/n*2*pi), 0))
    
    return p

def solve_module(force_vec):
    t = mm
    A = pi*t**2
    E = 1650e6 # Pa
    I = pi*t**4/2

    p = init_rbm()
    p.add_force('C', force_vec)
    p.solve_pose('BC', A, E, I)
    
    shifted_center = p.bodies['B'].rotmat@np.array([0, 0, -3*cm]) + p.bodies['B'].position
    shifted_offset = p.bodies['B'].rotmat@np.array([0, 15*mm, -3*cm]) + p.bodies['B'].position
    return (shifted_center, shifted_offset)

def get_center(i, j):
    # Get triangle center coordinates from indices
    x = i * side_length/2
    if (i + j % 2 == 0):
        return (x, (1/3 + j)*t_height)
    else:
        return (x, (2/3 + j)*t_height)


def main():
    forceg = ForceGenerator(0.6, 0.0075, 0.025, 1, 0.05, 80)
    grid_size = 15
    t_area = 0.5 * t_height * side_length

    c_x = []
    c_y = []
    o_x = []
    o_y = []
    # Ga langs (grid_size X grid_size) driehoekjes
    for i in tqdm(range(grid_size)):
        for j in tqdm(range(grid_size), leave=False):
            (x, y) = get_center(i, j)
            # todo better surface integration approximation
            force_vector = (forceg.shear_x(x, y), forceg.shear_y(x, y), -forceg.normal(x, y))
            p = init_rbm()
            (shifted_center, shifted_offset) = solve_module(force_vector)
            o = shifted_offset - shifted_center
            c_x.append(shifted_center[0] + x)
            c_y.append(shifted_center[1] + y)
            o_x.append(o[0])
            o_y.append(o[1])

    fig = plt.quiver(c_x, c_y, o_x, o_y)
    plt.show()


if __name__ == "__main__":
    main()
            

