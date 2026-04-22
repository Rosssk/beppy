from src.forcegen import ForceGenerator
import numpy as np
from numpy import cos, sin, pi
from src.rbm import PRBM
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

def get_triangle_center(i, j):
    # Get center coordinate of triangle at index (i, j)
    x = i * side_length/2
    if ((i + j) % 2 == 0):
        return (x, (1/3 + j)*t_height)
    else:
        return (x, (2/3 + j)*t_height)
    
def get_triangle_vertices(i, j):
    # Get corner coordinates of triangle at index (i, j)
    (c_x, c_y) = get_triangle_center(i, j)
    v_x = [c_x, c_x + 0.5*side_length, c_x - 0.5*side_length]
    if ((i + j) % 2 == 0):
        v_y = [c_y + (2/3) * t_height, c_y - t_height/3, c_y - t_height/3]
    else:
        v_y = [c_y - (2/3) * t_height, c_y + t_height/3, c_y + t_height/3]

    return (v_x, v_y)

def main():
    grid_size = 10
    forceg = ForceGenerator(0.005, 0.0075, 0.025, 3, t_height*grid_size, 80)
    # forceg.show()
    t_area = 0.5 * t_height * side_length

    c_x = []
    c_y = []
    o_x = []
    o_y = []
    # Ga langs (grid_size X grid_size) driehoekjes
    for i in tqdm(range(grid_size)):
        for j in tqdm(range(grid_size), leave=False):
            (x, y) = get_triangle_center(i, j)
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
            

