# pyright: strict

from calendar import c

import numpy as np
from numpy import c_

from forces2 import ForceGenerator2
from trigrid import TriangleGrid
import bep
import matplotlib as mpl
import matplotlib.pyplot as plt
import tqdm

def main():
    mpl.interactive(True)
    # np.random.seed(0)


    # Totale oppervlak van sensor grid
    grid_width = 0.1
    grid_height = 0.1

    # Eigenschappen driehoekig oppervlak aan de bovenkant
    module_area = 0.01*0.01 # 1cm^2 modules
    tris = TriangleGrid(grid_width, grid_height, module_area)

    # forceg = ForceGenerator(tris.width()*2, tris.height(), tris.height()*0.1, tris.width()*0.15, tris.width()*0.2, 3, 0, 15, -4, 4, -4, 4)
    # forceg = ForceGenerator2(tris.width*2, tris.height, tris.height*0.3, tris.width*0.15, tris.width*0.2)
    peak_normal = 5 / (0.01 * 0.01) # N/m^2 - hier dus 5N / cm^2
    forceg = ForceGenerator2(0.1, 0.1, 0.03, 0.01, 0.02, normal_peak=peak_normal, shear_peak=peak_normal * 0.2)
    forceg.show()

    plt.figure(figsize=(10, 10))

    forces = []
    for i in range(tris.n_x):
        forcess = []
        for j in range(tris.n_y):
            (tx, ty) = tris.get_triangle_center(i, j)
            (fx, fy, fz) = forceg(tx, ty)
            force_vector = [tris.t_area * fx, tris.t_area * fy, tris.t_area * fz]

            forcess.append(force_vector)
        forces.append(forcess)

    corners = []
    for i in tqdm.tqdm(range(tris.n_x)):
        corners_row = []
        for j in tqdm.tqdm(range(tris.n_y), leave=False):
            f_vec = forces[i][j]
            (c1, c2, c3) = bep.solve_module(f_vec, upside_down=((i + j) % 2 == 0))
            (tc_x, tc_y) = tris.get_triangle_center(i, j)
            original_center = [tc_x, tc_y, 0]
            c1 += original_center
            c2 += original_center      
            c3 += original_center
            plt.fill([c1[0], c2[0], c3[0]], [c1[1], c2[1], c3[1]], color="black")
            corners.append([c1, c2, c3])
        corners.append(corners_row)
            


    # plt.quiver(X, Y, U, V)
    plt.axis('equal')
    plt.show()

    # Hoekpunten output:



    input()


if __name__ == "__main__":
    main()