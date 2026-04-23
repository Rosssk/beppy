from forcegen import ForceGenerator
from trigrid import TriangleGrid
import bep
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import sim
import tqdm

def main():
    mpl.interactive(True)

    tri_grid_size = 10
    tris = TriangleGrid(0.01, tri_grid_size)
    
    forceg = ForceGenerator(tris.width(), tris.height(), tris.height()*0.1, tris.width()*0.08, tris.width()*0.1, 3)
    forceg.show()

    norm = mcolors.Normalize(vmin=-1, vmax=1)
    cmap = plt.get_cmap('viridis')

    plt.figure(figsize=(10, 10))

    forces = []
    for i in range(tri_grid_size):
        forcess = []
        for j in range(tri_grid_size):
            (tx, ty) = tris.get_triangle_center(i, j)
            # (X, Y) = tris.get_triangle_vertices(i, j)
            # plt.fill(X, Y, color=cmap(norm(forceg.shear_x(tx, ty))))

            force_vector = (forceg.shear_x(tx, ty), forceg.shear_y(tx, ty), -forceg.normal(tx, ty))
            forcess.append(force_vector)
        forces.append(forcess)
        
    X = []
    Y = []
    U = []
    V = []
    for i in tqdm.tqdm(range(tri_grid_size)):
        for j in tqdm.tqdm(range(tri_grid_size), leave=False):
            f_vec = forces[i][j]
            (local_center, local_offset) = bep.solve_module(f_vec)
            (tc_x, tc_y) = tris.get_triangle_center(i, j)
            original_center = [tc_x, tc_y, 0]
            center = original_center + local_center
            X.append(center[0])
            Y.append(center[1])
            U.append(local_offset[0])
            V.append(local_offset[1])


    plt.quiver(X, Y, U, V)
    plt.show()
    input()


if __name__ == "__main__":
    main()