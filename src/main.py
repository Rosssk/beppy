from forcegen import ForceGenerator
from trigrid import TriangleGrid
import bep
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import tqdm

def main():
    mpl.interactive(True)
    # np.random.seed(0)

    tri_grid_size = 10
    tris = TriangleGrid(0.0001, tri_grid_size)
    
    forceg = ForceGenerator(tris.width()*2, tris.height(), tris.height()*0.1, tris.width()*0.15, tris.width()*0.2, 3, 0, 15, -4, 4, -4, 4)
    forceg.show()

    norm = mcolors.Normalize(vmin=-1, vmax=1)
    cmap = plt.get_cmap('viridis')

    plt.figure(figsize=(10, 10))

    forces = []
    for i in range(tri_grid_size*2):
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
    for i in tqdm.tqdm(range(tri_grid_size*2)):
        for j in tqdm.tqdm(range(tri_grid_size), leave=False):
            f_vec = forces[i][j]
            (c1, c2, c3) = bep.solve_module(f_vec)
            (tc_x, tc_y) = tris.get_triangle_center(i, j)
            original_center = [tc_x, tc_y, 0]
            c1 += original_center
            c2 += original_center      
            c3 += original_center
            plt.fill([c1[0], c2[0], c3[0]], [c1[1], c2[1], c3[1]], color=cmap(norm(f_vec[0])))
            


    # plt.quiver(X, Y, U, V)
    plt.axis('equal')
    plt.show()
    input()


if __name__ == "__main__":
    main()