# pyright: strict

from forces2 import ForceGenerator2
from trigrid import TriangleGrid
import bep
import matplotlib as mpl
import matplotlib.pyplot as plt
import tqdm

class Beppy:
    def __init__(self, width: float, height: float, module_area: float):
        self.width: float = width
        self.height: float = height
        self.force_gen: ForceGenerator2 | None = None
        self.solution: float | None = None

        self.update_module_grid(module_area)

    def set_force_generator(self, safety_margin: float, r_min: float, r_max: float, smoothness: float, sphere_factor: float, normal_peak: float, shear_peak: float):
        self.force_gen = ForceGenerator2(self.width, self.height, safety_margin, r_min, r_max, smoothness, sphere_factor, normal_peak, shear_peak)

    def update_module_grid(self, module_area: float):
        self.module_area = module_area
        self.tri_grid = TriangleGrid(self.width, self.height, module_area)

    def create_solution(self):
        if self.force_gen is None:
            raise ValueError("No force generator set. Please set one first.")

        

        # Get coordinates of all the centers of the modules in the triangle grid        
        centers = self.tri_grid.get_triangle_centers()
        forces = []

        # for i in range(len(centers)):
        #     forces_row: list[tuple[float, float, float]] = []
        #     for j in range(len(centers[i])):
        #         c_x, c_y = centers[i][j]
        #         force_vector = self.force_gen(c_x, c_y)
        #         forces_row.append(force_vector)
        #     forces.append(forces_row)

        # # Get forces at the center of each module
        # f_x, f_y, f_z = self.force_gen.force(CX, CY)



def main():
    mpl.interactive(True)
    # np.random.seed(0)

    tri_grid_size = 10
    tris = TriangleGrid(0.0001, tri_grid_size)
    
    # forceg = ForceGenerator(tris.width()*2, tris.height(), tris.height()*0.1, tris.width()*0.15, tris.width()*0.2, 3, 0, 15, -4, 4, -4, 4)
    forceg = ForceGenerator2(tris.width()*2, tris.height(), tris.height()*0.3, tris.width()*0.15, tris.width()*0.2)
    forceg.show()

    plt.figure(figsize=(10, 10))

    forces = []
    for i in range(tri_grid_size*2):
        forcess = []
        for j in range(tri_grid_size):
            (tx, ty) = tris.get_triangle_center(i, j)
            # (X, Y) = tris.get_triangle_vertices(i, j)
            # plt.fill(X, Y, color=cmap(norm(forceg.shear_x(tx, ty))))

            # force_vector = (forceg.shear_x(tx, ty), forceg.shear_y(tx, ty), -forceg.normal(tx, ty))
            force_vector = forceg.force(tx, ty)
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
            plt.fill([c1[0], c2[0], c3[0]], [c1[1], c2[1], c3[1]])
            


    # plt.quiver(X, Y, U, V)
    plt.axis('equal')
    plt.show()
    input()


if __name__ == "__main__":
    main()