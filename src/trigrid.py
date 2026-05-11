# pyright: strict

import numpy as np
import optype.numpy as onp



class TriangleGrid:
    def __init__(self, width: float, height: float, triangle_area: float):
        """Helper class for managing grid of modules

        Args:
            width: Maximum width of module grid
            height: Maximum height of module grid
            triangle_area: Module surface area, used to determine dimensions of equilateral triangles in the grid.
        """
        self.height = height
        self.width = width
        self.t_area = triangle_area
        self.t_side = np.sqrt(4 * self.t_area / np.sqrt(3))
        self.t_height = np.sqrt(3)/2 * self.t_side
        self.n_x = int(self.width / (self.t_side / 2))
        self.n_y = int(self.height / (self.t_height))

    # Get center coordinate of triangle at index (i, j)
    def get_triangle_center(self, i: int, j: int):
        x: float = i * self.t_side/2
        if ((i + j) % 2 == 0):
            y: float = (1/3 + j)*self.t_height
        else:
            y: float = (2/3 + j)*self.t_height

        if (x > self.width or y > self.height):
            raise IndexError(f"Triangle index ({i}, {j}) is out of bounds for grid of width {self.width} and height {self.height}.")
        
        return (x, y)

    # Get corner coordinates of triangle at index (i, j)
    def get_triangle_vertices(self, i: int, j: int):
        (c_x, c_y) = self.get_triangle_center(i, j)
        v_x = [c_x, c_x + 0.5*self.t_side, c_x - 0.5*self.t_side]
        if (i + j) % 2 == 0:
            v_y = [c_y + (2/3) * self.t_height, c_y - self.t_height/3, c_y - self.t_height/3]
        else:
            v_y = [c_y - (2/3) * self.t_height, c_y + self.t_height/3, c_y + self.t_height/3]

        return v_x, v_y
