import numpy as np

class TriangleGrid:
    def __init__(self, triangle_area, grid_size):
        self.grid_size = grid_size
        self.t_area = triangle_area
        self.t_side = np.sqrt(4 * self.t_area / np.sqrt(3))
        self.t_height = np.sqrt(3)/2 * self.t_side
    
    def _bounds_check(self, i,j):
        if (i > self.grid_size*2 - 1 or j > self.grid_size - 1):
            raise Exception(f"Index ({i}, {j}) is out of bounds for {self.grid_size}*{self.grid_size} grid")

    # Get center coordinate of triangle at index (i, j)
    def get_triangle_center(self, i, j):
        self._bounds_check(i, j)

        x = i * self.t_side/2
        if ((i + j) % 2 == 0):
            return (x, (1/3 + j)*self.t_height)
        else:
            return (x, (2/3 + j)*self.t_height)

    # Get corner coordinates of triangle at index (i, j)
    def get_triangle_vertices(self, i, j):
        self._bounds_check(i, j)

        (c_x, c_y) = self.get_triangle_center(i, j)
        v_x = [c_x, c_x + 0.5*self.t_side, c_x - 0.5*self.t_side]
        if ((i + j) % 2 == 0):
            v_y = [c_y + (2/3) * self.t_height, c_y - self.t_height/3, c_y - self.t_height/3]
        else:
            v_y = [c_y - (2/3) * self.t_height, c_y + self.t_height/3, c_y + self.t_height/3]

        return (v_x, v_y)

    # get the width of this triangle grid
    def width(self):
        return self.grid_size * self.t_side * 0.5

    # get height of this triangle grid
    def height(self):
        return self.grid_size * self.t_height