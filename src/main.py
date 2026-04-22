from forcegen import ForceGenerator
import bep
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def main():
    mpl.interactive(True)

    forceg = ForceGenerator(0.005, 0.0075, 0.025, 3, 0.1, 80)
    forceg.show()

    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap('viridis')

    plt.figure(figsize=(10, 10))

    for i in range(15):
        for j in range(15):
            (tx, ty) = bep.get_triangle_center(i, j)
            (X, Y) = bep.get_triangle_vertices(i, j)

            force_vector = (forceg.shear_x(tx, ty), forceg.shear_y(tx, ty), -forceg.normal(tx, ty))

            plt.fill(X, Y, color=cmap(norm(forceg.normal(tx, ty))))

    plt.show()
    input()


if __name__ == "__main__":
    main()