from forcegen import ForceGenerator
import bep
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def main():
    mpl.interactive(True)

    # Initialiseer force generatie class
    forceg = ForceGenerator(0.005, 0.0075, 0.025, 3, 0.1, 80)
    forceg.show()

    norm = mpl.colors.Normalize(vmin = 0, vmax = 1)
    m = cm.ScalarMappable(norm = norm, cmap=cm.viridis)

    plt.figure(figsize = (10, 10))

    for i in range(15):
        for j in range(15):
            (tx, ty) = bep.get_triangle_center(i, j)
            (X, Y) = bep.get_triangle_vertices(i, j)

            force_vector = (forceg.shear_x(tx, ty), forceg.shear_y(tx, ty), -forceg.normal(tx, ty))


            plt.fill(X, Y, color = m.to_rgba(forceg.normal(tx, ty)))

    plt.show()

    input()

if __name__ == "__main__":
    main()
