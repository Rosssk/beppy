import numpy as np
from scipy.integrate import dblquad
import matplotlib.pyplot as plt

class ForceGenerator:
    def __init__(self, safe_distance, s_min, s_max, n_peaks_max, area, grid_size, min_normal=50, max_normal=150, min_x_shear=0, max_x_shear=10, min_y_shear=0, max_y_shear=100):
        self.safe_distance = safe_distance
        self.s_min = s_min
        self.s_max = s_max
        self.n_peaks_max = n_peaks_max
        self.area = area
        self.grid_size = grid_size
        self.min_normal = min_normal
        self.max_normal = max_normal
        self.min_x_shear = min_x_shear
        self.max_x_shear = max_x_shear
        self.min_y_shear = min_y_shear
        self.max_y_shear = max_y_shear

        self.reroll()

    def reroll(self):
        # Randomize variables
        self.n_peaks = np.random.randint(1, self.n_peaks_max + 1)  
        self.sigma_n = np.random.uniform(self.s_min, self.s_max, self.n_peaks_max)
        self.sigma_x = np.random.uniform(self.s_min, self.s_max, self.n_peaks_max)
        self.sigma_y = np.random.uniform(self.s_min, self.s_max, self.n_peaks_max)
        self.totale_kracht = np.random.uniform(self.min_normal, self.max_normal)
        self.totale_schuifkracht_x = np.random.uniform(self.min_x_shear, self.max_x_shear)
        self.totale_schuifkracht_y = np.random.uniform(self.min_y_shear, self.max_y_shear)
        self.r_max = self.area

        # Select random peak locations
        self.centers_x = np.random.uniform(-self.r_max* self.safe_distance, self.r_max * self.safe_distance, self.n_peaks)
        self.centers_y = np.random.uniform(-self.r_max* self.safe_distance, self.r_max * self.safe_distance, self.n_peaks)
        
        self.shear_x_signs = np.random.choice([-1, 1], self.n_peaks)
        self.shear_y_signs = np.random.choice([-1, 1], self.n_peaks)


    def show(self):
        # Grid
        x = np.linspace(-self.r_max, self.r_max, self.grid_size)
        y = np.linspace(-self.r_max, self.r_max, self.grid_size)
        X, Y = np.meshgrid(x, y)
        N_total = self.normal(X, Y)
        X_total = self.shear_x(X, Y)
        Y_total = self.shear_y(X, Y)

        # Plotten
        fig = plt.figure(figsize=(18, 7))

        # Normaal
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot_surface(X, Y, N_total, cmap='viridis')
        ax1.set_title(f'Normaal krachten\n{self.totale_kracht:.1f}N')

        # X
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.plot_surface(X, Y, X_total, cmap='viridis') # coolwarm is handig voor pos/neg
        ax2.set_title(f'Schuifkrachten X (Netto)\n{self.totale_schuifkracht_x:.1f}N')

        # Y
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.plot_surface(X, Y, Y_total, cmap='viridis')
        ax3.set_title(f'Schuifkrachten Y (Netto)\n{self.totale_schuifkracht_y:.1f}N')

        plt.tight_layout()
        plt.show()

    def sample_distr(self, x, y, centers_x, centers_y, sigma, signs=[]):
        if len(signs) == 0:
            signs = np.ones(self.n_peaks)

        # Return the normal force at a given point
        ret = 0
        for cx, cy, si, sign in zip(centers_x, centers_y, sigma, signs):
            ret += sign * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * si**2))
        return ret
    
    def normal(self, x, y):
        # Return the normal force at a given point
        return self.sample_distr(x, y, self.centers_x, self.centers_y, self.sigma_n)
    
    def shear_x(self, x, y):
        # Return the x component of the shear at a given point
        return self.sample_distr(x, y, self.centers_x, self.centers_y, self.sigma_x)
    
    def shear_y(self, x, y):
        # Return the y component of the shear at a given point
        return self.sample_distr(x, y, self.centers_x, self.centers_y, self.sigma_x)