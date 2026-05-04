import numpy as np
import numpy.typing as npt
from typing import TypeAlias
import matplotlib.pyplot as plt

ndfloat: TypeAlias = npt.NDArray[np.float64] | float
ndfloatlist: TypeAlias = npt.NDArray[np.float64] | list[float]

class ForceGenerator:
    def __init__(self, width: float, height: float, safe_distance: float, s_min: float, s_max: float, n_peaks_max: int, min_normal:float = 0, max_normal:float = 50, min_x_shear: float = -5, max_x_shear: float = 5, min_y_shear: float = -10, max_y_shear:float = 10):
        """Create ForceGenerator instance

        Args:
            width: x range
            height: y range
            safe_distance: Minimum distance from edge of peak
            s_min: Minimum distribution variance
            s_max: Maximum distribution variance    
            n_peaks_max: Max number of peaks    
            min_normal: Minimum normal peak normal force. Defaults to 50.
            max_normal: Maximum peak normal force. Defaults to 150.
            min_x_shear: Minimum peak shear force. Defaults to 0.
            max_x_shear: . Defaults to 10.
            min_y_shear: _description_. Defaults to 0.
            max_y_shear: _description_. Defaults to 100.
        """
        self.width = width
        self.height = height
        self.safe_distance = safe_distance
        self.s_min = s_min
        self.s_max = s_max
        self.n_peaks_max = n_peaks_max
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

        # Select random peak locations
        self.centers_x = np.random.uniform(self.safe_distance, self.width - self.safe_distance, self.n_peaks)
        self.centers_y = np.random.uniform(self.safe_distance, self.height - self.safe_distance, self.n_peaks)
        
        self.n_scales = np.random.uniform(self.min_normal, self.max_normal, self.n_peaks)
        self.sx_scales = np.random.uniform(self.min_x_shear, self.max_x_shear, self.n_peaks)
        self.sy_scales = np.random.uniform(self.min_y_shear, self.max_y_shear, self.n_peaks)

        self.shear_x_signs = np.random.choice([-1, 1], self.n_peaks)
        self.shear_y_signs = np.random.choice([-1, 1], self.n_peaks)

    def show(self):
        # Grid
        x = np.linspace(0, self.width, 50)
        y = np.linspace(0, self.height, 50)
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

    def sample_distr(self, x: ndfloat, y: ndfloat, centers_x: ndfloatlist, centers_y: ndfloatlist, sigma: ndfloatlist, scales: ndfloatlist, signs: ndfloatlist=[]):
        if len(signs) == 0:
            signs = list(np.ones(self.n_peaks))

        # Return the normal force at a given point
        ret = 0
        for cx, cy, si, scale, sign in zip(centers_x, centers_y, sigma, scales, signs):
            ret += scale * sign * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * si**2))
        return ret
    
    def normal(self, x: ndfloat, y: ndfloat):
        # Return the normal force at a given point
        return self.sample_distr(x, y, self.centers_x, self.centers_y, self.sigma_n, self.n_scales)
    
    def shear_x(self, x: ndfloat, y: ndfloat):
        # Return the x component of the shear at a given point
        return self.sample_distr(x, y, self.centers_x, self.centers_y, self.sigma_x, self.sx_scales, self.shear_x_signs)
    
    def shear_y(self, x: ndfloat, y: ndfloat):
        # Return the y component of the shear at a given point
        return self.sample_distr(x, y, self.centers_x, self.centers_y, self.sigma_x, self.sy_scales, self.shear_y_signs)