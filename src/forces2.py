import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# --- FUNCTIES ---
def generate_normal_force(x, y, x0, y0, rx, ry, smoothness=1.0, sphere_factor=1.2, peak_val=1.0):
    d_sq = ((x - x0)**2 / rx**2) + ((y - y0)**2 / ry**2)
    d_sq = np.clip(d_sq, 0, 1)
    z_norm = np.sqrt(1 - d_sq**sphere_factor)
    z_norm[d_sq >= 1] = 0
    noise = np.random.normal(0, 0.02, x.shape)
    z_norm = np.where(z_norm > 0, z_norm + noise, 0)
    z_norm_smooth = gaussian_filter(z_norm, sigma=smoothness)
    z_norm_smooth[d_sq > 0.95] = 0
    z_norm_smooth = np.maximum(z_norm_smooth, 0)
    return z_norm_smooth * peak_val, d_sq

def generate_shear_force(x, y, d_sq, peak_val):
    raw_profile = (d_sq**0.3) * ((1.0 - d_sq)**0.3)
    max_raw = np.max(raw_profile) if np.max(raw_profile) > 0 else 1
    normalized_profile = raw_profile / max_raw
    shear_noise = np.random.normal(0, 0.01, x.shape)
    shear_val = np.where(d_sq < 1.0, (normalized_profile + shear_noise) * peak_val, 0)
    return np.maximum(shear_val, 0)

# 1. Setup Grid
resolution = 0.0025 
side_range = np.arange(0, 0.1 + resolution, resolution)
x, y = np.meshgrid(side_range, side_range)

# 2. Randomize Parameters
x0, y0 = np.random.uniform(0.03, 0.07, 2)
rx, ry = np.random.uniform(0.02, 0.03, 2)

# 3. Kracht basisinstellingen
normal_peak_val = 125000 
shear_peak_val  = 2500   

# 4. Data Genereren
z_normal, d_sq = generate_normal_force(x, y, x0, y0, rx, ry, smoothness=1.2, peak_val=normal_peak_val)
z_shear_mag_raw = generate_shear_force(x, y, d_sq, shear_peak_val)

# 5. Vector Decompositie (Richting naar centrum)
dx, dy = x0 - x, y0 - y
dist = np.sqrt(dx**2 + dy**2); dist[dist==0]=1e-10
z_shear_x = z_shear_mag_raw * (dx / dist)
z_shear_y = z_shear_mag_raw * (dy / dist)

# 6. Zwaartekracht (Toevoegen aan Y)
r_avg = (rx + ry) / 2.0
g_total = 150 * (r_avg / 0.03)**2 + np.random.uniform(-25, 25)
z_shear_y_final = np.where(d_sq < 1.0, z_shear_y + (g_total * 200), 0)

# --- 7. PLOTTING ---
def get_sym_lim(data):
    limit = np.max(np.abs(data)) if np.max(np.abs(data)) > 0 else 1
    return -limit, limit

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Map 1: Normale Kracht
im1 = axes[0].pcolormesh(x, y, z_normal, cmap='viridis', shading='auto')
axes[0].set_title("Normale Kracht ($N/m^2$)")
axes[0].set_aspect('equal')
fig.colorbar(im1, ax=axes[0])

# Map 2: Schuifkracht X-Component
lx, hx = get_sym_lim(z_shear_x)
im2 = axes[1].pcolormesh(x, y, z_shear_x, cmap='RdBu_r', shading='auto', vmin=lx, vmax=hx)
axes[1].set_title("Schuifkracht X-Component\n(Wit = 0)")
axes[1].set_aspect('equal')
fig.colorbar(im2, ax=axes[1])

# Map 3: Schuifkracht Y-Component (Inclusief Zwaartekracht)
ly, hy = get_sym_lim(z_shear_y_final)
im3 = axes[2].pcolormesh(x, y, z_shear_y_final, cmap='RdBu_r', shading='auto', vmin=ly, vmax=hy)
axes[2].set_title(f"Schuifkracht Y-Component\n(Incl. {g_total:.1f}g)")
axes[2].set_aspect('equal')
fig.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.show()