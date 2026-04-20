import numpy as np
import matplotlib.pyplot as plt

#parameters
Veligheidsafstand = 0.6
min_sigma = 0.0075
max_sigma = 0.025
aantal_pieken = 5
grootte_opp = 0.05
precizie = 80

#krachten
min_normaal = 1
max_normaal = 30
min_X = 0
max_X = 1.5
min_Y = 0
max_Y = 3


# Parameters
n_peaks = np.random.randint(1, aantal_pieken + 1)  
sigma_n = np.random.uniform(min_sigma, max_sigma, n_peaks)
sigma_x = np.random.uniform(min_sigma, max_sigma, n_peaks)
sigma_y = np.random.uniform(min_sigma, max_sigma, n_peaks)
totale_kracht = np.random.uniform(min_normaal, max_normaal)
totale_schuifkracht_x = np.random.uniform(min_X, max_X)
totale_schuifkracht_y = np.random.uniform(min_Y, max_Y)
r_max = grootte_opp
grid_size = precizie

# Grid
x = np.linspace(-r_max, r_max, grid_size)
y = np.linspace(-r_max, r_max, grid_size)
X, Y = np.meshgrid(x, y)
N_total = np.zeros_like(X)
X_total = np.zeros_like(X)
Y_total = np.zeros_like(X)

# Willekeurige locaties bepalen
centers_x = np.random.uniform(-r_max* Veligheidsafstand , r_max* Veligheidsafstand , n_peaks)
centers_y = np.random.uniform(-r_max* Veligheidsafstand, r_max* Veligheidsafstand , n_peaks)

# 1. Normaalkracht (altijd positief/optellen)
for cx, cy, si in zip(centers_x, centers_y, sigma_n):
    N_peak = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * si**2))
    N_total += N_peak

# 2. Schuifkracht X (willekeurig optellen of aftrekken)
for cx, cy, si in zip(centers_x, centers_y, sigma_x):
    # np.random.choice([1, -1]) bepaalt of de piek omhoog of omlaag wijst
    direction = np.random.choice([1, -1])
    X_peak = direction * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * si**2))
    X_total += X_peak

# 3. Schuifkracht Y (willekeurig optellen of aftrekken)
for cx, cy, si in zip(centers_x, centers_y, sigma_y):
    direction = np.random.choice([1, -1])
    Y_peak = direction * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * si**2))
    Y_total += Y_peak

# Normaliseren
# Let op: bij schuifkrachten gebruiken we de som van de absolute waarden 
# om te voorkomen dat we delen door bijna nul als pieken elkaar opheffen.
N_total = (N_total / np.abs(N_total).sum()) * totale_kracht
X_total = (X_total / np.abs(X_total).sum()) * totale_schuifkracht_x
Y_total = (Y_total / np.abs(Y_total).sum()) * totale_schuifkracht_y

# Plotten
fig = plt.figure(f'{n_peaks} piek(en)', figsize=(16, 7))

# Normaal
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, N_total, cmap='viridis')
ax1.set_title(f'Normaal krachten\n{totale_kracht:.1f}N')

# X
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, X_total, cmap='viridis') # coolwarm is handig voor pos/neg
ax2.set_title(f'Schuifkrachten X (Netto)\n{totale_schuifkracht_x:.1f}N')

# Y
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X, Y, Y_total, cmap='viridis')
ax3.set_title(f'Schuifkrachten Y (Netto)\n{totale_schuifkracht_y:.1f}N')

plt.tight_layout()
plt.show()