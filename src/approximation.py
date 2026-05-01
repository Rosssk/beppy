from tqdm.contrib.concurrent import process_map
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

N_FZ = 30
N_FY = 30
MAX_FZ = 100
MAX_FY = 40
MAX_DISP = 5e-3  # m

def expensive_function(x, show=False, x0=None):
    r1, r2, r3, h1, h2, E, t, fx, fy, fz = x
    from rbm import PRBM
    from numpy import sin, cos, pi
    Emod = E * 1e9
    A = pi * t**2
    I = pi * t**4 / 2
    p = PRBM(3)
    p.add_body('A', (0, 0, 0))
    p.add_body('B', (0, 0, h1))
    p.add_body('C', (0, 0, h1 + h2))
    n = 3
    for i in range(n):
        p.add_flexure('A', (r1*cos(i/n*2*pi), r1*sin(i/n*2*pi), 0),
                      'B', (r2*cos((i+1)/n*2*pi), r2*sin((i+1)/n*2*pi), 0))
        p.add_flexure('C', (r3*cos((i-.5)/n*2*pi), r3*sin((i-.5)/n*2*pi), 0),
                      'B', (r2*cos((i+.5)/n*2*pi), r2*sin((i+.5)/n*2*pi), 0))
    p.add_force('C', [fx, fy, -fz])
    p.solve_pose('BC', A, Emod, I, x0=x0, method='Nelder-Mead',
                 options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-6})
    if not p.solution.success:
        p.solve_pose('BC', A, Emod, I, x0=None, method='Nelder-Mead',
                     options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-6})
    if show:
        p.show()

    pos_C = p.bodies['C'].position
    pos_C0 = p.bodies['C'].position0
    disp_C = norm(pos_C - pos_C0)

    success = p.solution.success and disp_C < MAX_DISP

    return p.solution.x, success, p.solution.fun, disp_C


def batched(args):
    r, fz, Y = args
    S, Z, F, D = [], [], [], []
    x0 = None
    for fy in Y:
        z, success, fun, disp_C = expensive_function(
            [r, r, r, 5e-3, 5e-3, 0.85, 1e-3, 0, fy, fz], x0=x0)
        if success:
            x0 = list(z)
        S.append(success)
        Z.append(z)       # full solution vector
        F.append(fun)
        D.append(disp_C)
    return S, Z, F, D


def main():
    r = 8.377e-3
    Y = np.linspace(0, MAX_FY, N_FY)
    X = [(r, fz, Y) for fz in np.linspace(0, MAX_FZ, N_FZ)]

    results = process_map(batched, X, max_workers=16, chunksize=1)
    S_all, Z_all, F_all, D_all = zip(*results)
    S_all = sum(S_all, [])
    Z_all = sum(Z_all, [])
    F_all = sum(F_all, [])
    D_all = sum(D_all, [])

    S_grid = np.array(S_all, dtype=bool).reshape(N_FZ, N_FY)
    F_grid = np.array(F_all, dtype=float).reshape(N_FZ, N_FY)
    D_grid = np.array(D_all, dtype=float).reshape(N_FZ, N_FY)
    Z_grid = np.array(Z_all, dtype=float).reshape(N_FZ, N_FY, 12)

    # --- Heatmaps ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    im1 = ax1.imshow(S_grid, cmap='RdYlGn', origin='lower',
                     extent=[0, MAX_FZ, 0, MAX_FY], aspect='auto')
    ax1.set_title('Solver success')
    ax1.set_xlabel('fz')
    ax1.set_ylabel('fy')
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(F_grid, cmap='viridis', origin='lower',
                     extent=[0, MAX_FZ, 0, MAX_FY], aspect='auto')
    ax2.set_title('Energy residual (solution.fun)')
    ax2.set_xlabel('fz')
    ax2.set_ylabel('fy')
    plt.colorbar(im2, ax=ax2)

    im3 = ax3.imshow(D_grid, cmap='viridis', origin='lower',
                     extent=[0, MAX_FZ, 0, MAX_FY], aspect='auto')
    ax3.set_title('Displacement of C (m)')
    ax3.set_xlabel('fz')
    ax3.set_ylabel('fy')
    plt.colorbar(im3, ax=ax3)

    plt.tight_layout()

    # --- Displacement/angle vs fy for a fixed fz slice ---
    # Pick the middle fz slice
    fz_idx = 3
    fz_val = np.linspace(0, MAX_FZ, N_FZ)[fz_idx]
    fy_vals = np.linspace(0, MAX_FY, N_FY)

    # Z_grid[fz_idx, :, :] -> shape (N_FY, 12)
    # solution vector: [pos_B(3), angles_B(3), pos_C(3), angles_C(3)]
    slice_Z = Z_grid[fz_idx]
    slice_S = S_grid[fz_idx]
    slice_D = D_grid[fz_idx]

    pos_C  = slice_Z[:, 6:9]    # x, y, z of C
    ang_C  = slice_Z[:, 9:12]   # rx, ry, rz of C

    # find cutoff index
    cutoff_idx = np.argmax(~slice_S) if (~slice_S).any() else None
    cutoff_fy  = fy_vals[cutoff_idx] if cutoff_idx is not None else None

    fig2, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig2.suptitle(f'Body C response vs fy  (fz = {fz_val:.1f} N)', fontsize=13)

    labels_pos = ['x (m)', 'y (m)', 'z (m)']
    labels_ang = ['rx (rad)', 'ry (rad)', 'rz (rad)']
    colors_pos = ['tab:blue', 'tab:orange', 'tab:green']
    colors_ang = ['tab:red', 'tab:purple', 'tab:brown']

    ax_pos = axes[0, 0]
    ax_ang = axes[0, 1]
    ax_dis = axes[1, 0]
    ax_suc = axes[1, 1]

    for i in range(3):
        ax_pos.plot(fy_vals, pos_C[:, i], color=colors_pos[i], label=labels_pos[i])
        ax_ang.plot(fy_vals, ang_C[:, i], color=colors_ang[i], label=labels_ang[i])

    ax_dis.plot(fy_vals, slice_D, color='tab:blue', label='|disp_C|')
    ax_dis.axhline(MAX_DISP, color='red', linestyle='--', label=f'cutoff ({MAX_DISP*1e3:.1f} mm)')

    ax_suc.plot(fy_vals, slice_S.astype(float), color='tab:green', label='success')

    for ax in [ax_pos, ax_ang, ax_dis, ax_suc]:
        if cutoff_fy is not None:
            ax.axvline(cutoff_fy, color='red', linestyle='--', alpha=0.6, label=f'cutoff fy={cutoff_fy:.1f}')
        ax.set_xlabel('fy (N)')
        ax.legend()
        ax.grid(True)

    ax_pos.set_title('Position of C')
    ax_ang.set_title('Angles of C')
    ax_dis.set_title('Displacement magnitude of C')
    ax_suc.set_title('Solver success')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()