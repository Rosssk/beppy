import numpy as np
from pyDOE3 import lhs
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# ── 0. Define your problem ────────────────────────────────────────────────────

N_INPUTS  = 10
N_OUTPUTS = 18
N_WORKERS = 16   # parallel workers — set to your CPU core count

# Bounding box for LHS: (min, max) per input
# LHS seeds the search; constraints below filter it
BOUNDS = np.array([
    [1e-5, 1], # r1
    [1e-5, 1], # r2
    [1e-5, 1], # r3
    [1e-4, 1], # h1
    [1e-4, 1], # h2
    [1e-2, 1e3],  # E in GPa
    [1e-6, 1e-1], # t
    [1e-2, 1e3], # fx
    [1e-2, 1e3], # fy
    [1e-2, 1e3], # fz
])  

# lower = np.array([1e-4,1e-4,1e-4,1e-3,1e-3,1,1e-5,1e-1,1e-1,1e-1])
# upper = np.array([1e-1,1e-1,1e-1,1,1,1e3,1e-2,1e3,1e3,1e3])

def physics_model(x: np.ndarray) -> np.ndarray | None:
    r1, r2, r3, h1, h2, E, t, fx, fy, fz = x
    from rbm import PRBM
    from numpy import sin, cos, pi, concatenate

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
    p.solve_pose('BC', A, Emod, I)


    # if p.solution.x[2] <= 0 or p.solution.x[8] <= 0:
    #     # Squished through ground?
    #     return None

    # angles = np.concatenate((p.solution.x[3:5], p.solution.x[9:11]))

    # if np.any(np.abs(angles) > np.pi * 0.5):
    #     # Rotated too far
    #     return None

    # xypos = np.concatenate((p.solution.x[0:2], p.solution.x[6:8]))
    # if np.any(np.abs(xypos) > 2*r1):
    #     # Shifted too far
    #     return None

    # if p.solution.x[2] > h1:
    #     # Went up somehow?
    #     return None

    state = []
    state += list(p.bodies['A'].position)
    state += list(p.bodies['A'].angles)
    state += list(p.bodies['B'].position)
    state += list(p.bodies['B'].angles)
    state += list(p.bodies['C'].position)
    state += list(p.bodies['C'].angles)


    return state

# ── 1. Generate candidate points via LHS ─────────────────────────────────────

def lhs_candidates(n: int) -> np.ndarray:
    """Draw n LHS candidates scaled to BOUNDS."""
    unit = lhs(N_INPUTS, samples=n, criterion="maximin")
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    return lo + unit * (hi - lo)

# ── 2. Evaluate one candidate (used by each worker) ──────────────────────────

def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Returns (x, y) if valid, else None."""
    y = physics_model(x)
    if y is None:
        return None

    return (x, y)

# ── 3. Gather samples with adaptive top-up ───────────────────────────────────

def gather_samples(
    n_target:    int   = 5_000,   # valid samples you want
    batch_size:  int   = 16*48,     # candidates per round
    max_rounds:  int   = 1_000_000,      # give up after this many rounds
) -> tuple[np.ndarray, np.ndarray]:

    X_valid, Y_valid = [], []

    with tqdm(total=n_target, desc="Sampling") as pbar:
        for round_ in range(max_rounds):
            if len(X_valid) >= n_target:
                break

            candidates = lhs_candidates(batch_size)

            with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
                results = list(pool.map(evaluate, candidates))

            new = [(x, y) for r in results if r is not None for x, y in [r]]
            X_valid.extend(x for x, _ in new)
            Y_valid.extend(y for _, y in new)
            pbar.update(len(new))

            acceptance = len(new) / batch_size
            print(f"  round {round_+1}: {len(new)}/{batch_size} valid  "
                  f"({acceptance:.0%} acceptance,  "
                  f"total {len(X_valid)})")

            if acceptance < 0.05:
                print("⚠ Low acceptance rate — check your bounds or constraints")

    X = np.array(X_valid[:n_target])
    Y = np.array(Y_valid[:n_target])
    return X, Y

# ── 4. Run and save ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    X, Y = gather_samples(n_target=100_000)
    np.save("X.npy", X)
    np.save("Y.npy", Y)
    print(f"Saved {X.shape[0]} samples  →  X.npy, Y.npy")

``