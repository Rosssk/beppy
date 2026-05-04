import numpy as np
import matplotlib.pyplot as plt
from pyDOE3 import lhs
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from scipy.optimize import brentq
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


# ── 0. Problem definition ─────────────────────────────────────────────────────

N_INPUTS  = 10
N_OUTPUTS = 6
N_WORKERS = 16   # set to your CPU core count

# Bounding box for LHS: lower and upper bounds per input
LO = np.array([0.0] * N_INPUTS)   # ← replace with your bounds
HI = np.array([1.0] * N_INPUTS)   # ← replace with your bounds


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
    p.solve_pose('BC', A, Emod, I, method='Nelder-Mead')

    sol = p.solution.x[0:6]

    # --- Validity checks ---

    # 1. Solver convergence
    if not p.solution.success:
        return None

    pos_B  = sol[0:3]
    pos_C  = sol[6:9]
    angles_B = p.solution.x[3:6]
    angles_C = p.solution.x[9:12]

    # 2. Large rotations (PRBM small-angle assumption, valid to ~20°)
    MAX_ANGLE = np.deg2rad(20)
    if np.any(np.abs(angles_B) > MAX_ANGLE) or np.any(np.abs(angles_C) > MAX_ANGLE):
        return None

    # 3. Bodies below ground or topological inversion (C below B)
    # if pos_B[2] < 0 or pos_C[2] < 0:
    #     return None
    # if pos_C[2] < pos_B[2]:
    #     return None

    # 4. Excessive axial strain in flexures (>5% indicates breakdown of rigid-body assumption)
    for flexure in p.flexures.values():
        vec = flexure.springpoint_globalB - flexure.springpoint_globalA
        strain = (np.linalg.norm(vec) - flexure.springlen0) / flexure.springlen0
        if abs(strain) > 0.05:
            return None

    return sol

def is_output_valid(y: np.ndarray) -> bool:
    # if np.abs(y[0]) > 0.005:   # x translation
    #     return False
    # if np.abs(y[1]) > 0.005:   # y translation
    #     return False
    # # y[2] = z translation — no filter, varies continuously
    # if np.abs(y[3]) > 0.02:    # rx
    #     return False
    # if np.abs(y[4]) > 0.02:    # ry
    #     return False
    # if np.abs(y[5]) > 0.01:    # rz
    #     return False
    return True

# ── 1. Sampling ───────────────────────────────────────────────────────────────

def lhs_candidates(n: int) -> np.ndarray:
    """Draw n LHS candidates scaled to bounds."""
    unit = lhs(N_INPUTS, samples=n, criterion="maximin")
    return LO + unit * (HI - LO)


def evaluate(x: np.ndarray) -> tuple | None:
    """Returns (x, y) if valid, else None. Logs nothing — caller handles."""
    try:
        y = physics_model(x)
        if y is None or not is_output_valid(y):
            return None
        return (x, y)
    except Exception:
        return None


def gather_samples(
    n_target:   int = 5_000,
    batch_size: int = 500,
    max_rounds: int = 50,
    save_invalid: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect n_target valid samples via LHS + rejection.
    Optionally saves rejected inputs to X_invalid.npy for later analysis.
    """
    X_valid, Y_valid = [], []
    X_invalid = []

    with tqdm(total=n_target, desc="Sampling") as pbar:
        for round_ in range(max_rounds):
            if len(X_valid) >= n_target:
                break

            candidates = lhs_candidates(batch_size)

            with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
                results = list(pool.map(evaluate, candidates))

            for x, result in zip(candidates, results):
                if result is not None:
                    X_valid.append(result[0])
                    Y_valid.append(result[1])
                elif save_invalid:
                    X_invalid.append(x)

            n_new = sum(1 for r in results if r is not None)
            pbar.update(n_new)

            acceptance = n_new / batch_size
            print(f"  round {round_+1}: {n_new}/{batch_size} valid  "
                  f"({acceptance:.0%} acceptance,  total {len(X_valid)})")

            if acceptance < 0.05:
                print("⚠ Low acceptance rate — consider tightening BOUNDS")

    X = np.array(X_valid[:n_target])
    Y = np.array(Y_valid[:n_target])

    if save_invalid and X_invalid:
        np.save("X_invalid.npy", np.array(X_invalid))
        print(f"Saved {len(X_invalid)} invalid inputs → X_invalid.npy")

    return X, Y


# ── 2. Threshold analysis ─────────────────────────────────────────────────────

def find_thresholds(Y: np.ndarray) -> None:
    """
    Fit a 2-component GMM to each output to find the natural
    valid/invalid boundary — use results to tune is_output_valid().
    Also plots histograms and the rotation vs translation scatter.
    """
    print("Data-driven threshold analysis:")
    for i in range(Y.shape[1]):
        vals = np.abs(Y[:, i]).reshape(-1, 1)
        gm = GaussianMixture(n_components=2, random_state=42).fit(np.log1p(vals))
        means = np.expm1(gm.means_.flatten())
        print(f"  output {i}: cluster means at {means[0]:.5f} and {means[1]:.5f}")

    # Histogram per output
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, ax in enumerate(axes.flat):
        ax.hist(np.abs(Y[:, i]), bins=200, log=True)
        ax.set_title(f"Output {i}")
        ax.set_xlabel("|value|")
    plt.suptitle("Output distributions — threshold sits in the valley")
    plt.tight_layout()
    plt.show()

    # Rotation vs translation scatter
    rot_mag   = np.linalg.norm(Y[:, 3:], axis=1)
    trans_mag = np.linalg.norm(Y[:, :3], axis=1)
    plt.figure(figsize=(8, 5))
    plt.scatter(trans_mag, rot_mag, s=3, alpha=0.2)
    plt.xlabel("‖translation‖")
    plt.ylabel("‖rotation‖")
    plt.yscale("log")
    plt.xscale("log")
    plt.title("Valid vs invalid — look for the gap")
    plt.show()


# ── 3. Outlier filtering ──────────────────────────────────────────────────────

def filter_outliers(X: np.ndarray, Y: np.ndarray,
                    max_rotation_deg: float = 15.0,
                    max_translation: np.ndarray | None = None,
                    smoothness_neighbors: int = 5,
                    smoothness_threshold: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove outliers using:
      1. Hard output thresholds (rotation + translation)
      2. Local smoothness check (catches chaotic near-boundary samples)
    """
    if max_translation is None:
        max_translation = np.array([0.1, 0.1, 0.5])

    max_rot = np.deg2rad(max_rotation_deg)

    # Hard threshold mask
    threshold_mask = (
        (np.abs(Y[:, 0]) < max_translation[0]) &
        (np.abs(Y[:, 1]) < max_translation[1]) &
        (np.abs(Y[:, 2]) < max_translation[2]) &
        (np.abs(Y[:, 3]) < max_rot) &
        (np.abs(Y[:, 4]) < max_rot) &
        (np.abs(Y[:, 5]) < max_rot)
    )
    X_f, Y_f = X[threshold_mask], Y[threshold_mask]
    print(f"After threshold filter: {threshold_mask.sum()} / {len(Y)}")

    # Smoothness mask
    nn = NearestNeighbors(n_neighbors=smoothness_neighbors).fit(X_f)
    _, idx = nn.kneighbors(X_f)
    Y_neighbor_mean = Y_f[idx].mean(axis=1)
    Y_std_global    = Y_f.std(axis=0)
    deviation       = np.abs(Y_f - Y_neighbor_mean) / (Y_std_global + 1e-10)
    smooth_mask     = deviation.max(axis=1) < smoothness_threshold

    X_clean, Y_clean = X_f[smooth_mask], Y_f[smooth_mask]
    print(f"After smoothness filter: {smooth_mask.sum()} / {len(Y_f)}")

    return X_clean, Y_clean


# ── 4. Surrogate fitting ──────────────────────────────────────────────────────

def fit_surrogate(X: np.ndarray, Y: np.ndarray,
                  degree: int = 3) -> tuple[Pipeline, StandardScaler, np.ndarray]:
    """
    Fit a polynomial surrogate with output scaling.
    Returns (surrogate_pipeline, output_scaler, r2_per_output).
    """
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    output_scaler  = StandardScaler()
    Y_train_scaled = output_scaler.fit_transform(Y_train)

    surrogate = Pipeline([
        ("scale", StandardScaler()),
        ("poly",  PolynomialFeatures(degree=degree, include_bias=False)),
        ("ridge", Ridge(alpha=1e-3)),
    ])
    surrogate.fit(X_train, Y_train_scaled)

    Y_pred = output_scaler.inverse_transform(surrogate.predict(X_test))
    r2     = r2_score(Y_test, Y_pred, multioutput="raw_values")

    print(f"\ndegree={degree}  "
          f"n_features={surrogate['poly'].n_output_features_}  "
          f"n_train={len(X_train)}")
    print("R² per output:")
    for i, v in enumerate(r2):
        print(f"  output {i}: {v:.6f}")

    return surrogate, output_scaler, r2


def predict(x_new: np.ndarray,
            surrogate: Pipeline,
            output_scaler: StandardScaler) -> np.ndarray:
    """x_new: (n, 10) or (10,) — returns predictions in original units."""
    x_new = np.atleast_2d(x_new)
    return output_scaler.inverse_transform(surrogate.predict(x_new))


def plot_parity(X_test: np.ndarray, Y_test: np.ndarray,
                surrogate: Pipeline, output_scaler: StandardScaler,
                r2: np.ndarray) -> None:
    """Predicted vs actual scatter per output."""
    Y_pred = predict(X_test, surrogate, output_scaler)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, ax in enumerate(axes.flat):
        ax.scatter(Y_test[:, i], Y_pred[:, i], s=5, alpha=0.3)
        lims = [min(Y_test[:, i].min(), Y_pred[:, i].min()),
                max(Y_test[:, i].max(), Y_pred[:, i].max())]
        ax.plot(lims, lims, 'r--', linewidth=1)
        ax.set_title(f"Output {i}  R²={r2[i]:.3f}")
        ax.set_xlabel("true")
        ax.set_ylabel("predicted")
    plt.tight_layout()
    plt.show()


# ── 5. Boundary scouting ──────────────────────────────────────────────────────

def find_boundary(x0: np.ndarray, direction: np.ndarray,
                  t_max: float = 10.0, tol: float = 1e-3) -> np.ndarray | None:
    """Binary search for the validity boundary along x0 + t*direction."""
    direction = direction / np.linalg.norm(direction)

    if not _is_valid_full(x0):
        raise ValueError("x0 must be a known valid point")

    if _is_valid_full(x0 + t_max * direction):
        return None  # never hits boundary in this direction

    t_boundary = brentq(
        lambda t: 0.0 if _is_valid_full(x0 + t * direction) else 1.0,
        0.0, t_max, xtol=tol
    )
    return x0 + t_boundary * direction


def _is_valid_full(x: np.ndarray) -> bool:
    try:
        y = physics_model(x)
        return y is not None and is_output_valid(y)
    except Exception:
        return False


def scout_boundary(x0: np.ndarray, n_directions: int = 200,
                   t_max: float = 10.0) -> np.ndarray:
    """Sample random directions, find boundary point in each."""
    boundary_points = []
    for i in range(n_directions):
        d  = np.random.randn(len(x0))
        pt = find_boundary(x0, d, t_max=t_max)
        if pt is not None:
            boundary_points.append(pt)
        if i % 20 == 0:
            print(f"  {i}/{n_directions}  "
                  f"found {len(boundary_points)} boundary points")
    return np.array(boundary_points)


def fit_mahalanobis(boundary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a Mahalanobis ellipsoid to boundary points.
    Returns (mean, inv_cov) for use as a fast validity pre-filter.
    """
    mean   = boundary.mean(axis=0)
    cov    = np.cov(boundary.T)
    covinv = np.linalg.inv(cov)

    dists = np.array([_mahal(b, mean, covinv) for b in boundary])
    print(f"Mahalanobis distances: mean={dists.mean():.2f}  std={dists.std():.2f}")
    print("If std is small, the valid region is well-described by an ellipsoid.")

    return mean, covinv


def _mahal(x, mean, covinv):
    d = x - mean
    return np.sqrt(d @ covinv @ d)


def is_probably_valid(x: np.ndarray, mean: np.ndarray,
                      covinv: np.ndarray, threshold: float = 3.0) -> bool:
    """Fast pre-filter — skip model evaluation if clearly outside valid region."""
    return _mahal(x, mean, covinv) < threshold


# ── 6. Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1: gather samples
    X, Y = gather_samples(n_target=5_000)
    np.save("X.npy", X)
    np.save("Y.npy", Y)

    # Step 2: inspect thresholds (run once, tune is_output_valid, re-sample)
    find_thresholds(Y)

    # Step 3: filter outliers
    X_clean, Y_clean = filter_outliers(X, Y)
    np.save("X_clean.npy", X_clean)
    np.save("Y_clean.npy", Y_clean)

    # Step 4: fit surrogate — try degree 3, fall back to MLP if R² < 0.95
    surrogate, output_scaler, r2 = fit_surrogate(X_clean, Y_clean, degree=3)

    # Step 5: scout validity boundary (optional)
    x0 = X_clean.mean(axis=0)   # centroid as safe starting point
    boundary = scout_boundary(x0, n_directions=300)
    np.save("boundary.npy", boundary)
    mean, covinv = fit_mahalanobis(boundary)

    X, Y = gather_samples(n_target=100)  # small quick run