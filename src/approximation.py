import numpy as np
import torch
import gpytorch

from scipy.stats import qmc
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tqdm.contrib.concurrent import process_map

from numpy import cos, sin, pi
from rbm import PRBM

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

N_INPUTS   = 10
N_OUTPUTS  = 6
N_INITIAL  = 480
N_ADAPTIVE = 1000
BATCH_SIZE = 120
TOLERANCE  = 0.01
N_WORKERS  = 12
MAX_POINTS = 1200  # cap for GP stability

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────
# EXPENSIVE FUNCTION
# ─────────────────────────────────────────────────────────────

def expensive_function(r1, r2, r3, h1, h2, E, t, fx, fy, fz):
    Emod = E * 1e9
    A = pi * t**2
    I = pi * t**4 / 2

    p = PRBM(3)
    p.add_body('A', (0, 0, 0))
    p.add_body('B', (0, 0, h1))
    p.add_body('C', (0, 0, h2))

    n = 3
    for i in range(n):
        p.add_flexure('A', (r1*cos(i/n*2*pi), r1*sin(i/n*2*pi), 0),
                      'B', (r2*cos((i+1)/n*2*pi), r2*sin((i+1)/n*2*pi), 0))
        p.add_flexure('C', (r3*cos((i-.5)/n*2*pi), r3*sin((i-.5)/n*2*pi), 0),
                      'B', (r2*cos((i+.5)/n*2*pi), r2*sin((i+.5)/n*2*pi), 0))

    p.add_force('C', [fx, fy, -fz])
    p.solve_pose('BC', A, Emod, I)

    pos = p.bodies["B"].position
    rot = p.bodies["B"].angles

    return np.concatenate((pos, rot))

# ─────────────────────────────────────────────────────────────
# PARALLEL EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate_single(x_row):
    try:
        return expensive_function(*x_row)
    except Exception:
        return np.full(N_OUTPUTS, np.nan)

def evaluate_batch(X_rows, desc="Evaluating"):
    results = process_map(
        evaluate_single,
        list(X_rows),
        max_workers=N_WORKERS,
        desc=desc,
        unit="sample",
    )
    return np.array(results)

# ─────────────────────────────────────────────────────────────
# MULTITASK GP MODEL
# ─────────────────────────────────────────────────────────────

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_tasks):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(),
            num_tasks=n_tasks
        )

        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.MaternKernel(nu=2.5),
            num_tasks=n_tasks,
            rank=1  # reduced for stability
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────

def train_gp(X, Y, iters=50):
    train_x = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    train_y = torch.tensor(Y, dtype=torch.float32).to(DEVICE)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=Y.shape[1],
        noise_constraint=gpytorch.constraints.GreaterThan(1e-5)
    ).to(DEVICE)

    model = MultitaskGPModel(train_x, train_y, likelihood, Y.shape[1]).to(DEVICE)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(iters):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    return model.eval(), likelihood.eval()

# ─────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────

def predict_gp(model, likelihood, X):
    x = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    with torch.no_grad(), \
         gpytorch.settings.fast_pred_var(), \
         gpytorch.settings.cholesky_jitter(1e-4):

        preds = likelihood(model(x))

    mean = preds.mean.detach().cpu().numpy()
    var  = preds.variance.detach().cpu().numpy()

    return mean, var

# ─────────────────────────────────────────────────────────────
# ACQUISITION
# ─────────────────────────────────────────────────────────────

def acquisition(model, likelihood, X_pool):
    _, var = predict_gp(model, likelihood, X_pool)
    return np.sum(var, axis=1)

def get_batch_candidates(model, likelihood, batch_size):
    sampler = qmc.LatinHypercube(d=N_INPUTS)
    X_pool = sampler.random(5000)

    scores = acquisition(model, likelihood, X_pool)
    idx = np.argsort(-scores)[:batch_size]

    return X_pool[idx], np.max(scores)

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    lower = np.array([1e-4,1e-4,1e-4,1e-3,1e-3,1,1e-5,1e-1,1e-1,1e-1])
    upper = np.array([1e-1,1e-1,1e-1,1,1,1e3,1e-2,1e3,1e3,1e3])

    print("Initial sampling...")
    sampler = qmc.LatinHypercube(d=N_INPUTS, seed=42)
    X_train_raw = qmc.scale(sampler.random(N_INITIAL), lower, upper)
    Y_train = evaluate_batch(X_train_raw, "Initial")

    valid = ~np.isnan(Y_train).any(axis=1)
    X_train_raw = X_train_raw[valid]
    Y_train = Y_train[valid]

    # Validation
    X_val_raw = qmc.scale(qmc.LatinHypercube(d=N_INPUTS).random(48), lower, upper)
    Y_val = evaluate_batch(X_val_raw, "Validation")

    valid_val = ~np.isnan(Y_val).any(axis=1)
    X_val_raw = X_val_raw[valid_val]
    Y_val = Y_val[valid_val]

    # Scaling
    x_scaler = MinMaxScaler()
    X_train = x_scaler.fit_transform(X_train_raw)
    X_val   = x_scaler.transform(X_val_raw)

    # 🔴 IMPORTANT: normalize outputs
    y_scaler = StandardScaler()
    Y_train = y_scaler.fit_transform(Y_train)
    Y_val   = y_scaler.transform(Y_val)

    print(f"\n{'Iter':>5} {'Batch':>6} {'MaxVar':>10} {'R2':>10}")
    print("-"*40)

    for i in range(N_ADAPTIVE):

        # cap dataset size
        if len(X_train) > MAX_POINTS:
            idx = np.random.choice(len(X_train), MAX_POINTS, replace=False)
            X_train = X_train[idx]
            Y_train = Y_train[idx]

        model, likelihood = train_gp(X_train, Y_train)

        Y_pred_scaled, _ = predict_gp(model, likelihood, X_val)
        Y_pred = y_scaler.inverse_transform(Y_pred_scaled)
        Y_val_real = y_scaler.inverse_transform(Y_val)

        r2 = r2_score(Y_val_real, Y_pred)

        X_cand_scaled, max_var = get_batch_candidates(
            model, likelihood, BATCH_SIZE
        )

        print(f"{i+1:5d} {BATCH_SIZE:6d} {max_var:10.4f} {r2:10.4f}")

        if max_var < TOLERANCE:
            print("Converged.")
            break

        X_cand_real = x_scaler.inverse_transform(X_cand_scaled)
        Y_cand = evaluate_batch(X_cand_real, f"Iter {i+1}")

        valid = ~np.isnan(Y_cand).any(axis=1)

        X_train = np.vstack([X_train, X_cand_scaled[valid]])
        Y_train = np.vstack([Y_train, y_scaler.transform(Y_cand[valid])])

    # Final model
    model, likelihood = train_gp(X_train, Y_train)

    X_new = qmc.scale(qmc.LatinHypercube(d=N_INPUTS).random(5), lower, upper)
    X_new_scaled = x_scaler.transform(X_new)

    Y_pred_scaled, Y_var = predict_gp(model, likelihood, X_new_scaled)
    Y_pred = y_scaler.inverse_transform(Y_pred_scaled)

    print("\nPredictions:")
    print(Y_pred)

    print("\nVariance:")
    print(Y_var)