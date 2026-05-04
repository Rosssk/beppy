

import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ── 1. Load and clean data ────────────────────────────────────────────────────
X = np.load("X.npy")
Y = np.load("Y.npy")

# MAX_ROTATION_RAD = np.deg2rad(15)
# MAX_TRANSLATION  = np.array([0.1, 0.1, 0.5])

# mask = (
#     (np.abs(Y[:, 0]) < MAX_TRANSLATION[0]) &
#     (np.abs(Y[:, 1]) < MAX_TRANSLATION[1]) &
#     (np.abs(Y[:, 2]) < MAX_TRANSLATION[2]) &
#     (np.abs(Y[:, 3]) < MAX_ROTATION_RAD) &
#     (np.abs(Y[:, 4]) < MAX_ROTATION_RAD) &
#     (np.abs(Y[:, 5]) < MAX_ROTATION_RAD)
# )
# X, Y = X[mask], Y[mask]
# print(f"Using {mask.sum()} / {len(mask)} samples after filtering")

# ── 2. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# ── 3. Scale outputs ──────────────────────────────────────────────────────────
# Fit scaler on training set only — never fit on test data
output_scaler = StandardScaler()
Y_train_scaled = output_scaler.fit_transform(Y_train)
Y_test_scaled  = output_scaler.transform(Y_test)

# ── 4. Build and fit surrogate ────────────────────────────────────────────────
surrogate = Pipeline([
    ("scale", StandardScaler()),
    ("poly",  PolynomialFeatures(degree=4, include_bias=False)),
    ("ridge", Ridge(alpha=1e-3)),
])
surrogate.fit(X_train, Y_train_scaled)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
Y_pred_scaled = surrogate.predict(X_test)
Y_pred        = output_scaler.inverse_transform(Y_pred_scaled)

r2 = r2_score(Y_test, Y_pred, multioutput="raw_values")
print("R² per output:")
for i, v in enumerate(r2):
    print(f"  output {i}: {v:.6f}")

# ── 6. Predict new points ─────────────────────────────────────────────────────
def predict(x_new: np.ndarray) -> np.ndarray:
    """x_new: (n, 10) or (10,) — returns predictions in original units."""
    x_new = np.atleast_2d(x_new)
    return output_scaler.inverse_transform(surrogate.predict(x_new))

