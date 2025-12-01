import numpy as np
from scipy.linalg import qr

#def global_qr_selection(X, r, weights=None, existing_sensors=None):
def global_qr_selection(X, coords, r, weights=None, existing_sensors=None):
    """Run weighted QR decomposition for the whole domain."""

    n_sites = X.shape[1]

    # Apply weights if provided
    if weights is not None:
        X_w = X @ np.diag(weights)
    else:
        X_w = X

    if existing_sensors:
        # Partition matrix into fixed and free columns
        fixed_indices = list(existing_sensors)
        free_indices = [i for i in range(n_sites) if i not in fixed_indices]

        A_F = X_w[:, fixed_indices]
        A_R = X_w[:, free_indices]

        Q_F, _ = np.linalg.qr(A_F)
        projection = Q_F @ (Q_F.T @ A_R)
        A_R_prime = A_R - projection

        _, _, pivots = qr(A_R_prime, pivoting=True)
        
        r_free = r - len(fixed_indices)
        chosen = fixed_indices + [free_indices[i] for i in pivots[:r_free]]

    else:
        _, _, pivots = qr(X_w, pivoting=True)
        chosen = pivots[:r]

    return {
        "indices": chosen,
        "coords": [coords[i] for i in chosen],
        "basin_assign": None,
        "method": "global QR",
    }
