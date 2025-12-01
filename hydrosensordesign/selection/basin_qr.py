import numpy as np
import geopandas as gpd
from scipy.linalg import qr


def per_basin_qr_selection(
    X, points_with_basin, basin_field, quotas, weights=None, existing_sensors=None
):
    """Run QR independently for each basin."""
    selected = []

    for basin, grp in points_with_basin.groupby(basin_field):
        k = quotas.get(basin, 0)
        if k == 0:
            continue

        cols = grp["id"].to_numpy()
        if len(cols) == 0:
            continue

        X_basin = X[:, cols]

        # Find fixed sensors inside basin
        fixed = []
        if existing_sensors is not None:
            fixed = [c for c in existing_sensors if c in cols]

        # Weighted matrix
        X_w = X_basin @ np.diag(weights[cols]) if weights is not None else X_basin

        # QR with fixed sensors
        if fixed:
            A_F = X_w[:, fixed]
            free = [c for c in cols if c not in fixed]
            A_R = X_w[:, [np.where(cols == cc)[0][0] for cc in free]]

            QF, _ = np.linalg.qr(A_F)
            A_R_prime = A_R - QF @ (QF.T @ A_R)

            _, _, piv = qr(A_R_prime, mode="economic", pivoting=True)
            need = k - len(fixed)
            chosen = fixed + [free[piv[i]] for i in range(need)]

        else:
            _, _, piv = qr(X_w, mode="economic", pivoting=True)
            chosen = [cols[piv[i]] for i in range(k)]

        selected.extend(chosen)

    return set(selected)