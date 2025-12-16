import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.linalg import qr


def per_basin_qr_selection(X, points_with_basin, basin_field, quotas, weights=None, existing_sensors=None):
    """
    Perform QR-pivot sensor selection for each basin.
    
    Parameters
    ----------
    X : np.ndarray or list, data matrix
    points_with_basin : GeoDataFrame with basin assignments 
        With columns: 'id'
    basin_field : column name with basin names from points_with_basin  
    quotas : dict{basin_name:number}

    Returns
    -------
    Tuple of (selected_sensors_df, selected_indices_list)
    """
    selected_rows = []
    selected_sensor_indices = []

    for basin, grp in points_with_basin.groupby(basin_field):
        k = quotas.get(basin, 0)
        if k == 0:
            continue

        cols = grp["id"].to_numpy()
        if len(cols) == 0:
            continue

        k = min(k, len(cols))

        X_basin = X[:, cols]

        # Find fixed sensors inside basin
        fixed_indices = []
        if existing_sensors is not None:
            fixed_indices = [c for c in existing_sensors if c in cols]

        # Weighted matrix
        if weights is not None:
            X_w = X_basin @ np.diag(weights[cols])  
        else:
            X_w = X_basin

        # QR with fixed sensors
        if fixed_indices:
            free_indices = [c for c in cols if c not in fixed_indices]
            
            A_F = X_w[:, fixed_indices]
            A_R = X_w[:, [np.where(cols == cc)[0][0] for cc in free_indices]]

            QF, _ = np.linalg.qr(A_F)
            projection = QF @ (QF.T @ A_R)
            A_R_prime = A_R - projection

            _, _, pivots = qr(A_R_prime, mode="economic", pivoting=True)
            r_free = k - len(fixed_indices)
            chosen = fixed_indices + [free_indices[pivots[i]] for i in range(r_free)]

        else:
            _, _, pivots = qr(X_w, mode="economic", pivoting=True)
            chosen = [cols[pivots[i]] for i in range(k)]

        selected_sensor_indices.extend(chosen)
        sel = grp.loc[grp["id"].isin(chosen),[basin_field,"matrix_col","lat","lon"]]
        selected_rows.append(sel)

    selected_sensors = pd.concat(selected_rows,ignore_index=True) if selected_rows else pd.DataFrame()
    print(f"\nSelected {len(selected_sensor_indices)} optimal sensor locations")

    return selected_sensors, selected_sensor_indices