import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.linalg import qr


def qr_selection(X_train, points_with_basin, flow_id_field, basin_field, quotas, existing_sensors, expansion = False, weights=None):
    """
    Perform QR-pivot sensor selection
    
    Parameters
    ----------
    X : np.ndarray or list, data matrix
    points_with_basin : GeoDataFrame with basin assignments 
        With columns: 'id'
    basin_field : column name with basin names from points_with_basin  
    quotas : dict{basin_name:number}

    Returns
    -------
    ranked column indices using qr-decomp
    """
    selected_rows = []
    selected_sensor_idx = []

    for basin, grp in points_with_basin.groupby(basin_field):
        # grp: candidate grid points in basin
        # k: num of existing sensors in basin
        k = quotas.get(basin, 0)
        if k == 0:
            continue
        
        basin_idx = grp["idx"].to_numpy()
        basin_id = grp[flow_id_field].to_numpy()
        if len(basin_id) == 0:
            continue
        
        k = min(k,len(basin_idx))
        
        id_to_idx = dict(zip(basin_id, basin_idx))

        # QR with fixed sensors
        if expansion:
            # Find fixed sensors inside basin
            mask = existing_sensors[flow_id_field].isin(basin_id)
            fixed_id = existing_sensors.loc[mask,flow_id_field].tolist()
            fixed_idx = existing_sensors.loc[mask,"idx"].tolist()

            free_id = [cid for cid in basin_id if cid not in fixed_id]
            free_idx = [id_to_idx[cid] for cid in free_id]
            A_F = X_train[:, fixed_idx]
            A_R = X_train[:, free_idx]

            QF, _ = np.linalg.qr(A_F)
            projection = QF @ (QF.T @ A_R)
            A_R_prime = A_R - projection

            _, _, pivots = qr(A_R_prime, mode="economic", pivoting=True)
            free_n = max(0, k - len(fixed_idx))
            chosen_idx = fixed_idx + free_idx[pivots[:free_n]]
        else:
            _, _, pivots = qr(X_train[:,basin_idx], mode="economic", pivoting=True)
            chosen_idx = basin_idx[pivots[:k]]

        selected_sensor_idx.extend(chosen_idx)
        sel = (grp.set_index("idx").loc[chosen_idx].reset_index()[[basin_field,flow_id_field,"idx","lat","lon"]])
        selected_rows.append(sel)
    
    sensor_location = pd.concat(selected_rows,ignore_index=True) if selected_rows else pd.DataFrame()
    print(f"\nSelected {len(selected_sensor_idx)} optimal sensor locations")

    return sensor_location