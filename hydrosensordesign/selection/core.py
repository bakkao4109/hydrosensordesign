import numpy as np
import geopandas as gpd
from .global_qr import global_qr_selection
from .basin_qr import per_basin_qr_selection
from hydrosensordesign.utils import (
    extract_coordinates, 
    align_points_to_grid, 
    assign_basins, 
    coords_to_points, 
    compute_basin_quotas
    )

def select_sensors(
    X,
    coords=None,
    boundaries=None,
    basin_field=None,
    per_basin_quota=None,
    existing_sensors=None,
    weights=None,
    r=None
    ):
    """
    Unified entry point for QR-based sensor network design.
    Handles:
        - global QR
        - per-basin QR
        - weights
        - existing sensors
        - reach or gridded data
    """

    n_sites = X.shape[1]
    all_cols = np.arange(n_sites)

    # --------------------------------------------------
    # (1) Global QR (no boundaries)
    # --------------------------------------------------
    if boundaries is None:
        if r is None:
            raise ValueError("Must provide number of sensors, r")
        return global_qr_selection(
            X=X,
            r=r,
            coords=coords,
            weights=weights,
            existing_sensors=existing_sensors,
        )
    # --------------------------------------------------
    # If boundaries are provided, basin_field is required
    # --------------------------------------------------
    if (boundaries is None) != (basin_field is None):
        raise ValueError(
            "Specify basin_field when providing boundaries. "
            "Example: basin_field='HUC6' or basin_field='RHI_NM'."
    )

    # --------------------------------------------------
    # (2) Per-basin mode — first assign each site to a basin
    # --------------------------------------------------
    points = coords_to_points(coords, all_cols, boundaries.crs)
    points_with_basin = assign_basins(points, boundaries, basin_field)

    # --------------------------------------------------
    # (3) Determine quota (minimum number of sensor per basin)
    # --------------------------------------------------
    quotas = compute_basin_quotas(
        points_with_basin, basin_field, r, per_basin_quota
    )

    # --------------------------------------------------
    # (4) Run per-basin QR
    # --------------------------------------------------
    selected_indices, selected_sensor_indices = per_basin_qr_selection(
        X,
        points_with_basin,
        basin_field,
        quotas,
        weights=weights,
        existing_sensors=existing_sensors,
    )

    return {
        "indices": selected_indices,
        "coords": [coords[i] for i in selected_indices],
        "basin_assign": points_with_basin.loc[selected_indices][basin_field].tolist(),
        "method": "per-basin QR (modular version)",
    }


