import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ----------------------------------------------------------
# XX. PARSE "(lat, lon)" STRING LABELS
# ----------------------------------------------------------

def extract_coordinates(coord_labels):
    """
    Parse coordinate labels like "(23.45, 90.11)" into floats.

    Parameters
    ----------
    coord_labels : list of strings

    Returns
    -------
    lat : ndarray
    lon : ndarray
    """
    lat = []
    lon = []
    for s in coord_labels:
        s = s.strip().replace("(", "").replace(")", "")
        a, b = s.split(",")
        lat.append(float(a))
        lon.append(float(b))
    return np.array(lat), np.array(lon)

# ----------------------------------------------------------
# XX. ALIGN POINTS TO GRID
# ----------------------------------------------------------

def align_points_to_grid(points_gdf, lat_vals, lon_vals):
    """
    Align point locations to nearest (lat, lon) grid cells.

    Parameters
    ----------
    points_gdf : GeoDataFrame
        With columns: geometry (Point), gauge_lat, gauge_lon.
    lat_vals : array-like
    lon_vals : array-like

    Returns
    -------
    indices : ndarray of ints
        Column indices in the stacked X matrix.
    """
    lat_pts = points_gdf.geometry.y.values
    lon_pts = points_gdf.geometry.x.values

    lat_idx = np.abs(lat_vals[:, None] - lat_pts).argmin(axis=0)
    lon_idx = np.abs(lon_vals[:, None] - lon_pts).argmin(axis=0)

    # Convert the 2D grid to a flat column index
    grid_idx = lat_idx * len(lon_vals) + lon_idx
    return grid_idx

# ----------------------------------------------------------
# XX. BASIN ASSIGNMENT
# ----------------------------------------------------------

def coords_to_points(coords, indices, target_crs):
    """
    Convert coordinate strings '(lat, lon)' into a GeoDataFrame and assign CRS.
    Handles longitude wrap-around and CRS validation.
    """

    if target_crs is None:
        raise ValueError("Boundaries must have a valid CRS.")

    lat_list, lon_list = [], []

    for s in coords:
        stripped = s.strip("()")
        lat, lon = map(float, stripped.split(","))

        # Fix 0–360 longitudes to -180 to 180
        if lon > 180:
            lon = ((lon + 180) % 360) - 180

        lat_list.append(lat)
        lon_list.append(lon)

    points = gpd.GeoDataFrame(
        {
            "id": indices,
            "coord": coords,
            "lat": lat_list,
            "lon": lon_list,
        },
        geometry=gpd.points_from_xy(lon_list, lat_list),
        crs="EPSG:4326"
    )

    # Reproject points to boundary CRS if needed
    if points.crs != target_crs:
        points = points.to_crs(target_crs)

    return points

# ----------------------------------------------------------
# XX. BASIN ASSIGNMENT
# ----------------------------------------------------------

def assign_basins(points_gdf, boundaries, basin_field):
    """
    Assign basin values to points using spatial join + nearest-basin fallback.
    Includes CRS validation and safe reprojection.
    """
    # -----------------------------
    # 1. CRS validation
    # -----------------------------
    if boundaries.crs is None:
        raise ValueError("boundaries GeoDataFrame has no CRS. Please set one.")

    if points_gdf.crs is None:
        raise ValueError("points_gdf has no CRS. Please set one.")

    # Reproject points if needed
    if points_gdf.crs != boundaries.crs:
        warnings.warn("boundaries and points_gdf have mismatching CRS. points_gdf will be reprojected to the CRS from boundaries")
        points_gdf = points_gdf.to_crs(boundaries.crs)

    # -----------------------------
    # 2. Spatial join
    # -----------------------------
    joined = gpd.sjoin(
        points_gdf,
        boundaries[[basin_field, "geometry"]],
        how="left",
        predicate="within",
    )

    # -----------------------------
    # 3. Nearest-basin assignment
    # -----------------------------
    missing_mask = joined[basin_field].isna()
    n_missing = missing_mask.sum()

    if n_missing > 0:
        print(f"Found {n_missing} gauges not inside any basin. Assigning nearest basin...")

        # Reproject to projected CRS for Euclidean distances
        proj_crs = "EPSG:4326"

        pts_proj = joined.loc[missing_mask].to_crs(proj_crs)
        basins_proj = boundaries.to_crs(proj_crs)

        for idx in pts_proj.index:
            geom = pts_proj.loc[idx].geometry
            dists = basins_proj.geometry.distance(geom)

            nearest_idx = dists.idxmin()

            # Assign basin attributes
            joined.at[idx, basin_field] = boundaries.at[nearest_idx, basin_field]

    # Cleanup
    if "index_right" in joined:
        joined = joined.drop(columns=["index_right"])

    return joined

# ----------------------------------------------------------
# 3. BASIN QUOTA COMPUTATION
# ----------------------------------------------------------

def compute_basin_quotas(basin_assignments, total_sensors):
    """
    Compute proportional per-basin sensor quotas.

    Parameters
    ----------
    basin_assignments : array-like of basin names, length = n_sites
    total_sensors : int

    Returns
    -------
    quota : dict {basin_name : k}
    """
    basins, counts = np.unique(basin_assignments, return_counts=True)
    proportions = counts / counts.sum()

    raw = proportions * total_sensors
    quota = {b: int(max(1, np.round(k))) for b, k in zip(basins, raw)}

    return quota

# ----------------------------------------------------------
# XX. Reconstruct 
# ----------------------------------------------------------

def reconstruction_evaluation(X_train, X_test, sensor_location, n_sensors):
    """Evaluate reconstruction performance for given sensor locations"""
    N_sensors = X_test.shape[1]
    all_sensors = np.arange(N_sensors)
    selected_sensors = sensor_location[:n_sensors]
    non_selected_sensors = np.setdiff1d(all_sensors, selected_sensors)

    X_train_selected = X_train[:, selected_sensors]  
    X_test_selected = X_test[:, selected_sensors]

    solution = np.linalg.lstsq(X_train_selected.T, X_test_selected.T, rcond=None)[0]
    X_test_reconstructed = solution.T @ X_train
    X_test_reconstructed = np.maximum(X_test_reconstructed, 1e-10)

    rmse = np.sqrt(np.mean((X_test - X_test_reconstructed) ** 2, axis=0))
    relative_error = np.linalg.norm(X_test_reconstructed - X_test,'fro') / np.linalg.norm(X_test,'fro')

    return X_test_selected, X_test_reconstructed, selected_sensors, non_selected_sensors, rmse, relative_error