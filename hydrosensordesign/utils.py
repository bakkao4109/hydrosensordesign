import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pyproj import CRS

'''
notes:
# for geographic coord sys (lat,lon), longitude wrapping
# - create separate functions for geo and proj crs 
# for projected coord sys (x,y), filter out points out-of-bounds 
'''

# ----------------------------------------------------------
# XX. Coordinates to Point Geometry
# ----------------------------------------------------------

def coords_to_points(coords, site_id, idx, bnd_crs):
    """
    Convert coordinate strings '(lat, lon)' into a GeoDataFrame
    """
    lat_list, lon_list = [], []

    for s in coords:
        stripped = s.strip("()")
        lat, lon = map(float, stripped.split(","))

        # Fix 0–360 longitudes to -180 to 180 for WGS84
        if lon > 180:
            lon = ((lon + 180) % 360) - 180
        lat_list.append(lat)
        lon_list.append(lon)

    points_gdf = gpd.GeoDataFrame(
        {
            "id": site_id,
            "idx": idx,
            "coord": coords,
            "lat": lat_list,
            "lon": lon_list,
        },
        geometry=gpd.points_from_xy(lon_list, lat_list),
        crs=bnd_crs
    )

    return points_gdf

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

def assign_points_to_basins(points_gdf: gpd.GeoDataFrame, 
                            flow_id_field,
                            boundaries, 
                            basin_field,
                            coord_crs=None):
    """
    Assign basin values to points using spatial join + nearest-basin fallback.
    Includes CRS validation and safe reprojection.
    """
    # -----------------------------
    # 1. CRS validation
    # -----------------------------
    # Reproject points if needed
    if points_gdf.crs != boundaries.crs:
        warnings.warn("boundaries and points_gdf have mismatching CRS. points_gdf will be reprojected to the CRS from boundaries")
        points_gdf = points_gdf.to_crs(boundaries.crs)

    # Set Projection Coordinate
    if coord_crs is None:
        warnings.warn('must provide crs for lat/lon')

    # -----------------------------
    # 2. Spatial join 
    # -----------------------------
    points_with_basin = gpd.sjoin(
        points_gdf,
        boundaries[[basin_field, "geometry"]],
        how="left",
        predicate="within",
    )

    # -----------------------------
    # 3. Nearest-basin assignment
    # -----------------------------
    missing_mask = points_with_basin[basin_field].isna()
    n_missing = missing_mask.sum()

    if n_missing > 0:
        print(f"Found {n_missing} gauges not inside any basin. Assigning nearest basin...")

        pts_proj = points_with_basin.loc[missing_mask].to_crs(coord_crs)
        basins_proj = boundaries.to_crs(coord_crs)

        for idx in pts_proj.index:
            geom = pts_proj.loc[idx].geometry
            dists = basins_proj.geometry.distance(geom)

            nearest_idx = dists.idxmin()

            # Assign basin attributes
            points_with_basin.at[idx, basin_field] = boundaries.at[nearest_idx, basin_field]
    # Cleanup
    if "index_right" in points_with_basin:
        points_with_basin = points_with_basin.drop(columns=["index_right"])
    points_with_basin[flow_id_field] = points_with_basin[flow_id_field].astype(int)
    
    return points_with_basin

# ----------------------------------------------------------
# XX. Quota  
# ----------------------------------------------------------

def quotas_from_existing(existing_sensors, boundaries, basin_field):
    """
    Quotas = how many existing sensors fall in each basin.
    points_with_basin must contain columns: ["id", basin_field]
    """
    existing_pts = gpd.sjoin(existing_sensors,
                            boundaries[[basin_field,"geometry"]],
                            how='left',
                            predicate='within')
    if "index_right" in existing_pts:
        existing_pts = existing_pts.drop(columns=["index_right"])
    quotas = existing_pts.groupby(basin_field).size().to_dict()

    return quotas

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