import numpy as np
import geopandas as gpd
from hydrosensordesign.selection.qr import qr_selection
from hydrosensordesign.utils import *
from hydrosensordesign.visualization import *

def select_sensors(
    X_train,
    boundaries,
    basin_field,
    flowlines,
    flow_id_field,
    existing_sensors,
    expansion = False,
    quotas=None,
    weights=None,
    ):
    """
    Unified entry point for QR-based sensor network design.

    ### Parameters
        X: Cleaned (n, m) matrix where n is days and m is potential site
        coords: String Tuple of all potential sites (lat, lon)
        site_id: potential site identifier number
        boundaries: boundary for basin-qr
        basin_field: field name for boundary
        existing_sensors: 
        weights:
        
    ### Returns
        sensor_location: column indices from qr-decomp
    """
    # --------------------------------------------------
    # (1) Assign each site to a basin
    # --------------------------------------------------
    points_with_basin = assign_points_to_basins(flowlines, flow_id_field, boundaries, basin_field,flowlines.crs)
    # --------------------------------------------------
    # (2) Filter existing sensors
    # --------------------------------------------------
    '''
    existing_sensors = existing_sensors.merge(
                                        points_with_basin[[flow_id_field, "idx"]],
                                        on=flow_id_field,
                                        how="left"
                                        )
    existing_sensors = points_with_basin[points_with_basin[flow_id_field].isin(existing_sensors[flow_id_field])]
    '''
    # --------------------------------------------------
    # (3) Quotas are determined by existing network per basin 
    # Note: quotas will be an input parameter where users can predefine the number of sensor per basin as a dict 
    # --------------------------------------------------
    if quotas is None:
        quotas = quotas_from_existing(existing_sensors, boundaries, basin_field)
    # --------------------------------------------------
    # (5) QR-decomp
    # --------------------------------------------------
    sensor_location = qr_selection(
                        X_train,
                        points_with_basin,
                        flow_id_field,
                        basin_field,
                        quotas,
                        existing_sensors,
                        expansion,
                        weights
                        )
    return sensor_location

def recon(X_train, X_test, sensor_location, n_sensors):
    """Creat reconstructed matrix for given sensor locations"""
    N_sensors = X_test.shape[1]
    all_sensors = np.arange(N_sensors)
    selected_sensors = sensor_location[:n_sensors]
    non_selected_sensors = np.setdiff1d(all_sensors, selected_sensors)

    X_train_selected = X_train[:, selected_sensors]  
    X_test_selected = X_test[:, selected_sensors]

    solution = np.linalg.lstsq(X_train_selected.T, X_test_selected.T, rcond=None)[0]
    X_test_reconstructed = solution.T @ X_train
    X_test_reconstructed = np.maximum(X_test_reconstructed, 1e-10)

    return X_test_reconstructed

def recon_evaluation(X_test, X_test_reconstructed, mode='nnse'):
    if mode == 'rmse':
        rmse = np.sqrt(np.mean((X_test - X_test_reconstructed) ** 2, axis=0))
        return rmse
    if mode == 'rel_error':
        rel_error = np.linalg.norm(X_test_reconstructed - X_test,'fro') / np.linalg.norm(X_test,'fro')
        return rel_error
    
    ss_res = np.sum((X_test - X_test_reconstructed) ** 2, axis=0)
    ss_tot = np.sum((X_test - np.mean(X_test, axis=0)) ** 2, axis=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        if mode =='r2':
            r_squared = np.where(ss_tot != 0, 1 - (ss_res / ss_tot), np.nan)
            return r_squared
        nse = np.where(ss_tot != 0, 1 - (ss_res / ss_tot), np.nan)
        if mode == 'nse':
            return nse
        if mode=='nnse':    
            nnse = 1 / (2 - nse)
            return nnse
        
__all__ = ["select_sensors", "recon","recon_evaluation"]