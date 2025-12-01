'''
from hydrosensordesign.selection.global_selection import select_sensors_global
from hydrosensordesign.selection.basin_selection import select_sensors_per_basin

from hydrosensordesign.utils import (
    filter_invalid_columns,
    assign_basins,
    compute_quota,
    apply_weights,
    align_points_to_grid,
    extract_coordinates,
)
'''
from .selection.core import select_sensors

__all__ = ["select_sensors"]