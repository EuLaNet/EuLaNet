"""
dataset.py — High-level builders for E, L, and correspondence (from build_semantic_el_dataset.py, build_full_el_dataset.py)
"""
from pathlib import Path
import json
import numpy as np
import pyvista as pv
from .eulerian import EULERIAN_FIELDS, load_eulerian_snapshot, flatten_eulerian_fields
from .transport import rolling_deformation, principal_quantities

def build_semantic_dataset(vtu_dir: Path, lagrangian_path: Path, output_path: Path, pattern="flow_{index:05d}.vtu", dt=0.016849, window=25, start=0, stop=149):
    lag=np.load(lagrangian_path)
    flow=lag["flow_map"]; ntime,npart,_=flow.shape
    # X0,Y0 from first frame or initial_positions
    if "initial_positions" in lag.files:
        X0=lag["initial_positions"][:,0]; Y0=lag["initial_positions"][:,1]
    else: X0=flow[0,:,0]; Y0=flow[0,:,1]
    snapshots=list(range(start, min(stop, ntime-1-window)+1))
    # For semantic builder we use the full EL dataset logic (see build_full_el_dataset.py)
    # Here we delegate to build_full style but store as semantic E[X,T] + L[particle]
    # Simplified: call full EL builder and reformat
    from .dataset import build_full_el_simple  # avoid circular, will define below
    return build_full_el_simple(vtu_dir, lag, output_path, pattern, dt, window, start, stop)

def build_full_el_simple(vtu_dir, lag_data, output_path, pattern, dt, window, start, stop):
    # Wrapper that reproduces build_full_el_dataset rolling logic
    import pandas as pd
    # This is a placeholder for the full logic already validated in build_full_el_dataset.py
    # For toolbox, we reuse that script's functions directly
    raise NotImplementedError("Use eulanet.cli build for full pipeline")
