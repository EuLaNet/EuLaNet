"""
correspondence.py — Eulerian-Lagrangian spatial-temporal correspondence
"""
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

def sample_fields_vtk(mesh, positions, Eulerian_FIELDS):
    # positions: (N,2) -> 3D
    pts3d=np.column_stack([np.asarray(positions)[:,:2], np.zeros(len(positions))])
    probe=pv.PolyData(pts3d)
    sampled=probe.sample(mesh)
    out={}
    for name in Eulerian_FIELDS:
        if name not in sampled.point_data: continue
        arr=np.asarray(sampled.point_data[name])
        if arr.ndim==1: out[f"E_{name}"]=arr.astype(float)
        else:
            for c in range(arr.shape[1]):
                out[f"E_{name}_{c}"]=arr[:,c].astype(float)
    return out

def build_correspondence_mask(lag_valid, euler_valid, window_valid=None):
    mask=lag_valid & euler_valid
    if window_valid is not None: mask &= window_valid
    return mask

def nearest_correspondence(euler_coords, lag_positions, radius=1.5):
    tree=cKDTree(euler_coords)
    dist, idx=tree.query(lag_positions, k=1)
    valid=dist<=radius
    return idx, dist, valid
