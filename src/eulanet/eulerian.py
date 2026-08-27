"""
eulerian.py — Full Eulerian environment extraction (from build_eulerian_baseline.py, build_semantic_el_dataset.py)
"""
from pathlib import Path
import numpy as np
import pyvista as pv

EULERIAN_FIELDS = {
    "Pressure": 1, "Velocity": 3, "Nu_Tilde": 1, "Grid_Velocity": 3,
    "Pressure_Coefficient": 1, "Density": 1, "Laminar_Viscosity": 1,
    "Heat_Capacity": 1, "Thermal_Conductivity": 1, "Temperature": 1,
    "Skin_Friction_Coefficient": 3, "Heat_Flux": 1, "Y_Plus": 1, "Eddy_Viscosity": 1,
}

def flatten_field(data, name):
    arr=np.asarray(data)
    if arr.ndim==1: arr=arr[:,None]
    elif arr.ndim==2: pass
    else: raise ValueError(f"Unexpected shape for {name}: {arr.shape}")
    return arr.astype(np.float32)

def load_eulerian_snapshot(vtu_path: Path):
    mesh=pv.read(str(vtu_path))
    coords=np.asarray(mesh.points[:,:2], dtype=np.float32)
    fields={}
    for name, comp in EULERIAN_FIELDS.items():
        if name not in mesh.point_data:
            raise KeyError(f"{vtu_path.name}: field '{name}' not found. Available: {list(mesh.point_data.keys())}")
        arr=flatten_field(mesh.point_data[name], name)
        if arr.shape[1]!=comp: raise ValueError(f"{name} has {arr.shape[1]} comps, expected {comp}")
        fields[name]=arr
    return coords, fields

def flatten_eulerian_fields(fields):
    cols=[]; names=[]
    for fname, comp in EULERIAN_FIELDS.items():
        arr=fields[fname]
        for c in range(comp):
            cols.append(arr[:,c])
            names.append(fname if comp==1 else f"{fname}_{c}")
    return np.column_stack(cols).astype(np.float32), names

def build_eulerian_stack(vtu_dir: Path, pattern="flow_{index:05d}.vtu", start=0, stop=149):
    import re
    files=[]
    for i in range(start, stop+1):
        p=vtu_dir / pattern.format(index=i)
        if not p.exists(): raise FileNotFoundError(p)
        files.append(p)
    E_list=[]; X_list=[]
    for p in files:
        coords, fields=load_eulerian_snapshot(p)
        E,names=flatten_eulerian_fields(fields)
        E_list.append(E); X_list.append(coords)
    E=np.stack(E_list)
    X=np.stack(X_list)
    return E, X, names
