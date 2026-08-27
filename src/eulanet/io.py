"""
io.py — SU2 config and VTU I/O (validated from lagrangian_flowmap.py, build_semantic_el_dataset.py)
"""
from __future__ import annotations
from pathlib import Path
import re
from dataclasses import dataclass
import numpy as np
import pyvista as pv

@dataclass
class FlowSnapshot:
    time: float
    coordinates: np.ndarray
    velocity: np.ndarray
    grid_velocity: np.ndarray | None

class SU2Config:
    def __init__(self, filename: Path):
        self.filename = Path(filename)
        if not self.filename.exists():
            raise FileNotFoundError(f"SU2 config not found:\n{self.filename}")
        self.values = self._read()
    def _read(self):
        values = {}
        with open(self.filename, "r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                line = raw_line.split("%")[0].strip()
                if not line or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip(); value = value.strip().split("%")[0].strip()
                values[key] = value
        return values
    def get(self, key, default=None): return self.values.get(key, default)
    def get_float(self, key, default=None):
        v=self.get(key,default)
        return float(v) if v is not None else None
    def get_int(self, key, default=None):
        v=self.get(key,default)
        return int(v) if v is not None else None
    @property
    def time_step(self):
        v=self.get_float("TIME_STEP")
        if v is None: raise ValueError("TIME_STEP not found in SU2 config.")
        return v
    @property
    def time_domain(self): return self.get("TIME_DOMAIN","NO").upper()
    @property
    def time_iter(self): return self.get_int("TIME_ITER")

class SU2Adapter:
    def __init__(self, config_file: Path, run_directory: Path, vtu_pattern: str="flow_*.vtu"):
        self.config = SU2Config(config_file)
        self.run_directory = Path(run_directory)
        self.vtu_files = self._discover_snapshots(vtu_pattern)
        if not self.vtu_files:
            raise FileNotFoundError(f"No VTU snapshots matching '{vtu_pattern}' found in {run_directory}")
        self.dt = self.config.time_step
    def _discover_snapshots(self, pattern):
        files = list(self.run_directory.glob(pattern))
        def snapshot_number(path):
            m=re.search(r"(\d+)(?=\.vtu$)", path.name)
            return int(m.group(1)) if m else -1
        files.sort(key=snapshot_number)
        return files
    def __len__(self): return len(self.vtu_files)
    def read_snapshot(self, index: int):
        filename = self.vtu_files[index]
        mesh = pv.read(str(filename))
        coordinates = np.asarray(mesh.points[:,:2], dtype=float)
        if "Velocity" not in mesh.point_data:
            raise KeyError(f"'Velocity' not found in {filename}. Available: {list(mesh.point_data.keys())}")
        velocity_raw = np.asarray(mesh.point_data["Velocity"])
        if velocity_raw.ndim!=2 or velocity_raw.shape[1]<2:
            raise ValueError(f"Unexpected Velocity shape: {velocity_raw.shape}")
        velocity = velocity_raw[:,:2].astype(float)
        grid_velocity=None
        if "Grid_Velocity" in mesh.point_data:
            grid_raw=np.asarray(mesh.point_data["Grid_Velocity"])
            grid_velocity=grid_raw[:,:2].astype(float)
        time=index*self.dt
        return FlowSnapshot(time=time, coordinates=coordinates, velocity=velocity, grid_velocity=grid_velocity)
    def report(self):
        print("="*70); print("SU2 INPUT ADAPTER"); print("="*70)
        print(f"Config:      {self.config.filename}")
        print(f"TIME_DOMAIN: {self.config.time_domain}")
        print(f"TIME_STEP:   {self.dt}")
        print(f"TIME_ITER:   {self.config.time_iter}")
        print(f"VTU snapshots: {len(self.vtu_files)}")
        first=self.read_snapshot(0)
        print(f"First snapshot: {self.vtu_files[0].name}")
        print(f"Mesh points: {len(first.coordinates):,}")
        print(f"Velocity field: {first.velocity.shape}")
        if first.grid_velocity is None: print("Grid velocity: NOT AVAILABLE")
        else:
            print("Grid velocity: AVAILABLE")
            print(f"Max grid speed: {np.max(np.linalg.norm(first.grid_velocity,axis=1)):.6e}")
        print("="*70)

def discover_vtus(run_dir: Path, pattern: str="flow_*.vtu"):
    files=list(Path(run_dir).glob(pattern))
    def num(p): 
        m=re.search(r"(\d+)(?=\.vtu$)", p.name)
        return int(m.group(1)) if m else -1
    files.sort(key=num)
    return files
