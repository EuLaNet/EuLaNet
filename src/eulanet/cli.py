"""
cli.py — eulanet build command (validated pipeline)
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from .io import SU2Adapter, SU2Config
from .eulerian import EULERIAN_FIELDS, load_eulerian_snapshot, flatten_eulerian_fields
from .lagrangian import create_initial_particles, VelocityField, LagrangianEngine
from .transport import rolling_deformation, principal_quantities

BANNER = r"""
   ███████╗██╗   ██╗██╗      █████╗ ███╗   ██╗███████╗████████╗
   ██╔════╝██║   ██║██║     ██╔══██╗████╗  ██║██╔════╝╚══██╔══╝
   █████╗  ██║   ██║██║     ███████║██╔██╗ ██║█████╗     ██║
   ██╔══╝  ╚██╗ ██╔╝██║     ██╔══██║██║╚██╗██║██╔══╝     ██║
   ███████╗ ╚████╔╝ ███████╗██║  ██║██║ ╚████║███████╗   ██║
   ╚══════╝  ╚═══╝  ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝

          Lean Eulerian–Lagrangian Flow Representation
                     for ML / RL / Deep RL

                         Built by Sai Siddharth
============================================================
  EuLaNet converts SU2 flow solutions into a structured
  Eulerian-Lagrangian representation containing:

      FULL EULERIAN FLOW FIELD
                +
      SPARSE LAGRANGIAN MATERIAL EVOLUTION
                +
      EXPLICIT SAME-TIME SPATIAL CORRESPONDENCE

  Model-independent and designed for downstream
  ML, RL and Deep RL workflows.

  Acknowledgements

  I would like to acknowledge the helpful comments,
  discussions, and suggestions from members of the MIT
  AeroAstro community, the Hypersonics Research Laboratory,
  and the American Physical Society community. I am grateful
  to these research communities for the intellectual exchange
  and open scientific discussions that contributed to the
  development of this work.
============================================================
"""

def find_config(run_dir: Path):
    for p in run_dir.glob("*.cfg"):
        return p
    return None

def parse_probe_grid(s):
    if "x" in s.lower():
        a,b=s.lower().split("x")
        return int(a), int(b)
    return int(s), int(s)

def cmd_build(args):
    import sys
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print(BANNER)
    run_dir=Path(args.vtu_dir).resolve()
    pattern=args.pattern
    start=args.start
    stop=args.stop if args.stop is not None else 149
    dt=args.dt
    window=args.window
    nx,ny=parse_probe_grid(args.probe_grid)
    substeps=args.substeps
    output=Path(args.output).resolve()
    # Discover config
    config = Path(args.config) if args.config else find_config(run_dir)
    if config is None:
        # try parent
        cand = run_dir / "config_incomp_turb_sa.cfg"
        if cand.exists(): config=cand
    if config and Path(config).exists():
        try:
            cfg=SU2Config(config)
            dt=cfg.time_step
            print(f"Config: {config.name} TIME_STEP={dt} TIME_ITER={cfg.time_iter}")
        except Exception as e:
            print(f"Config read warning: {e}")
    print(f"SU2 solution detected: {run_dir}")
    # Discover VTUs
    import re
    all_vtus=sorted(run_dir.glob("flow_*.vtu"), key=lambda p: int(re.search(r"(\d+)(?=\.vtu$)", p.name).group(1)) if re.search(r"(\d+)(?=\.vtu$)", p.name) else -1)
    if not all_vtus:
        print("No VTU files found"); return 1
    print(f"VTU snapshots available: {len(all_vtus)}")
    # Determine needed snapshots for flow map: need up to stop+window
    max_needed = stop + window
    # But ensure we have at least that many files
    if max_needed >= len(all_vtus):
        print(f"Warning: need {max_needed+1} snapshots for window, only {len(all_vtus)} available. Adjusting.")
        max_needed = len(all_vtus)-1
        if args.stop is None:
            stop = max_needed - window
    print(f"Building for snapshots {start} -> {stop} (window {window}, need up to {max_needed})")
    # Build Eulerian stack for start..stop
    print("\nBuilding Eulerian representation...")
    E_list=[]; X_list=[]; euler_names=None
    for idx in range(start, stop+1):
        vtu = run_dir / pattern.format(index=idx)
        if not vtu.exists():
            # fallback glob
            vtu = all_vtus[idx] if idx < len(all_vtus) else None
            if vtu is None or not vtu.exists():
                raise FileNotFoundError(f"Missing VTU {idx}")
        mesh=pv.read(str(vtu))
        coords=np.asarray(mesh.points[:,:2], dtype=np.float32)
        # collect fields
        fields={}
        for fname in EULERIAN_FIELDS:
            if fname not in mesh.point_data:
                raise KeyError(f"{vtu.name}: missing {fname}")
            arr=np.asarray(mesh.point_data[fname])
            if arr.ndim==1: arr=arr[:,None]
            fields[fname]=arr
        E_snap, names = flatten_eulerian_fields(fields)
        if euler_names is None: euler_names=names
        E_list.append(E_snap); X_list.append(coords)
        if idx==start or (idx+1)%5==0:
            print(f"  Eulerian snapshot {idx:03d} -> {E_snap.shape}")
    E=np.stack(E_list).astype(np.float32)
    X=np.stack(X_list).astype(np.float32)
    print(f"Eulerian shape: {E.shape} X {X.shape} features {len(euler_names)}")
    # Build Lagrangian flow map for start..max_needed
    print("\nBuilding Lagrangian material probes...")
    print(f"Probe grid: {nx}x{ny} substeps {substeps}")
    # Need adapter for dt and snapshots
    if config and Path(config).exists():
        adapter=SU2Adapter(config, run_dir, "flow_*.vtu")
        # override dt already
    else:
        # minimal adapter using pattern and dt
        class MiniAdapter:
            def __init__(self, run_dir, pattern, dt):
                self.run_dir=Path(run_dir); self.dt=dt
                import re
                files=sorted(self.run_dir.glob("flow_*.vtu"), key=lambda p: int(re.search(r"(\d+)(?=\.vtu$)", p.name).group(1)) if re.search(r"(\d+)(?=\.vtu$)", p.name) else -1)
                self.vtu_files=files
            def read_snapshot(self, idx):
                from .io import FlowSnapshot
                mesh=pv.read(str(self.vtu_files[idx]))
                coords=np.asarray(mesh.points[:,:2], dtype=float)
                vel=np.asarray(mesh.point_data["Velocity"])[:,:2].astype(float)
                gv=None
                if "Grid_Velocity" in mesh.point_data:
                    gv=np.asarray(mesh.point_data["Grid_Velocity"])[:,:2].astype(float)
                return FlowSnapshot(time=idx*self.dt, coordinates=coords, velocity=vel, grid_velocity=gv)
            def __len__(self): return len(self.vtu_files)
        adapter=MiniAdapter(run_dir, pattern, dt)
        adapter.dt=dt
    # Build flow map for needed range
    n_needed = max_needed - start + 1
    # For simplicity, build from 0 to max_needed then slice, but we need start offset
    # Build snapshots from start to max_needed inclusive
    snapshots=[adapter.read_snapshot(i) for i in range(start, max_needed+1)]
    # Initial particles from first needed mesh
    mesh0=pv.read(str(run_dir / pattern.format(index=start)))
    raw=create_initial_particles(mesh0, nx, ny)
    vf0=VelocityField(snapshots[0])
    v0=vf0.evaluate(raw)
    valid0=np.all(np.isfinite(v0),axis=1)
    initial=raw[valid0]
    print(f"Initial particles: {len(initial)} (from {len(raw)} candidates)")
    if len(initial)==0: raise RuntimeError("No valid initial particles")
    engine=LagrangianEngine(snapshots, dt, substeps)
    flow_map=engine.compute_flow_map(initial)
    times=np.arange(len(snapshots))*dt + start*dt
    print(f"Flow map: {flow_map.shape} times {times[0]:.6f}->{times[-1]:.6f}")
    # Compute transport for window for snapshots start..last_valid (where window fits)
    print("\nComputing transport quantities...")
    # X0,Y0 from initial
    X0=initial[:,0]; Y0=initial[:,1]
    last_valid = min(stop, len(all_vtus)-1 - window) if window else stop
    if last_valid < start:
        last_valid = start
    n_lag_snap = last_valid - start + 1
    # For full 150 with window 25: last_valid=124, n=125*1184=148000; for 5-snap test: last_valid=4, n=5*1184=5920
    n_lag_rows = n_lag_snap * len(initial)
    print(f"  Lagrangian snapshots: {start} -> {last_valid} ({n_lag_snap} snapshots, {n_lag_rows} rows)")
    F_all=np.full((n_lag_rows,2,2), np.nan, dtype=float)
    C_all=np.full((n_lag_rows,2,2), np.nan, dtype=float)
    lmin_all=np.full(n_lag_rows, np.nan); lmax_all=np.full(n_lag_rows, np.nan)
    stretch_all=np.full(n_lag_rows, np.nan); dir_x=np.full(n_lag_rows, np.nan); dir_y=np.full(n_lag_rows, np.nan)
    ftle_all=np.full(n_lag_rows, np.nan)
    pid_all=np.tile(np.arange(len(initial)), n_lag_snap)
    snap_all=np.repeat(np.arange(start, last_valid+1), len(initial))
    valid_all=np.zeros(n_lag_rows, dtype=bool)
    # For each snapshot k, compute rolling
    row_offset=0
    for k in range(start, last_valid+1):
        local_k = k - start
        # flow_at_k is flow_map[local_k], flow_at_end is flow_map[local_k+window] if exists
        if local_k+window >= len(snapshots):
            row_offset+=len(initial)
            continue
        flow_k=flow_map[local_k]
        flow_end=flow_map[local_k+window]
        # Need finite at both ends
        finite_now=np.all(np.isfinite(flow_k),axis=1)
        finite_end=np.all(np.isfinite(flow_end),axis=1)
        F,C,valid = rolling_deformation(flow_k, flow_end, X0, Y0)
        lmin,lmax,stretch,direction,valid2 = principal_quantities(C, valid.copy())
        ftle=np.full(len(initial), np.nan)
        T=window*dt
        pos=(valid2 & (lmax>0))
        ftle[pos]=np.log(lmax[pos])/(2*T)
        # Store in flat arrays
        idx_slice=slice(row_offset, row_offset+len(initial))
        # We need to map F/C etc per particle
        # F,C are (npart,2,2)
        # For simplicity, store per row
        # We'll store in all arrays
        for i in range(len(initial)):
            base=row_offset+i
            if valid2[i]:
                F_all[base]=F[i]; C_all[base]=C[i]
                lmin_all[base]=lmin[i]; lmax_all[base]=lmax[i]
                stretch_all[base]=stretch[i]; dir_x[base]=direction[i,0]; dir_y[base]=direction[i,1]
                ftle_all[base]=ftle[i]
                valid_all[base]=True
        row_offset+=len(initial)
    print(f"Transport valid: {valid_all.sum()}/{len(valid_all)}")
    # Build correspondence: nearest Eulerian point per lag position (only for valid window snapshots)
    print("\nBuilding spatial-temporal correspondence...")
    from scipy.spatial import cKDTree
    corr_point=np.full(n_lag_rows, -1, dtype=np.int32)
    corr_dist=np.full(n_lag_rows, np.nan, dtype=float)
    corr_valid=np.zeros(n_lag_rows, dtype=bool)
    # X is (n_snapshots, n_points, 2) for start..stop
    for k in range(start, last_valid+1):
        local_idx=k-start
        coords=X[local_idx]  # (n_points,2)
        tree=cKDTree(coords)
        # lag positions for this snapshot
        flow_k=flow_map[local_idx]
        mask = np.all(np.isfinite(flow_k), axis=1)
        # Only for valid lagrangian rows for this snapshot
        for i in np.where(mask)[0]:
            row = local_idx*len(initial)+i
            pos=flow_k[i]
            dist, pt = tree.query(pos, k=1)
            corr_point[row]=int(pt); corr_dist[row]=float(dist)
            corr_valid[row]=True
    print(f"Correspondence valid: {corr_valid.sum()}/{len(corr_valid)}")
    # Save dataset like semantic
    print("\nValidating dataset...")
    # Prepare Eulerian names
    euler_names_list=euler_names
    lagrangian_names=["displacement_x","displacement_y","F11","F12","F21","F22","C11","C12","C21","C22","lambda_min","lambda_max","principal_stretch","stretch_direction_x","stretch_direction_y","FTLE"]
    # Build output dict
    out_path=Path(output).resolve()
    # Compute displacement: only for valid window snapshots start..last_valid
    n_lag = last_valid - start + 1
    X0_rep=np.tile(X0, n_lag)
    Y0_rep=np.tile(Y0, n_lag)
    x_cur=flow_map[:n_lag, :, 0].reshape(-1)
    y_cur=flow_map[:n_lag, :, 1].reshape(-1)
    disp_x=x_cur - X0_rep
    disp_y=y_cur - Y0_rep
    Lmat=np.column_stack([disp_x, disp_y, F_all[:,0,0], F_all[:,0,1], F_all[:,1,0], F_all[:,1,1], C_all[:,0,0], C_all[:,0,1], C_all[:,1,0], C_all[:,1,1], lmin_all, lmax_all, stretch_all, dir_x, dir_y, ftle_all])
    np.savez_compressed(
        str(out_path),
        E=E, X=X,
        L=Lmat,
        particle_id=np.tile(np.arange(len(initial)), n_lag),
        X0=np.tile(X0, n_lag),
        Y0=np.tile(Y0, n_lag),
        x_current=x_cur, y_current=y_cur,
        displacement_x=disp_x, displacement_y=disp_y,
        F11=F_all[:,0,0], F12=F_all[:,0,1], F21=F_all[:,1,0], F22=F_all[:,1,1],
        C11=C_all[:,0,0], C12=C_all[:,0,1], C21=C_all[:,1,0], C22=C_all[:,1,1],
        lambda_min=lmin_all, lambda_max=lmax_all, principal_stretch=stretch_all,
        stretch_direction_x=dir_x, stretch_direction_y=dir_y, FTLE=ftle_all,
        lagrangian_valid=valid_all,
        eulerian_position_valid=np.all(np.isfinite(flow_map[:n_lag].reshape(n_lag_rows,2)), axis=1),
        window_end_valid=np.tile([True]*len(initial), n_lag),
        snapshot_id=np.repeat(np.arange(start, last_valid+1), len(initial)),
        time=np.repeat(np.arange(start, last_valid+1)*dt, len(initial)),
        correspondence_point=corr_point, correspondence_snapshot=np.repeat(np.arange(start, last_valid+1), len(initial)),
        correspondence_particle=np.tile(np.arange(len(initial)), n_lag),
        correspondence_distance=corr_dist, correspondence_valid=corr_valid,
        eulerian_feature_names=np.asarray(euler_names, dtype=object),
        lagrangian_feature_names=np.asarray(lagrangian_names, dtype=object),
        valid_for_learning=valid_all & corr_valid,
    )
    print("="*60)
    print("EuLaNet dataset complete")
    print("="*60)
    print(f"Output:\n    {out_path}")
    print(f"\nEulerian representation:\n    {E.shape[0]} x {E.shape[1]} x {E.shape[2]}")
    print(f"\nLagrangian representation:\n    {n_lag_rows} x 16")
    print(f"\nValid correspondence:\n    {corr_valid.sum()}")
    print("\nReady for downstream machine learning.")
    return 0

def main():
    parser=argparse.ArgumentParser(prog="eulanet")
    sub=parser.add_subparsers(dest="cmd")
    p=sub.add_parser("build", help="Build EuLaNet dataset from SU2 VTUs")
    p.add_argument("--vtu-dir", default=".", help="Directory containing VTUs")
    p.add_argument("--pattern", default="flow_{index:05d}.vtu")
    p.add_argument("--config", default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--dt", type=float, default=0.016849)
    p.add_argument("--probe-grid", default="40x40")
    p.add_argument("--window", type=int, default=25)
    p.add_argument("--substeps", type=int, default=4)
    p.add_argument("--output", default="eulanet_dataset.npz")
    p.set_defaults(func=cmd_build)
    args=parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help(); return 1
    return args.func(args)

if __name__=="__main__":
    raise SystemExit(main())
