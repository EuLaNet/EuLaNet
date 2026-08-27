"""
lagrangian.py — Lagrangian probe generation and flow-map integration (from lagrangian_flowmap.py)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pyvista as pv
from scipy.interpolate import LinearNDInterpolator
from .io import FlowSnapshot

def create_initial_particles(mesh, nx, ny):
    points=np.asarray(mesh.points[:,:2], dtype=float)
    xmin,xmax=points[:,0].min(), points[:,0].max()
    ymin,ymax=points[:,1].min(), points[:,1].max()
    xs=np.linspace(xmin,xmax,nx); ys=np.linspace(ymin,ymax,ny)
    X,Y=np.meshgrid(xs,ys)
    candidates=np.column_stack([X.ravel(), Y.ravel()])
    candidate_mesh=pv.PolyData(np.column_stack([candidates, np.zeros(len(candidates))]))
    surface=mesh.extract_surface()
    enclosed=candidate_mesh.select_enclosed_points(surface, tolerance=1e-8, check_surface=False)
    inside=np.asarray(enclosed.point_data["SelectedPoints"], dtype=bool)
    return candidates[inside]

class VelocityField:
    def __init__(self, snapshot: FlowSnapshot):
        pts=snapshot.coordinates
        vel=snapshot.velocity
        self.u=LinearNDInterpolator(pts, vel[:,0], fill_value=np.nan)
        self.v=LinearNDInterpolator(pts, vel[:,1], fill_value=np.nan)
        self.snapshot=snapshot
    def evaluate(self, positions):
        positions=np.asarray(positions,dtype=float)
        u=self.u(positions[:,0], positions[:,1])
        v=self.v(positions[:,0], positions[:,1])
        return np.column_stack([u,v])

class TimeInterpolatedFlow:
    def __init__(self, field_a: VelocityField, field_b: VelocityField, alpha: float):
        self.field_a=field_a; self.field_b=field_b; self.alpha=alpha
    def evaluate(self, positions):
        va=self.field_a.evaluate(positions); vb=self.field_b.evaluate(positions)
        return (1-self.alpha)*va + self.alpha*vb

def rk4_step(x, dt, f1,f2,f3,f4):
    return x + (dt/6.0)*(f1+2*f2+2*f3+f4)

def advect_interval(positions, field_a, field_b, dt_cfd, substeps):
    dt=dt_cfd/substeps
    x=positions.copy()
    for j in range(substeps):
        a0=j/substeps; am=(j+0.5)/substeps; a1=(j+1.0)/substeps
        k1=TimeInterpolatedFlow(field_a,field_b,a0).evaluate(x)
        k2=TimeInterpolatedFlow(field_a,field_b,am).evaluate(x+0.5*dt*k1)
        k3=TimeInterpolatedFlow(field_a,field_b,am).evaluate(x+0.5*dt*k2)
        k4=TimeInterpolatedFlow(field_a,field_b,a1).evaluate(x+dt*k3)
        valid=(np.all(np.isfinite(k1),axis=1) & np.all(np.isfinite(k2),axis=1) & np.all(np.isfinite(k3),axis=1) & np.all(np.isfinite(k4),axis=1) & np.all(np.isfinite(x),axis=1))
        if np.any(valid):
            x_valid=rk4_step(x[valid],dt,k1[valid],k2[valid],k3[valid],k4[valid])
            x_new=np.full_like(x,np.nan); x_new[valid]=x_valid; x_new[~valid]=np.nan; x=x_new
        else: x[:]=np.nan
    return x

class LagrangianEngine:
    def __init__(self, snapshots, dt, substeps=4):
        self.snapshots=snapshots; self.dt=dt; self.substeps=substeps
    def compute_flow_map(self, initial_positions):
        n_times=len(self.snapshots); n_particles=len(initial_positions)
        flow_map=np.full((n_times,n_particles,2), np.nan, dtype=float)
        flow_map[0]=initial_positions
        current=initial_positions.copy()
        for n in range(n_times-1):
            print(f"\rLagrangian integration: {n+1}/{n_times-1}", end="")
            field_a=VelocityField(self.snapshots[n]); field_b=VelocityField(self.snapshots[n+1])
            current=advect_interval(current, field_a, field_b, self.dt, self.substeps)
            flow_map[n+1]=current
        print()
        return flow_map

def build_flow_map(adapter, nx=40, ny=40, max_snapshots=None, substeps=4):
    n_available=len(adapter)
    n_use=n_available if max_snapshots is None else min(max_snapshots, n_available)
    snapshots=[adapter.read_snapshot(i) for i in range(n_use)]
    mesh=pv.read(str(adapter.vtu_files[0]))
    raw=create_initial_particles(mesh, nx, ny)
    vf0=VelocityField(snapshots[0]); v0=vf0.evaluate(raw)
    valid=np.all(np.isfinite(v0),axis=1)
    initial=raw[valid]
    if len(initial)==0: raise RuntimeError("No valid initial particles")
    engine=LagrangianEngine(snapshots, adapter.dt, substeps)
    flow_map=engine.compute_flow_map(initial)
    times=np.arange(n_use)*adapter.dt
    return initial, flow_map, times, snapshots
