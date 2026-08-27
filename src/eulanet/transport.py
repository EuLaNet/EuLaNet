"""
transport.py — Deformation gradient, Cauchy-Green, FTLE (validated: C=F^T F, FTLE=log(lambda_max)/(2T))
"""
import numpy as np

def finite_difference_jacobian_grid(flow_xy, x0, y0):
    x0=np.asarray(x0); y0=np.asarray(y0); flow_xy=np.asarray(flow_xy)
    n=len(x0)
    ux=np.unique(np.round(x0,12)); uy=np.unique(np.round(y0,12))
    nx=len(ux); ny=len(uy)
    if nx*ny != n: return local_jacobian(flow_xy, x0, y0)
    ix={v:i for i,v in enumerate(ux)}; iy={v:i for i,v in enumerate(uy)}
    index_grid=-np.ones((ny,nx), dtype=int)
    for p in range(n):
        index_grid[iy[round(y0[p],12)], ix[round(x0[p],12)]]=p
    J=np.full((n,2,2), np.nan); valid=np.zeros(n,dtype=bool)
    dx=np.median(np.diff(ux)) if nx>1 else np.nan; dy=np.median(np.diff(uy)) if ny>1 else np.nan
    if not np.isfinite(dx) or not np.isfinite(dy): return local_jacobian(flow_xy,x0,y0)
    for j in range(1,ny-1):
        for i in range(1,nx-1):
            p=index_grid[j,i]; pxm=index_grid[j,i-1]; pxp=index_grid[j,i+1]; pym=index_grid[j-1,i]; pyp=index_grid[j+1,i]
            if min(p,pxm,pxp,pym,pyp)<0: continue
            dFdx=(flow_xy[pxp]-flow_xy[pxm])/(2*dx); dFdy=(flow_xy[pyp]-flow_xy[pym])/(2*dy)
            J[p,:,0]=dFdx; J[p,:,1]=dFdy; valid[p]=np.all(np.isfinite(J[p]))
    return J, valid

def local_jacobian(flow_xy, x0, y0, k=8):
    n=len(x0); J=np.full((n,2,2), np.nan); valid=np.zeros(n,dtype=bool)
    pts=np.column_stack([x0,y0])
    for i in range(n):
        d=pts-pts[i]; r2=np.sum(d*d,axis=1); order=np.argsort(r2)
        neigh=order[1:k+1]
        if len(neigh)<4: continue
        A=d[neigh]; B=flow_xy[neigh]-flow_xy[i]
        try:
            Gt,_,rank,_=np.linalg.lstsq(A,B,rcond=None)
            if rank<2: continue
            J[i]=Gt.T; valid[i]=np.all(np.isfinite(J[i]))
        except np.linalg.LinAlgError: pass
    return J, valid

def rolling_deformation(flow_at_k, flow_at_end, x0, y0):
    Jk,vk=finite_difference_jacobian_grid(flow_at_k,x0,y0)
    Je,ve=finite_difference_jacobian_grid(flow_at_end,x0,y0)
    n=len(x0)
    F=np.full((n,2,2), np.nan); C=np.full((n,2,2), np.nan)
    valid=vk & ve
    for i in np.where(valid)[0]:
        try:
            Fi=Je[i] @ np.linalg.inv(Jk[i]); Ci=Fi.T @ Fi
            if np.all(np.isfinite(Fi)) and np.all(np.isfinite(Ci)):
                F[i]=Fi; C[i]=Ci
            else: valid[i]=False
        except np.linalg.LinAlgError: valid[i]=False
    return F,C,valid

def cauchy_green(F):
    # validated: C = F^T F
    return np.einsum("nji,njk->nik", F, F)

def principal_quantities(C, valid):
    n=len(valid)
    lmin=np.full(n,np.nan); lmax=np.full(n,np.nan); stretch=np.full(n,np.nan); direction=np.full((n,2),np.nan)
    for i in np.where(valid)[0]:
        try:
            vals,vecs=np.linalg.eigh(C[i])
            if vals[0]<=0 or vals[1]<=0: valid[i]=False; continue
            lmin[i]=vals[0]; lmax[i]=vals[1]; stretch[i]=np.sqrt(vals[1]); direction[i]=vecs[:,1]
        except np.linalg.LinAlgError: valid[i]=False
    return lmin,lmax,stretch,direction,valid

def ftle_from_lambda(lam_max, T):
    ftle=np.full_like(lam_max, np.nan, dtype=float)
    valid=(lam_max>0) & np.isfinite(lam_max) & np.isfinite(T) & (T>0)
    # allow T scalar or array
    if np.ndim(T)==0:
        ftle[valid]=np.log(lam_max[valid])/(2.0*T)
    else:
        ftle[valid]=np.log(lam_max[valid])/(2.0*T[valid])
    return ftle
