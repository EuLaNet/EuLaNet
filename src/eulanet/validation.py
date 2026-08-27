"""
validation.py — Sanity checks (from validate_flowmap.py, dataset_sanity_check.py, validate_semantic_correspondence.py)
"""
import numpy as np

def relative_error(a,b):
    a=np.asarray(a); b=np.asarray(b)
    mask=np.isfinite(a) & np.isfinite(b)
    if not np.any(mask): return np.nan
    return np.linalg.norm((a[mask]-b[mask]).ravel())/ (np.linalg.norm(b[mask].ravel()) or 1.0)

def check_identity(initial, flow_map, tol=1e-12):
    err=np.max(np.linalg.norm(flow_map[0]-initial, axis=1))
    return err, err<=tol

def check_ftle_recompute(lam_max, T, ftle):
    expected=np.log(lam_max)/(2*T)
    return np.max(np.abs(expected-ftle))

def check_cauchy(F,C):
    Ccalc=np.einsum("nji,njk->nik", F, F)
    return np.max(np.abs(Ccalc-C))
