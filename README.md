# EuLaNet

### Lean Eulerian–Lagrangian Flow Representation for ML / RL / Deep RL

---

EuLaNet converts SU2 flow solutions into a structured Eulerian–Lagrangian representation containing:

```
    FULL EULERIAN FLOW FIELD
              +
    SPARSE LAGRANGIAN MATERIAL EVOLUTION
              +
    EXPLICIT SAME-TIME SPATIAL CORRESPONDENCE
```

Model-independent and designed for downstream ML, RL and Deep RL workflows. EuLaNet does not prescribe or contain a specific learner; it produces the physically structured dataset architecture for any NN based models.

---

## Installation

```bash
git clone https://github.com/saisidd/EuLaNet
cd EuLaNet
pip install -e .
```

Dependencies: `numpy`, `scipy`, `pyvista`, `vtk`, `pandas`, `tqdm`

---

## Quick Start

```bash
cd D:\my_su2_case   # contains flow_*.vtu + .cfg
eulanet build
```

Explicit:

```bash
eulanet build \
  --vtu-dir "D:\my_su2_case" \
  --pattern "flow_{index:05d}.vtu" \
  --start 0 --stop 149 \
  --dt 0.016849 \
  --probe-grid 40x40 \
  --window 25 \
  --output "eulanet_dataset.npz"
```

**Input:** raw SU2 `.vtu` files
**Output:** `eulanet_dataset.npz`

---

## Representation

**1. Full Eulerian Environment** `E[t, x, field]` — 20 fields (`Pressure`, `Velocity_0/1/2`, `Nu_Tilde`, `Grid_Velocity`, `Pressure_Coefficient`, `Density`, `Laminar_Viscosity`, `Heat_Capacity`, `Thermal_Conductivity`, `Temperature`, `Skin_Friction_Coefficient`, `Heat_Flux`, `Y_Plus`, `Eddy_Viscosity`) and `X[t, x, 2]`

**2. Sparse Lagrangian Material** `L` — `40×40` probes → `1184` valid, displacement, `F`, `C = FᵀF`, `λ`, `FTLE = 1/(2T) ln(λ_max)`, validity preserved

**3. Explicit Same-Time Correspondence** — per-snapshot KDTree, same-time `x_L(t) → X[t]`, `correspondence_point/snapshot/particle/distance/valid`

Correspondence uses **same-time Eulerian snapshot + per-snapshot KDTree** (canonical Method B for moving/deforming mesh).

---

## Example

`examples/naca0015/` — NACA0015 pitching RANS (150 snapshots, 44,100 points, dt 0.016849, 40×40 → 1184 probes, 44,594 valid correspondences, `C=FᵀF` to `2.22e-16`).

---

### Acknowledgements

<sub>
Built by Sai Siddharth.<br>
With Helpful comments and discussions from members of the MIT AeroAstro community, and the American Physical Society community.<br>
Siddharthaerospace@gmail.com · saisidd@mit.edu
</sub>

## License

MIT — Sai Siddharth
