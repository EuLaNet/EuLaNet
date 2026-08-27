# NACA0015 Pitching Example

Reference case in `D:\su2\run_pitching_naca0015\` — 150 snapshots, dt=0.016849, 40×40 probes → 1184 valid.

To reproduce with toolbox:

```bash
cd D:\su2\run_pitching_naca0015
eulanet build --vtu-dir . --pattern "flow_{index:05d}.vtu" --start 0 --stop 149 --dt 0.016849 --probe-grid 40x40 --window 25 --output eulanet_dataset.npz
```

Compare to reference: `semantic_el_dataset_150.npz` (E 150×44100×20, L 148000×16, 44594 valid).
