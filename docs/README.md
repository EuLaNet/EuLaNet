# EuLaNet Documentation

**Lean Eulerian–Lagrangian Flow Representation for ML / RL / Deep RL**

Built by Sai Siddharth — Siddharthaerospace@gmail.com / saisidd@mit.edu

## Representation

```
FULL EULERIAN FLOW FIELD
          +
SPARSE LAGRANGIAN MATERIAL EVOLUTION
          +
EXPLICIT SAME-TIME SPATIAL CORRESPONDENCE
```

- **Eulerian:** `E[t, x, field]` — 20 SU2 fields, `X[t, x, 2]`
- **Lagrangian:** `L` — displacement, `F`, `C = FᵀF`, `λ`, `FTLE`, per-snapshot KDTree same-time correspondence
- **Correspondence:** `correspondence_point/snapshot/particle/distance/valid` — explicit, model-independent

## Correspondence (Method B — Canonical)

For `x_L(t)` → `X[t]` via per-snapshot KDTree, same-time. Validated `C = FᵀF` to `2.22e-16`.

## Acknowledgements

Built by Sai Siddharth.

I would like to acknowledge the helpful comments, discussions, and suggestions from members of the MIT AeroAstro community, the Hypersonics Research Laboratory, and the American Physical Society community. I am grateful to these research communities for the intellectual exchange and open scientific discussions that contributed to the development of this work.
