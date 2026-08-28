# EuLaNet — Eulerian–Lagrangian Flow Representation

[![Python](https://img.shields.io/badge/Python-3.x-3776AB)](https://www.python.org/)
[![CFD](https://img.shields.io/badge/CFD-SU2-orange)](https://su2code.github.io/)
[![Representation](https://img.shields.io/badge/Representation-Eulerian%20%2B%20Lagrangian-4c8eda)](#19-the-final-representation)

## Technical Overview

EuLaNet is a CFD-to-learning data toolbox that transforms conventional Eulerian CFD snapshots into a coupled Eulerian–Lagrangian representation. Given SU2 `.vtu` solutions $E(x,t)$, it retains the full flow state while integrating sparse material probes through the velocity field $\frac{dx_p}{dt}=u(x_p,t)$, producing finite-time trajectories and the associated flow map $x(t)=\Phi_t(x_0)$. From this evolution it computes the deformation gradient $F=\frac{\partial x(t)}{\partial x_0}$, the Cauchy–Green tensor $C=F^TF$, principal stretching $\lambda$, and FTLE.

The key construction is the explicit **same-time correspondence** between the Lagrangian material state and the Eulerian field. For every snapshot $t$, Lagrangian positions $x_L(t)$ are associated with the Eulerian coordinates $X(t)$ using a per-snapshot KDTree, giving $x_L(t)\rightarrow X(t)$. The resulting dataset therefore couples instantaneous flow physics, material transport, deformation and spatial correspondence in a consistent representation. EuLaNet is model-independent and is intended to provide a more structured physical state for ML, RL and DeepRL applications, particularly where temporal evolution and flow control are important.

