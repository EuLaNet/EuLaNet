# EuLaNet

### Lean Eulerian–Lagrangian Flow Representation
### for ML / RL / Deep RL

**Built by Sai Siddharth**

Siddharthaerospace@gmail.com
saisidd@mit.edu

---

## 1. The basic idea

A conventional CFD solution is primarily Eulerian.

It describes the flow field at spatial locations $x$ and times $t$:

$$E = E(x,t)$$

For many transport and flow-control problems, however, the instantaneous field is only part of the information.

A material element moves through the field according to

$$\frac{dx_p}{dt} = u(x_p, t)$$

where

$x_p(t)$ is the position of material particle $p$

$u(x,t)$ is the Eulerian velocity field

EuLaNet therefore keeps both descriptions:

```
Eulerian flow state + Lagrangian material evolution
```

and explicitly connects them:

```
Eulerian state ⟷ Lagrangian state
```

The resulting dataset is model-independent and can subsequently be used with CNNs, GNNs, Transformers, recurrent models, RL, DeepRL, or other learning methods.

---

## 2. What happens to a CFD .vtu file?

EuLaNet first reads the raw SU2 VTU solutions.

For each snapshot $t_n$, the CFD mesh provides spatial coordinates

$$X_n = \{x_i(t_n)\}_{i=1}^{N}$$

and the corresponding Eulerian variables

$$E_n = \{e_i(t_n)\}_{i=1}^{N}.$$

The resulting Eulerian representation is

$$E \in \mathbb{R}^{T \times N \times d_E}$$

with coordinates

$$X \in \mathbb{R}^{T \times N \times 2}.$$

For the validated NACA0015 case:

$$E \in \mathbb{R}^{150 \times 44100 \times 20}$$

and

$$X \in \mathbb{R}^{150 \times 44100 \times 2}.$$

So the complete CFD field is preserved rather than reduced to a small set of derived quantities.

---

## 3. Eulerian variables

The validated representation contains the following CFD quantities:

```
Pressure
Velocity_0
Velocity_1
Velocity_2
Nu_Tilde
Grid_Velocity_0
Grid_Velocity_1
Grid_Velocity_2
Pressure_Coefficient
Density
Laminar_Viscosity
Heat_Capacity
Thermal_Conductivity
Temperature
Skin_Friction_Coefficient_0
Skin_Friction_Coefficient_1
Skin_Friction_Coefficient_2
Heat_Flux
Y_Plus
Eddy_Viscosity
```

Conceptually:

$$E[t,i,:]$$

contains the Eulerian state at mesh point $i$ and snapshot $t$.

The corresponding coordinates are

$$X[t,i,:].$$

This gives the learning system access to the original spatial field rather than only a preselected set of CFD outputs.

---

## 4. Constructing the Lagrangian representation

EuLaNet then creates a configurable sparse set of material probes.

For a $40 \times 40$ probe grid, candidate locations are generated over the CFD domain.

A candidate is retained only if it lies inside the actual computational domain.

Thus the probe set is

$$P_0 = \{x_{p,0}\}_{p=1}^{N_p}.$$

For the validated reference case:

$$40 \times 40 \rightarrow 1184 \text{ valid particles.}$$

Points outside the actual CFD domain are not treated as physical particles.

---

## 5. Material advection

Each probe is treated as a passive material element.

Its trajectory satisfies

$$\frac{dx_p}{dt} = u(x_p, t).$$

The velocity field is available only at discrete CFD snapshots:

$$u_n(x) = u(x, t_n)$$

and

$$u_{n+1}(x) = u(x, t_{n+1}).$$

EuLaNet therefore temporally interpolates the velocity field between consecutive CFD snapshots.

For an interpolation parameter

$$\alpha \in [0,1],$$

the interpolated velocity is

$$u(x, t_n + \alpha \Delta t) = (1 - \alpha) u_n(x) + \alpha u_{n+1}(x).$$

This produces a continuous-in-time velocity estimate between the available CFD solutions.

---

## 6. RK4 integration

The particle trajectory is integrated using fourth-order Runge–Kutta.

For a time step $h$:

$$k_1 = f(x_n, t_n)$$
$$k_2 = f\left(x_n + \frac{h}{2} k_1, t_n + \frac{h}{2}\right)$$
$$k_3 = f\left(x_n + \frac{h}{2} k_2, t_n + \frac{h}{2}\right)$$
$$k_4 = f(x_n + h k_3, t_n + h)$$

and

$$x_{n+1} = x_n + \frac{h}{6} (k_1 + 2k_2 + 2k_3 + k_4).$$

The CFD interval can be divided into configurable substeps:

$$h = \frac{\Delta t_{\text{CFD}}}{N_{\text{substeps}}}.$$

The implementation evaluates the velocity using temporal interpolation between the surrounding CFD snapshots during these substeps.

This produces the discrete flow map:

$$\Phi_{t_0}^{t_n}(x_0) = x(t_n; x_0).$$

---

## 7–21. Lagrangian Transport Mathematics (Compressed)

**Flow map & displacement:** $\Phi_{t_0}^{t}(x_0)=x(t;x_0,t_0)$, $\Phi\in\mathbb{R}^{150\times1184\times2}$, $x_{p,0}\!\to\!x_p(t_T)$; $\Delta x_p(t)=x_p(t)-x_{p,0}$.

**Deformation:** $F_{t_0}^{t}=\partial\Phi/\partial x_0$, $F=\begin{bmatrix}\partial x/\partial x_0&\partial x/\partial y_0\\\partial y/\partial x_0&\partial y/\partial y_0\end{bmatrix}$, $\delta x_t\approx F\delta x_0$; $C=F^TF$ ($\|\delta x_t\|^2=\delta x_0^T C\delta x_0$, not $FF^T$, max error $2.22\times10^{-16}$); $Cn_i=\lambda_i n_i$, $\lambda_{\min},\lambda_{\max}$, $\sigma_{\max}=\sqrt{\lambda_{\max}}$.

**FTLE & validity:** $\text{FTLE}=1/(2|t-t_0|)\log\lambda_{\max}=1/|t-t_0|\log\sqrt{\lambda_{\max}}$; $x_p(t)\notin\Omega\Rightarrow\text{NaN}$.

**Lagrangian:** $L_p(t)=[x_0,y_0,x_p(t),y_p(t),\Delta x_p,\Delta y_p,\lambda,\sigma,\text{FTLE},\dots]$, $L\in\mathbb{R}^{148000\times16}$ ($1184$ probes).

**Correspondence:** $X_t=\{x_i(t)\}$, $K_t=\text{KDTree}(X_t)$, $i^*=\arg\min_i\|x_p(t)-x_i(t)\|$, $x_p(t)\to X_t[i^*]$, $d_p=\|x_p-X_t[i^*]\|$, invariant $\text{snapshot}=\text{correspondence\_snapshot}$ (same-time per-snapshot, not $X_{t_0}$); fields `point/snapshot/particle/distance/valid` $\Rightarrow$ $(p,t)\mapsto(i^*,t,d)$.

**Final:** $D=(E,X,L,C)$: $E(x,t),X(x,t)$ → same-time map → $L: x,\Delta x,F,C,\lambda,\text{FTLE}$ — instantaneous + finite-time response.

**Why built:** CFD is `many points/variables/snapshots/moving mesh`; EuLaNet adds missing material history `CFD→representation→dataset` without learner tie-in. EuLaNet does not claim automatic ML improvement — a paper applying this model with RL and ML is currently being written and will present those results — it tests if material evolution improves learning.


---

## 22. Model independence

EuLaNet is not a neural network.

It does not prescribe:

```
CNN
GNN
Transformer
LSTM
Autoencoder
RL algorithm
DeepRL algorithm
```

Instead:

```
EuLaNet = physical data representation
```

The researcher chooses the learning architecture afterwards.

This separation allows the same EuLaNet dataset to be used for different learning experiments without rebuilding the CFD representation each time.

---

## 23. Dataset structure

A EuLaNet `.npz` dataset contains the physical representation and the metadata needed to interpret it.

Core components include:

```
E
X
L
particle_id
X0
lagrangian_position
lagrangian_snapshot
lagrangian_time
correspondence_point
correspondence_snapshot
correspondence_particle
correspondence_distance
correspondence_valid
eulerian_feature_names
lagrangian_feature_names
metadata
```

The resulting file is therefore not just a matrix of numbers.

It preserves the relationship between:

```
field
particle
position
time
transport
deformation
correspondence
validity
```

---

## 24. Validated reference case

The implementation has been validated using a pitching NACA0015 CFD case.

The established reference dimensions are:

$$E = (150, 44100, 20)$$
$$X = (150, 44100, 2)$$
$$L = (148000, 16)$$

with

$$1184$$

Lagrangian particles.

The validated Cauchy–Green relation is

$$C = F^T F$$

with a maximum numerical error of approximately

$$2.22 \times 10^{-16}.$$

The reference validation also established the Eulerian, Lagrangian, transport and correspondence calculations against the original CFD-derived data.

These numbers describe the validation case and are not hard-coded requirements for other CFD simulations.

---

## 25. Usage

Install:

```bash
pip install -e .
```

Then point EuLaNet at a directory containing SU2 `.vtu` files:

```bash
eulanet build
```

or explicitly:

```bash
eulanet build \
  --vtu-dir "PATH_TO_SU2_CASE" \
  --pattern "flow_{index:05d}.vtu" \
  --start 0 \
  --stop 149 \
  --dt 0.016849 \
  --probe-grid 40x40 \
  --window 25 \
  --output "eulanet_dataset.npz"
```

The complete workflow is:

```
SU2
 │
 │ CFD solution
 ▼
.vtu snapshots
 │
 ▼
Eulerian extraction
 │
 ├───────────────┐
 ▼               │
Probe generation │
 │               │
 ▼               │
Lagrangian       │
advection        │
 │               │
 ▼               │
Flow map         │
 │               │
 ▼               │
F → C → λ → FTLE │
 │               │
 └───────┬───────┘
         ▼
Same-time correspondence
         │
         ▼
EuLaNet dataset
         │
         ▼
 ML / RL / DeepRL
```

No previously generated `.npz` dataset is required as input.

The input is the CFD solution itself.

---

## 26. In one equation

The entire purpose of EuLaNet can be summarized as:

$$\text{SU2 } E(x,t) \longrightarrow [E(x,t), \Phi_{t_0}^{t}(x_0), F, F^T F, \lambda, \text{FTLE}, C_{E \leftrightarrow L}]$$

where

$$\Phi_{t_0}^{t}$$

captures material evolution,

$$F$$

captures local deformation,

$$F^T F$$

captures finite-time metric deformation,

$$\lambda$$

captures principal stretching,

$$\text{FTLE}$$

captures finite-time stretching rate,

and

$$C_{E \leftrightarrow L}$$

explicitly connects the material representation to the Eulerian field at the same time.

That is EuLaNet.

---

## Acknowledgements

Built by Sai Siddharth.

I would like to acknowledge the helpful comments, discussions, and suggestions from members of the MIT AeroAstro community, the Hypersonics Research Laboratory, and the American Physical Society community. I am grateful to these research communities for the intellectual exchange and open scientific discussions that contributed to the development of this work.
