# EuLaNet — Eulerian–Lagrangian Flow Representation

**EuLaNet** is a model-independent physical data representation that augments conventional Eulerian CFD fields with sparse Lagrangian material evolution, finite-time deformation, and explicit same-time Eulerian–Lagrangian correspondence.

It is designed to turn raw CFD solutions into a structured dataset for downstream **ML, RL, and DeepRL** experiments without prescribing a particular learning architecture.

> **Core idea:** preserve the full Eulerian flow field while explicitly representing how material elements move and deform through it.

---

# 1. The basic idea

A conventional CFD solution is primarily Eulerian.

It describes the flow field at spatial locations x and times t:

E=E(x,t) 

For many transport and flow-control problems, however, the instantaneous field is only part of the information.

A material element moves through the field according to

\frac{dx_p}{dt}=u(x_p,t) 

where

- xp​(t) is the position of material particle p 
- u(x,t) is the Eulerian velocity field 

EuLaNet therefore keeps both descriptions:

**Eulerian flow state + Lagrangian material evolution** 

and explicitly connects them:

**Eulerian state ⟷ Lagrangian state** 

The resulting dataset is model-independent and can subsequently be used with CNNs, GNNs, Transformers, recurrent models, RL, DeepRL, or other learning methods.

---

# 2. What happens to a CFD `.vtu` file?

EuLaNet first reads the raw SU2 VTU solutions.

For each snapshot tn​, the CFD mesh provides spatial coordinates

Xn​={xi​(tn​)}i=1N​ 

and the corresponding Eulerian variables

En​={ei​(tn​)}i=1N​. 

The resulting Eulerian representation is

E\in\mathbb{R}^{T\times N\times d_E} 

with coordinates

X\in\mathbb{R}^{T\times N\times 2}. 

For the validated NACA0015 case:

E∈R150×44100×20 

and

X∈R150×44100×2. 

So the complete CFD field is preserved rather than reduced to a small set of derived quantities.

---

# 3. Eulerian variables

The validated representation contains the following CFD quantities:





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


Conceptually:

E[t,i,:] 

contains the Eulerian state at mesh point i and snapshot t.

The corresponding coordinates are

X[t,i,:]. 

This gives the learning system access to the original spatial field rather than only a preselected set of CFD outputs.

---

# 4. Constructing the Lagrangian representation

EuLaNet then creates a configurable sparse set of material probes.

For a 40×40 probe grid, candidate locations are generated over the CFD domain.

A candidate is retained only if it lies inside the actual computational domain.

Thus the probe set is

P0​={xp,0​}p=1Np​​. 

For the validated reference case:

40×40→1184 valid particles. 

Points outside the actual CFD domain are not treated as physical particles.

---

# 5. Material advection

Each probe is treated as a passive material element.

Its trajectory satisfies

\frac{dx_p}{dt}=u(x_p,t). 

The velocity field is available only at discrete CFD snapshots:

u_n(x)=u(x,t_n) 

and

u_{n+1}(x)=u(x,t_{n+1}). 

EuLaNet therefore temporally interpolates the velocity field between consecutive CFD snapshots.

For an interpolation parameter

\alpha\in[0,1], 

the interpolated velocity is

u(x,t_n+\alpha\Delta t)=(1-\alpha)u_n(x)+\alpha u_{n+1}(x). 

This produces a continuous-in-time velocity estimate between the available CFD solutions.

---

# 6. RK4 integration

The particle trajectory is integrated using fourth-order Runge–Kutta.

For a time step h:

k1​=f(xn​,tn​) k2​=f(xn​+2h​k1​,tn​+2h​) k3​=f(xn​+2h​k2​,tn​+2h​) k4​=f(xn​+hk3​,tn​+h) 

and

xn+1​=xn​+6h​(k1​+2k2​+2k3​+k4​). 

The CFD interval can be divided into configurable substeps:

h=Nsubsteps​ΔtCFD​​. 

The implementation evaluates the velocity using temporal interpolation between the surrounding CFD snapshots during these substeps.

This produces the discrete flow map:

\Phi_{t_0}^{t_n}(x_0)=x(t_n;x_0,t_0) 

---

# 7. The flow map

The flow map answers a fundamental question:

> Where does a material element that started at x0​ end up at time t?

Formally,

\Phi_{t_0}^{t}(x_0)=x(t;x_0,t_0) 

For every initial particle,

xp,0​→xp​(t1​)→xp​(t2​)→⋯→xp​(tT​). 

The complete Lagrangian flow map can therefore be represented as

Φ∈RT×Np​×2. 

For the reference case this corresponds to

150×1184×2 

particle positions before the derived transport quantities are flattened into the final Lagrangian representation.

---

# 8. Particle displacement

The displacement of a material particle is

\Delta x_p(t)=x_p(t)-x_{p,0} 

In two dimensions:

Δxp​(t)=[xp​(t)−xp,0​yp​(t)−yp,0​​]. 

EuLaNet retains the particle's initial position, current position and displacement.

This distinguishes:





where the particle started
        ↓
how it moved
        ↓
where it is now


rather than treating every observation as an independent spatial sample.

---

# 9. Deformation gradient

Tracking particle positions gives the flow map.

The next question is:

> How does a small neighborhood around a particle deform?

The answer is given by the deformation gradient:

Ft0​t​=∂x0​∂Φt0​t​​. 

In two dimensions:

F=[∂x0​∂x​∂x0​∂y​​∂y0​∂x​∂y0​∂y​​]. 

This is a local linearization of the flow map.

For a small initial displacement

δx0​, 

the corresponding final displacement is approximately

δxt​≈Fδx0​. 

Thus F contains local information about how the material neighborhood is stretched, compressed and rotated by the flow.

---

# 10. Cauchy–Green deformation tensor

EuLaNet then computes the right Cauchy–Green deformation tensor:

C=F^TF 

This is important because C removes the pure rotational component and describes deformation through the change of lengths.

For an infinitesimal vector δx0​,

∥δxt​∥2=δx0T​Cδx0​. 

Therefore C directly describes how the squared length of an infinitesimal material vector changes under the flow.

The implementation explicitly uses

C=F^TF 

rather than FFT.

For the validated implementation, the maximum numerical error in this relation was approximately

2.22\times10^{-16}, 

which is at machine precision.

---

# 11. Eigenvalue decomposition

The deformation tensor is symmetric:

C=C^T. 

It can therefore be decomposed into eigenvalues and eigenvectors:

Cn_i=\lambda_i n_i. 

In 2D:

\lambda_{\min},λmax​. 

The eigenvectors give the principal directions of deformation.

The eigenvalues give the corresponding squared stretches.

If

λmax​ 

is the largest eigenvalue, then the maximum principal stretch is

σmax​=\lambda_{\max}. 

Likewise,

σmin​=\lambda_{\min}​. 

So the Lagrangian representation does not merely say that a particle moved.

It also describes **how the neighborhood surrounding that particle deformed while it moved**.

---

# 12. FTLE

EuLaNet also derives the finite-time Lyapunov exponent from the largest eigenvalue of the Cauchy–Green tensor.

For a forward integration interval,

T=t−t0​, 

the FTLE is

\mathrm{FTLE}=\frac{1}{2|t-t_0|}\log\lambda_{\max}(C) 

or equivalently,

\mathrm{FTLE}=\frac{1}{|t-t_0|}\log\lambda_{\max}(C). 

**FTLE** therefore converts the finite-time deformation into a rate-like scalar quantity.

Large values indicate stronger finite-time material stretching.

---

# 13. Validity is preserved

Particles can leave the CFD domain.

EuLaNet does not fabricate values for those particles.

If the particle can no longer be evaluated inside the valid flow domain, its state becomes invalid and is preserved as such.

Conceptually:

x_p(t)\notin\Omega\Rightarrow x_p(t)=\mathrm{NaN} 

for subsequent invalid states.

This allows the dataset to retain the distinction between:





particle exists and was evaluated
particle left the domain
particle has invalid deformation data


rather than silently filling missing observations.

---

# 14. The Lagrangian feature vector

For each particle and time/window, EuLaNet collects the material information into a structured Lagrangian representation.

Conceptually:

Lp​(t)=[x0​,y0​,xp​(t),yp​(t),Δxp​,Δyp​,\lambda_{\min},λmax​,σmax​,FTLE,…] 

along with validity information and deformation metadata.

The complete Lagrangian representation is stored as

L\in\mathbb{R}^{N_L\times d_L}. 

For the validated reference case:

L\in\mathbb{R}^{148000\times16}. 

There are

1184 

material probes in the reference case.

---

# 15. Why correspondence is necessary

At this point we have two representations:

### Eulerian

E(x,t) 

describing the complete CFD field.

### Lagrangian

Lp​(t) 

describing the evolution and deformation of material probes.

But these cannot simply be concatenated row-by-row.

A material particle moves through the domain.

Therefore:

x_p(t)\ne x_p(t_0) 

in general.

The dataset needs to explicitly answer:

> Which Eulerian state corresponds to this material observation at this particular time?

This is the purpose of the correspondence layer.

---

# 16. Same-time Eulerian–Lagrangian correspondence

For every particle at time t, EuLaNet uses the Eulerian coordinates from the **same CFD snapshot**.

The Eulerian coordinates at time t are

Xt​={xi​(t)}i=1N​. 

A spatial KDTree is constructed:

Kt​=KDTree(Xt​). 

For a Lagrangian particle position xp​(t), the nearest Eulerian point is

i^*=\arg\min_i \|x_p(t)-x_i(t)\|. 

The correspondence is therefore

xp​(t)⟶Xt​[i∗]​ 

with correspondence distance

d_p(t)=\|x_p(t)-X_t[i^*]\|. 

This is performed independently for each snapshot.

The important invariant is:

`correspondence_snapshot = snapshot_id` 

for valid same-time correspondences.

---

# 17. Why the same-time mapping matters

Consider a moving CFD mesh.

Using a single spatial tree constructed from the first snapshot would implicitly perform

xp​(t)→Xt0​​. 

That is not the physical spatial relationship we want.

EuLaNet instead performs

xp​(t)→Xt​. 

Therefore the correspondence is simultaneously:





spatial
+
temporal
+
material


The downstream model can determine:

which particle→where it is→at what time→which Eulerian state it occupies. 

This same-time per-snapshot correspondence is the canonical EuLaNet implementation.

---

# 18. Correspondence fields

The correspondence layer stores explicit metadata including:





correspondence_point
correspondence_snapshot
correspondence_particle
correspondence_distance
correspondence_valid


Thus the relationship is not implicit.

It can be represented conceptually as

C:(p,t)\mapsto(i^*,t,d) 

where

- p = Lagrangian particle 
- t = time/snapshot 
- i∗ = Eulerian point 
- d = spatial correspondence distance 

This allows downstream processing to filter or weight observations according to correspondence quality.

---

# 19. The final representation

EuLaNet therefore produces three connected layers:





┌──────────────────────────────────────────┐
│          FULL EULERIAN FIELD             │
│                                          │
│              E(x,t)                      │
│              X(x,t)                      │
└───────────────────┬──────────────────────┘
                    │
                    │ same-time spatial map
                    ▼
┌──────────────────────────────────────────┐
│             CORRESPONDENCE               │
│                                          │
│     particle ↔ Eulerian point ↔ time     │
│                                          │
│       distance + validity metadata       │
└───────────────────┬──────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────┐
│       SPARSE LAGRANGIAN EVOLUTION        │
│                                          │
│     x(t), displacement, F, C, λ, FTLE    │
└──────────────────────────────────────────┘


The complete representation can therefore be thought of as

D=(E,X,L,C) 

where

E=Eulerian field, X=Eulerian coordinates, L=Lagrangian material evolution, 

and

C=Eulerian–Lagrangian correspondence. 

---

# 20. What this gives a learning system

A conventional dataset may provide only

E(x,t). 

EuLaNet additionally provides a material trajectory:

xp​(t0​)→xp​(t1​)→⋯→xp​(tT​) 

and its deformation:

F→C→λ→FTLE. 

It also tells the learning system where that material observation sits inside the Eulerian field:

$x_p(t)\leftrightarrow X_t$. 

This creates a structured state representation in which the learner can access both:

instantaneous flow state 

and

finite-time material response. 

The point is not that every learning problem requires all of these quantities.

The point is that EuLaNet makes them available in a consistent representation.

---

# 21. Why this was built

CFD simulations already contain a large amount of physical information.

The difficulty is that the information is primarily stored in the representation produced by the solver:





many spatial points
many variables
many snapshots
moving/deforming mesh


A machine-learning pipeline can consume this data, but the material history of the flow is not naturally represented as an explicit object.

EuLaNet was built to create that missing layer:

**CFD solution → structured physical representation → learning dataset** 

The intention is to make experiments on flow prediction, flow control, transport and reinforcement learning easier to formulate without tying the physical representation to one particular neural-network architecture.

EuLaNet therefore does **not** claim that the representation automatically produces better ML_performance.

That must be demonstrated experimentally.

Instead, it provides the representation needed to test questions such as:

Does material evolution improve the learned state representation? Does finite-time deformation provide useful predictive information? Does explicit Eulerian–Lagrangian correspondence improve learning efficiency? Can the same representation improve data efficiency for flow-control problems? 

These are downstream research questions.

---

# 22. Model independence

EuLaNet is not a neural network.

It does not prescribe:





CNN
GNN
Transformer
LSTM
Autoencoder
RL algorithm
DeepRL algorithm


Instead:

**EuLaNet = physical data representation** 

The researcher chooses the learning architecture afterwards.

This separation allows the same EuLaNet dataset to be used for different learning experiments without rebuilding the CFD representation each time.

---

# 23. Dataset structure

A EuLaNet `.npz` dataset contains the physical representation and the metadata needed to interpret it.

Core components include:





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


The resulting file is therefore not just a matrix of numbers.

It preserves the relationship between:





field
particle
position
time
transport
deformation
correspondence
validity


---

# 24. Validated reference case

The implementation has been validated using a pitching NACA0015 CFD case.

The established reference dimensions are:

E=(150,44100,20) X=(150,44100,2) L=(148000,16) 

with

1184 

Lagrangian particles.

The validated Cauchy–Green relation is

C=F^TF 

with a maximum numerical error of approximately

2.22\times10^{-16}. 

The reference validation also established the Eulerian, Lagrangian, transport and correspondence calculations against the original CFD-derived data.

These numbers describe the validation case and are not hard-coded requirements for other CFD simulations.

---

# 25. Usage

Install:





pip install -e .


Then point EuLaNet at a directory containing SU2 `.vtu` files:





eulanet build


or explicitly:





eulanet build ^
  --vtu-dir "PATH_TO_SU2_CASE" ^
  --pattern "flow_{index:05d}.vtu" ^
  --start 0 ^
  --stop 149 ^
  --dt 0.016849 ^
  --probe-grid 40x40 ^
  --window 25 ^
  --output "eulanet_dataset.npz"


The complete workflow is:





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


No previously generated `.npz` dataset is required as input.

The input is the CFD solution itself.

---

# 26. In one equation

The entire purpose of EuLaNet can be summarized as:

SU2 E(x,t)⟶[E(x,t),Φt0​t​(x0​),F,FTF,λ,FTLE,CE↔L​]​ 

where

Φt0​t​ 

captures material evolution,

F 

captures local deformation,

FTF 

captures finite-time metric deformation,

λ 

captures principal stretching,

FTLE 

captures finite-time stretching rate,

and

CE↔L​ 

explicitly connects the material representation to the Eulerian field at the same time.

That is EuLaNet.
