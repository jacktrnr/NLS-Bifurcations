# Half-Line NLS Bifurcation Analysis

A Julia program for the numerical bifurcation analysis of bound states of the
nonlinear Schrödinger equation on the half-line, with a compactly supported
potential.  The code finds solution branches, tracks their stability spectra,
computes resonances of the linearised operator, and verifies asymptotic
predictions for the bifurcation.

---

## Table of Contents

1. [Mathematical Problem](#1-mathematical-problem)
2. [Methods Overview](#2-methods-overview)
3. [File Structure](#3-file-structure)
4. [Dependencies](#4-dependencies)
5. [Running the Code](#5-running-the-code)
6. [Configuration Reference](#6-configuration-reference)
7. [Stage-by-Stage Description](#7-stage-by-stage-description)
8. [Potential Types](#8-potential-types)
9. [Output Files](#9-output-files)
10. [Detailed Method Notes](#10-detailed-method-notes)
11. [Rebuilding This Program](#11-rebuilding-this-program)

---

## 1. Mathematical Problem

### 1.1 The Equation

We look for **stationary bound states** of the focusing cubic NLS on the half-line:

```
-ψ''(x) + V(x) ψ(x) - ψ(x)³ = E ψ(x),    x > 0
```

subject to a **Dirichlet condition at the origin**:

```
ψ(0) = 0
```

and a **decay condition at infinity**:

```
ψ(x) → 0  as  x → ∞
```

Here:
- `E < 0` is the spectral parameter (energy / nonlinear eigenvalue).
- `V(x)` is a real-valued potential with **compact support on `[0, b]`**,
  meaning `V(x) = 0` for `x > b`.
- `ψ` is real-valued throughout.

The equation can be rewritten as a second-order ODE:

```
ψ'' = (V(x) - E) ψ - ψ³
```

### 1.2 The Shooting Parameter

Since `ψ(0) = 0`, the only free initial datum is the slope:

```
β = ψ'(0)
```

The problem becomes: for a given energy `E < 0`, find values of `β` for which
the solution of the ODE with initial conditions `ψ(0) = 0`, `ψ'(0) = β` decays
to zero as `x → ∞`.

### 1.3 The Tail Condition and Hamiltonian Residual

For `x > b` the potential vanishes and the equation reduces to the free NLS:

```
-ψ'' - ψ³ = E ψ,    x > b
```

This free equation has the conserved Hamiltonian:

```
H[ψ] = ½ (ψ')² + ½ E ψ² + ¼ ψ⁴
```

The unique decaying solution of the free equation satisfying `H = 0` is the
**soliton** (homoclinic orbit in phase space):

```
ψ_sol(x) = A sech(κ(x - x₀)),    A = √(-2E),  κ = √(-E)
```

Therefore the matching condition at `x = b` is simply `H[ψ(b)] = 0`, i.e.

```
F(β, E) = ½ ψ'(b)² + ½ E ψ(b)² + ¼ ψ(b)⁴ = 0
```

Finding zeros of `F` in `(β, E)` space gives the solution branches.

### 1.4 Norms

The L² norm (total mass) is split into an interior part and an analytic tail:

```
N[ψ] = ∫₀^b ψ² dx  +  ∫_b^∞ ψ² dx
```

The interior integral is evaluated numerically (trapezoidal rule on the ODE
grid).  The tail integral is computed analytically from the soliton formula
using only `ψ(b)` and `ψ'(b)`:

```
∫_b^∞ ψ² dx = (A²/κ)(1 + ψ'(b) / (κ ψ(b)))
```

The H¹ norm adds a similar kinetic tail term.

### 1.5 Stability Operators

Linearising around a bound state `ψ` gives the operators

```
L₊ = -∂ₓ² + V(x) - E - 3ψ²
L₋ = -∂ₓ² + V(x) - E - ψ²
```

on `L²(0,∞)` with Dirichlet BC at `x = 0`.

A bound state is **orbitally stable** in the Grillakis–Shatah–Strauss sense
when `n(L₊) = 1` and `n(L₋) = 0`, where `n(A)` denotes the number of
negative eigenvalues of operator `A`.

The **second eigenvalue** of `L₊`, denoted `μ = λ₂(L₊)`, plays a key role
near the bifurcation — it measures how close the solution is to a stability
boundary.

### 1.6 Resonances and Bifurcation Theory

The **linear operator** `H = -∂ₓ² + V(x)` on the half-line has resonances:
values `k ∈ ℂ` with `Im(k) < 0` for which `Hu = k² u` on `(0,b)` has a
solution satisfying

```
u(0) = 0,    u'(b) = ik u(b)    (outgoing radiation condition)
```

For a resonance on the imaginary axis, writing `k = -iγ` with `γ > 0`,
one has `k² = -γ²`, so the **bifurcation energy** is

```
E_bif = -γ²
```

The resonance eigenfunction `U_★` is normalised by the canonical shooting
convention `U_★'(0) = 1` and satisfies

```
-U_★'' + V(x) U_★ = -γ² U_★,    U_★(0) = 0,  U_★'(0) = 1
```

The outgoing BC at `x = b` becomes `U_★'(b) = γ U_★(b)` (since `ik = γ`
when `k = -iγ`).

The **bifurcation coefficients** needed for leading-order asymptotics are:

```
Ω  = U_★(b)⁴ / (2γ) - 2 ∫₀^b U_★(x)⁴ dx
ν₀ = 3Ω / (4γ)

A  = ∫₀^b U_★(x)² dx - U_★(b)² / (2γ)
dN/dE|₀ = -2/γ + 2A²/Ω
```

### 1.7 Asymptotic Predictions (Stage 5)

With `ε = β² = ψ'(0)²` as the bifurcation smallness parameter:

| Quantity | Leading-order prediction |
|---|---|
| Second eigenvalue of L₊ | `μ(ε) = ν₀ ε² + O(ε³)` |
| Mass-energy slope | `dN/dE|_{ε=0} = -2/γ + 2A²/Ω` |

Stage 5 verifies these predictions numerically along the continuation branch.

---

## 2. Methods Overview

### 2.1 ODE Integration — Tsit5

All initial-value problems are solved with **Tsit5** (Tsitouras' 4(5)
Runge-Kutta pair) from `OrdinaryDiffEq.jl`.  Default tolerances are
`reltol = 1e-10`, `abstol = 1e-12` for the nonlinear shooting, and tighter
(`1e-13`/`1e-15`) for linear resonance shooting.  Tsit5 is an excellent
default for smooth, non-stiff problems.

### 2.2 Seed Finding — Grid Scan + Bisection

For each energy `E` in `Estart`:
1. Evaluate `F(β, E)` on a uniform grid of `nβ` values in `(0, β_max]`.
2. Detect sign changes between consecutive grid points.
3. Refine each sign change to a root via **bisection** (up to 60 iterations,
   tolerance `1e-10`).

This yields a list of starting points `(β★, E)` for continuation.

### 2.3 Branch Continuation — Pseudo-Arclength (BifurcationKit)

Branches are continued in `E` using **pseudo-arclength continuation** (PALC)
from `BifurcationKit.jl`.  At each step:
- A **predictor** takes a step of length `ds` along the current tangent
  to the solution curve.
- A **Newton corrector** (tolerance `1e-8`, max 15 iterations) solves the
  augmented system `[F(β,E) = 0; arclength constraint = 0]`.
- The step size adapts between `dsmin` and `dsmax` based on Newton convergence.

`bothside=true` traces the branch in both directions from the seed.
A callback stops the branch when `|β| < β_min`, preventing continuation into
the degenerate neighbourhood of `β = 0`.

### 2.4 Spectral Analysis — FD Tridiagonal + Shooting Bisection

**Coarse eigenvalues** are computed by finite-difference discretisation of
`L₊` and `L₋` on `[0, Xmax]` with Dirichlet BCs at both ends.  The
resulting sparse tridiagonal matrix is diagonalised by `eigvals` (or `eigs`
for very large grids).

**Refined eigenvalues** (`compute_Lpm_eigenvalues_refined`) use a two-stage
pipeline:
1. FD on a coarser grid gives an initial bracket `[λ_fd ± Δ]`.
2. **Bisection** on the shooting residual
   `G(λ) = φ'(X_match) + κ(λ) φ(X_match)`
   refines to tolerance `1e-10`.  Here `φ` solves `(L± - λ)φ = 0` with
   `φ(0) = 0`, `φ'(0) = 1` integrated by Tsit5 to the matching point
   `X_match = b + 12/κ_E`, beyond which the soliton tail is negligible.

This removes the finite-domain truncation shift entirely from the FD result.

### 2.5 Resonance Computation — Companion Eigenvalue Problem

Resonances of `H = -∂ₓ² + V(x)` are found by posing the problem on `[0, b]`
with the exact outgoing condition `u'(b) = iku(b)`.

**Ghost-point elimination** at `x = b` converts the central-difference stencil
at the last node into the **quadratic eigenvalue problem (QEP)**:

```
(k²I + ik B - A) u = 0
```

where:
- `A` is the N×N FD matrix for `-∂ₓ² + V` (with row N modified by the
  ghost-point substitution `u_{N+1} = u_{N-1} + 2ihk u_N`).
- `B` is a rank-1 matrix with `B[N,N] = 2/h` (the k-linear term from the BC).

The QEP is **linearised** to a `2N × 2N` companion:

```
C = [ 0,   I  ]    satisfying   C [u; ku] = k [u; ku]
    [ A,  -iB ]
```

Solved by LAPACK's dense `eigen`.  Resonances with `Im(k) < -tol` and
`|k| < k_max` are filtered, and the eigenfunction is the upper N components
of the eigenvector, normalised so `U(b) = 1`.

### 2.6 Resonance Refinement — Newton Shooting

The companion EVP gives an initial estimate `γ_FD`.  Stage 5 refines it by
Newton iterations on the shooting residual

```
f(γ) = U'(b; γ) - γ U(b; γ) = 0
```

where `U(·; γ)` solves `U'' = (V(x) + γ²) U`, `U(0) = 0`, `U'(0) = 1`
with Tsit5.  The derivative `f'(γ)` is approximated by a forward finite
difference with step `δ = 10⁻⁶ γ`.  Convergence is declared at
`|Δγ| < 10⁻¹² γ`, typically achieved in 3–6 iterations.

### 2.7 Time Dynamics — Symmetric Split-Step with DST

The time-dependent NLS `iψₜ = -ψ'' + Vψ - |ψ|²ψ` is evolved by a
**symmetric (Strang) split-step**:

```
ψ(t+dt) ≈ exp(-i dt/2 · T) · exp(-i dt · (V - |ψ|²)) · exp(-i dt/2 · T) ψ(t)
```

The kinetic half-steps `exp(-i dt/2 · T)` are applied in the **DST-I**
(Discrete Sine Transform) frequency domain via FFTW, which diagonalises
`T = -∂ₓ²` exactly on `(0, Xmax)` with Dirichlet BCs.  The nonlinear step
is a pointwise multiplication.

A **complex absorbing potential** (CAP) layer near `x = Xmax` damps outgoing
radiation: `W(x) = s·((x-x_abs)/w)^p` (cubic ramp by default), applied as
`exp(-W dt)` per time step.

---

## 3. File Structure

```
Half Line/
├── run.jl          Main script — configuration and all pipeline stages
├── core.jl         ODE shooting, norms, continuation, stability operators
├── potentials.jl   Potential constructors and dispatcher
├── plotting.jl     All visualisation functions
├── save.jl         JLD2 serialisation and plot-saving utilities
├── dynamics.jl     Split-step time propagator and initial conditions
├── resonances.jl   Resonance computation via companion EVP + printing
└── results/        Auto-created output directory
    └── <potential_type>/
        ├── data/           JLD2 branch data
        ├── potential/      V(x) plot
        ├── mass_energy/    N vs E and H¹ vs E (full + zoomed)
        ├── profiles/       Solution profiles ψ(x) (wide + zoomed)
        ├── spectrum/       Eigenvalue evolution plots
        ├── stability/      Stability diagram
        └── verify/         Stage 5 CSVs and PDFs
```

### File Roles

| File | Responsibility |
|---|---|
| `run.jl` | Top-level driver; user edits the `CONFIGURATION` block and `include`s |
| `core.jl` | All numerics: shooting, residual, bisection, continuation wrapper, FD operators, spectral refinement |
| `potentials.jl` | Closure-based potential constructors and the `make_potential` dispatcher |
| `plotting.jl` | Pure visualisation; builds and displays Plots.jl figures |
| `save.jl` | Serialises branch data to JLD2 and dispatches plot saving to subdirectories |
| `dynamics.jl` | Time-domain solver: split-step integrator, DST kinetic step, CAP layer, GIF export |
| `resonances.jl` | Companion EVP construction, eigenvalue filtering, integral and coupling computation |

---

## 4. Dependencies

All packages are from the Julia general registry.  Add them with

```julia
using Pkg
Pkg.add([
    "OrdinaryDiffEq",
    "BifurcationKit",
    "Accessors",
    "Arpack",
    "FFTW",
    "Plots",
    "LaTeXStrings",
    "Colors",
    "JLD2",
])
```

| Package | Purpose |
|---|---|
| `OrdinaryDiffEq` | Tsit5 adaptive ODE solver |
| `BifurcationKit` | Pseudo-arclength continuation and Newton corrector |
| `Accessors` | `@optic` lens for BifurcationKit parameter handling |
| `Arpack` | Iterative sparse eigenvalue solver (`eigs`) |
| `FFTW` | Fast Discrete Sine Transform for split-step kinetics |
| `Plots` | Plotting backend (default: GR) |
| `LaTeXStrings` | `L"..."` macro for LaTeX axis labels |
| `Colors` | Distinguishable colour palettes for multi-branch plots |
| `JLD2` | HDF5-based serialisation of Julia objects |

`LinearAlgebra`, `SparseArrays`, and `Printf` are Julia standard-library
modules (no installation required).

---

## 5. Running the Code

Open a Julia REPL (e.g. the integrated terminal in VS Code with the Julia
extension):

```julia
include("run.jl")
```

Edit the `CONFIGURATION` block at the top of `run.jl`, then re-include.
No compilation step is needed — all stages execute sequentially.

To selectively enable or disable stages, set the corresponding flags before
including (or edit them in `run.jl`):

```julia
run_spectral      = true    # Stage 2: eigenvalue tracking
run_dynamics_flag = false   # Stage 3: time dynamics
run_resonances    = true    # Stage 4: resonance computation
run_verification  = true    # Stage 5: formula verification
```

---

## 6. Configuration Reference

### Potential

```julia
potential_type = :square   # see §8 for all types
b    = 1.0                 # support boundary: V(x) = 0 for x > b
V0   = -1.0                # potential depth (negative = attractive well)
```

Type-specific parameters (ignored when not relevant):

```julia
bump_amp_factor  = 2.3     # :square_bump — amplitude = factor × |V0|
bump_width_frac  = 0.1     # :square_bump — width = fraction × b
bump_center_frac = 1.0     # :square_bump — center = fraction × b
V1               = -6.0    # :step        — left-half depth
σ_frac           = 0.25    # :gaussian    — width σ = fraction × b
edge_height      = 0.1     # :threestep   — height of edge shelves
```

### Seed Finding

```julia
Estart   = [-7.02]   # list of energies to scan; use values near linear eigenvalues
β_max    = 10.0      # maximum β in the grid scan
nβ       = 800       # number of β gridpoints
N        = 3000      # ODE gridpoints on [0,b] for residual evaluations
seed_tol = 1e-9      # bisection tolerance for F = 0
```

### Continuation

```julia
ds        = 0.001    # initial arclength step
dsmin     = 1e-5     # minimum step (decrease near turning points)
dsmax     = 0.001    # maximum step
Emin      = -10.0    # lower E boundary
max_steps = 5000     # maximum continuation steps per branch
β_min     = 1e-3     # stop branch when |β| drops below this
```

### Spectral Analysis (Stage 2)

```julia
nev           = 2       # number of eigenvalues for L₊ and L₋
spectral_skip = 1       # compute spectrum every N-th branch point (1 = all)
Ngrid         = 2000    # FD grid size for tridiagonal operator
Xmax_spec     = 60.0    # domain truncation length
```

### Resonances (Stage 4)

```julia
run_resonances = true
res_N     = 400    # FD nodes; companion matrix is 2N × 2N
res_k_max = 10.0   # discard resonances with |k| > this
```

Increase `res_N` for better accuracy.  Increase `res_k_max` to see resonances
further from the imaginary axis.

### Verification (Stage 5)

```julia
run_verification = true
verif_n_pts  = 10          # branch points used for μ(ε) table
verif_Ngrid  = Ngrid       # FD grid for L₊ in verification
verif_Xmax   = Xmax_spec   # truncation domain
```

### Time Dynamics (Stage 3)

```julia
run_dynamics_flag   = false
dyn_ic              = :soliton    # :soliton | :groundstate | :gaussian
dyn_branch_idx      = 1           # which branch to use for the IC
dyn_use_endpoint    = true        # true = branch point closest to E = 0
dyn_Xmax            = 100.0       # spatial domain right boundary
dyn_Ngrid           = 2048        # interior gridpoints (power of 2 recommended)
dyn_Tmax            = 100.0       # total evolution time
dyn_dt              = 1e-2        # time step
dyn_ε_psi2          = 0.0         # ψ² perturbation amplitude (:soliton)
dyn_ε_psip          = 0.8         # ψ' perturbation amplitude (:soliton)
dyn_σ_gauss         = 5.0         # Gaussian IC width (:gaussian)
dyn_absorb_width    = 5.0         # CAP width at x = Xmax
dyn_absorb_strength = 15.0        # CAP damping coefficient
dyn_absorb_power    = 3           # CAP ramp exponent (3 = cubic)
dyn_save_every      = 100         # save a frame every N time steps
dyn_fps             = 20          # GIF frames per second
```

---

## 7. Stage-by-Stage Description

### Stage 1 — Seeds and Continuation

**Seed finding.**  For each `E` in `Estart`, the Hamiltonian residual
`F(β, E)` is evaluated on a grid of `nβ` points in `β ∈ (0, β_max]`.
Sign changes indicate a zero crossing; each is refined by bisection to give
a seed `(β★, E)`.  Seeds very close together (within `1e-4` in `β`) are
deduplicated.

**Continuation.**  Each seed is passed to `BifurcationKit.continuation` with
the `PALC()` algorithm.  The `finalise_solution` callback stops the branch
when `|β| < β_min` (near the bifurcation, where `F` becomes degenerate) or
when `E` leaves `[Emin, -1e-10]`.

**Output.**  Branch summary table; L² and H¹ norm vs E plots (full range and
auto-zoomed to the data extent); solution profile plots (wide view to `Xmax`
and zoomed to the support `[0, b]`).

### Stage 2 — Spectral Analysis

For each non-empty branch, `track_spectrum_branch` selects `n_grid = 50`
E values evenly spaced across the branch range, finds the nearest branch
point to each, and evaluates the smallest `nev` eigenvalues of `L₊` and
`L₋` via the FD tridiagonal method.

Two plots are produced per branch:
- **Top panel**: all tracked eigenvalues vs E, plus the essential spectrum
  threshold `λ = -E`.
- **Bottom panel**: the second eigenvalue `λ₂(L₊)` alone, which changes
  sign at stability transitions.

### Stage 2b — Reload from Saved Data (commented block)

A commented-out block at the bottom of `run.jl` shows how to reload a
`.jld2` file, reconstruct `Vfun` from the stored metadata, and re-run
spectral analysis without repeating the branch computation.

### Stage 3 — Time Dynamics

A split-step propagator evolves the full time-dependent NLS from one of three
initial conditions:

- `:soliton` — a bound state from the continuation branch, optionally
  perturbed in the `ψ²` or `ψ'` directions.
- `:groundstate` — the first mode of `-∂ₓ²` on `[0,b]`, scaled to the
  branch mass.
- `:gaussian` — `x exp(-x²/(2σ²))`, which satisfies the Dirichlet BC and
  has smooth compact support.

A GIF of `|ψ(x,t)|` is saved to `results/<type>/dynamics/`.

### Stage 4 — Resonances

`compute_resonances` builds the `2N×2N` companion matrix, solves the full
dense eigenvalue problem, and filters for `Im(k) < -tol`, `|k| < k_max`.

For each resonance the output reports:
- Wavenumber `k` and `κ = -ik`.
- The eigenfunction `U` (normalised `U(b) = 1`).
- The integral `I = ∫₀^b (U/U(b))⁴ dx` (trapezoidal rule).
- The coupling coefficient `c = -4κ I`.

### Stage 5 — Formula Verification

This stage tests the two asymptotic predictions from §1.7.

**Step 1: resonance selection.**  Branch data is sorted by `β` ascending, so
`Es[1]` is the energy closest to the bifurcation.  The resonance whose
`Re(k²)` is closest to `Es[1]` is selected from the Stage 4 results — this
picks the resonance that the branch bifurcates from.

**Step 2: Newton refinement.**  Newton iterations refine `γ` by solving
`f(γ) = U'(b;γ) - γ U(b;γ) = 0` with forward-difference Newton, converging
to `~10⁻¹²` relative accuracy in a few iterations.

**Step 3: U_★ computation.**  The linear IVP
`U'' = (V(x) + γ²) U`, `U(0) = 0`, `U'(0) = 1` is integrated by Tsit5
on 2001 points on `[0, b]`.  All integrals (`I2`, `I4`) are computed by the
trapezoidal rule.

**Step 4: eigenvalue table.**  The second eigenvalue `μ = λ₂(L₊)` is computed
at `verif_n_pts` branch points using `compute_Lpm_eigenvalues_refined`.
The ratio `(μ - μ_min) / ε²` is tabulated and plotted vs `ε = β²`; it should
converge to `ν₀` as `ε → 0`.

**Step 5: mass slope.**  `dN/dE` is estimated by centred finite differences
along the sorted branch and compared to the predicted value `dN/dE|₀`.

**Output.** Five PDFs and three CSVs saved to `results/<type>/verify/`.

---

## 8. Potential Types

All potentials have exact compact support: `V(x) = 0` for `x ≤ 0` or `x ≥ b`.

| Symbol | Shape | Key parameters |
|---|---|---|
| `:square` | Constant `V0` on `(0,b)` | `V0`, `b` |
| `:square_bump` | Square well plus a Gaussian bump near the boundary | `V0`, `b`, `bump_amp_factor`, `bump_width_frac`, `bump_center_frac` |
| `:step` | Piecewise constant: `V1` on `(0, b/2)`, `V0` on `[b/2, b)` | `V0`, `V1`, `b` |
| `:gaussian` | `V0 exp(-(x-b/2)²/σ²)` on `(0,b)` | `V0`, `σ_frac`, `b` |
| `:threestep` | `edge_height` on outer quarters, `V0` on middle half | `V0`, `edge_height`, `b` |
| `:smooth` | `V0 exp(-1/(1-(x/b)²))` on `(0,b)` — C^∞, all derivatives zero at endpoints | `V0`, `b` |

**Tip for Stage 5:** `:smooth` gives the cleanest bifurcation because the
potential vanishes smoothly at both endpoints, producing a well-separated
resonance spectrum.  Use a smaller `β_min` (e.g. `1e-4`) to follow the branch
close to the bifurcation.

---

## 9. Output Files

### Directory layout

```
results/<potential_type>/
    data/<label>.jld2
    potential/<label>.png
    mass_energy/<label>_L2.png
    mass_energy/<label>_H1.png
    mass_energy/<label>_L2_zoomed.png
    mass_energy/<label>_H1_zoomed.png
    profiles/<label>.png
    profiles/<label>_zoomed.png
    spectrum/<label>_branch1.png
    stability/<label>.png
    verify/<label>_branch.csv
    verify/<label>_mu.csv
    verify/<label>_resonance.csv
    verify/<label>_N_vs_eps.pdf
    verify/<label>_N_vs_E.pdf
    verify/<label>_mu_ratio.pdf
    verify/<label>_mu_loglog.pdf
    verify/<label>_dNdE.pdf
```

### Label format

Labels are auto-generated as `<potential_type>_b=<b>_V0=<V0>`, e.g.
`square_b=1.0_V0=-1.0`.

### JLD2 data format

```julia
data = load_run_data("results/square/data/square_b=1.0_V0=-1.0.jld2")
# data.branch_data[i].points[j]  →  (; β, E, param)
# data.seed_data[i]              →  (; β, E)
# data.b, data.V0, data.potential_type
```

### CSV columns

**`_branch.csv`:** `beta, E, N, dNdE`

**`_mu.csv`:** `beta, eps, E, mu_num, mu_shifted, nu0_eps2, rel_err, lambda1`

**`_resonance.csv`:** `gamma, E_bif, Omega, nu0, A, dNdE_pred, I2, I4`

---

## 10. Detailed Method Notes

### 10.1 Why the Hamiltonian Residual?

The residual `F(β, E) = H[ψ(b)] / ψ(b)` is chosen because it is smooth and
well-conditioned even when `ψ(b)` is small.  Dividing by `ψ(b)` removes a
trivial zero at `ψ ≡ 0`.  A penalty `100(|ψ(b)| - A)²` is added when
`|ψ(b)| > A` to keep the Newton corrector in the physical range.

### 10.2 Soliton Gluing

For profiles and norm computations, the ODE solution on `[0, b]` is extended
to `[b, Xmax]` by matching to the soliton `ψ_sol(x) = A sech(κ(x - x₀))`.
The centre `x₀` is determined by inverting `|ψ(b)| = A sech(κ(b - x₀))` and
choosing the branch where the derivative sign matches `ψ'(b)`.

### 10.3 Analytic Tail Integrals

The L² tail contribution is

```
∫_b^∞ A² sech²(κ(x - x₀)) dx = (A²/κ)(1 - tanh(κ(b - x₀)))
```

which, via the matching condition, reduces to a formula in `ψ(b)` and `ψ'(b)`
only — no numerical integration of a long tail is needed.

### 10.4 Arclength Continuation Details

PALC traces the zero set of `F(β, E) = 0` in the `(β, E)` plane.  The
tangent direction at each step is computed from the Jacobian.  The predictor
steps along the tangent; the corrector is Newton's method on the augmented
system `[F = 0; arclength condition = ds]`.

The `finalise_solution` callback stops the branch when `|β| < β_min`,
preventing the algorithm from trying to follow the branch all the way to
`β = 0` where `F` is degenerate.

### 10.5 Domain Truncation in Spectral Computations

`L₊` and `L₋` are defined on `[0, ∞)`.  The FD matrices truncate to
`[0, Xmax]` with a Dirichlet condition at `Xmax`, which shifts each
eigenvalue by approximately `(π/Xmax)²`.  For the coarse spectral tracking
(Stage 2) this error is acceptable.  For Stage 5, the shooting bisection
refinement removes the truncation error entirely.

### 10.6 Resonance Eigenfunction Normalisations

`compute_resonances` normalises the FD eigenvector so `U(b) = 1`.  Stage 5
uses the canonical **shooting normalisation** `U'(0) = 1` instead, which
makes the analytic formulas for `Ω`, `ν₀`, `A` self-consistent with the
derivation.  The two normalisations differ by a global scalar; all
dimensionless ratios (like `ν₀`) are unchanged.

### 10.7 Split-Step Accuracy and the DST

The symmetric (Strang) split-step has global error `O(dt²)`.  The DST-I
diagonalises `-∂ₓ²` on `(0, Xmax)` with Dirichlet BCs exactly, so the
kinetic step introduces no spatial discretisation error — accuracy is
controlled solely by `dt`.  FFTW's DST-I operates on N interior points
and yields eigenvalues `λₙ = (nπ / (Xmax + h))²` for `n = 1, …, N`.

### 10.8 Complex Absorbing Potential

The CAP is a purely imaginary addition to the potential: `V_eff = V - iW`.
In the split-step nonlinear substep, damping is applied as `exp(-W dt)`
pointwise.  The cubic ramp `W(x) ∝ ((x - x_abs)/w)³` is a standard choice
that minimises reflections while keeping the layer short.

---

## 11. Rebuilding This Program

The essential implementation order, from simplest to most complex:

1. **`potentials.jl`** — Write closure-returning functions for each shape;
   add a dispatcher `make_potential(::Symbol; kwargs...)`.

2. **`core.jl` — shooting and residual** — Implement `shoot_from_origin`
   with `ODEProblem` + `Tsit5`; write `F_residual` wrapping it; add
   `bisect_F`; implement `glue_solution` for the soliton tail; add
   `compute_L2_norm` using analytic tail integrals.

3. **`core.jl` — seed finding** — Grid scan over `β`, detect sign changes,
   call `bisect_F`, deduplicate.

4. **`core.jl` — continuation** — Wrap `BifurcationProblem` and
   `continuation(prob, PALC(), opts)` with a `finalise_solution` callback.

5. **`core.jl` — stability operators** — `compute_Lpm_eigenvalues`
   (tridiagonal FD + `eigvals`); then `compute_Lpm_eigenvalues_refined`
   (FD bracket → Tsit5 shooting → bisection on `G(λ)`).

6. **`resonances.jl`** — Assemble the `N×N` modified FD matrix `A`, the
   rank-1 matrix `B`, build the `2N×2N` companion `C`, call `eigen`, filter
   by `Im(k) < 0` and `|k| < k_max`, extract and normalise eigenfunctions.

7. **`plotting.jl`** — One function per plot type; use `L"..."` only for
   pure-math strings; use plain strings for human-readable titles; use
   `latexstring(sprintf(...))` for legend entries that mix text and numbers.

8. **`save.jl`** — `jldsave` / `load` via JLD2; `savefig` dispatched to
   labelled subdirectories via a small helper.

9. **`dynamics.jl`** — Build DST-I kinetic propagator via FFTW; Strang
   splitting loop; CAP layer as pointwise damping; save frames and write GIF.

10. **`run.jl`** — Wire all stages together with boolean flags; put all
    tuneable parameters in a single `CONFIGURATION` block at the top so the
    user only has to edit one place.
