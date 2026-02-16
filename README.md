# Half-Line NLS Bifurcation Analysis

Numerical bifurcation analysis for the focusing nonlinear Schrodinger equation on the half-line:

```
-u'' + V(x)u - u^3 = Eu,    x > 0
u(0) = 0                     (Dirichlet BC)
```

Solutions are found by shooting from the origin with slope `beta = u'(0)` and matching to the homoclinic (soliton) tail `u ~ A sech(kappa(x - x_R))` as `x -> infinity`, where `A = sqrt(-2E)` and `kappa = sqrt(-E)`.

## Files

| File | Description |
|------|-------------|
| `run.jl` | Main entry point. Configure parameters at the top, then `include("run.jl")` in the Julia REPL. |
| `core.jl` | Shooting ODE solver, Hamiltonian residual, seed finding (bisection), BifurcationKit continuation, L+/L- spectral operators. |
| `potentials.jl` | Potential constructors with compact support on `[0, b]`. Dispatcher `make_potential(:type; b, V0, ...)` builds any type from a symbol. |
| `plotting.jl` | All visualization: potential plots, mass-energy diagrams (L2/H1 vs E), solution profiles, spectral evolution, and stability diagrams. |
| `save.jl` | JLD2 serialization for branch data and organized PNG export for plots. |
| `dynamics.jl` | Split-step time evolution of a perturbed bound state. Produces a GIF of `|psi(x,t)|`. |

## Supported Potentials

| Symbol | Description |
|--------|-------------|
| `:square` | Constant well `V(x) = V0` on `(0, b)` |
| `:square_bump` | Square well + Gaussian bump near boundary |
| `:step` | Piecewise constant: `V1` on `[0, b/2)`, `V0` on `[b/2, b)` |
| `:gaussian` | Gaussian well centered at `b/2` |
| `:threestep` | Raised edge shelves with deep center |
| `:smooth` | C-infinity bump function, vanishes exactly at endpoints |

## Workflow

### Stage 1: Seeds + Continuation
1. Scan `beta in (0, beta_max]` at specified energy values to find zeros of the Hamiltonian residual `F(beta, E) = 0`.
2. Continue each seed as a branch in `E` using pseudo-arclength continuation (BifurcationKit / PALC).
3. Compute L2 and H1 norms along each branch.
4. Generate plots and save data to JLD2.

### Stage 2: Spectral Analysis (optional, `run_spectral`)
1. Discretize the linearized operators `L+` and `L-` on a finite-difference grid.
2. Track the smallest eigenvalues along a grid of 50 E values per branch.
3. Classify stability via the Vakhitov-Kolokolov criterion: stable when `n(L+) = 1` and `n(L-) = 0`.

### Stage 3: Time Dynamics (optional, `run_dynamics_flag`)
```
i \partial_t \Psi = - \partial_x^2 \Psi + V(x) \Psi - |\Psi|^2 \Psi
```
1. Extract a bound state from the first non-empty branch.
   - `dyn_use_endpoint = false`: use the branch point near `Estart` (beginning of the branch).
   - `dyn_use_endpoint = true`: use the branch point whose E is closest to 0.
2. Perturb: `psi(x,0) = psi_0(x) * (1 + epsilon)`.
3. Evolve the time-dependent NLS via symmetric split-step with DST (Dirichlet BCs).
4. Save an animated GIF of `|psi(x,t)|` with the unperturbed bound state shown for reference.

## Usage

```julia
# In the Julia REPL or VS Code:
include("run.jl")
```

Edit the `CONFIGURATION` block at the top of `run.jl` to change:
- `potential_type` and parameters (`b`, `V0`, etc.)
- Seed energies `Estart` and scanning resolution
- Continuation step sizes and bounds
- `run_spectral = true/false` to toggle spectral analysis
- `run_dynamics_flag = true/false` to toggle time dynamics
- `dyn_use_endpoint` to choose which bound state to perturb

## Output Structure

```
results/<potential_type>/
  data/            JLD2 files with branch + seed data
  potential/       V(x) plots
  mass_energy/     L2 and H1 norm vs E (full + zoomed)
  profiles/        Solution profiles (wide + zoomed near support)
  spectrum/        Eigenvalue evolution plots
  stability/       Stability diagrams (mass colored by stability)
  dynamics/        GIF animations of perturbed bound state evolution
```

## Dependencies

- [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) -- ODE integration (Tsit5)
- [BifurcationKit.jl](https://github.com/bifurcationkit/BifurcationKit.jl) -- Pseudo-arclength continuation
- [Arpack.jl](https://github.com/JuliaLinearAlgebra/Arpack.jl) -- Sparse eigenvalue problems
- [FFTW.jl](https://github.com/JuliaMath/FFTW.jl) -- Discrete sine transform for split-step dynamics
- [Plots.jl](https://github.com/JuliaPlots/Plots.jl) + [LaTeXStrings.jl](https://github.com/stevengj/LaTeXStrings.jl) -- Visualization
- [JLD2.jl](https://github.com/JuliaIO/JLD2.jl) -- Data serialization
- [Accessors.jl](https://github.com/JuliaObjects/Accessors.jl) -- Lens for BifurcationKit parameters
