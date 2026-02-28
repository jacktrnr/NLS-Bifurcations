potential_type = :smooth # :smooth, :threestep, :gaussian, :step, :square, :square_bump
b = 1.0                  # Support boundary: V(x) = 0 for x > b
V0 = -7.0 # Potential depth/height (negative = attractive well)

# Type-specific parameters (ignored if not relevant to chosen type):
bump_amp_factor = 2.3    # :square_bump — bump amplitude = factor * |V0|
bump_width_frac = 0.1    # :square_bump — bump width = fraction * b
bump_center_frac = 1.0   # :square_bump — bump center = fraction * b
V1 = -6.0               # :step — left-half depth
σ_frac = 0.25           # :gaussian — width σ = fraction * b
edge_height = 0.1       # :threestep — height of edge regions

# --- Seed finding ---
# Estart: initial energy guesses where we scan for solutions.
#   Each E in this list gets a full β-scan. Use values where you
#   expect bifurcations (usually just below the linear eigenvalue).
Estart = [-2.0]
β_max = 10.0             # Maximum shooting slope to scan
nβ = 800                 # Number of β gridpoints (higher = finer scan)
N = 3000                 # ODE integration gridpoints on [0, b]
seed_tol = 1e-13          # Tolerance for accepting a zero of F(β, E)

# --- Continuation ---
# BifurcationKit pseudo-arclength continuation parameters.
ds = 0.0005               # Initial step size
dsmin = 1e-12             # Minimum step size (smaller = more robust near folds)
dsmax = 0.001            # Maximum step size
Emin = -5.0           # Lower bound on E for continuation (p_min)
max_steps = 5000         # Maximum continuation steps per branch
β_min = 1e-4             # Stop branch when |β| drops below this

# --- Spectral analysis (Stage 2) ---
# Set run_spectral = false to skip entirely (much faster).
run_spectral = true

# --- Resonance computation (Stage 4) ---
# Finds resonances of H = -∂ₓ^2 + V(x) on the half-line via QEP.
# Set run_resonances = true to enable.
run_resonances = true
res_N     = 400    # FD nodes for the 2N×2N companion EVP
res_k_max = 10.0   # only show resonances with |k| < res_k_max
nev = 2                  # Number of eigenvalues to compute for L₊ and L₋
spectral_skip = 1        # Compute spectrum every `skip` branch points (1 = all)
μ_min_spec = -50.0       # Lower end of μ scan range (should be below all eigenvalues)
N_scan_spec = 500        # Number of μ scan points per branch point

# --- Verification (Stage 5) ---
# Verifies  μ(ε) ≈ ν_0 ε^2  and  dN/dE → predicted value  as ε → 0,
# where ε = β^2 = ψ'(0)^2 and ν_0, dN/dE come from the first resonance k = -iγ.
# Requires run_resonances = true (Stage 4) and at least one non-empty branch.
# Tip: use potential_type = :smooth and lower β_min (e.g. 1e-4) for best results.
run_verification = true
verif_n_pts  = 80      # branch points used to compute μ  (coarser grid)
verif_N_scan = N_scan_spec  # μ scan points for L₊ in verification
verif_μ_min  = μ_min_spec   # lower scan bound for L₊ in verification

# --- Output ---
save_data = true         # Save branch data to JLD2
save_plots_flag = true   # Save all plots as PNGs
results_dir = joinpath(@__DIR__, "results")

# --- Time dynamics (Stage 3) ---
# Set run_dynamics_flag = false to skip entirely.
run_dynamics_flag    = false
dyn_ic               = :soliton   # :soliton | :groundstate | :gaussian
dyn_branch_idx       = 1          # which branch to use for IC
dyn_use_endpoint     = true       # for :soliton: false = first branch pt, true = closest to E=0
dyn_Xmax             = 100.0      # right boundary for dynamics grid
dyn_Ngrid            = 2048       # interior grid points (power of 2 recommended)
dyn_Tmax             = 100.0       # total evolution time
dyn_dt               = 1e-2       # time step
dyn_ε_psi2           = 0.0         # amplitude of ψ_0^2 perturbation (relative to max|ψ_0|)  [:soliton]
dyn_ε_psip           = 0.8         # amplitude of ψ_0' perturbation (regularized at x=0)  [:soliton]
dyn_σ_gauss          = 5.0         # peak location σ for x·exp(-x^2/(2σ^2))  [:gaussian]
dyn_save_every       = 100        # save a frame every this many time steps
dyn_fps              = 20         # GIF frames per second
dyn_absorb_width     = 5.0        # CAP width at x = Xmax (0 = hard Dirichlet / reflecting)
dyn_absorb_strength  = 15.0        # CAP damping strength (typical 1–10)
dyn_absorb_power     = 3          # CAP ramp exponent (3 = cubic)

# --- Profile plot settings ---
Xmax_profiles = 10.0     # Wide-view x-axis limit for profile plots
n_profiles = 15          # Number of profiles to show per branch