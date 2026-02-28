###############################################
# verify_formulas.jl
#
# Numerical verification of two analytic predictions for bifurcated
# nonlinear bound states  ψ_ε  of
#
#   -ψ'' + V(x)ψ - ψ³ = E ψ   on ℝ₊,   ψ(0) = 0
#
# where V is the smooth compactly supported well on [0, b]:
#   V(x) = V0 · exp(-1 / (1 - (x/b)²))   for 0 < x < b,   V = 0 elsewhere.
#
# ε = β = ψ'(0)  is the bifurcation amplitude parameter.
#
# The bifurcation occurs at  E_bif = -γ²  where k = -iγ  (γ > 0) is the
# first resonance of  H = -∂ₓ² + V  with  Im(k) < 0.
#
# Predictions verified
# ─────────────────────
# (1) Eigenvalue splitting:
#       μ(ε) = ν₀ ε² + O(ε³),   ν₀ = 3Ω / (4γ)
#     where Ω = U⋆(b)⁴/(2γ) − 2 ∫₀ᵇ U⋆⁴ dx
#     and μ(ε) is the near-zero eigenvalue of L₊ = -∂ₓ² + V − 3ψ_ε² − E(ε).
#
# (2) Slope of the norm–energy curve:
#       dN/dE|_{ε=0} = −2/γ + 2A²/Ω,   A = ∫₀ᵇ U⋆² dx − U⋆(b)²/(2γ)
#     where N(ε) = ∫₀^∞ ψ_ε² dx.
#
# Usage:  include("verify_formulas.jl")
###############################################

using Printf, LinearAlgebra

include("core.jl")
include("potentials.jl")
include("resonances.jl")

# ============================================================
# CONFIGURATION  — edit here
# ============================================================

b  = 1.0      # support boundary
V0 = -6.0     # smooth well depth  (negative = attractive)

# Target ε = β = ψ'(0) values
eps_values = 10.0 .^ range(-1.0, -4.0; length = 7)

# ── Resonance solver ──────────────────────────────────────────
N_res   = 600      # FD nodes for the 2N × 2N QEP
k_max   = 30.0     # discard |k| ≥ k_max
im_tol  = -1e-6    # discard Im(k) ≥ im_tol  (keeps genuine resonances)

# ── Shooting / branch finding ─────────────────────────────────
N_shoot = 3000     # ODE grid on [0, b] for shoot_from_origin

# ── L₊ eigenvalue computation ────────────────────────────────
Ngrid_Lp = 4000    # FD grid size for L₊
# Xmax is set adaptively per ε (see get_Xmax below)

# ── dN/dE finite difference ───────────────────────────────────
δ_rel = 0.02       # relative step: perturb ε → ε ± δ_rel·ε

# ── Continuation (for N–E branch plot) ───────────────────────
ds        = 5e-4
dsmin     = 1e-7
dsmax     = 5e-3
max_steps = 4000
β_min_cont = 5e-5   # stop continuation when β drops below this

# ── Output ───────────────────────────────────────────────────
out_dir = joinpath(@__DIR__, "results", "verify")
mkpath(out_dir)

# ============================================================
# SETUP
# ============================================================

println("="^70)
println("VERIFY FORMULAS  —  smooth well,  b = $b,  V0 = $V0")
println("="^70)

Vfun = smooth_well(b, V0)

# ============================================================
# STEP 1: Resonance  →  γ, E_bif, U⋆, Ω, ν₀, slope_pred
# ============================================================

println("\n--- Step 1: resonance data ---")

resonances_all = compute_resonances(b, Vfun; N=N_res, k_max=k_max, im_tol=im_tol)

if isempty(resonances_all)
    error("No resonances found. Try increasing |V0|, b, or loosening im_tol.")
end

# The list is sorted by Im(k) most-negative first.
# We want the resonance with Im(k) closest to 0, i.e. the last entry.
res = resonances_all[end]

κ_res = real(res.κ)           # κ = −ik;  for a resonance Im(k) < 0 → κ < 0
@assert κ_res < 0  "Expected κ < 0 for a resonance, got κ = $κ_res"

γ     = -κ_res                # γ > 0
E_bif = -γ^2                  # bifurcation energy

U_star = real.(res.U)         # resonance eigenfunction, normalized U⋆(b) = 1
x_res  = res.x                # FD grid  h, 2h, …, b
h_res  = x_res[2] - x_res[1]
U_b    = U_star[end]          # should be 1.0 by construction

# Trapezoidal rule (node x₀ = 0 has u₀ = 0 by Dirichlet)
I2 = h_res * (sum(U_star[1:end-1].^2) + 0.5 * U_star[end]^2)
I4 = h_res * (sum(U_star[1:end-1].^4) + 0.5 * U_star[end]^4)

Ω          = U_b^4 / (2γ)  -  2*I4
ν₀         = 3*Ω / (4γ)
A_val      = I2 - U_b^2 / (2γ)
slope_pred = -2/γ  +  2*A_val^2 / Ω

@printf("  k        = %+.6f %+.6f i\n", real(res.k), imag(res.k))
@printf("  γ        = %.8f      (γ = −Im(k))\n", γ)
@printf("  E_bif    = %.8f      (bifurcation energy = −γ²)\n", E_bif)
@printf("  U⋆(b)   = %.8f      (should be 1)\n", U_b)
@printf("  ∫ U⋆² dx = %.8f\n", I2)
@printf("  ∫ U⋆⁴ dx = %.8f\n", I4)
@printf("  Ω        = %.8f\n", Ω)
@printf("  ν₀       = 3Ω/(4γ) = %.8f\n", ν₀)
@printf("  A        = ∫U²−U(b)²/(2γ) = %.8f\n", A_val)
@printf("  dN/dE predicted = %.8f\n", slope_pred)
println()
println("  Sign check:  Ω = $(Ω > 0 ? "positive" : "negative")",
        "  →  expect μ(ε) $(Ω > 0 ? "> 0" : "< 0")  for all ε")
println()

# ============================================================
# STEP 2: Continuation branch  (for N–E plot and profiles)
# ============================================================

println("--- Step 2: continuation branch ---")

# Seed near E_bif at a moderate β — scan a few E values just below E_bif
E_seed_list = [E_bif - δ for δ in (0.05, 0.1, 0.2, 0.5)]
E_seed_list = filter(e -> e < -1e-6, E_seed_list)

seeds = find_seeds(b, Vfun;
    E_list = E_seed_list,
    β_max  = 2.0,
    nβ     = 600,
    N      = N_shoot,
    tol    = 1e-9)

branches = Any[]
if !isempty(seeds)
    branches = continue_from_seeds(seeds, b, Vfun;
        N         = N_shoot,
        p_min     = E_bif - 5.0,
        p_max     = E_bif - 1e-8,
        ds        = ds,
        dsmin     = dsmin,
        dsmax     = dsmax,
        max_steps = max_steps,
        β_min     = β_min_cont,
        verbose   = 0)
else
    @warn "No seeds found for continuation — branch plot will be skipped."
end

# Extract (β, E, N) along the first non-empty branch
branch_β = Float64[]
branch_E = Float64[]
branch_N = Float64[]

for br in branches
    isempty(br.branch) && continue
    for sol in br.branch
        β_i = sol.β
        E_i = sol.param
        xi, ui, vi = shoot_from_origin(b, E_i, Vfun, β_i; N=N_shoot)
        Ni = compute_L2_norm(b, E_i, xi, ui, vi)
        isfinite(Ni) || continue
        push!(branch_β, β_i)
        push!(branch_E, E_i)
        push!(branch_N, Ni)
    end
    break   # use first non-empty branch
end

if !isempty(branch_E)
    idx = sortperm(branch_E)
    branch_β = branch_β[idx]
    branch_E = branch_E[idx]
    branch_N = branch_N[idx]
    @printf("  Branch: %d points,  E ∈ [%.5f, %.5f],  β ∈ [%.5f, %.5f]\n",
            length(branch_E),
            minimum(branch_E), maximum(branch_E),
            minimum(branch_β), maximum(branch_β))
else
    println("  (no branch data)")
end
println()

# ============================================================
# HELPERS
# ============================================================

"""
    find_E_for_eps(ε)

Given β = ε, find E ∈ (E_bif − 20, E_bif) such that F(β, E) = 0.
Uses a log-spaced scan in δE = E_bif − E to resolve zeros that are
exponentially close to E_bif for small ε.
"""
function find_E_for_eps(ε; n_scan=300, tol_F=1e-12, max_bisect=80)
    β = ε

    # Log-spaced δE from 1e-12 to 20: dense near E_bif, coarse far away
    log_lo = log10(1e-12)
    log_hi = log10(20.0)
    δE_vals = 10 .^ range(log_hi, log_lo; length=n_scan)   # decreasing δE
    E_grid  = E_bif .- δE_vals    # increasing E (towards E_bif)

    F_prev = F_residual(b, E_grid[1], Vfun, β; N=N_shoot)

    for i in 2:length(E_grid)
        F_curr = F_residual(b, E_grid[i], Vfun, β; N=N_shoot)

        if isfinite(F_prev) && isfinite(F_curr) && F_prev * F_curr < 0
            # Bisect in [E_grid[i-1], E_grid[i]]
            Elo, Ehi = E_grid[i-1], E_grid[i]
            for _ in 1:max_bisect
                Em = 0.5*(Elo + Ehi)
                Fm = F_residual(b, Em, Vfun, β; N=N_shoot)
                (abs(Fm) < tol_F || Ehi - Elo < 1e-14) && break
                F_residual(b, Elo, Vfun, β; N=N_shoot) * Fm < 0 ? (Ehi=Em) : (Elo=Em)
            end
            return 0.5*(Elo + Ehi)
        end

        isfinite(F_curr) && (F_prev = F_curr)
    end

    error("find_E_for_eps: no zero found for ε = $ε  " *
          "(check potential parameters or increase scan range)")
end

"""
    get_Xmax(ε, E)

Adaptive domain size: large enough to capture soliton center + tail.
Uses the estimate  x₀ ≈ b + log(√(−2E) / ε) / √(−E).
"""
function get_Xmax(ε, E)
    κ_E = sqrt(-E)
    A_E = sqrt(-2E)
    x0_est = b + log(max(A_E / ε, 1.0)) / κ_E
    return max(x0_est + 15.0/κ_E, 30.0)
end

# ============================================================
# STEPS 3–4: Loop over ε values
# ============================================================

println("--- Steps 3–4: ψ_ε, μ(ε), N(ε), dN/dE ---\n")

E_num    = Float64[]
N_num    = Float64[]
mu_num   = Float64[]
lam1_num = Float64[]     # most-negative L₊ eigenvalue (≈ −3γ²)
dNdE_num = Float64[]

for (i, ε) in enumerate(eps_values)
    @printf("  [%d/%d]  ε = %.2e\n", i, length(eps_values), ε)

    # ── Find E(ε) ──────────────────────────────────────────────
    E = find_E_for_eps(ε)
    push!(E_num, E)
    @printf("    E(ε)      = %.10e\n", E)
    @printf("    E − E_bif = %.4e\n",  E - E_bif)

    # ── N(ε) ───────────────────────────────────────────────────
    xi, ui, vi = shoot_from_origin(b, E, Vfun, ε; N=N_shoot)
    Nval = compute_L2_norm(b, E, xi, ui, vi)
    push!(N_num, Nval)
    @printf("    N(ε)      = %.8e\n", Nval)

    # ── μ(ε): near-zero eigenvalue of L₊ ──────────────────────
    Xmax = get_Xmax(ε, E)
    λp, _ = compute_Lpm_eigenvalues(b, E, Vfun, ε;
                                    nev = 3,
                                    Ngrid = Ngrid_Lp,
                                    Xmax = Xmax)
    push!(lam1_num, λp[1])
    push!(mu_num,   λp[2])
    @printf("    λ₋₁       = %.6e   (ground state of L₊, expect ≈ −3γ² = %.4f)\n",
            λp[1], -3γ^2)
    @printf("    μ(ε)      = %.6e\n", λp[2])
    @printf("    ν₀ε²      = %.6e   (predicted)\n", ν₀*ε^2)
    @printf("    rel.err   = %.4e\n",
            abs(λp[2] - ν₀*ε^2) / abs(ν₀*ε^2))

    # ── dN/dE via centered finite difference in ε ─────────────
    δε  = δ_rel * ε
    E_p = find_E_for_eps(ε + δε)
    E_m = find_E_for_eps(ε - δε)
    xp, up, vp = shoot_from_origin(b, E_p, Vfun, ε+δε; N=N_shoot)
    xm, um, vm = shoot_from_origin(b, E_m, Vfun, ε-δε; N=N_shoot)
    N_p = compute_L2_norm(b, E_p, xp, up, vp)
    N_m = compute_L2_norm(b, E_m, xm, um, vm)
    ΔE  = E_p - E_m
    ΔN  = N_p - N_m

    if abs(ΔE) < 1e-20
        push!(dNdE_num, NaN)
        println("    dN/dE: ΔE too small, skipped")
    else
        slope_num = ΔN / ΔE
        push!(dNdE_num, slope_num)
        @printf("    dN/dE_num = %.6e\n", slope_num)
        @printf("    dN/dE_pred= %.6e\n", slope_pred)
        @printf("    rel.err   = %.4e\n",
                abs(slope_num - slope_pred) / abs(slope_pred))
    end

    println()
end

# ============================================================
# STEP 5a: Print convergence tables
# ============================================================

println("="^70)
println("TABLE 1: Convergence of μ(ε)  [μ(ε) = ν₀ε² + O(ε³)]")
println("─"^70)
@printf("  %-12s  %-16s  %-16s  %-12s\n",
        "ε", "μ_numerical", "ν₀·ε²", "rel. error")
println("─"^70)

rel_err_mu = Float64[]
for (i, ε) in enumerate(eps_values)
    pred = ν₀ * ε^2
    relerr = abs(mu_num[i] - pred) / abs(pred)
    push!(rel_err_mu, relerr)
    @printf("  %-12.4e  %-16.8e  %-16.8e  %-12.4e\n",
            ε, mu_num[i], pred, relerr)
end
println("─"^70)

println()
println("TABLE 2: Convergence of dN/dE")
println("─"^70)
@printf("  %-12s  %-16s  %-16s  %-12s\n",
        "ε", "(dN/dE)_num", "predicted", "rel. error")
println("─"^70)

rel_err_slope = Float64[]
for (i, ε) in enumerate(eps_values)
    if isfinite(dNdE_num[i])
        relerr = abs(dNdE_num[i] - slope_pred) / abs(slope_pred)
        push!(rel_err_slope, relerr)
        @printf("  %-12.4e  %-16.8e  %-16.8e  %-12.4e\n",
                ε, dNdE_num[i], slope_pred, relerr)
    else
        push!(rel_err_slope, NaN)
        @printf("  %-12.4e  %-16s  %-16.8e  %-12s\n",
                ε, "NaN", slope_pred, "—")
    end
end
println("─"^70)

println()
println("CONSISTENCY CHECK")
println("─"^40)
@printf("  Ω = %.6e  (%s)\n", Ω, Ω > 0 ? "positive" : "negative")
pos_mu = count(x -> isfinite(x) && x > 0, mu_num)
neg_mu = count(x -> isfinite(x) && x < 0, mu_num)
@printf("  μ: %d positive, %d negative\n", pos_mu, neg_mu)
expected_sign = Ω > 0 ? "positive" : "negative"
actual_sign   = pos_mu >= neg_mu ? "positive" : "negative"
@printf("  Expected μ %s → %s\n", expected_sign,
        expected_sign == actual_sign ? "✓ PASS" : "✗ FAIL")

valid_slopes = filter(isfinite, dNdE_num)
if !isempty(valid_slopes)
    slope_mean = sum(valid_slopes) / length(valid_slopes)
    @printf("  dN/dE: mean numerical = %.4e, predicted = %.4e  → %s\n",
            slope_mean, slope_pred,
            sign(slope_mean) == sign(slope_pred) ? "✓ consistent sign" : "✗ sign mismatch")
end
println()

# ============================================================
# STEP 5b: Save CSVs
# ============================================================

csv_mu = joinpath(out_dir, "mu_convergence.csv")
open(csv_mu, "w") do f
    println(f, "eps,E,mu_numerical,nu0_eps2,rel_error,lambda1")
    for (i, ε) in enumerate(eps_values)
        pred = ν₀ * ε^2
        println(f, "$ε,$(E_num[i]),$(mu_num[i]),$pred,$(rel_err_mu[i]),$(lam1_num[i])")
    end
end
println("Saved: $csv_mu")

csv_slope = joinpath(out_dir, "slope_convergence.csv")
open(csv_slope, "w") do f
    println(f, "eps,E,N,dNdE_numerical,dNdE_predicted,rel_error")
    for (i, ε) in enumerate(eps_values)
        println(f, "$ε,$(E_num[i]),$(N_num[i]),$(dNdE_num[i]),$slope_pred,$(rel_err_slope[i])")
    end
end
println("Saved: $csv_slope")

csv_branch = joinpath(out_dir, "branch.csv")
if !isempty(branch_E)
    open(csv_branch, "w") do f
        println(f, "beta,E,N")
        for i in eachindex(branch_E)
            println(f, "$(branch_β[i]),$(branch_E[i]),$(branch_N[i])")
        end
    end
    println("Saved: $csv_branch")
end

# ============================================================
# STEP 5c: Plots
# ============================================================

using Plots, LaTeXStrings
pgfplotsx()
default(fontfamily = "Computer Modern",
        linewidth   = 2,
        markersize  = 5,
        legendfontsize = 10,
        tickfontsize   = 9,
        guidefontsize  = 11,
        titlefontsize  = 11,
        framestyle = :box,
        grid = true)

label_str = @sprintf("smooth well,  b = %.1f,  V₀ = %.1f", b, V0)

# ── Plot 1: μ(ε)/ε² vs ε  (semilog-x) ──────────────────────

ratio_mu = mu_num ./ eps_values.^2

p1 = plot(eps_values, ratio_mu;
    xscale  = :log10,
    xlabel  = L"\varepsilon",
    ylabel  = L"\mu(\varepsilon)\,/\,\varepsilon^2",
    label   = L"\mu_{\mathrm{num}} / \varepsilon^2",
    marker  = :circle,
    color   = :steelblue,
    title   = L"Eigenvalue splitting: $\mu(\varepsilon)/\varepsilon^2 \to \nu_0$"*
              "\n"*label_str)
hline!(p1, [ν₀];
    label     = @sprintf("predicted  ν₀ = %.5f", ν₀),
    linestyle = :dash,
    color     = :firebrick)

savefig(p1, joinpath(out_dir, "mu_ratio.pdf"))
println("Saved: $(joinpath(out_dir, "mu_ratio.pdf"))")

# ── Plot 2: relative error of μ  (log-log, expect O(ε)) ─────

finite_idx = findall(isfinite, rel_err_mu)
p2 = plot(eps_values[finite_idx], rel_err_mu[finite_idx];
    xscale  = :log10,
    yscale  = :log10,
    xlabel  = L"\varepsilon",
    ylabel  = "relative error",
    label   = L"|\mu_\mathrm{num} - \nu_0\varepsilon^2| / |\nu_0\varepsilon^2|",
    marker  = :circle,
    color   = :steelblue,
    title   = "Relative error in μ  (expect O(ε))\n"*label_str)

# O(ε) reference line
ref_vals = rel_err_mu[finite_idx[1]] / eps_values[finite_idx[1]] .* eps_values[finite_idx]
plot!(p2, eps_values[finite_idx], ref_vals;
    label     = L"O(\varepsilon)",
    linestyle = :dash,
    color     = :gray)

savefig(p2, joinpath(out_dir, "mu_relerr.pdf"))
println("Saved: $(joinpath(out_dir, "mu_relerr.pdf"))")

# ── Plot 3: dN/dE vs ε  (semilog-x) ─────────────────────────

valid = findall(isfinite, dNdE_num)
if !isempty(valid)
    p3 = plot(eps_values[valid], dNdE_num[valid];
        xscale  = :log10,
        xlabel  = L"\varepsilon",
        ylabel  = L"dN/dE",
        label   = L"(dN/dE)_\mathrm{num}",
        marker  = :diamond,
        color   = :seagreen,
        title   = L"Slope $dN/dE$ converging to predicted value"*
                  "\n"*label_str)
    hline!(p3, [slope_pred];
        label     = @sprintf("predicted = %.5f", slope_pred),
        linestyle = :dash,
        color     = :firebrick)
    savefig(p3, joinpath(out_dir, "slope_convergence.pdf"))
    println("Saved: $(joinpath(out_dir, "slope_convergence.pdf"))")
end

# ── Plot 4: relative error of dN/dE  (log-log) ───────────────

valid2 = findall(isfinite, rel_err_slope)
if !isempty(valid2)
    p4 = plot(eps_values[valid2], rel_err_slope[valid2];
        xscale  = :log10,
        yscale  = :log10,
        xlabel  = L"\varepsilon",
        ylabel  = "relative error",
        label   = L"|(dN/dE)_\mathrm{num} - \mathrm{pred}| / |\mathrm{pred}|",
        marker  = :diamond,
        color   = :seagreen,
        title   = "Relative error in dN/dE\n"*label_str)
    savefig(p4, joinpath(out_dir, "slope_relerr.pdf"))
    println("Saved: $(joinpath(out_dir, "slope_relerr.pdf"))")
end

# ── Plot 5: N vs E branch + verification points ──────────────

if !isempty(branch_E)
    p5 = plot(branch_E, branch_N;
        xlabel  = L"E",
        ylabel  = L"N(\varepsilon) = \|\psi_\varepsilon\|_{L^2}^2",
        label   = "continuation branch",
        color   = :royalblue,
        title   = L"$N$ vs $E$ branch"*"\n"*label_str)

    # Overlay the direct-shooting verification points
    scatter!(p5, E_num, N_num;
        label  = L"\varepsilon\text{ verification points}",
        marker = :circle,
        color  = :firebrick,
        ms     = 6)

    # Mark E_bif
    vline!(p5, [E_bif];
        label     = @sprintf("E_bif = %.4f", E_bif),
        linestyle = :dot,
        color     = :black)

    savefig(p5, joinpath(out_dir, "N_vs_E.pdf"))
    println("Saved: $(joinpath(out_dir, "N_vs_E.pdf"))")
end

# ── Plot 6: solution profiles at each ε ──────────────────────

Xmax_prof = get_Xmax(minimum(eps_values), minimum(E_num))
x_plot    = range(0.0, Xmax_prof; length=2000)
colors    = cgrad(:viridis, length(eps_values); categorical=true)

p6 = plot(xlabel = L"x",
          ylabel = L"\psi_\varepsilon(x)",
          title  = "Bound-state profiles\n"*label_str)

for (i, ε) in enumerate(eps_values)
    E = E_num[i]
    xi, ui, vi = shoot_from_origin(b, E, Vfun, ε; N=N_shoot)
    xf, ψf     = glue_solution(b, E, xi, ui, vi; Xmax=Xmax_prof)
    isempty(xf) && continue

    ψ_grid = linear_interp(xf, ψf, collect(x_plot))

    plot!(p6, collect(x_plot), ψ_grid;
        label  = @sprintf("ε = %.0e", ε),
        color  = colors[i],
        alpha  = 0.8)
end

vline!(p6, [b]; label="x = b", linestyle=:dash, color=:black)

savefig(p6, joinpath(out_dir, "profiles.pdf"))
println("Saved: $(joinpath(out_dir, "profiles.pdf"))")

println()
println("="^70)
@printf("DONE.  b = %.2f,  V0 = %.2f,  γ = %.6f,  E_bif = %.6f\n",
        b, V0, γ, E_bif)
@printf("       Ω = %.6f,  ν₀ = %.6f,  dN/dE_pred = %.6f\n",
        Ω, ν₀, slope_pred)
println("Results written to:  $out_dir")
println("="^70)
